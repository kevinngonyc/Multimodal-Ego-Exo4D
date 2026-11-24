"""Commentary-only pretraining, then a frozen level probe (run from idea_1).

usage:  python cls_lang/pretrain.py [--arm hard|weighted] [--aspect] [--repeats 4] [--seeds 2] [--smoke]

Why: in train.py the level loss overfits by epoch ~3, so the commentary heads barely train. Here the same transformer
(train.Net) is trained on the commentary losses ONLY (presence BCE + weighted polarity SE + weighted text 1-cos,
optionally + aspect soft-CE with --aspect), early-stopped on the inner participant split's commentary loss, with no
level gradient at all. Expert weights (hard / weighted) come from the outer training takes only; only training takes'
commentary is used. Then the network is frozen and every take (incl. takes with no commentary) is encoded:
  enc   = mean + std over clip-token outputs (2 x 128)   (the CLS gets no gradient without a level loss)
  heads = take summary of the commentary heads: mean / std presence prob, mean polarity per aspect, presence-weighted
          mean polarity per aspect (40)
Level is read out with the usual probe (scenario-centred, standardised, scenario one-hot, RidgeCV / LogisticRegressionCV),
alone and on top of the frozen PCA probe. Ridge predictions averaged over seeds.
Commentary diagnostics on held-out takes: presence AUC (vs a relative-position baseline), polarity sign AUC on covered
clips, text cosine (vs the training-mean embedding).
Protocol = train.py: 4 x 5-fold StratifiedGroupKFold by participant over the 2,111 official takes.
-> cls_lang/pretrain_{arm}[_aspect].pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from train import (DEV, EMB, N_ASP, SCEN, Net, Sentences, batchify, expert_rho, load_ratings,  # noqa: E402
                   load_takes, rho_w, weight_fn)

FEATS = ("probe", "enc", "heads", "probe+enc", "probe+heads")


class SentencesA(Sentences):
    """adds a per-clip aspect distribution target (incl. 'other') and the per-scenario valid-aspect mask."""
    def __init__(self, take_ix, n_clips):
        super().__init__(take_ix, n_clips)
        S = torch.load(Path("embeddings") / "cls_lang_sentences.pt", weights_only=False)
        self.asp_full = S["aspect"].astype(np.float32)
        self.valid = np.zeros((6, N_ASP), bool)
        for sc, names in S["aspect_names"].items():
            self.valid[SCEN.index(sc), :len(names)] = True

    def targets(self, j, w_of):
        out = super().targets(j, w_of)
        if out is None:
            return None
        n = self.n_clips[j]
        num = np.zeros((n, N_ASP), np.float32); den = np.zeros(n, np.float32)
        for k in self.by_take.get(j, []):
            w = w_of(self.expert[k])
            if w <= 0:
                continue
            a, b = min(self.first[k], n - 1), min(self.last[k], n - 1)
            num[a:b + 1] += w * self.asp_full[k]; den[a:b + 1] += w
        out["asp"] = np.divide(num, den[:, None], out=np.zeros_like(num), where=den[:, None] > 0)
        return out


class NetA(Net):
    def __init__(self):
        super().__init__()
        self.aspect = nn.Linear(self.d, N_ASP)

    def tokens(self, x, tpos, mask, s):
        h = self.inp(x) + self.time_enc(tpos)
        h = torch.cat([self.cls + self.scen(s)[:, None], h], 1)
        m = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=x.device), ~mask], 1)
        tok = self.norm(self.enc(h, src_key_padding_mask=m))[:, 1:]
        return tok, self.pres(tok).squeeze(-1), torch.tanh(self.pol(tok)), self.txt(tok), self.aspect(tok)


def add_asp(b, idx, T):
    A = torch.zeros(len(idx), b["x"].shape[1], N_ASP)
    for i, (j, o, n) in enumerate(zip(idx, b["offs"], b["lens"])):
        tg = T.get(int(j))
        if tg is not None:
            A[i, :n] = torch.from_numpy(tg["asp"][o:o + n])
    b["asp"] = A.to(DEV)


def comm_loss(net, b, valid, use_aspect):
    tok, pres, pol, txt, asp = net.tokens(b["x"], b["t"], b["mask"], b["s"])
    tm = b["mask"] & (b["has"][:, None] > 0)
    if tm.sum() == 0:
        return None
    l = F.binary_cross_entropy_with_logits(pres[tm], b["pres"][tm])
    l = l + ((pol - b["pol"]) ** 2 * b["polw"])[tm].sum() / tm.sum()
    cos = F.cosine_similarity(txt, b["txt"] + 1e-8, dim=-1)
    l = l + ((1 - cos) * b["txtw"])[tm].sum() / tm.sum()
    if use_aspect:
        lg = asp.masked_fill(~valid[b["s"]][:, None, :], -1e4)
        ce = -(b["asp"] * F.log_softmax(lg, -1)).sum(-1)
        l = l + (ce * b["txtw"])[tm].sum() / tm.sum()
    return l


def batch(idx, Xs, s, T, use_aspect, crop=None, rng=None):
    b = batchify(idx, Xs, s, crop=crop, rng=rng, T=T)
    if use_aspect:
        add_asp(b, idx, T)
    return b


@torch.no_grad()
def eval_loss(net, idx, Xs, s, T, valid, use_aspect, bs=32):
    net.eval()
    tot, cnt = 0.0, 0
    order = sorted(idx, key=lambda j: len(Xs[j]))
    for i in range(0, len(order), bs):
        ch = order[i:i + bs]
        l = comm_loss(net, batch(ch, Xs, s, T, use_aspect), valid, use_aspect)
        if l is not None:
            tot += float(l) * len(ch); cnt += len(ch)
    net.train()
    return tot / max(cnt, 1)


def train_comm(Xs, s, fit_i, iv_i, T, valid, use_aspect, seed, epochs, patience, smoke):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = NetA().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    best, best_state, bad, best_ep = np.inf, None, 0, 0
    for ep in range(1, epochs + 1):
        perm = rng.permutation(fit_i)
        for i in range(0, len(perm), 16):
            idx = perm[i:i + 16]
            l = comm_loss(net, batch(idx, Xs, s, T, use_aspect, crop=384, rng=rng), valid, use_aspect)
            opt.zero_grad(set_to_none=True); l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        cur = eval_loss(net, iv_i, Xs, s, T, valid, use_aspect)
        if cur < best - 1e-4:
            best, best_state, bad, best_ep = cur, {k: v.clone() for k, v in net.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= patience:
                break
        if smoke and ep >= 2:
            break
    net.load_state_dict(best_state)
    net.eval()
    return net, best_ep, best


@torch.no_grad()
def encode(net, Xs, s, bs=32):
    """per take: enc (256) and heads (40) features + per-clip head outputs (for diagnostics)."""
    order = sorted(range(len(Xs)), key=lambda j: len(Xs[j]))
    enc = np.zeros((len(Xs), 2 * net.d), np.float32); heads = np.zeros((len(Xs), 2 + 2 * N_ASP), np.float32)
    clip = {}
    for i in range(0, len(order), bs):
        ch = order[i:i + bs]
        b = batchify(ch, Xs, s)
        tok, pres, pol, txt, _ = net.tokens(b["x"], b["t"], b["mask"], b["s"])
        pp = torch.sigmoid(pres)
        for k, (j, n) in enumerate(zip(ch, b["lens"])):
            t, p, q = tok[k, :n], pp[k, :n], pol[k, :n]
            enc[j] = torch.cat([t.mean(0), t.std(0, unbiased=False)]).cpu().numpy()
            pw = (p[:, None] * q).sum(0) / (p.sum() + 1e-6)
            heads[j] = torch.cat([p.mean()[None], p.std(unbiased=False)[None], q.mean(0), pw]).cpu().numpy()
            clip[j] = (p.cpu().numpy(), q.cpu().numpy(), F.normalize(txt[k, :n], dim=-1).cpu().numpy())
    return enc, heads, clip


def blockify(B, s, fit):
    B = B.astype(np.float64).copy()
    for k in range(6):
        m = s == k
        if (fit & m).any():
            B[m] -= B[fit & m].mean(0)
    return B


def readout(Z, y, fit, held):
    Z = StandardScaler().fit(Z[fit]).transform(Z)
    r = RidgeCV(alphas=np.logspace(-2, 6, 17)).fit(Z[fit], y[fit]).predict(Z[held])
    pr = LogisticRegressionCV(Cs=np.logspace(-4, 2, 7), cv=5, max_iter=5000, n_jobs=4).fit(Z[fit], y[fit]).predict_proba(Z[held])
    return r, pr


def diagnostics(clip, te, Ttest, mean_txt):
    pres_y, pres_p, pos, pol_y, pol_p, cos_m, cos_b = [], [], [], [], [], [], []
    for j in te:
        tg = Ttest.get(int(j))
        if tg is None:
            continue
        p, q, txt = clip[int(j)]
        n = len(p)
        pres_y.append(tg["pres"] > 0); pres_p.append(p); pos.append(np.arange(n) / max(n - 1, 1))
        w = tg["pol_w"]; cov = w.sum(1) > 1e-3
        if cov.any():
            pol_y.append((tg["pol"][cov] * w[cov]).sum(1) / w[cov].sum(1) > 0)
            pol_p.append((q[cov] * w[cov]).sum(1) / w[cov].sum(1))
        c = tg["txt_w"] > 0
        if c.any():
            tt = tg["txt"][c] / (np.linalg.norm(tg["txt"][c], axis=1, keepdims=True) + 1e-8)
            cos_m.append((txt[c] * tt).sum(1)); cos_b.append(tt @ mean_txt)
    cat = np.concatenate
    py = cat(pres_y)
    return dict(pres_auc=float(roc_auc_score(py, cat(pres_p))), pres_auc_position=float(roc_auc_score(py, cat(pos))),
                pol_auc=float(roc_auc_score(cat(pol_y), cat(pol_p))), pol_frac_good=float(cat(pol_y).mean()),
                txt_cos=float(cat(cos_m).mean()), txt_cos_meanbaseline=float(cat(cos_b).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("hard", "weighted"), default="hard")
    ap.add_argument("--aspect", action="store_true")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    names, Xraw, s, y, p = load_takes()
    n = len(names)
    take_ix = {t: j for j, t in enumerate(names)}
    sents = SentencesA(take_ix, [len(x) for x in Xraw])
    valid = torch.as_tensor(sents.valid, device=DEV)
    R = load_ratings(take_ix)
    Xmean = np.stack([x.float().mean(0).numpy() for x in Xraw])
    tag = f"{a.arm}{'_aspect' if a.aspect else ''}{'_smoke' if a.smoke else ''}"
    print(f"{n} takes; commentary-only pretraining, arm {a.arm}, aspect head {a.aspect}", flush=True)
    reg = {k: np.zeros((a.repeats, n)) for k in FEATS}; prob = {k: np.zeros((a.repeats, n, 4)) for k in FEATS}
    fold_of = np.zeros((a.repeats, n), int); log = []
    t0 = time.time()
    for r in range(a.repeats):
        for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(n), y, p)):
            fold_of[r, te] = f
            fit_rel, iv_rel = next(StratifiedGroupKFold(7, shuffle=True, random_state=100 + r).split(np.zeros(len(tr)), y[tr], p[tr]))
            fit_i, iv_i = tr[fit_rel], tr[iv_rel]
            mu = np.stack([Xmean[tr][s[tr] == k].mean(0) for k in range(6)])
            sd = (Xmean[tr] - mu[s[tr]]).std(0) + 1e-6
            mu_t, sd_t = torch.as_tensor(mu, dtype=torch.float32), torch.as_tensor(sd, dtype=torch.float32)
            Xs = [(x.float() - mu_t[s[j]]) / sd_t for j, x in enumerate(Xraw)]
            wf = weight_fn(a.arm, expert_rho(R, s, y, tr), sents, s)   # expert weights from training takes only
            T = {int(j): tg for j in tr if (tg := sents.targets(int(j), wf(int(j)))) is not None}
            Ttest = {int(j): tg for j in te if (tg := sents.targets(int(j), wf(int(j)))) is not None}  # diagnostics only
            fit_c = np.array([j for j in fit_i if int(j) in T]); iv_c = np.array([j for j in iv_i if int(j) in T])
            mean_txt = np.mean([tg["txt"][tg["txt_w"] > 0].mean(0) for tg in T.values()], 0)
            mean_txt /= np.linalg.norm(mean_txt)
            fitm = np.zeros(n, bool); fitm[tr] = True
            # frozen PCA probe block (as in train.probe)
            P = [(s[:, None] == np.arange(6)[None]).astype(float)]
            for v in (slice(0, 768), slice(768, 1536)):
                P.append(PCA(64, svd_solver="full").fit(blockify(Xmean[:, v], s, fitm)[fitm]).transform(blockify(Xmean[:, v], s, fitm)))
            P = np.hstack(P)
            outs = {k: [] for k in FEATS}; diags = []
            for seed in range(a.seeds):
                net, ep, best = train_comm(Xs, s, fit_c, iv_c, T, valid, a.aspect, seed + 10 * r, a.epochs, a.patience, a.smoke)
                enc, heads, clip = encode(net, Xs, s)
                E, H = blockify(enc, s, fitm), blockify(heads, s, fitm)
                Zs = {"probe": P, "enc": np.hstack([P[:, :6], E]), "heads": np.hstack([P[:, :6], H]),
                      "probe+enc": np.hstack([P, E]), "probe+heads": np.hstack([P, H])}
                for k in FEATS:
                    if k == "probe" and seed > 0:
                        outs[k].append(outs[k][0]); continue
                    outs[k].append(readout(Zs[k], y, fitm, te))
                diags.append(diagnostics(clip, te, Ttest, mean_txt))
                log.append(dict(repeat=r, fold=f, seed=seed, best_epoch=ep, inner_loss=best, **diags[-1]))
                del net; torch.cuda.empty_cache()
            for k in FEATS:
                reg[k][r, te] = np.mean([o[0] for o in outs[k]], 0); prob[k][r, te] = np.mean([o[1] for o in outs[k]], 0)
            dm = {k: round(float(np.mean([d[k] for d in diags])), 3) for k in diags[0]}
            print(json.dumps(dict(repeat=r, fold=f, n_train_targets=len(T), epochs=[l["best_epoch"] for l in log[-a.seeds:]],
                                  **{f"rho_{k}": round(rho_w(reg[k][r, te], y[te], s[te]), 3) for k in FEATS}, **dm)),
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if a.smoke:
                break
        if a.smoke:
            break
    torch.save(dict(names=names, reg=reg, prob=prob, fold_of=fold_of, log=log, args=vars(a)),
               Path("cls_lang") / f"pretrain_{tag}.pt")


if __name__ == "__main__":
    main()
