"""CLS + language model on frozen TimeSformer K600 clip features: CLS -> official level, clip tokens -> commentary.

usage (from idea_1):  python cls_lang/train.py --arm level|weighted|hard [--repeats 4] [--smoke]

Input per take: one token per 1 s clip = ego + exo L12 patch-token std (1536), scenario-centred and standardised
with training-fold statistics, -> Linear(128) + sinusoidal time (s); a learned [CLS] (+ scenario embedding) in front;
2-layer pre-norm transformer encoder (4 heads, dropout). Heads: CLS -> level score (+ scenario bias) and 4 logits
(+ scenario bias); every clip token -> presence logit, per-aspect polarity (tanh), text embedding (384).
Loss per take = MSE(score) + CE(level) + lam * (BCE presence + w*SE polarity + w*(1 - cos) text), lam = 0.5
(arm "level": lam = 0). Commentary targets (cls_lang/build_targets.py sentences) are rebuilt for every training fold
from per-expert weights computed on the fold's training takes only:
  weighted: w_e = max(0, Spearman(rating_e, level)) within scenario, on training takes (>= 10 takes, else 0)
  hard:     per scenario, experts ranked by that Spearman; per take, weight 1 for the best-ranked expert (rho > 0) who
            commented on the take, 0 for the others
Soccer gets no commentary targets (all its experts' ratings correlate negatively with level).
A comment at pause time T covers clips floor(T) - 2 .. floor(T).
Protocol: 4 x 5-fold StratifiedGroupKFold by participant over the 2,111 official takes (= the main probe protocol);
early stopping on an inner participant-grouped split (1/7 of the training takes) by within-scenario Spearman;
2 seeds averaged. The frozen linear probe (scenario-centred PCA-64 per view, ridge / logistic) is fit on the same folds
from the same clip features.
-> cls_lang/results_{arm}.pt
"""
import argparse
import glob
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

DATA = next(p / "data" for p in (Path.cwd(), *Path.cwd().parents) if (p / "data" / "annotations").is_dir())
A = DATA / "annotations"
LEVELS = {"Novice": 1, "Early Expert": 2, "Intermediate Expert": 3, "Late Expert": 4}
SCEN = ["Basketball", "Cooking", "Dance", "Music", "Rock Climbing", "Soccer"]
FEAT_CACHE = Path("embeddings") / "cls_lang_clipfeats.pt"
N_ASP, EMB = 19, 384
DEV = "cuda"


# ---------------------------------------------------------------- data
def load_takes():
    meta = {}
    for fn in ("proficiency_demonstrator_train.json", "proficiency_demonstrator_val.json"):
        for m in json.load(open(A / fn, encoding="utf-8"))["annotations"]:
            meta[m["video_paths"]["ego"].split("/")[1]] = (SCEN.index(m["scenario_name"]), LEVELS[m["proficiency_score"]],
                                                           m["origin_participant_id"])
    if not FEAT_CACHE.exists():
        feats = {}
        for view in ("ego", "exo"):
            out = {}
            for name in (f"timesformer_k600_{view}8fps_clips.pt", f"timesformer_k600_{view}8fps_clips_extra.pt"):
                c = torch.load(Path("embeddings") / name, mmap=True, weights_only=False)["clips"]["squash"]
                out.update({t: c[t][:, 1, 2].clone().half() for t in c if t in meta})
            feats[view] = out
        both = sorted(t for t in meta if t in feats["ego"] and t in feats["exo"])
        X = {}
        for t in both:
            n = min(len(feats["ego"][t]), len(feats["exo"][t]))
            X[t] = torch.cat([feats["ego"][t][:n], feats["exo"][t][:n]], 1)
        torch.save(X, FEAT_CACHE)
    X = torch.load(FEAT_CACHE, weights_only=False)
    names = sorted(X)
    s = np.array([meta[t][0] for t in names]); y = np.array([meta[t][1] for t in names], float)
    p = np.array([meta[t][2] for t in names])
    return names, [X[t] for t in names], s, y, p


def load_ratings(take_ix):
    """(take index, expert) -> rating, for every expert file with a numeric rating."""
    R = {}
    for f in glob.glob(str(A / "expert_commentary" / "*" / "*" / "data.json")):
        take = Path(f).parent.parent.name
        if take not in take_ix:
            continue
        d = json.load(open(f, encoding="utf-8"))
        try:
            R[(take_ix[take], d.get("user_id"))] = float(d["proficiency"]["rating"])
        except Exception:
            pass
    return R


class Sentences:
    """per take: arrays of (first clip, last clip, expert, polarity value, aspect probs, embedding)."""
    def __init__(self, take_ix, n_clips):
        S = torch.load(Path("embeddings") / "cls_lang_sentences.pt", weights_only=False)
        self.by_take = defaultdict(list)
        ti = np.array([take_ix.get(t, -1) for t in S["take"]])
        keep = ti >= 0
        for k in np.flatnonzero(keep):
            self.by_take[int(ti[k])].append(k)
        self.expert = S["expert"]
        T = S["time"].astype(float)
        last = np.floor(T).astype(int)
        self.first, self.last = np.maximum(last - 2, 0), last
        self.v = 2 * S["p_good"] - 1
        asp = S["aspect"].astype(np.float32)
        self.asp_other = np.zeros(len(asp), np.float32)
        for sc, names in S["aspect_names"].items():
            m = S["scenario"] == sc
            self.asp_other[m] = asp[m, len(names) - 1]
            asp[np.ix_(m, [len(names) - 1])] = 0  # "other" = neutral: no polarity target
        self.asp = asp
        self.emb = S["emb"].astype(np.float32)
        self.n_clips = n_clips

    def experts_of(self, j):
        return {self.expert[k] for k in self.by_take.get(j, [])}

    def targets(self, j, w_of):
        """dense targets for take j given w_of(expert) -> weight; None if no weighted commentary."""
        idx = self.by_take.get(j, [])
        experts = {self.expert[k] for k in idx}
        W = sum(w_of(e) for e in experts)
        if W <= 0:
            return None
        n = self.n_clips[j]
        pres = np.zeros(n, np.float32)
        pol_num = np.zeros((n, N_ASP), np.float32); pol_den = np.zeros((n, N_ASP), np.float32)
        txt = np.zeros((n, EMB), np.float32); txt_w = np.zeros(n, np.float32)
        covered = defaultdict(set)
        for k in idx:
            w = w_of(self.expert[k])
            if w <= 0:
                continue
            a, b = min(self.first[k], n - 1), min(self.last[k], n - 1)
            sl = slice(a, b + 1)
            for t in range(a, b + 1):
                covered[t].add(self.expert[k])
            pol_num[sl] += w * self.asp[k] * self.v[k]
            pol_den[sl] += w * self.asp[k]
            txt[sl] += w * self.emb[k]
            txt_w[sl] += w
        for t, es in covered.items():
            pres[t] = sum(w_of(e) for e in es) / W
        pol = np.divide(pol_num, pol_den, out=np.zeros_like(pol_num), where=pol_den > 1e-6)
        return dict(pres=pres, pol=pol, pol_w=pol_den / W, txt=txt, txt_w=np.minimum(txt_w / W, 1.0))


def expert_rho(R, s, y, fit_idx):
    """within-scenario Spearman(rating, level) per expert on the given takes (>= 10 takes)."""
    fit = set(int(i) for i in fit_idx)
    by = defaultdict(list)
    for (j, e), r in R.items():
        if j in fit:
            by[e].append((r, y[j], s[j]))
    rho = {}
    for e, v in by.items():
        if len(v) < 10:
            continue
        r, l, _ = map(np.array, zip(*v))
        rho[e] = (float(spearmanr(r, l)[0]) if np.std(r) > 0 and np.std(l) > 0 else 0.0, int(np.bincount(_).argmax()))
    return rho


def weight_fn(arm, rho, sents, s):
    soccer = SCEN.index("Soccer")
    if arm == "weighted":
        def w(j):
            if s[j] == soccer:
                return lambda e: 0.0
            return lambda e: max(0.0, rho.get(e, (0.0,))[0])
        return w
    if arm == "hard":
        def w(j):
            if s[j] == soccer:
                return lambda e: 0.0
            cands = [e for e in sents.experts_of(j) if rho.get(e, (0.0,))[0] > 0]
            if not cands:
                return lambda e: 0.0
            best = max(cands, key=lambda e: rho[e][0])
            return lambda e: 1.0 if e == best else 0.0
        return w
    return lambda j: (lambda e: 0.0)


# ---------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, d_in=1536, d=128, layers=2, heads=4, drop=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_in, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.scen = nn.Embedding(6, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.reg = nn.Linear(d, 1); self.reg_b = nn.Parameter(torch.zeros(6))
        self.cl = nn.Linear(d, 4); self.cl_b = nn.Parameter(torch.zeros(6, 4))
        self.pres = nn.Linear(d, 1); self.pol = nn.Linear(d, N_ASP); self.txt = nn.Linear(d, EMB)
        self.d = d

    def time_enc(self, t):
        i = torch.arange(self.d // 2, device=t.device)
        f = torch.exp(-math.log(10000.0) * 2 * i / self.d)
        a = t[..., None].float() * f
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, x, tpos, mask, s):
        h = self.inp(x) + self.time_enc(tpos)
        c = (self.cls + self.scen(s)[:, None])
        h = torch.cat([c, h], 1)
        m = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=x.device), ~mask], 1)
        h = self.norm(self.enc(h, src_key_padding_mask=m))
        z = h[:, 0]
        tok = h[:, 1:]
        return (self.reg(z).squeeze(-1) + self.reg_b[s], self.cl(z) + self.cl_b[s],
                self.pres(tok).squeeze(-1), torch.tanh(self.pol(tok)), self.txt(tok))


def batchify(idx, Xs, s, crop=None, rng=None, T=None):
    seqs, offs = [], []
    for j in idx:
        x = Xs[j]
        o = 0
        if crop and len(x) > crop:
            o = int(rng.integers(0, len(x) - crop + 1)); x = x[o:o + crop]
        seqs.append(x); offs.append(o)
    L = max(len(q) for q in seqs)
    B = len(idx)
    X = torch.zeros(B, L, seqs[0].shape[1]); M = torch.zeros(B, L, dtype=torch.bool); P = torch.zeros(B, L)
    for b, (q, o) in enumerate(zip(seqs, offs)):
        X[b, :len(q)] = q; M[b, :len(q)] = True; P[b, :len(q)] = torch.arange(o, o + len(q))
    out = dict(x=X.to(DEV), mask=M.to(DEV), t=P.to(DEV), s=torch.as_tensor(s[idx]).to(DEV), offs=offs, lens=[len(q) for q in seqs])
    if T is not None:
        pres = torch.zeros(B, L); pol = torch.zeros(B, L, N_ASP); polw = torch.zeros(B, L, N_ASP)
        txt = torch.zeros(B, L, EMB); txtw = torch.zeros(B, L); has = torch.zeros(B)
        for b, (j, o, n) in enumerate(zip(idx, offs, out["lens"])):
            tg = T.get(int(j))
            if tg is None:
                continue
            has[b] = 1
            pres[b, :n] = torch.from_numpy(tg["pres"][o:o + n]); pol[b, :n] = torch.from_numpy(tg["pol"][o:o + n])
            polw[b, :n] = torch.from_numpy(tg["pol_w"][o:o + n]); txt[b, :n] = torch.from_numpy(tg["txt"][o:o + n])
            txtw[b, :n] = torch.from_numpy(tg["txt_w"][o:o + n])
        out.update(pres=pres.to(DEV), pol=pol.to(DEV), polw=polw.to(DEV), txt=txt.to(DEV), txtw=txtw.to(DEV), has=has.to(DEV))
    return out


@torch.no_grad()
def predict(net, idx, Xs, s, bs=32):
    net.eval()
    order = sorted(idx, key=lambda j: len(Xs[j]))
    reg, prob = {}, {}
    for i in range(0, len(order), bs):
        ch = order[i:i + bs]
        b = batchify(ch, Xs, s)
        r, l, *_ = net(b["x"], b["t"], b["mask"], b["s"])
        for j, rr, ll in zip(ch, r.cpu().numpy(), F.softmax(l, -1).cpu().numpy()):
            reg[j], prob[j] = rr, ll
    net.train()
    return np.array([reg[j] for j in idx]), np.array([prob[j] for j in idx])


def centred(v, g):
    v = np.asarray(v, float).copy()
    for k in np.unique(g):
        v[g == k] -= v[g == k].mean()
    return v


def rho_w(pred, y, s):
    return spearmanr(centred(pred, s), centred(y, s))[0]


def train_one(Xs, s, y, fit_i, iv_i, T, lam, seed, epochs, smoke):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    yt = torch.as_tensor(y, dtype=torch.float32, device=DEV)
    best, best_state, bad, best_ep = -2, None, 0, 0
    B = 16
    for ep in range(1, epochs + 1):
        perm = rng.permutation(fit_i)
        for i in range(0, len(perm), B):
            idx = perm[i:i + B]
            b = batchify(idx, Xs, s, crop=384, rng=rng, T=T if lam > 0 else None)
            reg, lg, pres, pol, txt = net(b["x"], b["t"], b["mask"], b["s"])
            yb = yt[torch.as_tensor(idx, device=DEV)]
            loss = F.mse_loss(reg, yb) + F.cross_entropy(lg, (yb - 1).long())
            if lam > 0 and b["has"].sum() > 0:
                tm = b["mask"] & (b["has"][:, None] > 0)
                l_pres = F.binary_cross_entropy_with_logits(pres[tm], b["pres"][tm])
                l_pol = ((pol - b["pol"]) ** 2 * b["polw"])[tm].sum() / tm.sum()
                cos = F.cosine_similarity(txt, b["txt"] + 1e-8, dim=-1)
                l_txt = ((1 - cos) * b["txtw"])[tm].sum() / tm.sum()
                loss = loss + lam * (l_pres + l_pol + l_txt)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        r, _ = predict(net, iv_i, Xs, s)
        cur = rho_w(r, y[iv_i], s[iv_i])
        if cur > best:
            best, best_state, bad, best_ep = cur, {k: v.clone() for k, v in net.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= 8:
                break
        if smoke and ep >= 2:
            break
    net.load_state_dict(best_state)
    return net, best_ep, best


def probe(Xmean, s, y, fit, held):
    parts = [(s[:, None] == np.arange(6)[None]).astype(float)]
    for v in (slice(0, 768), slice(768, 1536)):
        Z = Xmean[:, v].astype(np.float64).copy()
        for k in range(6):
            m = s == k
            if (fit & m).any():
                Z[m] -= Z[fit & m].mean(0)
        parts.append(PCA(64, svd_solver="full").fit(Z[fit]).transform(Z))
    Z = StandardScaler().fit(np.hstack(parts)[fit]).transform(np.hstack(parts))
    r = RidgeCV(alphas=np.logspace(-2, 6, 17)).fit(Z[fit], y[fit]).predict(Z[held])
    pr = LogisticRegressionCV(Cs=np.logspace(-4, 2, 7), cv=5, max_iter=5000, n_jobs=4).fit(Z[fit], y[fit]).predict_proba(Z[held])
    return r, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("level", "weighted", "hard"), required=True)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    names, Xraw, s, y, p = load_takes()
    n = len(names)
    take_ix = {t: j for j, t in enumerate(names)}
    n_clips = [len(x) for x in Xraw]
    Xmean = np.stack([x.float().mean(0).numpy() for x in Xraw])
    sents = Sentences(take_ix, n_clips) if a.arm != "level" else None
    R = load_ratings(take_ix)
    lam = 0.0 if a.arm == "level" else a.lam
    print(f"{n} takes, arm {a.arm}, lam {lam}; takes with sentences: {len(sents.by_take) if sents else 0}", flush=True)
    reg = np.zeros((a.repeats, n)); prob = np.zeros((a.repeats, n, 4))
    preg = np.zeros((a.repeats, n)); pprob = np.zeros((a.repeats, n, 4))
    fold_of = np.zeros((a.repeats, n), int); log = []
    t0 = time.time()
    for r in range(a.repeats):
        for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(n), y, p)):
            fold_of[r, te] = f
            fit_rel, iv_rel = next(StratifiedGroupKFold(7, shuffle=True, random_state=100 + r).split(np.zeros(len(tr)), y[tr], p[tr]))
            fit_i, iv_i = tr[fit_rel], tr[iv_rel]
            # standardisation from the training takes (scenario means of take-mean features, global sd)
            mu = np.stack([Xmean[tr][s[tr] == k].mean(0) for k in range(6)])
            sd = (Xmean[tr] - mu[s[tr]]).std(0) + 1e-6
            mu_t, sd_t = torch.as_tensor(mu, dtype=torch.float32), torch.as_tensor(sd, dtype=torch.float32)
            Xs = [(x.float() - mu_t[s[j]]) / sd_t for j, x in enumerate(Xraw)]
            T = None
            if lam > 0:
                rho = expert_rho(R, s, y, fit_i)
                wf = weight_fn(a.arm, rho, sents, s)
                T = {}
                for j in fit_i:
                    tg = sents.targets(int(j), wf(int(j)))
                    if tg is not None:
                        T[int(j)] = tg
            preds = []
            for seed in range(a.seeds):
                net, ep, inner = train_one(Xs, s, y, fit_i, iv_i, T, lam, seed + 10 * r, a.epochs, a.smoke)
                preds.append(predict(net, te, Xs, s))
                log.append(dict(repeat=r, fold=f, seed=seed, best_epoch=ep, inner_rho=inner))
            reg[r, te] = np.mean([q[0] for q in preds], 0); prob[r, te] = np.mean([q[1] for q in preds], 0)
            fitm = np.zeros(n, bool); fitm[tr] = True
            preg[r, te], pprob[r, te] = probe(Xmean, s, y, fitm, te)
            print(json.dumps(dict(repeat=r, fold=f, n_targets=len(T) if T else 0,
                                  test_rho=rho_w(reg[r, te], y[te], s[te]), probe_rho=rho_w(preg[r, te], y[te], s[te]),
                                  epochs=[l["best_epoch"] for l in log[-a.seeds:]])), f"{(time.time() - t0) / 60:.1f} min",
                  flush=True)
            if a.smoke:
                break
        if a.smoke:
            break
    torch.save(dict(names=names, reg=reg, prob=prob, probe_reg=preg, probe_prob=pprob, fold_of=fold_of, log=log,
                    args=vars(a)), Path("cls_lang") / f"results_{a.arm}{'_smoke' if a.smoke else ''}.pt")


if __name__ == "__main__":
    main()
