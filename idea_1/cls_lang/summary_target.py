"""Expert-summary target -> transformer CLS -> level probe (run from idea_1).

usage:  python cls_lang/summary_target.py [--level_weight 1.0] [--k 16] [--repeats 4] [--smoke]

Proxy for "Qwen summarises everything the experts said + the level into one vector": a take-level target built from
5 equally weighted blocks, all from the outer training fold only:
  level      one-hot official level (4)
  judgement  weighted mean of per-expert z-scored ratings; weighted share of praise (mean p_good over sentences)  (2)
  praise     weighted, p_good-weighted mean MiniLM embedding of the take's comment sentences (384)
  tip        same with (1 - p_good) (384)
  why        weighted mean MiniLM embedding of the experts' proficiency.why (384)
Expert weight w_e = max(0, within-scenario Spearman(rating_e, level)) on the outer training takes (Soccer -> 0).
Each block: centred within scenario (training means; missing -> 0), every dim standardised, block / sqrt(dims)
(equal block variance; --level_weight scales the level block's variance). PCA(k) fit on training takes -> target.
Model: train.Net's encoder on frozen ego+exo K600 clip tokens; CLS -> Linear(k), MSE. Trained only on takes with
weighted commentary (W > 0). Cross-fitting: the outer training takes are split 5 ways by participant; net i trains on
4/5 (early-stopped on the held-out 1/5) and predicts that 1/5, so every training take gets an out-of-sample summary
for the level readout; test takes get the mean of the 5 nets. Level readout = ridge / logistic on scenario one-hot +
predicted summary (and on top of the frozen PCA probe), as in train.py.
-> cls_lang/summary_lw{level_weight}.pt
"""
import argparse
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from train import A, DEV, SCEN, Net, batchify, expert_rho, load_ratings, load_takes, rho_w  # noqa: E402
from pretrain import blockify, readout  # noqa: E402

WHY_CACHE = Path("embeddings") / "cls_lang_why.pt"
FEATS = ("probe", "summary", "probe+summary")


def why_embeddings(take_ix):
    if not WHY_CACHE.exists():
        keys, texts = [], []
        for f in glob.glob(str(A / "expert_commentary" / "*" / "*" / "data.json")):
            take = Path(f).parent.parent.name
            if take not in take_ix:
                continue
            d = json.load(open(f, encoding="utf-8"))
            w = ((d.get("proficiency") or {}).get("why") or "").strip()
            if w:
                keys.append((take, d.get("user_id"))); texts.append(w)
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
        E = st.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        torch.save({k: e.astype(np.float16) for k, e in zip(keys, E)}, WHY_CACHE)
        del st; torch.cuda.empty_cache()
    W = torch.load(WHY_CACHE, weights_only=False)
    return {(take_ix[t], e): v.astype(np.float32) for (t, e), v in W.items()}


class Commentary:
    """per (take, expert) sufficient statistics of the comment sentences, z-rating and why embedding."""
    def __init__(self, take_ix, R):
        S = torch.load(Path("embeddings") / "cls_lang_sentences.pt", weights_only=False)
        ti = np.array([take_ix.get(t, -1) for t in S["take"]])
        pg = S["p_good"].astype(np.float32); E = S["emb"].astype(np.float32)
        acc = {}
        for k in np.flatnonzero(ti >= 0):
            key = (int(ti[k]), S["expert"][k])
            a = acc.setdefault(key, [np.zeros(384, np.float32), 0.0, np.zeros(384, np.float32), 0.0, 0])
            a[0] += pg[k] * E[k]; a[1] += pg[k]; a[2] += (1 - pg[k]) * E[k]; a[3] += 1 - pg[k]; a[4] += 1
        self.sent = acc
        by_e = defaultdict(list)
        for (j, e), r in R.items():
            by_e[e].append(r)
        mu = {e: np.mean(v) for e, v in by_e.items()}; sd = {e: np.std(v) + 1e-6 for e, v in by_e.items()}
        self.z = {(j, e): (r - mu[e]) / sd[e] for (j, e), r in R.items()}
        self.why = why_embeddings(take_ix)
        self.experts = defaultdict(set)
        for (j, e) in list(self.sent) + list(self.z) + list(self.why):
            self.experts[j].add(e)

    def blocks(self, n, w):
        """w(expert) -> weight. Returns dict of (n, d) arrays with NaN where missing, and the per-take total weight."""
        jud = np.full((n, 2), np.nan); pra = np.full((n, 384), np.nan); tip = np.full((n, 384), np.nan)
        why = np.full((n, 384), np.nan); Wt = np.zeros(n)
        for j, es in self.experts.items():
            ws = {e: w(j, e) for e in es}
            Wt[j] = sum(ws.values())
            if Wt[j] <= 0:
                continue
            zn = sum(ws[e] * self.z[(j, e)] for e in es if (j, e) in self.z); zd = sum(ws[e] for e in es if (j, e) in self.z)
            st = [(ws[e], self.sent[(j, e)]) for e in es if (j, e) in self.sent and ws[e] > 0]
            if zd > 0:
                jud[j, 0] = zn / zd
            if st:
                jud[j, 1] = sum(x * a[1] for x, a in st) / sum(x * a[4] for x, a in st)
                if sum(x * a[1] for x, a in st) > 0:
                    pra[j] = sum(x * a[0] for x, a in st) / sum(x * a[1] for x, a in st)
                if sum(x * a[3] for x, a in st) > 0:
                    tip[j] = sum(x * a[2] for x, a in st) / sum(x * a[3] for x, a in st)
            wy = [(ws[e], self.why[(j, e)]) for e in es if (j, e) in self.why and ws[e] > 0]
            if wy:
                why[j] = sum(x * v for x, v in wy) / sum(x for x, _ in wy)
        return dict(judgement=jud, praise=pra, tip=tip, why=why), Wt


def build_target(com, s, y, tr, R, k, level_weight):
    n = len(s)
    rho = expert_rho(R, s, y, tr)
    soccer = SCEN.index("Soccer")
    w = lambda j, e: 0.0 if s[j] == soccer else max(0.0, rho.get(e, (0.0,))[0])
    blocks, Wt = com.blocks(n, w)
    blocks = {"level": np.eye(4)[y.astype(int) - 1], **blocks}
    trm = np.zeros(n, bool); trm[tr] = True
    valid = Wt > 0
    parts = []
    for name, B in blocks.items():
        B = B.astype(np.float64).copy()
        ok = ~np.isnan(B).any(1)
        for sc in range(6):
            m = s == sc
            ref = m & trm & ok
            B[m] -= B[ref].mean(0) if ref.any() else 0.0
        B[np.isnan(B)] = 0.0
        sd = B[trm & ok].std(0) + 1e-6
        B = B / sd / np.sqrt(B.shape[1])
        if name == "level":
            B *= np.sqrt(level_weight)
        parts.append(B)
    Y = np.hstack(parts)
    fitv = trm & valid
    pca = PCA(k, svd_solver="full").fit(Y[fitv])
    return pca.transform(Y).astype(np.float32), valid, pca.explained_variance_ratio_.sum()


class SumNet(Net):
    def __init__(self, k):
        super().__init__()
        self.sum = nn.Linear(self.d, k)

    def summary(self, x, tpos, mask, s):
        h = self.inp(x) + self.time_enc(tpos)
        h = torch.cat([self.cls + self.scen(s)[:, None], h], 1)
        m = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=x.device), ~mask], 1)
        return self.sum(self.norm(self.enc(h, src_key_padding_mask=m))[:, 0])


@torch.no_grad()
def predict_sum(net, idx, Xs, s, bs=32):
    net.eval()
    order = sorted(idx, key=lambda j: len(Xs[j]))
    out = {}
    for i in range(0, len(order), bs):
        ch = order[i:i + bs]
        b = batchify(ch, Xs, s)
        for j, v in zip(ch, net.summary(b["x"], b["t"], b["mask"], b["s"]).cpu().numpy()):
            out[j] = v
    net.train()
    return np.array([out[j] for j in idx])


def train_sum(Xs, s, Yt, fit_i, val_i, k, seed, epochs, patience, smoke):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SumNet(k).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    Yg = torch.as_tensor(Yt, device=DEV)
    best, best_state, bad, best_ep = np.inf, None, 0, 0
    for ep in range(1, epochs + 1):
        perm = rng.permutation(fit_i)
        for i in range(0, len(perm), 16):
            idx = perm[i:i + 16]
            b = batchify(idx, Xs, s, crop=384, rng=rng)
            loss = F.mse_loss(net.summary(b["x"], b["t"], b["mask"], b["s"]), Yg[torch.as_tensor(idx, device=DEV)])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        cur = float(((predict_sum(net, val_i, Xs, s) - Yt[val_i]) ** 2).mean())
        if cur < best - 1e-5:
            best, best_state, bad, best_ep = cur, {kk: v.clone() for kk, v in net.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= patience:
                break
        if smoke and ep >= 2:
            break
    net.load_state_dict(best_state)
    return net, best_ep, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level_weight", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    names, Xraw, s, y, p = load_takes()
    n = len(names)
    take_ix = {t: j for j, t in enumerate(names)}
    R = load_ratings(take_ix)
    com = Commentary(take_ix, R)
    Xmean = np.stack([x.float().mean(0).numpy() for x in Xraw])
    print(f"{n} takes; summary target, level weight {a.level_weight}, k {a.k}; takes with any expert data "
          f"{len(com.experts)}, with why {len({j for j, _ in com.why})}", flush=True)
    reg = {f: np.zeros((a.repeats, n)) for f in FEATS}; prob = {f: np.zeros((a.repeats, n, 4)) for f in FEATS}
    fold_of = np.zeros((a.repeats, n), int); log = []
    t0 = time.time()
    for r in range(a.repeats):
        for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(n), y, p)):
            fold_of[r, te] = f
            mu = np.stack([Xmean[tr][s[tr] == q].mean(0) for q in range(6)])
            sd = (Xmean[tr] - mu[s[tr]]).std(0) + 1e-6
            mu_t, sd_t = torch.as_tensor(mu, dtype=torch.float32), torch.as_tensor(sd, dtype=torch.float32)
            Xs = [(x.float() - mu_t[s[j]]) / sd_t for j, x in enumerate(Xraw)]
            Yt, valid, evr = build_target(com, s, y, tr, R, a.k, a.level_weight)
            pred = np.zeros((n, a.k), np.float32); test_preds, eps = [], []
            for i, (a_rel, b_rel) in enumerate(StratifiedGroupKFold(a.inner, shuffle=True, random_state=100 + r)
                                               .split(np.zeros(len(tr)), y[tr], p[tr])):
                part_fit, part_out = tr[a_rel], tr[b_rel]
                fv, ov = part_fit[valid[part_fit]], part_out[valid[part_out]]
                net, ep, _ = train_sum(Xs, s, Yt, fv, ov, a.k, 1000 * r + 10 * f + i, a.epochs, a.patience, a.smoke)
                pred[part_out] = predict_sum(net, part_out, Xs, s)
                test_preds.append(predict_sum(net, te, Xs, s)); eps.append(ep)
                del net; torch.cuda.empty_cache()
            pred[te] = np.mean(test_preds, 0)
            tv = te[valid[te]]
            r2 = 1 - ((pred[tv] - Yt[tv]) ** 2).sum(0) / ((Yt[tv] - Yt[tv].mean(0)) ** 2).sum(0)
            fitm = np.zeros(n, bool); fitm[tr] = True
            P = [(s[:, None] == np.arange(6)[None]).astype(float)]
            for v in (slice(0, 768), slice(768, 1536)):
                Bv = blockify(Xmean[:, v], s, fitm)
                P.append(PCA(64, svd_solver="full").fit(Bv[fitm]).transform(Bv))
            P = np.hstack(P)
            Sm = blockify(pred, s, fitm)
            Zs = {"probe": P, "summary": np.hstack([P[:, :6], Sm]), "probe+summary": np.hstack([P, Sm])}
            for ft in FEATS:
                reg[ft][r, te], prob[ft][r, te] = readout(Zs[ft], y, fitm, te)
            log.append(dict(repeat=r, fold=f, epochs=eps, pca_evr=float(evr), n_valid_train=int(valid[tr].sum()),
                            r2_held=r2.tolist()))
            print(json.dumps(dict(repeat=r, fold=f, epochs=eps, pca_evr=round(float(evr), 3),
                                  r2_pc1_3=[round(float(x), 3) for x in r2[:3]], r2_mean=round(float(r2.mean()), 3),
                                  **{f"rho_{ft}": round(rho_w(reg[ft][r, te], y[te], s[te]), 3) for ft in FEATS})),
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if a.smoke:
                break
        if a.smoke:
            break
    tag = f"lw{a.level_weight:g}{'_smoke' if a.smoke else ''}"
    torch.save(dict(names=names, reg=reg, prob=prob, fold_of=fold_of, log=log, args=vars(a)),
               Path("cls_lang") / f"summary_{tag}.pt")


if __name__ == "__main__":
    main()
