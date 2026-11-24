"""MLP decoder vs the ridge / logistic probe on frozen TimeSformer K600 features, official level.

usage (from idea_1):  python decoder/mlp_probe.py [--repeats 4] [--smoke]
Protocol = motion_stats.ipynb: 2,111 official takes, 4 x 5-fold StratifiedGroupKFold by participant, scenario-centred
PCA-64 per block fit in the fold, scenario one-hot, standardised; predictions centred within (repeat, fold, scenario);
within-scenario Spearman + 4-class accuracy; participant bootstrap of paired differences.
Inputs:  base = ego + exo L12 patch-token std (the current best, 0.452 with ridge)
         all  = ego + exo x (L6, L12) x (CLS, mean, std): 12 blocks
MLP: d -> H -> GELU -> dropout -> (1 regression + 4 class logits), loss MSE + CE, full-batch AdamW, 5-seed ensemble.
H, dropout, weight decay and the number of epochs are chosen per outer fold by inner 3-fold GroupKFold (participant)
on within-scenario Spearman; nothing is tuned on the held-out fold.
-> decoder/mlp_probe_results.pt
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(8)
DATA_ROOT = next(p / "data" for p in (Path.cwd(), *Path.cwd().parents) if (p / "data" / "annotations").is_dir())
LEVELS = {"Novice": 1, "Early Expert": 2, "Intermediate Expert": 3, "Late Expert": 4}
GRID = list(itertools.product((32, 128, 512), (0.2, 0.5), (1e-3, 1e-2, 1e-1)))  # hidden, dropout, weight decay
EPOCHS = (25, 50, 100, 200, 400)
SEEDS = 5


def load():
    meta = {}
    for split, fname in (("train", "proficiency_demonstrator_train.json"), ("val", "proficiency_demonstrator_val.json")):
        for m in json.loads((DATA_ROOT / "annotations" / fname).read_text(encoding="utf-8"))["annotations"]:
            meta[m["video_paths"]["ego"].split("/")[1]] = dict(split=split, scenario=m["scenario_name"],
                                                               score=LEVELS[m["proficiency_score"]],
                                                               participant=m["origin_participant_id"])
    feats = {}
    for view in ("exo", "ego"):
        out = {}
        for name in (f"timesformer_k600_{view}8fps_clips.pt", f"timesformer_k600_{view}8fps_clips_extra.pt"):
            c = torch.load(Path("embeddings") / name, mmap=True)["clips"]["squash"]
            out.update({t: c[t].float().mean(0).numpy() for t in c if t in meta})  # (2, 3, 768)
        feats[view] = out
    names = sorted(t for t in meta if all(t in f for f in feats.values()))
    M = {k: np.array([meta[t][k] for t in names]) for k in ("split", "scenario", "score", "participant")}
    X = {v: np.stack([feats[v][t] for t in names]) for v in feats}
    return names, M, X


def centred(v, groups):
    v = np.asarray(v, float).copy()
    for g in np.unique(groups):
        v[groups == g] -= v[groups == g].mean()
    return v


def design(X, blocks, scen, fit):
    S = np.unique(scen)
    parts = [(scen[:, None] == S[None]).astype(float)]
    for view, l, s in blocks:
        B = X[view][:, l, s].astype(np.float64).copy()
        for k in S:
            m = scen == k
            B[m] -= B[fit & m].mean(0)
        parts.append(PCA(64, svd_solver="full").fit(B[fit]).transform(B))
    Z = np.hstack(parts)
    return StandardScaler().fit(Z[fit]).transform(Z)


def train_mlp(Z, y, H, drop, wd, epochs, seed):
    """full-batch; returns predictions function snapshots at each epoch in `epochs` (sorted)."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Z.shape[1], H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, 5))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    Zt, yt = torch.as_tensor(Z, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    ymu = float(yt.mean())
    snaps, done = {}, 0
    for e in sorted(epochs):
        net.train()
        for _ in range(e - done):
            out = net(Zt)
            loss = F.mse_loss(out[:, 0] + ymu, yt) + F.cross_entropy(out[:, 1:], (yt - 1).long())
            opt.zero_grad(); loss.backward(); opt.step()
        done = e
        snaps[e] = {k: v.clone() for k, v in net.state_dict().items()}
    def predict(Zq, e):
        net.load_state_dict(snaps[e]); net.eval()
        with torch.no_grad():
            out = net(torch.as_tensor(Zq, dtype=torch.float32))
        return out[:, 0].numpy() + ymu, F.softmax(out[:, 1:], 1).numpy()
    return predict


def rho_within(pred, y, scen):
    return spearmanr(centred(pred, scen), centred(y, scen))[0]


def select(Z, y, scen, part, fit_idx):
    """inner 3-fold GroupKFold -> best (H, drop, wd, epochs) by within-scenario Spearman."""
    score = np.zeros((len(GRID), len(EPOCHS)))
    for tr, va in GroupKFold(3).split(fit_idx, groups=part[fit_idx]):
        tr, va = fit_idx[tr], fit_idx[va]
        for gi, (H, drop, wd) in enumerate(GRID):
            pred = train_mlp(Z[tr], y[tr], H, drop, wd, EPOCHS, seed=0)
            for ei, e in enumerate(EPOCHS):
                score[gi, ei] += rho_within(pred(Z[va], e)[0], y[va], scen[va]) / 3
    gi, ei = np.unravel_index(score.argmax(), score.shape)
    return (*GRID[gi], EPOCHS[ei]), float(score[gi, ei])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    global GRID, EPOCHS
    if a.smoke:
        GRID, EPOCHS = GRID[:2], (25, 50)
    names, M, X = load()
    y, scen, part = M["score"].astype(float), M["scenario"], M["participant"]
    n = len(names)
    print(f"{n} takes, {len(np.unique(part))} participants", flush=True)
    INPUTS = {"base": [(v, 1, 2) for v in ("ego", "exo")],
              "all": [(v, l, s) for v in ("ego", "exo") for l in (0, 1) for s in (0, 1, 2)]}
    METHODS = [f"{i}: {m}" for i in INPUTS for m in ("ridge", "mlp", "ridge+mlp")]
    reg = {m: np.zeros((a.repeats, n)) for m in METHODS}
    prob = {m: np.zeros((a.repeats, n, 4)) for m in METHODS}
    fold_of = np.zeros((a.repeats, n), int)
    chosen = []
    t0 = time.time()
    for r in range(a.repeats):
        for f, (fit_i, held) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(n), y, part)):
            fold_of[r, held] = f
            fit = np.zeros(n, bool); fit[fit_i] = True
            for iname, blocks in INPUTS.items():
                Z = design(X, blocks, scen, fit)
                rr = RidgeCV(alphas=np.logspace(-2, 6, 17)).fit(Z[fit], y[fit]).predict(Z[held])
                lr = LogisticRegressionCV(Cs=np.logspace(-4, 2, 7), cv=5, max_iter=5000, n_jobs=-1).fit(Z[fit], y[fit])
                lp = lr.predict_proba(Z[held])
                cfg, inner = select(Z, y, scen, part, fit_i)
                H, drop, wd, ep = cfg
                outs = [train_mlp(Z[fit_i], y[fit_i], H, drop, wd, (ep,), seed=s)(Z[held], ep) for s in range(SEEDS)]
                mr, mp = np.mean([o[0] for o in outs], 0), np.mean([o[1] for o in outs], 0)
                # ridge+mlp: average of standardised within-scenario predictions / of probabilities
                zr = centred(rr, scen[held]); zm = centred(mr, scen[held])
                ens = zr / (zr.std() + 1e-9) + zm / (zm.std() + 1e-9)
                for m, (pr, pp) in {"ridge": (rr, lp), "mlp": (mr, mp), "ridge+mlp": (ens, (lp + mp) / 2)}.items():
                    reg[f"{iname}: {m}"][r, held], prob[f"{iname}: {m}"][r, held] = pr, pp
                chosen.append({"repeat": r, "fold": f, "input": iname, "H": H, "dropout": drop, "wd": wd, "epochs": ep,
                               "inner_rho": inner})
                print(json.dumps(chosen[-1]), f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if a.smoke:
                break
        if a.smoke:
            break
    torch.save({"names": names, "reg": reg, "prob": prob, "fold_of": fold_of, "chosen": chosen},
               Path("decoder") / f"mlp_probe_results{'_smoke' if a.smoke else ''}.pt")
    if a.smoke:
        return
    # pooled metrics
    lev_c = centred(y, scen)
    def pooled(m):
        P = reg[m].copy()
        for r in range(a.repeats):
            for f in range(5):
                for k in np.unique(scen):
                    g = (fold_of[r] == f) & (scen == k)
                    P[r, g] -= P[r, g].mean()
        return P.mean(0), prob[m].mean(0)
    Pm = {m: pooled(m) for m in METHODS}
    U = np.unique(part); rows = {u: np.flatnonzero(part == u) for u in U}; rng = np.random.default_rng(0)
    B = [np.concatenate([rows[u] for u in rng.choice(U, len(U))]) for _ in range(1000)]
    sp = lambda x, i: spearmanr(x[i], lev_c[i])[0]
    acc = lambda p, i: (p[i].argmax(1) + 1 == y[i]).mean()
    allidx = np.arange(n)
    print(f"\n{'method':22s} {'rho':>7s} {'acc':>7s}")
    for m in METHODS:
        print(f"{m:22s} {sp(Pm[m][0], allidx):7.3f} {acc(Pm[m][1], allidx):7.3f}")
    print("\npaired vs base: ridge (participant bootstrap 95% CI)")
    ref = "base: ridge"
    for m in METHODS:
        if m == ref:
            continue
        dr = [sp(Pm[m][0], i) - sp(Pm[ref][0], i) for i in B]
        da = [acc(Pm[m][1], i) - acc(Pm[ref][1], i) for i in B]
        print(f"{m:22s} rho {sp(Pm[m][0], allidx) - sp(Pm[ref][0], allidx):+.3f} [{np.percentile(dr, 2.5):+.3f}, "
              f"{np.percentile(dr, 97.5):+.3f}]   acc {acc(Pm[m][1], allidx) - acc(Pm[ref][1], allidx):+.3f} "
              f"[{np.percentile(da, 2.5):+.3f}, {np.percentile(da, 97.5):+.3f}]")
    print("\nper scenario rho")
    for k in np.unique(scen):
        mk = np.flatnonzero(scen == k)
        print(f"  {k:14s} n={len(mk):4d} " + "  ".join(f"{m.split(': ')[0][0]}:{m.split(': ')[1]} {sp(Pm[m][0], mk):.3f}"
                                                         for m in METHODS))
    C = chosen
    for iname in INPUTS:
        cs = [c for c in C if c["input"] == iname]
        print(iname, "chosen H", np.unique([c["H"] for c in cs], return_counts=True),
              "epochs", np.unique([c["epochs"] for c in cs], return_counts=True))


if __name__ == "__main__":
    main()
