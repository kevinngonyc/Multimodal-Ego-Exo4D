"""Pool cls_lang/summary_lw*.pt (usage from idea_1: python cls_lang/summary_report.py).
Within-scenario Spearman + accuracy of each level readout, paired participant bootstrap vs the frozen probe,
commentary diagnostics on held-out takes, best epochs."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

D = Path("cls_lang")
A = next(p / "data" for p in (Path.cwd(), *Path.cwd().parents) if (p / "data" / "annotations").is_dir()) / "annotations"
LEVELS = {"Novice": 1, "Early Expert": 2, "Intermediate Expert": 3, "Late Expert": 4}
meta = {}
for fn in ("proficiency_demonstrator_train.json", "proficiency_demonstrator_val.json"):
    for m in json.load(open(A / fn, encoding="utf-8"))["annotations"]:
        meta[m["video_paths"]["ego"].split("/")[1]] = (m["scenario_name"], LEVELS[m["proficiency_score"]], m["origin_participant_id"])

runs = {p.stem.replace("summary_", ""): torch.load(p, weights_only=False) for p in sorted(D.glob("summary_lw*.pt"))
        if "smoke" not in p.stem}
first = next(iter(runs.values()))
names = first["names"]
s = np.array([meta[t][0] for t in names]); y = np.array([meta[t][1] for t in names], float)
p = np.array([meta[t][2] for t in names])


def pooled(reg, prob, fold_of):
    P = reg.copy()
    for r in range(len(P)):
        for f in range(5):
            for k in np.unique(s):
                g = (fold_of[r] == f) & (s == k)
                P[r, g] -= P[r, g].mean()
    return P.mean(0), prob.mean(0)


M = {"frozen probe": pooled(first["reg"]["probe"], first["prob"]["probe"], first["fold_of"])}
for t, r in runs.items():
    for k in r["reg"]:
        if k != "probe":
            M[f"{t}: {k}"] = pooled(r["reg"][k], r["prob"][k], r["fold_of"])
lev_c = y.copy()
for k in np.unique(s):
    lev_c[s == k] -= y[s == k].mean()
U = np.unique(p); rows = {u: np.flatnonzero(p == u) for u in U}; rng = np.random.default_rng(0)
B = [np.concatenate([rows[u] for u in rng.choice(U, len(U))]) for _ in range(1000)]
sp = lambda x, i: spearmanr(x[i], lev_c[i])[0]
acc = lambda q, i: (q[i].argmax(1) + 1 == y[i]).mean()
allx = np.arange(len(y))
print(f"{len(y)} takes, {len(U)} participants")
print(f"{'readout':28s} {'rho':>6s} {'acc':>6s}   rho - probe [95% CI]        acc - probe [95% CI]")
for m, (rr, pp) in M.items():
    line = f"{m:28s} {sp(rr, allx):6.3f} {acc(pp, allx):6.3f}"
    if m != "frozen probe":
        a0, b0 = M["frozen probe"]
        dr = [sp(rr, i) - sp(a0, i) for i in B]; da = [acc(pp, i) - acc(b0, i) for i in B]
        line += (f"   {sp(rr, allx) - sp(a0, allx):+.3f} [{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
                 f"   {acc(pp, allx) - acc(b0, allx):+.3f} [{np.percentile(da, 2.5):+.3f}, {np.percentile(da, 97.5):+.3f}]")
    print(line)
print("per scenario rho")
for k in np.unique(s):
    i = np.flatnonzero(s == k)
    print(f"  {k:14s} n={len(i):4d} " + "  ".join(f"{m}={sp(M[m][0], i):.3f}" for m in M if "+" not in m))
for t, r in runs.items():
    L = r["log"]
    R2 = np.array([l["r2_held"] for l in L])
    ep = [e for l in L for e in l["epochs"]]
    print(f"{t}: best epoch median {np.median(ep):.0f} (range {min(ep)}-{max(ep)}); PCA var explained "
          f"{np.mean([l['pca_evr'] for l in L]):.3f}; held-out R2 per component (mean over folds): "
          + " ".join(f"{v:.2f}" for v in R2.mean(0)))
