"""Pool cls_lang/results_{arm}.pt: within-scenario Spearman + accuracy, paired participant bootstrap.

usage (from idea_1):  python cls_lang/report.py
Predictions centred within (repeat, fold, scenario), averaged over repeats (= the main probe protocol).
"""
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

res = {arm: torch.load(D / f"results_{arm}.pt", weights_only=False) for arm in ("level", "weighted", "hard")
       if (D / f"results_{arm}.pt").exists()}
first = next(iter(res.values()))
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


M = {"frozen probe": pooled(first["probe_reg"], first["probe_prob"], first["fold_of"])}
for arm, r in res.items():
    M[f"CLS: {arm}"] = pooled(r["reg"], r["prob"], r["fold_of"])
lev_c = y.copy()
for k in np.unique(s):
    lev_c[s == k] -= y[s == k].mean()
U = np.unique(p); rows = {u: np.flatnonzero(p == u) for u in U}; rng = np.random.default_rng(0)
B = [np.concatenate([rows[u] for u in rng.choice(U, len(U))]) for _ in range(1000)]
sp = lambda x, i: spearmanr(x[i], lev_c[i])[0]
acc = lambda q, i: (q[i].argmax(1) + 1 == y[i]).mean()
allx = np.arange(len(y))
print(f"{len(y)} takes, {len(U)} participants")
for m, (rr, pp) in M.items():
    print(f"  {m:18s} rho {sp(rr, allx):.3f}   acc {acc(pp, allx):.3f}")


def diff(a, b):
    dr = [sp(M[a][0], i) - sp(M[b][0], i) for i in B]
    da = [acc(M[a][1], i) - acc(M[b][1], i) for i in B]
    print(f"  {a} - {b}: rho {sp(M[a][0], allx) - sp(M[b][0], allx):+.3f} [{np.percentile(dr, 2.5):+.3f}, "
          f"{np.percentile(dr, 97.5):+.3f}]   acc {acc(M[a][1], allx) - acc(M[b][1], allx):+.3f} "
          f"[{np.percentile(da, 2.5):+.3f}, {np.percentile(da, 97.5):+.3f}]")


print("paired differences (95% participant bootstrap)")
for m in M:
    if m != "frozen probe":
        diff(m, "frozen probe")
for m in ("CLS: weighted", "CLS: hard"):
    if m in M and "CLS: level" in M:
        diff(m, "CLS: level")
if "CLS: weighted" in M and "CLS: hard" in M:
    diff("CLS: weighted", "CLS: hard")
print("per scenario rho")
for k in np.unique(s):
    i = np.flatnonzero(s == k)
    print(f"  {k:14s} n={len(i):4d} " + "  ".join(f"{m.replace('CLS: ', '')} {sp(M[m][0], i):.3f}" for m in M))
for arm, r in res.items():
    ep = [l["best_epoch"] for l in r["log"]]
    print(f"{arm}: best epoch median {np.median(ep):.0f} (range {min(ep)}-{max(ep)})")
