"""Official level read out from the frozen CLS token of the rating model (run from idea_1, after
`python cls_lang/rating.py --arm rating --head per_scenario --pred_scenario --level_probe --tag lvlprobe`).

Per fold and seed: CLS token (128) of every take from that fold's rating network (per-scenario heads, scenario predicted
from video at test time) -> 4-level LogisticRegressionCV + RidgeCV on [predicted-scenario one-hot, CLS], fit on the
fold's training takes; probabilities / scores averaged over seeds and repeats. Reference in the same folds: the same
readout on the frozen PCA-64 ego + exo features. Also: scenario-only (most frequent training level per scenario) and the
earlier level-trained CLS models (cls_lang/results_{level,hard}.pt, same 4 x 5 folds).
Metrics on all 2,111 takes: 4-class accuracy, balanced accuracy, within-scenario Spearman; participant bootstrap.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from train import A, LEVELS, SCEN  # noqa: E402

R = torch.load("cls_lang/rating_rating_lvlprobe.pt", weights_only=False)
names = R["names"]
meta = {}
for fn in ("proficiency_demonstrator_train.json", "proficiency_demonstrator_val.json"):
    for m in json.load(open(A / fn, encoding="utf-8"))["annotations"]:
        meta[m["video_paths"]["ego"].split("/")[1]] = (SCEN.index(m["scenario_name"]), LEVELS[m["proficiency_score"]],
                                                       m["origin_participant_id"])
s = np.array([meta[t][0] for t in names]); y = np.array([meta[t][1] for t in names]); p = np.array([meta[t][2] for t in names])
n, reps = len(y), len(R["reg"])
fold_of = R["fold_of"]

# scenario-only: most frequent training level per scenario (as one-hot probabilities)
so = np.zeros((reps, n, 4))
for r in range(reps):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(n), y, p):
        for k in range(6):
            mode = np.bincount(y[tr][s[tr] == k], minlength=5)[1:].argmax()
            so[r, te[s[te] == k], mode] = 1


def pooled(reg):
    P = reg.copy()
    for r in range(len(P)):
        for f in range(5):
            for k in range(6):
                g = (fold_of[r] == f) & (s == k)
                P[r, g] -= P[r, g].mean()
    return P.mean(0)


M = {"scenario only": (None, so.mean(0)),
     "frozen PCA probe": (pooled(R["fz_reg"]), R["fz_prob"].mean(0)),
     "rating CLS -> level probe": (pooled(R["lvl_reg"]), R["lvl_prob"].mean(0))}
for arm, lab in (("level", "level-trained CLS (earlier)"), ("hard", "level + hard commentary CLS (earlier)")):
    f = Path("cls_lang") / f"results_{arm}.pt"
    if f.exists():
        q = torch.load(f, weights_only=False)
        assert list(q["names"]) == list(names)
        M[lab] = (pooled(q["reg"]), q["prob"].mean(0))
JOINT = {"joint": "joint model, [LEVEL] token", "joint_comm": "joint + commentary, [LEVEL]",
         "joint_lam1": "joint + commentary lam 1, [LEVEL]", "joint_lam2": "joint + commentary lam 2, [LEVEL]",
         "joint_lam4": "joint + commentary lam 4, [LEVEL]", "joint_comm_rank": "joint + commentary + rank, [LEVEL]",
         "joint_comm_vjepa": "joint + commentary V-JEPA, [LEVEL]"}
for tag, lab in JOINT.items():
    f = Path("cls_lang") / f"rating_{tag}.pt"
    if f.exists():
        q = torch.load(f, weights_only=False)
        assert list(q["names"]) == list(names)
        M[lab] = (pooled(q["lvl_reg"]), q["lvl_prob"].mean(0))

lev_c = y.astype(float).copy()
for k in range(6):
    lev_c[s == k] -= y[s == k].mean()
U = np.unique(p); rows = {u: np.flatnonzero(p == u) for u in U}; rng = np.random.default_rng(0)
B = [np.concatenate([rows[u] for u in rng.choice(U, len(U))]) for _ in range(1000)]
acc = lambda P, i: (P[i].argmax(1) + 1 == y[i]).mean()
bacc = lambda P, i: np.mean([(P[i][y[i] == c].argmax(1) + 1 == c).mean() for c in range(1, 5) if (y[i] == c).any()])
sp = lambda x, i: spearmanr(x[i], lev_c[i])[0]
allx = np.arange(n)
print(f"{n} takes, {len(U)} participants; level counts {np.bincount(y)[1:].tolist()}")
print(f"{'readout':40s} {'accuracy':>9s} {'balanced':>9s} {'Spearman':>9s}")
for k, (rg, pr) in M.items():
    print(f"{k:40s} {acc(pr, allx):9.3f} {bacc(pr, allx):9.3f} {(sp(rg, allx) if rg is not None else float('nan')):9.3f}")
for ref in ("frozen PCA probe", "level + hard commentary CLS (earlier)", "joint model, [LEVEL] token", "joint + commentary, [LEVEL]"):
  if ref not in M:
    continue
  print(f"paired vs {ref} (95% participant bootstrap)")
  for k, (rg, pr) in M.items():
    if k == ref or (ref != "frozen PCA probe" and not k.startswith("joint")) or (ref.startswith("joint") and list(M).index(k) < list(M).index(ref)):
        continue
    da = [acc(pr, i) - acc(M[ref][1], i) for i in B]
    db = [bacc(pr, i) - bacc(M[ref][1], i) for i in B]
    line = (f"  {k:38s} acc {acc(pr, allx) - acc(M[ref][1], allx):+.3f} [{np.percentile(da, 2.5):+.3f}, {np.percentile(da, 97.5):+.3f}]"
            f"   balanced {bacc(pr, allx) - bacc(M[ref][1], allx):+.3f} [{np.percentile(db, 2.5):+.3f}, {np.percentile(db, 97.5):+.3f}]")
    if rg is not None:
        ds = [sp(rg, i) - sp(M[ref][0], i) for i in B]
        line += f"   rho {sp(rg, allx) - sp(M[ref][0], allx):+.3f} [{np.percentile(ds, 2.5):+.3f}, {np.percentile(ds, 97.5):+.3f}]"
    print(line)
print("per scenario accuracy")
for k in range(6):
    i = np.flatnonzero(s == k)
    print(f"  {SCEN[k]:14s} n={len(i):4d} " + "  ".join(f"{m.split(' (')[0]} {acc(pr, i):.3f}" for m, (_, pr) in M.items()))
