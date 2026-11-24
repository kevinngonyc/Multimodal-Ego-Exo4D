"""Official-split report for `joint.py --official` (run from idea_1).

usage:  python cls_lang/official_report.py [TAG]     (default joint_comm_official)
Test takes = official val takes with ego + exo features. Level (4-class accuracy, balanced accuracy, within-scenario
Spearman) for the model's [LEVEL] output, the frozen PCA-64 level probe fit on the same split, and the scenario prior
(most frequent training level per scenario). Rating vs held-out experts (raw / per-expert z, rows = (take, expert) on
val takes with >= 2 experts, as rating.py --report) for the [RATING] output, the frozen ridge probe and one human
expert. 95% intervals: 1,000 bootstrap resamples of val participants. Also on the SkillFormer test half of val
(models/SkillFormer/annotation_test.jsonl), the subset our midterm SkillFormer reproduction was scored on.
-> stdout (cls_lang/official_report[_SUFFIX].log); CSV copies in final_report/tables/official_split[_SUFFIX]*.csv,
   SUFFIX = TAG minus "joint_comm_official_" (none for the default tag); per-seed spread when the run saved seed_out.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from train import A, LEVELS, SCEN, centred, load_ratings  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "joint_comm_official"
q = torch.load(Path("cls_lang") / f"rating_{TAG}.pt", weights_only=False)
names = q["names"]; n = len(names)
meta, split = {}, {}
for fn, sp_ in (("proficiency_demonstrator_train.json", "train"), ("proficiency_demonstrator_val.json", "val")):
    for m in json.load(open(A / fn, encoding="utf-8"))["annotations"]:
        t = m["video_paths"]["ego"].split("/")[1]
        meta[t] = (SCEN.index(m["scenario_name"]), LEVELS[m["proficiency_score"]], m["origin_participant_id"]); split[t] = sp_
s = np.array([meta[t][0] for t in names]); y = np.array([meta[t][1] for t in names]); p = np.array([meta[t][2] for t in names])
te = np.array([split[t] == "val" for t in names]); tr = ~te
assert np.isfinite(q["lvl_reg"][0][te]).all() and np.isnan(q["lvl_reg"][0][tr]).all()
with open("models/SkillFormer/annotation_test.jsonl", encoding="utf-8") as f:
    sf = {json.loads(line)["video_paths"][0].split("/")[1] for line in f}
half = np.array([t in sf for t in names]) & te

prior = np.zeros((n, 4))
for k in range(6):
    prior[s == k, np.bincount(y[tr & (s == k)], minlength=5)[1:].argmax()] = 1
LV = {"Final model [LEVEL]": (q["lvl_prob"][0], q["lvl_reg"][0]),
      "Frozen PCA level probe": (q["fz_prob"][0], q["fz_reg"][0]),
      "Scenario prior": (prior, None)}
RT = {"Final model [RATING]": q["reg"][0]}
if np.isfinite(q["probe_reg"][0][te]).all():  # skipped for --view ego|exo runs
    RT["Frozen ridge probe"] = q["probe_reg"][0]

R = load_ratings({t: j for j, t in enumerate(names)})
by_e = defaultdict(list)
for (j, e), r in R.items():
    by_e[e].append(r)
mu = {e: np.mean(v) for e, v in by_e.items()}; sd = {e: (np.std(v) or 1.0) for e, v in by_e.items()}


def rows(mask):
    by_t = defaultdict(list)
    for (j, e), r in R.items():
        if mask[j]:
            by_t[j].append((r, (r - mu[e]) / sd[e]))
    H = []
    for j, g in by_t.items():
        if len(g) < 2:
            continue
        for i, (r, z) in enumerate(g):
            o = [x for k, x in enumerate(g) if k != i]
            H.append((j, r, z, np.mean([x[0] for x in o]), np.mean([x[1] for x in o])))
    return np.array(H)


spr = lambda a, b: spearmanr(a, b)[0]
ROWS, SROWS = [], []  # -> final_report/tables/official_*.csv


def report(mask, title):
    idx = np.flatnonzero(mask)
    lev_c = centred(y[idx].astype(float), s[idx])
    H = rows(mask); hi = H[:, 0].astype(int); hs = s[hi]
    cols = [centred(H[:, c], hs) for c in (1, 2, 3, 4)]  # one, one_z, oth, oth_z
    rated = np.zeros(n, bool); rated[hi] = True
    U = np.unique(p[idx]); rng = np.random.default_rng(0); draws = [rng.choice(U, len(U)) for _ in range(1000)]
    pos = {u: np.flatnonzero(p[idx] == u) for u in U}; posh = {u: np.flatnonzero(p[hi] == u) for u in U}
    Bl = [np.concatenate([pos[u] for u in d]) for d in draws]; Bh = [np.concatenate([posh[u] for u in d]) for d in draws]

    def lvl(P, Rg, b):
        yy = y[idx][b]; pr = P[idx][b].argmax(1) + 1
        a_ = (pr == yy).mean(); ba = np.mean([(pr[yy == c] == c).mean() for c in range(1, 5) if (yy == c).any()])
        rh = spr(centred(Rg[idx], s[idx])[b], lev_c[b]) if Rg is not None else np.nan
        return np.array([a_, ba, rh])

    def rat(x, b):
        xc = x.copy(); xc[rated] = centred(x[rated], s[rated]); xh = xc[hi]
        return np.array([spr(xh[b], cols[2][b]), spr(xh[b], cols[3][b])])

    ci = lambda D, i: f"[{np.percentile(D[:, i], 2.5):.3f}, {np.percentile(D[:, i], 97.5):.3f}]"
    cid = lambda D, i: f"[{np.percentile(D[:, i], 2.5):+.3f}, {np.percentile(D[:, i], 97.5):+.3f}]"
    alll = np.arange(len(idx)); allh = np.arange(len(hi))
    print(f"\n== {title}: {len(idx)} takes, {len(U)} participants; {len(hi)} (take, expert) rows on {len(np.unique(hi))} takes")
    print("level counts", np.bincount(y[idx], minlength=5)[1:].tolist())
    print(f"{'level':26s} {'accuracy':>24s} {'balanced':>24s} {'Spearman':>24s}")
    L = {}
    for k, (P, Rg) in LV.items():
        v = lvl(P, Rg, alll); D = np.array([lvl(P, Rg, b) for b in Bl]); L[k] = D
        for i, met in enumerate(("level_acc", "level_bal_acc", "level_spearman")):
            if np.isfinite(v[i]):
                ROWS.append([title, k, met, v[i], *np.percentile(D[:, i], [2.5, 97.5])])
        print(f"{k:26s} " + " ".join(f"{v[i]:.3f} {ci(D, i):>17s}" if np.isfinite(v[i]) else f"{'-':>24s}" for i in range(3)))
    d = L["Final model [LEVEL]"] - L["Frozen PCA level probe"]
    v = lvl(*LV["Final model [LEVEL]"], alll) - lvl(*LV["Frozen PCA level probe"], alll)
    for i, met in enumerate(("level_acc", "level_bal_acc", "level_spearman")):
        ROWS.append([title, "final - frozen probe", met, v[i], *np.percentile(d[:, i], [2.5, 97.5])])
    print(f"{'final - frozen probe':26s} " + " ".join(f"{v[i]:+.3f} {cid(d, i):>17s}" for i in range(3)))
    d = L["Final model [LEVEL]"] - L["Scenario prior"]
    v = lvl(*LV["Final model [LEVEL]"], alll) - lvl(*LV["Scenario prior"], alll)
    print(f"{'final - scenario prior':26s} " + " ".join(f"{v[i]:+.3f} {cid(d, i):>17s}" for i in range(2)))
    print(f"{'rating vs experts':26s} {'raw':>24s} {'z':>24s}")
    hum = np.array([spr(cols[0], cols[2]), spr(cols[1], cols[3])])
    Dh = np.array([[spr(cols[0][b], cols[2][b]), spr(cols[1][b], cols[3][b])] for b in Bh])
    print(f"{'Human: one expert':26s} " + " ".join(f"{hum[i]:.3f} {ci(Dh, i):>17s}" for i in range(2)))
    for i, met in enumerate(("rating_raw", "rating_z")):
        ROWS.append([title, "Human: one expert vs the others", met, hum[i], *np.percentile(Dh[:, i], [2.5, 97.5])])
    Q = {}
    for k, x in RT.items():
        v = rat(x, allh); D = np.array([rat(x, b) for b in Bh]); Q[k] = (v, D)
        for i, met in enumerate(("rating_raw", "rating_z")):
            ROWS.append([title, k, met, v[i], *np.percentile(D[:, i], [2.5, 97.5])])
        print(f"{k:26s} " + " ".join(f"{v[i]:.3f} {ci(D, i):>17s}" for i in range(2)))
    if "Frozen ridge probe" in Q:
        v = Q["Final model [RATING]"][0] - Q["Frozen ridge probe"][0]; d = Q["Final model [RATING]"][1] - Q["Frozen ridge probe"][1]
        print(f"{'final - frozen probe':26s} " + " ".join(f"{v[i]:+.3f} {cid(d, i):>17s}" for i in range(2)))
        for i, met in enumerate(("rating_raw", "rating_z")):
            ROWS.append([title, "final - frozen probe (rating)", met, v[i], *np.percentile(d[:, i], [2.5, 97.5])])
    print("per scenario: takes, level acc final / probe / prior, level Spearman final / probe, rating z final / probe / human")
    for k in range(6):
        m = s[idx] == k; mh = hs == k
        if not m.any():
            continue
        a_ = [(P[idx][m].argmax(1) + 1 == y[idx][m]).mean() for P, _ in LV.values()]
        r_ = [spr(centred(Rg[idx], s[idx])[m], lev_c[m]) for P, Rg in list(LV.values())[:2]]
        z_ = [spr(rat_c(x, rated)[hi][mh], cols[3][mh]) for x in RT.values()] if mh.sum() >= 10 else [np.nan] * len(RT)
        z_ = (z_ + [np.nan])[:2]
        hz = spr(cols[1][mh], cols[3][mh]) if mh.sum() >= 10 else np.nan
        SROWS.append([title, SCEN[k], int(m.sum()), int(mh.sum()), *a_, *r_, *z_, hz])
        print(f"  {SCEN[k]:14s} {m.sum():4d}  acc {a_[0]:.3f} / {a_[1]:.3f} / {a_[2]:.3f}   rho {r_[0]:.3f} / {r_[1]:.3f}"
              f"   z {z_[0]:.3f} / {z_[1]:.3f} / {hz:.3f}")


def rat_c(x, rated):
    xc = x.copy(); xc[rated] = centred(x[rated], s[rated]); return xc


report(te, "official val")
report(half, "SkillFormer test half of val")

SEED_ROWS = []
if q.get("seed_out"):
    idx = np.flatnonzero(te); lev_c = centred(y[idx].astype(float), s[idx])
    H = rows(te); hi = H[:, 0].astype(int); oth_z = centred(H[:, 4], s[hi])
    rated = np.zeros(n, bool); rated[hi] = True
    print(f"\n== per training seed, official val ({len(q['seed_out'])} seeds; the run's reported model averages them)")
    for so in q["seed_out"]:
        pr_full = np.full((n, 4), np.nan); pr_full[idx] = so["lvl_prob"]
        lr_full = np.full(n, np.nan); lr_full[idx] = so["lvl_reg"]
        rg_full = np.full(n, np.nan); rg_full[idx] = so["reg"]
        pr = pr_full[idx].argmax(1) + 1; yy = y[idx]
        a_ = (pr == yy).mean(); ba = np.mean([(pr[yy == c] == c).mean() for c in range(1, 5) if (yy == c).any()])
        rh = spr(centred(lr_full[idx], s[idx]), lev_c)
        z = spr(rat_c(rg_full, rated)[hi], oth_z)
        SEED_ROWS.append([so["seed"], a_, ba, rh, z])
        print(f"  seed {so['seed']}: level acc {a_:.3f}  balanced {ba:.3f}  Spearman {rh:.3f}  rating z {z:.3f}")
    A_ = np.array([r_[1:] for r_ in SEED_ROWS])
    print("  mean +- sd: " + "  ".join(f"{m} {A_[:, i].mean():.3f} +- {A_[:, i].std(ddof=1):.3f} (min {A_[:, i].min():.3f}, max {A_[:, i].max():.3f})"
                                     for i, m in enumerate(("acc", "balanced", "Spearman", "rating z"))))

import csv  # noqa: E402
OUT = Path("final_report") / "tables"; OUT.mkdir(parents=True, exist_ok=True)
SUF = "" if TAG == "joint_comm_official" else "_" + TAG.replace("joint_comm_official_", "")
if SEED_ROWS:
    with open(OUT / f"official_split{SUF}_per_seed.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["seed", "level_acc", "level_bal_acc", "level_spearman", "rating_z"])
        for r_ in SEED_ROWS:
            w.writerow([r_[0]] + [f"{x:.3f}" for x in r_[1:]])
with open(OUT / f"official_split{SUF}.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["eval_set", "model", "metric", "value", "ci_lo", "ci_hi"])
    for r_ in ROWS:
        w.writerow(r_[:3] + [f"{x:.3f}" for x in r_[3:]])
with open(OUT / f"official_split{SUF}_per_scenario.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["eval_set", "scenario", "takes", "rating_rows", "acc_final", "acc_frozen_probe", "acc_scenario_prior",
                "spearman_final", "spearman_frozen_probe", "rating_z_final", "rating_z_frozen_probe", "rating_z_human"])
    for r_ in SROWS:
        w.writerow(r_[:4] + [f"{x:.3f}" for x in r_[4:]])
print(f"\nCSV tables written to {OUT}")
