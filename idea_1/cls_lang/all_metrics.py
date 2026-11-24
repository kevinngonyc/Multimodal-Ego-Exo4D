"""Every model on both targets, same folds (run from idea_1, CPU, after the runs it reads).

Rating: the model's rating output (for level-only models, its level score) vs held-out experts, raw and per-expert z,
exactly as `rating.py --report`. Level: the model's own level classifier and score where it has one; for rating-only
models, Spearman of the rating score with level and a quantile readout for accuracy: within each (repeat, fold,
scenario) the test takes are ranked by score and cut at that scenario's training-fold level proportions (training labels
only), then the one-hot classes are averaged over repeats. Level Spearman on all 2,111 takes. Paired differences vs the
final model (joint + commentary weight 0.5), 1,000 participant bootstrap resamples.
-> cls_lang/all_metrics_report.log (stdout); CSV copies in final_report/tables/cv_*.csv
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from train import A, LEVELS, SCEN, centred, load_ratings  # noqa: E402

C = Path("cls_lang")
ld = lambda f: torch.load(C / f, weights_only=False)
LP = ld("rating_rating_lvlprobe.pt")
names = LP["names"]
meta = {}
for fn in ("proficiency_demonstrator_train.json", "proficiency_demonstrator_val.json"):
    for m in json.load(open(A / fn, encoding="utf-8"))["annotations"]:
        meta[m["video_paths"]["ego"].split("/")[1]] = (SCEN.index(m["scenario_name"]), LEVELS[m["proficiency_score"]],
                                                       m["origin_participant_id"])
s = np.array([meta[t][0] for t in names]); y = np.array([meta[t][1] for t in names]); p = np.array([meta[t][2] for t in names])
n = len(names); reps = 4
fold_of = LP["fold_of"]
for r in range(reps):  # the folds every file uses
    fo = np.zeros(n, int)
    for f, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(n), y, p)):
        fo[te] = f
    assert (fo == fold_of[r]).all()

# expert-rating rows (as rating.py report)
R = load_ratings({t: j for j, t in enumerate(names)})
rated = np.zeros(n, bool)
for (j, _) in R:
    rated[j] = True
by_e = defaultdict(list)
for (j, e), r in R.items():
    by_e[e].append(r)
mu = {e: np.mean(v) for e, v in by_e.items()}; sd = {e: (np.std(v) or 1.0) for e, v in by_e.items()}
by_t = defaultdict(list)
for (j, e), r in R.items():
    by_t[j].append((e, r, (r - mu[e]) / sd[e]))
H = []
for j, g in by_t.items():
    if len(g) < 2:
        continue
    for i, (e, r, z) in enumerate(g):
        o = [x for k, x in enumerate(g) if k != i]
        H.append((j, r, z, np.mean([x[1] for x in o]), np.mean([x[2] for x in o])))
H = np.array(H); hi = H[:, 0].astype(int); hs = s[hi]
oth, oth_z = centred(H[:, 3], hs), centred(H[:, 4], hs)
one, one_z = centred(H[:, 1], hs), centred(H[:, 2], hs)
lev_c = y.astype(float).copy()
for k in range(6):
    lev_c[s == k] -= y[s == k].mean()


def pooled(P0, sub):
    P = np.array(P0, float).copy()
    for r in range(reps):
        for f in range(5):
            for k in range(6):
                g = (fold_of[r] == f) & (s == k) & sub
                P[r, g] -= P[r, g].mean()
    return P.mean(0)


def quantile_prob(reg):
    out = np.zeros((reps, n, 4))
    for r in range(reps):
        for f in range(5):
            te, tr = fold_of[r] == f, fold_of[r] != f
            for k in range(6):
                g = np.flatnonzero(te & (s == k))
                if not len(g):
                    continue
                prop = np.bincount(y[tr & (s == k)], minlength=5)[1:] / (tr & (s == k)).sum()
                order = g[np.argsort(reg[r, g], kind="stable")]
                cut = np.round(np.cumsum(prop) * len(g)).astype(int)
                cls = np.searchsorted(cut, np.arange(len(g)), side="right")
                out[r, order, np.minimum(cls, 3)] = 1
    return out.mean(0)


# model -> (rating score (reps,n), level score (reps,n), level probs (n,4), note)
M = {}
fz = LP["fz_reg"]
q = ld("rating_rating.pt")
M["Frozen probe, shared ridge"] = (q["probe_reg"], fz, LP["fz_prob"].mean(0), "rating ridge; level PCA probe")
M["Frozen probe, per-scenario ridge"] = (LP["probe_ps_reg"], LP["probe_ps_reg"], quantile_prob(LP["probe_ps_reg"]), "rating only")
M["Rating-only CLS, shared head"] = (q["reg"], q["reg"], quantile_prob(q["reg"]), "rating only")
q = ld("rating_rating_comm.pt")
M["Rating-only CLS + all-expert commentary"] = (q["reg"], q["reg"], quantile_prob(q["reg"]), "rating only")
q = ld("rating_rating_ps.pt")
M["Rating-only CLS, per-scenario heads, true scenario"] = (q["reg"], q["reg"], quantile_prob(q["reg"]), "rating only")
q = ld("rating_rating_ps_video.pt")
M["Rating-only CLS, per-scenario heads"] = (q["reg"], q["reg"], quantile_prob(q["reg"]), "rating only")
M["Rating-only CLS -> level probe on CLS"] = (LP["reg"], LP["lvl_reg"], LP["lvl_prob"].mean(0), "rerun of the row above + 128-d CLS level probe")
q = ld("rating_ens_video.pt")
M["Ensemble: rating CLS + per-scenario probe"] = (q["reg"], q["reg"], quantile_prob(q["reg"]), "rating only")
for arm, lab in (("level", "Level-only CLS (earlier)"), ("hard", "Level + hard commentary CLS (earlier)")):
    q = ld(f"results_{arm}.pt"); assert list(q["names"]) == list(names)
    M[lab] = (q["reg"], q["reg"], q["prob"].mean(0), "level only")
for tag, lab in (("joint", "Joint, no commentary"), ("joint_comm", "Joint + commentary 0.5 (final)"),
                 ("joint_lam1", "Joint + commentary 1"), ("joint_lam2", "Joint + commentary 2"),
                 ("joint_lam4", "Joint + commentary 4"), ("joint_comm_rank", "Joint + commentary 0.5 + ranking loss"),
                 ("joint_comm_vjepa", "Joint + commentary 0.5, V-JEPA 2")):
    q = ld(f"rating_{tag}.pt"); assert list(q["names"]) == list(names)
    M[lab] = (q["reg"], q["lvl_reg"], q["lvl_prob"].mean(0), "")
FINAL = "Joint + commentary 0.5 (final)"

P = {k: (pooled(a, rated), pooled(b, np.ones(n, bool)), pr, note) for k, (a, b, pr, note) in M.items()}
sp = lambda a, b: spearmanr(a, b)[0]
acc = lambda pr, i: (pr[i].argmax(1) + 1 == y[i]).mean()
bacc = lambda pr, i: np.mean([(pr[i][y[i] == c].argmax(1) + 1 == c).mean() for c in range(1, 5) if (y[i] == c).any()])
U = np.unique(p); rng = np.random.default_rng(0); draws = [rng.choice(U, len(U)) for _ in range(1000)]
rows_h = {u: np.flatnonzero(p[hi] == u) for u in U}; rows_a = {u: np.flatnonzero(p == u) for u in U}
Bh = [np.concatenate([rows_h[u] for u in d]) for d in draws]; Ba = [np.concatenate([rows_a[u] for u in d]) for d in draws]
alla = np.arange(n)

so = np.zeros((reps, n, 4))
for r in range(reps):
    for f in range(5):
        te, tr = fold_of[r] == f, fold_of[r] != f
        for k in range(6):
            so[r, te & (s == k), np.bincount(y[tr & (s == k)], minlength=5)[1:].argmax()] = 1
so = so.mean(0)

print(f"{n} takes ({rated.sum()} rated, {len(hi)} take-expert rows), {len(U)} participants")
print(f"{'model':44s} {'rat raw':>8s} {'rat z':>7s} {'acc':>7s} {'bal':>7s} {'rho':>7s}  note")
print(f"{'Human: one expert vs the others':44s} {sp(one, oth):8.3f} {sp(one_z, oth_z):7.3f} {'-':>7s} {'-':>7s} "
      f"{sp(one, lev_c[hi]):7.3f}  rho = one expert's rating vs level, rated rows")
print(f"{'Scenario only':44s} {'-':>8s} {'-':>7s} {acc(so, alla):7.3f} {bacc(so, alla):7.3f} {'-':>7s}")
for k, (rt, lv, pr, note) in P.items():
    print(f"{k:44s} {sp(rt[hi], oth):8.3f} {sp(rt[hi], oth_z):7.3f} {acc(pr, alla):7.3f} {bacc(pr, alla):7.3f} "
          f"{sp(lv, lev_c):7.3f}  {note}")
print(f"\npaired vs {FINAL} (95% participant bootstrap)")
F_ = P[FINAL]
ci = lambda d: f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"
for k, (rt, lv, pr, _) in P.items():
    if k == FINAL:
        continue
    dz = [sp(rt[hi][b], oth_z[b]) - sp(F_[0][hi][b], oth_z[b]) for b in Bh]
    da = [acc(pr, b) - acc(F_[2], b) for b in Ba]
    dr = [sp(lv[b], lev_c[b]) - sp(F_[1][b], lev_c[b]) for b in Ba]
    print(f"  {k:44s} z {sp(rt[hi], oth_z) - sp(F_[0][hi], oth_z):+.3f} {ci(dz)}   "
          f"acc {acc(pr, alla) - acc(F_[2], alla):+.3f} {ci(da)}   rho {sp(lv, lev_c) - sp(F_[1], lev_c):+.3f} {ci(dr)}")
print("\nper scenario: rating z / level accuracy / level Spearman")
for k in range(6):
    m, i = hs == k, np.flatnonzero(s == k)
    print(f"  {SCEN[k]:14s} rows={m.sum():4d} takes={len(i):4d}  human z {sp(one_z[m], oth_z[m]):.3f}")
    for mm, (rt, lv, pr, _) in P.items():
        print(f"      {mm:44s} {sp(rt[hi][m], oth_z[m]):6.3f} {acc(pr, i):6.3f} {sp(lv[i], lev_c[i]):6.3f}")

# Dance-only runs (8 vs 30 FPS): own folds over the 501 Dance takes, rating z and level within Dance
DANCE = []
print("\nDance only (own 4 x 5 folds over Dance takes): rating raw / z, level accuracy / balanced / Spearman")
for tag, lab in (("dance8", "Rating-only CLS, Dance, 8 FPS"), ("dance30", "Rating-only CLS, Dance, 30 FPS")):
    q = ld(f"rating_rating_{tag}.pt")
    ix = np.array([names.index(t) for t in q["names"]]); sub = np.zeros(n, bool); sub[ix] = True
    fold_of_save = fold_of
    fold_of = np.full((reps, n), -1); fold_of[:, ix] = q["fold_of"]
    for key, lab2 in (("reg", lab), ("probe_reg", lab.replace("Rating-only CLS", "Frozen probe"))):
        full = np.zeros((reps, n)); full[:, ix] = q[key]
        rt = pooled(full, rated & sub); lv = pooled(full, sub)
        pr = quantile_prob(np.where(sub, full, 0))
        m = (hs == SCEN.index("Dance")); i = ix
        print(f"  {lab2:44s} {sp(rt[hi][m], oth[m]):6.3f} {sp(rt[hi][m], oth_z[m]):6.3f} {acc(pr, i):6.3f} "
              f"{bacc(pr, i):6.3f} {sp(lv[i], lev_c[i]):6.3f}")
        DANCE.append([lab2, sp(rt[hi][m], oth[m]), sp(rt[hi][m], oth_z[m]), acc(pr, i), bacc(pr, i), sp(lv[i], lev_c[i])])
    fold_of = fold_of_save

# ---- CSV copies for the final report (tracked in git; .log / .pt are not) ----
import csv  # noqa: E402
OUT = Path("final_report") / "tables"
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "cv_all_models.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "rating_raw", "rating_z", "level_acc", "level_bal_acc", "level_spearman", "note"])
    w.writerow(["Human: one expert vs the others", f"{sp(one, oth):.3f}", f"{sp(one_z, oth_z):.3f}", "", "",
                f"{sp(one, lev_c[hi]):.3f}", "level_spearman = one expert's rating vs level on rated rows; z CI [0.737, 0.798]"])
    w.writerow(["Scenario only", "", "", f"{acc(so, alla):.3f}", f"{bacc(so, alla):.3f}", "", "most frequent training level per scenario"])
    for k, (rt, lv, pr, note) in P.items():
        w.writerow([k, f"{sp(rt[hi], oth):.3f}", f"{sp(rt[hi], oth_z):.3f}", f"{acc(pr, alla):.3f}", f"{bacc(pr, alla):.3f}",
                    f"{sp(lv, lev_c):.3f}", note])
with open(OUT / "cv_paired_vs_final.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "d_rating_z", "lo", "hi", "d_level_acc", "lo", "hi", "d_level_spearman", "lo", "hi"])
    for k, (rt, lv, pr, _) in P.items():
        if k == FINAL:
            continue
        dz = [sp(rt[hi][b], oth_z[b]) - sp(F_[0][hi][b], oth_z[b]) for b in Bh]
        da = [acc(pr, b) - acc(F_[2], b) for b in Ba]
        dr = [sp(lv[b], lev_c[b]) - sp(F_[1][b], lev_c[b]) for b in Ba]
        vals = [sp(rt[hi], oth_z) - sp(F_[0][hi], oth_z), *np.percentile(dz, [2.5, 97.5]),
                acc(pr, alla) - acc(F_[2], alla), *np.percentile(da, [2.5, 97.5]),
                sp(lv, lev_c) - sp(F_[1], lev_c), *np.percentile(dr, [2.5, 97.5])]
        w.writerow([k] + [f"{v:+.3f}" for v in vals])
with open(OUT / "cv_per_scenario.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["scenario", "rating_rows", "takes", "model", "rating_z", "level_acc", "level_spearman"])
    for k in range(6):
        m, i = hs == k, np.flatnonzero(s == k)
        w.writerow([SCEN[k], int(m.sum()), len(i), "Human: one expert vs the others", f"{sp(one_z[m], oth_z[m]):.3f}", "", ""])
        w.writerow([SCEN[k], int(m.sum()), len(i), "Scenario only", "", f"{acc(so, i):.3f}", ""])
        for mm, (rt, lv, pr, _) in P.items():
            w.writerow([SCEN[k], int(m.sum()), len(i), mm, f"{sp(rt[hi][m], oth_z[m]):.3f}", f"{acc(pr, i):.3f}",
                        f"{sp(lv[i], lev_c[i]):.3f}"])
with open(OUT / "cv_dance_only_fps.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "rating_raw", "rating_z", "level_acc", "level_bal_acc", "level_spearman", "note"])
    for r_ in DANCE:
        w.writerow([r_[0]] + [f"{v:.3f}" for v in r_[1:]] + ["own 4x5 folds over the 501 Dance takes; level read from the rating score"])
print(f"\nCSV tables written to {OUT}")
