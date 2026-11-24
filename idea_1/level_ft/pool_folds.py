"""Pool the 5 CV folds (+ official split) of level_ft/finetune.py and compare with the frozen references.

usage (from idea_1):  python level_ft/pool_folds.py [RUN]
References recomputed per fold on the same takes and the same 16 probe clips:
  lp      = frozen linear head the fine-tuning starts from (ridge on standardised 1536-d, alpha by grouped CV)
  pca     = frozen PCA-64-per-view ridge / logistic probe (the standing protocol)
Within-scenario Spearman of predictions centred within (fold, scenario); participant bootstrap of the paired diffs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegressionCV, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent / "ssl_adapt"))
from common import BANK_DIR  # noqa: E402
from finetune import SCEN, centred, frozen_reference, official_table, split_fold  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "lpft_r8_top6"
D = Path("models") / "level_ft" / RUN
ix = torch.load(BANK_DIR / "probe_index.pt", weights_only=False)
takes = ix["takes"]
y, s, p, split = official_table(takes)
fr = torch.load(BANK_DIR / "probe_feats.pt", weights_only=False)["frozen"]
X = np.hstack([fr["ego"], fr["exo"]])
with open("models/SkillFormer/annotation_test.jsonl", encoding="utf-8") as f:
    sf_test = {json.loads(line)["video_paths"][0].split("/")[1] for line in f}


def references(fold):
    fit_i, iv_i, te_i = split_fold(fold, y, p, split)
    mu = np.stack([X[fit_i][s[fit_i] == k].mean(0) for k in range(6)])
    sd = (X[fit_i] - mu[s[fit_i]]).std(0) + 1e-6
    Z = (X - mu[s]) / sd
    yc = y - np.array([y[fit_i][s[fit_i] == k].mean() for k in range(6)])[s]
    cv = list(GroupKFold(5).split(Z[fit_i], y[fit_i], p[fit_i]))
    lp = GridSearchCV(Ridge(fit_intercept=False), {"alpha": np.logspace(1, 6, 11)}, cv=cv,
                      scoring="neg_mean_squared_error").fit(Z[fit_i], yc[fit_i]).predict(Z[te_i])
    oh = (s[:, None] == np.arange(6)[None]).astype(float)
    lpc = LogisticRegressionCV(Cs=np.logspace(-4, -1, 7), cv=cv, max_iter=3000, n_jobs=7).fit(
        np.hstack([Z, oh])[fit_i], y[fit_i]).predict_proba(np.hstack([Z, oh])[te_i])
    fitm, tem = np.zeros(len(y), bool), np.zeros(len(y), bool)
    fitm[np.concatenate([fit_i, iv_i])] = True; tem[te_i] = True
    pr, pl = frozen_reference(fr["ego"], fr["exo"], y, s, fitm, tem)
    return te_i, {"lp": (lp, lpc), "pca": (pr, pl)}


def rho(pred, idx, centre_groups):
    return spearmanr(centred(pred, centre_groups), centred(y[idx].astype(float), centre_groups))[0]


def report(name, rows):
    """rows: list of (te_i, {method: (reg, logit)}) per fold -> pooled metrics + paired bootstrap ft - ref"""
    idx = np.concatenate([r[0] for r in rows])
    fold_id = np.concatenate([np.full(len(r[0]), k) for k, r in enumerate(rows)])
    grp = fold_id * 10 + s[idx]
    M = {m: (np.concatenate([centred(r[1][m][0], s[r[0]]) for r in rows]),
             np.concatenate([r[1][m][1] for r in rows])) for m in rows[0][1]}
    print(f"\n== {name}: {len(idx)} takes, {len(set(p[idx]))} participants")
    for m, (reg, lg) in M.items():
        print(f"  {m:4s} rho {rho(reg, idx, grp):.3f}   acc {(lg.argmax(1) + 1 == y[idx]).mean():.3f}")
    for k in range(6):
        mk = s[idx] == k
        print(f"  {SCEN[k]:14s} n={mk.sum():4d} " + "  ".join(f"{m} {rho(M[m][0][mk], idx[mk], grp[mk]):.3f}" for m in M))
    rng, U = np.random.default_rng(0), np.unique(p[idx])
    pos = {u: np.flatnonzero(p[idx] == u) for u in U}
    for ref in [m for m in M if m != "ft"]:
        d_r, d_a = [], []
        for _ in range(2000):
            b = np.concatenate([pos[u] for u in rng.choice(U, len(U))])
            d_r.append(rho(M["ft"][0][b], idx[b], grp[b]) - rho(M[ref][0][b], idx[b], grp[b]))
            yb = y[idx][b] - 1
            d_a.append((M["ft"][1][b].argmax(1) == yb).mean() - (M[ref][1][b].argmax(1) == yb).mean())
        point_r = rho(M["ft"][0], idx, grp) - rho(M[ref][0], idx, grp)
        point_a = (M["ft"][1].argmax(1) + 1 == y[idx]).mean() - (M[ref][1].argmax(1) + 1 == y[idx]).mean()
        print(f"  ft - {ref}: rho {point_r:+.3f} [{np.percentile(d_r, 2.5):+.3f}, {np.percentile(d_r, 97.5):+.3f}]"
              f"   acc {point_a:+.3f} [{np.percentile(d_a, 2.5):+.3f}, {np.percentile(d_a, 97.5):+.3f}]")


def load(fold):
    P = torch.load(D / f"fold{fold}_preds.pt", weights_only=False)
    te_i, refs = references(fold)
    assert P["takes"] == [takes[t] for t in te_i]
    refs["ft"] = (P["reg"], P["logit"])
    print(f"fold {fold}: best epoch {P['final']['best_epoch']}, inner-val rho {P['final']['best_innerval_rho']:.3f}")
    return te_i, {m: refs[m] for m in ("ft", "lp", "pca")}


cv = [load(str(k)) for k in range(5) if (D / f"fold{k}_preds.pt").exists()]
if cv:
    report(f"CV ({len(cv)} folds)", cv)
if (D / "foldofficial_preds.pt").exists():
    te_i, refs = load("official")
    report("official val", [(te_i, refs)])
    m = np.isin(np.array(takes)[te_i], list(sf_test))
    report("SkillFormer test half", [(te_i[m], {k: (v[0][m], v[1][m]) for k, v in refs.items()})])
