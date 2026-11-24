"""Units vs takes-per-unit learning curve for the frozen probe on official level (run from idea_1).

usage:  python label_units/learning_curve.py [--repeats 4] [--draws 5]

A label unit = one (participant, scenario) pair; level is constant within it (243 of 256 multi-take units).
Within every outer training fold (4 x 5 StratifiedGroupKFold by participant over the 2,111 official takes, as in the
main protocol) the training set is thinned two ways, the test fold untouched:
  units: keep ALL takes of a random fraction (25 / 50 / 75 / 100 %) of the units, sampled per scenario
  cap:   keep EVERY unit but at most k random takes of it (k = 1, 2, 4, 8, all)
Probe = the frozen probe (ego + exo K600 L12 token std, take mean; scenario-centred PCA-64 per view on the kept
training takes; scenario one-hot; standardised; ridge). The ridge penalty is chosen by 5-fold GroupKFold by
participant on the kept training takes (RidgeCV's default leave-one-out would leak across a unit's sibling takes,
and how much it leaks depends on the cap, which would bias the comparison).
Metric = within-scenario Spearman, predictions centred within (repeat, fold, scenario) and averaged over repeats,
computed per random draw; reported as mean and range over draws.
-> label_units/learning_curve.pt, figures/learning_curve_units.png
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cls_lang"))
from train import SCEN, load_takes  # noqa: E402

ALPHAS = np.logspace(-2, 6, 17)
FRACS = (0.25, 0.5, 0.75, 1.0)
CAPS = (1, 2, 4, 8, None)


def design(X, s, keep, held):
    rows = np.concatenate([keep, held])
    parts = [(s[rows, None] == np.arange(6)[None]).astype(float)]
    nk = len(keep)
    for v in (slice(0, 768), slice(768, 1536)):
        B = X[rows, v].astype(np.float64).copy()
        for k in range(6):
            m = s[rows] == k
            mk = m[:nk]
            if mk.any():
                B[m] -= B[:nk][mk].mean(0)
        try:
            pca = PCA(64, svd_solver="full").fit(B[:nk])
        except np.linalg.LinAlgError:  # rare LAPACK gesdd non-convergence on a subset; same PCA via the covariance
            pca = PCA(64, svd_solver="covariance_eigh").fit(B[:nk])
        parts.append(pca.transform(B))
    Z = np.hstack(parts)
    Z = StandardScaler().fit(Z[:nk]).transform(Z)
    return Z[:nk], Z[nk:]


def ridge_group_cv(Z, y, groups):
    err = np.zeros(len(ALPHAS))
    for tr, va in GroupKFold(5).split(Z, y, groups):
        for i, a in enumerate(ALPHAS):
            err[i] += ((Ridge(alpha=a).fit(Z[tr], y[tr]).predict(Z[va]) - y[va]) ** 2).sum()
    return Ridge(alpha=ALPHAS[err.argmin()]).fit(Z, y)


def centred(v, g):
    v = np.asarray(v, float).copy()
    for k in np.unique(g):
        v[g == k] -= v[g == k].mean()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--draws", type=int, default=5)
    a = ap.parse_args()
    names, Xraw, s, y, p = load_takes()
    n = len(names)
    X = np.stack([x.float().mean(0).numpy() for x in Xraw])
    unit = np.array([f"{q}|{k}" for q, k in zip(p, s)])
    conds = [("units", f) for f in FRACS] + [("cap", c) for c in CAPS]
    ndraw = {c: (1 if c in (("units", 1.0), ("cap", None)) else a.draws) for c in conds}
    pred = {c: np.zeros((ndraw[c], a.repeats, n)) for c in conds}
    ntake = {c: [] for c in conds}; nunit = {c: [] for c in conds}
    fold_of = np.zeros((a.repeats, n), int)
    t0 = time.time()
    for r in range(a.repeats):
        for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(n), y, p)):
            fold_of[r, te] = f
            units = {u: tr[unit[tr] == u] for u in np.unique(unit[tr])}
            for c in conds:
                for d in range(ndraw[c]):
                    rng = np.random.default_rng(1000 * r + 100 * f + 10 * d + conds.index(c))
                    if c[0] == "units":
                        keep_u = []
                        for k in range(6):
                            us = sorted(u for u in units if s[units[u][0]] == k)
                            m = max(2, int(np.ceil(c[1] * len(us))))
                            keep_u += list(rng.choice(us, min(m, len(us)), replace=False))
                        keep = np.concatenate([units[u] for u in keep_u])
                    else:
                        keep = np.concatenate([idx if c[1] is None or len(idx) <= c[1] else rng.choice(idx, c[1], replace=False)
                                               for idx in units.values()])
                        keep_u = list(units)
                    Zk, Zt = design(X, s, keep, te)
                    pred[c][d, r, te] = ridge_group_cv(Zk, y[keep], p[keep]).predict(Zt)
                    ntake[c].append(len(keep)); nunit[c].append(len(keep_u))
            print(f"repeat {r} fold {f} done, {(time.time() - t0) / 60:.1f} min", flush=True)
    lev = centred(y, s)
    res = {}
    for c in conds:
        rhos = []
        for d in range(ndraw[c]):
            P = pred[c][d].copy()
            for r in range(a.repeats):
                for f in range(5):
                    for k in range(6):
                        g = (fold_of[r] == f) & (s == k)
                        P[r, g] -= P[r, g].mean()
            rhos.append(spearmanr(P.mean(0), lev)[0])
        res[c] = dict(rho=np.array(rhos), takes=float(np.mean(ntake[c])), units=float(np.mean(nunit[c])))
        print(f"{c[0]:5s} {str(c[1]):5s} train takes {res[c]['takes']:7.0f}  units {res[c]['units']:5.0f}  "
              f"rho {np.mean(rhos):.3f} (range {np.min(rhos):.3f}-{np.max(rhos):.3f})", flush=True)
    torch.save(dict(res=res, args=vars(a)), Path("label_units") / "learning_curve.pt")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    for kind, col, lab in (("units", blue, "fewer units (all takes of each kept unit)"),
                           ("cap", orange, "fewer takes per unit (every unit kept)")):
        cs = [c for c in conds if c[0] == kind]
        xs = [res[c]["takes"] for c in cs]; ms = [res[c]["rho"].mean() for c in cs]
        lo = [res[c]["rho"].mean() - res[c]["rho"].min() for c in cs]; hi = [res[c]["rho"].max() - res[c]["rho"].mean() for c in cs]
        ax.errorbar(xs, ms, yerr=[lo, hi], color=col, lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2, capsize=0, label=lab)
        for c, x_, m_ in zip(cs, xs, ms):
            t = f"{int(c[1] * 100)}% units" if kind == "units" else (f"cap {c[1]}" if c[1] else "")
            if t:
                ax.annotate(t, (x_, m_), textcoords="offset points", xytext=(0, 10 if kind == "units" else -16),
                            ha="center", fontsize=8, color=muted)
    ax.set_xscale("log")
    ax.set_xlabel("training takes per fold (log scale)", color=ink)
    ax.set_ylabel("within-scenario Spearman with level", color=ink)
    ax.set_title("Frozen probe: dropping units vs dropping takes per unit", color=ink, loc="left", fontsize=11)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.spines["left"].set_color(muted); ax.spines["bottom"].set_color(muted)
    ax.tick_params(colors=muted); ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(Path("figures") / "learning_curve_units.png", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
