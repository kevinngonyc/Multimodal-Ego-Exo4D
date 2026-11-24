"""Report figures from the tracked CSVs only (no data, GPU or .pt needed). Run from idea_1:
    python final_report/make_figures.py      -> final_report/figures/*.png
Palette: categorical slots 1-3 of the dataviz reference palette (blue, orange, aqua), neutral gray for references
(humans, priors); thin marks, recessive grid, one y-scale per panel, direct labels on the key points.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T = Path("final_report") / "tables"; F = Path("final_report") / "figures"; F.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#8d8c86", "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 200})
rd = lambda f: list(csv.DictReader(open(T / f, encoding="utf-8")))
num = lambda x: float(x) if x not in ("", None) else None


def save(fig, name):
    fig.tight_layout(); fig.savefig(F / name, facecolor=SURF); plt.close(fig); print("wrote", F / name)


# 1. rating vs level trade-off: every CV model on both targets
rows = {r["model"]: r for r in rd("cv_all_models.csv")}
GROUP = {"Frozen probe": GRAY, "Rating-only": ORANGE, "Ensemble": ORANGE, "Level": AQUA, "Joint": BLUE}
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for k, r in rows.items():
    x, y = num(r["rating_z"]), num(r["level_acc"])
    if x is None or y is None or k.startswith("Rating-only CLS -> level probe"):
        continue
    c = next(v for g, v in GROUP.items() if k.startswith(g))
    final = "final" in k
    ax.scatter(x, y, s=70 if final else 30, color=c, edgecolor=SURF, linewidth=1.5, zorder=3)
hz = num(rows["Human: one expert vs the others"]["rating_z"]); pr = num(rows["Scenario only"]["level_acc"])
ax.axvline(hz, color=GRAY, ls="--", lw=1); ax.text(hz - 0.005, 0.475, "one human expert", ha="right", color=INK2, fontsize=8)
ax.axhline(pr, color=GRAY, ls="--", lw=1); ax.text(0.445, pr + 0.003, "scenario prior", color=INK2, fontsize=8)
LAB = {"Joint + commentary 0.5 (final)": ("final model (joint + commentary)", (-0.005, 0.006), "right"),
       "Frozen probe, shared ridge": ("frozen probe", (0.0, -0.009), "center"),
       "Rating-only CLS, per-scenario heads": ("rating-only CLS", (0.007, -0.001), "left"),
       "Level + hard commentary CLS (earlier)": ("level-only CLS", (0.008, 0.0), "left"),
       "Joint + commentary 0.5, V-JEPA 2": ("joint, V-JEPA 2", (0.004, -0.009), "left"),
       "Frozen probe, per-scenario ridge": ("per-scenario ridge", (-0.006, -0.008), "right")}
for k, (t, (dx, dy), ha) in LAB.items():
    ax.annotate(t, (num(rows[k]["rating_z"]), num(rows[k]["level_acc"])),
                (num(rows[k]["rating_z"]) + dx, num(rows[k]["level_acc"]) + dy), ha=ha, fontsize=8, color=INK)
ax.set_xlim(0.43, 0.80); ax.set_ylim(0.33, 0.49)
ax.set_xlabel("Expert rating: Spearman with held-out experts (per-expert z)"); ax.set_ylabel("Official level: 4-class accuracy")
ax.set_title("Only the joint models do well on both targets (4×5 CV, 2,111 takes)", loc="left")
for g, c in (("frozen probes", GRAY), ("rating-only models", ORANGE), ("level-only models", AQUA), ("joint models", BLUE)):
    ax.scatter([], [], color=c, s=30, label=g)
ax.legend(loc="lower right", fontsize=8)
save(fig, "fig1_rating_vs_level.png")

# 2. per-scenario rating z: human vs frozen probe vs final
ps = rd("cv_per_scenario.csv")
get = lambda sc, m: next(num(r["rating_z"]) for r in ps if r["scenario"] == sc and r["model"] == m)
scen = sorted({r["scenario"] for r in ps}, key=lambda sc: -get(sc, "Human: one expert vs the others"))
fig, ax = plt.subplots(figsize=(6.4, 3.6))
w = 0.26
for i, (m, c, lab) in enumerate((("Human: one expert vs the others", GRAY, "one human expert"),
                                 ("Frozen probe, shared ridge", ORANGE, "frozen probe"),
                                 ("Joint + commentary 0.5 (final)", BLUE, "final model"))):
    xs = [j + (i - 1) * w for j in range(len(scen))]
    v = [get(sc, m) for sc in scen]
    ax.bar(xs, v, width=w - 0.03, color=c, label=lab, zorder=3)
    if c == BLUE:
        for x, y in zip(xs, v):
            ax.text(x, max(y, 0) + 0.015, f"{y:.2f}", ha="center", fontsize=7, color=INK)
ax.axhline(0, color=INK2, lw=0.8)
rows_n = {r["scenario"]: r["rating_rows"] for r in ps}
ax.set_xticks(range(len(scen)), [f"{sc}\n({rows_n[sc]} rows)" for sc in scen])
ax.set_ylabel("Spearman with held-out experts (z)"); ax.set_ylim(-0.12, 0.95)
ax.set_title("Expert rating per scenario (4×5 CV)", loc="left"); ax.legend(loc="upper right", ncol=3, fontsize=8)
ax.grid(axis="x", visible=False)
save(fig, "fig2_rating_per_scenario.png")

# 3. official split: level accuracy, ours with participant-bootstrap CIs, published single numbers
off = {(r["eval_set"], r["model"], r["metric"]): r for r in rd("official_split.csv")}
pub = {r["method"]: r for r in rd("published_baselines.csv")}
items = [("Scenario prior (ours)", off[("official val", "Scenario prior", "level_acc")], GRAY),
         ("Frozen PCA probe (ours)", off[("official val", "Frozen PCA level probe", "level_acc")], ORANGE),
         ("Final model (ours, 2 seeds)", off[("official val", "Final model [LEVEL]", "level_acc")], BLUE)]
s5 = [r for r in rd("official_split_s5.csv") if r["eval_set"] == "official val" and r["model"] == "Final model [LEVEL]"
      and r["metric"] == "level_acc"]
if s5:
    items.append(("Final model (ours, 5 seeds)", s5[0], BLUE))
fig, ax = plt.subplots(figsize=(7.6, 3.3))
labels = []
for i, (lab, r, c) in enumerate(items):
    v, lo, hi = 100 * num(r["value"]), 100 * num(r["ci_lo"]), 100 * num(r["ci_hi"])
    ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter(v, i, color=c, s=40, zorder=3, edgecolor=SURF, linewidth=1.5)
    ax.text(hi + 0.8, i, f"{v:.1f}  [{lo:.1f}, {hi:.1f}]", va="center", fontsize=8, color=INK)
    labels.append(lab)
for j, (m, lab) in enumerate((("SkillFormer", "SkillFormer, Ego+Exos (paper)"), ("ProfVLM (AGP)", "ProfVLM, Ego+Exos (paper)"))):
    i = len(items) + j; v = num(pub[m]["acc_ego_exos"])
    ax.scatter(v, i, facecolor=SURF, edgecolor=INK2, s=40, linewidth=1.5, zorder=3)
    ax.text(v + 0.8, i, f"{v:.1f}  (paper; no interval)", va="center", fontsize=8, color=INK)
    labels.append(lab)
ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis(); ax.set_xlim(15, 88)
ax.set_xlabel("Level accuracy on official val (%), 95% participant-bootstrap interval")
ax.set_title("Official split: level accuracy vs published methods", loc="left")
ax.grid(axis="y", visible=False)
save(fig, "fig3_official_level.png")

# 4. commentary weight sweep: rating / level unchanged, commentary outputs improve
cv = {r["model"]: r for r in rd("cv_all_models.csv")}; cm = {r["model"]: r for r in rd("cv_commentary.csv")}
W = [(0, "Joint, no commentary", None), (0.5, "Joint + commentary 0.5 (final)", "Joint + commentary 0.5 (FINAL)"),
     (1, "Joint + commentary 1", "Joint + commentary 1"), (2, "Joint + commentary 2", "Joint + commentary 2"),
     (4, "Joint + commentary 4", "Joint + commentary 4")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.9))
xs = [w_ for w_, _, _ in W]
for key, c, lab in (("rating_z", BLUE, "rating (z)"), ("level_acc", ORANGE, "level accuracy")):
    v = [num(cv[m][key]) for _, m, _ in W]
    a1.plot(xs, v, color=c, lw=2, marker="o", ms=4, label=lab); a1.text(4.1, v[-1], lab, color=INK, fontsize=8, va="center")
a1.set_xlim(-0.2, 5.6); a1.set_ylim(0.40, 0.70); a1.set_xlabel("commentary loss weight"); a1.set_ylabel("score")
a1.set_title("Rating and level: flat", loc="left")
xs2 = [w_ for w_, _, k in W if k]
for key, c, lab in (("pres_auc", BLUE, "presence AUC"), ("txt_cos", AQUA, "text cosine"), ("pol_auc", ORANGE, "praise-vs-tip AUC")):
    v = [num(cm[k][key]) for _, _, k in W if k]
    a2.plot(xs2, v, color=c, lw=2, marker="o", ms=4); a2.text(4.1, v[-1], lab, color=INK, fontsize=8, va="center")
a2.set_xlim(0.3, 6.4); a2.set_ylim(0.55, 0.75); a2.set_xlabel("commentary loss weight"); a2.set_ylabel("held-out score")
a2.set_title("Commentary outputs: improve", loc="left")
a1.axvline(0.5, color=GRAY, ls=":", lw=1); a2.axvline(0.5, color=GRAY, ls=":", lw=1)
a1.text(0.55, 0.415, "final", color=INK2, fontsize=7); a2.text(0.55, 0.555, "final", color=INK2, fontsize=7)
save(fig, "fig4_commentary_weight.png")

# 6. by view on official val: ours (one exo camera, CI) vs the papers (all exo cameras, single runs)
vv = rd("official_views_vs_papers.csv")
VIEWS = ["Ego", "Exo", "Ego+Exo"]
SER = [("Ours (final model)", BLUE, "ours (exo = best camera; 2 seeds)"), ("SkillFormer", ORANGE, "SkillFormer (paper; exo = all cameras)"),
       ("ProfVLM (AGP)", AQUA, "ProfVLM AGP (paper; exo = all cameras)")]
fig, ax = plt.subplots(figsize=(6.6, 3.2))
for i, (m, c, lab) in enumerate(SER):
    for j, v in enumerate(VIEWS):
        r = next(r for r in vv if r["model"] == m and r["view"] == v and "5 seeds" not in r["note"])
        x = j + (i - 1) * 0.22; y = 100 * float(r["level_acc"])
        if r["acc_ci_lo"]:
            ax.plot([x, x], [100 * float(r["acc_ci_lo"]), 100 * float(r["acc_ci_hi"])], color=c, lw=2, solid_capstyle="round")
        ax.scatter(x, y, color=c, s=36, zorder=3, edgecolor=SURF, linewidth=1.2, label=lab if j == 0 else None)
        ax.text(x + 0.05, y, f"{y:.1f}", fontsize=7, color=INK, va="center")
ax.axhline(31.1, color=GRAY, ls="--", lw=1); ax.text(2.45, 32.0, "majority / scenario prior", fontsize=7, color=INK2, ha="right")
ax.set_xticks(range(3), ["ego only", "exo only", "ego + exo"]); ax.set_xlim(-0.5, 2.6); ax.set_ylim(25, 65)
ax.set_ylabel("level accuracy on official val (%)")
ax.set_title("Level accuracy by input view vs published methods", loc="left")
ax.legend(loc="upper left", fontsize=7.5); ax.grid(axis="x", visible=False)
save(fig, "fig6_views_vs_papers.png")
