"""Qualitative commentary outputs of the final model on official val (run from idea_1, CPU, after
`python cls_lang/joint.py --lam 0.5 --official --seeds 5 --tag joint_comm_official_s5`).

Per 1 s clip the model outputs a presence logit (would an expert comment here?), a polarity per aspect (praise > 0 >
tip, tanh; trained on polarity centred per expert, so "praise" means warmer than that expert's usual tone) and a sentence embedding (MiniLM space) of what the expert would say. "Which aspect" is read from the
predicted embedding with a per-scenario logistic regression trained on the TRAIN-split comments (MiniLM embedding ->
the comment's aspect tag from build_targets.py); praise vs tip for that aspect comes from the polarity head. No comment
text is used or shown anywhere.

Example takes: per scenario, the val take with the median per-take presence AUC (typical, not the best). Highlighted
moments: the 4 highest presence peaks at least max(15 s, 1/6 of the take) apart. Reference: the real comments within +-3 s (their aspect tags
and praise / tip from build_targets.py, all experts).
Aggregate check on all val clips that carry a real comment: is the comment's aspect among the model's top-1 / top-3
aspects (vs always guessing the most frequent training aspect of that scenario)? And praise vs tip agreement.
-> final_report/tables/commentary_examples.csv, commentary_aspect_check.csv, final_report/figures/fig5_commentary_examples.png
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from train import A, LEVELS  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "joint_comm_official_s5"
q = torch.load(Path("cls_lang") / f"rating_{TAG}.pt", weights_only=False)
CO = q["clip_out"]
assert len(next(iter(CO.values()))) == 3, "run saved no text embeddings; rerun joint.py --official with lam > 0"
S = torch.load(Path("embeddings") / "cls_lang_sentences.pt", weights_only=False)
NAMES = S["aspect_names"]
meta = {}
for fn, sp in (("proficiency_demonstrator_train.json", "train"), ("proficiency_demonstrator_val.json", "val")):
    for m in json.load(open(A / fn, encoding="utf-8"))["annotations"]:
        meta[m["video_paths"]["ego"].split("/")[1]] = (m["scenario_name"], m["proficiency_score"], sp)
split = np.array([meta.get(t, ("?", "?", "?"))[2] for t in S["take"]])
asp_tag = np.array([int(np.argmax(S["aspect"][i, :len(NAMES[S["scenario"][i]])])) for i in range(len(S["take"]))])
unit = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)

# per-scenario aspect classifier on real train-split comment embeddings
CLF, PRIOR = {}, {}
rng = np.random.default_rng(0)
print("aspect classifier (train-split comments -> aspect tag), accuracy on val-split comments:")
for sc in NAMES:
    tr = np.flatnonzero((S["scenario"] == sc) & (split == "train")); va = np.flatnonzero((S["scenario"] == sc) & (split == "val"))
    tr = rng.choice(tr, min(len(tr), 20000), replace=False)
    clf = LogisticRegression(C=4.0, max_iter=2000).fit(unit(S["emb"][tr].astype(np.float32)), asp_tag[tr])
    CLF[sc] = clf; PRIOR[sc] = Counter(asp_tag[tr]).most_common(1)[0][0]
    acc = (clf.predict(unit(S["emb"][va].astype(np.float32))) == asp_tag[va]).mean()
    print(f"  {sc:14s} {len(NAMES[sc])} aspects, val comments {len(va)}: acc {acc:.3f} (most frequent '{NAMES[sc][PRIOR[sc]]}' "
          f"{(asp_tag[va] == PRIOR[sc]).mean():.3f})")

by_take = defaultdict(list)
for i, t in enumerate(S["take"]):
    if t in CO:
        by_take[t].append(i)


def take_view(t):
    pres_l, pol, txt = CO[t]
    n = len(pres_l); sc = meta[t][0]
    pres = 1 / (1 + np.exp(-pres_l))
    lab = np.zeros(n, bool); gt = defaultdict(list)  # clip -> real comments covering it (aspect, praise?)
    for i in by_take[t]:
        last = min(int(np.floor(S["time"][i])), n - 1)
        for c in range(max(last - 2, 0), last + 1):
            lab[c] = True; gt[c].append((asp_tag[i], S["p_good"][i] > 0.5))
    P = CLF[sc].predict_proba(unit(txt.astype(np.float32)))  # (n, classes) over aspect ids in CLF.classes_
    return sc, pres, pol, P, lab, gt


# aggregate aspect / polarity check over all val clips with a real comment
agg = defaultdict(lambda: Counter())
auc_take = {}
for t in CO:
    if t not in by_take:
        continue
    sc, pres, pol, P, lab, gt = take_view(t)
    if 0 < lab.mean() < 1:
        auc_take[t] = roc_auc_score(lab, pres)
    cls_ = CLF[sc].classes_; other = len(NAMES[sc]) - 1
    for c, lst in gt.items():
        order = cls_[np.argsort(-P[c])]
        for a_id, good in lst:
            if a_id == other:
                continue
            g = agg[sc]; g["n"] += 1
            g["top1"] += order[0] == a_id; g["top3"] += a_id in order[:3]; g["prior"] += PRIOR[sc] == a_id
            g["pol_ok"] += (pol[c, a_id] > 0) == good; g["praise"] += good; g["pred_praise"] += pol[c, a_id] > 0
rows_chk = []
print("\nmodel's aspect readout at clips with a real comment (comment aspect != other):")
for sc in NAMES:
    g = agg[sc]
    if g["n"]:
        r_ = [sc, g["n"], g["top1"] / g["n"], g["top3"] / g["n"], g["prior"] / g["n"], g["pol_ok"] / g["n"],
              g["praise"] / g["n"], g["pred_praise"] / g["n"]]
        rows_chk.append(r_)
        print(f"  {sc:14s} n={g['n']:6d}  top-1 {r_[2]:.3f}  top-3 {r_[3]:.3f}  (most-frequent-aspect {r_[4]:.3f})  "
              f"praise/tip agreement {r_[5]:.3f} (real praise share {r_[6]:.3f}, predicted {r_[7]:.3f})")

# examples: per scenario the median-AUC val take
ex = []
for sc in NAMES:
    c_ = sorted([t for t in auc_take if meta[t][0] == sc], key=lambda t: auc_take[t])
    if c_:
        ex.append(c_[len(c_) // 2])
rows_ex = []
for t in ex:
    sc, pres, pol, P, lab, gt = take_view(t)
    cls_ = CLF[sc].classes_; other = len(NAMES[sc]) - 1
    peaks = []; sep = max(15, len(pres) // 6)
    for c in np.argsort(-pres):
        if all(abs(c - p) >= sep for p in peaks):
            peaks.append(int(c))
        if len(peaks) == 4:
            break
    for c in sorted(peaks):
        order = [a for a in cls_[np.argsort(-P[c])] if a != other][:2]
        pr = {a: P[c][list(cls_).index(a)] for a in order}
        near = [x for cc in range(max(c - 3, 0), min(c + 4, len(pres))) for x in gt.get(cc, [])]
        near_s = "; ".join(f"{NAMES[sc][a]} ({'praise' if g else 'tip'})" for a, g in sorted(set(near)))
        rows_ex.append([t, sc, meta[t][1], f"{auc_take[t]:.3f}", c, f"{pres[c]:.3f}",
                        NAMES[sc][order[0]], f"{pr[order[0]]:.2f}", "praise" if pol[c, order[0]] > 0 else "tip", f"{pol[c, order[0]]:+.2f}",
                        NAMES[sc][order[1]], f"{pr[order[1]]:.2f}", "praise" if pol[c, order[1]] > 0 else "tip", f"{pol[c, order[1]]:+.2f}",
                        len(near), near_s])

OUT = Path("final_report"); (OUT / "tables").mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
with open(OUT / "tables" / "commentary_examples.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["take", "scenario", "level", "take_presence_auc", "time_s", "presence_prob", "aspect_1", "aspect_1_prob",
                "aspect_1_call", "aspect_1_polarity", "aspect_2", "aspect_2_prob", "aspect_2_call", "aspect_2_polarity",
                "real_comments_within_3s", "real_comment_aspects_within_3s"])
    w.writerows(rows_ex)
with open(OUT / "tables" / "commentary_aspect_check.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["scenario", "commented_clip_comment_pairs", "aspect_top1", "aspect_top3", "most_frequent_aspect_baseline",
                "praise_tip_agreement", "real_praise_share", "predicted_praise_share"])
    w.writerows([[r_[0], r_[1]] + [f"{x:.3f}" for x in r_[2:]] for r_ in rows_chk])
print("\nexample takes:", ex)
print(f"per-take presence AUC on val: median {np.median(list(auc_take.values())):.3f} over {len(auc_take)} takes")

# figure: presence over time + real comment times + annotated peaks, one panel per example take
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#8d8c86", "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(len(ex), 1, figsize=(7.6, 1.75 * len(ex)))
for ax, t in zip(np.atleast_1d(axs), ex):
    sc, pres, pol, P, lab, gt = take_view(t)
    x = np.arange(len(pres))
    ax.plot(x, pres, color=BLUE, lw=1.2, zorder=3)
    for i in by_take[t]:
        tm = S["time"][i]
        ax.plot([tm, tm], [-0.13, -0.03], color=AQUA if S["p_good"][i] > 0.5 else ORANGE, lw=0.8, alpha=0.6, zorder=2)
    for k_, r_ in enumerate([r for r in rows_ex if r[0] == t]):
        c = r_[4]
        ax.scatter(c, pres[c], color=BLUE, s=14, zorder=4, edgecolor=SURF, linewidth=0.8)
        ha = "left" if c < 0.08 * len(pres) else ("right" if c > 0.92 * len(pres) else "center")
        ax.annotate(f"{r_[6]} ({r_[8]})", (c, pres[c]), (c, min(pres[c] + (0.12 if k_ % 2 == 0 else 0.3), 1.1)),
                    ha=ha, fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5))
    ax.set_ylim(-0.15, 1.2); ax.set_xlim(0, len(pres)); ax.set_yticks([0, 0.5, 1])
    ax.set_title(f"{sc} (level {meta[t][1]}, take presence AUC {auc_take[t]:.2f})", loc="left", fontsize=8.5,
                 fontweight="bold", color=INK)
np.atleast_1d(axs)[-1].set_xlabel("time in take (s)")
fig.text(0.01, 0.5, "predicted probability an expert comments", rotation=90, va="center", color=INK2)
fig.suptitle("Final model's per-second commentary outputs on official val (typical takes: median presence AUC)\n"
             "line: predicted presence; ticks below: real comments (aqua praise, orange tip); labels: top predicted aspect (praise / tip)",
             x=0.01, ha="left", fontsize=8.5, color=INK)
fig.tight_layout(rect=(0.02, 0, 1, 0.97))
fig.savefig(OUT / "figures" / "fig5_commentary_examples.png", dpi=200, facecolor=SURF)
print("wrote", OUT / "figures" / "fig5_commentary_examples.png")
