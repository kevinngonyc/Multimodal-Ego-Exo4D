"""Builds the tables in final_report/tables/ that the two result scripts do not write (run from idea_1, CPU, needs
the local data/annotations, embeddings/ caches and cls_lang/*.pt run outputs, none of which are in git).

usage (from idea_1):
    python cls_lang/all_metrics.py      > cls_lang/all_metrics_report.log    # -> tables/cv_*.csv
    python cls_lang/official_report.py  > cls_lang/official_report.log       # -> tables/official_split*.csv
    python final_report/build_tables.py                                      # -> the rest + logs/*.txt
Writes:
  tables/dataset_stats.csv          takes / participants / levels / ratings / experts / comments per scenario and split
  tables/cv_commentary.csv          held-out commentary metrics per joint variant (mean and sd over the 20 folds)
  tables/cv_training.csv            checkpoint epochs (rating / level) per joint variant, run time, model size
  tables/scenario_classifier.csv    takes whose scenario the video classifier got wrong (CV and official split)
  tables/published_baselines.csv    SkillFormer / ProfVLM / Ego-Exo4D baselines, copied from the papers
  tables/official_views_vs_papers.csv  ours by view (ego / exo / both; 2 and 5 seeds) next to the papers' view columns
  logs/*.txt                        copies of the report logs (.log is gitignored)
"""
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "cls_lang")
from train import A, LEVELS, SCEN, load_ratings  # noqa: E402

OUT = Path("final_report") / "tables"; OUT.mkdir(parents=True, exist_ok=True)
LOGS = Path("final_report") / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
C = Path("cls_lang")
ld = lambda f: torch.load(C / f, weights_only=False)


def write(name, header, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    print("wrote", OUT / name, len(rows), "rows")


# ---------------- dataset statistics ----------------
meta = {}
for fn, sp in (("proficiency_demonstrator_train.json", "train"), ("proficiency_demonstrator_val.json", "val")):
    for m in json.load(open(A / fn, encoding="utf-8"))["annotations"]:
        meta[m["video_paths"]["ego"].split("/")[1]] = dict(s=m["scenario_name"], y=LEVELS[m["proficiency_score"]],
                                                           p=m["origin_participant_id"], split=sp)
names = ld("rating_joint_comm.pt")["names"]  # the 2,111 takes with ego + exo features
used = set(names)
take_ix = {t: j for j, t in enumerate(names)}
R = load_ratings(take_ix)
S = torch.load(Path("embeddings") / "cls_lang_sentences.pt", weights_only=False)
sent_take = Counter(t for t in S["take"] if t in used)
rows = []
for scen in SCEN + ["All"]:
    for split in ("train", "val", "all"):
        ts = [t for t in names if (scen == "All" or meta[t]["s"] == scen) and (split == "all" or meta[t]["split"] == split)]
        if not ts:
            continue
        tj = {take_ix[t] for t in ts}
        lv = Counter(meta[t]["y"] for t in ts)
        rr = [(j, e) for (j, e) in R if j in tj]
        by_t = Counter(j for j, _ in rr)
        rows.append([scen, split, len(ts), len({meta[t]["p"] for t in ts}),
                     len({(meta[t]["p"], meta[t]["s"]) for t in ts}), *[lv.get(k, 0) for k in (1, 2, 3, 4)],
                     len(by_t), sum(1 for v in by_t.values() if v >= 2), len(rr), len({e for _, e in rr}),
                     sum(sent_take[t] for t in ts), sum(1 for t in ts if sent_take[t] > 0)])
all_official = Counter(m["split"] for m in meta.values())
write("dataset_stats.csv",
      ["scenario", "split", "takes", "participants", "participant_x_scenario_groups", "novice", "early_expert",
       "intermediate_expert", "late_expert", "rated_takes", "takes_with_2plus_experts", "ratings", "experts",
       "timestamped_comments", "takes_with_comments"], rows)
print(f"  official annotations: {dict(all_official)} takes; with ego + exo features: {len(names)}")

# ---------------- commentary metrics + training details per joint variant ----------------
VAR = [("joint", "Joint, no commentary", "joint.py"), ("joint_comm", "Joint + commentary 0.5 (FINAL)", "joint.py --lam 0.5 --tag joint_comm"),
       ("joint_lam1", "Joint + commentary 1", "joint.py --lam 1 --tag joint_lam1"),
       ("joint_lam2", "Joint + commentary 2", "joint.py --lam 2 --tag joint_lam2"),
       ("joint_lam4", "Joint + commentary 4", "joint.py --lam 4 --tag joint_lam4"),
       ("joint_comm_rank", "Joint + commentary 0.5 + ranking loss 1", "joint.py --lam 0.5 --rank 1 --tag joint_comm_rank"),
       ("joint_comm_vjepa", "Joint + commentary 0.5, V-JEPA 2", "joint.py --lam 0.5 --feat_cache embeddings/cls_lang_clipfeats_vjepa.pt --clip_sec 2 --tag joint_comm_vjepa"),
       ("joint_comm_official", "FINAL on official split (train -> val)", "joint.py --lam 0.5 --official --tag joint_comm_official"),
       ("joint_comm_official_s5", "FINAL on official split, 5 seeds", "joint.py --lam 0.5 --official --seeds 5 --tag joint_comm_official_s5"),
       ("joint_comm_official_ego", "FINAL on official split, ego view only", "joint.py --lam 0.5 --official --view ego --tag joint_comm_official_ego"),
       ("joint_comm_official_exo", "FINAL on official split, exo view only", "joint.py --lam 0.5 --official --view exo --tag joint_comm_official_exo")]
KEYS = ["pres_auc", "pres_auc_position", "pol_auc", "pol_auc_takemean", "pol_auc_within", "txt_cos", "txt_cos_meanbaseline"]
crow, trow = [], []
for tag, lab, cmd in VAR:
    q = ld(f"rating_{tag}.pt")
    if q.get("comm"):
        crow.append([lab, tag, len(q["comm"])] + [f"{np.mean([c[k] for c in q['comm']]):.3f}" for k in KEYS]
                    + [f"{np.std([c[k] for c in q['comm']]):.3f}" for k in ("pres_auc", "pol_auc", "txt_cos")])
    L = q["log"]
    be = [l["best_epoch"] for l in L]; bl = [l["best_epoch_level"] for l in L]
    trow.append([lab, tag, "python cls_lang/" + cmd, len(L), np.median(be), min(be), max(be), np.median(bl), min(bl), max(bl)])
write("cv_commentary.csv", ["model", "tag", "folds"] + KEYS + ["sd_pres_auc", "sd_pol_auc", "sd_txt_cos"], crow)

sys.path.insert(0, str(C))
from joint import JointNet  # noqa: E402
n_par = sum(p.numel() for p in JointNet(1536).parameters())
n_par_768 = sum(p.numel() for p in JointNet(768).parameters())
n_par_v = sum(p.numel() for p in JointNet(ld("rating_joint_comm_vjepa.pt")["args"].get("d_in", 2048)).parameters()) \
    if False else None
RUNTIME = {"joint": "about 20 min", "joint_comm": "about 27 min", "joint_lam1": "about 27 min", "joint_lam2": "about 27 min",
           "joint_lam4": "about 27 min", "joint_comm_rank": "about 45 min", "joint_comm_vjepa": "about 24 min",
           "joint_comm_official": "2.4 min", "joint_comm_official_s5": "3.9 min", "joint_comm_official_ego": "1.2 min",
           "joint_comm_official_exo": "1.6 min"}
write("cv_training.csv", ["model", "tag", "command (from idea_1)", "trainings (folds x seeds)", "rating_epoch_median",
                          "rating_epoch_min", "rating_epoch_max", "level_epoch_median", "level_epoch_min", "level_epoch_max",
                          "wall_time_rtx4070super", "trainable_params_jointnet"],
      [r + [RUNTIME[r[1]], n_par if not r[1].endswith(("_ego", "_exo")) else n_par_768] for r in trow])

# ---------------- scenario classifier errors ----------------
srow = []
for tag in ("joint_comm", "joint_comm_official"):
    q = ld(f"rating_{tag}.pt")
    errs = q.get("scenario_errors", [])
    total = sum(len(e) for e in errs)
    srow.append([tag, len(errs), total, "; ".join(sorted({t for e in errs for t in e}))])
write("scenario_classifier.csv", ["run", "test_folds", "misclassified_take_instances", "takes"], srow)

# ---------------- published baselines (copied by hand from the papers) ----------------
pub = [
    ["Random", "", "24.9", "24.9", "24.9", "", "", "", "SkillFormer Table 1 / ProfVLM Table 3"],
    ["Majority class", "", "31.1", "31.1", "31.1", "", "", "", "equals the per-scenario training majority on val (we get 0.311 on 466 takes)"],
    ["TimeSformer", "none", "42.3", "40.1", "40.8", "16", "121M", "15", "Ego-Exo4D baseline"],
    ["TimeSformer", "K400", "42.9", "39.1", "38.6", "16", "121M", "15", "Ego-Exo4D baseline"],
    ["TimeSformer", "HowTo100M", "46.8", "38.2", "39.7", "16", "121M", "15", "Ego-Exo4D baseline"],
    ["TimeSformer", "EgoVLP", "44.4", "40.6", "39.5", "16", "121M", "15", "Ego-Exo4D baseline"],
    ["TimeSformer", "EgoVLPv2", "45.9", "38.0", "37.8", "16", "121M", "15", "Ego-Exo4D baseline"],
    ["EgoPulseFormer", "EgoPPG-DB", "45.3", "35.9", "42.4", "16", "121M", "15", ""],
    ["PandaGPT (0-shot)", "", "23.5", "25.6", "25.8", "", "52M", "", "ProfVLM Table 3"],
    ["SmolVLM (0-shot)", "", "34.2", "30.7", "31.1", "", "256M", "", "ProfVLM Table 3"],
    ["SmolVLM2 (0-shot)", "", "26.8", "30.5", "32.2", "", "256M", "", "ProfVLM Table 3"],
    ["SkillFormer", "K600", "45.9", "46.3", "47.5", "32 / 24 / 16", "14M / 20M / 27M", "4", "LoRA TimeSformer + CrossViewFusion; arXiv 2505.08665"],
    ["ProfVLM (MLP)", "", "38.9", "44.2", "42.9", "8", "2.2M", "6", "frozen TimeSformer + LM; arXiv 2509.26278"],
    ["ProfVLM (AGP)", "", "44.2", "45.1", "48.2", "8", "5.3M", "6", "also BERTScore F1 85.53 for generated feedback"],
]
write("published_baselines.csv", ["method", "pretrain", "acc_ego", "acc_exos", "acc_ego_exos", "frames", "trainable_params",
                                  "epochs", "note"], pub)
pub_s = [  # per-scenario accuracy (%) on official val, from the papers
    ["Basketball", 36.19, 55.24, 77.88, 41.00], ["Cooking", 50.00, 35.00, 60.53, 51.00], ["Dance", 51.61, 42.74, 13.68, 60.35],
    ["Music", 58.97, 56.41, 68.10, 56.26], ["Rock Climbing (Bouldering)", 0.00, 17.28, 31.87, 38.74], ["Soccer", 62.50, 75.00, 66.67, 69.75]]
# ours by view on official val (from official_report.py CSVs) next to the papers' Ego / Exos / Ego+Exos columns
def off(suffix, metric, model="Final model [LEVEL]"):
    for r in csv.DictReader(open(OUT / f"official_split{suffix}.csv", encoding="utf-8")):
        if r["eval_set"] == "official val" and r["model"] == model and r["metric"] == metric:
            return r["value"], r["ci_lo"], r["ci_hi"]
    return "", "", ""
vrow = []
for view, suf, note in (("Ego", "_ego", "ego only, 2 seeds"), ("Exo", "_exo", "best exo camera only, 2 seeds"),
                        ("Ego+Exo", "", "ego + best exo, 2 seeds (headline run)"), ("Ego+Exo", "_s5", "ego + best exo, 5 seeds")):
    a_ = off(suf, "level_acc"); b_ = off(suf, "level_bal_acc"); r_ = off(suf, "level_spearman")
    z_ = off(suf, "rating_z", "Final model [RATING]")
    vrow.append(["Ours (final model)", view, note, *a_, b_[0], r_[0], *z_])
for m, k in (("SkillFormer", "SkillFormer"), ("ProfVLM (AGP)", "ProfVLM (AGP)"), ("ProfVLM (MLP)", "ProfVLM (MLP)")):
    pr = next(r for r in pub if r[0] == k)
    for view, col, note in (("Ego", 2, "ego"), ("Exo", 3, "all exo cameras"), ("Ego+Exo", 4, "ego + all exo cameras")):
        vrow.append([m, view, note + " (paper, single run)", f"{float(pr[col]) / 100:.3f}", "", "", "", "", "", "", ""])
write("official_views_vs_papers.csv", ["model", "view", "note", "level_acc", "acc_ci_lo", "acc_ci_hi", "level_bal_acc",
                                       "level_spearman", "rating_z", "rating_z_ci_lo", "rating_z_ci_hi"], vrow)
write("published_per_scenario.csv", ["scenario", "majority", "baseline_ego_exos", "skillformer_ego_exos", "profvlm_agp_ego_exos"], pub_s)

# ---------------- log copies ----------------
for f in ["all_metrics_report", "official_report", "lamsweep_rating_report", "lamsweep_level_report", "comm_rank_rating_report",
          "comm_rank_level_report", "step23_rating_report", "step23_level_report", "joint_rating_report", "joint_level_report",
          "rating_report", "rating_ps_report", "rating_ps_video_report", "rating_dance_report", "rating_ens_video_report",
          "joint_comm", "joint_comm_official", "joint_comm_rank", "joint_comm_vjepa", "joint", "joint_lam1", "joint_lam2", "joint_lam4",
          "report", "summary_report", "pretrain_report", "comm_auc_sweep", "level_from_rating_cls",
          "joint_comm_official_s5", "joint_comm_official_ego", "joint_comm_official_exo", "official_report_s5",
          "official_report_ego", "official_report_exo", "commentary_examples"]:
    src = C / f"{f}.log"
    if src.exists():
        shutil.copy(src, LOGS / f"{f}.txt")
print("copied logs to", LOGS)
