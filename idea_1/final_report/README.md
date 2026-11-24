# idea_1 final report pack: proficiency estimation on Ego-Exo4D

Everything needed to write the final report for idea_1, in files that are tracked by git. Read this first.

**Important:** `.gitignore` excludes `data/`, `embeddings/`, every `*.pt`, `*.log` and `*.out`. On a fresh clone, none of
the run outputs, feature caches or logs exist and **no experiment can be rerun or rescored** (feature extraction alone
took days on an RTX 4070 SUPER, and the Ego-Exo4D data needs a licence). Every number therefore lives here as CSV,
Markdown or `.txt`. Treat these files as the source of truth; do not try to recompute.

## Contents

| Path | What it is |
|---|---|
| `report_doc.pdf` | The full results write-up (29 pages, exported from the Claude Doc on 2026-11-23), with the architecture diagram |
| `report_doc.md` | The same document as Markdown; easier to search and quote. The diagram is replaced by a placeholder. **Claude instances: read this copy.** Claude Code's file reader reports the PDF as password-protected; it is not (a normal Chromium PDF with an incremental-update section) and opens fine in any PDF viewer |
| `tables/*.csv` | Every result number used in the doc, machine-readable (see "Tables" below) |
| `figures/*.png` | Six report figures (see "Figures") |
| `logs/*.txt` | Copies of the raw report and training logs the CSVs come from |
| `build_tables.py`, `make_figures.py` | How the tables and figures were made (`make_figures.py` runs on a fresh clone; it needs only the CSVs) |

Online version of the doc (may be newer than the PDF): https://claude.ai/artifact/4WX6GpqoYueZrJtpN1N4sT

## The project in one paragraph

Ego-Exo4D "proficiency demonstrator": 2,111 takes (ego + exo video) of people doing Basketball, Cooking, Dance, Music,
Rock Climbing and Soccer. The **main target is the 1–10 expert rating** that Ego-Exo4D's expert commentators gave each
take; the **official 4-class proficiency level** (Novice, Early, Intermediate, Late Expert) is the second output. The
final model is **one small transformer (648k parameters) over frozen TimeSformer-K600 clip features** of the ego and the
best exo video, with two task tokens ([RATING], [LEVEL]), trained jointly on ratings, levels and 188k timestamped expert
comments (auxiliary per-second commentary heads, loss weight 0.5). **Inference is video only**, and even the scenario is
predicted from video. Headline numbers (4×5 participant-grouped CV): rating agreement with held-out experts 0.638 z
(one human expert: 0.772); level accuracy 0.460 (frozen probe 0.451, scenario prior 0.345). Official split: level
45.8% with 2 seeds / 44.5% with 5 seeds (single seeds 41–46%; SkillFormer 47.5%, ProfVLM 48.2%; our 95% interval ±14
points, so indistinguishable), rating 0.720 / 0.708 z (human 0.769). The per-second commentary heads predict *when*
experts comment well (per-take presence AUC median 0.79) but *what* aspect only at the level of each scenario's typical
aspects.

## Decisions and constraints (set by the user; keep them in the report)

- **Target**: expert rating first (pivot on 2026-11-22, after showing level is a person-level label with only about 280
  independent units); level is the second output. Level results are still reported for comparison with the literature.
- **Video-only inference.** Scenario is predicted from video (misclassifies 1 take of 2,111: `sfu_basketball_04_38`).
- **No audio** (most Dance, Music and Cooking takes have no usable sound) and **no pose** (an oracle showed pose helps
  expert ratings but hurts level; the user ruled it out on 2026-11-19 and again on 2026-11-23).
- **Multimodal = training signals**: video + expert ratings + official levels + timestamped expert commentary. Language is
  never an input at test time; the model outputs per-second commentary predictions.
- **Final model = commentary weight 0.5** (`joint.py --lam 0.5`, tag `joint_comm`), the original default, kept for
  comparability with earlier runs. Weight 4 was slightly better on commentary outputs and identical on rating and level;
  the user chose 0.5. Do not present weight 4 as the final model.
- **Adoption rule** used throughout: a change is adopted only if it gains ≥ +0.02 z on rating without a significant level
  drop. Nothing after the joint model passed it.

## Evaluation protocol (needed to read every number)

- **Data**: all official train + val demonstrator takes with both an ego and an exo video: 2,111 takes, 227
  participants (`tables/dataset_stats.csv`). 1,511 takes are rated, 5,601 ratings from 38 experts; each expert rates
  one scenario only.
- **CV**: 5-fold StratifiedGroupKFold grouped by participant (no person in train and test), 4 repeats
  (random_state 0–3), 2 training seeds averaged per fold. Early stopping on an inner participant split (1/7 of
  training participants), per task. Predictions are centred within (repeat, fold, scenario) and averaged over repeats.
- **Rating metric**: held-out-expert protocol. For every take with ≥ 2 rated experts and each of its experts, the
  model's prediction is compared with the mean rating of the *other* experts: 5,565 (take, expert) rows. Within-scenario
  Spearman, on raw ratings and on per-expert z-scored ratings ("z", the main number). The human reference is the held-out
  expert's own rating on the same rows (0.700 raw, 0.772 z [0.737, 0.798]).
- **Level metrics**: 4-class accuracy, balanced accuracy, within-scenario Spearman of the level score, on all 2,111 takes.
- **Intervals**: 95% from 1,000 participant bootstrap resamples; paired differences resample the same participants.
- **Cross-target readouts** (marked † / ‡ in the doc): a rating-only model's level accuracy comes from ranking its rating
  score within scenario and cutting at training-fold class proportions; a level-only model's rating agreement uses its
  level score. They show what a specialist gives up, not its best possible score on the other target.
- **Official split**: train on official train (1,655 takes), test on official val (456 takes, 56 participants; the papers
  list 466, 10 lack ego or exo features here). One split, 2 seeds; intervals from val-participant bootstrap.

## Tables (`tables/`)

| File | Content | Made by |
|---|---|---|
| `cv_all_models.csv` | Every model on both targets: rating raw / z, level accuracy / balanced / Spearman (main results table) | `cls_lang/all_metrics.py` |
| `cv_paired_vs_final.csv` | Each model minus the final model, with 95% CIs, for rating z, level accuracy, level Spearman | same |
| `cv_per_scenario.csv` | Per scenario: rating z, level accuracy, level Spearman for every model, human, scenario prior | same |
| `cv_dance_only_fps.csv` | Dance-only 8 FPS vs 30 FPS ablation (own folds) | same |
| `cv_commentary.csv` | Held-out commentary metrics per joint variant: presence AUC (+ position baseline), praise-vs-tip AUC (+ baselines), text cosine (+ mean-comment baseline), fold SDs | `final_report/build_tables.py` |
| `cv_training.csv` | Command, checkpoint epochs, wall time, parameter count per joint variant | same |
| `official_split.csv` | Official val and the SkillFormer test half: level and rating metrics with CIs; final − probe differences | `cls_lang/official_report.py` |
| `official_split_per_scenario.csv` | Per-scenario official-split numbers | same |
| `official_split_{s5,ego,exo}.csv` (+ `_per_scenario`, `_per_seed`) | Same, for the 5-seed run and the ego-only / exo-only runs; `_per_seed` = each training seed alone | `official_report.py <tag>` |
| `official_views_vs_papers.csv` | Ours by view (ego / exo / both; 2 and 5 seeds) next to SkillFormer's and ProfVLM's Ego / Exos / Ego+Exos columns | `build_tables.py` |
| `commentary_examples.csv` | Per example val take (one per scenario, median presence AUC): the 4 highest presence peaks, top-2 predicted aspects with praise / tip, and the real comments' aspects within ±3 s (no comment text) | `cls_lang/commentary_examples.py` |
| `commentary_aspect_check.csv` | Over all val clips with a real comment: is its aspect the model's top-1 / top-3 aspect, vs the most frequent aspect; praise / tip agreement | same |
| `published_baselines.csv`, `published_per_scenario.csv` | SkillFormer / ProfVLM / Ego-Exo4D baseline numbers copied from the papers | hand-entered in `build_tables.py` |
| `dataset_stats.csv` | Takes, participants, level counts, ratings, experts, comments per scenario and split | `build_tables.py` |
| `scenario_classifier.csv` | Takes whose scenario the video classifier got wrong | `build_tables.py` |

Model names in the CSVs: "Joint + commentary 0.5 (final)" is the final model; "Joint, no commentary" is weight 0;
"Level + hard commentary CLS (earlier)" is the best level-target model from before the rating pivot.

Results that exist only in the doc (earlier experiments without saved per-take outputs, mostly the level-target phase)
are in `report_doc.md`: sections "Earlier level-target results", "Commentary outputs and the inter-expert ceiling",
"Ablations: how the language is used", "Ablations: encoder, representation, adaptation and decoder", "Ablations: other
supervision and modality oracles", "Why level is hard: the label ceiling". Note that the earliest studies used the 1,338
takes with commentary, not all 2,111; the doc marks this per row.

## Figures

`figures/` (made from the CSVs by `make_figures.py`):

| File | Shows |
|---|---|
| `fig1_rating_vs_level.png` | Every CV model on both targets; only the joint models are good at both |
| `fig2_rating_per_scenario.png` | Rating z per scenario: one human vs frozen probe vs final model |
| `fig3_official_level.png` | Official-split level accuracy with CIs next to SkillFormer and ProfVLM |
| `fig4_commentary_weight.png` | Commentary weight sweep: rating and level flat, commentary outputs improve |
| `fig5_commentary_examples.png` | Per-second presence over one typical val take per scenario, real comment times (praise / tip), top predicted aspect at the peaks (made by `cls_lang/commentary_examples.py`, needs local data) |
| `fig6_views_vs_papers.png` | Official-split level accuracy by input view, ours (with CIs) vs SkillFormer and ProfVLM |

Older figures from the exploration phase are in `idea_1/figures/` (tracked), each made by the notebook of the same name
(e.g. `pose_oracle.png` by `pose_oracle.ipynb`, `expert_rating.png` by `expert_rating.ipynb`).

## Headline results with provenance

| Claim | Number | Source |
|---|---|---|
| Final model, rating vs held-out experts (CV) | 0.660 raw / 0.638 z | `cv_all_models.csv` |
| One human expert vs the others | 0.700 raw / 0.772 z [0.737, 0.798] | `cv_all_models.csv` note |
| Final vs frozen probe, rating | +0.095 z [+0.057, +0.136] | `cv_paired_vs_final.csv` (sign flipped) |
| Final model, level (CV) | acc 0.460, balanced 0.458, Spearman 0.464 | `cv_all_models.csv` |
| Frozen probe, level (CV) | 0.451 / 0.438 / 0.452; scenario prior 0.345 / 0.321 | `cv_all_models.csv` |
| Rating-only model's level loss vs final | acc −0.075 [−0.139, −0.015], Spearman −0.165 [−0.278, −0.056] | `cv_paired_vs_final.csv` |
| Level-only model's rating loss vs final | −0.182 z [−0.259, −0.111] | `cv_paired_vs_final.csv` |
| Commentary outputs (final) | presence AUC 0.690 (position baseline 0.509), praise-vs-tip 0.570, text cosine 0.587 (baseline 0.571) | `cv_commentary.csv` |
| Commentary weight 0 → 4 | rating and level within noise (all CIs span 0); presence 0.690 → 0.721 | `cv_paired_vs_final.csv`, `cv_commentary.csv` |
| Pairwise ranking loss | rating −0.002 z [−0.013, +0.008]; not adopted | `cv_paired_vs_final.csv` |
| V-JEPA 2 instead of TimeSformer | rating −0.016 z [−0.042, +0.009]; not adopted | `cv_paired_vs_final.csv` |
| Official split, level accuracy | 45.8 [31.9, 60.3] (probe 45.2, prior 31.4; SkillFormer 47.5, ProfVLM 48.2) | `official_split.csv`, `published_baselines.csv` |
| Official split, rating | 0.734 raw / 0.720 z (human 0.689 / 0.769); +0.107 z [+0.036, +0.183] over probe | `official_split.csv` |
| SkillFormer test half of val (229 takes) | ours 44.5%; our midterm SkillFormer runs 44.5% (ego) and 44.5% (exo) | `official_split.csv`; `idea_1/midterm/idea_1_midterm.ipynb` cell 0 |
| Official split, 5 seeds | level 44.5 [30.6, 58.7], rating 0.708 z; single seeds: acc 41.4–46.3 (43.4 ± 1.8), rating 0.638–0.705 z | `official_split_s5.csv`, `official_split_s5_per_seed.csv` |
| Official split by view (2 seeds) | ego only 43.9% / 0.673 z; exo only (best camera) 45.8% / 0.704 z; both 45.8% / 0.720 z. Papers: SkillFormer 45.9 / 46.3 / 47.5, ProfVLM 44.2 / 45.1 / 48.2 | `official_views_vs_papers.csv` |
| Commentary outputs on val (5-seed run) | presence AUC 0.723 overall (position 0.521), per-take median 0.788; aspect top-1 0.13–0.49, beats the most-frequent-aspect guess only in Music (0.281 vs 0.184); praise / tip agreement 0.50–0.60 | `cv_commentary.csv`, `commentary_aspect_check.csv` |

## Experiment history (chronological, with verdicts and where to find each)

**Phase 1: midterm, LLM route (Nov 2026, 1,338 commentary takes; tested on the SkillFormer test half of val).**
Qwen3.5-4B fine-tuned (LoRA) to output the level from video + commentary, scored on the 164 test-half takes with
commentary: 40.2% with commentary, 29.9% video only, 42.1% commentary only, majority 35.4% (`midterm/idea_1_midterm.ipynb`). Zero-shot Qwen video features: about 0.35
Spearman. Our SkillFormer runs on the same half: 44.5% ego, 44.5% exo. Failure analysis vs clip duration
(`midterm/idea_1_midterm_failure_analysis.ipynb`), paired video/commentary embeddings (`midterm/idea_1_midterm_representation.ipynb`).

**Phase 2: representation search for level (notebooks in `idea_1/`, mostly 1,338 takes, linear probes).**
`continuous_sampling`, `view_comparison` (1 FPS windows, ego vs exo views), `pooling` (patch-token std beats mean,
+0.05–0.09), `good_tip_cca`, `skillformer_goodtip`, `goodtip_aspects` (good-execution vs tip moments: not separable
within a take from video), `why_cca`, `why_cca_cv`, `why_concepts` (expert "why" text as supervision: no gain over label
PLS), `exo_8fps` and `timesformer_stage1` (frozen TimeSformer-K600 on exo at 8 FPS: 0.464 vs Qwen 0.350, the key
encoder switch), `smollm_stage2` (SmolLM2-135M decoder over clip tokens: 0.295, worse than a linear probe),
`fusion_vjepa` (ego + exo fusion +0.016 n.s.; V-JEPA 2 −0.091), `expert_rating` (experts agree with each other at
0.70 / 0.77 z but with level at only 0.41; rating is the observable target), `pose_oracle` (pose: −0.088 on level,
+0.089 on ratings), `motion_stats`, `multicam`, `person_crop`, `ssv2_encoder` (all no gain on 2,111 takes).
Script folders: `ssl_adapt/` (self-supervised LoRA adaptation, no gain), `level_ft/` (supervised LoRA fine-tune on level,
−0.028 n.s.), `decoder/` (MLP probes, no gain), `label_units/` (learning curve in label units), `dense/` (30 FPS Dance
features), `vjepa/` (V-JEPA 2 features). Summary table: doc section "Ablations: encoder, representation, adaptation and decoder".

**Phase 3: CLS transformer with time-aligned commentary, level target (`cls_lang/train.py`, 4×5 CV on 2,111 takes).**
Level-only CLS 0.450 Spearman; + hard-selected expert commentary 0.458 / 0.472 acc (best level model, n.s. vs probe
0.452); commentary-only pretraining (`pretrain.py`), expert-summary target (`summary_target.py`), commentary target
variants (`comm_auc.py`): no level gain. Doc sections "Earlier level-target …" and "Ablations: how the language is used".

**Phase 4: rating target (2026-11-22, `cls_lang/rating.py`).** CLS transformer on the z-scored expert rating beats the
frozen probe clearly (+0.084 raw, CI excludes 0), the first win of a learned temporal model. Per-scenario heads,
video-predicted scenario, all-expert commentary loss, 30 FPS (Dance), ensemble with the per-scenario probe: no further
gain. Doc section "Ablations: rating target"; `cv_all_models.csv`, `cv_dance_only_fps.csv`.

**Phase 5: joint rating + level model (2026-11-22/23, `cls_lang/joint.py`).** Ties each specialist on its own target and
is the only model good on both. + commentary loss weight 0.5 = **final model**. Then: weight sweep 1/2/4 (rating, level
flat; commentary outputs improve), V-JEPA 2 encoder (−0.016 z), within-expert pairwise ranking loss (−0.002 z). A LoRA
fine-tune pilot of TimeSformer layers 8–11 with the ranking loss (`finetune/rating_ft.py`, Dance fold 0) was **stopped
early** by the user because it changed two things at once; it has no reportable result. Official-split run
(`joint.py --official`) for the comparison with SkillFormer and ProfVLM, then (2026-11-23, user-approved) ego-only and
exo-only versions (`--view`), a 5-seed run (`--seeds 5`), and qualitative commentary examples with an aspect readout
(`commentary_examples.py`).

## Code map (`idea_1/cls_lang/`, run everything from `idea_1/`)

| Script | Role |
|---|---|
| `train.py` | Data loading (`load_takes`, `load_ratings`, `Sentences`), the CLS `Net`, level-target CLS runs |
| `build_targets.py` | Builds the per-sentence commentary targets (timestamps, polarity, aspects, MiniLM embeddings) |
| `rating.py` | Rating-only CLS runs and the rating report (`--report --report_tags …`) |
| `joint.py` | **Final model** (`--lam 0.5`); options `--rank`, `--feat_cache/--clip_sec` (V-JEPA), `--official` (also saves per-seed predictions and per-clip commentary outputs), `--view ego\|exo`, `--seeds` |
| `level_from_rating_cls.py` | Level report for the joint models and level-readout comparisons |
| `all_metrics.py` | Every model on both targets → `final_report/tables/cv_*.csv` |
| `official_report.py` | Official-split report → `final_report/tables/official_split*.csv` (argument: run tag) |
| `commentary_examples.py` | Commentary examples, aspect readout check and fig 5 (from the 5-seed official run) |
| `comm_auc.py`, `pretrain*.py`, `summary_*.py`, `report.py` | Phase 3 commentary experiments and reports |

Reproduction commands with run times are in the doc's "Reproducibility" section and in `tables/cv_training.csv`.

## Pitfalls when writing the report

- Do not mix CV numbers (4×5 folds over train + val) with official-split numbers (one split, val only); they answer
  different questions. Only the official-split numbers are comparable with SkillFormer / ProfVLM.
- Level Spearman for rating-only models is 0.30 on all 2,111 takes; older drafts showed about 0.42, which was on the
  1,511 rated takes only.
- "Rating z" means per-expert z-scored ratings; raw and z are both reported, z is the headline.
- The rating-model CLS → level probe is not cross-fitted (training-take CLS vectors are in-sample).
- The V-JEPA commentary baselines differ (2 s clips): position 0.519, text 0.578.
- Published numbers come with no intervals; our ±14-point level interval on val applies to theirs as well.
- Training is not bit-reproducible on the GPU: rerunning the same seed gives different early-stopping epochs and
  slightly different numbers. On the official split, single seeds span 41–46% level accuracy; quote the official level
  accuracy as "about 44–46%" or give both the 2-seed (45.8) and 5-seed (44.5) numbers, not 45.8 alone.
- Our "exo" is the single best exo camera; the papers' "Exos" uses all exo cameras.
- The commentary polarity is trained on per-expert-centred targets, so "praise" from the model means warmer than that
  expert's usual tone; the aspect labels come from an auxiliary classifier on the predicted embedding, not from a head
  trained for aspects. No comment text is generated or shown.
- Earliest notebook studies used 1,338 commentary takes; compare each row with its own baseline.

## Suggested report outline

1. Task and data (`dataset_stats.csv`; why rating is the main target: doc "Why level is hard").
2. Method: features, joint transformer, losses, training (doc "Current model", architecture diagram in the PDF).
3. Evaluation protocol (above).
4. Results: main table (`cv_all_models.csv`, fig 1), per scenario (fig 2), commentary outputs (`cv_commentary.csv`).
5. Comparison with published methods (`official_split.csv`, `official_views_vs_papers.csv`, figs 3 and 6), seed spread.
5b. What the commentary heads say (`commentary_aspect_check.csv`, `commentary_examples.csv`, fig 5).
6. Ablations: rating target, commentary weight (fig 4), ranking loss, encoder, earlier representation search.
7. Limitations and future work (doc "Limitations", "Room for improvement").
