# Ego-Exo4D Proficiency: Joint Expert-Rating + Level Model from Video

Nov 22, 2026 · @Kevin

## Summary

Our current best model is one transformer over frozen TimeSformer clip features from ego + exo video, trained with an auxiliary loss on 188k timestamped expert comments. It outputs an expert-style 1–10 rating, the official proficiency level and, per second, whether an expert would comment, praise vs tip and what they would say. It runs on video alone; even the scenario is predicted from video (99.95% correct).

| Output | Joint + commentary model (final) | Frozen linear probe | Human reference |
| --- | --- | --- | --- |
| Expert rating vs held-out experts, within-scenario Spearman (raw / per-expert z) | 0.660 / 0.638 | 0.560 / 0.544 | 0.700 / 0.772 (one expert vs the others) |
| Official level: 4-class accuracy / balanced accuracy / Spearman | 0.460 / 0.458 / 0.464 | 0.451 / 0.438 / 0.452 | scenario-only prior: 0.345 / 0.321 / –; one expert's rating vs level: Spearman 0.403 |
| Commentary: presence AUC / praise-vs-tip AUC / text cosine | 0.690 / 0.570 / 0.587 | – | position-only presence 0.509; praise-vs-tip chance 0.5; mean-comment text 0.571 |

The rating is where the model clearly works. The transformer beats the frozen probe by +0.100 raw / +0.095 z (95% participant bootstrap on z \[+0.057, +0.136\], excludes zero). In Rock Climbing (0.806 vs human 0.845) and Basketball (0.764 vs 0.823) it is close to a single human expert. The remaining gap is mostly Dance, Music and Cooking.

Level does not move past the probe (accuracy +0.009 \[−0.022, +0.039\], Spearman +0.012 \[−0.022, +0.044\]). Level is constant within a participant and scenario, so the 2,111 takes carry only about 280 independent labels; expert ratings vary from take to take, which is why the same architecture learns them.

Training both targets in one network is what makes one model good at both. Each single-target model gives up the other target: the rating-only model's score classifies level worse than the final model (accuracy −0.075 \[−0.139, −0.015\], Spearman −0.165 \[−0.278, −0.056\]), and the level-only models rank takes much worse than the experts do (rating −0.18 z \[−0.26, −0.11\]). The joint model ties each specialist on its own target: rating-only vs final −0.001 z \[−0.030, +0.025\]; best level-only model vs final, accuracy +0.011 \[−0.007, +0.029\], Spearman −0.006 \[−0.028, +0.015\].

Multimodality lives in training: video, the experts' ratings, official levels and 188k timestamped expert comments. Inference is video only. The commentary loss at its original weight 0.5 costs nothing on rating or level (vs no commentary: rating +0.003 z \[−0.007, +0.017\], level accuracy −0.002 \[−0.009, +0.005\]) and adds the per-second commentary outputs. Nothing tried on top moved rating or level: a higher commentary weight (up to 4), a within-expert pairwise ranking loss, and V-JEPA 2 in place of TimeSformer all stay within 0.016 z of the final model, none significant.

On the official Ego-Exo4D split, the final model's level accuracy (45.8% with 2 seeds, 44.5% with 5; single seeds 41–46%) is within noise of the published SkillFormer (47.5%) and ProfVLM (48.2%) results while using one exo camera instead of all; its rating agreement with held-out experts (0.720 z) approaches a single human expert's (0.769). Neither paper reports the expert rating.

## Current model: joint rating + level transformer

- **Clip features:** each 1 s clip (8 frames at 8 FPS) goes through frozen TimeSformer-base (Kinetics-600); we keep the standard deviation of the layer-12 patch tokens (768 dims) for the ego camera and the best exo camera and concatenate them (1,536 dims). Features are standardised with training-fold per-scenario means and a pooled standard deviation.
- **Scenario from video:** per fold, a logistic regression (standardise, PCA 128) on take-mean features predicts each test take's scenario. Over 4 repeats it misclassifies one take (sfu\_basketball\_04\_38, as Soccer). Using the predicted instead of the true scenario costs −0.002 \[−0.005, 0.000\].
- **Encoder:** Linear 1536→128 plus a sinusoidal time encoding; two learned task tokens, \[RATING\] and \[LEVEL\], each plus a scenario embedding, in front of the clip tokens; a 2-layer pre-norm transformer (4 heads, dropout 0.2).
- **Heads:** \[RATING\] feeds six per-scenario linear heads and the predicted scenario picks one. \[LEVEL\] feeds a level score and 4 class logits, each with a per-scenario bias. Every clip token feeds three commentary heads: presence (does an expert comment here), polarity per aspect (praise vs tip) and a sentence-embedding of what they say.
- **Rating target:** each expert's ratings are z-scored with that expert's mean and spread on training-fold takes; a take's target is the mean of its experts' z-scores.
- **Loss:** MSE + cross-entropy on level for all training takes, plus MSE on the rating for rated training takes, equal weights; plus 0.5 × the commentary loss (presence BCE + polarity squared error + text 1 − cosine) on clips of takes with comments. A comment at time T labels the clips covering T − 2 to T seconds; all experts' comments are used, polarity centred per expert with training-fold means.
- **Training:** random 384-clip crops, AdamW (lr 3e-4, weight decay 0.05), batch 16, up to 60 epochs, 2 seeds averaged. Early stopping uses an inner participant split (1/7 of training participants) separately per task: the rating checkpoint is the epoch with the best inner rating Spearman (median epoch 7), the level checkpoint the best inner level Spearman (median epoch 3). Training stops once neither has improved for 8 epochs.
- **Inference:** the whole take through each task's checkpoint; the commentary outputs come from the rating checkpoint. Deployed, that is two saved copies of one network; a single compromise checkpoint would cost a little on one task.

## Evaluating the rating

- 1,511 rated takes carry 5,601 ratings from 38 experts. Every expert rates exactly one scenario.
- **Held-out-expert protocol:** for each take with at least two rated experts and each of its experts, the model's prediction is compared with the mean of the *other* experts' ratings, raw and per-expert z-scored: 5,565 (take, expert) rows. The human reference is the held-out expert's own rating in the same rows.
- Same 4×5 participant-grouped folds over all 2,111 takes as the level evaluation (see Task, data and evaluation). Predictions are centred within (repeat, fold, scenario) and averaged over repeats; within-scenario Spearman; 95% intervals from 1,000 participant bootstrap resamples.

## Results: every model on both targets

Every model is scored on both targets in the same 4×5 participant-grouped folds (cls\_lang/all\_metrics.py). Rating: within-scenario Spearman with the held-out experts on 5,565 (take, expert) rows. Level: accuracy, balanced accuracy and within-scenario Spearman on all 2,111 takes. Each model's own output is used where it has one; the cross-target cells are marked:

- † Rating-only model, no level output: its rating score is read as level. Spearman directly; for accuracy, the test takes of each scenario are ranked by score and cut at that scenario's training-fold level proportions (training labels only).
- ‡ Level-only model, no rating output: its level score is compared with the held-out experts.

| Model | Rating, raw | Rating, z | Level accuracy | Balanced accuracy | Level Spearman |
| --- | --- | --- | --- | --- | --- |
| Human: one expert vs the others | 0.700 | 0.772 \[0.737, 0.798\] | – | – | 0.403 (rating vs level) |
| Scenario only (most frequent training level) | – | – | 0.345 | 0.321 | – |
| Frozen probe (ridge on rating; PCA probe on level) | 0.560 | 0.544 | 0.451 | 0.438 | 0.452 |
| Frozen probe, per-scenario ridge on rating | 0.625 | 0.609 | 0.377 † | 0.366 † | 0.280 † |
| Rating-only CLS, per-scenario heads | 0.656 | 0.638 | 0.386 † | 0.377 † | 0.299 † |
| Rating-only CLS, level probe on its CLS (rerun) | 0.658 | 0.640 | 0.393 | 0.354 | 0.450 |
| Level-only CLS (earlier, true scenario) | 0.476 ‡ | 0.453 ‡ | 0.465 | 0.460 | 0.450 |
| Level + hard commentary CLS (earlier, true scenario) | 0.480 ‡ | 0.457 ‡ | 0.472 | 0.467 | 0.458 |
| Joint model, no commentary | 0.655 | 0.635 | 0.462 | 0.459 | 0.465 |
| **Joint + commentary, weight 0.5 (final)** | **0.660** | **0.638** | **0.460** | **0.458** | **0.464** |

Earlier versions of this table showed a level Spearman of about 0.42 for the rating-only rows; that was on the 1,511 rated takes only. On all 2,111 takes it is 0.30.

Difference from the final model (other model − final, 95% participant bootstrap; bold where the interval excludes zero):

| Model | Rating, z | Level accuracy | Level Spearman |
| --- | --- | --- | --- |
| Frozen probe | **−0.095 \[−0.136, −0.057\]** | −0.009 \[−0.039, +0.022\] | −0.012 \[−0.044, +0.022\] |
| Frozen probe, per-scenario ridge | −0.029 \[−0.070, +0.013\] | **−0.084 \[−0.150, −0.025\]** | **−0.183 \[−0.305, −0.058\]** |
| Rating-only CLS, per-scenario heads | −0.001 \[−0.030, +0.025\] | **−0.075 \[−0.139, −0.015\]** | **−0.165 \[−0.278, −0.056\]** |
| Rating-only CLS, level probe on its CLS | +0.002 \[−0.028, +0.028\] | −0.068 \[−0.144, +0.006\] | −0.014 \[−0.060, +0.035\] |
| Level-only CLS | **−0.185 \[−0.263, −0.111\]** | +0.005 \[−0.014, +0.023\] | −0.014 \[−0.037, +0.007\] |
| Level + hard commentary CLS | **−0.182 \[−0.259, −0.111\]** | +0.011 \[−0.007, +0.029\] | −0.006 \[−0.028, +0.015\] |
| Joint model, no commentary | −0.003 \[−0.017, +0.007\] | +0.002 \[−0.005, +0.009\] | +0.002 \[−0.006, +0.010\] |

Only the joint models are never significantly worse than another model on either target. The level-only models' scores rank takes worse than even the frozen rating probe (0.453 vs 0.544 z).

Per scenario, each cell is rating z / level accuracy / level Spearman († and ‡ as above):

| Scenario | Rating rows / takes | Human (rating z) | Scenario-only accuracy | Frozen probe | Per-scenario ridge † | Rating-only CLS † | Level + hard CLS ‡ | Joint, no commentary | Final model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rock Climbing | 1,246 / 722 | 0.845 | 0.076 | 0.754 / 0.328 / 0.449 | 0.730 / 0.280 / 0.175 | 0.818 / 0.289 / 0.196 | 0.551 / 0.403 / 0.451 | 0.805 / 0.393 / 0.463 | 0.806 / 0.395 / 0.457 |
| Basketball | 1,410 / 540 | 0.823 | 0.443 | 0.708 / 0.606 / 0.625 | 0.751 / 0.485 / 0.535 | 0.759 / 0.470 / 0.585 | 0.704 / 0.578 / 0.628 | 0.762 / 0.554 / 0.640 | 0.764 / 0.556 / 0.643 |
| Soccer | 230 / 77 | 0.745 | 0.675 | 0.557 / 0.584 / 0.277 | 0.604 / 0.468 / −0.303 | 0.575 / 0.442 / −0.324 | −0.451 / 0.753 / 0.190 | 0.573 / 0.740 / 0.217 | 0.559 / 0.753 / 0.257 |
| Dance | 1,641 / 501 | 0.698 | 0.485 | 0.261 / 0.393 / 0.374 | 0.480 / 0.361 / 0.317 | 0.370 / 0.395 / 0.311 | 0.316 / 0.403 / 0.339 | 0.375 / 0.413 / 0.360 | 0.414 / 0.395 / 0.367 |
| Music | 565 / 172 | 0.727 | 0.570 | 0.261 / 0.610 / −0.017 | 0.216 / 0.442 / 0.096 | 0.245 / 0.448 / 0.233 | 0.269 / 0.541 / −0.009 | 0.325 / 0.506 / −0.057 | 0.318 / 0.535 / −0.027 |
| Cooking | 473 / 99 | 0.495 | 0.424 | −0.077 / 0.424 / 0.015 | 0.124 / 0.384 / 0.182 | 0.261 / 0.424 / 0.283 | 0.165 / 0.404 / 0.164 | 0.236 / 0.424 / 0.089 | 0.199 / 0.394 / 0.105 |

Commentary helps Dance most on rating (0.414 vs 0.375 z without it; 0.411–0.423 at every weight tried). The experts' rationales in Dance and Music talk about rhythm, tempo, pitch and finger technique: audio and fine motion that a frozen whole-frame encoder at 1 s resolution does not see. On level, the final model is below the scenario-only prior in Dance (0.395 vs 0.485) and Music (0.535 vs 0.570), and its Music level Spearman is about zero. Soccer (54 rated takes, 77 takes) and Cooking (99) are small and swing between models; the level-only model's score even runs against the experts in Soccer (−0.451 z).

All level outputs are within noise of the frozen probe (final model vs probe: accuracy +0.009 \[−0.022, +0.039\], balanced +0.020 \[−0.014, +0.054\], Spearman +0.012 \[−0.022, +0.044\]). A level probe on the rating model's CLS keeps the ranking (0.450) but loses classification (balanced −0.084 vs the frozen probe, interval excludes zero), most of all in Rock Climbing and Cooking: experts' ratings and the official level measure different things.

## Official split and published methods

SkillFormer and ProfVLM report 4-class level accuracy on the official val split, one run, no intervals. To compare, the final model was trained once on the official train takes (1,655 with ego + exo features; inner participant split for early stopping as usual) and tested on official val: 456 takes from 56 participants (the papers list 466; 10 lack ego or exo features here). Same code, `joint.py --official`, 2 seeds. Intervals: 1,000 bootstrap resamples of val participants.

| Model | Views at test | Level accuracy (%) | Balanced accuracy (%) | Level Spearman |
| --- | --- | --- | --- | --- |
| Majority class / scenario prior (papers: 31.1) | – | 31.4 \[18.2, 47.1\] | 33.2 | – |
| Ego-Exo4D TimeSformer baselines (papers) | ego, exos or both | 37.8–46.8 | – | – |
| SkillFormer, Ego+Exos (paper) | ego + all exos | 47.5 | – | – |
| ProfVLM, Ego+Exos (paper) | ego + all exos | 48.2 | – | – |
| Frozen PCA probe (ours) | ego + best exo | 45.2 \[32.2, 59.1\] | 41.2 | 0.369 |
| **Final model (ours)** | ego + best exo | **45.8 \[31.9, 60.3\]** | **41.7 \[29.7, 55.3\]** | **0.398 \[0.12, 0.64\]** |

Expert rating on the same val takes (1,335 (take, expert) rows on 332 takes), which neither paper predicts:

| Model | Rating, raw | Rating, z |
| --- | --- | --- |
| Human: one expert vs the others | 0.689 \[0.564, 0.759\] | 0.769 \[0.662, 0.823\] |
| Frozen ridge probe | 0.631 | 0.613 |
| **Final model** | **0.734 \[0.631, 0.790\]** | **0.720 \[0.610, 0.779\]** |
| Final − frozen probe | +0.103 \[+0.031, +0.181\] | +0.107 \[+0.036, +0.183\] |

- Level: the final model is 1.7 points below SkillFormer and 2.4 below ProfVLM, against a 95% interval about ±14 points wide (level is fixed per participant and val has 56 participants). The three cannot be told apart on this split. On the SkillFormer test half of val (229 takes), the final model scores 44.5%, the same as our midterm SkillFormer runs on those takes (44.5% with ego, 44.5% with exo).
- Rating: the gain over the frozen probe holds on the official split, interval excluding zero, and the model is within 0.05 z of a single human expert. Per scenario it is above one human in Dance (0.693 vs 0.598, 123 takes) and close in Rock Climbing (0.821 vs 0.876).
- Differences in setup: both papers fine-tune or fuse all exo cameras and sample 8–32 frames per take; ours keeps TimeSformer frozen, uses the best exo camera and reads the whole take as 1 s clips. ProfVLM also generates feedback text (BERTScore F1 85.5); our commentary outputs are per-second presence, praise vs tip and a sentence embedding, so the commentary metrics are not comparable.
- Sources: SkillFormer, arXiv 2505.08665 (Table 1); ProfVLM, arXiv 2509.26278 (Table 3).

### Input views and training seeds (official split)

The papers report each view separately. Same final model, one view's 768 features only (`joint.py --official --view ego|exo`, 2 seeds), and the both-view model with 5 seeds instead of 2 (`--seeds 5`):

| Model | Ego only | Exo only | Ego + Exo |
| --- | --- | --- | --- |
| Ours, level accuracy (%) \[95% CI\] | 43.9 \[31.8, 57.3\] | 45.8 \[32.5, 59.7\] | 45.8 \[31.9, 60.3\] (2 seeds); 44.5 \[30.6, 58.7\] (5 seeds) |
| Ours, rating vs held-out experts (z) | 0.673 | 0.704 | 0.720 (2 seeds); 0.708 (5 seeds) |
| SkillFormer (paper), level accuracy (%) | 45.9 | 46.3 | 47.5 |
| ProfVLM AGP (paper), level accuracy (%) | 44.2 | 45.1 | 48.2 |

- Views: exo carries more than ego for both targets (rating 0.704 vs 0.673 z), and adding ego helps rating a little more (0.720). Our exo input is one camera (the best one); the papers' Exos column uses all exo cameras. Per view, ours is within 2.5 points of both papers (at most 2.4 below, ego + exo vs ProfVLM).
- Seeds: single training seeds range 41.4–46.3% level accuracy (mean 43.4 ± 1.8) and 0.638–0.705 rating z (mean 0.682 ± 0.027); averaging 5 seeds gives 44.5% and 0.708. GPU training is not bit-reproducible: seeds 0 and 1 stopped at different epochs than in the 2-seed run. The 2-seed headline (45.8%) sits at the upper end of this spread; 44–46% is the fair summary. The seed spread alone (±2 points) is as large as our gap to the papers.
- The rating gain over the frozen probe holds with 5 seeds: +0.095 z \[+0.011, +0.186\].

## Commentary outputs on official val: what the model says, per second

The final model outputs, for every 1 s clip, the probability that an expert comments, a praise-vs-tip polarity per aspect and an embedding of what the expert would say. To show which aspects it highlights without generating text, a per-scenario logistic regression trained on the train-split comments' embeddings maps each predicted embedding to an aspect tag (it labels real val comments correctly 93–98% of the time, 67% in Rock Climbing); the polarity head gives praise or tip for that aspect. Source: 5-seed official-split run, `cls_lang/commentary_examples.py`; figure `final_report/figures/fig5_commentary_examples.png`, one typical take per scenario (median per-take presence AUC).

- When: the presence output is the strong part. Per-take presence AUC on val has a median of 0.788 over 332 takes (overall 0.723; position-only baseline 0.521).
- What: at clips that carry a real comment, the model's top aspect is right 13–49% of the time, which beats always guessing the scenario's most frequent aspect only in Music (0.281 vs 0.184). The predicted content mostly falls back on each scenario's typical aspect: clarity in Dance, hips in Rock Climbing, posture and contact in Soccer, knife and handling in Cooking, footwork and follow-through in Basketball.
- Praise vs tip: agreement with the real comment is 0.50–0.60. The polarity head is trained on polarity centred per expert, so it predicts warmer or cooler than that expert's usual tone rather than the raw share of praise.

| Scenario | Comment–clip pairs | Top-1 aspect | Top-3 aspect | Most frequent aspect | Praise / tip agreement |
| --- | --- | --- | --- | --- | --- |
| Basketball | 30,016 | 0.211 | 0.454 | 0.249 | 0.598 |
| Cooking | 8,715 | 0.133 | 0.278 | 0.219 | 0.524 |
| Dance | 38,739 | 0.485 | 0.753 | 0.485 | 0.584 |
| Music | 13,498 | 0.281 | 0.611 | 0.184 | 0.547 |
| Rock Climbing | 20,568 | 0.457 | 0.569 | 0.457 | 0.500 |
| Soccer | 16,198 | 0.367 | 0.990 | 0.552 | 0.556 |

So the commentary heads learn when experts speak up much better than what they say, which matches the text-cosine result (0.587 vs 0.571 for the mean comment in CV).

## Ablations: rating target

Rating-only models (no level loss), so every level cell is the rating score read as level (†); the frozen probe row uses its own level probe. Same 4×5 folds.

| Model | Rating, raw | Rating, z | Level accuracy † | Balanced accuracy † | Level Spearman † |
| --- | --- | --- | --- | --- | --- |
| Frozen probe, shared ridge (level: PCA probe) | 0.560 | 0.544 | 0.451 | 0.438 | 0.452 |
| Frozen probe, per-scenario ridge | 0.625 | 0.609 | 0.377 | 0.366 | 0.280 |
| CLS, shared head | 0.644 | 0.622 | 0.384 | 0.376 | 0.304 |
| CLS, shared head + all-expert commentary loss (weight 0.5) | 0.643 | 0.622 | 0.380 | 0.373 | 0.311 |
| CLS, per-scenario heads, true scenario | 0.658 | 0.640 | 0.390 | 0.382 | 0.297 |
| CLS, per-scenario heads, scenario from video | 0.656 | 0.638 | 0.386 | 0.377 | 0.299 |
| Equal-weight ensemble: per-scenario CLS + per-scenario ridge | 0.638 | 0.623 | 0.377 | 0.370 | 0.302 |
| Dance only, 8 FPS: CLS / frozen probe | 0.370 / 0.451 | 0.350 / 0.440 | 0.357 / 0.327 | 0.245 / 0.222 | 0.324 / 0.311 |
| Dance only, 30 FPS: CLS / frozen probe | 0.364 / 0.426 | 0.344 / 0.413 | 0.351 / 0.315 | 0.235 / 0.207 | 0.335 / 0.321 |

The Dance-only rows use their own 4×5 folds over the 501 Dance takes, so compare them only with each other.

- Transformer vs frozen probe: with a shared head the CLS transformer already beats the probe on rating, +0.084 \[+0.054, +0.120\] raw. This is the first clear win of a learned temporal model in the project; on level the same model never beat the probe. Ratings vary within a participant (39% of within-scenario variance vs 1% for level), so there is take-level signal to learn. Its score is a worse level readout than the level probe (accuracy 0.384 vs 0.451).
- Per-scenario heads: CLS +0.013 raw / +0.017 z (not significant); ridge probe +0.065 \[+0.013, +0.120\] raw (significant). Level read from either score stays at 0.38–0.39 accuracy.
- Scenario predicted from video instead of given: rating −0.002 \[−0.005, 0.000\]; level accuracy 0.386 vs 0.390.
- All-expert commentary as an auxiliary loss on the rating-only model: no change on either target (rating 0.643 / 0.622 vs 0.644 / 0.622; level Spearman 0.311 vs 0.304).
- 30 FPS instead of 8 FPS (Dance only; 8 consecutive frames = 0.27 s per clip, 3.75× as many tokens): CLS −0.005 \[−0.049, +0.044\] raw; probe 0.413 vs 0.440 z; level unchanged (Spearman 0.335 vs 0.324). Frame rate is not the limit for these features.
- Equal-weight ensemble of the CLS model and the per-scenario probe: −0.018 \[−0.037, +0.002\] raw vs the CLS alone; level unchanged (accuracy 0.377). It helps Dance (0.37 → 0.45) and Soccer but hurts Rock Climbing and Cooking.
- Joint (final) vs rating-only: rating +0.001 z \[−0.025, +0.030\]; level accuracy +0.075 \[+0.015, +0.139\], Spearman +0.165 \[+0.056, +0.278\]; Music rating 0.245 → 0.318.

## Ablations: commentary weight, ranking loss and encoder

The final model keeps the commentary weight at its original 0.5, so it stays comparable with the earlier runs. Raising the weight improves the commentary outputs steadily and leaves rating and level unchanged; a pairwise ranking loss and V-JEPA 2 change neither. All runs: joint model, 4×5 folds × 2 seeds, same folds.

| Variant | Rating, raw | Rating, z | Level accuracy | Balanced accuracy | Level Spearman | Presence AUC | Praise-vs-tip AUC | Text cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Commentary weight 0 | 0.655 | 0.635 | 0.462 | 0.459 | 0.465 | – | – | – |
| **Commentary weight 0.5 (final)** | **0.660** | **0.638** | **0.460** | **0.458** | **0.464** | **0.690** | **0.570** | **0.587** |
| Commentary weight 1 | 0.657 | 0.637 | 0.465 | 0.461 | 0.471 | 0.701 | 0.578 | 0.592 |
| Commentary weight 2 | 0.660 | 0.638 | 0.459 | 0.454 | 0.466 | 0.708 | 0.581 | 0.597 |
| Commentary weight 4 | 0.663 | 0.643 | 0.460 | 0.457 | 0.469 | 0.721 | 0.586 | 0.610 |
| Weight 0.5 + pairwise ranking loss (weight 1) | 0.658 | 0.636 | 0.459 | 0.458 | 0.466 | 0.680 | 0.569 | 0.580 |
| Weight 0.5, V-JEPA 2 ViT-L encoder | 0.643 | 0.623 | 0.455 | 0.452 | 0.479 | 0.707 | 0.590 | 0.596 |
| Commentary-only model (reference) | – | – | – | – | – | 0.738 | – | 0.636 |

Difference from the final model (variant − final, 95% participant bootstrap; no interval excludes zero):

| Variant | Rating, z | Level accuracy | Level Spearman |
| --- | --- | --- | --- |
| Commentary weight 0 | −0.003 \[−0.017, +0.007\] | +0.002 \[−0.005, +0.009\] | +0.002 \[−0.006, +0.010\] |
| Commentary weight 1 | −0.002 \[−0.007, +0.003\] | +0.004 \[−0.009, +0.017\] | +0.008 \[−0.002, +0.017\] |
| Commentary weight 2 | +0.000 \[−0.007, +0.007\] | −0.002 \[−0.014, +0.010\] | +0.002 \[−0.013, +0.017\] |
| Commentary weight 4 | +0.005 \[−0.006, +0.014\] | +0.000 \[−0.015, +0.014\] | +0.005 \[−0.012, +0.025\] |
| + pairwise ranking loss | −0.002 \[−0.013, +0.008\] | −0.002 \[−0.015, +0.010\] | +0.002 \[−0.009, +0.014\] |
| V-JEPA 2 encoder | −0.016 \[−0.042, +0.009\] | −0.005 \[−0.031, +0.020\] | +0.016 \[−0.017, +0.051\] |

- Rating and level: every variant is within noise of the final model on all three numbers, and none reaches the +0.02 z adoption rule.
- Commentary: weight 4 beats weight 0.5 on presence and text in 20 of 20 folds and on praise-vs-tip in 16 of 20. The position-only presence baseline is 0.509 and the mean-comment text baseline 0.571 (V-JEPA, 2 s clips: 0.519 / 0.578). The scores have not levelled off; the commentary-only model is the ceiling. If the commentary outputs matter, weight 4 gets them at no cost to rating or level.
- Why rating does not follow: the loss clearly shapes the per-clip tokens, but whatever the comments carry is either already in the frozen features or not what drives the take-level scores.
- Pairwise ranking loss: each training step adds 8 pairs of training takes rated by the same expert in the same task with different ratings (about 79k such pairs per fold; pairs sampled so each scenario counts in proportion to its rated takes), with loss softplus(−(r\_a − r\_b) · sign) on the \[RATING\] score and weight 1. Encoder frozen, everything else as the final model. No gain: the rating loss already uses per-expert z-scored targets, so two takes one expert rated differently already get different targets and the pairs mostly repeat that. Soccer level accuracy fell 0.753 → 0.662 (77 takes); the commentary outputs dropped slightly.
- V-JEPA 2 ViT-L instead of TimeSformer (2 s clips of 16 frames at 8 FPS, layer-12 token std, ego + exo, commentary weight 0.5): fails the +0.02 switch rule on rating. Better in Cooking (0.303), Music (0.347), Soccer (0.604) and Rock Climbing (0.817), worse in Dance (0.344) and Basketball (0.734); the highest level Spearman of any model (0.479), not significant. Through the transformer V-JEPA gains +0.104 z over its own probe, so attentive readout does help it, just not past TimeSformer.

## Room for improvement

Where the gap is: weighting each scenario's gap to a human expert by its rows, Dance is 46% of the remaining rating gap, Music 23% and Cooking 14%; Rock Climbing and Basketball are within 0.04–0.06 of a human expert. Level is bound by its \~280 label units. Every change tried on the training side with the frozen TimeSformer (commentary weight, ranking loss, per-scenario heads, ensembling, frame rate) and the encoder swap to V-JEPA 2 stays within 0.02 z of the final model, so the limit is what the frozen features carry about skill, not how they are read out.

Out of scope by decision: pose (including pose estimated from video) and audio. A LoRA fine-tune of TimeSformer layers 8–11 was piloted on Dance fold 0 and stopped because it changed two things at once (fine-tuning and the ranking loss); it is not part of the final model. Remaining options that keep the frozen encoder and video-only input: choose the readout per scenario by inner CV (the per-scenario ridge still beats the CLS on Dance, 0.480 vs 0.414; at best about +0.02); model each expert's own rating (≤ +0.01); more seeds (+0.005–0.01).

## Task, data and evaluation

The main target is the 1–10 expert rating; the official Ego-Exo4D proficiency level (Novice, Early Expert, Intermediate Expert, Late Expert) from the proficiency demonstrator benchmark is the second output. Both are predicted from video alone. Audio and pose are out of scope.

**Data.** All official train + val demonstrator takes with both an ego and an exo video: 2,111 takes from 227 participants.

| Scenario | Takes |
| --- | --- |
| Rock Climbing | 722 |
| Basketball | 540 |
| Dance | 501 |
| Music | 172 |
| Cooking | 99 |
| Soccer | 77 |

**Protocol.** Every number in this document uses the same protocol unless marked otherwise.

1. 5-fold StratifiedGroupKFold grouped by participant, so no participant is in both training and test. Repeated with 4 random seeds (4×5 folds).
2. All preprocessing (scenario means, standardisation, PCA, expert rankings) is fit on the training folds only.
3. **Within-scenario Spearman:** held-out predictions are centred within each (repeat, fold, scenario), averaged over repeats, then correlated with the scenario-centred level. This scores the ranking of people within a scenario, not the recognition of which scenario it is.
4. **Accuracy:** 4-class accuracy of the averaged class probabilities.
5. **Uncertainty:** paired differences with 95% intervals from 1,000 participant-level bootstrap resamples.

**Reference point.** The frozen probe: scenario-centred PCA-64 of the ego and exo clip features (averaged over the take) + scenario one-hot → RidgeCV (Spearman) and LogisticRegressionCV (accuracy). It scores 0.452 / 0.451 and is the best non-multimodal result we have found.

## Earlier level-target model: CLS + time-aligned expert commentary, hard expert selection

One transformer reads a take as a sequence of 1 s clip tokens: the CLS token is trained to predict the level, and each clip token is trained to predict what the best-agreeing expert said at that moment.

&#91;embedded content: model architecture · inference path and training-only targets\]

Only the solid path runs at inference; the expert ratings and comments shape training and are never seen at test time.

### Video input

- **Encoder:** TimeSformer-base fine-tuned on Kinetics-600, frozen. Each 1 s clip is 8 frames at 8 FPS, the whole frame squashed to 224×224.
- **Clip feature:** the standard deviation of the layer-12 patch tokens (768 dims). Token std beat the mean and the CLS token in every comparison.
- **Views:** the ego clip feature concatenated with the best exo camera's (1,536 dims per clip), truncated to the shorter view.
- **Normalisation:** each dimension is centred on its scenario mean and divided by a global standard deviation, both from the training folds.

### Commentary targets

Each expert file in `expert_commentary` holds that expert's rating (1–10), a written justification (`why`) and timestamped spoken comments with transcripts. On the 2,111 official takes this gives 57,059 comments on 1,509 takes, split into 188,256 sentences.

- **Text:** each sentence is embedded with all-MiniLM-L6-v2 (384 dims, unit norm).
- **Polarity:** a logistic regression on the embedding + scenario gives P(praise) vs P(tip). It is trained on the official good-execution / tip-for-improvement lists (40,279 sentences; CV accuracy 0.829) and checked on 11,718 raw comments that match a list entry (AUC 0.926). Target value = 2·P(praise) − 1.
- **Aspect:** what the sentence is about, from a per-scenario vocabulary (11–19 aspects). A per-scenario logistic regression is trained on Qwen3.5-4B tags of the list sentences (CV accuracy 0.58–0.65). Rock Climbing has no lists, so its 23,067 sentences are tagged by Qwen directly. The last aspect, "other", marks neutral sentences that get no polarity target.
- **Timing:** a comment made at pause time T labels clips floor(T)−2 … floor(T).

### Hard expert selection

Experts disagree with the official level to very different degrees, and every expert rates exactly one scenario. Inside each training fold:

1. For each expert with at least 10 rated training takes, compute the Spearman correlation between their ratings and the official level.
2. For each take, keep only the comments of the highest-ranked expert with a positive correlation who commented on that take.
3. Soccer gets no commentary targets, because all its experts correlate negatively with level (−0.21 to −0.36).

About 1,100–1,200 training takes per fold carry targets. Selection uses only training-fold levels, so no held-out label leaks in.

### Architecture

- Clip tokens: dropout 0.1 → Linear(1,536 → 128) + sinusoidal encoding of the clip's time in seconds.
- A learned \[CLS\] token plus a scenario embedding is prepended.
- 2-layer pre-norm transformer encoder, 4 heads, feed-forward 512, dropout 0.2, final LayerNorm.
- **CLS heads:** a level score (+ per-scenario bias) and 4 class logits (+ per-scenario bias).
- **Clip-token heads:** a presence logit, 19 polarity outputs (tanh, one per aspect slot) and a 384-dim text vector.

### Loss

```latex
L = \mathrm{MSE}(\hat{s}, y) + \mathrm{CE}(\hat{p}, y) + \lambda\left[\mathrm{BCE}_{\text{presence}} + \sum_{a} w_a (\hat{v}_a - v_a)^2 + w_{\text{txt}}\,(1 - \cos(\hat{e}, e))\right], \quad \lambda = 0.5
```

The polarity term for aspect a at a clip is weighted by how strongly the clip's comments are about aspect a. The text term is weighted by whether the clip is covered by a comment. Commentary terms apply only to takes with targets.

### Training and inference

- AdamW, learning rate 3e-4, weight decay 0.05, batch 16, random crops of up to 384 clips, gradient clipping at 1.0.
- Early stopping on a participant-grouped inner split (1/7 of the training takes) by within-scenario Spearman, patience 8, at most 60 epochs. The best epoch has a median of 3.
- 2 seeds per fold, predictions averaged.
- At inference the model sees only the video clip tokens and the scenario.

## Earlier level-target results

Hard expert selection has the highest point estimates on both metrics, but every model sits within noise of the frozen probe and of each other.

| Model | Spearman | Accuracy |
| --- | --- | --- |
| Frozen probe (PCA + ridge / logistic) | 0.452 | 0.451 |
| CLS model, level loss only | 0.450 | 0.465 |
| CLS model + expert-weighted commentary | 0.451 | 0.471 |
| **CLS model + hard-selected commentary** | **0.458** | **0.472** |

The weighted arm uses the same targets but mixes every expert, weighted by max(0, ρ) where ρ is their in-fold correlation with level, instead of keeping only the best one.

**Paired differences** (95% participant bootstrap):

| Comparison | Spearman | Accuracy |
| --- | --- | --- |
| Hard − frozen probe | +0.006 \[−0.039, +0.046\] | +0.020 \[−0.011, +0.049\] |
| Weighted − frozen probe | −0.001 \[−0.045, +0.039\] | +0.020 \[−0.010, +0.049\] |
| Level-only − frozen probe | −0.002 \[−0.047, +0.040\] | +0.014 \[−0.019, +0.044\] |
| Hard − level-only | +0.008 \[−0.007, +0.022\] | +0.007 \[−0.005, +0.021\] |
| Weighted − level-only | +0.001 \[−0.011, +0.014\] | +0.006 \[−0.007, +0.021\] |
| Weighted − hard | −0.007 \[−0.017, +0.003\] | −0.000 \[−0.009, +0.008\] |

The small accuracy gain over the probe (+0.014 to +0.020) comes from the transformer itself, since the level-only arm shows it too; it is not significant.

**Per-scenario Spearman:**

| Scenario | Takes | Frozen probe | Level-only | Weighted | Hard |
| --- | --- | --- | --- | --- | --- |
| Basketball | 540 | 0.625 | 0.636 | 0.627 | 0.628 |
| Rock Climbing | 722 | 0.449 | 0.435 | 0.437 | 0.451 |
| Dance | 501 | 0.374 | 0.337 | 0.346 | 0.339 |
| Soccer | 77 | 0.277 | 0.160 | 0.178 | 0.190 |
| Cooking | 99 | 0.015 | 0.151 | 0.155 | 0.164 |
| Music | 172 | −0.017 | −0.002 | −0.013 | −0.009 |

The transformer helps Cooking and hurts Soccer and Dance relative to the probe; both of the swinging scenarios are small. Music is near zero for every model, consistent with its experts judging tempo and tone, which the video cannot carry.

## Commentary outputs and the inter-expert ceiling

The clip-token heads predict when an expert comments better than one expert predicts another, and predict within-take praise vs tip as well as experts agree with each other.

These numbers come from commentary-only training of the same network: hard-expert targets, no level loss, 1 repeat × 5 folds × 2 seeds. They were not measured for the joint model in Main results, which stops earlier (median epoch 3 vs 8) and so likely scores slightly lower. All are on held-out takes, with targets built from the training fold's expert ranking.

| Output | Model | Baseline | Experts vs each other |
| --- | --- | --- | --- |
| Presence: does an expert comment on this clip? (AUC) | 0.736 | 0.509 (relative position in the take) | 0.677 |
| Polarity: praise vs tip, all covered clips (AUC) | 0.654 | 0.647 (every clip gets its take's mean prediction) | — |
| Polarity within a take (AUC) | 0.565 | 0.500 (chance) | 0.565 |
| Text: cosine to the comment embedding | 0.611 | 0.561 (always the average comment) | — |

**Experts vs each other** is leave-one-expert-out on takes with two or more commenting experts. Presence: the number of other experts commenting at a clip, scored against the held-out expert's comments. Polarity: the other experts' mean polarity at clips both covered, scored within each take. The comparison is not exact: each expert is scored against one to three colleagues, while the model learns from about 1,200 training takes.

Three readings:

- **Presence is real moment-level signal.** It is far above the position baseline and above inter-expert agreement.
- **Polarity is mostly person-level.** Overall polarity (0.654) barely beats giving every clip its take's average (0.647). What the model knows is that this person draws praise, which is the level signal again. Within a take it reaches 0.565, the same as experts agree with each other.
- **Text is generic.** The predicted vector is only modestly closer to the true comment than the average comment is. The model outputs an embedding, not words; turning it into text would need retrieval of the nearest real comments or a generator, neither of which we built.

Two target changes were also tested; see the ablations below. Higher AUCs against a single expert's labels would mostly mean fitting that expert's quirks.

## Ablations: how the language is used

No way of injecting the commentary moved level beyond noise; the ways that compress it hardest did worst.

### Expert selection (joint training, 4×5 folds)

Covered in Main results: level-only 0.450, weighted 0.451, hard 0.458. Hard beats weighted by +0.007 \[−0.017, +0.003\] on Spearman. That fits the per-expert agreement with level, which varies widely within a scenario:

| Scenario | Spearman(rating, level) per expert | Best expert's coverage of takes |
| --- | --- | --- |
| Basketball | 0.51 to 0.73 | 78% |
| Cooking | 0.28 to 0.72 | 96% |
| Rock Climbing | 0.31 to 0.52 | about 50% |
| Dance | 0.19 to 0.44 | — |
| Music | 0.18 to 0.37 | — |
| Soccer | −0.36 to −0.21 | excluded |

An unweighted arm (all experts equal) was not run: earlier linear tests of unweighted text supervision had already failed.

### Commentary-only pretraining, then a frozen level probe (4×5 folds)

Why: in joint training the level loss stops training by epoch 3, before the commentary heads learn much. Here the network is trained on commentary alone (best epoch about 8), frozen, and level is read out with ridge / logistic. The CLS token gets no gradient without a level loss, so the readout uses the mean and std of the clip-token outputs ("tokens") or a 40-number summary of the head predictions ("heads").

| Readout | Spearman | vs frozen probe | Accuracy |
| --- | --- | --- | --- |
| Tokens | 0.426 | −0.026 \[−0.068, +0.018\] | 0.423 |
| Tokens, with an aspect head added | 0.442 | −0.010 \[−0.052, +0.032\] | 0.424 |
| Head summary | 0.307 | −0.145 \[−0.220, −0.067\] | 0.356 |
| Frozen probe + tokens | 0.448 | −0.004 \[−0.024, +0.017\] | 0.415 |
| Frozen probe + tokens (aspect) | 0.456 | +0.004 \[−0.017, +0.023\] | 0.428 |
| Frozen probe + head summary | 0.447 | −0.005 \[−0.017, +0.008\] | 0.428 |

The network learns the commentary, but what it learns reads out as level no better than the raw features. Music improves (−0.02 → 0.13–0.21); Soccer, which has no targets, collapses.

### Expert-summary target (4×5 folds)

A cheap stand-in for asking an LLM to summarise everything the experts said, plus the level, into one vector. Five equal-variance blocks were combined: level one-hot, expert judgement (weighted z-rating and share of praise), and the weighted mean MiniLM embedding of praise sentences, tip sentences and `why` texts. These were reduced with PCA to 16 components (66% of variance). The CLS token regresses this vector; level is read out from the prediction, cross-fitted over 5 inner folds.

| Readout | Spearman | vs frozen probe | Accuracy |
| --- | --- | --- | --- |
| Predicted summary | 0.377 | −0.075 \[−0.150, +0.003\] | 0.399 |
| Frozen probe + predicted summary | 0.441 | −0.011 \[−0.026, +0.003\] | 0.446 |

Only the first component is predictable from video (held-out R² 0.34); components 2–6 are at zero. Later components reach R² 0.2–0.3 but likely describe which drill, dish or route it is, not skill. An LLM-generated summary was not run (8–16 GPU hours) because this proxy was clearly negative.

### Commentary target variants (commentary-only, 1×5 folds)

| Variant | Presence AUC (all-expert labels) | Presence AUC (hard labels) | Polarity AUC | Within-take polarity | Text cosine (baseline) |
| --- | --- | --- | --- | --- | --- |
| Hard expert for every target | 0.715 | 0.736 | 0.654 | 0.565 | 0.611 (0.561) |
| Presence + text from all experts, polarity from hard expert | 0.738 | 0.731 | 0.666 | 0.569 | 0.636 (0.552) |

Taking presence from all experts adds +0.023 on all-expert labels and lets training run longer (median epoch 12). Five timing windows, from floor(T)−1 … floor(T) to floor(T)−4 … floor(T) and up to one clip after the pause, all land within 0.01 of each other. Neither change was carried into the joint model.

## Ablations: encoder, representation, adaptation and decoder

Frozen TimeSformer-K600 with token-std pooling, ego + exo, read out by a linear model, is the best video representation we found; every attempt to adapt it or replace the readout did no better.

The earliest studies used the 1,338 takes with expert commentary rather than all 2,111, so their numbers are not directly comparable with the 0.452 reference; each row is compared with its own baseline.

| Change | Result (Spearman) | Against | Takes |
| --- | --- | --- | --- |
| Zero-shot Qwen3.5-4B ego features | about 0.35 | duration-only 0.24 | 1,338 |
| Pooling: patch-token std instead of mean (Qwen) | +0.05 to +0.09 | mean pooling | 1,338 |
| TimeSformer-K600 instead of Qwen (exo) | 0.464 vs 0.350, +0.11 (CI excludes 0) | Qwen exo | 1,338 |
| Ego + exo concatenated (TimeSformer) | 0.492, +0.016 \[−0.027, +0.052\] | best exo-only | 1,338 |
| Pool only the top-25% predicted comment windows (exo) | +0.012 (CI excludes 0); hurts ego (−0.055); no gain on official val | uniform pooling | 1,338 |
| V-JEPA 2 ViT-L instead of TimeSformer | −0.091 (CI excludes 0) | TimeSformer exo | 1,338 |
| TimeSformer-SSv2, a motion-centric encoder | 0.405, −0.047 (CI excludes 0) | K600 0.452 | 2,111 |
| Person crops from a learned detector | −0.044 (crop only), −0.013 (whole frame + crop), n.s. | whole frame | 2,111 |
| Features from all exo cameras | −0.015, n.s. | best exo camera | 2,111 |
| Motion statistics of the 1 s feature trajectory | −0.004 to −0.021, n.s. | 0.452 | 2,111 |
| Self-supervised LoRA adaptation (ego↔exo sync, 3,818 unlabelled takes) | exo degrades 0.422 → 0.365; mixed probe 0.444–0.449 vs 0.445 | frozen | 2,111 |
| Supervised LoRA fine-tuning on level (top 6 blocks) | 0.395 vs 0.423, −0.028 n.s. | frozen PCA probe, same folds | 2,111 (1×5) |
| MLP decoder instead of ridge | 0.449, −0.003 \[−0.024, +0.016\] | ridge 0.452 | 2,111 |
| MLP on all 12 feature blocks (2 layers × 3 statistics × 2 views) | 0.459, +0.007 \[−0.039, +0.052\] | ridge 0.452 | 2,111 |
| SmolLM2-135M + LoRA over clip tokens | 0.295 vs 0.422 | linear probe | 1,338 (2×5) |
| CLS transformer, level loss only | 0.450, −0.002 \[−0.047, +0.040\] | ridge 0.452 | 2,111 |

The pattern: more capacity (MLP, transformer, LLM decoder, fine-tuning) never helps and the largest models do worst, which is what a label-limited problem looks like. Representation choices that matter are the encoder (K600 over Qwen, V-JEPA 2 and SSv2), the pooling statistic (std over mean) and using both views.

## Ablations: other supervision and modality oracles

Even with oracle access to the true expert ratings, the gain on level is capped at +0.035; pose helps expert ratings but hurts level.

| Experiment | Result on level (Spearman) | Takes |
| --- | --- | --- |
| Text supervision on Qwen features: why-CCA, LLM-extracted concepts, label PLS | All equal; none beats label PLS | 1,338 |
| PLS on rating + `why` text vs ridge (TimeSformer ego + exo) | 0.531 vs 0.499, +0.033 (CI spans 0) | 1,338 |
| Video vs the true mean expert rating as a predictor | video 0.499, true rating 0.464 | 1,338 |
| Video + the true mean expert rating (oracle) | +0.035 \[+0.002, +0.071\], almost all from Cooking | 1,338 |
| Video + Ego-Exo4D automatic 3D body pose (oracle) | −0.088 (CI excludes 0); pose alone 0.14 | 748 |

The same pose features raise agreement with expert ratings by +0.089 \[+0.026, +0.159\], mostly in Dance and Basketball. So experts and the official level reward different things, and pose follows the experts. Pose and audio were set out of scope after this oracle.

The rating oracle bounds every "learn from the experts, predict level" route, including the commentary model: if perfect knowledge of the experts' scores adds at most +0.035, a model that imperfectly predicts their comments cannot be expected to add more.

## Why level is hard: the label ceiling

The video models already match a panel of human experts on this label, and the label carries only about 300 independent data points.

- **Level belongs to the person, not the take.** It is constant within participant × scenario in 258 of 272 groups with several takes (median 5 takes, up to 62). The 2,111 takes reduce to roughly 300 independent units, which explains why every model overfits within a few epochs.
- **Experts and level disagree.** On the 1,338 commentary takes, one expert's rating agrees with level at 0.41 within scenario, and the mean of all experts at 0.48. Our video probe on the same takes reaches 0.48–0.50. Experts agree with each other far better (0.70 raw, 0.77 after z-scoring each expert).
- **Ratings vary within a person; level does not.** Within a scenario, 39% of the variance in expert ratings is between takes of the same person, against 1% for level. The video, like the experts, sees take-to-take variation that the level label ignores.

The observable target is therefore the expert rating, not the level: trained on mean rating, the same frozen probe reaches 0.546 against held-out experts (humans 0.70). This is why the project made the expert rating its main target (Nov 22), with level kept as the second output.

## Limitations

**Current joint model:**

- **Language is a training signal only.** The final model learns from video, expert ratings, official levels and expert commentary, but its input is RGB video alone. The commentary loss adds per-second commentary outputs and does not change rating or level.
- **Two checkpoints.** The rating and level outputs come from the same network at different epochs (median 7 vs 3); the commentary outputs are read at the rating checkpoint, which costs them a little against a commentary-only model (presence AUC 0.690 vs 0.738).
- **The level result is still a tie with the frozen probe.** The rating result is the real gain.
- **Weak scenarios:** Dance (0.414 vs human 0.698), Music (0.318 vs 0.727) and Cooking (0.199 vs 0.495) remain far from human; Soccer (54 rated takes) and Cooking (99) swing by ±0.1 between models.
- **Rating-model CLS level probe is not cross-fitted:** training-take CLS vectors are in-sample for that rating network.
- Cross-target cells (level read from a rating-only model's score, rating from a level-only model's score) score an output the model was not trained for. They show what each specialist gives up, not the best its features could do on the other target.
- **One official-split run.** The official-split numbers come from a single train/val split (2 seeds) and have ±14-point level intervals; every other number uses 4×5 participant-grouped CV over train + val.

**Earlier level-target model.** The headline gain is not significant, and the multimodality lives in training and output rather than in the input.

- **No significant level gain.** Hard selection beats the frozen probe by +0.006 Spearman and +0.020 accuracy, both with intervals spanning zero. It should be presented as equal to the probe, with added explanatory outputs.
- **Commentary metrics come from a sibling model.** The presence, polarity and text numbers were measured with commentary-only training, not with the joint model whose level numbers are reported.
- **Text output is an embedding.** The model does not produce words, and the predicted embedding is only modestly better than the average comment (0.611 vs 0.561).
- **Polarity is person-level.** Within a take, praise vs tip is only at inter-expert agreement (0.565), so the model cannot reliably say which moment was good.
- **No official-split result for this model.** Everything here uses 4×5 participant-grouped CV on all 2,111 takes. On the official train/val split, earlier probes reached Spearman 0.37 \[0.11, 0.60\] and accuracy 0.44–0.46. Val accuracy there is dominated by a label shift: Rock Climbing is 33% Novice in train and 0% in val, and a duration-only model reaches 0.45. SkillFormer reports 46.3 (exo) / 47.5 (ego + exo) accuracy on its test half.
- **Expert selection uses levels.** Choosing experts by their agreement with level is done inside the training folds, but it does make the language targets partly level-derived.
- **Small scenarios swing.** Cooking (99 takes) and Soccer (77) move by ±0.1–0.15 between models; their per-scenario numbers are not reliable.

## Reproducibility

All commands run from `idea_1/`, one GPU job at a time (on the 12 GB RTX 4070 SUPER, two concurrent jobs spill into system memory and stall).

Rating target and joint model (from `idea_1/`):

| Step | Command | Output | Time |
| --- | --- | --- | --- |
| Rating-only CLS, per-scenario heads, video-predicted scenario | `python cls_lang/rating.py --arm rating --head per_scenario --pred_scenario --tag rating_ps_video` | `cls_lang/rating_rating_ps_video.pt` | about 10 min |
| + commentary auxiliary loss | `python cls_lang/rating.py --arm rating_comm` | `cls_lang/rating_rating_comm.pt` | about 16 min |
| Level probe on the rating CLS | `python cls_lang/rating.py --arm rating --head per_scenario --pred_scenario --level_probe --tag lvlprobe` | `cls_lang/rating_rating_lvlprobe.pt` | about 12 min |
| Joint rating + level model | `python cls_lang/joint.py` | `cls_lang/rating_joint.pt` | about 20 min |
| Rating report | `python cls_lang/rating.py --report --report_tags rating_ps_video joint` | `cls_lang/joint_rating_report.log` | under 1 min |
| Level report | `python cls_lang/level_from_rating_cls.py` | `cls_lang/joint_level_report.log` | under 1 min |
| Dance 30 FPS features | `python dense/extract_dense.py` | `embeddings/cls_lang_clipfeats_dense30_Dance.pt` | about 55 min |
| Final model: joint + commentary, weight 0.5 | `python cls_lang/joint.py --lam 0.5 --tag joint_comm` | `cls_lang/rating_joint_comm.pt` | about 27 min (weight sweep: --lam 1 / 2 / 4 with --tag joint\_lam1 / joint\_lam2 / joint\_lam4, same time) |
| V-JEPA 2 features for the takes missing from the main caches | `python vjepa/extract_extra.py exo; python vjepa/extract_extra.py ego; python vjepa/extract_extra.py build` | `embeddings/cls_lang_clipfeats_vjepa.pt` | about 33 min |
| V-JEPA 2 joint + commentary | `python cls_lang/joint.py --lam 0.5 --feat_cache embeddings/cls_lang_clipfeats_vjepa.pt --clip_sec 2 --tag joint_comm_vjepa` | `cls_lang/rating_joint_comm_vjepa.pt` | about 24 min |
| Rating and level reports for the sweep | `python cls_lang/rating.py --report --report_tags joint joint_comm joint_lam1 joint_lam2 joint_lam4; python cls_lang/level_from_rating_cls.py` | `cls_lang/lamsweep_{rating,level}_report.log` | about 2 min |
| Final model + pairwise ranking loss | `python cls_lang/joint.py --lam 0.5 --rank 1 --tag joint_comm_rank` | `cls_lang/rating_joint_comm_rank.pt` | about 45 min |
| Every model on both targets (all tables in Results and Ablations: rating target) | `python cls_lang/all_metrics.py` | `cls_lang/all_metrics_report.log` | a few minutes (CPU) |
| Final model on the official split | `python cls_lang/joint.py --lam 0.5 --official --tag joint_comm_official` | `cls_lang/rating_joint_comm_official.pt` | about 3 min |
| Official-split report (level, rating, SkillFormer test half) | `python cls_lang/official_report.py` | `cls_lang/official_report.log` | under 1 min |
| Official split: ego only / exo only / 5 seeds | `python cls_lang/joint.py --lam 0.5 --official --view ego --tag joint_comm_official_ego (--view exo …_exo; --seeds 5 …_s5); python cls_lang/official_report.py <tag>` | `cls_lang/rating_joint_comm_official_{ego,exo,s5}.pt; final_report/tables/official_split_{ego,exo,s5}*.csv` | 1.2 / 1.6 / 3.9 min |
| Commentary examples and aspect check | `python cls_lang/commentary_examples.py` | `final_report/tables/commentary_*.csv; final_report/figures/fig5_commentary_examples.png` | about 2 min (CPU) |

Report tags are the file suffix after `rating_`. The level-target runs:

| Step | Command | Output | Time |
| --- | --- | --- | --- |
| Commentary targets | `python cls_lang/build_targets.py` | `embeddings/cls_lang_sentences.pt` | about 40 min (Qwen tags Rock Climbing) |
| Main model, three arms | `python cls_lang/train.py --arm level\|weighted\|hard` | `cls_lang/results_{arm}.pt` | about 20 min per arm |
| Main report | `python cls_lang/report.py` | `cls_lang/report.log` | under 1 min |
| Commentary-only pretraining | `python cls_lang/pretrain.py [--aspect]` | `cls_lang/pretrain_hard[_aspect].pt` | about 18 min |
| Pretraining report | `python cls_lang/pretrain_report.py` | `cls_lang/pretrain_report.log` | under 1 min |
| Expert-summary target | `python cls_lang/summary_target.py` | `cls_lang/summary_lw1.pt` | about 20 min |
| Summary report | `python cls_lang/summary_report.py` | `cls_lang/summary_report.log` | under 1 min |
| Commentary target variants | `python cls_lang/comm_auc.py --pres_src hard\|all --window=-2,0` | `cls_lang/comm_auc_*.pt` | about 5 min each |

Inputs:

- `embeddings/timesformer_k600_{ego,exo}8fps_clips[_extra].pt`: frozen TimeSformer clip features, assembled into `embeddings/cls_lang_clipfeats.pt` on first run.
- `embeddings/qwen35_goodtip_aspects.pt`: Qwen aspect tags of the official good/tip lists.
- `cls_lang/vocab.py`: the aspect vocabulary per scenario.
- The inter-expert ceiling was computed by a one-off script (`expert_ceiling.py`) kept outside the repository.
