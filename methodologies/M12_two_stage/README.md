# M12 — Two-stage Friday forecast (predict qualifying, then feed it to the frozen H2 model)

**Status:** done (R10; dev 2021-23 only; the test blocks and the fresh holdout are not scored)
**Plan reference:** PLAN3.md §4 (M12), §5, §6 (R10); horizons H0/H1/H2 from PLAN2.md §4
**Factors:** the H2 model is `latent_111` (F1 dynamic = on, F2 decomposition = on, F3 ranking = on)

## Expected gain and failure criterion (written before the first run)
Expected gain: **PLAN3 states none for M12** (§8 has no M12 line). My own expectation: no gain over the direct H1 model. Stage 2 can only see
what stage 1 knows, and stage 1 has the same inputs as H1 (history and practice); the two-stage route adds a nonlinear route (rank of predicted
qualifying position into a model fitted on real qualifying) but no new information, and real qualifying is noisy to predict. A change within
+-0.01 rho either way is the expectation; I would not be surprised by a negative one.
Failure: the paired 95% bootstrap CI of the gain over the plan-2 H1 reference on the same dev races includes 0 or is negative. Primary
metrics rho and MAE, each with its own verdict; exact-position rate overall, on classified entries and by actual place (1-3, 4-10, 11+) is reported
too. `results/plan3/attribution.csv` has one `verdict` column: it holds the rho verdict; all verdicts are in `results/verdicts.csv`.
**Sample:** practice laps exist in reality for 2021-22 as well, but they are not in our data: FastF1 timing needs a login and OpenF1 starts in 2023 (PLAN2.md:240), so practice features exist for 2023 only.
On the development block the comparison is therefore **one season: 21 races of 2023** (the 22nd, race_id 193 = round 12 of 2023, has no practice data). It is one season, not a general result.
The 44 races of 2021-22 (H1 = H0, no practice columns) are reported separately as descriptive only. Note that the `no_practice_2021_22` subset in the csv files and the run output is
**45 races: those 44 plus race 193 of 2023**; the label is wrong, and the card gives the true 44-race numbers (computed afterwards from the stored per-race tables, same bootstrap, seed 0; not in the csv files).
The leaderboard row is the mean over all 66 dev races (two-stage minus H1 rho +0.003, `all_66` in `results/paired_gains_dev.csv`); the attribution row (`results/plan3/attribution.csv`) is the 21-race comparison (rho +0.022). They are different samples and are not the same number.

## Which H2 model is "frozen H2" (decision recorded here; no ambiguity found)
H2 in PLAN2 §4 = "H0 + qualifying position, grid, Q1 gap = plan 1's information, the reference". The frozen plan-1 model at that information set is
`latent_111` (`configs/frozen.yaml`, M03 cell 111: lam 32, half-life 50; its four weekend inputs are the plan-1 signals). It is identical to
`m06_L_r0_111` in the plan-2 leaderboard (rho 0.645967 on both). The H3 model (richer qualifying, practice, sprint, weather) is **not** used.
Its four qualifying-derived inputs and how each is filled from stage 1:

| H2 input (`LatentModel._x`) | Filled with | Source |
|---|---|---|
| `quali_norm - 0.5` | (stage-1 predicted qualifying rank - 1) / (n - 1) - 0.5 | stage 1 |
| `grid_norm - quali_norm` (penalties) | 0 | constant (penalties are not known on Friday) |
| `q1_gap * 10` | mean `q1_gap` at that qualifying rank over earlier races (expanding, races before r only) | history |
| `pit_start` | 0 | constant |

The H2 model itself is fitted on **real** qualifying of earlier races (refit before every race); only the test race's four inputs are overwritten, and
restored after the prediction.

## Stage 1 (qualifying position from the H1 information set)
Ridge regression (alpha 1, fixed, no tuning) of `quali_norm` on nine columns, refit before every race on all earlier races:
decayed means of earlier qualifying position per driver and per constructor lineage (half-lives 8 and 32 races; same shrunk-mean construction as the
plan-1 finishing-form features), the plan-1 finishing form (`drv_ew16`, `con_ew16`), and the three practice signals of H1 (`x_fp_gap`, `x_fp_long`,
`x_fp_laps`; 0 when absent, as in H1). The predicted values are converted to a rank within the race (ties by driverId). Nothing about race r's
qualifying or result is used. Practice enters only for 2023 races; before that the practice columns are 0.
Quality of stage 1 is reported as Spearman rho and MAE of predicted vs actual qualifying position (actual = rank of `quali_pos` among the race's
entries, because a few races have a qualifying position above the number of starters), against: (a) the previous race's qualifying order (the
driver's last qualifying position, 0.5 if new), (b) the H0 strength order (stored plan-2 `h0_latent_111` scores), (c) the H1 score order, and stage 1
without the practice columns.

## Reference and reproduction
The reference is the plan-2 H1 model (`h1_latent_111`: `configs/frozen_plan2.yaml` H1/111) re-run in the same script; it must reproduce the stored plan-2
dev run (difference reported in the run output and `results/reproduction_check.json`).

## Leakage notes
Stage-1 features use races before r (decayed means) and practice of weekend r (available before qualifying, the same as H1). The history columns are checked by
truncation in the run script (features recomputed on the data cut before race r with race r's qualifying blanked must be identical). The H2 model never sees
race r's real qualifying.

## How to run
```
PYTHONPATH=src python methodologies/M12_two_stage/run.py [--quick]
```

## Results
Dev 2021-23. Comparison set = the **21 dev races with practice data (all in 2023: one season)**; two-stage minus H1 (positive = two-stage better), 95% bootstrap CI over
races (10,000 resamples, seed 0), ties by driverId. Run folders `results/20261116-2047_m12_*_dev` hold all 66 dev races (the leaderboard row is the 66-race mean and is not the comparison);
`results/verdicts.csv`, `results/paired_gains_dev.csv` (subsets practice_2023, no_practice_2021_22, all_66), `results/context_practice_races.csv`.

**Reproduction.** Re-run plan-2 H1 against the stored plan-2 dev run (66 races, 1,311 entries): scores identical (max |difference| 0.0); per-race metrics max |difference| 8.9e-16
(csv round trip). Leakage check of the qualifying-history features by truncation (4 races): 0.0. A one-off scratch check scrambled the real qualifying, grid and Q1 columns of one predicted race (id 197) and found
no change in any two-stage score; this check is not in `run.py` or in the tests, so it is not reproducible from the repository. A repository test is being added in `tests/test_plan3_r10.py`; until it exists this claim is unverified.

Means on the 21 practice races: H1 rho 0.6077, MAE 3.724; two-stage rho 0.6295, MAE 3.542; for context H0 rho 0.5948, MAE 3.763; H2 with the REAL qualifying (plan-1 `latent_111`, the upper bound for stage 2) rho 0.6596, MAE 3.242.

| Two-stage minus H1, 21 races of 2023 | gain [95% CI] | verdict |
|---|---|---|
| rho | +0.0218 [-0.0101, +0.0569] | no measurable difference |
| MAE (places) | +0.182 [-0.034, +0.391] | no measurable difference |
| exact all (points) | +1.7 [-2.4, +6.1] | no measurable difference |
| exact classified (points) | +1.6 [-2.9, +6.2] | no measurable difference |
| exact places 1-3 / 4-10 / 11+ (points) | +3.2 [-6.3, +14.3] / +2.7 [-4.1, +10.9] / +0.6 [-4.8, +6.7] | all no measurable difference |
| rho on finishers | +0.0363 [-0.0023, +0.0889] | no measurable difference |
| MAE on finishers | +0.253 [+0.033, +0.500] | CI excludes 0 (secondary metric, 21 races of one season: dev only) |

Descriptive subsets (not the planned comparison): the 44 races of 2021-22 (no practice, H1 = H0): rho -0.0039 [-0.0174, +0.0095], MAE +0.044 [-0.045, +0.139] (the csv subset labelled `no_practice_2021_22` has 45 races, adding race 193 of 2023: rho -0.0052 [-0.0184, +0.0082], MAE +0.034 [-0.055, +0.127]); all 66 races: rho +0.0034 [-0.0104, +0.0187],
MAE +0.081 [-0.011, +0.175]. Context on the 21 races (descriptive): two-stage minus H0 rho +0.0346 [+0.0140, +0.0555], MAE +0.221 [+0.092, +0.351]; H1 minus H0 rho +0.0129 [-0.0207, +0.0435];
two-stage minus H2-with-real-qualifying rho -0.0302 [-0.0849, +0.0232], MAE -0.300 [-0.642, +0.028].

**How well is qualifying itself predicted** (mean per race; actual = rank of qualifying position among the race's entries; predicted = rank of the stage-1 value):

| Predictor | rho, 21 races of 2023 | MAE (places), 21 races of 2023 | rho, 44 races 2021-22 | MAE, 44 races 2021-22 |
|---|---|---|---|---|
| Stage 1 (history + practice) | 0.637 | 3.53 | 0.776 | 2.79 |
| Stage 1 without practice columns | 0.624 | 3.62 | 0.776 | 2.79 (identical to stage 1: no practice columns exist) |
| H1 score order (finishing-model score used as a qualifying forecast) | 0.604 | 3.84 | 0.728 | 3.16 |
| H0 strength order (stored plan-2 `h0_latent_111`) | 0.581 | 3.98 | 0.728 | 3.16 |
| Previous race's qualifying order | 0.477 | 4.45 | 0.651 | 3.58 |

Paired, stage 1 minus baseline, 2023 (21 races): against the previous race's order rho +0.160 [+0.073, +0.255], MAE +0.917 [+0.471, +1.385]; against the H0 order rho +0.056 [+0.030, +0.081], MAE +0.442 [+0.266, +0.623];
against the H1 score order rho +0.033 [-0.003, +0.084], MAE +0.309 [+0.051, +0.632]; against stage 1 without practice rho +0.012 [-0.007, +0.028], MAE +0.087 [-0.029, +0.201]. On all 66 races
stage 1 beats the previous order and the H0 order with intervals excluding 0 (`results/quali_prediction_paired.csv`). The 2021-22 columns are 44 races (recomputed from the stored stage-1 predictions; the csv files hold the 45-race version with race 193 added: 0.777 / 2.78 for stage 1). Predicting the qualifying order from Friday information is much harder in 2023
(rho 0.64) than in 2021-22 (0.78), which may be a different field or only the 21-race sample; not resolved here. The two eras also differ in what stage 1 sees (practice columns only in 2023).

## What we learned
- **No measurable difference from the direct pre-qualifying model.** Two-stage minus H1 on the 21 practice races: rho +0.022 [-0.010, +0.057], MAE +0.18 places [-0.03, +0.39]. Failure criterion met
  (intervals include 0) for both primary metrics. The +0.022 rests largely on one race: 12 of 21 races have a positive rho gain, the per-race gain ranges from -0.114 to +0.256 (race 185 is the largest),
  and without that race the mean gain is +0.010. This is not evidence for the two-stage route; the point estimate should not be read as favouring it. Over the 44 races of 2021-22, where the two routes
  differ only through the qualifying stand-in, the rho gain is -0.004 [-0.017, +0.010].
  The only interval excluding 0 among the nine metrics on the 21 races is MAE on finishers (secondary): reported as dev only, not a finding.
- Stage 1 is better than the simple baselines at predicting qualifying (previous order, H0 order; intervals exclude 0), and also better than the H1 score order used as a qualifying forecast: in 2023
  MAE +0.309 places [+0.051, +0.632] (rho +0.033 [-0.003, +0.084]), and over all 66 races rho +0.042 [+0.025, +0.061], MAE +0.340 [+0.216, +0.470] (`results/quali_prediction_paired.csv`). Practice adds
  about +0.012 rho [-0.007, +0.028] to stage 1 in 2023, not distinguishable from 0. A better qualifying forecast has not turned into a measurably better race forecast on 21 races.
- Stage 2 with predicted qualifying (rho 0.630) stays below stage 2 with the real qualifying (0.660) by 0.03 [-0.08, +0.02] on these races, i.e. the information gap between Friday and Saturday is
  visible in the point estimates but not resolved by 21 races.
- The expectation written beforehand (nothing, within +-0.01) was exceeded in the point estimate (+0.022), but that estimate depends on one race and the interval includes 0.
- Decision rule (PLAN3 §5): does not pass alone on dev; the fresh-holdout column is left empty. One season of data is the limit of what dev can say on practice-based questions.
