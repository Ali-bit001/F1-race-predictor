# M19 — In-race forecast (H4 after lap 1, H5 at 25% of the distance, H6 at 50%)

**Headline.** The running order is a strong exact-position baseline that the score-sorted model does not beat: the two-stage models have a lower exact rate than the running order at all six model/horizon pairs
(the interval excludes 0 for 2 of 6). They gain a little on rho and on MAE at H4 and H5; at H6 the rho gain is marginal (+0.010 to +0.012).

**Status:** revised after a referee review (v2); dev 2021-23 only; the test blocks and the fresh holdout are not scored. v2 supersedes the first run (leaderboard rows `m19_*` are kept and marked superseded; their summary tables are in `results/superseded_v1/`).
**Plan reference:** PLAN3.md §3 ("In-race state"), §4 (M19), §5, §6 (R13)
**Factors:** latent model `latent_111` (F1 dynamic on, F2 decomposition on, F3 ranking on) and `gbm_100` (F1 on, F2 off, F3 off). MAIN model: the frozen pre-race score of each plus the in-race columns in a second stage. Variant (labelled, not main): the v1 joint refit.

**Reported separately.** These numbers are in-race forecasts. They are never put in one table with the pre-race results (H0-H3) without the horizon stated,
and there are no rows in `results/plan3/attribution.csv`. A stored pre-race run appears only as labelled context.

## Expected gain and failure criterion (written before the first run)
The baseline at each horizon is the running order itself (the order of the cars at the end of lap k, retirees behind the runners).
PLAN3 states no numeric expectation for M19. Mine: the running order carries almost all of the information at H5 and H6, so the model can add only
what the running order does not say, which is the pre-race strength of the cars (a fast car stuck in traffic comes back; a slow one drops back),
pit stops still to come (stops made relative to the field), and the gap to the car ahead.
Expected paired gain over the running order (positive = model better):

| Horizon | rho | MAE (places) | exact-position rate |
|---|---|---|---|
| H4 (lap 1) | +0.02 to +0.05 | -0.3 to -0.6 | +1 to +4 points |
| H5 (25%) | +0.01 to +0.03 | -0.2 to -0.4 | 0 to +3 points |
| H6 (50%) | 0 to +0.02 | -0.1 to -0.3 | 0 to +2 points |

The gain should shrink with the horizon. Failure criterion, per horizon and per model: the paired 95% bootstrap CI (10,000 resamples of races, seed 0,
outcome-free tie-break by driverId) of the gain in rho over the running order includes 0 or is negative. MAE and exact rate (overall, by actual-place
band) are read with their own intervals; the verdict column carries rho. If the CI includes 0 at a horizon, the finding is "the running order is
already as good as the model at this horizon" and is reported as such. I also expect LightGBM to be no better than the latent model (it has few
races to learn five new columns from); a gap between them inside the noise is a null result.

## Method
**Data.** `data/interim/plan3/laps.parquet` (Jolpica, all 2014-2023 races used here) and `pits.parquet` (Jolpica). One source per race, never mixed:
OpenF1 lap 1 differs from Jolpica lap 1 by about 0.35 s, and OpenF1 is not used in dev. The 2023 lap table has no position column (both sources), so the running
order is derived from cumulative lap times for every race (`src/pitlane/features/inrace.py`, `race_state`):
- runners = entries with a lap-k record, ordered by the sum of their lap times 1..k. On 2021-22, where Jolpica also stores a position, this order equals the stored one in
  all 44 races at H4, and at H5 and H6 in all but one race: 2022 round 18 (red flag, 28 laps; 83% of the cars agree at H5, 72% at H6) - see `results/data_checks.csv`.
- **Entries that retired before lap k** (no lap-k record, including cars that never completed lap 1) are known to be out at lap k, which is legitimate in-race
  information. Decision: they are *kept* in the field and scored like every other entry (all entries have a finishing position, `pos`, 1..n, retirees behind
  the classified cars in the order they retired). In the running-order baseline they sit behind all runners, ordered by laps completed (more laps = better), ties by
  driverId. In the models they carry the flags `ir_out` and `ir_def`; the fitted coefficients decide where they go.
- A few 2014 and 2018 drivers have gaps in their lap sequence (9 drivers, none in 2021-23); they are used as they are (training only).

**Race distance L (corrected in v2).** L places the horizons, so it must be known before the race. v1 took the last lap in the lap table; the v1 text said this equals the scheduled distance except for
red-flag races. That was wrong: the lap table gives what was run, which differs from the scheduled distance after any early stop (red flag, time limit, aborted start), and is not known before the race.
v2: dev races use the scheduled distance in `data/external/scheduled_laps.csv` (66 rows, a source per row). The repository holds no scheduled lap count (Jolpica results give laps run). The rule
"same as the most recent earlier race at that circuit" was tried and rejected: it disagrees with the announced distance in 14 of the 66 dev races: 5 circuits with no earlier race (Zandvoort, Losail, Jeddah, Miami, Las Vegas) and 9 where the previous count differs
(Bahrain 2021 after the 2020 outer layout, Yas Marina 2021, Bahrain 2022, Suzuka 2022, Marina Bay 2023, Monza 2023, and Spa 2022, Monaco 2023, Suzuka 2023 whose previous race was itself cut short). So the table holds the distance in force at lights-out.
**Rule: L is the race distance in force when racing begins (lights-out). Cuts before lights-out count; cuts during the race do not** (the rule is also in the `rule` column of the csv).
For the 61 races that ran to distance it equals the lap table (the race article's infobox laps agree for 58 of them; three, 2021 R2, 2021 R22 and 2022 R4, now cite the FIA final classification, 63, 58, 63 laps).
The five races that did not run to the lap-table count: 2023 R14 Monza **51** (53 cut by two laps after an aborted start, before lights-out; FIA final classification and Wikipedia); 2022 R7 Monaco **77** (the two formation laps cut 78 to 77 before the
rolling start; F1 race report 'Lap 3 of 77' and Wikipedia; 64 run because of the time limit, which does not count); 2021 R12 Spa **44** (1 run; the value rests on the Wikipedia infobox and the FIA 2022 Spa classification, same 308.052 km;
whether the safety-car formation laps cut it before the start is **unverified**); 2022 R17 Singapore **61** (59 run, time limit during the race); 2022 R18 Suzuka **53** (28 run; F1 report 'two of 53 planned laps').
A pre-race model would use 53 for Monza and 78 for Monaco; nothing pre-race uses this table now. For these two the corrected values change no horizon lap: ceil(0.25 x 77) = ceil(0.25 x 78) = 20, ceil(0.5 x 77) = ceil(0.5 x 78) = 39, and Monza stays 13 and 26. Races before 2021 (training history of the in-race coefficients, already run) use the last lap of their lap table.
k(H4) = 1, k(H5) = ceil(0.25 L), k(H6) = ceil(0.5 L). If a race stops before the horizon lap, the last available lap <= horizon is used. This happens for **one** dev race, 2021 R12 Spa (H5: lap 11 -> 1, H6: lap 22 -> 1;
`results/capped_horizons.csv`); Monaco, Singapore, Suzuka and Monza all ran past their H6 lap. Descriptive subsets: `excl_4_shortened` (the four shortened races, 62 left) and `excl_4_shortened_and_monza` (61). The four "shortened" races are those that ran fewer laps than the table distance (Spa, Monaco, Singapore, Suzuka); Monza is the one cut before lights-out.

**In-race state at lap k (laps 1..k and stops on laps <= k only; five columns):**
`ir_pos` running position, centred and scaled to [-0.5, 0.5]; `ir_gap` gap to the car ahead in tens of seconds (capped at 60 s; 0 for the leader and for
retirees); `ir_pits` stops made minus the median of the runners; `ir_out` out at lap k; `ir_def` share of the k laps not completed (0 for runners).
**Red-flag stops (corrected in v2).** v1 used the stored `red_flag_suspect` flag, which compares each stop with the median of the whole race and so used stops after lap k (a leak). v2 ignores the stored column: a stop on lap <= k is
dropped from the count if its duration exceeds the median duration of the stops on laps <= k by more than 20 s (`configs/ingest3.yaml`, `pits.red_flag_margin_s`).

**Models.**
1. **MAIN, two stages** (`src/pitlane/models/stage2.py`). Stage 1 is the frozen pre-race model (`latent_111`, `gbm_100`, `configs/frozen.yaml`), run walk-forward over every earlier race
   (`walk_scores`: refit before each race on earlier races only), so the training seasons 2014-2020 carry out-of-sample pre-race scores too (8 s for both models; the walk-forward scores equal the stored
   plan-1 dev scores to 3.7e-12 for the latent model and exactly for gbm). Stage 2 never refits stage 1 and learns only the in-race coefficients: latent `u = c * score + beta . x`
   (Plackett-Luce, **c free** with a ridge of 1e-4, ridge `lam_x` on beta, time decay with the frozen half-life 50); LightGBM on `[score] + x` (frozen gbm_100 hyper-parameters, half-life 32, no tuning).
   Trained on earlier races with race_id >= 20 only (the first season has too little history for a score to mean anything; fixed, not tuned). `lam_x` in {0.1, 1, 10} per horizon is the only tuned quantity.
2. **Variant, joint refit** (the v1 model): the five columns enter the additive term of `latent_111` / extra columns of `gbm_100`, and the pre-race strengths are re-estimated together with the in-race coefficients. This is
   a different model from "the frozen pre-race score plus the in-race features" (PLAN3 M19), so it is labelled a variant; same tuning rule.

Tuning (dev, refit every 4 races, pick = max mean rho): 3 `lam_x` x 3 horizons for the two-stage latent plus 3 x 3 for the joint latent = **18 configurations** in total (budget 30); all values in `results/tuning_lam_x.csv`, choices in `results/reproduction_check.json`.
Switches off: the harness runs of `latent_111` and `gbm_100` reproduce the stored plan-1 dev runs (max |score difference| 0, max per-race metric difference 8.9e-16); stage-2 latent with no in-race columns and c fixed to 1 returns the pre-race score (max |difference| 0).

**Metrics** (PLAN3 §5): rho, MAE, exact-position rate overall and by actual place band (1-3, 4-10, 11+), also on classified entries; per-race table, paired bootstrap against
the running order, 10,000 resamples, seed 0, driverId tie-break (outcome-free). Retirees are scored (all entries have a position).

## Leakage notes
`race_state(laps, pits, drivers, k)` discards rows with lap > k before anything is computed. Checked by hand on Monaco 2022: multiplying lap times and setting stop durations to 500 s after lap k, and overwriting the stored flag, leave the lap-k state unchanged.
The models see in-race columns of earlier races at their own horizon (training) and of race r at lap k only; the frozen scores of race r come from a fit on races < r. The distance L of a dev race comes from the pre-race table; the lap table of race r is used only
to cap k when the race stops early (Spa 2021). Test: changing laps and stops after lap k leaves the lap-k state and forecast unchanged (written by the tester, `tests/test_plan3_m19.py`).

## How to run
```
PYTHONPATH=src python methodologies/M19_inrace/run.py [--quick]
```

## Results
Run folders `results/20261117-0225_m19v2b_h{4,5,6}_{running_order,latent_s2,gbm_s2,latent_joint,gbm_joint}_dev` (15 leaderboard rows, `results/plan3/leaderboard.csv`, method M19). The `m19v2_*` rows and folders (run 20261117-0159, same code, before the Monaco 77 / source corrections) and the 9 `m19_*` rows of the first run stay in the leaderboard; the v2b run reproduces v2 exactly (every leaderboard metric difference 0), because the corrected distances change no horizon lap. 66 dev races, outcome-free ties.
`_s2` = MAIN two-stage; `_joint` = joint-refit variant. Chosen `lam_x`, two-stage latent: H4 10.0, H5 0.1, H6 1.0; joint latent: H4 1.0, H5 10.0, H6 1.0.

Mean over the 66 races (the pre-race row is a different horizon and is context only):

| Horizon | Predictor | rho | MAE | exact | exact 1-3 | exact 4-10 | exact 11+ |
|---|---|---|---|---|---|---|---|
| H4 | running order | 0.679 | 3.10 | 0.191 | 0.379 | 0.173 | 0.148 |
| H4 | latent two-stage | 0.707 | 3.00 | 0.153 | 0.273 | 0.149 | 0.120 |
| H4 | gbm two-stage | 0.718 | 2.97 | 0.168 | 0.394 | 0.117 | 0.136 |
| H5 | running order | 0.742 | 2.66 | 0.244 | 0.475 | 0.221 | 0.191 |
| H5 | latent two-stage | 0.770 | 2.53 | 0.213 | 0.384 | 0.178 | 0.187 |
| H5 | gbm two-stage | 0.772 | 2.51 | 0.220 | 0.419 | 0.186 | 0.184 |
| H6 | running order | 0.846 | 1.97 | 0.313 | 0.576 | 0.258 | 0.273 |
| H6 | latent two-stage | 0.858 | 1.89 | 0.297 | 0.510 | 0.238 | 0.274 |
| H6 | gbm two-stage | 0.857 | 1.93 | 0.264 | 0.520 | 0.188 | 0.240 |
| pre-race (context) | latent_111 / gbm_100 | 0.646 / 0.649 | 3.36 / 3.34 | 0.140 / 0.142 | | | |

Paired gain over the running order (positive = model better; 95% bootstrap CI over races, 10,000 resamples, seed 0), MAIN two-stage:

| Horizon | Model | rho | MAE (places, positive = fewer errors) | exact rate |
|---|---|---|---|---|
| H4 | latent | +0.028 [0.011, 0.047] | +0.10 [-0.02, 0.23] | -0.038 [-0.071, -0.008] |
| H4 | gbm | +0.039 [0.024, 0.056] | +0.13 [0.02, 0.25] | -0.023 [-0.055, 0.006] |
| H5 | latent | +0.028 [0.013, 0.044] | +0.13 [0.02, 0.24] | -0.031 [-0.067, 0.003] |
| H5 | gbm | +0.030 [0.015, 0.048] | +0.16 [0.04, 0.29] | -0.024 [-0.062, 0.014] |
| H6 | latent | +0.012 [0.003, 0.020] | +0.08 [-0.01, 0.16] | -0.017 [-0.050, 0.015] |
| H6 | gbm | +0.010 [0.004, 0.018] | +0.04 [-0.03, 0.12] | -0.049 [-0.083, -0.017] |

Joint-refit variant (not the main model), rho: H4 +0.036 [0.017, 0.055] latent, +0.041 [0.020, 0.063] gbm; H5 +0.031 [0.010, 0.054], +0.029 [0.012, 0.048]; H6 +0.011 [0.002, 0.021], +0.008 [-0.001, 0.017].
Exact rate of the variant: H4 -0.031 and -0.025, H5 -0.047 and -0.034, H6 -0.043 and -0.042 (latent, gbm); the interval excludes 0 for H5 latent, H6 latent and H6 gbm.
All metrics with intervals: `results/verdicts.csv` (main model, **54 rows** = 3 horizons x 2 models x 9 metrics, all dev only), `results/verdicts_joint_variant.csv`, `results/paired_gains_dev.csv` (also by season and without the shortened races).
Of the 54 main verdict rows, 20 are "passes alone" (all 12 rho and rho-on-finishers rows, 3 of 6 MAE, 5 of 6 MAE-on-finishers), 27 are no measurable difference and 7 are "worse" (all of them exact-rate rows).
Variant minus main (rho): H4 latent +0.008 [-0.000, 0.016], H5 +0.003 [-0.005, 0.012], H6 -0.001 [-0.004, 0.003]; gbm +0.002, -0.001, -0.002 (all intervals include 0).
Gbm minus latent (two-stage, rho): +0.011 [-0.000, 0.023], +0.003 [-0.004, 0.009], -0.001 [-0.006, 0.004]: no measurable difference.

Without the four shortened races (62 races, descriptive, `excl_4_shortened`), rho gain latent / gbm: H4 +0.030 [0.012, 0.049] / +0.043 [0.027, 0.060]; H5 +0.030 [0.015, 0.047] / +0.034 [0.018, 0.053]; H6 +0.013 [0.005, 0.022] / +0.012 [0.005, 0.020].
Exact-rate gain without them, latent: -0.029 [-0.059, -0.001], -0.024 [-0.059, 0.010], -0.001 [-0.029, 0.028]; gbm: -0.017 [-0.045, 0.011], -0.012 [-0.048, 0.024], -0.034 [-0.062, -0.005] (H4, H5, H6).
Dropping them moves each main rho gain by at most +0.004 (the same with Monza also dropped); no conclusion changes.

## What we learned
- **The running order is a strong exact-rate baseline that the score-sorted model does not beat.** The exact rate of the two-stage models is below the running order at every horizon for both models; the interval excludes 0 at H4 latent and H6 gbm and includes 0 elsewhere.
  None of the 54 verdict rows has the model better on any exact-rate metric. The expected exact-rate gain (0 to +4 points) did not appear.
- **rho: small, and shrinking with the horizon.** The H4 and H5 gains (+0.03 to +0.04, intervals above 0) are driven by 2022 and 2023; in 2021 they are not distinguishable from 0 (H4 latent -0.000 [-0.022, 0.023], gbm +0.021 [-0.001, 0.043];
  H5 +0.006 and +0.005, intervals include 0). That is one block of 22 races per season, not a general result.
- **H6 rho is marginal.** +0.012 latent and +0.010 gbm, lower bounds 0.003 and 0.004; I read it as no practical difference rather than as a gain. By season, the two-stage latent excludes 0 only in 2021 (+0.015 [0.002, 0.031]); 2022 and 2023 include 0
  (2023 lower bound -0.0003); gbm excludes 0 only in 2021. **The statement "no season excludes 0 at H6" that I was asked to write does not hold for these runs** (2021 excludes 0 for both); what holds is that the H6 gain is thin and not stable across seasons.
- **MAE:** gains of 0.04 to 0.16 places; the interval excludes 0 for 3 of 6 rows (gbm at H4 and H5, latent at H5).
- **Where the exact-rate deficit sits** (`results/exact_hits_by_group.csv`, `results/exact_contribution_gain_by_group.csv`; contribution = exact hits in the group / field size). Cars already out at lap k are 2.0%, 4.5% and 7.3% of the field at H4, H5 and H6, and their places are near-free hits:
  62% to 83% of them are placed exactly by the running order and 65% to 78% by the two-stage models. The deficit is on the runners: contribution to the exact rate minus the running order's is -0.039 [-0.071, -0.009] (H4 latent), -0.024 (H4 gbm), -0.032 (H5 latent),
  -0.022 (H5 gbm), -0.010 (H6 latent) and -0.041 [-0.076, -0.008] (H6 gbm), against a retiree contribution within +-0.003 at H4 and H5. At H6 the retirees lose a little too (latent -0.007 [-0.016, 0.001], gbm -0.008 [-0.016, -0.002]),
  so "the deficit is on runners, not retirees" holds at H4 and H5 and only approximately at H6.
- **Two-stage versus joint refit:** no measurable difference in rho (above); at H6 the joint latent is worse than the two-stage on the exact rate (-0.026 [-0.048, -0.005]). The reading of "frozen score" does not change any conclusion; the literal one is the main model.
- **Latent versus LightGBM:** not distinguishable on rho and MAE. On the exact rate at H6 gbm is lower than latent (-0.032 [-0.059, -0.006], one of nine comparisons).
- **`lam_x` selection optimism.** `lam_x` is picked on the same 66 races on which the gains are reported (tuning refit every 4 races, final runs every race). Mean rho over the three values, two-stage: H4 0.7053, 0.7044, 0.7067; H5 0.7697, 0.7697, 0.7685; H6 0.8565, 0.8575, 0.8507.
  At H6 the chosen value is 0.0026 above the grid mean, about a fifth of the H6 gain (+0.012), so part of the H6 latent gain is selection. The untuned gbm gives the same size (+0.010 [0.004, 0.018]), so I do not dismiss H6 outright, but I do not call it a gain.
- Context: the pre-race models (rho 0.646, 0.649) are behind the running order already after lap 1 (-0.033 and -0.030, intervals include 0), clearly behind at H5 (-0.096, -0.093) and H6 (-0.200, -0.197) (`results/context_pre_race_vs_running_order.csv`).
- Two-stage latent coefficients at the last dev refit (`results/reproduction_check.json`): running position -0.9 (H4), -2.0 (H5), -3.5 (H6); out -0.8, -1.6, -1.7; stops relative to the field +0.1 to +0.4; gap to the car ahead -0.2 to -0.1;
  weight on the frozen score c = 0.69, 0.57, 0.48. `ir_out` and `ir_def` are the same column at H4 (k = 1), so the ridge splits their effect.
- Data: 26 entries in the dev races have no lap record at all (retired on lap 1) and are treated as out. No drivers with gaps in the dev lap tables. 2021 round 12 (1 lap run) stays in; H5 and H6 there are lap 1 (capped).
- About 54 verdict rows for the main model (108 with the variant), all dev only. Under PLAN3 §5 a secondary result counts as a finding only if it also holds on the fresh holdout; none of these has been tested there.

## Open questions / next tries
- Next step (not run): a dead-band / stickiness decoder judged on the exact rate. Keep the running order and let the model move a car only when its strength difference to the neighbour exceeds a threshold (cf. M18). Not run here: it is a new component with its own tuning,
  and tuning it on the exact rate of these 66 races would be a second look at the same data.
- Fresh-holdout use is not allowed yet; nothing here is scored beyond dev.
- Only the Spa 2021 distance (44) is still not verified against a primary document (see above).
