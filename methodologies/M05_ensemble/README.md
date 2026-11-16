# M05 — Ensemble of the latent model and LightGBM; monotone LightGBM

**Status:** done (R10; dev 2021-23 only; the test blocks and the fresh holdout are not scored)
**Plan reference:** PLAN3.md §4 (M05), §5 (attribution protocol, decision rules), §6 (R10), §8 (expected outcome)
**Factors:** F1 dynamic = on · F2 decomposition = on for `latent_111`, off for `gbm_100` · F3 ranking = on for `latent_111`, off for `gbm_100`

## Expected gain and failure criterion (written before the first run)
Expectation (PLAN3 §8): about +0.005 rho from the rank-average ensemble. No separate expectation is stated in the plan for the monotone
LightGBM; my own expectation is a change within the noise (it removes a freedom, not checked before the run).
Failure: the paired 95% bootstrap CI of the per-race gain on dev includes 0 or is negative. Primary metrics: rho and MAE; each gets its own
verdict. Exact-position rate is reported overall, on classified entries and by actual place (1-3, 4-10, 11+). `results/plan3/attribution.csv`
holds one `verdict` column: it is the rho verdict; the other verdicts are in `results/<run>/verdicts.csv` and in the table below.
Secondary comparisons: a result counts as a finding only if it also holds on the fresh holdout (PLAN3 §5).
With the ensemble's members correlated at about 0.9 or more on dev, +0.005 rho is smaller than the width of a typical 66-race paired interval
(about +-0.01), so a null result is the most likely outcome.

## Method
Frozen plan-1 hyper-parameters, no re-tuning, walk-forward on dev with a refit before every race (`refit_every=1`):
`latent_111` = `configs/frozen.yaml` M03 cell 111 (lam 32, half-life 50); `gbm_100` = M04 cell 100 (4 leaves, 200 trees, lr 0.05,
min child 40, half-life 32). Both references are re-run in the same script and compared with the stored plan-1 dev runs.

| Component | Definition |
|---|---|
| `M05_rank_avg` | Per race: the mean of the within-race ranks of the two models' scores, equal weights, nothing fitted. Ranks and the average are tie-broken by driverId (`pitlane.plan3.decode.break_ties`), never by input row order. The score handed to the harness is minus the mean rank. The leaderboard row `M05_rank_avg` carries the label "cell 111" (the latent member's cell), but it averages the cell-111 latent model and the cell-100 LightGBM; it is not the cell-111 model. |
| `M05_gbm_monotone` | `gbm_100` with LightGBM `monotone_constraints`. The target is `y` (normalised finishing position, 0 = win). Constraint **+1 (predicted y never decreases)** on `grid_norm` and `quali_norm`; **0 (free)** on the ten other columns (`q1_gap`, `pit_start`, `n`, `drv_circ`, `con_circ`, `drv_dnf_ew32`, `con_dnf_ew32`, `drv_cnt`, `drv_ew32`, `con_ew32`). Same features, same hyper-parameters, same seed. |

`q1_gap` (larger = slower in Q1) is not constrained: the plan names qualifying position and grid slot only. The constraint holds for each
feature with the others fixed (LightGBM semantics); it does not make the final order equal to the grid order.

Comparisons: `M05_rank_avg` against `latent_111` and against `gbm_100` (two attribution rows); `M05_gbm_monotone` against `gbm_100`. Exact-position
rate is also compared with grid order (descriptive paired number; outcome-free grid convention of M17: grid 0 = back, ties by driverId).

## Leakage notes
No new inputs. Both members use only races before the predicted one (walk-forward harness, `refit_every=1`). The ensemble uses their scores for the
same race and nothing else.

## How to run
```
PYTHONPATH=src python methodologies/M05_ensemble/run.py [--quick]
```

## Results
Dev 2021-23, 66 races, 1,311 entries; refit before every race; frozen plan-1 hyper-parameters, no tuning. Paired gain = candidate minus
reference (positive = candidate better), 95% bootstrap CI over races (10,000 resamples, seed 0). Ties broken by driverId, never by row order.
Run folders: `results/20261116-2045_m05_*_dev`; per-metric verdicts: `results/verdicts.csv`; all gains: `results/paired_gains_dev.csv`.

Means: `latent_111` rho 0.6460, MAE 3.355, exact 13.97%; `gbm_100` rho 0.6489, MAE 3.337, exact 14.20%; `M05_rank_avg` rho 0.6508, MAE 3.303, exact 14.89%;
`M05_gbm_monotone` rho 0.6522, MAE 3.318, exact 14.57%; grid order (outcome-free) exact 16.38%, rho 0.5916.

| Component vs reference | rho | MAE (places) | exact all (points) | exact classified (points) | rho verdict | MAE verdict | exact verdict |
|---|---|---|---|---|---|---|---|
| `M05_rank_avg` vs `latent_111` | +0.0048 [-0.0042, +0.0133] | +0.052 [-0.012, +0.117] | +0.9 [-1.5, +3.3] | +0.8 [-1.9, +3.4] | no measurable difference | no measurable difference | no measurable difference |
| `M05_rank_avg` vs `gbm_100` | +0.0019 [-0.0070, +0.0110] | +0.034 [-0.026, +0.092] | +0.7 [-1.1, +2.5] | +0.9 [-1.0, +2.8] | no measurable difference | no measurable difference | no measurable difference |
| `M05_gbm_monotone` vs `gbm_100` | +0.0033 [-0.0017, +0.0082] | +0.018 [-0.012, +0.049] | +0.4 [-0.8, +1.5] | +0.4 [-0.9, +1.8] | no measurable difference | no measurable difference | no measurable difference |

Exact rate by actual place band (points, candidate minus reference): `rank_avg` vs `latent_111` 1-3 +3.0 [-3.0, +9.1], 4-10 -0.6 [-4.1, +2.8], 11+ +1.3 [-1.6, +4.3];
`rank_avg` vs `gbm_100` 1-3 -3.0 [-8.1, +1.5], 4-10 +0.9 [-2.8, +4.3], 11+ +1.6 [-0.9, +4.2]; `gbm_monotone` vs `gbm_100` 1-3 -1.0 [-4.0, +2.0], 4-10 +0.2 [-2.2, +2.6], 11+ +0.9 [-0.6, +2.4]. All include 0.

Finishers (classified entries only, re-ranked): rho `rank_avg` vs `latent_111` +0.0114 [+0.0044, +0.0185] (CI excludes 0), vs `gbm_100` -0.0019 [-0.0099, +0.0059];
`gbm_monotone` vs `gbm_100` +0.0017 [-0.0031, +0.0068]. MAE on finishers: `rank_avg` vs `latent_111` +0.103 [+0.048, +0.157] (excludes 0), vs `gbm_100` +0.023 [-0.033, +0.079];
`gbm_monotone` vs `gbm_100` -0.001 [-0.031, +0.030]. Multiplicity: this card shows 27 intervals (3 comparisons x 9 metrics: rho, MAE, exact all, exact classified, three place bands, finishers' rho and MAE); two exclude 0
(both finishers, `rank_avg` vs `latent_111`). `results/plan3/attribution.csv` holds 36 intervals for the three M05 rows (3 x 12 metrics, adding within-1, within-2 and the log-loss of P(points finish)); three exclude 0
(finishers' rho, finishers' MAE and log-loss of P(points finish), all `rank_avg` vs `latent_111`). The 20 exact-rate-versus-grid intervals below are a separate family, one of which excludes 0.
With this many intervals, a few excluding 0 is expected by chance, and none of the three holds against `gbm_100`. The finishers' intervals say only that the latent model is somewhat worse than LightGBM on
finishers (the rank average moves it toward LightGBM). They are "dev only" in the sense of PLAN3 §5.

**Log-loss of P(points finish)** (PLAN3 §5 primary probability metric; paired gain, positive = candidate better; `results/plan3/attribution.csv`, column `gain_logloss_points`; dev only):
`rank_avg` vs `latent_111` +0.0101 [+0.0053, +0.0149] (excludes 0); `rank_avg` vs `gbm_100` +0.0021 [-0.0029, +0.0073]; `gbm_monotone` vs `gbm_100` -0.0002 [-0.0015, +0.0011].
So on this metric the ensemble beats the latent model and not LightGBM, the same pattern as the finishers' metrics; the attribution `verdict` column is the rho verdict and does not reflect this.

**Exact-position rate against grid order** (descriptive paired number, candidate minus grid; negative = below grid order; outcome-free grid convention): `latent_111`
-2.4 points [-6.1, +0.7], `gbm_100` -2.2 [-5.7, +0.9], `M05_rank_avg` -1.5 [-5.0, +1.7], `M05_gbm_monotone` -1.8 [-5.3, +1.3]. None of the models beats grid order on exact
position; all four intervals include 0. In places 1-3 `latent_111` is below grid by 10.1 points [-19.7, -0.5] (the only interval excluding 0); the others include 0
(`results/exact_vs_grid_dev.csv`).

**Reproduction.** Re-run references against the stored plan-1 dev runs: predicted scores identical (max |difference| 0.0, 1,311 entries each); per-race metrics max |difference|
8.9e-16 for `latent_111` and 4.4e-16 for `gbm_100` (this is the csv text round trip of the floats, as in M11; 66 races). `results/reproduction_check.json`.

**Monotone constraint check** (last dev race, 20 entries, sweep of the feature over 0..1 with all others fixed): plain `gbm_100` has a predicted y that decreases
somewhere along the sweep for 20/20 entries in `grid_norm` and 1/20 in `quali_norm`; `M05_gbm_monotone` 0/20 and 0/20 (`results/monotone_check.json`). So the constraint
binds on the grid slot: the unconstrained trees do predict a better finish for a worse grid slot over parts of the range (one race checked, descriptive). The count uses a decrease threshold of
-1e-12 on the predicted y; the size of the violations was not measured, so 20/20 may include tiny decreases.

**Pit-lane starts.** A review remark said a pit-lane start has `grid_norm` < 0 (`src/pitlane/data/ingest.py:115`) and so would count as better than pole under the constraint. Checked, and not so: `ingest.py` first sets
`pit_start` = 1 and replaces grid 0 by n, then computes `grid_norm` = (grid - 1)/(n - 1), so a pit-lane start has `grid_norm` = 1 (back of the grid). In `data/processed/entries_feat.parquet` the minimum of `grid_norm` is 0.0 and no row is negative
(48 pit-lane starts in dev, all with `grid_norm` = 1.0). The constraint therefore does not rank them above pole; `pit_start` is free and can still adjust them.

**Tie-break sensitivity of the rank average.** 471 of 1,311 rank-average entries sit in a tie of the mean rank (two members that disagree by the same number of places in opposite directions give equal means), broken by driverId; verified from the stored member scores
(the recomputation reproduces rho 0.6508 and exact 14.89%). Other outcome-free tie-breaks give: by latent rank rho 0.6494, exact 14.43%; by LightGBM rank 0.6514, 14.66%; by grid slot 0.6497, 14.74%; by reversed driverId 0.6500, 14.49%
(rho 0.6494 to 0.6514, exact 14.4% to 14.7%, against the reported 0.6508 and 14.9%). The reported exact-rate gain of the ensemble over the members (+0.9 and +0.7 points) is therefore partly the tie-break; it is not distinguishable from 0 either way.
(Scratch check, not in the repository.)

**Agreement of the members.** Mean per-race rank correlation between the scores of `latent_111` and `gbm_100`: 0.943 (minimum 0.883 over 66 races), the reason why a
rank average can add little.

## What we learned
- **Q6 answer on dev: the ensemble gives +0.005 rho over the latent model (CI [-0.004, +0.013]) and +0.002 over LightGBM (CI [-0.007, +0.011]): no measurable difference.**
  The expected gain of about +0.005 was reached as a point estimate against `latent_111` only, and it is inside the noise. Failure criterion met (interval includes 0) for rho and MAE
  against both references. The ensemble does not beat LightGBM on any of the primary or exact metrics.
- **Monotone LightGBM: +0.003 rho [-0.002, +0.008] over `gbm_100`, no measurable difference.** The constraint is active (20/20 entries had a non-monotone response in the grid slot without it, one race, -1e-12 threshold, size not measured) but
  forcing monotonicity does not change the order accuracy measurably.
- No variant beats grid order on exact positions (16.4% on dev). Nothing here changes the finding that exact-position rate before the race is at grid level at best.
- Secondary (finishers' rho and MAE of the ensemble over `latent_111`) have intervals that exclude 0 but are not primary and do not hold against `gbm_100`.
- Decision rule (PLAN3 §5): neither component passes alone on rho or MAE; neither enters the final model on dev evidence. Fresh-holdout column left empty.
