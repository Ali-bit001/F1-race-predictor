# M18 — Overtaking threshold (dead-band on the strength difference)

**Status:** done (R7); dev 2021-23 only. The test blocks and the fresh holdout are not scored.
**Plan reference:** PLAN3.md §4 (M18, with M17's grid-slot finding), §5, §6 (R7)
**Factors:** plan-2 simulator S3 on the `latent_111` weekend score (F1 dynamic on, F2 decomposition: the score itself is unchanged, the dead-band is a pre-processing step on it, F3 ranking on).

## Reading of the plan (written BEFORE the first run)
PLAN3 says: the pace term between a car and the car ahead on the grid counts only above a threshold `d * stick[circuit]`, one fitted `d`; if A3 fails and there is no pace offset, the same dead-band is applied to the model's strength difference. A3 failed as declared (car persistence 0.096 < 0.2), so there is no `con_rq` and M09 was not built. Choices I make where the plan is silent, each the simplest one consistent with it:

1. **Reference = frozen plan-2 S3.** Its `kappa` is 0 (frozen_plan2.yaml), so the "linear" stickiness term of the simulator is off in the reference. The dead-band is a separate, additional step, and `d = 0` is exactly S3, bitwise. All S3 parameters (tau, dnf_scale, regime) stay frozen; only `d` is fitted. A secondary check refits `tau` once at the chosen `d`, because the dead-band shrinks score differences (declared as secondary, counted in the configuration budget).
2. **`stick`** is the plan-2 circuit grid stickiness: shrunk mean Pearson correlation (not Spearman) of grid and finishing position at earlier races of the circuit, where grid 0 (pit lane) counts as front in that correlation (both inherited from plan 2; not changed here) (`features/weekend.py`, column `stick`, range 0.49-0.73 on dev, mean 0.62). It is used uncentred, so the threshold is positive. (The simulator's `stick_c` is centred; it is not used here because a negative threshold has no meaning.)
3. **Where the dead-band enters.** On the M06 score `u` the simulator already uses, in score units (before the `tau` scaling). Order the field by starting grid (pit-lane starts at the back, ties by driverId, never by row order). For each car, `diff = u(car) - u(car ahead on the grid)`. A car behind that is stronger counts only by the part above `thr = d * stick`: `max(0, diff - thr)`; a car ahead that is stronger counts in full (the dead-band is for passing, not for being passed). The adjusted score is `u - cumulative sum of min(max(diff, 0), thr)` along the grid, and it replaces `u` in the unchanged simulator. Soft dead-band (subtract), not a hard cut, so the term stays continuous in `d`.
4. **Fitting `d`:** coordinate over a fixed list, objective = the simulator's own plan-2 objective (mean RPS, 300 draws, streams 1000 + i), dev races only. Not tuned on exact rate, rho or MAE.
5. **Judging:** per-race paired gain against the reference S3 and against grid order (driverId tie-break), 10,000 resamples of races, seed 0, on rho, MAE, exact rate overall and by band (1-3, 4-10, 11+), plus RPS and points log-loss. Decoder: D0 (sort by expected position, the plan-2 convention) is the main row; D2 is reported too, after the grid-slot calibration of the matrix. The declared S6 x D2 interaction is not part of this package. Tuning and judging use the same 66 races, so the verdict is optimistic by construction.

## Expected gain and failure criterion (written before the first run)
Mechanism: M17 showed the plan-2 matrix puts 11.8% on "finishes in the grid slot" against 16.4% realised. A dead-band cuts the chance that a slightly faster car passes, so mass moves to the grid slot.
Expected: grid-slot probability rises from 11.8% to about 13-15%; exact rate +0.5 to +2 points over the reference S3 (D0), still at or below grid order (16.4%); rho within -0.01..+0.01; MAE within +-0.1; RPS better by about 0.001-0.003 at the fitted `d`. With 66 races an exact-rate gain under about 2 points cannot be resolved, so I expect a "no measurable difference" verdict on exact rate unless the effect is larger than I expect.
Failure: the exact-rate gain against the reference has a 95% CI that includes 0 (null), or the fitted `d` sits at 0 or at the end of the list (no dead-band preferred by RPS), or rho or MAE is measurably worse. The package would count as "meeting the plan's target" only if exact rate is not below grid order; I expect that not to happen.
Acceptance (R7): `d = 0` reproduces the reference matrices exactly (max abs difference < 1e-12; expected bitwise 0).

## Leakage notes
`u`, `stick`, `grid` are the plan-2 inputs: scores from walk-forward M06 (history plus the weekend's own qualifying), `stick` from earlier races of the circuit only, the grid is known before the start. The tie-break is by driverId. Nothing from the race result enters except through the evaluation.

## How to run
```
python methodologies/M18_overtaking/run.py --config methodologies/M18_overtaking/config.yaml
```

## Results (dev 2021-23, 66 races, 1,311 entries; run `20261117-0243_*`)
**Failures first.** The failure criterion is met: the fitting objective (RPS) prefers `d = 0`. RPS rises monotonically with `d` (0.12040 at 0, 0.12047 at 0.03, 0.12111 at 0.1, 0.12503 at 0.5, 0.12859 at 3.0; 16 configurations, budget 30), and refitting `tau` at `d` = 0.2 and 0.5 does not help (0.12195, 0.12362, 0.12401, 0.12691). No dead-band version beats the reference on the fitting objective. To still give a paired comparison I evaluated the smallest positive grid value with the lowest RPS, `d = 0.03` (labelled "best positive d", not a fitted optimum). The grid values 0.03 and 0.05 and the tau-at-d configurations were added after seeing the first tuning pass (0.1 was the smallest value first; RPS was already rising); this is a declared post-hoc extension, inside the 30-configuration budget.

Acceptance: `d = 0` reproduces the frozen S3 matrices bitwise (max absolute difference 0.0 over all 66 races).

Paired gains of `d = 0.03` against the reference S3 (same streams, D0 decoder, driverId tie-break), positive = better, 95% CI over races, 10,000 resamples, seed 0:

| Metric | vs S3 D0 | vs grid order |
|---|---|---|
| rho | -0.0013 [-0.0048, +0.0022] | +0.0449 [+0.0242, +0.0673] |
| MAE (places) | +0.008 [-0.016, +0.034] | +0.186 [+0.053, +0.324] |
| exact rate (points) | +0.77 [-0.28, +1.84] | -2.04 [-5.32, +0.94] |
| exact places 1-3 / 4-10 / 11+ (points) | 0.0 [-2.0, +2.0] / +1.3 [-0.9, +3.9] / +0.7 [-0.8, +2.1] | -8.1 [-16.7, +0.5] / +0.2 [-4.3, +4.8] / -1.9 [-5.5, +1.5] |
| RPS | -0.0001 [-0.0004, +0.0002] | (same matrix as reference; not meaningful for grid order) |
| log-loss, top 10 | -0.0023 [-0.0045, -0.0002] (worse) | same |

Verdicts (CI rule, dev): vs S3 rho, MAE, exact all "no measurable difference"; vs grid order rho and MAE "passes alone" (inherited from S3, which already beats grid order on rho: this is not a dead-band effect), exact "no measurable difference" (the dead-band version stays 2.0 points below grid order, 14.3% against 16.4%). Rows `M18_deadband`, `M18_deadband_D2` in `results/plan3/attribution.csv` (references `S3_D0`, `grid`, and `S3_D2` for D2).

Grid-slot calibration (mean predicted probability of finishing in the grid slot, realised 16.4%): reference 11.78%, `d = 0.03` 12.12% (the matrix is barely moved). D2 at `d = 0.03`: exact 14.5%, +0.9 [-1.3, +3.1] vs S3 D0, +0.6 [-0.8, +2.0] vs S3 D2, -1.9 [-5.6, +1.3] vs grid; rho vs S3 D0 -0.066 [-0.086, -0.047], MAE worse by 0.22 [0.12, 0.32] (the same cost of assignment decoding as in M17).

**Two different nulls (read separately).**
(a) *Fitted d = 0 is a result of the RPS objective, not of the dead-band idea.* The objective was RPS, as declared. Plan 2 had already fitted the linear grid-attachment `kappa` by RPS on the same simulator and chose 0 (configs/frozen_plan2.yaml, S2/S3), so RPS was predictably going to prefer `d = 0` again; my pre-run prediction above (RPS better by 0.001-0.003) expected otherwise and was wrong. The choice of objective cannot be verified as made before running: the M18 folder was untracked, and its config snapshot already contains the post-hoc grid values 0.03 and 0.05. Scored on the stored matrices (`revise.py`, paired over 66 races, gain = d = 0 loss minus d = 0.03 loss, positive = d = 0.03 better, 95% CI, 10,000 resamples, seed 0): grid-slot binary log-loss prefers `d = 0.03` by +0.0023 [+0.0010, +0.0038] (0.4211 at d = 0); full exact-slot log-loss (-log P of the realised place, probabilities floored at 1e-3) +0.0019 [-0.0041, +0.0078] (floor 1e-4: same). These reproduce the referee's numbers. At `d = 0.5` the grid-slot log-loss gain is +0.0101 [-0.0008, +0.0229] and the exact-slot log-loss is worse by 0.081 [0.036, 0.127] (floor 1e-4: 0.095); at `d = 1.5` +0.0090 [-0.0064, +0.0273] and worse by 0.146 [0.081, 0.212]. So a grid-slot objective would choose a small positive `d`; the plan's objective (RPS) does not, and the verdict rows below say "fitted d = 0, component inactive".
(b) *Mechanism: M18 cannot beat grid order on exact rate under any objective.* The dead-band lowers a stronger car behind to at most equal strength with the car ahead (a coin flip under Gumbel/PL noise); it never gives the car ahead an advantage. As `d` grows the model converges to grid order with ties, so D0 exact does not move (13.5%, 13.9% at d = 0.5, 1.5) and D2 can at best tie grid order. Even at the descriptive `d` that calibrates the grid slot, D2 is -0.3 points vs grid order.

**Descriptive, not fitted, no leaderboard or attribution row (post hoc): large dead-bands.** `d = 0.5` and `d = 1.5` bring the grid-slot probability to 15.4% and 16.5% (calibrated), so the mechanism works on the matrix. D0 exact stays at 13.5% and 13.9%; D2 exact is 16.1% (`d = 0.5`, +2.5 points vs S3 D0 [+0.3, +4.8], -0.3 vs grid [-3.5, +2.6]) and 15.6% (`d = 1.5`, +2.0 [-0.7, +4.6]). The cost: RPS worse by 0.0046 [0.0017, 0.0075] and 0.0079; D0 rho -0.026 [-0.042, -0.011] and -0.035; D2 rho -0.082 and -0.105; D2 MAE worse by 0.27 and 0.41. The +2.5 [+0.3, +4.8] is not a finding: `d = 0.5` was picked after seeing two values, it is 1 of 8 descriptive comparisons (2 values x 2 decoders x 2 references) with no multiplicity control, the interval does not include Monte Carlo noise (M17 estimated about 1.2 points sd on D2 minus D0 for one 1000-draw run), and the size is about that of grid order's own advantage over S3 D0. Run folders for these two runs (config snapshot, metrics.json, per_race_metrics.csv, predictions.parquet, matrices): `results/20261117-0627_m18_descriptive_d0.5_{D0,D2}_dev`, `..._d1.5_{D0,D2}_dev` (written by `revise.py`; same seed and matrices as the first pass; `metrics.json` reproduces the figures above). The paired table is `results/20261117-0243_descriptive_large_d_paired_dev.csv`; log-loss scores are in `results/20261117-0627_logloss_objectives_dev.csv`.

## What we learned
- The dead-band in the form the plan allows (strength difference, one `d`) does not improve the probability matrix on its own objective: any positive `d` lowers RPS quality, and the best positive value is a no-op within noise on rho, MAE and exact rate.
- The 11.8% vs 16.4% grid-slot gap of M17 is closable with a large `d`, but then the exact-rate gain appears only with D2 decoding (the declared S6 x D2 interaction is outside this package), and it costs rho and calibration. Decoded by sorting expected position (D0), the exact rate does not move at any `d`.
- Reading choices that could matter: soft rather than hard threshold; one-sided (only passes are dead-banded); cumulative along the grid; `stick` uncentred. Other readings were not tested. The reference has `kappa = 0`, so the plan's "linear version" (a `stick * grid` term) is absent in the frozen S3; I did not refit `kappa` (that would change a frozen plan-2 parameter).
- Tuning and judging use the same 66 races; the result is optimistic if anything, and still null.

## Open questions / next tries
- A "grid-slot attachment" candidate for the fresh holdout: `d` chosen on dev by grid-slot binary log-loss, decoded with D2, primary comparison against grid order on exact rate. This is an M18 x D2 interaction, which PLAN3 section 5 does not declare (it declares only M09 x M18, S6 x D2, M11 finishers-only x M07). It needs the user's declaration before any fresh race is scored. Not run. By (b) above I expect it at best to tie grid order, so it is a weak candidate.
- Refit `tau` and noise jointly for a large `d` on exact-slot log-loss (suggested in the R2 log entry); not done, it changes the objective.
