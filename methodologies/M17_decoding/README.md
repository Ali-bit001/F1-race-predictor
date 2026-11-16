# M17 — Metric-matched decoding

**Status:** done, revised after tester and referee review (dev 2021-23 only; test blocks and fresh holdout not scored)
**Plan reference:** PLAN3.md §4 (M17), §5 (attribution protocol), §6 (R2)
**Factors:** F1 dynamic = n/a · F2 decomposition = n/a · F3 ranking = decoding of the plan-2 S3 simulator output

## Hypothesis
Turning the simulator's position-probability matrix into one order with the assignment that maximises the summed
probability of exact places (D2) raises the exact-position rate over sorting by score (D0). Open question: does it
beat starting-grid order (16.3% on dev)? Failure: D2's paired gain over grid order has a 95% CI including 0 or is negative.

## Method
Plan-2 simulator S3 re-run on dev with the frozen parameters (config hash `0d1b7c074d80`, 1000 draws, same random
streams as the plan-2 run). The re-simulated matrices reproduce the stored M08 S3 dev scores exactly (max difference 0).
Matrices are stored in `results/<D0 run>/position_matrices.npz`. Nothing is fitted.

| Decoder | Rule |
|---|---|
| D0 | sort by the stored score. In plan 2 the stored score is -E[position], so D0 and D1 are the same ordering |
| D1 | sort by E[position] computed from the matrix (identical to D0 by construction) |
| D2 | `linear_sum_assignment` maximising the sum of P(driver i finishes in place j). Remaining ties: closeness of the place to E[position], then a 1e-11 term keyed on the rank of the row's own values. Row shuffles do not change any assignment |
| grid | starting grid order (grid 0 = pit-lane start goes to the back) |
| D0u | extra diagnostic, not in the plan: sort by the simulator's latent input score u (before the simulation) |

Retirements: the matrix has no separate DNF state. Inside each draw a retiree is placed behind all finishers in random
order, so its probability mass is spread over the last places. D2 therefore also assigns every driver to a place
1..n, and a driver with high retirement probability is pushed to the back only through that spread mass.
Actual positions of non-classified drivers are their derived finishing order (ingest.py), the same convention as plan 2:
"overall" metrics use all entries, "finishers" metrics use classified entries only.
Tied D0 utilities occur in 2 of 66 races.

**Two tie conventions** (the row order of the entries is the finishing order, i.e. the outcome):
- *plan-2 convention* (runs `D0`, `D1`, `grid`): exact ties broken by input row order. Kept only so the plan-2 acceptance numbers (D0 13.57, grid 16.31) reproduce.
- *outcome-free convention* (runs `D0_free`, `grid_free`, `D2`, `D0u`): ties broken by driverId (alphabetical); D2 by row content. All attribution rows use this convention; the csv names `D0_free` and `grid_free` as references `D0` and `grid`.

Old vs new, mean-of-races exact (%): D0 13.573 -> D0_free 13.573 (2 of 66 races have tied E[pos]; driverId order happens to give the same result);
grid 16.308 -> grid_free 16.384; D2 13.828 (first version) -> 13.904 (6 assigned places changed).
Averaged over 200 random outcome-free tie keys (`results/plan3/m17_tie_break_sensitivity.csv`): D0 13.54 (range 13.50-13.57), grid 16.31 (16.23-16.38),
D2 13.82 (13.68-13.98). So the referee's "D0 about 13.50-13.53, grid 16.31" holds on average over random keys; the driverId key sits at the upper end for
D0 and grid. The tie effect (under 0.1 points) is far below the interval widths.
Not changed: the plan-2 simulator assigns its Gumbel noise by row index (frozen, untouched). Draws for a driver therefore depend on its row position, which is
finishing order; I did not test whether this biases the matrices (unverified); the matrices are reproduced exactly as in plan 2.

## Leakage notes
All inputs are the frozen plan-2 pre-race predictions (M06 score, M07 p_dnf, grid, circuit and forecast columns).
Decoding uses only the matrix. Grid order is known at the start.

## How to run
```
PYTHONPATH=src python methodologies/M17_decoding/run.py --config methodologies/M17_decoding/config.yaml          # full
PYTHONPATH=src python methodologies/M17_decoding/run.py --check                                                   # acceptance + old-vs-new tie numbers, writes nothing
```

## Results (dev, 66 races, 1,311 entries; paired gain = candidate minus reference, 95% bootstrap CI over races, 10,000 resamples, seed 0, one independent RNG stream per metric)
Acceptance (plan-2 convention): D0 exact 13.57%, grid 16.31%; grid by band 35.9 / 19.2 / 12.1 / 12.1 / 10.3% reproduced.
Convention of the table: D0, D2, D0u as in the method section (outcome-free); the grid row is `grid_free` (plan-2 convention in brackets).
"exact (finishers)" = the full-field exact hit (predicted place == actual place in the full order of all entries) restricted to
classified entries. It is NOT a re-ranking among finishers (that number is in the diagnostics below).

| Decoder | exact | exact (finishers) | places 1-3 | 4-10 | 11+ | rho | MAE |
|---|---|---|---|---|---|---|---|
| D0 = D1 | 13.6 | 14.4 | 27.8 | 14.1 | 8.9 | 0.638 | 3.42 |
| D2 | 13.9 | 14.8 | 28.3 | 12.6 | 10.5 | 0.582 | 3.63 |
| grid | 16.4 (16.3) | 17.7 (17.6) | 35.9 | 15.2 | 11.4 (11.2) | 0.592 | 3.60 |
| D0u (off-plan) | 14.3 | 15.1 | 28.8 | 14.3 | 9.8 | 0.638 | 3.40 |

Paired gains, in points for rates (all metrics: `results/<stamp>_paired_gains_dev.csv` and `results/plan3/attribution.csv`):

| Candidate | vs | exact | exact finishers | rho | MAE (positive = better) |
|---|---|---|---|---|---|
| D2 | D0 | +0.3 [-1.9, +2.5] | +0.4 [-2.2, +2.7] | -0.056 [-0.075, -0.037] | -0.22 [-0.32, -0.11] |
| D2 | grid | -2.5 [-6.1, +0.7] | -2.9 [-6.7, +0.4] | -0.010 [-0.037, +0.018] | -0.04 [-0.20, +0.13] |
| grid | D0 | +2.8 [-0.2, +6.0] | +3.3 [+0.1, +6.7] | -0.046 [-0.069, -0.025] | -0.18 [-0.32, -0.04] |

### Descriptive diagnostics (not components, no verdicts)
Files: `results/<stamp>_m17_diagnostics/` and `results/plan3/m17_*.csv`; none of them is in attribution.csv.

Expected exact rate under the model's own matrix vs realised (mean of races, %), and exact rate re-ranked among classified finishers:

| Order | expected | realised | full-field hit, classified only | re-ranked among finishers |
|---|---|---|---|---|
| D0 | 13.8 | 13.6 | 14.4 | 18.1 |
| D2 | 14.9 | 13.9 | 14.8 | 16.9 |
| grid | 11.8 | 16.4 | 17.7 | 20.4 |

(referee: grid re-ranked 20.5; mine is 20.4 with the outcome-free tie-break.)

Grid-slot calibration (`pitlane.plan3.calibration.grid_slot_calibration(matrix, grid, pos)`): mean P(finish = grid slot) under
the matrix 11.78% vs realised 16.40%. Equal-count quintiles of the predicted probability, predicted -> observed:
0.053 -> 0.053, 0.086 -> 0.126, 0.104 -> 0.141, 0.125 -> 0.191, 0.222 -> 0.309. The referee's bins
(0.08 -> 0.095, 0.12 -> 0.18, 0.17 -> 0.24, 0.23 -> 0.32, 0.36 -> 0.47) were NOT reproduced bin by bin (binning not specified;
mine are quintiles over all entries). Direction and size of the shortfall in the upper bins agree.
All-cell calibration (fixed probability bins) is good except at the extremes, predicted -> observed: 0.004 -> 0.005, 0.034 -> 0.037,
0.073 -> 0.072, 0.127 -> 0.123, 0.244 -> 0.210, 0.554 -> 0.575. The matrix is calibrated cell by cell on average and under-weights the grid slot.

Per-season exact gain, grid minus D0 (points, 22 races each): 2021 +8.5 [+2.3, +15.8]; 2022 +1.8 [-2.0, +5.7]; 2023 -1.9 [-6.4, +2.0]
(referee: +8.3 / +1.8 / -1.9). D2 minus grid: -7.5 [-16.3, -0.3] / -2.3 [-6.2, +1.3] / +2.4 [-1.5, +6.7]. D2 minus D0: +1.0 / -0.5 / +0.5, all intervals include 0.

Blend `(1-w)*P + w*(point mass at grid slot)` decoded with D2, exact %: w = 0: 13.9; 0.05: 15.35; 0.10: 15.55; 0.15: 16.32; 0.20: 16.54; 0.30: 16.38
(referee: 15.4, 15.5, 16.2, 16.5, 16.31). In-sample diagnostic, w not tuned out of sample; not a component and not in attribution.csv.

Retirement draws off (S3 re-simulated with p_dnf = 0, same random streams; diagnostic): realised exact D0 13.82, D2 14.86; mean P(grid slot)
13.51% vs realised 16.40% (11.78% with retirements). Retirements off lifts the grid-slot probability by 1.7 points, about 37% of the 4.6-point gap,
so spreading of retirement mass is part of the cause but not all of it. (Expected-hit columns of that run are not comparable with realised ones,
because the real outcomes contain retirements.)

Simulation noise: four extra simulation seeds (1-4) give D2 - D0 of -1.13, -0.75, +0.78, +1.45 points (seed 0: +0.33); sd 1.23 points
(four seeds, crude). The referee's approximate 0.8 points (approximation, not independently derived) is smaller than this four-seed estimate.
Either way the D2 - D0 gain (+0.3) is well inside the simulation noise of one 1000-draw run.

## What we learned
- Conclusion (defensible wording): D2 does not beat grid order. The simulator's matrix puts about 11.8% on the grid slot against 16.3% realised, so it
  underweights the grid slot by about 4.5 points. "Grid is better than the simulator" is not established: the paired interval includes 0
  (+2.8 [-0.2, +6.0]) and the gap comes almost entirely from 2021 (+8.5 points; 2023 is -1.9).
- D2's expected rate under its own matrix (14.9%) is already below the grid's realised rate (16.3%), so no decoder of this matrix can reach grid order unless
  the matrix itself changes. D2 realises 13.9% against its 14.9% expectation.
- D2 gain over D0 on exact rate is +0.3 points, inside the noise (and inside the simulation noise, about 0.8 to 1.2 points).
- D2 costs ordering quality: rho -0.056 and MAE +0.22 against D0, both CIs excluding 0. It trades the middle places (4-10, -1.5 points) for the back (11+, +1.6 points), none of it distinguishable from 0.
- D1 equals D0 exactly, because the plan-2 stored score is already -E[position]. The plan lists them as different decoders; on this simulator they are not.
  D0u (latent input score) is within noise of D0 and is off-plan (`declared = no`, no verdict used).
- RPS and log-loss do not change with a decoder (same matrix); those rows are zero by construction. The grid row has no probabilities of its own; its RPS and log-loss columns repeat the D0 values and mean nothing.
- Verdict on exact rate (CI excluding 0 on dev): no measurable difference for every declared decoder pair.

## Consequences for plan 3
- M17 is not useful on this matrix. The relevant fix is grid-slot attachment: the M18 dead-band, or re-fitting the simulator's noise and stickiness
  parameters on exact-slot log-loss instead of RPS.
- S4-S6 are not expected to help exact hits (they do not address the grid-slot shortfall), and the declared S6 x D2 interaction is expected to be negative.
- Any future matrix (M16/M18 and later) must pass the grid-slot calibration check (`grid_slot_calibration`: predicted mean close to the realised rate,
  quintiles close to the diagonal) before D2 is run on it.

## Open questions / next tries
D2 should be re-run on every later model (the plan says so). Exact-position rate per race has SE of about 1.1 points,
so gains under about 2 points cannot be resolved with 66 dev races.
