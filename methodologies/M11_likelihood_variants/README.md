# M11 — Likelihood variants of the latent model (cell 111)

**Status:** done, revised after the referee's review (R5; dev 2021-23 only; test blocks and fresh holdout not scored). Every result below is "dev only".
**Plan reference:** PLAN3.md §4 (M11), §5 (attribution protocol), §6 (R5)
**Factors:** F1 dynamic = on · F2 decomposition = on · F3 ranking = on, with three changes to the Plackett-Luce likelihood, each alone

## Summary (read this first)
- Against the re-run reference `latent_111` (lam 32, half-life 50), **reverse passes alone on MAE and on exact-position rate, not on rho**; finishers-only and top-10 pass on none of the three primary metrics. The first version of this card recorded the verdict on rho only; PLAN3 §5 names rho and MAE as primary and the exact rate as the Amendment-2 headline, so the verdict is now given per metric.
- The reverse gain on places 1-3 (+18.2 points) **repairs a defect of the reference**, it does not beat the benchmark: the reference is worse than grid order on places 1-3 (-10.1 [-19.7, -0.5]); reverse minus grid on places 1-3 is +8.1 [-2.0, +18.2]. **No variant beats grid order on overall exact rate** (reverse +0.6 [-3.3, +4.1] points; the upper bound is 4.06). The Amendment-2 target is not met.
- Of 36 paired intervals, 10 exclude 0; under Holm (alpha 0.05, 36 tests) 4 survive: reverse places 1-3, reverse log-loss of P(points), reverse MAE on finishers, finishers-only MAE on finishers. Reverse's overall exact rate and overall MAE do not survive.
- Tuning each variant on the plan-1 grid (12 points) does not change the picture: no variant has a rho interval excluding 0 (reverse tuned: +0.0183 [-0.0004, +0.0362]).
- The declared interaction finishers-only x M07 was built (it can be built from frozen plan-2 pieces) and shows no measurable RPS or rho gain as defined; see below for the scale problem.

## Expected gain and failure criterion (stated from the plan; fixed before the numbers were looked at)
Expectation from PLAN3: small or no change, |delta rho| under about 0.01 for each variant (retirees are about 12% of dev entries). Failure: the paired 95% bootstrap CI of the per-race gain on dev includes 0 or is negative. Rule (PLAN3 §5): a variant "adds value" if the 95% paired bootstrap CI over races excludes 0. Primary metrics for order: rho and MAE; headline for exact positions: exact rate with grid order as the baseline (Amendment 2). The rule is applied separately to each of rho, MAE and exact rate below. A secondary result counts as a finding only if it also holds on the fresh holdout.

## Method
Model, features, penalties and the frozen plan-1 hyper-parameters of `latent_111` are unchanged (`configs/frozen.yaml`, M03 cell 111: lam 32, half-life 50) for the main comparison; the per-variant tuning is a separate, labelled block below. Walk-forward on dev with a refit before every race (`refit_every=1`), warm starts as in plan 1. The reference `latent_111` is re-run in the same script, not read from a file.

Plackett-Luce, entries in finishing order j = 1..n, strengths u (higher = faster). Plan-1 likelihood: stage j picks entry j among entries j..n with probability exp(u_j) / sum_{k>=j} exp(u_k).

| Switch (`LatentModel` argument) | Change |
|---|---|
| `finishers_only=True` | non-classified entries are dropped from their race entirely: no stage of their own and not in anyone's choice set (censored). Classified entries keep their official order. |
| `topk=10` | only stages j <= k contribute to the likelihood; every entry stays in the choice sets of those k stages (stage j still sums over all entries j..n). |
| `reverse=True` | the order is built from last place upward: stage j picks entry j among entries 1..j with probability exp(-u_j) / sum_{k<=j} exp(-u_k), i.e. Plackett-Luce on the reversed order with negated strengths. |

All three have analytic gradients (`LatentModel._pl_variant`). Finite-difference check: `methodologies/M11_likelihood_variants/gradient_check.py` (writes `results/gradient_check.csv`; fixed seeds, 40 random coordinates, central differences, real training window of 2808 rows and 254 parameters, max relative error): all off 6.9e-7, finishers 6.9e-7, top-10 6.3e-8, reverse 1.4e-7 (finishers + reverse 8.2e-8, not run as a component). The synthetic-data version of the same check is in `tests/test_plan3_m11.py`. (The numbers in the first version of this card, 3.8e-7 / 5.0e-7 / 4.7e-8 / 6.7e-8 on "3047 rows", came from an ad hoc session and could not be traced to a script; they are replaced by the reproducible ones above.)

**How a prediction is obtained from the reverse model.** The parameters are the same `a`, `b`, `g`, `beta`; the fitted linear predictor u is the strength (higher = faster) in all three variants and is used as the score directly (sort of u descending). The sign convention of the reverse model sits inside the likelihood (exp(-u)), so no transformation is applied at prediction time. The probability columns (`p_win`, `p_podium`, `p_points`) come from the harness's Plackett-Luce sampler on u with the online-calibrated temperature, as in plan 1.

**Classification status in the model.** The entries table holds the boolean `classified` (Jolpica `positionText` is a number); the derived finishing order puts classified entries first. `LatentModel` reads it once (`row_cls`) only when `finishers_only=True`, and only for races already in the training window, so nothing about the predicted race is used. With all switches off the original code path runs line for line. Change to shared code: `src/pitlane/models/latent.py` only.

Tie-break: driverId (`pitlane.plan3.decode.break_ties`), never input row order (which is finishing order). No race had tied scores (0 of 66 in all four runs). Grid order for the comparison below uses the same outcome-free tie-break (M17 `grid_free`).

## Leakage notes
Training rows are races before the predicted one. The classification flag of a race is used only after that race has been run. No new inputs. Tuning (below) uses dev only.

## How to run
```
PYTHONPATH=src python methodologies/M11_likelihood_variants/run.py --config methodologies/M11_likelihood_variants/config.yaml      # original run (about 15 s)
PYTHONPATH=src python methodologies/M11_likelihood_variants/revision.py --stage post|mech|tune|inter|final                       # revision (about 2.5 min in total)
PYTHONPATH=src python methodologies/M11_likelihood_variants/gradient_check.py
```
Run times of the revision on this machine: post 1 s, mech under 1 s, tune 106 s (48 walk-forward runs, 1-6 s each), inter 2 s, gradient check a few seconds.

## Reproduction check (all switches off)
Re-run `latent_111` and `latent_110` on dev against the stored plan-1 runs: predicted scores identical (max |difference| 0.0, 1311 entries); the saved `per_race_metrics.csv` of the re-run reference equals the stored plan-1 file in all 20 columns exactly. (`results/reproduction_check.json` lists 8.9e-16: in-memory table vs the stored csv, i.e. csv text round trip.) A separate check against the pre-change `latent.py` from git HEAD at one refit gives 0.0 for cells 111, 110, 011, 101 and for 111, 110 with `lam_a=64`. The tuning grid below contains the reference at (32, 50) and gives rho 0.6460, MAE 3.3552, exact 13.97%, the same numbers.

## Results at the frozen setting (dev, 66 races, 1,311 entries; paired gain = variant minus reference, positive = variant better, 95% bootstrap CI over races, 10,000 resamples, seed 0)
Reference `latent_111`: rho 0.646, MAE 3.355, exact 13.97%, exact on classified 15.13%, rho on finishers 0.769.

**Per-metric verdicts** (`results/verdicts_per_metric_dev.csv`; rule: CI excludes 0):

| Variant | rho | MAE (positions) | exact all (points) |
|---|---|---|---|
| finishers-only | +0.0014 [-0.0100, +0.0130] none | +0.065 [-0.005, +0.134] none | +1.9 [-0.3, +4.2] none |
| top-10 | -0.0085 [-0.0205, +0.0036] none | -0.007 [-0.083, +0.071] none | +1.4 [-0.7, +3.6] none |
| reverse | +0.0062 [-0.0053, +0.0184] none | +0.076 [+0.007, +0.147] **gain** | +3.0 [+0.9, +5.1] **gain** |

Attribution rows carry this as a compact string, for example `rho: none; MAE: gain; exact: gain; dev only`.

Beside it (from `results/paired_gains_dev.csv`): rho on finishers: finishers +0.0075 [-0.0036, +0.0184], top-10 -0.0022 [-0.0128, +0.0082], reverse +0.0070 [-0.0026, +0.0168]. MAE on finishers: finishers +0.117 [+0.044, +0.185], top-10 +0.030 [-0.037, +0.098], reverse +0.102 [+0.043, +0.161]. Exact in places 1-3: finishers +11.6 [+3.0, +19.7], top-10 +6.6 [-1.0, +13.6], reverse +18.2 [+11.1, +24.7] points; places 4-10 and 11+: all intervals include 0.

### Multiplicity (`results/multiplicity_dev.csv`)
36 intervals (3 variants x 12 metrics), 10 exclude 0. p-values are normal approximations from the bootstrap interval (SE = width / 3.92), Holm at 0.05 over 36. Surviving: reverse places 1-3 (p 2e-7), reverse log-loss of P(points) (p 2e-6), reverse MAE on finishers (p 7e-4), finishers-only MAE on finishers (p 1.1e-3). Not surviving: reverse overall exact rate (p 4.5e-3, threshold 1.6e-3), top-10 log-loss, finishers-only places 1-3, reverse exact on classified, finishers-only log-loss, reverse overall MAE (p 0.033). The two metrics on which the per-metric verdict above says "gain" for reverse therefore do not survive a correction over the family. With about thirty components in the plan a few chance passes are expected.

### Comparison with grid order (`results/vs_grid_dev.csv`; outcome-free tie-break; positive = run better than grid; exact in points)
| Run | exact all | exact classified | places 1-3 | places 4-10 | places 11+ | rho | MAE |
|---|---|---|---|---|---|---|---|
| reference | -2.4 [-6.1, +0.7] | -2.6 [-6.4, +0.7] | **-10.1 [-19.7, -0.5]** | -1.1 [-5.4, +3.0] | -1.1 [-4.8, +2.2] | +0.054 [+0.032, +0.079] | +0.240 [+0.090, +0.388] |
| finishers-only | -0.5 [-3.8, +2.6] | -0.4 [-3.8, +2.8] | +1.5 [-6.6, +9.6] | -0.2 [-4.5, +4.3] | -1.3 [-5.0, +2.0] | +0.056 [+0.034, +0.081] | +0.305 [+0.156, +0.466] |
| top-10 | -1.0 [-4.5, +2.1] | -1.1 [-4.8, +2.2] | -3.5 [-12.1, +4.0] | +0.6 [-3.5, +4.5] | -1.5 [-5.4, +2.1] | +0.046 [+0.024, +0.071] | +0.233 [+0.085, +0.391] |
| reverse | +0.6 [-3.3, +4.1] | +0.6 [-3.6, +4.3] | +8.1 [-2.0, +18.2] | -0.4 [-6.1, +5.0] | -1.1 [-5.0, +2.4] | +0.061 [+0.035, +0.089] | +0.316 [+0.139, +0.500] |

Grid order itself: exact 16.4%, places 1-3 35.9%, rho 0.592, MAE 3.595. No variant is better than grid order on any exact-rate row; the latent family is better than grid order on rho and MAE (as in plan 1), with or without the likelihood change. The +18.2 points of reverse on places 1-3 lift a reference that sits 10 points below grid order to a level whose gap to grid (+8.1) includes 0.

Exact rate against grid order by season (points): reference -8.0 [-16.6, -1.6] / -0.7 [-5.1, +3.4] / +1.5 [-2.9, +5.9] (2021 / 2022 / 2023); finishers-only -4.8 / -0.5 / +3.8; top-10 -7.8 / +0.4 / +4.3 (2023 gain, interval excludes 0); reverse -6.4 [-15.7, +1.1] / +1.1 [-2.6, +5.0] / +7.0 [+2.6, +11.5]. The sign is not stable across seasons.

### Per-season spread of the gains vs the reference (`results/per_season_gains_dev.csv`; 22 races per season; exact in points)
| Variant | metric | 2021 | 2022 | 2023 |
|---|---|---|---|---|
| finishers-only | rho | +0.005 | +0.006 | -0.007 |
| | MAE | +0.073 | +0.128 | -0.007 |
| | exact | +3.2 | +0.2 | +2.4 |
| | places 1-3 | **+28.8 [+15.2, +42.4]** | +6.1 | 0.0 |
| top-10 | rho | -0.010 | +0.003 | **-0.019 [-0.031, -0.006]** |
| | exact | +0.2 | +1.1 | +2.8 |
| | places 1-3 | +10.6 [0.0, +21.2] | +6.1 | +3.0 |
| reverse | rho | +0.002 | +0.021 | -0.004 |
| | MAE | +0.022 | **+0.160 [+0.014, +0.323]** | +0.046 |
| | exact | +1.6 | +1.8 | **+5.5 [+2.8, +8.4]** |
| | places 1-3 | **+25.8 [+13.6, +37.9]** | **+16.7 [+1.5, +31.8]** | **+12.1 [+6.1, +18.2]** |

Only the places 1-3 gain of reverse is positive in every season. The rho gain is within +-0.02 in every season and no season interval excludes 0 except top-10 in 2023 (worse). Most of the gain on places 1-3 sits in 2021.

### Face validity: predicted winners (`results/predicted_winner_by_season.csv`, `results/winner_hits_by_season_dev.csv`; 22 races per season)
The reference `latent_111` predicts Hamilton to win 19 of 22 races in 2021 (actual winners: Verstappen 10, Hamilton 8) and 13 of 22 in 2022 (Hamilton won none), Verstappen once in 2022 (actual 15). Reverse predicts Verstappen 16 and Perez 5 in 2022. Correct winner calls (of 22): reference 7 / 3 / 19, finishers-only 13 / 8 / 19, top-10 9 / 8 / 17, reverse 13 / 13 / 19 (2021 / 2022 / 2023). The reference's front-of-field error is a concrete, visible defect; the variants reduce it. The cause is not tested here (see the mechanism check for the fitted strengths). Pole sitters in the file are grid position 1 (21 of 22 in 2021; one race has no entry on grid 1).

### Mechanism check (`results/fitted_components.csv`; descriptive, no verdict)
One cold fit on all races before the first 2022 race and before the first 2023 race, at the frozen setting (lam 32, half-life 50). Convergence is recorded per fit in `fitted_components.csv` (`converged` = scipy success and max |gradient| under 1e-5, `max_abs_grad`, `n_iter`). Seven of the eight fits met it at the first attempt (maxiter 3000; they stopped after 30-46 iterations). The finishers-only fit before 2022 stopped at max |gradient| 4.5e-5 and was refitted from that point (ftol 1e-15, gtol 1e-8, up to 20000 iterations; final max |gradient| 9e-9); beta quali_pos moved from -3.457 to -3.457 (-3.4574 to -3.4567), sd of u from 1.287 to 1.287, weekend share from 62.6% to 62.6%, so nothing in the table changes. Caveat: this applies to these eight cold fits only. The walk-forward prediction runs were not re-run and their per-refit convergence was not stored, so whether their finishers-only refits also stop early (and how many) is unverified. Weekend (qualifying) term, weight on `quali_pos` and sd across entries of the last training season:

| Fit | beta quali_pos | sd weekend part | sd car part | sd driver part | sd of u | weekend share of var(u) |
|---|---|---|---|---|---|---|
| reference, before 2022 | -1.96 | 0.54 | 0.19 | 0.09 | 0.71 | 57% |
| reference, before 2023 | -1.76 | 0.48 | 0.19 | 0.07 | 0.63 | 59% |
| finishers-only, 2022 / 2023 | -3.46 / -3.38 | 1.02 / 0.99 | 0.28 / 0.30 | 0.10 / 0.08 | 1.29 / 1.24 | 63% / 63% |
| top-10, 2022 / 2023 | -3.80 / -3.76 | 1.12 / 1.09 | 0.20 / 0.25 | 0.08 / 0.07 | 1.31 / 1.29 | 73% / 71% |
| reverse, 2022 / 2023 | -3.38 / -3.31 | 0.93 / 0.92 | 0.35 / 0.38 | 0.11 / 0.10 | 1.29 / 1.26 | 52% / 53% |

**The qualifying weight differs between variants**: in raw units it is 1.7 to 2.2 times the reference's in every variant. The whole strength scale is also about 1.8 times larger (sd of u 1.24-1.31 against 0.63-0.71), so in relative terms the weekend share of the variance of u moves little for finishers-only (63% against 57-59%) and reverse (52-53%), and more for top-10 (71-73%). The variants fit a sharper u; they do not simply lean harder on qualifying. Strengths: before 2022 the reference ranks Hamilton (+0.35), Perez, Norris, Sainz, Ricciardo as top-5 drivers and Red Bull 2021 only 4th among cars (+0.06); finishers-only has Hamilton +0.29, Verstappen +0.28 and Red Bull 2021 2nd (+0.25); reverse Hamilton +0.36, Verstappen +0.23, and Red Bull 2021 (+0.55) ahead of Mercedes (+0.52). Before 2023 the reference ranks Verstappen second among drivers by a margin of 0.17 below Hamilton and Mercedes-2022 above Red Bull-2022; reverse puts Red Bull-2022 first (+0.69). So the variants re-rank cars and drivers, mainly Red Bull and Verstappen, in addition to rescaling. Why the forward likelihood under-rates Red Bull 2021 is not tested here (a hypothesis only: its retirements count against it in every stage, which finishers-only removes, but reverse also moves the same way without dropping retirees).

## Tuning per variant (`results/tuning_grid_dev.csv`, `results/tuned_vs_untuned_dev.csv`)
Grid used by plan 1 for cell 111 (`pitlane.pipeline.LAM_GRID x HL_GRID`): lam in {0.5, 2, 8, 32} x half-life in {8, 20, 50} = 12 configurations (cap 30 respected), each variant and the reference, walk-forward with a refit before every race, selection by mean rho (plan-1 rule; plan 1 used a refit every 4 races while tuning, here every race). 48 runs, 106 s.

| Run | chosen (lam, half-life) | rho | MAE | exact % | gain vs reference at frozen setting: rho | MAE | exact (points) | places 1-3 (points) |
|---|---|---|---|---|---|---|---|---|
| reference (frozen = its own best) | 32, 50 | 0.6460 | 3.355 | 13.97 | | | | |
| finishers-only, tuned | 8, 50 | 0.6571 | 3.251 | 16.18 | +0.0111 [-0.0032, +0.0257] | +0.105 [+0.018, +0.189] | +2.2 [-0.1, +4.6] | +13.6 [+5.6, +21.7] |
| finishers-only, untuned | 32, 50 | 0.6474 | 3.291 | 15.89 | +0.0014 [-0.0100, +0.0130] | +0.065 [-0.005, +0.134] | +1.9 [-0.3, +4.2] | +11.6 [+3.0, +19.7] |
| top-10, tuned | 2, 50 | 0.6537 | 3.334 | 14.41 | +0.0077 [-0.0066, +0.0216] | +0.022 [-0.076, +0.121] | +0.4 [-1.9, +2.9] | +5.1 [-1.0, +11.1] |
| top-10, untuned | 32, 50 | 0.6374 | 3.362 | 15.35 | -0.0085 [-0.0205, +0.0036] | -0.007 [-0.083, +0.071] | +1.4 [-0.7, +3.5] | +6.6 [-1.0, +13.6] |
| reverse, tuned | 2, 50 | 0.6642 | 3.221 | 15.55 | +0.0183 [-0.0004, +0.0362] | +0.134 [+0.026, +0.240] | +1.6 [-0.6, +3.9] | +14.6 [+7.6, +21.7] |
| reverse, untuned | 32, 50 | 0.6522 | 3.279 | 16.97 | +0.0062 [-0.0053, +0.0184] | +0.076 [+0.007, +0.147] | +3.0 [+0.9, +5.1] | +18.2 [+11.1, +24.7] |

Choosing the setting and judging it on the same 66 races is optimistic: the tuned rows can only be better on the selection metric (rho) than the untuned rows by construction. **The untuned comparison (same frozen setting for all four) is the conservative one**, and it is the one the verdict above uses. Reading: tuning raises each variant's rho point estimate (finishers-only +0.010, top-10 +0.016, reverse +0.012), because all three prefer a weaker ridge (lam 2-8) than the reference's lam 32; no tuned rho interval excludes 0 (reverse: lower bound -0.0004). Tuned reverse gains MAE (+0.134 [+0.026, +0.240]) and tuned finishers-only gains MAE (+0.105 [+0.018, +0.189]) against the reference; tuned exact-rate gains lose significance for reverse (+1.6 [-0.6, +3.9]) and stay null for the other two. Two caveats: the reference's best setting is lam 32, the largest value of the plan-1 grid (it may prefer more ridge; not checked, since the grid was not extended), and the tuned variants are compared with a reference whose own optimum was found by the same procedure on the same races (plan 1 used the same dev block).

## Interaction declared in PLAN3 §5: finishers-only likelihood x retirement model (M11 x M07)
**Built from frozen plan-2 pieces; no design decision was needed.** Order model: finishers-only `latent_111` (frozen setting), reference: forward `latent_111`; retirement probabilities: the stored M07 dev predictions (plan 2); simulator: plan-2 S3 with its frozen parameters unchanged (tau, dnf_scale, kappa, p0, s_dis, a_vol, a_rain from `configs/frozen_plan2.yaml`), 1000 draws, same random streams for both; decoding: expected position (D0, outcome-free tie-break). `pitlane.plan3.calibration.grid_slot_calibration` is in `results/interaction_calibration_dev.csv`. (The first card said the interaction "has no motivation from this null result"; that is not a reason the plan allows, since the interaction is declared. It is now run.)

Gain of finishers-only x S3 minus forward x S3 (positive = finishers-only better; `results/interaction_paired_gains_dev.csv`):

| metric | gain | 95% CI |
|---|---|---|
| RPS | +0.0007 | [-0.0015, +0.0029] none |
| rho | +0.0007 | [-0.0092, +0.0111] none |
| MAE | +0.047 | [-0.012, +0.108] none |
| exact (points) | +1.5 | [-0.6, +3.7] none |
| exact places 1-3 (points) | +12.1 | [+4.5, +20.2] |
| MAE on finishers | +0.094 | [+0.023, +0.161] |
| log-loss of P(points) | -0.0196 | [-0.0378, -0.0033] worse |

Contrast (gain inside the simulator minus gain with plain order, `results/interaction_contrast_dev.csv`): rho -0.0007 [-0.0059, +0.0046], MAE -0.017 [-0.053, +0.020], exact -0.4 [-2.6, +1.8], places 1-3 +0.5 [-6.6, +7.6]. There is no measurable interaction: the simulator neither adds to nor removes the finishers-only gain. M07 does not change what the likelihood change does to the order.

**Scale problem (read before using the probability rows).** The frozen S3 tau was fitted on a score with a within-race sd of about 0.70 (M06) and the reference u has 0.66, but the finishers-only u has 1.24 (see mechanism). The same tau therefore makes the finishers-only matrix about 1.9 times too sharp: P(finish in grid slot) 0.156 against 0.114 for the reference and 0.164 realised, and the log-loss of P(points) is worse. The RPS and log-loss rows are thus confounded with scale. As a descriptive diagnostic only (no verdict, not in attribution.csv, scale chosen after seeing the problem), multiplying the finishers-only score by 0.536 (the ratio of mean within-race sd, a pre-race quantity) gives RPS +0.0017 [+0.0004, +0.0032], log-loss +0.003 [-0.004, +0.010], exact +2.0 [-0.2, +4.2] (`results/interaction_scale_matched_dev.csv`). That would be the only RPS interval excluding 0 in this work package, and it rests on a post hoc rescaling and one comparison; it is not a finding. The attribution row records the as-defined result (`RPS: none; rho: none; MAE: none; exact: none; dev only`). Refitting tau per variant would be the principled version; it changes what "frozen plan-2 pieces" means, so it is left to a decision by the plan's owner.

## What we learned
- On rho, none of the three variants measurably differs from the plan-1 likelihood at the frozen setting, and none does after tuning (all intervals include 0). On MAE and exact rate, reverse passes alone; neither survives Holm over 36 intervals, and the exact-rate gain is mostly the front of the field.
- The front gain is a repair: the reference is below grid order on places 1-3 and wrongly names Hamilton as the favourite in 2021-22. After the repair reverse is not better than grid order on any exact-rate row. The Amendment-2 target (beat grid order) is not met by any variant.
- The variants change the strength scale (sd of u about 1.8 times larger) and re-rank cars; the weekend weight rises in raw units but not in share (finishers-only, reverse).
- Reverse is the one to carry to the fresh holdout if M11 is carried at all: it has the largest and most consistent places 1-3 gain (positive in each season) and the best MAE, but a rho gain within noise.
- Retirements in the likelihood: finishers-only does not change rho, the plan's own prior that retirements are mostly luck is consistent with this; M07 in the simulator does not interact with it.

## Limits
- Dev only (66 races, three seasons, one rule change in 2022). No result is confirmed on the fresh holdout; all verdicts are "dev only". Nothing was scored on 2024-26.
- Hyper-parameters of the reference were tuned on the same dev races as everything here, so the reference comparison is not independent of dev; the tuned-vs-untuned numbers add a second use of the same 66 races for the variants.
- About thirty components in the plan: chance passes are expected. 36 intervals here, 10 exclude 0, 4 survive Holm. Holm uses normal approximations from bootstrap intervals; the 66 races are not independent of each other across seasons.
- RPS: the stand-alone variants have no RPS (the harness's plain Plackett-Luce sampler does not produce a position matrix in the stored runs; `gain_rps` stays empty in the three alone-rows). RPS exists only for the interaction runs and carries the scale confound above.
- The tuned grid is the plan-1 grid; the reference's optimum lies at its upper edge (lam 32).
- Per-season intervals have 22 races each and are wide; no multiplicity correction is applied to them.
- The mechanism table is descriptive and rests on two cold fits per variant, not on the walk-forward chain.

## Open questions / next tries
- Combined finishers-only + reverse (the code supports it, gradient checked) was not scored because it is not a declared component.
- Refit the simulator's tau for each order model (needs a decision, see the interaction section).
- The reverse gain on places 1-3 is the lead to check on the fresh holdout, against grid order.
