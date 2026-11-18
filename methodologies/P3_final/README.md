# P3_final — final plan-3 pre-race model and freeze preparation

**Status:** done (dev only); frozen. Holdout scoring rules: `HOLDOUT_PREREG.md`. Freeze record: `FREEZE.md`.
**Plan reference:** PLAN3.md §4 "Final model", §5 attribution protocol and decision rules, §6 R14
**Factors:** F1 dynamic = on · F2 decomposition = on · F3 ranking = on (latent cell 111, reverse likelihood)

## Hypothesis
Pre-stated in `prestatement.md` (written before any run). The components that pass alone on dev (M11 reverse likelihood, M09 car
race-pace history) survive leave-one-out; the car-only penalty re-tune adds 0.000 to +0.005 rho (failure: interval includes 0);
the final model beats `latent_111` on dev, and does **not** beat grid order on exact rate (the expected outcome, as in every earlier package).

## Method
- **Reference** `latent_111` (plan-1 frozen: lam 32, half-life 50), re-run in the same harness: reproduces the stored plan-1 dev run
  exactly (max score difference 0.0, 66 races, 1,311 entries).
- **Component `car_penalty_retune`** (declared by the main session): car penalty `lam` over {2, 4, 8, 16, 32, 64, 128, 256}, driver penalty
  held at 32, pick by dev rho at `refit_every = 4`, then a full run. 8 configurations.
- **Final candidate** = `latent_111` + `M11_reverse` + `M09_car_pace_hist` (`pace_hist_x`, D1 median clean lap, half-life 50, scaled by the SD
  over seasons <= 2020, `lam_x` 3.0). Not included, as decided before this package: finishers-only, top-k, M09 Q1 offset (own increment null)
  and its off-ladder "alone" version (not declared), M05, M12, M17 decoders, M18, M20, M16 S4-S7, M14 supplier pooling.
- **`lam_x` for the combination:** checked once over {0.3, 1, 3, 10} (4 configurations, rule: switch only if better by more than 0.002 rho).
  Best 0.3 (rho 0.6640) against 3.0 (0.6634): within the margin, so M09's 3.0 is kept.
- **Leave-one-out:** full against full minus one component, paired per race, 10,000 resamples, seed 0, 66 dev races.
- **Simulator:** plan-2 S3 re-fitted on dev (same coordinate search, grids and two passes as plan 2) on the final model's walk-forward
  scores, M07 retirement model as in plan 2, decoder D0. Also re-fitted on `latent_111` scores as the fair reference.
- **Freeze:** `frozen_config.yaml`, SHA-256 in `FREEZE.md`, `predict.py` (checks the hash, refuses if the config was edited).
- Budget: 8 (car penalty) + 4 (`lam_x` check) = 12 order-model configurations of 30; the S3 re-fit is the plan-2 coordinate search
  (61 evaluations per fit, as plan 2; it is a fit of 7 simulator parameters, not a variant comparison).

## Leakage notes
- `pace_hist_x` for race r uses only races before r (decayed mean of the car lineage's earlier race-pace gaps, from laps). Laps of 2024-26 come
  from OpenF1, laps up to 2023 from Jolpica; the extended pace table equals the stored A3 table exactly up to 2023, and the column equals M09's
  column on dev bit for bit (tests).
- `predict_race` keeps only rows with race_id <= r, blanks the outcome columns of race r, and uses a pace table without the laps of r and later
  in the tests. Predictions are bit-identical when data after r are removed, or when the outcomes of r and later races are permuted
  (`tests/test_plan3_final.py`). The M07 retirement model and the simulator draw only on earlier races.
- The scale constant (SD over seasons <= 2020) is stored in the frozen config; nothing is re-tuned on 2024-2026.

## How to run
```
python methodologies/P3_final/run.py            # dev run, writes run folders, leaderboard and attribution rows (append only), frozen config
python methodologies/P3_final/run.py --quick    # 6 races, nothing written
python methodologies/P3_final/predict.py --season S --round R [--horizon H2|H1|both] [--entries PATH]
```
`run.py` appends rows to `results/plan3/attribution.csv` and `results/plan3/leaderboard.csv` and refuses to run twice (rows exist).

## Results (dev 2021-2023, 66 races, outcome-free tie-break, 95% bootstrap CI over races, positive = candidate better)

**Failures and caveats first.**
- `car_penalty_retune` **fails alone**: pick lam 16 (dev rho at refit 4: 0.6486 against 0.6454 at 32), rho +0.0019 [-0.0040, +0.0074], MAE -0.017
  [-0.054, +0.019], exact -0.003 [-0.020, +0.012]. Not included. The under-dispersion suspicion from R6 is not confirmed by a weaker or stronger car
  penalty on this data; the gain that a decayed car-form column gave is not reproduced by changing lam.
- **The final model's rho gain over `latent_111` has its lower bound at 0:** +0.0172 [-0.0002, +0.0340] (the skeptic's independent bootstrap gives [+0.0002, +0.0344]; read as neither a pass nor a clean null). MAE (+0.119 [+0.013, +0.222]) and the
  places 1-3 exact rate (+10.6 points [+4.6, +17.2]) exclude 0. Finishers' rho +0.0193 [+0.0031, +0.0351], points log-loss +0.0115 [+0.0057, +0.0170].
  Both components were chosen by passing alone on these same 66 races, so all of this is optimistic; per season rho gains are +0.016 / +0.028 / +0.008, each
  with an interval including 0.
- **The pre-race exact-rate target is not met:** final model 0.1512 against grid order 0.1638, paired -0.0127 [-0.0508, +0.0207]. The simulator order is
  lower still (0.1457; -0.0181 [-0.0567, +0.0165]). `M09_car_pace_hist` lowers the exact rate inside the final model (-0.0185 [-0.0382, -0.0000]).
- M09's own caveats stand (Amendment 3 item 6; R6 control): the pace column is a car-form proxy, credited to car form by the pre-stated control, and counts only if
  the fresh holdout confirms it. M11 reverse "passes" on MAE and exact, not on rho, and fails a Holm correction over its 36 intervals (R5).
- `predict_race` is not bit-identical to the harness: it uses the universe of rows up to r and a 15-race warm start. Scores differ by up to 6.9e-4
  over 3 dev races (coder) and 1.0e-3 over 12 dev races (tester), with no score-order flip; a test bounds the difference at 5e-3 and 0.999.
- **The simulator (D0) order from `predict.py` is not the dev D0 order.** The harness draws all races from one random stream; `predict.py` draws one
  race from seed 0 with 1,000 draws. On 11 of 12 dev races the order differs by 1-3 places (max |E[position]| difference 0.22-0.78), and permuting a
  race's entry rows shifts it by up to 1 place. This is Monte-Carlo noise, not leakage, but adjacent places in the D0 order are not stable. That is one
  reason `HOLDOUT_PREREG.md` scores the score order, and treats the D0 order and `predict.py` output as descriptive.
- **The full model has a worse exact rate than full minus the pace column:** 0.1512 against 0.1697, paired -0.0185 [-0.0382, -0.0000]; places 1-3
  -0.076 [-0.126, -0.025]. The pace column is retained under the pre-stated rule (which looks only at the metric it passed on, rho); holdout test T3
  decides whether it stays supported.

**Leave-one-out (full = reverse + pace_hist; positive = full better than full minus the component)**

| Removed | rho | MAE | exact | verdict |
|---|---|---|---|---|
| `M11_reverse` | +0.0010 [-0.0105, +0.0124] | +0.078 [+0.004, +0.152] | +0.0245 [+0.0040, +0.0452] | retained (criterion MAE or exact) |
| `M09_car_pace_hist` | +0.0110 [+0.0012, +0.0203] | +0.044 [-0.025, +0.110] | -0.0185 [-0.0382, -0.0000] | retained (criterion rho) |

Neither component is redundant under the pre-stated rule; no elimination round was needed. Re-runs of full-minus-pace_hist and full-minus-reverse reproduce the
stored M11 reverse and M09 history runs bit for bit (score difference 0.0).

**Final model against `latent_111` and grid order (means: grid order / `latent_111` / final)**

| Metric | Grid | `latent_111` | Final | Final minus `latent_111` | Final minus grid |
|---|---|---|---|---|---|
| Spearman rho | 0.5916 | 0.6460 | 0.6632 | +0.0172 [-0.0002, +0.0340] | +0.0716 [+0.0425, +0.1027] |
| rho, finishers | 0.7146 | 0.7690 | 0.7883 | +0.0193 [+0.0031, +0.0351] | +0.0737 [+0.0425, +0.1054] |
| MAE | 3.595 | 3.355 | 3.236 | +0.119 [+0.013, +0.222] | +0.359 [+0.151, +0.566] |
| MAE, finishers | 2.620 | 2.373 | 2.206 | +0.167 [+0.069, +0.257] | +0.414 [+0.213, +0.607] |
| Exact rate | 0.1638 | 0.1397 | 0.1512 | +0.0115 [-0.0106, +0.0337] | -0.0127 [-0.0508, +0.0207] |
| Exact, finishers | 0.1770 | 0.1513 | 0.1609 | +0.0096 [-0.0157, +0.0353] | -0.0161 [-0.0563, +0.0206] |
| Exact, places 1-3 | 0.3586 | 0.2576 | 0.3636 | +0.1061 [+0.0455, +0.1717] | +0.0051 [-0.0960, +0.1111] |
| Exact, places 4-10 | 0.1515 | 0.1407 | 0.1277 | -0.0130 [-0.0476, +0.0238] | -0.0238 [-0.0779, +0.0281] |
| Exact, places 11+ | 0.1138 | 0.1030 | 0.1027 | -0.0003 [-0.0291, +0.0279] | -0.0111 [-0.0470, +0.0241] |
| Within +-1 | 0.3609 | 0.3639 | 0.3759 | +0.0120 [-0.0158, +0.0394] | +0.0150 [-0.0268, +0.0549] |
| Within +-2 | 0.5120 | 0.5332 | 0.5443 | +0.0110 [-0.0132, +0.0352] | +0.0323 [-0.0076, +0.0699] |
| Points log-loss | n/a | 0.5282 | 0.5168 | +0.0115 [+0.0057, +0.0170] | n/a |

(Within +-3, reported only: grid 0.6280, `latent_111` 0.6622, final 0.6758. Top-10 and top-5 overlap are in `per_race_metrics.csv` of the run.)

**Re-fitted simulator S3 (RPS and log-loss of P(points finish))**

| Model through S3 | S3 parameters fitted on dev | RPS | log-loss (points) | exact (D0) |
|---|---|---|---|---|
| `latent_111` scores | tau 3.5, dnf_scale 1.25, kappa 0, p0 0.2, s_dis 2, a_vol 0.5, a_rain 2 (= plan-2 values) | 0.1201 | 0.4861 | 0.1366 |
| final scores | tau 1.7, dnf_scale 1.25, kappa 0, p0 0.05, s_dis 1.5, a_vol 1.0, a_rain 2 | 0.1149 | 0.4671 | 0.1457 |
| final scores, unchanged plan-2 theta (descriptive) | plan-2 | 0.1159 | 0.4807 | 0.1435 |
| plan-2 S3 on the M06 H3 score (stored run; more weekend information, descriptive) | plan-2 | 0.1204 | 0.4888 | 0.1357 |

Final through re-fitted S3 against `latent_111` through re-fitted S3: RPS +0.0052 [+0.0029, +0.0074], log-loss +0.0191 [+0.0093, +0.0287], rho +0.0195
[+0.0046, +0.0342]. Against plan-2 S3 on the M06 score: RPS +0.0055 [+0.0029, +0.0082] (descriptive: different information). Re-fit against unchanged plan-2
theta on the final score: RPS +0.0010 [-0.0000, +0.0020], so most of the RPS gain comes from the score, not the re-fit. Both S3 fits are in-sample on the same 66
races, so RPS 0.1149 is not comparable with a holdout RPS. Four of the seven fitted parameters sit at a grid edge (p0, s_dis, a_vol, a_rain); the grids are not extended. The re-fitted matrix puts 12.0% on "finishes in the grid slot" against 16.4% realised (still under-confident there, as in R2/R12). The re-fit moves tau from 3.5 to 1.7,
consistent with a wider score scale under the final model (that reverse is the cause is untested); its gain over plan-2 theta is in log-loss only, in-sample.

**H1 (before qualifying)** is the plan-2 frozen H1 `111` model, unchanged: the components were never tested at H1. Descriptive only, 22 races of 2023: H1 with
reverse and `pace_hist_x` switched on, rho +0.0193 [-0.0087, +0.0533], MAE +0.191 [+0.008, +0.390], exact +0.0346 [-0.0045, +0.0733] against plan-2 H1 (no decision weight).

## What we learned
- Only two components made it. Their dev evidence is weak in different ways (see caveats): reverse passes on MAE and exact but not rho; the pace column is a car-form
  proxy; the combined rho interval touches 0. Both stay in the frozen model by the pre-stated rules and are to be confirmed or rejected on the fresh holdout.
- The car-only penalty does not change much: dev rho (refit 4) over lam runs from 0.625 (lam 2) to 0.649 (lam 16); lam 32 is within 0.003 of the best, and the gain of lam 16 at refit 1 is inside the noise.
- With its own in-sample S3 re-fit the final model improves RPS over the same pipeline on `latent_111` by 0.005; almost all of that comes from the score, not the re-fit.

## Open questions / next tries
- No model beats grid order on the pre-race exact rate with an interval excluding 0 (`reverse_only` is above grid at 0.1697 against 0.1638, inside the noise; the final model is below it at 0.1512, mostly because of the pace column).
- The entries row of a not-yet-run race has to be built from pre-race sources (grid from qualifying and penalties): the repository builds `entries2` only after a race, so
  `predict.py` takes the entries frame; a live pre-race entries builder does not exist yet.
- Features for 2027 need the 2027 entry list (lineage map, circuit types).

## Files
`prestatement.md` (before the runs), `HOLDOUT_PREREG.md` (holdout tests, before 2026 R21), `config.yaml`, `run.py`, `predict.py`, `frozen_config.yaml`, `FREEZE.md`, `results/<run_id>/` (nine run folders) and
`results/*.csv`, `results/reproduction_check.json`. Cross-method rows: `results/plan3/attribution.csv` (7 new rows, `loo_gain` filled on the two `P3_LOO_*` rows),
`results/plan3/leaderboard.csv` (9 new rows, method `P3`).

## R15 scoring (written 2026-11-18, data through 2026 R20, race_id 271)

Script `methodologies/P3_final/score.py` (helpers `src/pitlane/plan3/holdout.py`, tests `tests/test_plan3_score.py`). It scores frozen models only: nothing is tuned,
selected or switched. Run folder `results/20261118-1407_p3_R15_score/`; per-row folders `results/20261118-1407_p3_R15_<model>_<split>_<split>/` (the split is in the label, which must be unique in the plan-3 leaderboard, and `save_run` appends it again); tables
`results/plan3/third_look.csv` and `results/plan3/holdout_status.json`.

**Fresh-holdout race count: n = 0.** The data end at 2026 R20 (race_id 271). There is no fresh number and no verdict, and none is possible before 20 races
(HOLDOUT_PREREG section 4). `holdout_status.json` records n = 0 and "no verdict (n < 20)". Every number below is a **third look** on 2024 to 2026 R20, labelled
"third look, no decision weight": those races were scored in plans 1 and 2, the final model's components were chosen on dev, and nothing here confirms or
rejects T1-T3. No Holm correction is applied here (descriptive only). Intervals: paired bootstrap over races, 10,000 resamples, seed 0, 95%, positive = candidate better
(same code as `attribution.csv`).

### Failures, nulls and caveats first
- **The plan-3 dev rho gain does not replicate on the old test blocks:** pooled T1 +0.0006 [-0.009, +0.010], an interval that excludes the dev point estimate +0.0172; the 2024-25 sign is negative.
  There is currently no evidence that the frozen plan-3 model beats `latent_111`; the fresh holdout has 0 races.
  T1 detail: pooled +0.0006 [-0.0090, +0.0099] (68 races); 2024-25 -0.0028 [-0.0143, +0.0086] (48); 2026 R1-20 +0.0086 [-0.0064, +0.0251] (20). The exact rate on places 1-3 is
  lower for the final model than for `latent_111` on 2024-25 (-0.069 [-0.139, +0.000]).
- **T3 (the pace column, `final` against `reverse_only`, rho) does not help on the old blocks:** pooled -0.0036 [-0.0108, +0.0037]; 2024-25 -0.0069 [-0.0165, +0.0029];
  2026 +0.0044 [-0.0038, +0.0134]. On the 2024-25 position error the final model is worse than `reverse_only` with intervals excluding 0: T3 MAE (all entries) -0.070 [-0.138, -0.003]
  against the pace column, and finishers' MAE -0.082 [-0.150, -0.019]. These are uncorrected. A null on rho cannot tell car form from race pace; the old blocks do not decide what the pace column measures.
- **T2 (reverse, MAE) is positive only in 2026 R1-20:** +0.078 [+0.001, +0.158] (20 races). The lower bound sits on 0 and changes sign with another bootstrap seed. It is uncorrected and is not support;
  2024-25 -0.002 [-0.060, +0.057]; pooled +0.021 [-0.028, +0.070].
- **The exact-rate target against grid order is not met by any model on the old blocks.** Pooled exact: grid 0.1624, `final` 0.1414 (-0.0210 [-0.0487, +0.0057]),
  `reverse_only` 0.1504 (-0.0121 [-0.0373, +0.0134]), `latent_111` 0.1458 (-0.0166 [-0.0420, +0.0081]). In 2026 R1-20 `latent_111` is clearly below grid: -0.0540 [-0.0767, -0.0328].
- **The final model is below grid order on exact in 2025:** -0.029 [-0.052, -0.006]; on places 1-3 in 2025 -0.139 [-0.264, -0.014] (uncorrected).
- **Multiplicity:** `third_look.csv` holds 516 contrast intervals, many of them duplicated across blocks and names; 75 exclude 0 (54 positive, 21 negative), mostly model-versus-grid rho. No single cell is read as support.
- The `logloss_points` rows of the plain models in `third_look.csv` come from the harness Plackett-Luce probabilities, not from S3; the column `prob_source` says which (`S3 matrix` rows are the simulator).
- Per-row `metrics.json` files carry `scored: <split>: third look, no decision weight`.
- Probabilities: final through S3 against `latent_111` through its own S3 re-fit: RPS pooled +0.0009 [-0.0006, +0.0023]; 2024-25 -0.0001 [-0.0016, +0.0016]; 2026 +0.0031 [-0.0001, +0.0062].
  Points log-loss pooled +0.0063 [-0.0028, +0.0152]; 2026 +0.0164 [+0.0032, +0.0315] (20 races, uncorrected). Nothing in 2024-25.
- `plan2_H2` is ambiguous in the frozen pre-registration (`configs/frozen_plan2.yaml` has no H2 key). It is scored as `latent_111` (plan-2 H2 vocabulary: H2 = plan 1,
  `methodologies/M12_two_stage/README.md`; identical to the stored plan-2 `m06_L_r0_111` run), so final vs plan2_H2 = T1 and adds no information. The plan-2 H3 main model (M06L/111) is not scored under this reading.
- Metric choice: "Spearman rho, all classified" is `spearman` (all entries, retirees at their derived positions), as in `attribution.csv`; `spearman_fin` is the finishers' version.
- The old blocks are scored on the data through race_id 271 (universe as at the freeze, so they reproduce the stored runs exactly). The fresh holdout will use all rows then in the table (the pre-registered "full universe"); the row-hash check keeps the old rows fixed.
- The 2024-25 block has 48 races (24 per season), not 44.

### Third look, means (outcome-free tie-break; test_stable 48 races, test_shift 20, pooled 68)
| Block | Model | rho | MAE | exact | rho fin. | MAE fin. | exact 1-3 | exact 4-10 | exact 11+ | within 1 | within 2 | within 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_stable | grid | 0.6903 | 3.099 | 0.1657 | 0.7574 | 2.402 | 0.382 | 0.176 | 0.092 | 0.389 | 0.567 | 0.684 |
| test_stable | latent_111 (= plan2_H2) | 0.7029 | 3.064 | 0.1647 | 0.7800 | 2.318 | 0.368 | 0.182 | 0.091 | 0.390 | 0.551 | 0.687 |
| test_stable | reverse_only | 0.7070 | 3.043 | 0.1603 | 0.7863 | 2.293 | 0.354 | 0.182 | 0.087 | 0.385 | 0.564 | 0.690 |
| test_stable | final_minus_reverse | 0.6997 | 3.111 | 0.1375 | 0.7802 | 2.346 | 0.306 | 0.119 | 0.100 | 0.385 | 0.547 | 0.675 |
| test_stable | final | 0.7001 | 3.113 | 0.1481 | 0.7783 | 2.375 | 0.299 | 0.158 | 0.095 | 0.369 | 0.548 | 0.681 |
| test_stable | H1 (plan-2 frozen) | 0.6049 | 3.772 | 0.1038 | 0.6920 | 3.019 | 0.215 | 0.092 | 0.078 | 0.296 | 0.446 | 0.585 |
| test_shift | grid | 0.6663 | 3.360 | 0.1546 | 0.7596 | 2.422 | 0.333 | 0.157 | 0.103 | 0.403 | 0.562 | 0.677 |
| test_shift | latent_111 (= plan2_H2) | 0.7108 | 3.252 | 0.1006 | 0.8017 | 2.276 | 0.267 | 0.079 | 0.068 | 0.365 | 0.546 | 0.707 |
| test_shift | reverse_only | 0.7150 | 3.175 | 0.1265 | 0.8120 | 2.158 | 0.283 | 0.107 | 0.094 | 0.376 | 0.563 | 0.697 |
| test_shift | final_minus_reverse | 0.7116 | 3.245 | 0.1252 | 0.8100 | 2.241 | 0.267 | 0.079 | 0.115 | 0.357 | 0.544 | 0.695 |
| test_shift | final | 0.7194 | 3.167 | 0.1254 | 0.8223 | 2.101 | 0.300 | 0.086 | 0.100 | 0.370 | 0.572 | 0.694 |
| test_shift | H1 (plan-2 frozen) | 0.6554 | 3.800 | 0.0896 | 0.7295 | 2.827 | 0.083 | 0.071 | 0.102 | 0.270 | 0.470 | 0.592 |
| pooled | grid | 0.6832 | 3.176 | 0.1624 | 0.7580 | 2.408 | 0.368 | 0.170 | 0.095 | 0.393 | 0.566 | 0.682 |
| pooled | latent_111 (= plan2_H2) | 0.7052 | 3.120 | 0.1458 | 0.7864 | 2.306 | 0.338 | 0.151 | 0.084 | 0.382 | 0.550 | 0.693 |
| pooled | reverse_only | 0.7093 | 3.082 | 0.1504 | 0.7939 | 2.253 | 0.333 | 0.160 | 0.089 | 0.382 | 0.564 | 0.692 |
| pooled | final_minus_reverse | 0.7032 | 3.151 | 0.1339 | 0.7890 | 2.315 | 0.294 | 0.107 | 0.105 | 0.377 | 0.546 | 0.681 |
| pooled | final | 0.7058 | 3.129 | 0.1414 | 0.7913 | 2.295 | 0.299 | 0.137 | 0.097 | 0.369 | 0.555 | 0.685 |
| pooled | H1 (plan-2 frozen) | 0.6197 | 3.780 | 0.0996 | 0.7030 | 2.962 | 0.177 | 0.086 | 0.085 | 0.288 | 0.453 | 0.587 |

Probabilities (S3 matrix, frozen theta for `final`; `latent_111` through its own R14 S3 re-fit; seeds `0 + 10000 + i`, i = race index within the block):

| Block | Model | RPS | points log-loss |
|---|---|---|---|
| test_stable | final | 0.1097 | 0.4434 |
| test_stable | latent_111 | 0.1096 | 0.4456 |
| test_shift | final | 0.1072 | 0.4470 |
| test_shift | latent_111 | 0.1103 | 0.4634 |
| pooled | final | 0.1090 | 0.4445 |
| pooled | latent_111 | 0.1098 | 0.4508 |

### Third look, paired contrasts (candidate minus reference; positive = candidate better; 95% bootstrap interval; third look, no decision weight)
| Contrast (metric) | test_stable (48) | test_shift (20) | pooled (68) |
|---|---|---|---|
| T1 final - latent_111 (rho) | -0.0028 [-0.0143, +0.0086] | +0.0086 [-0.0064, +0.0251] | +0.0006 [-0.0090, +0.0099] |
| T2 final - final_minus_reverse (MAE gain) | -0.0022 [-0.0596, +0.0565] | +0.0780 [+0.0009, +0.1576] | +0.0214 [-0.0276, +0.0695] |
| T3 final - reverse_only (rho) | -0.0069 [-0.0165, +0.0029] | +0.0044 [-0.0038, +0.0134] | -0.0036 [-0.0108, +0.0037] |
| exact: final - grid | -0.0176 [-0.0540, +0.0173] | -0.0291 [-0.0682, +0.0040] | -0.0210 [-0.0487, +0.0057] |
| exact: reverse_only - grid | -0.0054 [-0.0397, +0.0271] | -0.0281 [-0.0570, -0.0000] | -0.0121 [-0.0373, +0.0134] |
| exact: latent_111 - grid | -0.0010 [-0.0361, +0.0316] | -0.0540 [-0.0767, -0.0328] | -0.0166 [-0.0420, +0.0081] |
| final - plan2_H2 (rho) | same as T1 (same model) | same as T1 | same as T1 |
| S3 RPS gain, final - latent_111 | -0.0001 [-0.0016, +0.0016] | +0.0031 [-0.0001, +0.0062] | +0.0009 [-0.0006, +0.0023] |
| S3 points log-loss gain | +0.0021 [-0.0088, +0.0130] | +0.0164 [+0.0032, +0.0315] | +0.0063 [-0.0028, +0.0152] |

All other metrics, per-season rows (2024, 2025, 2026) and the finishers' versions of every contrast are in `results/plan3/third_look.csv`. By season (rho, T1): 2024 -0.0018 [-0.0182, +0.0141],
2025 -0.0037 [-0.0204, +0.0123], 2026 R1-20 +0.0086 [-0.0064, +0.0251]; T3: -0.0061, -0.0076, +0.0044, all with intervals including 0.

### H1 (descriptive)
The frozen H1 model is the plan-2 `h1_latent_111` model; the re-run reproduces the stored plan-2 runs exactly (below). It is far behind every H2 model
(pooled rho 0.6197 against 0.7052 for `latent_111`), as expected for a model without qualifying information.

### Reproduction checks (third look against stored runs; nothing differs)
| Check | Races / entries | max abs difference, per-race metrics | max abs difference, scores |
|---|---|---|---|
| `latent_111` against stored plan-1 run, test_stable | 48 / 952 | 8.9e-16 | 0.0 |
| `latent_111` against stored plan-1 run, test_shift | 20 / 432 | 8.9e-16 | 0.0 |
| `plan2_H2` (= `latent_111`) against stored plan-2 `m06_L_r0_111`, both blocks | 48 / 952; 20 / 432 | 8.9e-16 | 0.0 |
| H1 against stored plan-2 `h1_latent_111`, both blocks | 48 / 952; 20 / 432 | 8.9e-16 | 0.0 |
| grid order (outcome-free tie-break) against stored plan-1 `grid`, both blocks | 48; 20 | 8.9e-16 | n/a |
| P0 check: stored plan-1 `latent_111` against stored plan-2 `m06_L_r0_111`, per-race rho | 48; 20 | 0.0 | n/a |

The "plan-2 leakage-test reproduction" was read as the plan-2 P0 reproduction check (plan-1 latent models re-run under later code against the stored runs, `report2.py`); it holds to 8.9e-16.
No stored run exists for the S3 re-fit on the test blocks (plan 2 stored S3 with the plan-2 theta on the M06 H3 score), so the S3 numbers are not checked against a stored run. The S3 theta of `latent_111` is
read from the stored R14 re-fit run; the theta of `final` is the one in `frozen_config.yaml`.

### Integrity and re-running
- `freeze_rowhash.json` (written once, while the files still equalled the `FREEZE.md` hashes) stores a row-order-free hash per race (entries2, entries) and per season-round (laps, pits) for all rows with
  race_id <= 271, and the ordered row hashes of the four external tables (rows may be appended, existing rows may not change). `score.py` checks it and `verify_freeze` first and stops on any mismatch.
- With the same data, `score.py` is a no-op (it checks status and race ids, not the models). New fresh races give a new run folder, new `fresh` leaderboard rows (label `p3_R15_<model>_fresh_n<n>`), `results/plan3/fresh_holdout_latest.csv`
  (overwritten each time) and an updated `holdout_status.json`. The readings at n >= 20 and n >= 40 are written once to `results/plan3/holdout_verdict_n20.json` and `_n40.json` (exclusive create, never overwritten);
  a reading uses the first 20 (40) races in date order, T1-T3 with Holm over the three tests (percentile interval at 1 - 0.05/(3 - rank + 1), stop at the first failure).
- Leaderboard: rows appended for `final`, `reverse_only`, `final_minus_reverse` and the two S3 re-fits on each old block (10 rows, label `p3_R15_<model>_<split>`). `latent_111`, `plan2_H2`, grid and H1 already
  have third-look rows from plans 1 and 2 (stored, reproduced exactly above) and were not appended again.
- Dry run of the fresh path (not saved): the pipeline was run with the freeze boundary moved to 2026 R8 and to 2025 R24 (12 and 20 "fresh" races, which are old races); at 20 races it produced a Holm reading and at 12 "no verdict (n < 20)".
  At 20 races (2026 R1-20) it reproduced the third-look test_shift numbers exactly. Nothing from these runs was written to the repository.

### Main-session decisions taken 2026-11-18, before any holdout race (applied in this version of `score.py`)
1. **Holm ordering.** The hypotheses are one-sided ("in the stated direction"). T1-T3 are ranked by the one-sided bootstrap p, p = (1 + #{boot gain <= 0}) / (n_boot + 1); ties are broken in the pre-registered order T1, T2, T3.
   Rank k is confirmed iff the lower bound of the two-sided percentile interval at level 1 - 0.05/(3 - k + 1) is above 0 and every earlier rank was confirmed; the first failure stops. A wrong-way test ranks last and cannot block a right-way one.
2. **Constants are hard-coded** in `src/pitlane/plan3/holdout.py` (alpha 0.05, levels 20 and 40, 10,000 resamples, seed 0, the eight FREEZE.md data files); `score.py` refuses a config that disagrees. Only the freeze boundary and the row-hash file name are read from `config.yaml`.
   `score.py` also verifies the `HOLDOUT_PREREG.md` hash against `FREEZE.md`, and records the `freeze_rowhash.json` sha256 in `metrics.json` and `holdout_status.json` (a later run stops if it changed).
3. **A level-t reading uses only data up to the t-th holdout race** (walk-forward on the rows through that race), so it does not depend on when the script is run (tested: the n20 reading is identical at n = 20 and n = 23).
4. **Verdicts.** `--no-write` is refused when a reading level is due and its verdict file does not exist. Each verdict file stores the row digests of the holdout rows it used, checked on every later run, and the commit hash.
   Writing a verdict is refused when the git tree is dirty for `src/`, `methodologies/P3_final/` (outside `results/`) or `configs/`.
5. Entries are asserted sorted by (race_id, pos) as the harness requires; the CSV integrity message is corrected (all columns are hashed).
6. Stored reference runs (plan-1 and plan-2 leaderboards, the `latent_111` S3 theta) are pinned to explicit run_ids.
7. Truncation: removing every row after race r from the harness path moves scores by up to 1.2e-3 (`reverse_only`), with the order unchanged in all tested races (tester); the first draft of this section said "about 2e-4" for the final model only.
