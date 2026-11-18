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
