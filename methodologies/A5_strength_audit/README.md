# A5 — Strength audit

**Status:** done (R1), revised after a referee review (every number below is from the re-run of 2026-11-16, run folders `20261116-2010_*`)
**Plan reference:** PLAN3.md §4 "A5", §6 row R1
**Factors:** n/a — audit of the frozen M03 cell `110` (dynamic on, decomposed on, Gaussian) and the history-only H0 cell `110`

## Hypothesis
The explainability claim needs believable strengths: every team shown, intervals attached, one-race drivers not at
the extremes. Against: the extremes are real signal, or the fix costs prediction accuracy.

## Method
- Two tables, **do not mix them up**:
  - `quali_in` = frozen plan-1 `M03/110` (qualifying position and grid shift are in the model). Its `a` is a driver's
    **race-day gain beyond the grid slot**, not overall skill.
  - `h0` = frozen plan-2 `H0/110` (history only, no weekend signals). Its `a` carries **overall skill**.
- **The MAIN driver table and figure are the models as frozen** (`drivers_final_fit.csv` rows `quali_in`, `h0`;
  `figures/driver_strength_{quali_in,h0}.png`), with 95% intervals; drivers with fewer than 10 starts are in a separate panel
  (A5 item 2). The `*_lam_a` tables and `figures/*_lam_a_sensitivity.png` are a **labelled sensitivity to the driver
  penalty, not the model's explanation**: the predictions do not use `lam_a`.
- Fits: the same model and hyper-parameters as the frozen configs, but converged (`maxiter` 5000, gradient max 5e-8; the
  walk-forward fits stop at 150 iterations, max parameter difference 2.3e-4). One fit per season, through that season's
  last race (`strengths.csv`, "snapshot") plus one through the last race (2026 round 20) (`drivers_final_fit.csv`,
  `cars_final_fit.csv`, "final fit").
  **What the snapshot ranks are:** leak-free (each uses only races up to that season's end) but in-sample descriptions of what
  the model believed at that season's end. `a` is static with a 50-race half-life, so a snapshot blends 2-3 seasons; it is not
  that season's skill.
- **Laplace intervals**: covariance = inverse of the Hessian of the unnormalised penalised objective at the optimum
  (negative log-likelihood + ridge terms), interval = estimate +/- 1.96 se. Hessian by central differences of the analytic
  gradient (all 254 parameters, <0.1 s). Checks: (i) equals the closed-form Gaussian Hessian J'WJ/sigma^2 + penalty,
  max abs difference 2.0e-9 (max entry 2134); (ii) on a 117-parameter problem (first 50 races) it matches second
  differences of the objective value, relative max difference 1.9e-8 (Gaussian) and 2.5e-7 (Plackett-Luce); (iii)
  smallest eigenvalue 16.1 > 0. Caveats: sigma is the model's fixed 0.25, not estimated; the time-decay weights make the
  likelihood a power likelihood; so these are model-conditional intervals, not verified frequentist coverage.
  Marginal intervals on `a` include the uncertainty of the shared level (a driver/car trade-off), so they are wide.
  **Differences between two strengths are not better determined than the marginals**: for the 26 drivers with >= 30 starts
  active 2024-26 the mean pairwise se of a difference is 0.078 against a mean marginal se of 0.058 (ratio 1.36, close to
  sqrt(2) = 1.41; `resolved_pairs.csv`). The referee's estimate (unverified, not tested here) is that decay weighting may make
  the Laplace se conservative, by about a factor 0.7 if true skill is stable.
- **Provenance**: the run configs and `a5_summary.json` record `git_commit: aac399b+dirty-<sha1 of the files used>`
  (`code_stamp()` in `run.py`). The R1 code was uncommitted when the runs were made; the 12-hex digest is a sha1 over
  `audit.py`, `latent.py`, `registry.py`, `run.py` and `config.yaml` as they were. The stamp replaces the plain `aac399b` that
  the harness writes.
- **Shrinkage fix (a sensitivity switch)**: new switch `lam_a` in `LatentModel` (ridge on driver effects only). `None`
  (default) = `lam`, i.e. the old behaviour; verified bit-identical `theta` for cells 000, 010, 110, 111 against the HEAD
  version of the file, and the dev runs with the switch off reproduce the leaderboard (|diff| 0 for Spearman and MAE, 6e-17 for log-loss).
- Units: for the Gaussian cells `a`, `b` are in units of normalised finishing position (0..1), not log-odds as the
  plan-1 fig6 axis says.

## Shrinkage cause (item 3)
Not a bug, not centring (mean `a` = -0.003), not time decay (the two drivers' single races carry weight 0.97-0.99), and
there is no random-walk start value (`a` is static). It is the **penalty**: `lam = 32` on `a` is a Gaussian prior with sd
`1/sqrt(2*lam) = 0.125`, tuned for the car terms and the ranking metric, while the driver effects the data supports
spread only 0.028 (fitted) or 0.038 (unpenalised) among drivers with >= 30 starts. A driver's data weight is
`w/(w + 2*lam*sigma^2)`: 0.95 for Verstappen (69 effective starts) but 0.20 for a driver with one recent start
(rafael_camara -0.0633, ryo_hirakawa +0.0588; their unpenalised values are -0.32 and +0.30, i.e. one noisy race).
Those 0.06 are as large as the whole spread among established drivers, so they land at the extremes of the as-frozen table.
The first-order condition reproduces every `a` (max error 2e-7, `shrinkage_diagnosis.csv`), which rules out a coding error.
Why the plan-1 tuning missed it: dev Spearman barely depends on `lam_a`, so nothing penalised it.

### The `lam_a` sensitivity: how it was chosen and what it is
Tuned on dev 2021-23 only: `lam_a` in {64,...,4096} plus the old value, per model; selection history in `config.yaml`.
- Rule 1 (written before the grid was scored: largest `lam_a` whose paired Spearman difference to the dev-best contains 0)
  picked the grid edge 4096 for both models, where the spread of `a` is 0.002 (a flat table).
- Rule 2 (used) was chosen **after seeing rule 1's result**. It is empirical Bayes on the unpenalised `a` of drivers with
  >= 30 starts in the fit through 2023, `lam_a = 1/(2 sd^2)` = 272 (`quali_in`), 111 (`h0`), snapped to the grid: **256 and 128**.
  In effect it picks a `lam_a` for which the acceptance check passes with a stated, not tuned, rule. No data after 2023 entered it
  (test `test_eb_sd_uses_only_data_through_2023`).
- **The EB input ignores sampling noise**, so it overstates the spread of true skill and understates `lam_a`. With the
  noise-corrected variance (next section, `driver_variance.csv`, final fit through the last race) the empirical-Bayes value is
  `lam_a` = 985 (sigma 0.25) / 758 (sigma 0.229, dev out-of-sample residual sd) / 587 (sigma 0.20) for `quali_in`, i.e. 2-4 times the used 256, and
  124 / 122 / 115 for `h0`, about the used 128. (These come from the final fit; the 2023-fit inputs behind the used values would be
  somewhat different. Not re-derived on the 2023 fit.)
- **`lam_a` is NOT adopted for prediction.** Whether it should be (and at what value) is to be tested later as one
  attribution component under PLAN3 §5; not run here.

Dev numbers (tuning grid: refit every 4 races, as in plan 1; the run folders: refit every race):

| model | lam_a | dev rho (refit 4) | paired d-rho vs old [95% CI] (refit 4) | dev rho (refit 1) |
|---|---|---|---|---|
| quali_in | 32 (frozen) | 0.6522 | 0 | 0.6510 |
| quali_in | 64 (dev-best of 8) | 0.6550 | +0.0027 [+0.0003, +0.0054] | |
| quali_in | 256 (used) | 0.6547 | +0.0025 [-0.0013, +0.0065] | 0.6507 |
| h0 | 32 (frozen) | 0.6005 | 0 | 0.6053 |
| h0 | 64 (dev-best of 8) | 0.6011 | +0.0006 [-0.0027, +0.0040] | |
| h0 | 128 (used) | 0.5989 | -0.0016 [-0.0070, +0.0038] | 0.6065 |

`lam_a` = 64 for `quali_in` is the best of 8 values and its interval excludes 0. Across the 14 comparisons against the old value
(7 grid values x 2 models) exactly one interval excludes 0 (this one), where about 0.7 are expected by chance; it is not evidence of a
gain, and it is not "within the noise" in the sense of an interval containing 0. The used values (256, 128) have intervals that
contain 0 and are dev-indistinguishable from the old value and from the dev-best.

Win-probability log-loss gets **worse** with larger `lam_a`, monotonically in both models. Refit every 4 (tuning grid, `tuning_lam_a.csv`):
`quali_in` 0.1539 -> 0.1553 at 256, `h0` 0.1616 -> 0.1625 at 128. Refit every race (run folders, `paired_dev_refit1.csv`): `quali_in`
0.1544 -> 0.1551, paired difference +0.00077 [+0.00033, +0.00121]; `h0` 0.1598 -> 0.1610, +0.00119 [+0.00079, +0.00158] (intervals
exclude 0: prediction mildly prefers *less* shrinkage of the driver term; the Spearman differences at refit 1 are
-0.0003 [-0.0044, +0.0039] and +0.0013 [-0.0042, +0.0067], i.e. null). The display argument for more shrinkage and the
log-loss argument against it are both real.

## Between-driver variance of the true `a` (referee correction)
An earlier version of this card said the quali-in `a` has about zero true between-driver variance. That subtracted
sigma^2 / sum(w), the noise of an unweighted mean over the effective sample, which is too small for a decay-weighted mean. The noise of
a weighted mean of residuals is sigma^2 * sum(w^2) / (sum(w))^2 (`A.driver_variance`). Method of moments, tau^2 = var(unpenalised `a`) - mean noise,
drivers with >= 30 starts (43 drivers), final fit; the noise ignores the car separation (a lower bound on the sampling noise), so tau is
an upper bound on the true spread:

| model | sigma used for the noise | var(unpenalised a) | mean noise | tau^2 | tau |
|---|---|---|---|---|---|
| quali_in | 0.25 (model) | 0.00146 | 0.00096 | 0.00051 | 0.023 |
| quali_in | 0.229 (dev out-of-sample residual sd) | 0.00146 | 0.00080 | 0.00066 | 0.026 |
| quali_in | 0.20 (referee sensitivity) | 0.00146 | 0.00061 | 0.00085 | 0.029 |
| h0 | 0.25 (model) | 0.00497 | 0.00096 | 0.00402 | 0.063 |
| h0 | 0.240 (dev out-of-sample residual sd) | 0.00497 | 0.00088 | 0.00409 | 0.064 |
| h0 | 0.20 (referee sensitivity) | 0.00497 | 0.00061 | 0.00436 | 0.066 |

The referee's figures (0.00146, 0.00096, tau 0.023, 0.029 at sigma 0.20; h0 0.00497, tau 0.063) are reproduced. The out-of-sample residual sd
is sd(y - (0.5 - score)) on the dev predictions of the repro runs. Honest statement: the quali-in `a` is **small** (sd about 0.02-0.03 in normalised
finishing position), not zero; **h0 carries most of the driver skill** (tau 0.063, 2.5-3 times larger).

Resolved differences (`resolved_pairs.csv`, `driver_vs_field.csv`; 26 drivers with >= 30 starts active 2024-26, 325 pairs, full Laplace
covariance): the share of pairs with |diff| > 1.96 se of the difference is **0.6%** (2 of 325) for `quali_in` and **10.5%** (34 of 325) for `h0`
(referee: about 1% and 10%). Against the group mean, only Verstappen is distinguishable in `quali_in` (+0.099); in `h0` Verstappen (+0.195) and
Tsunoda (-0.096).

## Leakage notes
Snapshot fits for season s use only races up to the end of s (decay reference = last race + 1). The tables for 2024-26 are
descriptive; tuning and the dev runs use 2021-23 only. The in-sample correlations are a face-validity check, not a prediction;
the out-of-sample one below uses only the previous season's snapshot.

## How to run
```
python -u methodologies/A5_strength_audit/run.py --config methodologies/A5_strength_audit/config.yaml   # ~5 min; --quick for a smoke test
```
Outputs in `results/plan3/`: `strengths.csv` (models `quali_in`, `quali_in_lam_a`, `h0`, `h0_lam_a`; every driver and car
for every season, est/se/lo/hi, `n_season`, `n_total`, `low_sample`, `rank`, `rank_main`), `drivers_final_fit.csv`,
`cars_final_fit.csv`, `champion_ranks.csv`, `correlations_per_season.csv`, `teammate_oos.csv`, `identification.csv`,
`shrinkage_diagnosis.csv`, `driver_variance.csv`, `resolved_pairs.csv`, `driver_vs_field.csv`, `paired_dev_refit1.csv`,
`tuning_lam_a.csv` (also in this folder's `results/`), `acceptance_*.csv`, `a5_summary.json`, `leaderboard.csv`
(one row per run), `figures/`.

## Results
| Run ID | Split | Spearman rho | MAE | Win log-loss | Notes |
|--------|-------|-----------|-----|---|-------|
| 20261116-2010_a5_quali_in_repro_lam_a_off_dev | dev | 0.651014 | 3.355 | 0.15437 | reproduces leaderboard `latent_110` dev exactly (|diff| 0, logloss 6e-17) |
| 20261116-2010_a5_h0_repro_lam_a_off_dev | dev | 0.605252 | 3.690 | 0.15984 | reproduces plan-2 `h0_latent_110` dev exactly |
| 20261116-2010_a5_quali_in_lam_a_256_dev | dev | 0.650737 | 3.344 | 0.15513 | sensitivity variant, not adopted |
| 20261116-2010_a5_h0_lam_a_128_dev | dev | 0.606505 | 3.679 | 0.16103 | sensitivity variant, not adopted |

### Acceptance (R1: no driver with fewer than 10 starts in the top or bottom 3 without a stated reason)
Under the final layout (`acceptance_main_panel.csv`) the main panel holds drivers active 2024-26 with >= 10 starts and the others are listed
separately. **Main panel, as frozen: 0 of 6 top/bottom slots hold a driver with < 10 starts**, for both models (`quali_in` bottom 3 bottas, tsunoda,
sargeant, top 3 norris, hamilton, verstappen; `h0` bottom 3 sargeant, tsunoda, stroll, top 3 leclerc, norris, verstappen). This holds by
construction of the layout, so the check is passed trivially; the stated reason is that the penalty, not the data, put them there. The
as-frozen `quali_in` ranking *without* the separation still has rafael_camara (bottom 3) and ryo_hirakawa (top 3) in 2 of 6 slots
(`acceptance_summary.csv`); `h0` as frozen 0 of 6; the sensitivity variants 0 of 6. Season-end snapshots, all drivers racing that season (78
slots per model): `quali_in` 5 (hartley 2017, resta 2017, colapinto 2024 with 9 starts, camara and hirakawa 2026), `quali_in_lam_a` 0, `h0` 3
(hartley and resta 2017, pietro_fittipaldi 2020), `h0_lam_a` 0. The separate panel (< 10 starts, 2024-26): rafael_camara, doohan, ryo_hirakawa.

### Named checks (`champion_ranks.csv`; rank among the drivers racing that season, 1 = best `a`, snapshot at season end)
2026 is the points leader after 20 rounds, not a champion. Verstappen:

| season | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| `quali_in` (as frozen) | 2 | 1 | 1 | 1 | 1 | 1 |
| `h0` (as frozen) | 2 | 1 | 1 | 1 | 1 | 1 |

(Verstappen is not the champion in 2025-26; the champion/leader ranks are in the csv: `quali_in` 5 and 16 for Norris 2025 and Antonelli 2026,
`h0` 3 and 17.) Hamilton is rank 1 in 2014-15 and 2017-20 in `h0`, Rosberg 2nd in 2016 in `h0` (5th in `quali_in`). The champion's teammate
sits mid-table (4th-10th) in `quali_in` and 1st-4th for the Mercedes years in `h0`; Perez falls to rank 18 (`quali_in`) and 20 (`h0`) of 24 in 2024.

**Antonelli 2026 is a face-validity failure of the static driver term.** He has 44 starts (32.7 effective), not little history. `h0` ranks him 17th of
25 (a = -0.004) and below his teammate Russell (a = +0.031, rank 10), although he outscored him 359-228 points in 2026. Cause: `a` is static with a
50-race half-life, so his 2025 season dominates what the model believes about him. The `quali_in` table has him at 16 (a = -0.001) next to Russell 17.

**Teammate and car correlations** (plan-1 definition, pairs with >= 8 starts each). Pooled, final fit, in-sample: `quali_in` **+0.54** (129 pairs,
bootstrap CI [0.36, 0.68]) and **+0.92** (133 constructor-seasons), i.e. plan 1 reproduced; `h0` +0.62 and +0.95. Per season, ~10 pairs each, so wide.
Two bases, both in `correlations_per_season.csv`:
- *Snapshot basis* (a and b as estimated at the end of that same season): teammate correlation 0.64-0.94 in every season for `quali_in` (2014-17: 0.90, 0.92, 0.74,
  0.92) and 0.54-1.00 for `h0` (2014-17: 1.00, 0.91, 0.80, 0.97); car correlation 0.93-0.99 / 0.97-1.00.
- *Final-fit basis* (the model through 2026 evaluated on every season): `quali_in` -0.52 (2014), 0.13, -0.26, 0.09 (2017), ..., 0.91 (2021), 0.80 (2025);
  `h0` -0.11, 0.05, -0.17, 0.24 for 2014-17. These low values are an **artefact of a static `a` plus decay**: in the final fit the 2014-17 data of those
  drivers is decayed to about 2-3 effective starts, so their `a` is shrunk and says little about the seasons being scored. They do not show seasons where the driver term is
  unrelated to the teammate gap. On the snapshot basis the same file gives 0.74-0.92 and 0.80-1.00.

**Out-of-sample teammate check** (`teammate_oos.csv`; descriptive, no decision uses it). For each season S >= 2015, the difference in snapshot `a` at the end of S-1 between two
teammates against their mean finishing-position gap in S (pairs of season S with >= 8 starts each, both with a snapshot value at S-1; 92 pairs pooled, 5-9 per season). Pooled r with a
bootstrap CI over pairs:

| model | pooled r | 95% CI |
|---|---|---|
| quali_in | +0.28 | [0.01, 0.49] |
| quali_in_lam_a (sensitivity) | +0.32 | [0.06, 0.52] |
| h0 | +0.39 | [0.14, 0.59] |
| h0_lam_a (sensitivity) | +0.43 | [0.18, 0.61] |

Per season (`h0`, as frozen): 2015 0.14, 2016 0.00, 2017 0.83, 2018 0.26, 2019 0.51, 2020 0.88, 2021 0.55, 2022 -0.09, 2023 -0.45, 2024 0.53, 2025 0.82, 2026 0.05; `quali_in` is
negative in 2016, 2022, 2023 and 2026 (-0.03, -0.32, -0.63, -0.20). With 5-9 pairs per season, these are wide. The out-of-sample correlations are much lower than the in-sample +0.54 / +0.62
(the in-sample one is partly mechanical: the fit used those same finishing positions) and the intervals of the as-frozen and `lam_a` variants overlap; the sensitivity
variants are not worse here, unlike in the in-sample check.

**In-sample correlation under `lam_a`** drops (0.54 -> 0.475 for `quali_in`, 0.615 -> 0.583 for `h0`, final-fit basis). This is in-sample with overlapping intervals and partly mechanical
(less freedom to fit the same positions), so it is weak evidence; the out-of-sample table above shows no such drop.

**Identification** (`identification.csv`): 65 drivers, 21 flagged (fewer than 10 starts: 8; exactly one teammate over the whole
career: 13, including piastri, antonelli, bortoleto and rosberg; two drivers, bianchi and chilton, sit in a team-season
component of their own, disconnected from the rest). A driver who never leaves a team is separated from the car only through
that single teammate, so their `a` is really "better or worse than that teammate". The Laplace se of `a` for these (0.07-0.12)
is higher than the roughly 0.05 of established multi-team drivers, which says the same thing.

## What we learned
- All 13 lineages are now shown (`figures/car_strength_*.png`, 4x4 grid; single-season lineages such as cadillac and caterham are drawn with an error bar, since a one-point band is invisible). The
  plan-1 fig6 omitted five because of the `nlargest(8)` filter; it was a plotting bug only (the model fits all 13).
- The one-race extremes were a penalty effect, so a single noisy race was displayed as ability.
- `quali_in` `a` is small (tau about 0.02-0.03) and `h0` `a` larger (0.063); the `h0` table is the one to read for skill. It ranks Verstappen 1st from 2022 and
  Hamilton 1st in most 2014-2020 seasons, but fails face validity on Antonelli 2026 (above).
- Intervals are wide (se of `a` about 0.058 for established drivers, 0.07-0.12 otherwise, against a between-driver sd of 0.02-0.06): only 0.6% (`quali_in`) and 10.5% (`h0`) of
  driver pairs among established current drivers are resolved.
- Win log-loss prefers the old penalty; Spearman is flat. Display and prediction pull in opposite directions on `lam_a`.

## Open questions / consequences for plan 3
(PLAN3.md not edited here.)
- **Driver-car separation, per model.** `quali_in` `a` is a gain beyond the grid slot (the car and the qualifying performance are in the grid term), tau about 0.02-0.03, and is hardly
  resolved; `h0` `a` is overall skill (tau 0.063), separated from the car only through teammates and team changes. Any attribution of "driver versus car" has to be stated per model.
- **`lam_a` as an attribution component** (PLAN3 §5), tested there against the as-frozen penalty on prediction, not adopted from this audit. The noise-corrected EB value (about 590-985 for `quali_in`,
  about 115-125 for `h0`) is the natural starting point; the display benefit is measured here, the prediction cost (log-loss) is not "noise" (paired intervals exclude 0).
- **M15 rookie prior** to be tuned together with `lam_a` (both govern low-count drivers) but attributed separately.
- **S6 depends strongly on lambda**: mean Laplace se of `a` (26 established current drivers) falls from 0.058 to 0.034 in `quali_in` (referee: 0.055 to 0.033) and from 0.058 to 0.042 in `h0` when `lam_a` rises (as-frozen vs sensitivity), so the width of a parameter-uncertainty draw in the simulator follows a penalty choice
  that prediction metrics barely see.
- **A per-season or random-walk driver term** as a candidate component: a static `a` makes the Antonelli 2026 failure and the final-fit artefact above unavoidable; "who was better in 2016" cannot be asked.
- Rank intervals (posterior draws from the Laplace covariance) rather than point ranks.
