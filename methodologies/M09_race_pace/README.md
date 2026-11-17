# M09 — Race-pace model (exploratory, post-gate)

**Status:** done (R6, dev only). Exploratory: counted only if the fresh holdout confirms it.
**Plan reference:** PLAN3.md Amendment 3 item 6, §4 M09, §5
**Factors:** latent_111 (F1 on, F2 on, F3 on) and gbm_100 (F1 on with decayed features at half-life 32, F2 off, F3 off) plus the columns below in the weekend term / as extra columns

## Status of the evidence
The A3 gate failed as declared (car persistence 0.096 < 0.2); Q2 stays answered as a null. M09 was run anyway under Amendment 3
item 6, with the reference definition chosen after seeing the gate fail. A dev pass is therefore weaker than a declared one.

## Corrections after skeptic review (read first)
Task A (car-form control, below) and the skeptic's checks change how the headline reads. The +0.016 rho of latent_111 + `pace_hist_x` is
**not shown to be race pace**. It is a car-form proxy: within-race rank correlation of `pace_hist_x` with the decayed finishing form
`con_ew32` is 0.963 (0.969 at half-life 50), `q1off_x` 0.865-0.923; in my control `pace_hist_x` vs `form_x` is 0.969. A column that
only carries finishing form gives +0.0138 rho [+0.0027, +0.0245] (C1), and race pace on top of it adds +0.0012 [-0.0002, +0.0028]
(C2 vs C1, interval includes 0). The gain above the amendment's expected 0.00-0.01 is a warning sign, not extra strength.
* Rung 2 (`hist + Q1 offset`) is a bundle, not the offset's own effect. The offset's increment over history is +0.0006 [-0.0012, +0.0023], null.
  The `attribution.csv` row `M09_car_q1_offset, latent_111` saying "passes alone" is wrong as a reading of the offset; the file is append-only and
  keeps it, this card corrects it.
* The off-ladder `q1off_alone` row (+0.0057 [+0.0006, +0.0111]) is an extra comparison not in Amendment 3. It passes only uncorrected; a
  6-row correction (99.17% intervals) gives [-0.0012, +0.0131]. The main hist pass survives the same correction ([+0.0031, +0.0291]), and it is
  also a car-form proxy.
* The latent_111 + history gain does not fix the Mercedes-2022 defect; it makes it worse (Mercedes predicted winner 17 -> 19 of 22 races in
  2022). Removing Mercedes from the comparison gives +0.020 [+0.009, +0.032]. (Skeptic numbers, not recomputed here.)
* gbm_100 null: the likely reason is redundancy, not "trees split on noise": gbm_100 already carries `con_ew32` (`gbm_feature_cols`), with
  which the column has 0.963 rank correlation. I found nothing contradicting this.
* The history runs through the 2022 regulation change with no reset and leans on 2020-21 cars, a wrong prior for a regime change.
* Half-life 50 is the primary half-life of A3, not stated in Amendment 3 item 6; the comment in `config.yaml` attributes it to item 6. I left that
  comment as it is (editing it changes the code stamp of the stored runs) and note it here.

## Hypothesis (stated before the run)
Adding a car's earlier race pace to the weekend term moves the predicted order toward finishing order. Expected gain 0.00-0.01 rho.
Failure: the paired dev interval includes 0. The Q1 offset adds little beyond the race-pace history, since A3's sensitivity found
the offset's persistence is mostly the car's earlier race pace.

## Method
D1 pace (median clean lap, `src/pitlane/features/pace.py`, saved table `results/plan3/a3_pace_table_2014_2023.parquet`, laps of 2014-2023 only).
Both columns are per entry, from the car (lineage) history, decayed mean with half-life 50 (races, race_id as time, prior 0, one
pseudo-observation, the plan-1 history machinery) over EARLIER races only, divided by a fixed constant (SD over seasons <= 2020).
* `pace_hist_x`: car's field-relative race-pace gap (race gap % minus the race median), mean over the car's drivers.
* `q1off_x`: car's field-relative offset `race gap % - Q1 gap %` (same construction as `rq_c`), Q1 gap missing left missing (the
  plan-1 0.10 imputation in `entries.q1_gap` is NOT used; `q1_missing == 1` entries give no observation).
Ladder: reference -> + `pace_hist_x` -> + `pace_hist_x` + `q1off_x`. Off-ladder diagnostic (declared "no"): `q1off_x` alone. No
driver rung and no stickiness rung. Families: frozen `latent_111` (`x_cols`, ridge `lam_x`) and frozen `gbm_100` (`extra_cols`);
plan-1 hyper-parameters unchanged; `latent.py` and `gbm.py` unchanged.
**Missing values (decided in the config before the run):** a column is NaN for a car with no earlier valid observation. LatentModel
fills NaN with 0.0 (= the centred value, since prior 0 and field-relative); LightGBM takes NaN natively. In dev no entry was missing
(share 0.0).
**Tuning:** `lam_x` in {0.3, 1, 3, 10} per rung, latent only, dev mean rho at refit_every=4, then final runs at refit_every=1.
Configurations for M09: 12 latent + 3 LightGBM (no ridge to tune) = 15, budget 30. Table: `results/tuning_lam_x.csv`. Chosen:
hist 3.0, hist+q1off 0.3, q1off alone 3.0 (the differences across lam_x are about 0.001 rho; the choice is dev-optimistic by that much).
LightGBM and the references are not tuned.

## Leakage notes
Features of race r use rows with race_id < r only. In-run truncation check: blank the pace rows and the Q1 times of race 171 and
later and recompute: max abs difference of both columns for all races <= 171 is 0.0 (`results/feature_info.json`); the run asserts it.
The scale constant comes from seasons <= 2020. No 2024+ data read.

## How to run
```
python methodologies/M09_race_pace/run.py [--quick]
```

## Results (dev 2021-2023, 66 races; 10,000 race resamples, seed 0, driverId tie-break; gain > 0 = better than the reference)
References re-run in the same harness reproduce the stored plan-1 dev runs: max |per-race metric difference| 8.9e-16 (latent_111),
4.4e-16 (gbm_100); max |score difference| 0 (66 races, 1311 entries).

| Component | Family | rho gain [95% CI] | rho on finishers [95% CI] | MAE gain [CI] | exact gain [CI] | verdict (rho) |
|---|---|---|---|---|---|---|
| M09_car_pace_hist | latent_111 | +0.0162 [+0.0065, +0.0258] | +0.0138 [+0.0038, +0.0241] | +0.042 [-0.023, +0.106] | -0.0130 [-0.0283, +0.0015] | passes alone (dev) |
| M09_car_pace_hist | gbm_100 | -0.0020 [-0.0091, +0.0048] | -0.0008 [-0.0081, +0.0070] | -0.019 [-0.065, +0.030] | -0.0018 [-0.0169, +0.0135] | no measurable difference |
| M09_car_q1_offset (ladder rung 2, hist + offset) | latent_111 | +0.0168 [+0.0068, +0.0267] | +0.0143 [+0.0045, +0.0244] | +0.045 [-0.022, +0.110] | -0.0107 [-0.0251, +0.0029] | passes alone (dev) |
| M09_car_q1_offset | gbm_100 | +0.0043 [-0.0020, +0.0109] | +0.0039 [-0.0031, +0.0112] | +0.007 [-0.041, +0.056] | -0.0077 [-0.0220, +0.0069] | no measurable difference |
| M09_car_q1_offset_alone (off-ladder) | latent_111 | +0.0057 [+0.0006, +0.0111] | +0.0055 [+0.0004, +0.0108] | +0.013 [-0.023, +0.049] | -0.0031 [-0.0161, +0.0097] | passes alone (dev) |
| M09_car_q1_offset_alone | gbm_100 | +0.0018 [-0.0036, +0.0069] | +0.0009 [-0.0053, +0.0074] | +0.006 [-0.030, +0.041] | -0.0022 [-0.0168, +0.0122] | no measurable difference |

Ladder increment of the Q1 offset over race-pace history: latent +0.0006 rho [-0.0012, +0.0023]; gbm +0.0063 [+0.0001, +0.0128]
(secondary, not an attribution row; a post-hoc pair).
Means (rho): latent_111 0.6460, +hist 0.6622, +hist+off 0.6628; gbm_100 0.6489, +hist 0.6469, +hist+off 0.6532.
Exact rate against grid order (0.1638): reference latent_111 0.1397, gbm_100 0.1420; every M09 variant is below grid order; for latent
with hist the difference is -0.037 [-0.073, -0.005], i.e. worse than grid order on exact rate. M09 does not beat grid order on
exact positions in any variant (`results/exact_vs_grid.csv`). By season, latent hist rho gain: 2021 +0.014, 2022 +0.018,
2023 +0.016 (each interval touches or just excludes 0; `results/paired_gains_by_season.csv`).
Verdict labels come from the declared rule (interval excludes 0); they are dev only, exploratory.

## What we learned
* In the latent family adding the column raises rho by about 0.016 on dev with an interval excluding 0, above the 0.00-0.01
  expected, in the same direction in all three seasons. It is not established that this is race pace: see Task A (car-form control).
  In LightGBM the same column gives nothing, most likely because gbm_100 already has `con_ew32` (redundancy).
* The Q1 offset adds nothing measurable on top of race-pace history in the latent family (+0.0006), consistent with A3's reading that its persistence is the car's race pace.
* Exact-position rate does not improve and is lower than the reference's point value in the latent family (not significant); the rho gain is order-wide, not position-exact.
* Car form alone (finishing-position history) reproduces most of the gain; race pace adds nothing measurable beyond it (see below).
* Evidence is weak by construction: pair found after the gate failed, rung and family chosen after seeing A3, `lam_x` tuned on the same dev races, ~6 attribution rows added in one go. Nothing counts unless the fresh holdout confirms it.

## Task A - car-form control (diagnostic, post hoc, NO decision weight)
Pre-statement written before the runs: `control_prestatement.md`. Column `form_x`: decayed mean (half-life 50, earlier races only, same
lineage map, `min_prior_obs`, prior 0, scaled by SD over seasons <= 2020 = 0.1783) of the car's mean normalised finishing position `y`
centred by the race mean (the quantity behind `con_ew*`; built in `run_control.py`, `src/pitlane` untouched). Truncation check (outcomes of
race 171 and later blanked): max abs diff 0.0. Not an attribution component; no attribution rows. Budget 15 + 8 = 23 <= 30.
Rungs (latent_111, `lam_x` tuned on dev at refit_every=4, final refit_every=1): C1 = reference + `form_x` (lam_x 1.0); C2 = reference +
`form_x` + `pace_hist_x` (lam_x 3.0). Means rho: reference 0.6460, C1 0.6598, hist 0.6622, C2 0.6610. Tuning grid: `results/control_tuning_lam_x.csv`
(differences across lam_x about 0.001).

| Comparison | rho gain [95% CI] | rho finishers | MAE gain | exact gain |
|---|---|---|---|---|
| C1 vs reference | +0.0138 [+0.0027, +0.0245] | +0.0128 [+0.0010, +0.0249] | +0.020 [-0.057, +0.095] | -0.0185 [-0.0348, -0.0026] |
| hist vs C1 | +0.0024 [-0.0006, +0.0055] | +0.0010 [-0.0021, +0.0038] | +0.021 [-0.002, +0.046] | +0.0055 [-0.0051, +0.0169] |
| C2 vs C1 | +0.0012 [-0.0002, +0.0028] | +0.0007 [-0.0007, +0.0022] | +0.006 [-0.006, +0.018] | +0.0000 [-0.0061, +0.0063] |

Pre-stated rule: the hist vs C1 interval on rho includes 0, so the gain of `pace_hist_x` over the reference is **credited to car form, not race
pace**; C2 vs C1 includes 0, so race pace adds no measurable value beyond car form. By season (rho): C1 vs ref +0.017 / +0.012 / +0.012
(2021/22/23; only 2021 excludes 0); hist vs C1 -0.002 / +0.006 [+0.001, +0.011] / +0.004; C2 vs C1 -0.001 / +0.001 / +0.004 [+0.0003, +0.007]
(isolated single-season exclusions, not the declared pooled test; `results/control_paired_gains_by_season.csv`). C1 lowers the exact rate
below the reference, like the hist runs. Caveats: `lam_x` tuned on the same dev races; diagnostic, post hoc. Files:
`results/control_paired_gains_dev.csv`, `control_paired_gains_by_season.csv`, `control_tuning_lam_x.csv`, `control_feature_info.json`
(also within-race rank correlations on dev: form_x vs pace_hist_x 0.969, vs q1off_x 0.898, vs con_ew32 0.988; pace_hist_x vs con_ew32 0.963).
Run folders `*_m09_ctrl_{form,form_hist}_latent_111_dev`; 2 leaderboard rows, labels ending `_diagnostic`.

## Open questions / next tries
* Fresh-holdout check (R-final). Leave-one-out and the M09 x M18 interaction are not run here.
