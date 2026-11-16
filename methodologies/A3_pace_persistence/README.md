# A3 — Race-pace persistence and the M09 gate

**Status:** done (R4); gate FAILED
**Plan reference:** PLAN3.md §3 (pace definitions, `rq`), §4 A3 and M09, §5, §6 R4
**Factors:** n/a — analysis (an oracle that uses race data on purpose, never a model; it is not a candidate for anything)

## Written before the run (2026-11-17; the claim that nothing below "Results" had been computed is NOT verifiable)
Referee correction: this README was last saved after the run, so the timestamps do not prove the text below predates the
results. The only independent evidence is the run's config snapshot, which shows `primary_definition: med` (set before the
run). Treat the expectations and definitions below as stated by the author, not as a verified pre-registration; the
reconstruction in section "Post-hoc sensitivity" at the end is explicitly post hoc.

### Hypothesis, expected result, failure criterion
Plan 2 (22 races of 2023) found finishers ranked by their own race pace beat `latent_111` by 0.073 rho. A3 asks (1) is that gap
still there on all 66 dev races, (2) is a car's race-minus-qualifying offset `rq` persistent from race to race, so that it could
be learned from earlier races, and (3) how much of the gap is persistent car offset, persistent driver offset, or race-specific.

Expected, stated before running (PLAN3 §8: "the most likely single outcome is that the offset is only weakly persistent"):
* Extended oracle gap on finishers: 0.05 to 0.08 rho, so the first gate condition passes. A gap below 0.03 would mean the 2023
  result was a small-sample artefact.
* Car-level persistence of `rq` (pooled correlation over dev car-races, half-life 50): 0.10 to 0.30. This is the uncertain
  half of the gate; my guess is that it lands near the 0.2 threshold, slightly below. Driver-level persistence (raw
  `drv_rq`, which contains the car) a little lower; teammate-relative driver persistence near 0.05 to 0.15.
* Corrected pace: persistence a little higher than the median lap (PLAN3 §8), by less than 0.05.
* Split of the gap: most of it race-specific (not predictable from earlier races); the persistent car and driver parts together
  under a third of the gap and under 0.02 rho.

Failure criteria (what counts against the plan's hope for M09): car-level persistence below 0.2 on dev (gate fails; M09 is
not built, Q2 is answered "not persistent enough", and work moves to M10-M12), or the persistent car part of the gap having a
paired interval that includes 0 and a point value under 0.005 rho. The gate is mechanical: both conditions on point estimates
as in plan 2; intervals are reported beside, not used to decide.

### Definitions fixed before the run (so that they cannot be chosen after seeing results)
* **Laps:** Jolpica laps for all dev races (2021-2023 complete; OpenF1 is not used, lap-1 times are not interchangeable and
  lap 1 is dropped anyway). Pit stops: Jolpica (2021 round 12 is an empty answer: a one-lap race, no pit data, corrected pace
  falls back to the median there). `stints.parquet` (OpenF1, 2023 only, no compound before 2023) is **not used**: stint age is
  derived from pit-stop laps so that it is defined identically in every season. Tyre compound is not modelled.
* **Median clean lap (D1, the plan's base definition, section 3):** drop lap 1, in-laps, out-laps and laps above 107% of the
  race median (median of all laps >= 2 in the race, as in plan 2); at least 10 clean laps; gap in % to the best such median in
  the race. **Corrected pace (D2, the plan's "variant"):** per race, least squares on those clean laps of
  `lap time = driver + f * laps remaining + t * stint age`, driver pace evaluated at the race-average fuel and age; fallback to
  D1 for the race if no pit data, fewer than 8 drivers, or the demeaned fuel and age columns correlate above 0.98; the fallback
  count is reported.
* **Primary definition for the gate: D1.** PLAN3 names no gating definition. It lists the median clean lap as the feature
  (section 3, table row 1) and the corrected fit as "(variant)", and M09 says the persistence numbers are "reported for both".
  I therefore gate on D1 and report the gate for D2 as well. If the two disagree, the report says so and the definition choice
  swings the gate.
* **rq:** `race gap % - qualifying gap %`, qualifying gap = `q_best_gap` (best of Q1/Q2/Q3 against pole, existing plan-2 field).
  For persistence the offset is **field-relative** (`rq_c` = rq minus the median rq of that race): the raw rq contains a
  race-wide constant that cannot be persistent and does not matter for ordering cars. Raw `rq` is stored in the table.
* **Car level:** mean of the car's drivers' `rq_c` in the race, per car lineage. **Driver level:** the driver's own `rq_c`
  (contains the car part; this is the plan's `drv_rq`). Extra, descriptive: **teammate-relative** driver offset.
* **History features:** shrunk decayed mean over earlier races only (one pseudo-observation of prior 0, race ordinal as time),
  same code as plan-1 history features. **Primary half-life 50** (the frozen `latent_111` value, M09 is built on the latent
  model); 32 (frozen `gbm_100`), 16 and 8 reported. Rows enter the persistence correlation only with at least 5 earlier valid
  observations of that car or driver.
* **Persistence statistic:** pooled Pearson correlation over all dev car-races (and by season), 95% CI by resampling races
  (10,000, seed 0). Gate: pooled dev correlation >= 0.2.
* **Oracle gap (gate part 1):** mean over dev races of rho_fin(oracle) - rho_fin(model), classified finishers with a pace value
  (>= 8 per race), oracle = finishers ranked by their own race pace, model = `latent_111` plan-1 dev run, both on the same
  subset of drivers. Gate: >= 0.03. The plan-2 convention (model rho_fin over all classified finishers, plan-2 lap filter with
  pit laps kept and >= 5 laps) is also computed and reported; its 2023 value is checked against `results/plan2/ceiling.json`.
* **Split of the gap** (all on the same finishers; pace-unit predictors, nothing fitted, coefficient 1 on every offset):
  `q` = qualifying gap; `Pc` = `con_rq` (car, earlier races); `Pd` = teammate-relative driver offset (earlier races);
  `C` = the car's actual rq_c in this race; `D` = the driver's actual deviation from the car's rq_c.
  rho of `q`, `q+Pc`, `q+Pc+Pd`, `q+C+Pd`, `q+C+D` (the last equals the oracle). Terms: model-vs-q, +persistent car, +persistent
  driver, +race-specific car, +race-specific driver. Order matters for the attribution of overlapping terms; stated, not tuned.
* 107% threshold sensitivity (1.05, 1.10) reported for the oracle gap; PLAN3 section 7 asks for it.
* Dev only (2021-2023 scored). Burn-in 2014-2020 laps feed the history means and nothing else. No 2024-2026 data is read.

## Method
See the definitions above; code in `src/pitlane/features/pace.py` (per-race pace table, `rq`, history features) and
`src/pitlane/plan3/pace_gate.py` (oracle, pooled correlation, bootstrap). Orchestration in `run.py`.

## Leakage notes
The pace table of a race uses only that race's laps and stops. The history features for race r use only rows with
`race_id < r`; a test-ready check is to blank a race's laps (and result) and confirm that race's `con_rq_*`, `drv_rq_*`,
`drvtm_rq_*` do not change (the tester writes it). The oracle and the pace of the race itself are outcomes: they are targets
and oracles here, never inputs for the same race.

## How to run
```
python methodologies/A3_pace_persistence/run.py --config methodologies/A3_pace_persistence/config.yaml
```

## Results
Run `20261117-0131_a3_pace_oracle_dev` (folder `results/20261117-0131_a3_pace_oracle_dev/`; code stamp
`c3b4fbc+dirty-815ab92f2a1b`, i.e. the working tree with the new files, not a commit). Dev 2021-2023, 66 races; 65 enter the
oracle (2021 round 12, a one-lap race, has no driver with 10 clean laps). All CIs: 10,000 resamples of races, seed 0.

Reproduction check: the plan-2 oracle code path re-run on the 22 races of 2023 matches `results/plan2/ceiling_per_race.csv` to
1.1e-16 and gives gap 0.0733, as in plan 2.

**1. Extended pace oracle (finishers, rho; model = plan-1 `latent_111`, dev)**

| Quantity | Mean | 95% CI | Races |
|---|---|---|---|
| Plan-2 convention: oracle (median lap, plan-2 filter) minus model rho_fin over all classified finishers | **0.0724** | [0.038, 0.106] | 65 |
| D1 median clean lap, matched drivers: oracle 0.842, model 0.767, **gap** | **0.0750** | [0.042, 0.109] | 65 |
| D2 corrected pace, matched drivers: oracle 0.886, model 0.767, **gap** | 0.1189 | [0.092, 0.146] | 65 |
| D1 gap with the lap filter at 105% / 110% | 0.0739 / 0.0755 | [0.042, 0.106] / [0.040, 0.110] | 65 |

By season (D1 gap): 2021 0.057, 2022 0.091, 2023 0.076. The gap is not an artefact of the 2023 sample. The D2 oracle is higher
because the fit gives each driver a pace free of tyre age and fuel; it is an oracle, not evidence that the correction is better
for prediction. Fits: 203 of 204 races in 2014-2023 fitted, 1 fallback (2021 round 12, no pit data). In 2021-2023 the fuel
coefficient had the expected sign in 98.5% of races (median 0.054 s per lap of fuel) and the tyre coefficient in 89.4% (median
0.040 s per lap of age).

**2. Persistence of the race-minus-qualifying offset** (field-relative `rq`, car lineage and driver, decayed mean of earlier
races, rows with at least 5 earlier observations; pooled Pearson r, 95% CI over races)

| Level, definition, half-life | Dev pooled | 2021 | 2022 | 2023 |
|---|---|---|---|---|
| **Car, D1 median, 50 (gate)** | **0.096** [0.016, 0.182] | -0.000 | 0.095 | 0.132 |
| Car, D2 corrected, 50 | 0.119 [0.044, 0.207] | 0.136 | 0.096 | 0.142 |
| Driver (incl. car), D1, 50 | 0.103 [0.063, 0.158] | 0.081 | 0.092 | 0.125 |
| Driver (incl. car), D2, 50 | 0.100 [0.056, 0.164] | 0.101 | 0.071 | 0.125 |
| Driver minus teammate, D1, 50 | 0.089 [0.065, 0.134] | 0.190 | 0.043 | 0.082 |
| Driver minus teammate, D2, 50 | 0.090 [0.068, 0.136] | 0.249 | 0.051 | 0.082 |

Car level D1 by half-life 8 / 16 / 32 / 50: 0.068 / 0.071 / 0.086 / 0.096; D2: 0.095 / 0.098 / 0.112 / 0.119. Earlier races of
the same season only (D1, half-life 50): 0.059. No half-life, definition or season of the decayed-mean car level reaches 0.2 (largest 0.142, D2 2023). The
same-season-only car variant (half-life 50, at least 3 earlier same-season races) reaches 0.239 [0.087, 0.393] for D2 in 2021
(D1 2021 0.090; D2 2022 0.022, 2023 0.052; dev pooled 0.059 D1 and 0.067 D2), so the earlier sentence "no season reaches 0.2"
was wrong for that variant: the largest seasonal car-level value in `a3_persistence.csv` is 0.239. Intervals for the 22-race season cells are wide. Sanity of the machinery: with the same code, the field-relative race gap
itself is persistent at car level 0.659 [0.528, 0.778] and the qualifying gap at 0.464 [0.390, 0.617]
(`results/plan3/a3_sanity_pace_level.csv`), so the weak persistence is a property of the offset, not of a broken pipeline.

**3. Split of the oracle gap** (same finishers, D1, rho_fin means over 65 races; pace-unit predictors with nothing fitted,
offsets with coefficient 1; the five terms add up to the total gap)

| Term | beta = 1 (pre-declared) | beta = in-sample OLS slope (optimistic diagnostic) |
|---|---|---|
| model minus qualifying gap order (rho_q - rho_model) | -0.018 [-0.034, -0.004] | same |
| + persistent car offset (earlier races) | -0.069 [-0.090, -0.048] | -0.040 [-0.055, -0.024] |
| + persistent driver offset (earlier races, vs teammate) | -0.047 [-0.068, -0.027] | -0.023 [-0.036, -0.011] |
| + race-specific car offset | +0.064 [0.021, 0.103] | +0.049 [0.012, 0.085] |
| + race-specific driver offset | +0.147 [0.103, 0.193] | +0.108 [0.068, 0.150] |
| total gap (oracle minus model, this subset) | 0.076 [0.038, 0.113] | same |

The OLS slopes (dev, in-sample) are 0.72 (car) and 0.34 (driver vs teammate). D2 gives the same pattern (persistent car -0.070,
persistent driver -0.050, race-specific car +0.089, race-specific driver +0.170 at beta = 1). Reading (narrow): with this offset
`rq = race gap - q_best_gap` and coefficient 1, adding the earlier-race car and driver offsets to the qualifying gap lowers rho
against the finishing order, and also at the in-sample slope; the gain sits in what is only known after the race, mostly the
driver's own deviation from the car's mean (traffic, damage, strategy and incidents are inside it, not only skill). This does
not show that no offset learnable from earlier races could explain part of the gap: this `rq` correlates with the same race's
qualifying gap (field-relative `rq_c` against the field-relative `q_best` gap, within-race mean Pearson -0.48 D1 / -0.53 D2; the
uncentred pooled -0.90 is dominated by large gaps), so it carries much race-specific qualifying noise, and `q_best`
mixes Q1, Q2 and Q3 times. The terms are increments
along an ordered sequence, not shares, and depend on the order (car before driver).

**Gate (mechanical, point estimates; thresholds from PLAN3 section 4):**

| Definition | Oracle gap >= 0.03 | Car persistence >= 0.2 | Gate |
|---|---|---|---|
| D1 median clean lap (**primary, fixed before the run**) | 0.0750 yes | 0.096 **no** | **FAIL** |
| D2 corrected | 0.1189 yes | 0.119 **no** | FAIL |

The definition choice does not swing the gate: both fail on the same condition by a wide margin (the persistence CI upper bound
is 0.182 for D1 and 0.207 for D2; the D2 interval touching 0.2 does not change a point-estimate rule). The plan-2 convention gap
(0.0724) and the 105% / 110% filters also pass the first condition. **The gate fails as declared** (rq = race gap - q_best_gap, coefficient 1,
half-life 50, point estimates), so under the plan's rule M09 is not built (R6 skipped) and work moves to M10-M12. This is a
statement about that offset, not a general answer to Q2: see the sensitivity section, where the outcome depends on the
qualifying reference, which the plan did not fix.

## What we learned
- The 2023 oracle gap of plan 2 holds on 66 races (0.075 vs 0.073 on 22): knowing a finisher's own race pace after the race is
  worth a lot. That information is not available before the race.
- The gap is not carried by a stable car or driver offset of the declared form `race gap - q_best_gap`. Car-level persistence
  of that offset is about 0.1 (CI upper bound 0.18 for D1), half the 0.2 threshold, and the same for the driver. Other
  qualifying references give higher values (post-hoc section below). Corrected pace is slightly more persistent than the
  median lap at car level (0.119 vs 0.096, inside the intervals), as PLAN3 section 8 expected, by less than 0.05.
- Against the expectations written before the run: oracle gap in the predicted 0.05-0.08 range; car persistence at the bottom of
  the guessed 0.10-0.30 range; the persistent parts of the gap were predicted "under 0.02 rho" and came out negative.
- A caution on the split: qualifying order already encodes the grid, which drives the finishing order. Pace-unit shifts by a noisy
  offset move cars away from grid order and cost rho. That argues against an additive form of M09, but the gate was decided by
  the persistence number, not by the split.

## Limits
- The qualifying gap `q_best_gap` mixes Q1, Q2 and Q3 times (a driver knocked out in Q1 has a Q1 time), so `rq` carries
  session-composition noise; a cleaner qualifying reference was not tried (it would be a new definition after seeing results).
- Tyre compound is not modelled (none before 2023); used-tyre age at stint start is ignored; stint age comes from Jolpica pit
  laps. Safety-car and red-flag laps are removed only by the 107% filter.
- Outliers (tester findings): `q_best_gap` reaches 45.6% (Sargeant, 2023 round 2; wet 2016 round 11 above 30%). Dropping the
  38 dev entries with |rq_c| > 10 raises the gate persistence from 0.096 to 0.156 [0.077, 0.226], still below 0.2.
- Implausible fuel coefficients (0.25-0.61 s per lap) in 8 of the 203 fitted races, 3-4 of them in dev (2021 round 2, 2022
  rounds 7 and 17, possibly 18), probably a drying track. D2 numbers, including the 0.119 oracle gap, are affected; the D1
  gate is not.
- Two drivers have laps but no `entries2` row and drop out: norris 2019 round 18 and alonso 2023 round 19.
- 2014 round 4: the lap table keeps 56 laps and `rem` is counted from lap 56 (burn-in only, not scored).
- Persistence rows need 5 earlier observations; new lineages and the 2022 regulation reset start with shrunk values. Half-life 50
  was declared the gate value before the run; the others are sensitivity only.
- The 2021 car-level value (-0.000) and the other season cells rest on about 21 races. The split uses coefficient 1 on the
  offsets; the OLS version is fitted in-sample on the same dev races.
- The leaderboard row for this run is an oracle (`spearman_fin` = oracle rho_fin, other metrics empty, `cell = oracle`). Plan 2's
  A1 kept its oracle out of the leaderboard while CLAUDE.md asks for one row per run; the main session decides whether to keep it.
- Data gaps: 2021 round 12 (no pit data, one lap, not in the oracle). 26 of 1,311 dev entries have no lap rows. Checked: all 26
  are unclassified with no Jolpica lap row (a driver who completed lap 1 has one; 7 other retirees have exactly one lap), so
  they did not complete a lap with a time; 18 carry an accident or collision status. The other 8 carry "Retired", "Power
  Unit" or "Hydraulics"; that they went out on lap 1 is the reading of the missing rows, not independently confirmed. In
  one race (race_id 148) a retiree with 2 laps is ranked below four lapless drivers, an ordering oddity in the source. These
  are retirements, not missing source data. Pace coverage: 1,227 of the 1,311 dev entries (18.6 per race over all 66
  races) have a D1 pace; the earlier "17.2 of 19-20" is the mean number of classified finishers with a D1 pace in the
  65 oracle races, not drivers per race. 2024-2026 are not read.

## Post-hoc sensitivity to the qualifying reference (referee review; descriptive, not a gate re-run)
Chosen after seeing the gate result, so it cannot change the gate. Script `sensitivity.py` (this folder), table
`results/plan3/a3_reference_sensitivity.csv`. Same statistic as the gate (car level, pooled Pearson, half-life 50, at least 5
earlier observations, field-relative offset, 10,000 race resamples, seed 0); the first row reproduces the gate values.

| Variant (car level) | D1 r [95% CI] | D2 r [95% CI] | Referee D1 / D2 |
|---|---|---|---|
| Gate: rq vs `q_best_gap` | 0.096 [0.016, 0.182] | 0.119 [0.044, 0.207] | |
| rq vs `q1_gap`, missing Q1 = NaN (the faithful analogue of the gate; 16 dev entries; n = 645) | 0.268 [0.137, 0.454] | 0.393 [0.279, 0.511] | 0.268 [0.141, 0.450] / 0.393 [0.284, 0.508] |
| rq vs `q1_gap`, Q1 filled with plan-1 imputation 0.10 (`entries2.q1_gap`; n = 646) | 0.184 [0.079, 0.328] | 0.227 [0.120, 0.355] | |
| Per-race residual of race gap on `q_best_gap`, median-centred like `rq_c` (main) | 0.282 [0.156, 0.459] | 0.471 [0.415, 0.526] | 0.282 [0.157, 0.457] / 0.471 [0.416, 0.527] |
| Same residual, raw (mean-zero) | 0.306 [0.181, 0.471] | 0.486 [0.429, 0.541] | |
| Winsorised 5% (driver-level `rq_c` clipped at its 5/95% quantiles, 2014-2023) | 0.183 [0.102, 0.259] | 0.201 [0.124, 0.273] | 0.167 (other winsorising) |
| Spearman (ranks over the pooled dev rows) | 0.146 [0.059, 0.230] | 0.147 [0.063, 0.228] | 0.146 |
| Lag-1 (previous race's car offset) | 0.119 [0.006, 0.264] | 0.114 [-0.000, 0.269] | 0.116 |

Partial correlation of the Q1-based offset (missing Q1 = NaN) with its history (car level, pooled, field-relative controls):

| Controls | D1 | D2 | Referee |
|---|---|---|---|
| this race's `q_best` and `q1` gaps | 0.391 [0.236, 0.578] | 0.565 [0.491, 0.635] | 0.392 / 0.568 |
| the same, plus the decayed mean of the car's earlier race-pace gap | -0.004 [-0.128, 0.145] | 0.088 [0.019, 0.162] | -0.010 / 0.083 |

Offset against the same race's qualifying gap: the comparable figure is the field-relative `rq_c` against the field-relative
`q_best` gap, mean of per-race Pearson over the 65 races: -0.476 (D1) and -0.525 (D2). The pooled -0.90 (D1) and -0.93 (D2)
are for the uncentred `rq` against the raw gap and are dominated by large qualifying gaps.

Remaining small differences from the referee's figures (third decimal, bootstrap intervals, partial-correlation controls) are
within what implementation detail of the controls and resampling explain. The earlier mismatch was caused by
`entries2.q1_gap` filling a missing Q1 time with 0.10 (`src/pitlane/data/ingest.py`), which the gate pipeline never does for
`q_best_gap` (NaN); the residual comparison differed only in median-centring; the first winsorising attempt differed in where
the clip is applied (driver level matches the gate pipeline). The Q1 gap is also clipped at 0.10 in `entries2`; kept.

Reading, with these limits. (a) The gate outcome depends on the qualifying reference, which the plan did not fix: the point
value of car-level persistence moves between 0.10 and 0.47 across references on the same data; the Q1-based (0.268 D1, 0.393
D2) and the median-centred residual (0.282 D1, 0.471 D2) clear 0.2 for both definitions, winsorised only for D2 (0.201),
while the Spearman and lag-1 versions stay near 0.12 to 0.15. (b) The persistent signal appears to be car race pace in earlier races: once the
decayed mean of the car's earlier race-pace gap is controlled for, the partial correlation of the Q1-based offset with its
history falls to about 0 (-0.004, D1) and 0.09 (D2). This is a reading of correlations, not a causal test, and it does not say that an
M09 built on such an offset would improve finishing order: rho of the pace-unit split was not re-run with these references.
(c) These variants were chosen after seeing the gate result and several of them were tried at once; none carries a
multiplicity adjustment. (d) Whether to run M09 on another definition is a plan decision for the main session and the user.
It is not made here, the gate result above stands as declared, and no gate, threshold or definition in the plan was changed.

## Open questions / next tries
- If M09 is revisited, a cleaner qualifying reference (per-segment gap) would be a separate, declared definition.
- The persistent offset could matter in a lap-level simulator (plan 4) even if the additive rank term does not use it. Not tested.
