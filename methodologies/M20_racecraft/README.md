# M20 — Race-craft and pit crew (phase 1: persistence check only)

**Status:** phase 1 done (persistence); phase 2 done (consistency alone, dev): no measurable difference on any metric
**Plan reference:** PLAN3.md §4 M20, §6 R8
**Factors:** F1 dynamic = n/a · F2 decomposition = n/a · F3 ranking = n/a (no model in this phase)

This card was written before any persistence number was computed. Data-structure facts looked at before
writing (no feature values): laps and pit stops for 2014-2023 come from Jolpica for every race (one source per
race); Jolpica laps carry a stored `position` for 2014-2022 and none for 2023.

## Hypothesis
Four small race-craft traits are persistent enough from race to race to be worth testing in the weekend term:
driver consistency, driver tyre degradation, driver lap-1 gain, pit-crew time loss (with slow-stop rate).
Expected effect if they enter a model: under 0.003 rho each (PLAN3 §8); lap-1 gain is the likeliest to show.
Result that counts against a feature: persistence below 0.2 on dev (decision rule below). A3 found that the
related car race-pace offset was not persistent (0.096), and that driver-level traits are confounded with the
car, so a pass is not the expected outcome for the three driver features.

## Method: the race value of each feature
All race values are outcomes of that race (computed from that race's own laps and stops, nothing from any other
race). They are used (a) as the target `y` of the persistence check and (b) as the inputs of the history means
that would become features of LATER races. They are never features of the race they are computed from.
Laps and stops: Jolpica, all seasons 2014-2023 (one source per race, so OpenF1's missing 2023 rounds 1-6 pit
data and the cross-source lap-1 offset do not matter). Laps are marked by `pitlane.features.pace.mark_laps`
(clean lap = lap >= 2, <= 107% of the race median lap, not an in-lap or out-lap; stints are the laps between
Jolpica pit-stop laps; compound is not known for 2014-2022, so it is not used for any season).

**1. Driver consistency (`cons`).** Per driver, per stint with at least 5 clean laps: regress the clean lap
times (as % of the driver's median clean lap time in the race) on lap number within the stint; take the
residuals. `cons` = 1.4826 x median absolute residual over all such stints of the driver (robust spread, % of lap
time; lower = more consistent). Driver needs at least 10 clean laps in qualifying stints. Field-relative:
`cons_c = cons - median over eligible drivers of the race`. Unit: driver.

**2. Driver tyre degradation (`deg`).** Same regressions as `cons`: `deg` = pooled within-stint slope (stint
fixed effects: sum of stint Sxy over sum of stint Sxx) of the lap-time % on lap number, in % of lap time per lap.
This is the net of tyre wear and fuel burn (they are collinear within a stint and cannot be separated without a
model; the common fuel part is removed by the field-relative step). Field-relative:
`deg_c = deg - median over eligible drivers of the race`. Same eligibility as `cons`. Unit: driver.

**3. Driver lap-1 gain (`lap1`).** `gain = grid - position after lap 1` (positive = places gained). Position
after lap 1 = the stored Jolpica `position` of lap 1 where the lap table has one (2014-2022); for 2023 (no stored
position) the rank of the lap-1 time among drivers with a lap-1 time. Agreement of the two rules on 2021-2022 is
reported. Not used for lap-1 positions: any time comparison across sources. Excluded: pit-lane starters
(`pit_start`) and drivers without a lap-1 record (retired on lap 1; this selects on an outcome of the same race, it
is a property of the race value, not a leak). Because gain depends on grid slot (back-markers can only gain),
the race value is the residual `lap1_c` of a per-race least-squares fit of `gain` on `grid` (field-relative,
slot-adjusted). Needs at least 8 drivers in the race. Unit: driver.

**4. Pit-crew time loss and slow-stop rate (`crew_loss`, `crew_slow`).** Per stop, duration = Jolpica pit
duration (total pit-lane time, NOT stationary time; stationary time exists in OpenF1 only from late 2024, so
this feature measures pit lane plus stop). Stops with duration > race median stop duration + 20 s are dropped as
red-flag or repair stops (the plan-3 margin `pits.red_flag_margin_s` = 20 s). The median is the race-wide median
of that race: this is allowed because the value of race r is only ever used for races after r (a history
feature); the in-race code in `inrace.py` recomputes the flag from stops up to lap k because there it feeds a
same-race forecast, which is not the case here. **Amendment 1 (made after the first run, which I had seen the numbers of):** the first run showed team values of
hundreds of seconds in red-flag races (2016 R20, 2023 R3, a 2021 race), because a red flag makes most stops long, the
race median itself becomes long and "median + 20 s" keeps them. The first-run crew time-loss persistence on dev was
-0.010 (declared) and 0.178 (winsorised), which is an artefact of those races, not a result. Fix, a bug fix and not
a tuned threshold: stops longer than 60 s are dropped first (no normal stop in 2014-2023 takes that long), then
the median + 20 s rule is applied to the rest. The first-run numbers are not used for the decision. Both the old and the new
number are reported in the run summary below.
Time loss of a stop = duration - race median of the kept stops.
`crew_loss` = median time loss over the team's kept stops in the race (needs >= 2 kept stops); `crew_slow` = share
of the team's kept stops with time loss > 3.0 s ("slow stop"; threshold declared here, not tuned). Field-relative:
`crew_loss_c = crew_loss - median over teams`, `crew_slow_c = crew_slow - mean over teams`. **Level: team (car
lineage), not driver** - the unit is a team-race, as `con_rq` is in A3. `crew_loss` is the primary component of
feature 4; `crew_slow` gets its own row and its own decision. Both are reported; neither is merged.

## Method: the persistence statistic (same as A3)
For each feature and unit, the history value of an observation is the shrunk, time-decayed mean (half-life 50
races, race ordinal as time, one pseudo-observation of prior 0, `pitlane.features.pace._decayed`) of the same
unit's field-relative values in EARLIER races only. Persistence = pooled Pearson correlation, over all
observations with at least 5 earlier valid observations, of the race's field-relative value with that history
value (`pitlane.plan3.pace_gate.pooled_corr`), with a 95% percentile CI from resampling races (10,000 resamples,
seed 0). Driver features are pooled over drivers; crew over teams. Also reported (none used for the decision):
half-lives 8, 16 and 32; per season 2021, 2022, 2023; 2014-2020 as burn-in (history only from 2014, so early
rows are dropped by the 5-observation rule; not a decision block and not tuned on); Spearman; a teammate-relative
version of the three driver features (value minus the mean of the team's eligible drivers, needs two); and a
version of each feature with the 1% and 99% values winsorised across all races.

## Decision rule
Dev (2021-2023), half-life 50, field-relative value, Pearson point estimate **>= 0.2**: the feature is
persistent and goes to phase 2 (each alone in the weekend term). Below 0.2: reported as not persistent and not
run. As in A3 the point estimate decides; where the CI straddles 0.2 this is said. Sensitivity rows cannot rescue
a feature that fails the declared row (A3's lesson: a post-hoc reference changed the verdict; here the declared
definitions are fixed first).

## Leakage notes
History features for race r use only rows with race_id < r (shared code and the same truncation tests as A3;
M20-specific tests are for the tester). The race-wide red-flag median and the per-race slot regression are
properties of the race being summarised and only reach other races through the history means. Not a pre-race
feature set yet: nothing here is added to any model in phase 1.

## How to run
```
python methodologies/M20_racecraft/run.py --config methodologies/M20_racecraft/config.yaml
```
Code: `src/pitlane/features/racecraft.py`. Outputs: `results/plan3/m20_persistence.csv`,
`results/plan3/m20_summary.json`, `methodologies/M20_racecraft/results/<run_id>/`. No leaderboard row (no model).

## Results
Run `20261117-0243_m20_persistence_dev` (final, after amendment 1; `...-0242_...` is the first run with the
red-flag bug and is superseded). Dev 2021-2023, half-life 50, declared value, pooled Pearson, 95% CI by race
bootstrap (10,000, seed 0), threshold 0.2:

| Feature | Level | r | 95% CI | n obs | races | Decision |
|---|---|---|---|---|---|---|
| Driver consistency | driver | 0.256 | [0.153, 0.355] | 1187 | 65 | persistent (CI straddles 0.2) |
| Driver tyre degradation | driver | 0.121 | [0.058, 0.177] | 1187 | 65 | not persistent |
| Driver lap-1 gain | driver | 0.163 | [0.107, 0.220] | 1222 | 66 | not persistent (CI straddles 0.2) |
| Pit-crew time loss | team | 0.186 | [0.102, 0.282] | 603 | 65 | not persistent (CI straddles 0.2) |
| Pit-crew slow-stop rate | team | 0.119 | [0.036, 0.204] | 603 | 65 | not persistent (CI straddles 0.2) |

Full table (all half-lives, seasons, burn-in 2014-2020, sensitivities): `results/plan3/m20_persistence.csv`.

## What we learned
The declared rule decides (point estimate at half-life 50, field-relative value). Everything marked post hoc
below was computed after the result was known, by `robustness.py` into `results/plan3/m20_robustness.csv`
(CIs: race bootstrap, 10,000, seed 0), and has no decision weight. Corrected statement: an earlier version of this
card said that no sensitivity version turns a failing feature into a passing one. That is wrong. The Spearman
column of `m20_persistence.csv` is 0.220 for lap-1 gain and 0.280 for crew time loss at half-life 50, and
0.204 for tyre degradation at half-life 8; all three are above 0.2 although their declared Pearson values fail.
(Winsorised Pearson values stay below 0.2: crew loss 0.195, lap-1 0.171.)

**Consistency: passes as declared (0.256 [0.153, 0.355]), but it is not a driver trait (post hoc).**
* Within a race, `cons_c` correlates with grid position (mean per-race Spearman 0.328 [0.247, 0.402]) and with
  finishing position (0.311 [0.240, 0.379]); 64 dev races. Slow cars are less consistent.
* Partialling out the current race's grid position: 0.132 [0.049, 0.217]. Partialling out the car's own
  consistency history (car mean of `cons_c`, decayed): 0.124 [0.034, 0.215].
* Car-level persistence (mean `cons_c` of the car's drivers, lineage history): 0.295 [0.192, 0.401].
* Single drivers: without Mazepin 0.191 [0.104, 0.278]; without Mazepin, M. Schumacher and Zhou 0.128 [0.034, 0.223].
  Requiring at least 20 earlier observations: 0.166 [0.072, 0.260]. Teammate-relative (declared in the card as a
  sensitivity): 0.091 [0.018, 0.169].
* Reading (interpretation, not tested): the feature mostly measures slow-car traffic, blue flags and car
  quality, which the model already has through car strength and grid. The declared rule still sends it to phase 2.

**Tyre degradation: fails (0.121 [0.058, 0.177]).** PLAN3 §2.1 words the driver trait as "against the
teammate". That version is 0.102 [0.012, 0.188], also below 0.2. Spearman 0.144 at half-life 50; at half-life 8
the Pearson value is 0.165 and the Spearman 0.204 (post hoc, not a decision).

**Lap-1 gain: fails as declared (0.163 [0.107, 0.220]), but the field-relative decision is not robust (post
hoc).** Spearman 0.220; values clipped to +-5: 0.209 [0.153, 0.263]; dropping rows with |value| > 6 from the
scored set: 0.229 [0.171, 0.287]. The teammate-relative version is about 0.05 under every trim (0.046 declared
scale, 0.045 clipped, 0.048 dropped; intervals include 0), so a driver-specific part is absent: what persists is
the car and grid-slot effect. (A first draft of the review quoted 0.203 for the clipped version; the code here gives
0.209, with the clip applied before the history is built. The difference was not traced.)

**Crew time loss: fails as declared (0.186 [0.102, 0.282]), but the decision is not robust (post hoc).**

| Variant | r | 95% CI |
|---|---|---|
| declared (cap 60 s, margin 20 s, median, >= 2 stops) | 0.186 | [0.102, 0.282] |
| drop races with a stop > 300 s (13 dev races, scored rows only) | 0.259 | [0.170, 0.354] |
| mean loss instead of median | 0.219 | [0.141, 0.304] |
| margin 10 s | 0.215 | [0.130, 0.301] |
| minimum 1 stop | 0.200 | [0.119, 0.285] |
| cap 40 s | 0.201 | [0.130, 0.276] |
| cap 45 s / 50 s | 0.170 / 0.170 | [0.077, 0.269] / [0.079, 0.270] |
| margin 30 s | 0.160 | [0.103, 0.238] |
| cap 100 s / 300 s | 0.187 / 0.187 | [0.102, 0.283] |
| driver level (median loss of the driver's stops) | 0.083 | [0.027, 0.147] |
| leaving out Haas (scored rows) | 0.090 | [0.001, 0.180] |
| season 2021 / 2022 / 2023 | 0.084 / 0.297 / 0.227 | [-0.031, 0.197] / [0.170, 0.434] / [0.067, 0.418] |

* Not reproduced: the review quotes 0.241 for "drop the 22 red-flag races". I defined a red-flag race as one with
  a stop above 300 s: 22 such races exist in 2014-2023 but only 13 in dev, and dropping them gives 0.259. The
  review's 22-race definition was not available to me, so the 0.241 is unverified.
* Amendment 1 is a real bug fix, not a tuning step: any cap from 60 to 300 s gives 0.186-0.187, because Jolpica has
  426 stops above 300 s and only 27 between 60 and 300 s. Without a cap the value was -0.010.
* Jolpica's duration is total pit-lane time, which also contains penalties, double-stacking and drive-through
  effects; it is not crew speed. Dependence on one team (Haas) and on the margin and the cap, with 603
  observations, makes the 0.186 versus 0.2 distinction uninformative either way.
* Slow-stop rate: 0.119 [0.036, 0.204] declared, Spearman 0.100; no variant checked.

Other facts: crew features measure pit-lane time (Jolpica), not stationary time. The folder is
`M20_racecraft`; PLAN3 §6 names it `M20_racecraft_pitcrew`.

## Phase 2 plan (written before any model run; phase 2 is not started)
* Run consistency only, on the declared definition (`cons_c`, half-life 50 history), added to the weekend term of
  `latent_111` and to `gbm_100`, each against the same model re-run in the same harness (dev, 66 races, paired
  by race).
* Expected gain: about 0, well under 0.003 rho. Failure: the dev interval of the paired difference includes 0 (on
  rho and MAE), which is the expected outcome.
* Report the teammate-relative version beside it, labelled post hoc, with no decision weight.
* Not run: tyre degradation, lap-1 gain and slow-stop rate (not persistent as declared). Crew time loss is not run
  unless the user amends the plan, because it failed the declared rule.

## Phase 2 results (run 2026-11-17 06:44, dev 2021-2023 only, 66 races; code stamp `1db3fdc+dirty-...`, uncommitted at run time)
**Set-up, as fixed in the plan above.** Feature `cons_x` = history of the declared `cons_c` (shrunk decayed mean, half-life 50,
race_id as time, prior 0, earlier races of the same driver only), divided by one constant (SD of the history over seasons <= 2020,
0.0718), so the column has unit scale. Latent: `latent_111` (frozen M03 cell 111) with `x_cols=[cons_x]`, `lam_x=1.0`
(the `LatentModel` constructor default, `latent.py:36`, and the A2 value; it is not "the value used for every plan-2 extra column":
plan-2 M06 tuned `lam_x`, mostly 3.0, sometimes 0.3, see `methodologies/M06_weekend_info/README.md`. The same wrong reason is written in a
comment in `config_phase2.yaml` line 7 and in `run_phase2.py`; both are left unedited so that the code stamp recorded in the run configs
still matches. `latent.py` needed no change). Trees: `gbm_100` (frozen M04 cell 100)
with `extra_cols=[cons_x]`. **No hyper-parameter was tuned**: both models keep their frozen plan-1 values, `lam_x` is the default,
the scale is a fixed constant. **Deviation from PLAN3 §5**, which says each component is tuned separately with the same budget: here
neither side was tuned (budget 0 on both the candidate and the reference). The reference is the plan-1 tuned model and the candidate
carries an untuned extra column with an untuned penalty, which leans against the candidate. Refit before every race, paired by race with the same model re-run in the same harness,
outcome-free tie-break by driverId, 10,000 resamples, seed 0.

**Join (deviation from the wording of the brief, same intent).** The history is computed on a frame with one row per entry (value
NaN where the driver's own race value is invalid, plus the value rows that have no entry row), so for race r it is the decayed
mean of all valid values of races < r. A literal "latest stored history row of races < r" holds only races before that row, so it
would miss the most recent valid value. Truncation check inside the run (blank the values of race 171 and later, recompute): max
difference 0. Values exist to 2023 only; rows after 2023 are NaN and never used.

**Reproduction.** The re-run references equal the stored plan-1 dev runs: `latent_111` max |per-race metric difference| 8.9e-16,
max |score difference| 0; `gbm_100` 4.4e-16 and 0 (66 races, 1311 entries). Means: rho 0.6460 / 0.6489, MAE 3.355 / 3.337.

Paired gain on dev, candidate minus reference (positive = candidate better; MAE and log-loss signed so that positive = better); 95% CI:

| Metric | vs `latent_111` | vs `gbm_100` |
|---|---|---|
| rho | +0.0030 [-0.0005, +0.0067] | +0.0031 [-0.0022, +0.0085] |
| MAE | +0.0149 [-0.0128, +0.0423] | +0.0183 [-0.0246, +0.0628] |
| exact, all | -0.0001 [-0.0123, +0.0120] | +0.0077 [-0.0061, +0.0214] |
| exact, classified | -0.0012 [-0.0144, +0.0119] | +0.0080 [-0.0081, +0.0239] |
| exact, places 1-3 / 4-10 / 11+ | 0.0000 [-0.025, +0.025] / +0.0152 [-0.009, +0.041] / -0.0111 [-0.030, +0.007] | -0.0101 [-0.051, +0.035] / +0.0065 [-0.024, +0.037] / +0.0131 [-0.010, +0.035] |
| rho, finishers | +0.0019 [-0.0016, +0.0054] | -0.0011 [-0.0066, +0.0046] |
| MAE, finishers | +0.0140 [-0.0147, +0.0417] | -0.0124 [-0.0533, +0.0272] |
| points log-loss | +0.0008 [-0.0004, +0.0020] | -0.0008 [-0.0027, +0.0011] |

Verdicts (CI excluding 0 on dev): rho, MAE and exact are all "no measurable difference" for both models (every row above has an
interval including 0). This is the pre-stated failure outcome. The plan expected "about 0, well under 0.003 rho"; the point
estimates of rho (+0.00304 and +0.00313) are slightly above that, inside the noise, and are not read as an improvement.
**Power.** With 66 dev races the rho intervals reach +0.0067 and +0.0085, two to three times the plan's 0.003 ceiling, so a gain of the
expected size could not have passed. Dev cannot distinguish 0 from about +0.007. The null says the feature is not worth adding on this
evidence, not that the effect is zero. Per season (rho / MAE / exact,
point estimates; intervals in `results/phase2_paired_gains_by_season.csv`), vs `latent_111`: 2021 +0.0000 / -0.0002 / -0.0068,
2022 +0.0064 / +0.0407 / -0.0023, 2023 +0.0027 / +0.0043 / +0.0089; vs `gbm_100`: 2021 +0.0073 / +0.0505 / +0.0252,
2022 +0.0036 / +0.0452 / +0.0112, 2023 -0.0015 / -0.0409 / -0.0135. Only one of 18 per-season intervals on rho, MAE and exact excludes 0
(gbm_100, 2021, exact +0.0252 [+0.0002, +0.0457], 22 races). Two per-season points log-loss intervals also exclude 0 (latent_111 2021
+0.0026 [+0.0008, +0.0045]; gbm_100 2023 -0.0050 [-0.0086, -0.0019], a worsening), and 8 of the 36 teammate-relative per-season intervals
(rho, MAE, exact, finisher rho, finisher MAE, log-loss; 2 references x 3 seasons x 6 metrics) exclude 0, all on gbm_100 (2021 rho +0.0140
[+0.0045, +0.0244] and MAE +0.101; 2022 finisher rho and MAE negative; 2023 exact -0.032). Overall 11 of 120 per-season intervals exclude 0
against about 6 expected by chance, with mixed signs; none is interpreted, no correction is applied. Exact-position rate against grid order is not improved (reference exact rates 0.140 / 0.142 against 0.164 for grid order;
candidates 0.140 / 0.150).

**Teammate-relative version (post hoc, no decision weight, no attribution row).** Same pipeline on `cons_c` minus the mean of the
team's eligible drivers (history scaled by 0.0360). Paired gain vs `latent_111`: rho -0.0000 [-0.0012, +0.0012], MAE -0.0030
[-0.0140, +0.0077], exact -0.0014 [-0.0082, +0.0054]; vs `gbm_100`: rho +0.0009 [-0.0055, +0.0071], MAE -0.0015 [-0.0404, +0.0396],
exact -0.0060 [-0.0211, +0.0091]. On the full dev block the teammate-relative result is null with no exceptions (the exact interval for
places 1-3 vs `latent_111`, +0.0101 [0.0000, 0.0253], has a lower bound of exactly 0 and does not exclude it; the verdict file says
no measurable difference). The driver-specific part carries nothing.

**Reading (interpretation, not tested).** Consistency as declared has a point estimate of +0.003 rho for both models, inside the noise (intervals include
0), and the teammate-relative version adds nothing, consistent with the phase-1 finding that the persistent part of `cons_c` is the car
and grid slot, which both models already have. Not run: tyre degradation, lap-1 gain, slow-stop rate, crew time loss (failed the
declared rule). Files: `run_phase2.py`, `config_phase2.yaml`, `results/phase2_*.csv|json`, six run folders
`results/20261117-0645_m20p2_*_dev` (two references, two declared candidates, two post hoc), `results/plan3/attribution.csv`
(component `M20_consistency`, references `latent_111` and `gbm_100`), `results/plan3/leaderboard.csv` (method M20, six rows).

## Open questions / next tries
None planned for M20: the declared feature shows no measurable gain; the other four features did not pass the persistence rule.
