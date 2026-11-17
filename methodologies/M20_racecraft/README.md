# M20 — Race-craft and pit crew (phase 1: persistence check only)

**Status:** in progress (phase 1: persistence of the four features; no model has been run)
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

## Open questions / next tries
Phase 2 (each persistent feature alone in the weekend term, paired against the reference) is not started.
