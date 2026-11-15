# M06 — Weekend-information models (plan 2 main model)

**Status:** run · **Plan reference:** PLAN2.md §3–§4 · **Role:** tests whether practice, richer qualifying, sprint and weather information add value beyond qualifying position
**Factors:** cells `000` and `111` of both families (F1 dynamic · F2 decomposition · F3 ranking), at information set H3

## Hypothesis
Qualifying position already carries most of the available signal (plan 1: every model ties with qualifying order at rho ≈ 0.70). Practice pace, within-session qualifying gaps, the sprint result and a weather mismatch flag carry information that the qualifying position does not, so H3 beats H2 (plan 1). Evidence against: paired gains whose CI includes 0 on development and test, or gains with opposite signs.

## Method
- **Family L (latent):** the plan-1 additive model `u = driver + car + car×circuit-type + β·weekend`, with the weekend term extended by pre-scaled columns. Groups: *qrich* (best-lap gap to pole, gap within the last session reached, session reached, gap to teammate), *practice* (mean best-lap gap, long-run pace gap, laps completed), *sprint* (sprint finishing position, positions gained), *weather* (qualifying position × wet qualifying, × race rain forecast, × forecast/observed mismatch). Extra coefficients are ridge-penalised (`lam_x`, tuned). Race-wide flags are constant within a race and cannot change an order on their own, so in family L they only act through interactions with qualifying position.
- **Family G (LightGBM):** the plan-1 feature set plus the raw columns of all groups and circuit stickiness / circuit DNF rate / circuit volatility; NaN where a source is missing (pre-2023 practice).
- **Information ladder:** H2 → +qrich → +practice → +sprint → +weather (= H3), and leave-one-group-out from H3. Every rung is tuned separately on dev with the same grid; `r0` is the plan-1 frozen model.
- **Decision rule (fixed before running):** a group adds value if its paired gain on dev has a 95% bootstrap CI above 0 and the gain has the same sign on test 2024–25.

## Leakage notes
Practice, qualifying, sprint and weather sessions finish before the race. History features (circuit stickiness, circuit DNF rate, incident/mechanical rates) use earlier races only (`tests/test_leakage_plan2.py`). Weather: observed precipitation on the qualifying day (ERA5) and a modelled forecast for the race window (Open-Meteo historical-forecast archive, from 2020).

## How to run
```
python scripts/run_plan2.py          # whole plan 2; resumable
```

## Results
See `results/plan2/REPORT.md` section 4 and `results/plan2/leaderboard.csv` (labels `m06_L_r0..r4_*`, `m06_L_no_*_111`, `m06_G_h3_*`).

## What we learned
Filled in from the report: see `experiments/LOG.md` (auto entry) and `results/plan2/REPORT.md`.

## Open questions / next tries
Practice exists only for 2023+ (FastF1 is blocked, OpenF1 starts in 2023), so practice effects are estimated from few races. Tyre-degradation slopes and fuel-corrected pace are not modelled.
