# M07 — DNF hazard model

**Status:** run · **Plan reference:** PLAN2.md §4 · **Role:** retirement probabilities for the simulator (M08) and a stand-alone probabilistic target
**Factors:** n/a (not a ranking model)

## Hypothesis
Retirements are the largest single error source (plan 1: rho 0.703 on all starters vs 0.780 on finishers). A logistic model of "not classified" using decayed driver incident rate, decayed constructor mechanical rate, circuit DNF rate, grid band, weather and regulation-change flags predicts them better than a constant rate. Evidence against: log-loss no better than the constant-rate baseline.

## Method
Logistic regression (standardised features, ridge `C` tuned on dev by log-loss), refitted before every race on all earlier races. Features: `drv_inc_ew16` (driver incident rate, half-life 16 races), `drv_dnf_ew16`, `con_mech_ew16` (constructor mechanical rate), `con_dnf_ew16`, `circ_dnf`, `grid_norm`, `pit_start`, `wet_q`, `rain_fc`, `newregs` (first season of a regulation set), `rookie`, `new_team`, `fp_laps`, `fp_missing`. Incident vs mechanical is read from the Jolpica `status` text. Baselines: constant expanding rate; decayed driver/constructor rate average. Scores: Brier, log-loss, AUC per race.

## Leakage notes
All history features use earlier races only (`tests/test_leakage_plan2.py`).

## How to run
```
python scripts/run_plan2.py
```

## Results
`results/plan2/REPORT.md` section 5; `results/plan2/leaderboard_dnf.csv`; coefficients in each run folder (`coefficients.json`).

## What we learned
See the report and `experiments/LOG.md`.

## Open questions / next tries
Cause-specific hazards (incident vs mechanical as separate targets); lap-of-retirement timing; safety-car proneness.
