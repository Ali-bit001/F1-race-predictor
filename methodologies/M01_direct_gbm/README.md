# M01 — Direct regression (RF and LightGBM)

**Status:** run (see results below) · **Plan reference:** PLAN.md section 4 · **Role:** Deck Baseline 2
**Factors:** all OFF (cell 000)

## Hypothesis
A conventional model regressing normalised finishing position should beat the heuristics but is entangled, static and pointwise.

## Method
`rf_000`: RandomForest. `gbm_000`: LightGBM regression. Features: qualifying/grid/Q1 gap, expanding driver and constructor form, circuit history, DNF rates. Predictions are sorted to give the order. Same code path as M04 cell 000.

## Leakage notes
Form features are shrunk expanding means over strictly earlier races (tested by `tests/test_leakage.py`).

## How to run
```
python scripts/run_all.py   # whole study; or import pitlane.pipeline for single runs
```
Per-run artifacts (config snapshot, `metrics.json`, `per_race_metrics.csv`, `predictions.parquet`) are in `results/<run_id>/`;
tuning grids in `results/tuning_<cell>.csv`. Cross-method comparison: `/results/REPORT.md`, `/results/leaderboard.csv`.

## What we learned
See `results/REPORT.md` section for this method and the dated entries in `experiments/LOG.md`.

## Open questions / next tries
See PLAN.md section 7 (extensions X1-X11).
