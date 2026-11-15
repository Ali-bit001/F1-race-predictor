# M02 — GBM ranking (LambdaRank)

**Status:** run (see results below) · **Plan reference:** PLAN.md section 4 · **Role:** Deck Baseline 3
**Factors:** F3 only (cell 001)

## Hypothesis
Optimising the ordering directly (race = query group) should help a conventional feature set. Mimics the public XGBoost-ranking systems.

## Method
LightGBM `lambdarank` with linear label gain; relevance = field size − position. Same features as M01. Same code path as M04 cell 001.

## Leakage notes
As M01.

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
