# M00 — Heuristic baselines

**Status:** run (see results below) · **Plan reference:** PLAN.md section 4 · **Role:** Deck Baseline 1
**Factors:** n/a — baseline

## Hypothesis
How much of the finishing order do simple pre-race signals already explain? Anything we build must beat these.

## Method
Utility = one pre-race column: `grid` (starting grid), `quali` (qualifying order), `standings` (championship order before the race), `last_race` (previous finishing position), `season_mean` (season-to-date mean finish). No fitting.

## Leakage notes
All columns are known before lights-out (grid/quali) or are shifted history (`drv_last`, `drv_season`, `pts_cum`).

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
