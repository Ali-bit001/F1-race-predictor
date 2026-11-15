# M04 — GBM 2x2x2 factorial (Family G, robustness)

**Status:** run (see results below) · **Plan reference:** PLAN.md section 4 · **Role:** Does the conclusion survive on a conventional strong model?
**Factors:** all 8 cells (000/001 reported as M01/M02)

## Hypothesis
If a factor helps in both families the claim is strong; if only in one, it is model-specific.

## Method
F1: EWMA form features + time-decayed sample weights (OFF: expanding means, uniform weights). F2: replace raw driver/constructor form with walk-forward latent strengths (a, b, g) from M03 cells 010/110 (Gaussian, so F3 is not leaked into F2) (OFF: raw form). F3: LambdaRank (OFF: regression). 4 or 12 configs per cell tuned on dev.

## Leakage notes
Latent strengths are computed walk-forward (each row's strengths were fit on earlier races only).

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
