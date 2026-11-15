# M03 — Latent strength model (proposed, Family L)

**Status:** run (see results below) · **Plan reference:** PLAN.md section 4 · **Role:** Proposed model; primary 2x2x2 ablation
**Factors:** all 8 cells

## Hypothesis
F1/F2/F3 each improve out-of-time ranking; the deck's full model (111) is best. Evidence against: 111 not better than 000 on test, or decomposition failing the face-validity checks.

## Method
u = driver a + car b(lineage, season) + car x circuit-type g + beta . weekend. F1: exponential time decay + season-wise car states with carry-over (OFF: static). F2: drop b and g (OFF: driver absorbs car). F3: Plackett-Luce likelihood (OFF: Gaussian on normalised position). Penalised ML via L-BFGS with analytic gradients (checked against finite differences in `tests/test_latent_recovery.py`, which also tests parameter recovery on synthetic data). Hyperparameters (ridge `lam`, half-life) tuned on dev: 4 or 12 configs per cell.

## Leakage notes
Only races strictly before the target race enter the fit; decay is relative to the target race.

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
