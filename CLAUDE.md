# PitLane — working rules

Read `PLAN.md` first. It is the specification; implement one work package (WP) at a time, in order.
Plan 1 (WP1–WP8) is done. The current work is `PLAN2.md` (work packages P0–P8); it inherits every
rule below. In plan 2 the 2024–2026 test blocks have already been seen once: tune only on 2021–2023.

## Non-negotiables
- **CPU only.** No neural networks, no LLM training, no GPU libraries. Stack: pandas, numpy, scipy,
  scikit-learn, lightgbm, shap, fastf1, matplotlib. Ask before adding a heavy dependency.
- **No temporal leakage.** Anything used to predict race *r* must be known before race *r* starts.
  Never shuffle across time. The truncation test in `tests/` must pass before any result is recorded.
- **The test block (2024–2026) is locked** until WP7. Tune only on the development block (2021–2023).
- **Do not change the research design in code.** Splits, metrics, and the definitions of the three
  factors live in `PLAN.md`. If one looks wrong, stop and raise it instead of working around it.

## Where things go
- `src/pitlane/` — reusable code (data, features, models, evaluation, explain). No experiment logic here.
- `methodologies/<ID>_<name>/` — one folder per approach: `README.md` (method card, copy
  `_TEMPLATE`), `config.yaml`, `run.py`, and `results/<run_id>/`.
- `results/` — cross-method outputs only: `leaderboard.csv` and comparison figures/tables.
- `experiments/LOG.md` — append an entry for every run worth remembering, **including failures and
  dead ends**. Never delete or rewrite old entries.
- `data/raw|interim|processed/` — git-ignored, regenerable. `data/external/` — small hand-made tables, committed.
- `notebooks/` — exploration only; nothing in the pipeline may depend on a notebook.

## Every run must write
`methodologies/<ID>/results/<run_id>/` containing `config.yaml` (snapshot), `metrics.json`,
`per_race_metrics.csv`, `predictions.parquet`, and the git commit hash; then append one row to
`results/leaderboard.csv`. `run_id` = `YYYYMMDD-HHMM_<short-label>`.

## Style
- Python 3.10, type hints, small functions, `pytest` for tests. Fixed seeds everywhere.
- Config in YAML, not hard-coded constants.
- One WP or one methodology change per commit; message says what was tried and what the dev metric did.
- Report results honestly: if a variant is worse, the log and the method card say so.
