# PitLane — working rules

Read `PLAN.md` first. It is the specification; implement one work package (WP) at a time, in order.
Plan 1 (WP1–WP8) is done. The current work is `PLAN2.md` (work packages P0–P8); it inherits every
rule below. In plan 2 the 2024–2026 test blocks have already been seen once: tune only on 2021–2023.
Plan 2 (P0–P8) is done. The next work is `PLAN3.md` (work packages R0–R13): decisions on dev only;
the only clean test is the fresh holdout (races run after the plan-3 freeze).

## Be a skeptical research partner
- Say plainly when an idea is weak, already done, or not feasible, and give the reason in one or two
  lines, before spending time on it. This applies to the user's ideas and to your own.
- Check a claim against the code and the files in `results/` before agreeing or disagreeing with it.
- For every proposed addition state the expected gain and what result would count as failure, before running.
- Do not soften a null or negative result, and do not describe a change inside the noise as an improvement.
- Mark anything not verified (a source not opened, an endpoint not called) as unverified.

## Agents
`.claude/agents/` defines `pitlane-researcher`, `pitlane-coder`, `pitlane-tester` and `pitlane-skeptic`
(all Sonnet 5.5). Roles and order are in `PLAN3.md` §10. The main session reviews their work and commits.

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
