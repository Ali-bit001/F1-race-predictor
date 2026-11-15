# PitLane — F1 race-order prediction

**Research question:** can explicit separation of car strength, driver strength and circuit effects
improve out-of-time F1 race-order prediction?

Three ideas are tested, alone and in combination, on races the models have never seen:

1. **Dynamic performance** — recent races count more.
2. **Driver–car separation** — driver and constructor effects are estimated separately.
3. **Structured ranking** — the model optimises the whole finishing order, not one position at a time.

Everything runs on a CPU; no deep learning.

## Start here
- [`PLAN.md`](PLAN.md) — the research plan and work packages (plan 1, done).
- [`PLAN2.md`](PLAN2.md) — plan 2: practice, qualifying, sprint and weather information, a DNF model and a race-level Monte Carlo simulator (not started).
- [`results/REPORT.md`](results/REPORT.md) — plan 1 results.
- [`CLAUDE.md`](CLAUDE.md) — working rules for whoever writes the code.
- [`methodologies/`](methodologies/README.md) — every approach tried, each with its own results.
- [`experiments/LOG.md`](experiments/LOG.md) — chronological journal, failures included.
- [`docs/proposal/`](docs/proposal/) — the original proposal deck.

## Layout
```
PLAN.md, CLAUDE.md, README.md
configs/            shared configuration (splits, paths)
data/               raw / interim / processed (git-ignored), external (committed)
src/pitlane/        reusable library: data, features, models, evaluation, explain
methodologies/      one folder per approach: method card, config, run script, results
results/            cross-method leaderboard and comparison figures
experiments/        LOG.md
notebooks/          exploration only
tests/              unit tests, leakage test
scripts/            one-off utilities
docs/               proposal, final report
```

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the study
```
pip install -e . && pip install -r requirements.txt
python scripts/run_all.py          # ingest -> tune on dev -> freeze -> test once -> report  (~1 h on a laptop CPU)
python -m pytest                   # unit + leakage tests
```
Outputs: `results/REPORT.md` (tables, factorial effects, explainability), `results/figures/*.png`,
`results/leaderboard.csv`, and one folder per run under `methodologies/<ID>/results/`.

## Status
WP1-WP8 implemented. FastF1 practice pace / wet flags, sprint results and extensions X1-X11 are not done yet (see `PLAN.md`).
