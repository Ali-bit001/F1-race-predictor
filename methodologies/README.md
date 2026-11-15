# Methodologies

One folder per approach. Each contains:

```
<ID>_<name>/
  README.md      method card (copy _TEMPLATE/README.md)
  config.yaml    hyperparameters and switches
  run.py         entry point; calls into src/pitlane
  results/
    <run_id>/    config snapshot, metrics.json, per_race_metrics.csv, predictions.parquet, figures/
```

Folders are created when a methodology is first run. Abandoned approaches stay here with
status `abandoned` and the reason — they are part of the record.

## Index

| ID | Name | Role | Status | Best dev Spearman ρ | Best dev MAE |
|----|------|------|--------|---------------------|--------------|
| M00 | Heuristic baselines (grid, quali, standings, last race) | Deck Baseline 1 | run | see leaderboard | see leaderboard |
| M01 | Direct GBM / RF regression | Deck Baseline 2 | run | see leaderboard | see leaderboard |
| M02 | GBM ranking (LambdaRank) | Deck Baseline 3 | run | see leaderboard | see leaderboard |
| M03 | Latent strength model — 2×2×2 factorial (Family L) | **Proposed model, primary ablation** | run | see leaderboard | see leaderboard |
| M04 | GBM 2×2×2 factorial (Family G) | Robustness check | run | see leaderboard | see leaderboard |
| M05 | Hybrid latent + GBM | Optional | planned (optional, not built) | | |
| M06 | Weekend-information models (practice, richer qualifying, sprint, weather) | Plan 2 main model | run | see `results/plan2/` | see `results/plan2/` |
| M07 | DNF hazard model (logistic) | Plan 2, retirements | run | log-loss, see `results/plan2/` | |
| M08 | Race-level Monte Carlo simulator | Plan 2, structure | run | RPS, see `results/plan2/` | |
| A1 | Ceiling analysis (oracles) | Plan 2, analysis | run | | |
| A2 | Forecast-horizon analysis (H0/H1 factorial) | Plan 2, analysis | run | | |

M06–M08, A1 and A2 are specified in `PLAN2.md` §4; their results are in `results/plan2/REPORT.md`.

Extensions (X1–X11 in `PLAN.md` §7) get folders named `X<n>_<name>` when started.

Status values: `planned` · `in progress` · `done` · `abandoned`.
