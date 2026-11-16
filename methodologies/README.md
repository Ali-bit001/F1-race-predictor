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
| M05 | Ensemble (rank average of latent + GBM) and monotone GBM | Plan 3 (R10) | done | dev: no gain with an interval excluding 0 on rho or MAE | [M05_ensemble](M05_ensemble/README.md) |
| M06 | Weekend-information models (practice, richer qualifying, sprint, weather) | Plan 2 main model | run | see `results/plan2/` | see `results/plan2/` |
| M07 | DNF hazard model (logistic) | Plan 2, retirements | run | log-loss, see `results/plan2/` | |
| M08 | Race-level Monte Carlo simulator | Plan 2, structure | run | RPS, see `results/plan2/` | |
| A1 | Ceiling analysis (oracles) | Plan 2, analysis | run | | |
| A2 | Forecast-horizon analysis (H0/H1 factorial) | Plan 2, analysis | run | | |

| A3 | Race-pace persistence analysis | Plan 3, gate | done (R4), gate FAILED | dev: oracle gap 0.075 rho passes, car-level rq persistence 0.096 < 0.2 fails; M09 not built | [A3_pace_persistence](A3_pace_persistence/README.md) |
| M09 | Race-pace model (race-minus-qualifying offset) | Plan 3 primary hypothesis | planned | | |
| M10 | Upgrade signal (FIA car presentation documents) | Plan 3 | planned | | |
| M11 | Likelihood variants (finishers-only, truncated, reverse) | Plan 3 | done (R5, revised) | dev only: reverse passes alone on MAE and exact rate (not rho, not Holm over 36), no variant beats grid order, tuning and M11 x M07 change nothing measurable | [M11_likelihood_variants](M11_likelihood_variants/README.md) |
| M12 | Two-stage Friday forecast | Plan 3 (R10) | done | dev, 21 races of 2023: no measurable difference from H1 | [M12_two_stage](M12_two_stage/README.md) |
| M13 | Market benchmark (price collection) | Plan 3, data collection | planned | | |
| A4 | Rolling-origin evaluation and rolling re-tune | Plan 3, amendment 1 | done (R3) | rolling re-tune vs fixed on dev: no gain, keep 2021-23 tuning | [A4_rolling_origin](A4_rolling_origin/README.md) |
| A5 | Strength audit (all teams, intervals, shrinkage) | Plan 3, amendment 1, run first | done (R1) | descriptive; see `results/plan3/` | |
| M14 | Car traits (engine supplier, trait × circuit) | Plan 3, amendment 1 | planned | | |
| M15 | Rookie prior from junior-series record | Plan 3, amendment 1 | planned | | |
| M16 | Simulator extensions (S4–S7) | Plan 3, amendment 1 | planned | | |
| M17 | Metric-matched decoding (assignment) | Plan 3, amendment 2, run early | done (dev) | D2 exact 13.8% vs grid 16.3%, CI includes 0 | [M17_decoding](M17_decoding/README.md) |
| M18 | Overtaking threshold | Plan 3, amendment 2 | planned | | |
| M19 | In-race forecast (after lap 1, 25%, 50%) | Plan 3, amendment 2, reported separately | done (R13, dev) | rho above the running order at H4/H5 (+0.036, +0.035, CIs exclude 0), H6 +0.011 (latent) / not measurable (gbm); exact-position rate is lower | [M19_inrace](M19_inrace/README.md) |
| M20 | Race-craft traits and pit crew | Plan 3, amendment 2 | planned | | |
| M21 | Neural-network control (optional, needs rule change) | Plan 3, amendment 2 | planned | | |

M09–M21 and A3–A5 are specified in `PLAN3.md` §4; M05 (ensemble) is taken up there too.
M06–M08, A1 and A2 are specified in `PLAN2.md` §4; their results are in `results/plan2/REPORT.md`.

Extensions (X1–X11 in `PLAN.md` §7) get folders named `X<n>_<name>` when started.

Status values: `planned` · `in progress` · `done` · `abandoned`.
