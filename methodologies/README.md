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

Create the folder when work on that methodology starts. Abandoned approaches stay here with
status `abandoned` and the reason — they are part of the record.

## Index

| ID | Name | Role | Status | Best dev Spearman ρ | Best dev MAE |
|----|------|------|--------|---------------------|--------------|
| M00 | Heuristic baselines (grid, quali, standings, last race) | Deck Baseline 1 | planned | | |
| M01 | Direct GBM / RF regression | Deck Baseline 2 | planned | | |
| M02 | GBM ranking (LambdaRank) | Deck Baseline 3 | planned | | |
| M03 | Latent strength model — 2×2×2 factorial (Family L) | **Proposed model, primary ablation** | planned | | |
| M04 | GBM 2×2×2 factorial (Family G) | Robustness check | planned | | |
| M05 | Hybrid latent + GBM | Optional | planned | | |

Extensions (X1–X11 in `PLAN.md` §7) get folders named `X<n>_<name>` when started.

Status values: `planned` · `in progress` · `done` · `abandoned`.
