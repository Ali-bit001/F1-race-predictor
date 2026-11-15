# <ID> — <Name>

**Status:** planned | in progress | done | abandoned
**Plan reference:** PLAN.md §<n>
**Factors:** F1 dynamic = on/off · F2 decomposition = on/off · F3 ranking = on/off (or "n/a — baseline")

## Hypothesis
What this approach is expected to show, and what result would count against it.

## Method
Model, inputs, target, loss. Enough detail to re-implement without reading the code.

## Leakage notes
Which inputs are time-dependent and how each is restricted to pre-race information.

## How to run
```
python methodologies/<ID>_<name>/run.py --config methodologies/<ID>_<name>/config.yaml --split dev
```

## Results
| Run ID | Split | Spearman ρ | MAE | Notes |
|--------|-------|-----------|-----|-------|

## What we learned
Including what did not work and why.

## Open questions / next tries
