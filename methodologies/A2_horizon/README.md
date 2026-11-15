# A2 — Forecast-horizon analysis (plan-1 extension X1)

**Status:** run · **Plan reference:** PLAN2.md §4 (A2) · **Role:** where do the three plan-1 factors matter?
**Factors:** full 2×2×2 factorial of family L at H0 (history only); two cells at H1 (history + practice)

## Hypothesis
Dynamic weighting, driver–car separation and structured ranking show larger effects before qualifying is known (H0) than at H2 (plan 1), because qualifying no longer masks them. Evidence against: main effects at H0 no larger than at H2.

## Method
Family L with `base_x=False` (no weekend signal at all) for H0; practice columns only for H1 (practice exists from 2023, so H1 = H0 on 2021–22). Tuned on dev with the plan-1 grids; the H2 reference is the plan-1 `latent_*` runs.

## Results
`results/plan2/REPORT.md` section 3; leaderboard labels `h0_latent_<cell>`, `h1_latent_<cell>`.

## What we learned
See the report and `experiments/LOG.md`.
