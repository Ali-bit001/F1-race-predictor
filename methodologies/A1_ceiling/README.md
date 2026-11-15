# A1 — Ceiling analysis (analysis, not a model)

**Status:** run · **Plan reference:** PLAN2.md §4 (A1) · **Role:** how much order accuracy is left to gain, and the gate for plan 2's headline metric

## Hypothesis
If a race-pace oracle beats the plan-1 proposed model by less than 0.03 rho on finishers, order accuracy is saturated and probabilities are the headline.

## Method
On the development block, for plan-1 `latent_111`: (1) rho against orders sampled from the model's own calibrated Plackett–Luce distribution; (2) a retirement oracle (retirees at their true places); (3) a race-pace oracle: classified finishers ranked by median clean race lap (lap 1 and laps above 107% of the race median dropped), computed for the 2023 development season only because Jolpica's rate limit (~500 requests/hour) makes laps for all of dev too slow. Oracles use race data on purpose and never enter a leaderboard.

## Results
`results/plan2/ceiling.json`, `results/plan2/ceiling_per_race.csv`, `results/plan2/REPORT.md` section 1.

## Open questions / next tries
Extend the pace oracle to the full development block once the laps cache is built over several hours.
