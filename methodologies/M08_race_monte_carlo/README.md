# M08 — Race-level Monte Carlo simulator

**Status:** run · **Plan reference:** PLAN2.md §4 · **Role:** full driver × position probabilities and structure test
**Factors:** builds on M06 cell `111` at H3 and M07

## Hypothesis
Sampling whole races, with retirements modelled separately from pace, a circuit-specific weight on the grid, and a "disrupted race" regime, gives better probability forecasts (RPS, log-loss of P(points finish)) than plain Plackett–Luce sampling of the same scores. Evidence against: no step improves RPS on dev and test with a paired CI above 0.

## Method
Per race, 1 000 draws (300 while fitting): (1) each entry retires with probability `dnf_scale · p_dnf` (M07) and retirees finish behind all finishers in random order; (2) finishers are ordered by `tau · (M06 score + kappa · stickiness_c · grid_z) + Gumbel noise` (Plackett–Luce sampling); (3) with probability `p_dis` (logit-linear in the circuit's historical grid volatility and the rain forecast) the draw is disrupted and the noise is multiplied by `s_dis`. Output: position-probability matrix; point prediction = expected finishing position. Parameters are fitted by coordinate search on dev (minimising mean RPS, common random numbers) and then frozen. Ladder: S0 (PL only, tau) → S1 (+retirements) → S2 (+stickiness) → S3 (+regime). Lap-by-lap tyre/pit/overtake simulation is deliberately not attempted.

## Leakage notes
Inputs are the walk-forward out-of-sample M06 scores and M07 probabilities; circuit stickiness and volatility come from earlier races only. Parameters are frozen on 2021–23.

## How to run
```
python scripts/run_plan2.py
```

## Results
`results/plan2/REPORT.md` section 6; `results/plan2/leaderboard.csv` (labels `m08_S0..S3`).

## What we learned
See the report and `experiments/LOG.md`.

## Open questions / next tries
Per-position retirement timing (lap of retirement) and safety-car-driven reshuffles.
