# R14 pre-statement (written before any R14 run; dev 2021-2023 only)

Date: 2026-11-18. Nothing below uses 2024-2026 results or the fresh holdout.

## Component `car_penalty_retune` (declared by the main session as one component)
- Reference: `latent_111` (plan-1 frozen: cell 111, lam 32, half-life 50, driver penalty lam_a = lam = 32), re-run in the same harness.
- Candidate: `latent_111` with the **car** penalty `lam` re-tuned and the **driver** penalty `lam_a` held at 32. `lam` multiplies
  the car carry-over penalty and (times `fit_mult` = 5) the car x circuit-type penalty; both move together, as in the model code.
  `lam = 32` with `lam_a = 32` is bit for bit `latent_111`.
- Grid, fixed now, 8 configurations: `lam` in {2, 4, 8, 16, 32, 64, 128, 256}. Pick = highest dev mean Spearman at `refit_every = 4`
  (as plan 1 tuning). The final run refits before every race (`refit_every = 1`). Budget used by this component: 8 of 30.
- Why this is not already done: plan-1 tuned `lam` over {0.5, 2, 8, 32} with the driver penalty tied to it, so a car-only change
  was never tried, and 32 sat at the edge of the grid (larger values not tried).
- Expected gain (rho, vs `latent_111`): 0.000 to +0.005. Reason: the R6 diagnostic found the car order looks under-dispersed
  (a decayed car-form column adds +0.014 rho); a weaker car penalty would let the car level move more, but plan 1 found weaker
  tied penalties worse, so the car-only direction is uncertain. A larger `lam` is expected to be worse.
- Failure: the paired 95% CI of the gain (66 dev races, 10,000 resamples, seed 0) includes 0 on rho, MAE and exact rate, or the
  pick is `lam = 32` (component inactive).
- Pass rule (the one used for M11_reverse and M09 in the decided list): the CI excludes 0 on rho, MAE or exact rate, positive side.
  Because the pick is chosen and judged on the same 66 races the gain is optimistic; per-season gains are reported next to it.

## Final candidate
- Components entering: `M11_reverse` (reverse likelihood), `M09_car_pace_hist` (`pace_hist_x`, D1 median clean lap, half-life 50,
  scaled by the SD over seasons <= 2020), and `car_penalty_retune` only if it passes alone. Everything else is not included
  (decided list: no measurable difference, worse, or not declared).
- Settings re-used from each component's own dev tuning: `lam_x` = 3.0 for `pace_hist_x` (M09, chosen on the bare latent model).
  Because the reverse likelihood widens the strength scale (about 1.8 times, R5), `lam_x` is checked once for the combination:
  4 configurations {0.3, 1, 3, 10} at `refit_every = 4`. Rule: keep 3.0 unless another value beats it by more than 0.002 mean
  rho (about two to ten times the optimiser and surface noise seen in R6 and R3). Budget for the combination: 4 configurations.
- Hyper-parameters otherwise stay as tuned on 2021-2023 (A4 re-tune null). Simulator: plan-2 S3 re-fitted on dev (same coordinate
  search, same grids, 2 passes) on top of the final score; M07 retirement model as in plan 2. Decoder D0 (M17 found none better).

## Leave-one-out (PLAN3 section 5)
- Full model against full minus one component, paired per race, 10,000 resamples, seed 0, dev 66 races.
- A component is retained if its LOO gain is positive with a CI excluding 0 on at least one of the metrics on which it passed alone
  (M11_reverse: MAE or exact; M09_car_pace_hist: rho; car_penalty_retune: the metric(s) on which it passed).
- Otherwise it is reported redundant and removed. If more than one fails, remove the one with the smallest point gain on its
  first criterion metric, re-run, and repeat the leave-one-out on the smaller model, until every remaining component is retained.
- Failure of the whole exercise: the final model's gain over `latent_111` has a CI including 0 on rho, MAE and exact rate.
- Expected, stated in advance: final vs `latent_111` rho about +0.015 to +0.025 (the M09 gain, partly overlapping with what reverse
  repairs at the front), exact rate not above grid order (0.1638). Failure for the exact-rate target (beat grid order with an
  interval excluding 0) is the expected outcome, as in every earlier package.

## Horizons in the freeze
- H2 (after qualifying): the final model, simulator S3 re-fitted, decoder D0.
- H1 (before qualifying, practice only): the plan-2 frozen H1 `111` model unchanged. The final components were only tested at H2;
  a descriptive run with them switched on at H1 is made on 2023 (22 races) and carries no decision weight.
