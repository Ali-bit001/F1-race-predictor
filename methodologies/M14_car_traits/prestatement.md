# M14 part 1 (engine-supplier pooling) - pre-statement, written before any full run

Written 2026-11-17, before any tuning or final run of this component. Only the unit checks (gradient, off-switch) and
a 6-race `--quick` smoke run had been executed at the time of writing; no dev rho of the e-model had been seen.

Model: `b_total[car] = b[car] + e[supplier, season]`, both ridge-penalised; `b` keeps the plan-1 car penalty and
carry-over; `e` has its own penalty `lam_e` and no carry-over across seasons. Reference: `latent_111` (plan-1 frozen
hyper-parameters), re-run in the same harness. Dev 2021-2023, walk-forward, refit before every race.

Expected gain on dev: about 0 rho. Reasons: dev holds one chassis change (2022) and no engine change; the supplier
of every constructor is the same in 2021, 2022 and 2023, so e can only pool within a season (every season starts at
e = 0), and teammates plus customer teams of a supplier are already informed by their own history. The case for this
component is 2026 (new power units, no history), which lies in the locked test block and is not touched here.

Failure: the paired per-race rho interval (10,000 resamples, seed 0, driverId tie-break) includes 0. A pass on dev would
still count only after the fresh holdout (secondary comparison, Section 5). Also reported with the same pairing: rho
on finishers, MAE, exact rate, by season, and exact rate against grid order.

lam_e: grid [0.3, 1, 3, 10, 32, 100], chosen by dev mean rho at refit_every=4 (6 configurations, budget 30). No gbm
variant: the plan's term is in car strength.

Descriptive only, no decision weight: fitted e by supplier-season and the share of car-strength variance it takes
(end of dev, and after 2022 round 5); 2022 rounds 1-5 results separately (5 races, no interval).

---

# POST-HOC variant: carry-over on the total car strength (section dated 2026-11-17, written before the variant was run)

**Label: post hoc.** This variant was added after the dev result of the declared model above was seen (null, lam_e at the
grid edge). It is not part of the pre-statement, carries no decision weight beyond the Section 5 rule, and is labelled post hoc in the
config, run folder, leaderboard and attribution row.

Why: in the declared model the carry-over penalty `(b_s - rho * b_{s-1})^2` acts on the car's own part `b` only, not on `b + e`.
So `e` behaves as "reset the supplier share every January" and not as pooling of a persistent car level. This was raised by the
skeptic review and is a real flaw in the parameterisation.

Variant: total strength `T = b + e[supplier, season]`. Penalty `lam * (T_s - rho * T_{s-1})^2` (the plan-1 `lam` and `rho`, frozen),
`lam_e * e'e` on the supplier part, and a plain ridge `lam_b * b'b` on the own part with `lam_b = lam` (frozen, not tuned).
The ridge on `b` is needed: with the carry-over on `T` and no penalty on `b`, `b` and `e` only enter through `T`, the ridge on `e`
pushes `e` to 0 and the model is the reference (checked: with `lam_b = 0` the fitted utilities equal the reference up to the L-BFGS
tolerance). The choice `lam_b = lam` is mine and is a design decision of this variant; the limit `lam_e -> inf` is therefore
not the reference but "carry-over on `b` plus an extra ridge on `b`".

Expected gain on dev: about 0 rho (same reasons as above: no engine change in 2021-2023, teams' own history is informative).
Failure: the paired per-race rho interval (pitlane.plan3.attribution.paired_gain, 10,000 resamples, seed 0, driverId tie-break,
dev 2021-2023 only, against `latent_111` re-run in the same harness) includes 0. Also reported with the same pairing: rho on
finishers, MAE, exact rate, by season, exact rate against grid order. No decision weight beyond the Section 5 rule; passing on dev
would still count only after the fresh holdout.

Budget: lam_e grid [1, 3, 10, 32, 100, 300], chosen by dev mean rho at refit_every = 4: 6 configurations. With the 6 of the declared
model the M14 total is 12 of 30. No other setting is tuned. If the chosen value is at a grid edge it is reported as such and the grid
is not extended.

Also measured (not part of the decision): L-BFGS convergence (iterations, final gradient norm) for the reference and every lam_e
configuration, and the per-race tuning metrics of every lam_e, saved to `results/`.

---

# Correction note, dated 2026-11-17 (revision 2; added below the post-hoc section, nothing above is rewritten)

1. Limit description. The post-hoc section above says the limit `lam_e -> inf` is "carry-over on `b` plus an extra ridge on `b`". That is wrong for the
   variant as implemented: with `lam_e = inf` the model switches the e block off and is the plain `LatentModel` (carry-over on `b`, no lam_b term). The finite-`lam_e` variant has `e -> 0` as
   `lam_e -> inf`, which does give "carry-over on `b` plus an extra ridge `lam_b` on `b`", but that limit is not reachable through the code path `lam_e = inf`.
2. Look-ahead defect, fixed. The first post-hoc run let a car of a later season with no rows yet carry a free `b` (pulled by the carry-over, ridged by `lam_b`), so race r's scores
   depended on the future entry list. Fix: a car with no training row is pinned (no carry-over term, no `lam_b` ridge; predicted `b` = rho * b of last season's car, plus the supplier level's e).
   The same 12 configurations were re-run, no new configuration, no extra budget. `lam_b` is written as 32.0 (= lam) in the config instead of null; the value is unchanged. The first post-hoc results are marked superseded.
