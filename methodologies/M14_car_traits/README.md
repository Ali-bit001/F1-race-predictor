# M14 — Car traits, part 1: engine-supplier pooling

**Status:** done (R11 part 1, dev only; revised before its first commit). Part 1 is a null on dev, as pre-stated. Parts 2 and 3 (trait x circuit, learned car x circuit factors) are NOT cut: they come later, in their own commits. M15 (rookie prior) is cut (Amendment 3 item 4).
**Budget:** 12 distinct configurations of 30 (6 declared + 6 post-hoc lam_e values); about 4 tuning sweeps were seen (declared sweep twice, post-hoc sweep twice: 1828 superseded by the look-ahead fix, 1854); the post-hoc form and lam_b = 32 were fixed after the declared result and are not counted as tuned.
**Plan reference:** PLAN3.md section 4 M14, Amendment 3 item 4, section 5
**Factors:** latent_111 (F1 on, F2 on, F3 on) plus a supplier term in car strength. No gbm variant: the plan's term is in car strength, which LightGBM does not have.

## Hypothesis (pre-stated in `prestatement.md`, written before any tuning or final run)
Expected gain on dev about 0 rho: one chassis change (2022), no engine change, and the supplier of every constructor is the same in
2021, 2022 and 2023. The case for the component is 2026 (new power units, no history), which lies in the locked test block and is not
touched here. Failure: the paired rho interval includes 0. A pass would still need the fresh holdout (secondary comparison).

## Method
`b_total[car] = b[car] + e[supplier, season]`, car = (lineage, season). `b` keeps the plan-1 penalty and carry-over decay (lam 32, rho 0.8,
half-life 50); `e` has its own ridge `lam_e * e'e` and no carry-over across seasons, so each (supplier, season) level starts at 0 and
is learned from the cars sharing that supplier in that season. The supplier comes from `data/external/power_units.csv` joined on
(season, constructorId): pooled by supplier value, so Alfa/Sauber/Audi lineages and rebadged units (Honda RBPT) share levels correctly.
Implementation: `src/pitlane/models/latent_supplier.py` (`LatentSupplierModel`, subclass of `LatentModel`, registry kind
`latent_supplier`), analytic gradient, `lam_e = inf` removes the e parameters and runs the `LatentModel` code unchanged.
Experiment: `run.py` / `config.yaml` here. Reference `latent_111` re-run in the same harness; paired gains with
`pitlane.plan3.attribution.paired_gain` (10,000 resamples, seed 0, driverId tie-break).
`lam_e` grid [0.3, 1, 3, 10, 32, 100], dev mean rho at refit_every 4: 6 configurations (budget 30).

## Leakage notes
The supplier of race r is the season-start value of the table row of r's season. Check in every run (`results/checks.json`): scrambling
the outcomes of race r and all later races and replacing the supplier of all later seasons leaves race r's scores unchanged
(max abs diff 1.6e-14, regenerated with the current code); deleting all later races changes scores by 1.1e-3, a numerical dependence on the later entry list, about 1e-3 at the default tolerance, the same as plain `LatentModel` (1.0e-3), so it is not
an e effect. The tight-tolerance truncation test (<1e-6) passes. This leakage check runs at lam_e = 10, not at the chosen 100, and the post-hoc arm's leakage is covered only by tests in `tests/test_plan3_m14.py`. The earlier explanation here ("L-BFGS stopping at maxiter 150") was wrong: every fit converges well before 150 iterations (see Convergence).
The cause is not isolated; the likely one is that a cold fit with a different parameter set stops at a different point inside the L-BFGS-B tolerance (unverified). The full entry universe is loaded
(as in earlier plan-3 runs: it is needed for bit-for-bit reproduction), but only dev races are scored and only rows before the target
race are fitted; the run asserts no score outside 2021-2023.

Level pinning (revision): a (supplier, season) level with no training row has no data term; its parameter is now pinned to exactly 0 by
construction (gradient masked, warm start reset), instead of being ~1e-20 from round-off in the penalty gradient. The same holds for a car without a training row in the post-hoc arm: its gradient is about 4e-18 (numerically zero; a strict xfail test documents that it is not bitwise 0). The stored declared
run reproduces under this change (re-derived tuning surface identical, max |rho diff| 0.0; reference per-race metrics 8.9e-16).

## History of the first run
The first run of this component used a universe cut at 2023 (the dev horizon; the final run uses `universe_last_season: 2026`, and its 1617 reference reproduces the stored plan-1 run with score diff 0.0); its rows were removed from the two plan-3 CSV files and re-appended after
a rerun with the full universe (the config notes that the full universe is needed for bit-for-bit reproduction of `latent_111`), before anything was committed. Only the
second run's rows exist. I (a later agent session) did not see the first run and know this only from the instruction I was given. Pre-statement created 16:14:25; declared results written 16:17:34. That it preceded every full run cannot be shown: a first full run existed and its outputs were overwritten, and the hash of the declared text was pinned later. Treat the declared expectation as stated, not as proven by timing.

## How to run
```
python methodologies/M14_car_traits/run.py [--quick] [--checks-only]
python methodologies/M14_car_traits/run.py --posthoc     # revision: per-lam_e per-race metrics, convergence, post-hoc variant
```

## Results (dev 2021-2023, 66 races, refit before every race)
| Run | Split | rho | MAE | exact | Notes |
|---|---|---|---|---|---|
| m14_ref_latent_111 | dev | 0.6460 | 3.355 | 0.1397 | reproduces stored plan-1 dev run (max abs per-race metric diff 8.9e-16, scores 0) |
| m14_supplier_pool_latent_111 (lam_e = 100) | dev | 0.6457 | 3.364 | 0.1366 | paired gains below |

Paired gain of the candidate over latent_111 (positive = better, 95% CI over 66 races): rho -0.0003 [-0.0027, +0.0022]; rho on finishers
-0.0015 [-0.0037, +0.0006]; MAE (lower is better, sign flipped so positive = better) -0.0091 [-0.0258, +0.0065]; exact rate -0.0030
[-0.0113, +0.0047]. All include 0: no measurable difference. By season rho: 2021 +0.0001 [-0.0021, +0.0027], 2022 -0.0008 [-0.0064, +0.0048],
2023 -0.0002 [-0.0040, +0.0039]. (Two of the 15 season intervals exclude 0, both on the wrong side: 2022 rho on finishers -0.0039 [-0.0079, -0.0001] and 2022
MAE on finishers -0.0412 [-0.0782, -0.0076]; they are not independent and not a finding.) Exact rate against grid order (0.1638): candidate 0.1366,
gap -0.0272 [-0.0643, +0.0049]; latent_111 0.1397, gap -0.0242 [-0.0611, +0.0069]. Declared -0.0272 [-0.0643, +0.0049], reference -0.0242 [-0.0611, +0.0069], post-hoc -0.0311 [-0.0616, -0.0050]: both point estimates are about 2.5 points below grid order, and the post-hoc interval excludes 0.

**Tuning.** Dev rho at refit_every 4 rises monotonically with lam_e: 0.3 -> 0.6062, 1 -> 0.6167, 3 -> 0.6285, 10 -> 0.6361, 32 -> 0.6449, 100 -> 0.6455;
reference at refit_every 4: 0.6454. The chosen value (100) is the edge of the grid and the trend points to "no supplier term" (lam_e = inf is the
reference exactly). The grid was not extended (it was fixed in the pre-statement); a larger lam_e can only approach the reference. A small lam_e
costs up to 0.04 rho. The per-race metrics of every lam_e (and of the reference) at refit_every 4 are saved in `results/tuning_per_race_declared.csv`
(re-derived in the revision; identical to `tuning_lam_e.csv`, max |rho diff| 0.0).

**Convergence (measured in the revision: `results/convergence_declared.csv`, `convergence_posthoc.csv`, `solver_probe.json`).** L-BFGS-B (maxiter 150) converges
for the reference and for every declared lam_e: 21 fits each, status 0 in all, none reaches 150 iterations (max 40, median 23-28), final gradient inf-norm
median 8e-6, max 3e-5 (the default tolerance is 1e-5; 86-100% of fits end under it, none above 3e-5). The declared tuning surface is therefore not a
non-convergence artefact. The 0.33 maximum score difference at lam_e = 1e8 (`solver_probe.json`: 0.330; also in `checks.json`) is not a maxiter effect either: with maxiter 2000 it is still 0.27.
At lam_e = 1e8 the fits stop far from a stationary point (final gradient inf-norm median 1e-2, max 0.27, against 1e-5 for the reference; median 8 iterations at maxiter 150
and 1 at maxiter 2000), consistent with the ridge 2e8 on the e coordinates making the problem very stiff so that the stopping rule fires early. Measured on 6 races;
the stopping rule itself was not isolated, so this is an explanation, not a proof. It has no bearing on the grid values (lam_e <= 100), which converge.

**Descriptive (no decision weight).** Fitted e at the end of dev (lam_e = 100) is small: all |e| <= 0.066 (Honda 2023 +0.066, Ferrari 2022 -0.034,
Mercedes 2022 +0.029) against a car-strength SD of about 0.17-0.24. Share of the variance of b + e (= T) across the 10 car-seasons taken by e alone, var(e)/var(T):
2021 0.8%, 2022 1.5%, 2023 2.3%. The covariance term 2 cov(b, e)/var(T) is -6.5%, +9.5%, +13.0% (e and the cars' own parts are positively correlated in 2022-23, negatively in 2021; cause not examined).
`descriptive.json` was not regenerated (it is only written by a full run); `test_descriptive_fitted_e_reproduced` re-fits it under the current code within 6e-5 (e) and 1e-6 (share). Computed in `run.py: variance_share` (unweighted over the 10 car-seasons of the season; var(T) = var(b) + var(e) + 2 cov(b, e)). The supplier share is therefore
not just 1-2%: giving the whole covariance term to e it is 2023 15.3%, 2022 11.0%, 2021 -5.7% (an upper bound); splitting the covariance equally between b and e gives
8.8%, 6.3% and -2.4%. The allocation of the covariance is a convention. Honest statement: about 2% on its own, 9% to 15% in 2023 depending on the convention, with e small in absolute size (|e| <= 0.066). After 2022 round 5: e Ferrari +0.010, Honda -0.028, Mercedes +0.024,
Renault -0.006; e share 1.7%. 2022 rounds 1-5 (5 races, no interval): rho difference 0, 0, -0.0045, 0, +0.015 (mean +0.002); at round 1 e is exactly 0 under the current code (in the stored run 1617 it was about 1e-20)
(new season, no carry-over), so the only difference then is the 2021 fit.

## POST-HOC variant: carry-over on the total strength (added in the revision; post hoc; rule in `prestatement.md`, dated section written before the run)
The skeptic review pointed out that in the declared model the carry-over penalty acts on b only, so e behaves as "reset the supplier share each January" and
not as pooling; this probably pushed lam_e to the grid edge. The variant puts the carry-over on T = b + e, keeps lam_e * e'e and adds a plain ridge lam_b * b'b with
lam_b = lam = 32 (frozen, not tuned; written as 32.0 in the config since revision 2, identical to the earlier null; without a ridge on b the model is the reference). lam_e grid [1, 3, 10, 32, 100, 300] at refit_every 4, 6 configurations.

**Revision 2: look-ahead defect, fixed, results re-recorded.** The first post-hoc run (run folder `20261117-1828_m14_supplier_pool_carry_total_posthoc_dev`, files `*_superseded_lookahead.*`)
had a look-ahead defect found by the tester: a car of a later season with no rows yet still had a free b; carry-over pulled it to rho*T_s and the lam_b ridge then left a residual
penalty on the earlier car's T_s, so race r's scores depended on which lineages appear in later seasons (the future entry list). Truncating the universe changed scores by up to 0.051 in 3 races
(`test_carry_on_total_truncated_universe_gives_the_same_score_at_the_exact_optimum`). Fix (`latent_supplier.py`, post-hoc variant only): a car with no training row is pinned: no carry-over
term and no lam_b ridge in the objective; its predicted b is rho * b of last season's car (0 if none), the value LatentModel's carry-over gives a car without data (disclosure: this is rho*b_prev, whereas this variant's own carry-over acts on the total T, so its own prior would be rho*T_prev; with e at about 80% of var(T), first-race cars lose most of their carried level; round-1 delta rho against the reference: 2021 -0.0015, 2022 -0.0391, 2023 -0.0376; mean gain without round 1 is -0.0150, against -0.0155 with it, so the verdict does not move), plus the supplier level's e (as before, 0
until the level's first race). The declared arm and the defaults are unchanged: the stored declared supplier run reproduces with max |score diff| 0.0 and max |per-race metric diff| 1.1e-16; the reference
re-run reproduces at 8.9e-16; the re-derived declared tuning surface is identical (max |rho diff| 0.0). The truncation test passes. The same 12 configurations were re-run (no extra budget).
Verdict unchanged, but the chosen lam_e moved (32 -> 1) and the e share of variance changed a lot (see below); the old post-hoc numbers are no longer valid.

Dev rho at refit_every 4, new run: 0.6302, 0.6296, 0.6294, 0.6299, 0.6292, 0.6292 (lam_e 1 ... 300), against 0.6454 for the reference: flat (range 0.001) and about 0.015 below the reference. **lam_e = 1 chosen**
(highest by 0.0003, a tie in practice; the previous run chose 32, which is 0.0003 behind here). All fits converge (status 0, none at maxiter; max final gradient inf-norm 9.3e-5, 24-57% of fits under 1e-5).
Files: `results/*_posthoc.csv`, `posthoc_summary.json`, run folder `20261117-1854_m14_supplier_pool_carry_total_posthoc_dev`; the old ones are marked `superseded: look-ahead defect`.

Refit before every race, 66 dev races, paired against `latent_111` (10,000 resamples, seed 0; positive = variant better): rho -0.0155 [-0.0249, -0.0066]; rho on finishers -0.0190 [-0.0275, -0.0111];
MAE (sign flipped) -0.0740 [-0.1310, -0.0208]; exact rate -0.0069 [-0.0236, +0.0104]. By season rho: 2021 -0.0181 [-0.0376, -0.0018], 2022 -0.0031 [-0.0161, +0.0097], 2023 -0.0252 [-0.0395, -0.0116].
(Old run for comparison: rho -0.0156 [-0.0254, -0.0058]; rho on finishers -0.0179; MAE -0.0725; exact -0.0015; by season -0.0113, -0.0059, -0.0296.)
Exact rate against grid order (0.1638): variant 0.1327, gap -0.0311 [-0.0616, -0.0050], now excluding 0 on the wrong side (the old run's gap was -0.0256 [-0.0561, +0.0003]). Attribution verdict: worse.
The variant is worse than the reference, with an interval excluding 0 on the wrong side, as before. Pooling is very active here: at the end of dev e takes 76%, 83% and 81% of var(T) in 2021-2023 (old run 24%, 38%, 32%; declared model 1-2%);
the own part takes 22%, 15%, 17% and the covariance term is small (1-2%). Total variance of T is small (0.004-0.017), so the picture is "the cars of one supplier are fitted almost alike and the own part is shrunk hard". Caveat: lam_b = lam adds an own-part ridge that the reference does not have, so part of the loss may come from that extra shrinkage and not from pooling;
the design cannot separate the two (a lam_b grid would use more configurations and was not run). Post hoc, no decision weight; it does not rescue the component.

Reading of the post-hoc arm (main evidence that the fixed lam_b ridge, not pooling, drives the loss): var(T) falls from 0.031, 0.045, 0.056 (declared, 2021-2023, `descriptive.json`) to 0.004, 0.011, 0.017 (post-hoc, `posthoc_summary.json`), and the lam_e tuning surface is flat across lam_e 1-300 (0.6292 to 0.6302, spread about 0.001 rho; `tuning_lam_e_posthoc.csv`). Penalty arithmetic: moving a supplier group by x costs about x^2 via e (lam_e = 1) against 64-256 x^2 via own b (lam_b = 32, 2-8 cars), so the 76-83% e share is the mechanical result of the penalties, not a defect. The post-hoc arm changes two things at once (carry-over on T, plus the own-part ridge) and is not evidence for or against pooling; R11 part 1 has not tested pooling of a persistent car level (that would need a lam_b sweep or e carried across seasons).

## What we learned
- Part 1 is a null on dev: the declared supplier term does nothing measurable (rho -0.0003 [-0.0027, +0.0022]), and when lam_e is small it hurts. Expected: with e per (supplier, season) and no carry-over,
  e cannot help at a season's first race, and within a season the cars' own history already carries what the engine would add. The post-hoc variant that makes pooling active is worse
  (-0.0155 [-0.0249, -0.0066]).
- M14 parts 2 and 3 (trait x circuit from timing data; learned car x circuit factors) are not tested by this commit and are not cut; they come later in their own commits.
- The plan's motivating case (a new regulation, no history) is not testable on dev. With the present design e starts at 0 for every new
  (supplier, season) level, so even in 2026 it would help only after the first races, through customer teams sharing the supplier. A carry-over of e
  across seasons within an engine formula (or a prior from the supplier's last season) would be the version that could help at round 1; that
  is a design change not declared in the plan and was not built.
- Not shown: any benefit at a regulation change. The 2026 block stays locked.

## Open questions / next tries
- Fresh holdout (2026 races after the freeze) is the only place the component can be judged. Expected gain there is also small; unverified.
- Works flag in `power_units.csv` is not used by this model (it enters no term).
