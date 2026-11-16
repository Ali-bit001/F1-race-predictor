# Experiment log

Append-only, newest at the bottom. One entry per run or decision worth remembering — failures and
dead ends included. Do not edit old entries; add a correction entry instead.

Entry format:

```
## YYYY-MM-DD — <short title>
- Methodology / WP:
- Run ID(s):
- What was tried:
- Result (dev metrics, vs. grid baseline):
- Interpretation:
- Next:
```

---

## 2026-11-15 — Project initialised
- Methodology / WP: WP0
- What was tried: Read the proposal deck; wrote `PLAN.md`; set up the repository structure.
- Result: No code or data yet.
- Next: WP1 — data ingestion.

## 2026-11-15 — Full pipeline run (auto-generated)
- Methodology / WP: WP1-WP8 via `scripts/run_all.py`
- Frozen config hash: `5980a1f6ddba` (tuned on dev 2021-23 only; test blocks run once afterwards)
- Result (mean per-race Spearman rho, higher is better):
  - dev: grid baseline 0.592; top-3: gbm_001 0.661, rf_000 0.658, gbm_000 0.657
  - test_stable: grid baseline 0.690; top-3: rf_000 0.712, latent_111 0.703, gbm_010 0.702
  - test_shift: grid baseline 0.666; top-3: gbm_100 0.720, gbm_110 0.714, gbm_101 0.714
- Interpretation: see `results/REPORT.md` (paired bootstrap CIs and factorial effects).
- Deviations from PLAN.md: FastF1 practice pace and wet-weather flags not yet used (deferred); sprint results not used; circuit types are analyst judgement (`data/external/circuits.csv`); no git tag was created for the freeze, the config hash above identifies it.

## 2026-11-15 — Plan 2 written
- Methodology / WP: planning (`PLAN2.md`)
- What was tried: Read the plan-1 results. All trained models tie with qualifying order on test
  (0.70–0.71 rho); only the dynamic factor gained (+0.010, p = 0.04); `latent_111` scores 0.703 on
  all starters and 0.780 on finishers.
- Interpretation: the limit is information beyond qualifying and the handling of retirements, not
  model capacity.
- Decision: plan 2 adds practice, richer qualifying, sprint and pre-race weather data; a DNF hazard
  model; a race-level Monte Carlo simulator; a ceiling analysis and a forecast-horizon analysis.
  Considered and not built: lap-by-lap simulator, hidden Markov model for car form, full F2
  ingestion (reduced to an optional rookie-prior table). Reasons are in `PLAN2.md` §2.
- Note: the 2024–2026 test blocks were scored in plan 1, so plan-2 test numbers are a second look.
- Next: P0 (harness additions), then P1 (ceiling and horizon analysis on existing data).

## 2026-11-15 — Plan 2 full run (auto-generated)
- Methodology / WP: P0-P8 via `scripts/run_plan2.py` (A1, A2, M06, M07, M08)
- Frozen config hash: `0d1b7c074d80` (tuned on dev 2021-23 only; test blocks are a second look, see PLAN2.md section 5)
- A1 ceiling (dev): model rho 0.646, self-consistency 0.456, retirement oracle 0.843; pace-oracle gap on finishers 0.073 -> not saturated (gate not met)
- Result (mean per-race Spearman rho, higher is better):
  - dev: qualifying order 0.617; top-4: gbm_001 0.661, rf_000 0.658, gbm_000 0.657, gbm_010 0.656
  - test_stable: qualifying order 0.701; top-4: rf_000 0.712, m06_L_no_qrich_111 0.705, m06_G_h3_000 0.703, latent_111 0.703
  - test_shift: qualifying order 0.710; top-4: m06_G_h3_111 0.725, gbm_100 0.720, m06_L_r2_000 0.718, m06_L_r3_000 0.718
  - M08 RPS on test_stable: m08_S0 0.1139, m08_S1 0.1098, m08_S2 0.1098, m08_S3 0.1093
- Interpretation: see `results/plan2/REPORT.md` (information ladder, leave-one-out verdicts, DNF model, simulator ladder).
- Deviations from PLAN2.md: FastF1 is blocked (HTTP 401 without an F1 login), so practice laps and session weather come from OpenF1 and exist for 2023+ only; practice is judged on dev-2023 plus test; race laps for the pace oracle were fetched for 2023 only (Jolpica rate limit); the race-day forecast comes from Open-Meteo's historical-forecast archive (2020+); every ladder rung is tuned separately on dev; no git tag was created for the freeze (the config hash identifies it).

## 2026-11-15 — Plan 3 written
- Methodology / WP: planning (`PLAN3.md`)
- What was tried: Read the plan-2 results and searched the literature and public repositories (search summaries only; sources listed in `PLAN3.md` §9).
- Result: Plan 2 improved probabilities (RPS 0.1218 to 0.1093 on test 2024-25) but not the order (rho 0.702 against 0.703 for plan 1 and 0.701 for qualifying order). The only measured gap in order accuracy is race pace (oracle gap 0.073 rho on finishers, 22 races).
- Decision: plan 3 tests a race-minus-qualifying pace offset learned from earlier races (primary, gated by a persistence check), FIA upgrade documents, three likelihood variants, a two-stage Friday forecast and an ensemble; it collects market prices for a later benchmark. Considered and not built: news and sentiment scraping, a lap-by-lap simulator (deferred, gated), an MCMC version. Reasons are in `PLAN3.md` §2.
- Note: the 2024-2026 test blocks have been scored twice, so plan-3 decisions use dev only and the clean test is the fresh holdout (races run after the freeze).
- Correction: `PLAN.md` §9 cited "van Kesteren & Bergsma"; the second author is Bergkamp. Fixed.
- Next: R0 (lap ingestion, several hours in the background), then R1 (race-pace features and the A3 gate).

## 2026-11-16 — Plan 3 amended (Amendment 1), before any plan-3 code or run
- Methodology / WP: planning (`PLAN3.md` §2.1, §4, §6, §10)
- What was tried: Eight further ideas reviewed against the code and stored results. No model was run.
- Checked facts:
  - Training is already walk-forward with a refit before every race (`harness.walk_forward`, `refit_every=1`); only hyper-parameters were fixed on 2021-23.
  - Verstappen in the stored walk-forward driver strengths (`data/processed/latent_strengths.parquet`): dynamic model mean rank 11.5 (2017), 9.0 (2018), 3.6 (2021), 1.8 (2022), 1.2 (2023), 1.0 (2025-26); static model 3.3-5.2 over 2021-26. The plan-1 chart ranks him 1st of 30.
  - Defects found: `report.py` plots only the 8 lineages with the most rows, which drops McLaren, Haas, Cadillac, Manor and Caterham; drivers with one race sit at the extremes of the driver chart (Hirakawa 2nd, Camara last); no intervals are shown; plan 2 did not re-check strengths.
- Decision: added A5 (strength audit, run first), A4 (season-by-season table and rolling re-tune), M14 (engine-supplier pooling, trait x circuit), M15 (rookie prior from a hand table), M16 (simulator steps S4-S7, judged on RPS); M10 extended with keyword categories. Rejected: car specification sheets and win rate as target, learned text features, scraping junior-series histories, lap-level processes. Work packages renumbered R0-R13. Secondary results now need the fresh holdout to count as findings (about fifteen secondary comparisons).
- Also: four agent definitions in `.claude/agents/` (researcher, coder, tester, skeptic; Sonnet 5.5) and a "skeptical research partner" section in `CLAUDE.md`.
- Not verified: the Amendment-1 references in `PLAN3.md` §9 are cited from memory; OpenF1 speed-trap and sector fields; the cause of the weak shrinkage.
- Next: R0 (lap ingestion in the background), R1 (A5 audit) and R2 (A4) in parallel.

## 2026-11-16 — Plan 3 amended (Amendment 2), before any plan-3 code or run
- Methodology / WP: planning (`PLAN3.md` §2.2, §4, §5, §6)
- What was tried: A 25% exact-position target was proposed. Checked on existing files with throw-away scripts (no model fitted, nothing written to `results/`).
- Result (dev 2021-23, 66 races, 1,311 entries): exact-position rate of starting-grid order 16.3%; plan-2 simulator S3 sorted by score 13.6% (below grid order); model order with all retirements known in advance 28.0% (18.2% on finishers); per-race standard error about 1.1 points. On 2023 laps (22 races): running order after lap 1 18.6%, at 25% distance 22.0%, at 50% 31.9%, at 75% 42.5% (grid 14.2% on the same races).
- Interpretation: 25% before the race is at oracle level and is not adopted. The simulator loses to the grid on this metric because sorting by score is not the right decoder for exact hits. Earlier statements that the simulator improved the exact-position rate (16.5% to 17.7% on test 2024-25) were within noise.
- Decision: added M17 (assignment decoding, run early), M18 (overtaking threshold), M19 (in-race forecasts, reported separately), M20 (driver consistency, tyre degradation, lap-1 gain, pit crew), fuel- and tyre-corrected pace as a switch in M09, learned car x circuit factors in M14, and an attribution protocol: every component alone against a re-run reference, leave-one-out from the final model, one row each in `results/plan3/attribution.csv`. Work packages renumbered R0-R17.
- Rejected: a neural network as a model (about 5,500 rows; LightGBM already shows flexibility is not the limit) — kept only as an optional control M21, which needs the "no neural networks" rule relaxed first; raw telemetry and "optimal decision" scoring (no public car model; lap times carry the usable part); volatility, diffusion and Mallows-type models.
- Not verified: Jolpica pit-stop endpoint and the meaning of its duration field; OpenF1 `car_data`; the two Amendment-2 references.
- Next: R0 in the background; R1 (audit), R2 (attribution harness and M17), R3 (rolling origin).

## 2026-11-16 — R1: A5 strength audit (run, reviewed by tester and skeptic, revised once)
- Methodology / WP: `methodologies/A5_strength_audit/` (R1). Code in `src/pitlane/plan3/audit.py`; new `lam_a` switch in `LatentModel` (default off). Runs `20261116-2010_*`; outputs in `results/plan3/`.
- Reproduction: with the switch off, dev numbers match plan 1 and plan 2 exactly (`latent_110` rho 0.651014, MAE 3.355; `h0_latent_110` rho 0.605252; teammate correlation +0.538, car correlation +0.920). 62 tests pass (35 new).
- Missing teams: all 13 lineages are now in every car table and figure. The plan-1 figures are not changed.
- One-race drivers at the extremes: not a bug. The driver penalty (lam = 32, prior sd 0.125) is much wider than the spread of established drivers, so one recent race keeps a data weight of about 0.20 (Verstappen 0.95). Camara and Hirakawa have unpenalised values of about -0.32 and +0.30 from one race each.
- Driver spread (43 drivers with 30 or more starts, noise = sigma^2 * sum(w^2) / (sum w)^2): with qualifying in the model the between-driver sd is about 0.023-0.029 (normalised position); in the history-only model (H0) about 0.063. In the qualifying model only Verstappen is distinguishable from the field mean and 0.6% of driver pairs are resolved; in H0, Verstappen and Tsunoda, and 10.5% of pairs.
- Correction of a first-draft claim: the first version of the method card said the driver term in the qualifying model had about zero true variance. That used the wrong noise formula (sigma^2 / sum w). The skeptic re-derived it; the corrected statement is the line above.
- Verstappen, season-end snapshot rank 2015-2026: qualifying model 5, 7, 10, 4, 5, 3, 2, 1, 1, 1, 1, 1; H0 7, 7, 6, 4, 2, 2, 2, 1, 1, 1, 1, 1.
- Face-validity failure: H0 ranks Antonelli (17th of 25 in 2026) below teammate Russell (10th) although he outscored him 359 to 228. Cause: the driver term is one static value with a 50-race half-life, so 2025 dominates.
- Out-of-sample teammate check (strength at the end of season S-1 against the teammate finishing gap in S, 92 pairs): qualifying model r = +0.28 [0.01, 0.49]; H0 +0.39 [0.14, 0.59]. Well below the in-sample +0.54 and +0.62. Per-season values vary widely and are negative in 2022 and 2023.
- The near-zero in-sample teammate correlations for 2014-17 in the final fit are an artefact of decay (those drivers have 2-3 effective starts left); at each season's end they are 0.74 or higher.
- Stronger driver penalty (`lam_a` 256 with qualifying, 128 for H0): removes low-sample drivers from the extremes; dev rho unchanged (-0.0003 [-0.0044, +0.0039]; +0.0013 [-0.0042, +0.0067]); win log-loss worse (+0.00077 [0.00033, 0.00121]; +0.00119 [0.00079, 0.00158]). **Not adopted for prediction.** The rule that chose these values was changed after the first rule's result was seen (the first rule picked the grid edge and flattened the table); they are shown only as a labelled sensitivity. The main tables are the as-frozen model with intervals, with drivers under 10 starts in a separate panel.
- Acceptance check: passes for the main panel by construction of the layout (low-start drivers are shown separately); in an unseparated ranking Camara and Hirakawa are still in 2 of 6 extreme slots.
- Limits: intervals are conditional on sigma = 0.25 and the decay-weighted likelihood, coverage not verified; all strength rankings are in-sample descriptions.
- Consequences for later packages: state driver-car separation per model (overall skill from H0; with qualifying, `a` is race-day gain beyond grid slot and is small); `lam_a` and a season-varying driver term are candidate components under the attribution protocol (PLAN3 §5), not yet run; S6 depends strongly on the driver penalty (se 0.058 to 0.034).
- Next: R2 (attribution harness and M17), R3 (rolling origin).

## 2026-11-16 — R0: lap and pit-stop ingestion (code done and tested; Jolpica laps 2014-2022 still downloading)
- Methodology / WP: R0. Code `src/pitlane/data/ingest3.py`, `configs/ingest3.yaml`, `scripts/ingest_plan3.py`; tests `tests/test_ingest3.py` (82 pass, 3 expected failures documenting that the time parser accepts some non-times that do not occur on disk).
- Sources verified by calling them (2026-11-16): Jolpica laps (100 timings per page, 500 requests per hour), Jolpica pit stops (from 2011), OpenF1 race laps, stints, pit and drivers (2023-2026, no key, CC BY-NC-SA 4.0). TUM database exists (2014-2019) but is not used.
- Downloaded: OpenF1 laps, stints and drivers for all 90 races of 2023-2026; Jolpica pit stops for all 272 races. No failed requests, no unmatched drivers. Tables in `data/interim/plan3/` (git-ignored).
- Not finished: Jolpica laps 2022 back to 2014 run detached at 7.5 s per request (about 5 hours). Resume with `python -u scripts\ingest_plan3.py --laps`; check with `--check`; rebuild tables with `--tables`. The acceptance check (files per season match race counts) is therefore not yet met for laps.
- Facts that change later packages:
  - Jolpica pit-stop duration is total pit-lane time (median 22-25 s), not stationary time. Stationary time exists only in OpenF1 from late 2024 and is incomplete. The M20 crew feature will measure pit lane plus stop, not crew speed alone.
  - Red-flag stoppages appear as pit stops of many minutes and increment the stop counter; 5.6% of stops are flagged `red_flag_suspect` (duration above race median + 20 s), which also catches slow repair stops. Rows are kept; M09 and M20 must filter.
  - Tyre compound and stints exist only for 2023-2026. For 2014-2022 stint age can only come from pit-stop laps, compound unknown.
  - OpenF1 pit data is missing for 2023 rounds 1-6; 43 of 5,256 stint rows have `lap_end < lap_start` and must be dropped.
  - 2023 cross-check, Jolpica against OpenF1: 99.97% of driver-laps agree within 0.05 s outside lap 1; lap-1 times differ by about 0.35 s and are not interchangeable between sources. Jolpica 2023 laps (plan-2 cache) have no position column.
- Not tested: real rate-limit behaviour over the long run; concurrent writers on the request log; laps for 2014-2022 (not yet arrived).

## 2026-11-16 — R2: attribution harness and M17 metric-matched decoding (negative result; revised once after tester and skeptic)
- Methodology / WP: `methodologies/M17_decoding/` (R2). Code `src/pitlane/plan3/attribution.py`, `decode.py`, `calibration.py`; runs `20261116-2028_*`; `results/plan3/attribution.csv` (new, with a `declared` column) and `results/plan3/m17_*.csv`. Dev 2021-23 only (66 races, 1,311 entries). 28 tests in `tests/test_plan3_decode.py` pass.
- Reproduction: S3 re-simulated with the frozen parameters (hash `0d1b7c074d80`) matches the stored plan-2 scores exactly; D0 exact 13.57%, grid 16.31% under the plan-2 tie convention.
- Result (exact-position rate, outcome-free tie-breaks): D0 = D1 13.6%, D2 13.9%, grid order 16.4%. Paired: D2 vs D0 +0.3 points [-1.9, +2.5]; D2 vs grid -2.5 [-6.1, +0.7]; grid vs D0 +2.8 [-0.2, +6.0]. **No measurable difference for every declared pair. The pre-stated failure criterion (D2 not above grid) is met.**
- D2 costs ordering quality against D0: rho -0.056 [-0.075, -0.037], MAE worse by 0.22 [0.11, 0.32].
- D1 is not a separate decoder here: the plan-2 stored score is already -E[position].
- **Correction of the plan (my statement in PLAN3 §2.2):** the plan said the simulator loses to grid order on exact hits because sorting by score is the wrong decoder for that metric. That is not what the data show. The best decoder of this matrix expects 14.9% under the matrix itself, below the 16.3% grid order actually gets. The cause is the matrix: it puts 11.8% on "finishes in the grid slot" against 16.4% realised. Cell by cell it is otherwise calibrated.
- Limits of the grid comparison: "grid order beats the simulator on exact hits" is not established. The interval includes 0 and the gap is almost all 2021 (+8.5 [+2.3, +15.8]; 2022 +1.8; 2023 -1.9).
- Diagnostics (descriptive, no verdicts, not in attribution.csv): with retirement draws off the grid-slot probability rises 11.8% -> 13.5%, about 37% of the gap; mixing the matrix with a point mass on the grid slot gives 15.4-16.5% at weights 0.05-0.30 (in-sample, not tuned out of sample); simulation noise of one 1000-draw run on D2 - D0 is about 1.2 points (four extra seeds, crude).
- Bugs found in review and fixed: `upsert_attribution` wrote preserved holdout columns to the wrong row when one component had two references (tester); ties in D0 and grid utility were broken by input row order, which is finishing order, an outcome leak of under 0.1 points (skeptic). Both tie conventions are kept as separate runs; attribution rows use the outcome-free one. The off-plan diagnostic D0u is marked `declared = no`.
- Not verified: the frozen plan-2 simulator assigns its noise draws by row index, and rows are in finishing order. The draws are independent and identically distributed, so I expect no bias, but this was not tested.
- Skeptic's numbers not reproduced exactly: blend values differ by up to 0.12 points and the grid-slot bins use a different binning; direction and size agree. Not traced.
- Consequences: M17 is not useful on this matrix. The relevant fix is grid-slot attachment (M18 dead-band, or re-fitting simulator noise and stickiness on exact-slot log-loss). S4-S6 are not expected to raise exact hits; the declared S6 x D2 interaction is expected to be negative. Every later matrix must pass `grid_slot_calibration` before D2 is run on it. With 66 dev races, exact-rate gains under about 2 points cannot be resolved.
- The leaderboard rows and the index row for M17 are committed with R3, because both files also hold R3 edits in progress.

## 2026-11-16 — R3: A4 rolling-origin evaluation and rolling re-tune (null result; reviewed by tester and skeptic, card revised once)
- Methodology / WP: `methodologies/A4_rolling_origin/` (R3). Code `src/pitlane/plan3/rolling.py`; one new branch in `harness.split_races` for `seasons:a-b` (named splits unchanged, tested against the previous version). Runs `20261116-2024_*`; outputs `results/plan3/season_table.csv`, `a4_*.csv`, `a4_gain_over_qualifying.csv`. 71 new tests; full suite 243 passed, 3 expected failures.
- Rolling re-tune (hyper-parameters tuned on the three seasons before each season, 12 configurations per model, same budget as plan 1) against the fixed plan-1 settings, dev 66 races, paired: `latent_111` rho -0.0037 [-0.0104, +0.0029]; `gbm_100` +0.0005 [-0.0059, +0.0068]. Within the plan's expectation (PLAN3 §8, within 0.005). **Decision: keep the hyper-parameters tuned on 2021-23.**
- Six paired tests were run; one interval excludes 0: latent MAE worse by 0.043 [0.006, 0.081] under rolling. All of it comes from 2022, the one season where the latent configuration differed.
- What the test can and cannot say: the configurations that differed won their tuning windows by 0.0005-0.0023 rho, and the tester found that optimiser noise alone moves tuning rho by up to 0.0002. So the test mostly measures noise in picking among near-tied settings. It rests on one season for the latent model (2022) and two for LightGBM. It does not test staleness under a regime change larger than those in 2021-23. For LightGBM the comparison is biased toward the fixed settings and the unbiased version was not run; for the latent model the bias does not apply (tuning on 2018-20 picks the frozen settings).
- Season table 2015-2026 (every race predicted from earlier races only; pre-2021 rows use settings tuned on later data and are labelled not clean; 2024-26 is a third look, no decision): stored plan-1 predictions for 2021-26 match exactly (2,695 entries, difference 0.0).
- Model minus qualifying order, rho, paired by race: 2015-20 latent +0.009 [-0.000, +0.018], LightGBM +0.017 [+0.003, +0.032]; dev +0.029 [+0.011, +0.047] and +0.032 [+0.011, +0.053]; 2024-26 +0.002 [-0.008, +0.012] and -0.001 [-0.019, +0.018]. Per season the latent model is above qualifying in 2017, 2019, 2023 and below in 2015; LightGBM above in 2019 and 2023 (24 season intervals, about 1 expected by chance). Without 2017 and 2023 the pooled gain is not distinguishable from 0. No trend over seasons (slopes +0.0007 and -0.0004 per season, p 0.72 and 0.87).
- **Reading:** the models do not decay after 2023; they are refitted before every race. The gain over qualifying order is small everywhere (about 0.01-0.02 rho), largest on the block the settings were tuned on, and absent in 2024-26. The holdout comparison against qualifying order should be expected to be about level.
- Exact-position rate: qualifying order is at or above both models in 7 of 12 seasons; pooled LightGBM minus qualifying -1.8 points [-3.2, -0.6] by race, [-3.6, -0.0] by season.
- Observation, not evidence: the gain over qualifying is about 0 in the first half of seasons and larger in the second half (LightGBM H2 minus H1 +0.032 [+0.017, +0.047], latent +0.013 [+0.000, +0.025], by season). Mostly non-dev seasons, found after looking; a season-start component would have to be declared and tested on dev on its own.
- Rejected: a half-life per regulation period. The short half-life picked for 2022 came from pre-2022 data, won by 0.0005 and lost 0.011 rho in 2022.
- Not verified: the coder's full re-run agreement of 5.5e-12 was not saved to disk; an end-to-end re-run of `run.py` and the full 12-configuration tuning were not repeated by the tester.
- Note: this commit also carries the M17 leaderboard and index rows (R2) and four M11 leaderboard rows written early by the R5 coder; the M11 runs are committed with R5.

## 2026-11-17 — R0 (completion): Jolpica laps 2014-2022 downloaded; one source discrepancy documented
- All 182 Jolpica lap files 2014-2023 are on disk (`--check`: every season's file count equals its race count, 0 failed). One race (2020 round 1) failed once with HTTP 520 and succeeded on resume. Tables rebuilt: laps 322,579 rows, pits 13,197, stints 5,256 (`data/interim/plan3/`, git-ignored).
- One mismatch: 2014 round 4 (China). The lap table has 56 laps; the Jolpica results record laps and race time at lap 54 (the winner's 56 lap times minus laps 55-56 equal the results time 1:33:28.338 exactly). The lap table is kept; it is the only race of 182 with laps beyond the classified distance. `tests/test_ingest3.py` accepts this one case with a comment. The real race distance (56 laps) is from the coder's own knowledge; the files show only the exact time match.
- R0 acceptance is now met for laps.

## 2026-11-17 — R5: M11 likelihood variants (dev only; reviewed by tester and skeptic, revised once)
- Methodology / WP: `methodologies/M11_likelihood_variants/` (R5). Code: three switches in `src/pitlane/models/latent.py` (`finishers_only`, `topk`, `reverse`; off by default, all-off is bitwise identical to the previous version, tested). Runs `20261116-2036_*` (frozen setting), `20261116-2049_*` (tuned), `20261116-2050_*` (interaction). Tests `tests/test_plan3_m11.py` (67). Full suite 353 passed, 3 expected failures.
- Reference `latent_111` re-run equals the stored plan-1 dev run exactly (rho 0.646, MAE 3.355, exact 13.97%).
- Verdicts per primary metric (paired gain vs reference, 95% CI, dev 66 races):
  - finishers-only: rho +0.0014 [-0.0100, +0.0130], MAE +0.065 [-0.005, +0.134], exact +1.9 [-0.3, +4.2] points: no measurable difference on any.
  - top-10: rho -0.0085 [-0.0205, +0.0036], MAE -0.007, exact +1.4 [-0.7, +3.6]: no measurable difference.
  - reverse: rho +0.0062 [-0.0053, +0.0184] none; MAE +0.076 [+0.007, +0.147] and exact +3.0 [+0.9, +5.1] exclude 0. **Reverse is a candidate component, dev only.** Under Holm over the 36 intervals its overall MAE and exact-rate gains do not survive; places 1-3 (+18.2), points log-loss and finishers' MAE do.
- **No variant beats grid order** on exact rate (grid 16.38%): reverse minus grid +0.6 [-3.3, +4.1]; 2021 -6.4, 2022 +1.1, 2023 +7.0 [+2.6, +11.5]. The plan-3 target is not met.
- What the reverse gain is: it repairs a weak reference at the front. The reference is 10.1 points below grid on places 1-3 [-19.7, -0.5] and predicts Hamilton to win 13 of 22 races in 2022 (correct winner calls 7 / 3 / 19 per season; reverse 13 / 13 / 19). This is a defect of `latent_111` and goes to the strength audit.
- Mechanism (descriptive): all three variants fit a strength scale about 1.8 times wider than the reference; the qualifying weight in raw units is 1.7-2.2 times larger, mostly rescaling (weekend share of var(u) 57-59% reference, 52-63% variants). The variants also rank Red Bull 2021 and Verstappen higher before 2022. Not tested causally.
- Tuning (plan-1 grid, 12 configurations each, chosen and judged on the same 66 races, so optimistic): the reference keeps its frozen setting (lam 32, at the grid edge; larger values not tried). No tuned variant has a rho interval excluding 0; reverse tuned rho +0.0183 [-0.0004, +0.0362]. The untuned comparison is the conservative one.
- Declared interaction finishers-only x M07 (frozen S3, same random streams): RPS +0.0007 [-0.0015, +0.0029], no measurable interaction. Confounded: the frozen S3 noise scale fits scores with within-race sd about 0.70, finishers-only scores have 1.24, so its matrix is too sharp. Refitting the scale per variant would change the frozen plan-2 pieces; not done, open decision.
- Review findings and fixes: my R5 brief told the coder to judge on rho only; the plan names rho and MAE as primary and exact rate as headline, so the first verdicts understated reverse (corrected). RPS was missing and the interaction was skipped for a wrong reason (both done in the revision). The card's first finite-difference numbers were untraceable (replaced by `gradient_check.py`, max relative error under 1e-6). The card claimed every fitted-component fit converged to gradient below 1e-5; the finishers-only fit before 2022 stopped at 4.5e-5 (refitted to 9e-9, beta, sd and weekend share unchanged to 3 decimals). Whether the finishers-only walk-forward refits also stop early is unverified. The skeptic left a scratch file `T.pkl` in the repo root; deleted.
- Not tested: the finishers-only + top-k combination cuts by padded slot, not by rank among finishers (not a declared component). Not scored: test blocks, fresh holdout. The holdout must check reverse against grid order as well as `latent_111`.
- Formatting churn: the R5 and R10 upserts rewrote the M17 rows of `attribution.csv` and `leaderboard.csv` at lower float precision; I restored their HEAD text so both files only gain rows.

## 2026-11-17 — R10: M05 ensemble and M12 two-stage Friday forecast (null results; reviewed by tester and skeptic, cards revised once)
- Methodology / WP: `methodologies/M05_ensemble/`, `methodologies/M12_two_stage/` (R10). Code `src/pitlane/models/ensemble.py`, `two_stage.py`, `src/pitlane/plan3/runutil.py`. Runs `20261116-2045_*`, `20261116-2047_*`. Tests `tests/test_plan3_r10.py` (43). References re-run equal the stored plan-1 and plan-2 dev runs exactly.
- M05 (dev 66 races): rank average of `latent_111` and `gbm_100` vs `latent_111` rho +0.0048 [-0.0042, +0.0133], MAE +0.052 [-0.012, +0.117]; vs `gbm_100` rho +0.0019 [-0.0070, +0.0110]; monotone GBM vs `gbm_100` rho +0.0033 [-0.0017, +0.0082]. **No measurable difference on rho, MAE or exact rate.** Points log-loss: rank average vs `latent_111` +0.0101 [+0.0053, +0.0149], but not vs `gbm_100` (+0.0021 [-0.0029, +0.0073]); the finishers' rho and MAE gains vs `latent_111` are the same story. So the ensemble is not better than LightGBM alone. Dev only.
- 471 of 1,311 rank-average entries sit in a mean-rank tie (broken by driverId); other outcome-free tie-breaks move exact 14.4-14.9%. The ensemble's exact-rate point estimate depends on the tie-break.
- No model beats grid order on exact rate (all 1.5-2.4 points below, intervals include 0).
- M12 (21 races of 2023: practice is in our data only from 2023, and race 193 has none): two-stage minus H1 rho +0.0218 [-0.0101, +0.0569], MAE +0.182 [-0.034, +0.391]. **No measurable difference.** The point estimate rests on one race (without race 185 +0.010). On the 44 races of 2021-22 (stage 1 from history only) rho -0.0039 [-0.0174, +0.0095]. Finishers' MAE excludes 0 (secondary, dev only).
- Stage 1 (Ridge, qualifying history + practice) predicts qualifying better than the H1 score order: 66-race rho +0.042 [+0.025, +0.061], MAE +0.340 [+0.216, +0.470]; practice adds +0.012 [-0.007, +0.028]. Qualifying is harder to predict in 2023 (rho 0.64) than 2021-22 (0.78); cause not known.
- Review findings: the card cited a scramble test that was not in the repo (now in `tests/test_plan3_r10.py`, passes); the "45 races of 2021-22" subset was 44 races plus race 193 of 2023 (cards relabelled; the stored csv label `no_practice_2021_22` in M12 results still means those 45 races); the M12 leaderboard row is a 66-race mean (+0.003) while attribution uses 21 races (+0.022); the `M05_rank_avg` leaderboard row is labelled cell 111 but averages cells 111 and 100. The skeptic claimed pit-lane starts have negative `grid_norm`; that is wrong (`ingest.py:113-115` moves them to the back first).
- Observation: cutting the race universe changes two-stage scores by up to 3.5e-4 (optimizer path with fewer parameters), not leakage; altering later rows at fixed universe changes nothing.
- Not done: full M05 walk-forward re-run by the tester (first 3 races only); test blocks; fresh holdout.

## 2026-11-17 — R4: race-pace features and the A3 persistence gate (gate fails as declared; the null is not robust; reviewed by tester and skeptic, card revised twice)
- Methodology / WP: `methodologies/A3_pace_persistence/` (R4). Code `src/pitlane/features/pace.py` (per-race pace table: median clean lap D1 and fuel/stint-corrected fit D2; field-relative offset `rq` = race gap - q_best gap; decayed history `con_rq`, `drv_rq`, `drvtm_rq`, `con_rq_ssn` from earlier races only), `src/pitlane/plan3/pace_gate.py`. Run `20261117-0131_a3_pace_oracle_dev`; outputs `results/plan3/a3_*`. Tests `tests/test_plan3_a3.py` (65; leakage by truncation and scrambling on three races, brute-force history check on 1,500 rows). Full suite 417 passed, 3 expected failures.
- Gating definition: the plan names none; the coder chose D1 (the plan's base feature, D2 is called a variant). The config snapshot shows it set before the run; that the README text was written before the run cannot be verified.
- Results (dev, 65 races; 2021 round 12 has no pace): extended oracle gap on finishers 0.075 [0.042, 0.109] (passes 0.03). Car-level persistence of `rq`, pooled Pearson, half-life 50: D1 0.096 [0.016, 0.182], D2 0.119 [0.044, 0.207]; threshold 0.2. **Gate fails as declared, on both definitions.** Sanity: the same code gives 0.66 persistence for race pace itself and 0.46 for the qualifying gap.
- **The null is not robust (skeptic; post hoc, chosen after seeing the result):** `rq` against q_best (best of Q1/Q2/Q3, coefficient 1) is dominated by this race's qualifying noise (within-race correlation with the same race's qualifying gap -0.48). Against the Q1 gap (the qualifying signal `latent_111` uses; missing Q1 left missing, as the gate does for q_best) car persistence is 0.268 [0.14, 0.45] D1 / 0.393 [0.28, 0.51] D2; per-race regression residual 0.282 / 0.471. Spearman 0.146, lag-1 0.119, winsorised 0.183 / 0.201. With the car's earlier race-pace gap controlled, the partial correlation falls to about 0-0.09: the persistent signal is car race pace in earlier races, which `latent_111` does not contain. The gate outcome depends on a reference the plan did not fix.
- A first post-hoc run by the coder gave 0.184 for the Q1 row because `entries2.q1_gap` imputes a missing Q1 time as 0.10 (`ingest.py` 118-120); reconciled, both sides now agree to the third decimal.
- Gap split (D1, coefficient 1): adding the persistent q_best-based offsets lowers rho (-0.069 car, -0.047 driver); this shows that this one predictor hurts, not that nothing learnable from earlier races explains the gap.
- Data findings (tester): q_best outliers up to 45.6% (2023 R2) and >30% (wet 2016 R11); dropping the 38 entries with |rq_c| > 10 raises D1 persistence to 0.156 [0.077, 0.226], still below 0.2. Fuel coefficients of 0.25-0.61 s/lap (about ten times plausible) in 8 of 203 fitted races, 3-4 in dev, likely drying tracks: D2 numbers, including its 0.119 oracle gap, are affected; the D1 gate is not. norris 2019 R18 and alonso 2023 R19 have laps but no `entries2` row. The 26 dev entries without laps are retirements (18 accidents or collisions), not source gaps.
- Leaderboard: one A3 row (oracle rho on finishers), kept as an oracle row; plan 2 kept its oracle off the leaderboard, so this is an open choice for the user.
- Consequence: per the plan M09 (R6) is not built and M18 applies its dead-band to the model's strength difference. **Open decision for the user:** amend the plan and run M09 as a post-gate exploratory component on a definition fixed before running (car race-pace history plus the Q1-based offset), judged on dev and counted only if it holds on the fresh holdout.
- Process note: the plan says to run A3 first; R5 and R10 were committed before it. This does not change any number.
- This commit also carries the M19 index row and nine M19 leaderboard rows (R13, under review); the M19 code and runs follow with R13.

## 2026-11-17 — R13: M19 in-race forecast (dev only; reported separately from pre-race results; revised twice after tester, skeptic and researcher)
- Methodology / WP: `methodologies/M19_inrace/` (R13). Code `src/pitlane/features/inrace.py` (lap-k race state), `src/pitlane/models/stage2.py` (frozen pre-race score + in-race terms), hand table `data/external/scheduled_laps.csv` (66 dev rows, sourced). Final runs `20261117-0225_m19v2b_*`. Tests `tests/test_plan3_m19.py` (95). Full suite 512 passed, 3 expected failures.
- Horizons H4 after lap 1, H5 at 25% and H6 at 50% of the distance; baseline = the running order at that lap (derived from cumulative Jolpica lap times, one source per race; matches stored positions in 2021-22 except 2022 R18 after the red flag).
- Main model (literal reading of the plan): the frozen walk-forward `latent_111` / `gbm_100` pre-race score plus five in-race columns in a second stage (latent: PL on c*score + beta*x; LightGBM: score + x). The first version instead refitted the whole model jointly; it is kept as a labelled variant and is not distinguishable on rho. `lam_x` tuned on dev (18 configurations).
- Results, paired gain over the running order (dev, 66 races):
  - rho: H4 latent +0.028 [0.011, 0.047], gbm +0.039 [0.024, 0.056]; H5 +0.028 [0.013, 0.044] / +0.030 [0.015, 0.048]; H6 +0.012 [0.003, 0.020] / +0.010 [0.004, 0.018].
  - Exact-position rate: worse than the running order at all six pairs (two intervals exclude 0); the deficit is on running cars, not retirees.
  - MAE: +0.04 to +0.16 places, 3 of 6 intervals exclude 0.
- **Reading:** the running order is a strong exact-rate baseline that the score-sorted model does not beat; sorting by score moves cars off the place they hold (the model keeps only 24-46% of cars in their current slot), the same mechanism as the M17 grid-slot finding. The H4/H5 rho gains come from 2022-23 only (2021 null). H6 is marginal and not stable across seasons; `lam_x` selection optimism is about a fifth of the H6 gain. 54 verdict rows, all dev only.
- Review findings and fixes:
  - Leak (skeptic): the red-flag pit flag used the race-wide median duration, so stops after lap k changed lap-k `ir_pits` (3-6 dev races per horizon). Fixed: the flag is recomputed from stops at lap <= k. The acceptance test catches the old behaviour (checked by monkeypatching it back).
  - Race distance was taken from the race's own last lap (post-race; wrong for 2021 Spa, 2022 Monaco, Singapore, Suzuka). Now the distance in force at lights-out, from the hand table: Monza 2023 51 (cut after the aborted start, before lights-out), Monaco 2022 77 (cut by formation laps before the rolling start); cuts during the race do not count. A pre-race model would use 53 and 78. Researcher verified the values (FIA final classifications for full-distance rows; F1 race reports and Wikipedia for the cut races); Spa 2021 = 44 rests on Wikipedia plus the 2022 FIA classification. Only Spa 2021 is capped (H5/H6 at lap 1). Dropping the four shortened races moves each rho gain by at most +0.004.
  - "Frozen pre-race score" was first implemented as a joint refit (a design reinterpretation); now literal.
  - The coder fetched Wikipedia pages itself to build the table, which is the researcher's role; client name not checked. The researcher then verified the values.
- Superseded runs on the leaderboard (append-only): `m19_*` (v1, leaky flag) and `m19v2_*` (identical numbers to v2b; Monaco 77 vs 78 gives the same horizon laps).
- Not tested: the two-stage forecast leakage used a proxy pre-race score (stage 1 never reads laps); two-stage `lam_x` tuning not re-run by the tester. A NaN lap time within the first k laps would rank a car ahead (0 cases in dev).
- Next step, not run: a dead-band / stickiness-to-current-position decoder judged on exact rate (same idea as M18), tuned on dev, counted only if the fresh holdout confirms it.
