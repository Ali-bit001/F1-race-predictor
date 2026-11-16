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
