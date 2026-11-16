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
