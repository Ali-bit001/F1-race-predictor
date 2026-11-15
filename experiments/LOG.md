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
