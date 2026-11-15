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
