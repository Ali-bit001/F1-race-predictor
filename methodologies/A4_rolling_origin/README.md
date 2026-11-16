# A4 — Rolling-origin evaluation

**Status:** done (R3)
**Plan reference:** PLAN3.md §4 "A4", §5, §6 R3
**Factors:** `latent_111` (dynamic, decomposition, ranking all on), `gbm_100` (dynamic only), and the qualifying-order heuristic (n/a — baseline)

## Hypothesis
Models already refit before every race; only the hyper-parameters are fixed. Expected before running: tuning them on the
three seasons before each dev season gives no measurable difference to the fixed 2021-23 values (|Δρ| < 0.01, 95% CI
including 0). A CI excluding 0 in either direction would be the finding. The season table is descriptive.

## Method
1. **Season table.** Mean per-race Spearman ρ, MAE and exact-position rate for qualifying order, `latent_111` and `gbm_100`,
   per season 2015-2026 (2026 = rounds 1-20, the last race in the data; nothing after it is scored), first half / second half /
   whole season. Half = rounds 1..n//2 vs the rest, n = last round in the table for that season. Walk-forward, refit before every
   race (`refit_every=1`), hyper-parameters from `configs/frozen.yaml` (M03 cell 111, M04 cell 100).
   Labels: before 2021 "not clean" (the hyper-parameters were tuned on later data); 2021-23 "dev" (tuned on these seasons);
   2024-26 "third look, no decision" (the old test blocks, already seen twice).
2. **Rolling re-tune.** For S in 2021, 2022, 2023: grid search on the three seasons before S (walk-forward inside them,
   `refit_every=4` while tuning exactly as in plan 1, 15-race warm-up from earlier races, best mean ρ), same search space as
   plan 1 (`pipeline.grid_for`: 12 configs per model, budget ≤ 30), same seeds (defaults); then predict S with the chosen
   config and `refit_every=1`. The three seasons are concatenated into 66 dev races and compared per race with the fixed
   hyper-parameters (plan-1 dev run): paired difference rolling − fixed, 95% bootstrap CI (5000 resamples, seed 0).
3. **Decision rule** (PLAN3 §4 A4.3, §5): rolling re-tune is adopted for the final model iff the paired ρ gain on dev has a
   CI above 0; otherwise hyper-parameters stay as tuned on 2021-23.

**Bias.** The comparison favours the fixed hyper-parameters: they were chosen on 2021-23 itself (best of 12 configs on those
same races), while the rolling ones never saw the season they predict. A null or negative paired difference is therefore
weaker evidence against rolling re-tuning than a positive one would be evidence for it. The rule is applied as written.

**Reused vs re-run.** 2021-2026 (R20) rows reuse the stored plan-1 predictions (`dev`, `test_stable`, `test_shift` runs of
`latent_111`, `gbm_100`, `quali`); their per-race metrics are recomputed from the stored scores with the unchanged
`race_metrics` (the plan-1 `per_race_metrics.csv` files predate the exact-position metric). 2015-2020 are re-run
(one continuous walk-forward 2015-2020, warm-up from 2014). **Verification:** all of 2021-2026 was also re-run in full for all three
models, 134 races each: max |difference| in ρ, MAE, exact, RMSE, Kendall, NDCG, ρ_fin, MAE_fin = 0.0; entry scores differ by at
most 5e-12 (latent) and 0.0 (gbm, quali), 2695 of 2695 entries matched. Only the stored-prediction match (2695/2695 entries,
difference 0.0) can be verified from files on disk; the 5e-12 figure for the full re-run was printed during the run and was
not saved. Only the calibrated probabilities (p_win etc.) of
the stored runs differ in principle (online τ warm-up); they are not used here.

## Leakage notes
No new features. Models see only races before the one predicted (the harness is unchanged in behaviour; I added the split
spec `seasons:a-b`, which only chooses which races are counted; named splits are untouched, and the reproduction above
confirms identical numbers). Tuning windows end before the season predicted. Seasons 2015-2020 use hyper-parameters tuned
on later data (no leakage into the race scores' inputs, but not clean as a hyper-parameter estimate).

## How to run
```
python methodologies/A4_rolling_origin/run.py --config methodologies/A4_rolling_origin/config.yaml   # ~2.3 min
python methodologies/A4_rolling_origin/run.py --quick                                                 # smoke test
```
Outputs: `results/plan3/season_table.csv`, `results/plan3/figures/a4_season_table.png`, `results/plan3/a4_paired_dev.csv`,
`results/plan3/a4_summary.json`, `results/plan3/a4_tuning_{latent_111,gbm_100}.csv`, five run folders under `results/`, five
rows in `results/plan3/leaderboard.csv` (column `cell` = model cell: 111, 100, 0 for qualifying), and the gain over qualifying
order, `results/plan3/a4_gain_over_qualifying.csv` and `figures/a4_gain_over_qualifying.png`. The last two can be rebuilt from the
stored run folders without refitting: `python methodologies/A4_rolling_origin/run.py --postprocess-only`.

## Results
Rolling re-tune vs fixed, 66 dev races (Δ = rolling − fixed, 95% paired bootstrap CI):

| Model | ρ fixed | ρ rolling | Δρ [95% CI] | ΔMAE [CI] | Δexact [CI] |
|---|---|---|---|---|---|
| latent_111 | 0.6460 | 0.6422 | -0.0037 [-0.0104, +0.0029] | +0.043 [+0.006, +0.081] | -0.001 [-0.014, +0.011] |
| gbm_100 | 0.6489 | 0.6494 | +0.0005 [-0.0059, +0.0068] | -0.002 [-0.042, +0.038] | +0.006 [-0.007, +0.019] |

Chosen hyper-parameters (fixed in brackets):

| Model | 2021 (tuned 2018-20) | 2022 (2019-21) | 2023 (2020-22) |
|---|---|---|---|
| latent_111 (λ=32, halflife 50) | λ=32, hl 50 (same) | λ=8, hl 20 | λ=32, hl 50 (same) |
| gbm_100 (leaves 4, 200 trees, lr .05, mcs 40, hl 32) | leaves 7, 150 trees, lr .05, mcs 40, hl 32 | leaves 4, 200, .05, 40, hl 32 (same) | leaves 7, 150, .05, 40, hl 32 |

Per-season Δρ: latent 2021 0.000, 2022 -0.011, 2023 0.000; gbm 2021 -0.003, 2022 0.000, 2023 +0.005.

Season table (whole-season ρ; halves, MAE and exact rate in `results/plan3/season_table.csv`):

| Season | qualifying | latent_111 | gbm_100 | races | status |
|---|---|---|---|---|---|
| 2015 | 0.656 | 0.629 | 0.642 | 19 | not clean |
| 2016 | 0.650 | 0.659 | 0.669 | 21 | not clean |
| 2017 | 0.569 | 0.607 | 0.607 | 20 | not clean |
| 2018 | 0.597 | 0.603 | 0.616 | 21 | not clean |
| 2019 | 0.631 | 0.661 | 0.673 | 21 | not clean |
| 2020 | 0.616 | 0.607 | 0.608 | 17 | not clean |
| 2021 | 0.657 | 0.680 | 0.677 | 22 | dev |
| 2022 | 0.588 | 0.597 | 0.597 | 22 | dev |
| 2023 | 0.606 | 0.661 | 0.672 | 22 | dev |
| 2024 | 0.750 | 0.755 | 0.748 | 24 | third look, no decision |
| 2025 | 0.652 | 0.651 | 0.643 | 24 | third look, no decision |
| 2026 (R1-20) | 0.710 | 0.711 | 0.720 | 20 | third look, no decision |

Paired gain over qualifying order (model minus qualifying, per race; 95% bootstrap interval, 5000 resamples, seed 0;
race-level resampling within a block, season-cluster resampling where marked). Full table with exact-position rate:
`results/plan3/a4_gain_over_qualifying.csv`.

| Block (label) | latent_111 Δρ | gbm_100 Δρ | races |
|---|---|---|---|
| 2015-20 (not clean) | +0.009 [-0.000, +0.018] | +0.017 [+0.003, +0.032] | 119 |
| dev 2021-23 (dev) | +0.029 [+0.011, +0.047] | +0.032 [+0.011, +0.053] | 66 |
| 2024-26 (third look) | +0.002 [-0.008, +0.012] | -0.001 [-0.019, +0.018] | 68 |
| all seasons, by season | +0.012 [+0.001, +0.024] | +0.016 [+0.004, +0.030] | 253 |
| all except 2017, 2023, by season | +0.005 [-0.004, +0.014] | +0.009 [-0.001, +0.019] | 211 |

Season intervals (race-level) that exclude 0 in ρ: latent_111 above qualifying in 2017, 2019, 2023 and below in 2015;
gbm_100 above only in 2019 and 2023. H2 minus H1 gain in ρ (by season): gbm_100 +0.032 [+0.017, +0.047] (all seasons),
+0.037 [+0.022, +0.053] (excluding dev); latent_111 +0.013 [+0.000, +0.025] and +0.012 [-0.001, +0.024]. Linear trend of the
season gain on season: latent +0.0007 per season (p 0.72), gbm -0.0004 (p 0.87).

Exact-position rate, gain over qualifying: pooled gbm_100 -0.018 [-0.032, -0.006] at race level, [-0.036, -0.000] by season;
latent_111 -0.009 [-0.021, +0.003] at race level. Qualifying order is at or above both models in 7 of 12 seasons.

Run folders: see `results/a4_summary.json` (`run_ids`) and `results/plan3/leaderboard.csv` (method A4).

## Decision
Neither model's rolling re-tune has a paired ρ gain with a CI above 0 (latent -0.004, gbm +0.0005, both CIs include 0 and
both |Δρ| < 0.01), so **the final model's hyper-parameters stay as tuned on 2021-23**. This agrees with the expectation
stated before the run. The bias noted above means the result does not prove that rolling windows are useless, only that no
gain could be shown even before accounting for it in the other direction. The `latent_111` MAE is worse by 0.043 positions
with a CI excluding 0; MAE is secondary here and ρ is the decision metric, so it is reported, not acted on.

## What we learned
- No measurable staleness: the tuning surface is flat at the top (the picks that differed from the fixed configuration won by
  only 0.0005-0.0023 tuning ρ), and the search picks the 2021-23 values in 2 of 3 seasons for latent_111 and 1 of 3 for
  gbm_100. The one differing latent pick (λ=8, hl 20 for 2022) lost 0.011 ρ on 2022.
- The latent_111 MAE effect (+0.043 [+0.006, +0.081], rolling worse) comes entirely from 2022, the one season where its
  configuration differed (per-season mean ΔMAE: 2021 0.000, 2022 +0.128, 2023 0.000). Six paired tests were run (2 models x
  ρ, MAE, exact); one interval excludes 0.
- Season-to-season variation of ρ (0.527-0.774 per half-season) is far larger than anything the hyper-parameters change;
  the season table is descriptive and half-season values are means of 8-12 races.
- Gain over qualifying order (ρ): latent_111 is above qualifying in 2017, 2019 and 2023 and below it in 2015; gbm_100 is above
  only in 2019 and 2023. There are 24 season intervals, so about 1 is expected to exclude 0 by chance. Over all seasons the
  gain is about 0.01-0.02 ρ; it is not distinguishable from 0 without 2017 and 2023, absent in 2024-26, and largest on dev (the
  tuning block, so part of it is likely selection). No trend over time (slopes above).
- Exact-position rate: qualifying order is at or above both models in 7 of 12 seasons; pooled gbm_100 minus qualifying is
  -0.018 [-0.032, -0.006] at race level and [-0.036, -0.000] by season. Nothing here is a decision (2024-26 are third looks).
- Hypothesis only: the gain in ρ is larger in the second half of a season than in the first (gbm_100 +0.032
  [+0.017, +0.047], latent_111 +0.013 [+0.000, +0.025]). This is descriptive, comes mostly from non-dev seasons, and is not
  evidence for any model component.

## Limits
- The single weakest point: the rolling-vs-fixed test rests on one season per model for latent_111 (2022, n=22, the only
  season whose configuration differed) and two for gbm_100 (2021, 2023), with configurations separated by near-tie tuning
  margins. It does not test staleness under a larger regime shift.
- The bias toward the fixed configuration (see Bias) does not apply to latent_111: tuning on 2018-20 picks the frozen
  configuration. For gbm_100 the unbiased comparison (leaves 7 / 150 trees on 2022) was not run, and no decision depends on it.
- The pre-stated threshold is the plan's (PLAN3 §8: Δρ within ±0.005); both results (-0.004 and +0.0005) are within it.

## Open questions / next tries
- A rolling window longer than three seasons, or tuning with the final-model features after R14, is not tested here.
- `quick` mode is a smoke test only (2-config grids, one season, no files).
