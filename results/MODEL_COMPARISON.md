# PitLane — which model is best, by stage (written 2026-11-18, after plan 3)

**Short answer:** no model is *confirmed* better than a simple baseline yet. The fresh holdout (races after 2026 R20) has
n = 0, and the pre-registered tests need n ≥ 20. Everything below is from the development block (2021–23, 66 races, where
models were tuned) and from the 2024–26 "third look" (68 races, already seen, carries no decision weight). Picks are the
best-supported choice per stage, not proven winners.

Metrics: Spearman rho between predicted and actual order (higher is better), MAE in positions (lower), exact = share of drivers
whose place is predicted exactly. Intervals: paired per-race bootstrap, 10,000 resamples, seed 0, 95%.

## 1. Pick per stage

| Stage | Use for point order | Use for exact hits | Use for probabilities | Strength of evidence |
|---|---|---|---|---|
| Before qualifying (H1) | `h1_latent_111` (plan-2 frozen) | — | — | weak: about level with simple baselines |
| After qualifying (H2) | `final` (plan-3 frozen); `reverse_only` equal | **grid order** | `final` + S3 simulator (refit theta) | small gain over grid on 2024–26; not confirmed on holdout |
| After lap 1 (H4) | M19 two-stage `gbm_s2` | running order | — | exploratory, dev only, tuned on the same races |
| At 25% distance (H5) | M19 two-stage `gbm_s2` or `latent_s2` | running order | — | exploratory, dev only |
| At 50% distance (H6) | running order (M19 gain +0.01, marginal) | running order | — | exploratory, dev only |

## 2. Numbers behind the picks

### Before qualifying (H1, dev 2021–23) — `results/plan2/leaderboard.csv`, `results/leaderboard.csv`
| Model | rho | MAE | exact |
|---|---|---|---|
| `h1_latent_111` (frozen) | 0.6106 | 3.710 | 0.0947 |
| quali-order baseline (needs qualifying, so not available at H1; shown for scale) | 0.6170 | 3.506 | — |
| championship standings baseline | 0.6394 | 3.383 | — |

The frozen H1 model is **below** the standings baseline on dev. M12 (rho 0.6139) is not frozen and is inside the noise.
On 2024–26, H1 rho 0.6197, MAE 3.780. Honest reading: before qualifying, nothing we built beats "order by current standings".
The H1 model is kept because it is the frozen one, and because it outputs a full score vector the H2 model builds on.

### After qualifying (H2)
Dev 2021–23 (`results/plan3/leaderboard.csv`):

| Model | rho | MAE | exact | within ±1 | within ±2 | podium | win |
|---|---|---|---|---|---|---|---|
| `final` | **0.6632** | **3.236** | 0.1512 | 0.3759 | **0.5443** | 0.6465 | 0.6364 |
| `reverse_only` | 0.6522 | 3.279 | 0.1697 | 0.3767 | 0.5370 | 0.6768 | 0.6818 |
| `latent_111` (plan 1) | 0.6460 | 3.355 | — | — | — | — | — |
| plan-1 `gbm_001` / `rf_000` | 0.6614 / 0.6582 | — | — | — | — | — | — |
| grid order | 0.5918 | 3.595 | 0.1631 | 0.3617 | 0.5120 | 0.6667 | 0.5455 |

Dev T1 (`final` − `latent_111`, rho): +0.0172 [−0.0002, +0.0340]. The interval includes 0, so it does not pass.

Third look 2024–26 R20, 68 races (`results/plan3/third_look.csv`):

| Model | rho | MAE | exact |
|---|---|---|---|
| `reverse_only` | **0.7093** | **3.082** | 0.1504 |
| `final` | 0.7058 | 3.129 | 0.1414 |
| `latent_111` | 0.7052 | 3.120 | 0.1458 |
| grid order | 0.6832 | 3.176 | **0.1624** |

- Gain over grid, rho: `reverse_only` +0.0261 [0.0094, 0.0450]; `final` +0.0226 [0.0025, 0.0450]. This is the one comparison
  that holds on unseen-at-tuning seasons, but 2024–26 has been looked at, so it is not a clean test.
- `final` − `reverse_only`: −0.0036 [−0.0108, +0.0037]. `final` − `latent_111`: +0.0006 [−0.009, +0.010]. **The three model
  variants cannot be told apart.** The plan-3 additions (reverse, pace history) did not replicate their dev gain.
- Exact rate: grid order is never beaten (final vs grid −0.0210 [−0.0487, +0.0057]; places 1–3 pooled −0.0686
  [−0.1471, +0.0098]). If you want the most exact hits, predict the grid.
- Probabilities: S3 simulator points log-loss (dev refit) `final` 0.4671 vs `latent_111` 0.4861 vs simulator D0 0.4888.

**Why `final` is the pick and not `reverse_only`:** `final` is the frozen, pre-registered model, and swapping on the 2024–26
look would be choosing on seen data. The pre-registration already states that if T3 fails on the holdout, `reverse_only`
becomes the better-supported model (the frozen model is still not swapped). In practice the two give near-identical orders.

### During the race (M19, dev 2021–23 only) — `methodologies/M19_inrace/results/means_dev.csv`, `paired_gains_dev.csv`
| Horizon | running order rho / MAE / exact | `latent_s2` | `gbm_s2` | rho gain over running order (gbm_s2) |
|---|---|---|---|---|
| H4 after lap 1 | 0.679 / 3.10 / **0.191** | 0.707 / 3.00 / 0.153 | **0.718 / 2.97** / 0.168 | +0.039 [0.024, 0.056] |
| H5 25% | 0.742 / 2.66 / **0.244** | 0.770 / 2.53 / 0.213 | **0.772 / 2.51** / 0.220 | +0.030 [0.015, 0.048] |
| H6 50% | 0.846 / 1.97 / **0.313** | **0.858 / 1.89** / 0.297 | 0.857 / 1.93 / 0.264 | +0.010 [0.004, 0.018] |

Caveats: the in-race penalty `lam_x` was tuned on these same 66 races, so the gains are optimistic; the H4/H5 gains come
from 2022–23 only; exact rate is **worse** than the running order at every horizon; there is no 2024–26 or holdout check.
By H6 the running order already carries almost all the signal.

## 3. Comparison with published / online F1 predictors

Most public projects report classification accuracy (podium yes/no, winner yes/no) on a few races with unclear splits, so
they cannot be compared to our full-order metrics. Status: **verified** = page opened and the number read there;
**unverified** = number from a search snippet only.

| Source | What it is | Reported | Comparable? | Status |
|---|---|---|---|---|
| Weissbock & Mills, arXiv 2507.10966 | correlation study, 2010–23, top 20, sprints excluded | qualifying vs finish rho 0.763; grid vs finish 0.748; 2014–22 era 0.777; FP3 vs finish 0.676 | partly: their grid rho (0.748) is the baseline ours (0.59 on 2021–23, 0.68 on 2024–26) corresponds to. Different seasons and row rules (we keep all classified finishers; their DNF handling is unclear) | verified (DNF rule unverified) |
| INFO 523 student project (2025) | linear regression on Ergast 1994–2024 | MAE 3.32 on a 2019–24 test | roughly: our `final` MAE 3.24 (dev), 3.13 (2024–26); grid alone 3.18 on 2024–26 | **unverified** (page did not load) |
| fourcast substack (Aug 2024) | XGBoost race simulation | no metrics | no | verified |
| IRJET ranking model | ranking model | 52% exact on 3 races of 2024 | no: 3 races; our exact rate is ~15% over 66–68 races | unverified |
| Portfolio / app pages (e.g. rhianneg.github.io, mintlify "Formula-1-prediction") | RF / XGB classifiers | ~90% podium accuracy, top-3 accuracy ~0.86 | no: binary accuracy is dominated by the 17 "not podium" drivers | unverified |
| JCSI 2026 | category classification | 97% | no | unverified |
| Erdős Institute project | qualifying prediction | 12% exact, 33% within ±2 | different target (qualifying); for scale, our race H2 exact 15%, within ±2 54% | unverified |

Reading: the one verified, comparable baseline (the arXiv grid–finish correlation) says grid order alone gives rho ≈ 0.75 on
their sample; our models add ≈ +0.02 rho over grid on 2024–26. No public source we found reports a pre-registered holdout, so
the large accuracy figures online should not be read as beating this project. That is also not evidence that this project
is better; the comparisons are mostly not like for like.

Sources:
- https://arxiv.org/abs/2507.10966
- https://info-523-f25.github.io/final-project-team-1/
- https://fourcast.substack.com/p/modeling-formula-one-getting-to-the
- https://issuu.com/irjet/docs/a_ranking-based_prediction_model_fo_9732dcfc3bccb1
- https://ph.pollub.pl/index.php/jcsi/article/download/8462/5371/41814
- https://rhianneg.github.io/projects/project1_Race-Predictor/race-predictor.html
- https://erdosinstitute.org/_files/ugd/1baa6a_34b6ce49a94d4dc1b5a4211f60a1dc8e.pdf
- https://mintlify.com/TomandJerry0811/Formula-1-prediction/resources/model-performance

## 4. What would change these picks
- Holdout reading at n ≥ 20 (first eligible race 2026 R21): if T3 fails, `reverse_only` becomes the stated better-supported
  H2 model; if T1 fails, plan 3 adds nothing over plan 1.
- An in-race check on 2024–26 or the holdout with `lam_x` fixed in advance; until then M19 is exploratory.
- A pre-qualifying model that beats the standings baseline on dev with an interval excluding 0.

The chosen configurations are copied, unmodified, into `best_models/`.
