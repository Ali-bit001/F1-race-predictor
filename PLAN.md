# PitLane — Research Plan

Source proposal: [`docs/proposal/PitLane_RedBull_Custom_Car_slim.pptx`](docs/proposal/PitLane_RedBull_Custom_Car_slim.pptx)

Roles: this plan is the specification. The coding model implements it one work package at a time and
records what happened in `experiments/LOG.md`. Changes to the research design (splits, metrics, the
ablation definition) are made here first, never silently in code.

---

## 1. What we are testing

**Research question (from the deck, unchanged):**
Can explicit separation of car strength, driver strength and circuit effects improve *out-of-time*
F1 race-order prediction?

**The three contributions under test — these stay the core of the project:**

| # | Factor | Claim | OFF | ON |
|---|--------|-------|-----|----|
| F1 | **Dynamic performance** | Weighting recent races more lets the model track changing competitiveness | All history weighted equally | Time-decayed / online-updated |
| F2 | **Driver–car separation** | Estimating driver and constructor effects separately beats one entangled signal | One strength per driver (car absorbed into it) | Separate driver + constructor (+ circuit) effects |
| F3 | **Structured ranking** | Optimising the whole finishing order beats per-driver position regression | Pointwise regression on position | Listwise ranking likelihood, coherent P1–PN output with probabilities |

**Prediction cutoff:** before the race starts. Qualifying, practice and the starting grid are allowed;
nothing from the race itself is.

**Hard constraints**
- CPU only, no dedicated GPU. No neural networks, no LLM training. Every model must fit in minutes.
- Results must be explainable: every prediction should decompose into named contributions.
- No random train/test shuffling across time, anywhere.

**What success looks like:** a measurable, statistically supported improvement on unseen future races.
A null or negative result on any of the three factors is a valid finding and is reported as such.

---

## 2. Design decisions (and why)

### 2.1 A 2×2×2 factorial instead of only the A→D ladder
The deck's ladder (A direct → B +dynamic → C +decomposed → D +ranking) only measures each factor in
one context, so it cannot tell "ranking helps" from "ranking helps only once decomposition is in".
All 8 on/off combinations cost seconds each, so we run the full factorial. The deck's A, B, C, D are
four of the eight cells and are still reported as the headline ladder.

| Cell | F1 dyn | F2 decomp | F3 rank | Deck name |
|------|:-:|:-:|:-:|---|
| 000 | – | – | – | **A — Direct** |
| 100 | ✓ | – | – | **B — Dynamic** |
| 110 | ✓ | ✓ | – | **C — Decomposed** |
| 111 | ✓ | ✓ | ✓ | **D — Proposed** |
| 010, 001, 011, 101 | | | | extra cells (main effects + interactions) |

### 2.2 Two model families, so the conclusion is not an artefact of one model
- **Family L — latent strength model (primary, the "proposed" model).** A small additive model where
  the three factors are literal switches in the likelihood. Cleanest possible ablation and
  explainable by construction.
- **Family G — gradient-boosted trees (robustness check).** The same three switches applied to a
  conventional strong model (feature set / sample weights / objective). Answers "do the ideas still
  help when bolted onto what everyone else already uses?"

If a factor helps in both families, the claim is strong. If only in one, we say so.

### 2.3 Data scope and splits
| Block | Seasons | Use |
|-------|---------|-----|
| Burn-in | 2014–2020 | Fit / warm up only. Never scored. |
| Development | 2021–2023 | Walk-forward scoring for hyperparameter tuning and model selection. Contains the 2022 regulation reset, so adaptivity is tuned on a real regime change. |
| Test — stable | 2024–2025 | Locked. Scored once, after all choices are frozen. |
| Test — regime shift | 2026 | Locked. Reported **separately**: new regulations and an 11th team make this the direct test of gap G4 (generalisation when the order changes). Use every race completed at the data-freeze date. |

Evaluation is **walk-forward (prequential)**: to predict race *r*, a model may use everything that
happened strictly before race *r* starts — including earlier test races. Hyperparameters are frozen
from the development block. This mirrors real use.

Start at 2014 (hybrid era) so old eras don't dominate; older data is an optional sensitivity check.

### 2.4 Target
- Primary target: official classification order of all starters (Ergast `positionOrder`), DNFs
  included in their classified positions.
- Sensitivity analysis: same metrics on classified finishers only. DNFs are mostly noise and can
  swamp small model differences, so both views are always reported.
- Field size varies (20 or 22). Position errors are reported raw and field-size-normalised.

### 2.5 Metrics
Primary (pre-declared, used for the headline claim): **mean per-race Spearman ρ** and **position MAE**.

Secondary: RMSE, Kendall τ, NDCG@10, winner / podium (top-3) / points (top-10) accuracy,
log-loss and Brier score for P(win), P(podium), P(points), reliability curves + ECE,
ranked probability score over the full position distribution.

### 2.6 Statistical protocol
- Unit of analysis is the race. All comparisons are **paired per race**.
- 95% CIs by bootstrap over races (10 000 resamples); Wilcoxon signed-rank as a second opinion.
- Factorial main effects and interactions estimated from per-race metrics with a race-level
  blocking term.
- Stochastic models run with 5 seeds; report mean and spread.
- Every variant gets the **same tuning budget** (same grid size, same dev block).
- The test block is run once. If something is changed after seeing test numbers, that is logged in
  `experiments/LOG.md` and stated in the report.

---

## 3. Data

| Source | What | Notes |
|--------|------|-------|
| Jolpica-F1 API (Ergast successor) | Results, qualifying, grid, sprint results, status, circuits, driver/constructor IDs, standings | Core dataset. Rate-limited — check current limits, throttle, cache every response to `data/raw/`. Also reachable via `fastf1.ergast`. |
| FastF1 | Practice and qualifying lap times, session weather, tyre compounds (2018+) | Enable the on-disk cache. Slow first download; run once. |
| Hand-made table | Circuit descriptors: street/permanent, length, corner count, full-throttle share / power sensitivity, overtaking difficulty | `data/external/circuits.csv`, small, committed, with a source column. |
| Hand-made table | Regulation-change markers (2017, 2022, 2026) and team lineage (rebrands map to one constructor lineage) | `data/external/`, committed. |

**Known traps — each needs a test or an explicit handling note:**
1. Pit-lane starts are `grid = 0` in Ergast.
2. Sprint weekends have fewer practice sessions, and the 2021 sprint set the Sunday grid.
3. Grid ≠ qualifying order (penalties). Both are legitimate pre-race features; keep them distinct.
4. Team rebrands (e.g. Toro Rosso → AlphaTauri → RB) must not reset car strength.
5. Mid-season driver swaps and one-off substitutes; rookies have no history (cold start).
6. Missing FP data pre-2018 and for cancelled sessions — models must tolerate missing weekend signals.
7. Red-flagged / shortened races and double-points oddities: keep, but flag.
8. Disqualifications after the race change the classification; use the final official result.

**Leakage guard (mandatory test):** for a sample of races, build features from the full dataset and
from a dataset truncated just before that race. The two feature rows must be identical.

---

## 4. Methodologies

Each lives in `methodologies/<ID>_<name>/` with a method card, config, entry script and its own
`results/`. See `methodologies/README.md` for the index and the card template.

### M00 — Heuristic baselines (deck Baseline 1)
Predict finishing order = (a) starting grid, (b) qualifying order, (c) current championship order,
(d) previous-race finishing order, (e) season-to-date mean finish.
Grid order is expected to be very hard to beat; every later result is read against it.

### M01 — Direct GBM regression (deck Baseline 2)
Random Forest and LightGBM regressing finishing position on a conventional feature set
(grid, quali gap to pole, driver/constructor rolling form, circuit history, practice pace).
Order obtained by sorting predictions. This mimics public systems A–C in the deck.

### M02 — GBM ranking (deck Baseline 3)
Same features, LightGBM `lambdarank` (and XGBoost `rank:pairwise` if cheap), race = query group.

### M03 — Latent strength model, Family L (the proposed model + primary factorial)
Latent pace for entry *i* in race *r*:

```
s(i,r) = driver_i(t) + car_k(t) + fit(k, circuit_type) + β · weekend(i,r)
```

- `driver`, `car`: latent strengths. Identified by teammate comparisons and by drivers changing teams.
- `fit`: constructor × circuit-type interaction (shrunk hard toward 0).
- `weekend`: standardised quali gap, grid, practice long-run pace, wet flag. Missing → 0 with an indicator.

Switches:
- **F1 dynamic** — OFF: all past races weighted equally. ON: exponential time decay on the likelihood
  (half-life tuned on dev), separate half-lives for driver and car, plus extra shrinkage toward the
  mean at season boundaries. Online (Elo-style) updating is an alternative implementation to compare.
- **F2 decomposition** — OFF: one strength per driver, no car/fit terms. ON: full equation above.
- **F3 ranking** — OFF: Gaussian regression of (normalised) finishing position on `s`. ON:
  Plackett–Luce likelihood over the whole finishing order.

Fit by penalised maximum likelihood (L-BFGS in SciPy); a few hundred parameters, seconds per fit.
Probabilistic output: sample finishing orders from the fitted model → full driver × position
probability matrix, which is coherent by construction (each position assigned exactly once).

Explainability is free: each prediction is a sum of named terms
("car +1.1, driver +0.6, circuit fit −0.2, qualifying +0.4").

### M04 — GBM factorial, Family G (robustness check)
- **F1** — OFF: expanding all-history aggregates, uniform sample weights. ON: exponentially weighted
  aggregates + time-decayed sample weights.
- **F2** — OFF: raw driver and constructor aggregates (confounded: a driver's average reflects their car).
  ON: replace them with de-confounded driver and car strengths taken from M03's decomposition
  (computed walk-forward, so no leakage).
- **F3** — OFF: regression objective. ON: `lambdarank`; probabilities via Plackett–Luce sampling on scores.

### M05 — Hybrid (only if M03 and M04 disagree or both leave headroom)
M03 latent strengths + weekend signals into a small ranking GBM, or a simple average of M03 and M02
scores. Kept optional so the core study stays clean.

---

## 5. Explainability deliverables

1. **Additive breakdown per prediction** (M03): table + waterfall chart for any driver/race.
2. **Strength trajectories**: driver and car strength over time with regulation changes marked.
3. **Face-validity checks**: driver strengths vs. teammate qualifying/race head-to-heads; car
   strengths vs. constructors' standings. A decomposition that fails these is not trusted, whatever its score.
4. **Driver-move case studies**: what the model predicted when a driver changed teams vs. what an
   entangled model predicted.
5. **SHAP (TreeExplainer)** for Family G, global and per-race.
6. **Factorial effect plot**: main effect and interaction of F1/F2/F3 with CIs — the single figure
   that answers the research question.
7. **Error anatomy**: where the model loses to the grid baseline (wet races, safety cars, first-lap incidents, DNFs).

---

## 6. Work packages

Do them in order. Each ends with a commit, a `LOG.md` entry and its acceptance check passing.

| WP | Deliverable | Acceptance check |
|----|-------------|------------------|
| **WP0** Scaffold | Repo structure, plan, conventions *(done in the initial commit)* | — |
| **WP1** Data ingestion | `src/pitlane/data/`: cached download of 2014–2026 results, qualifying, sprints, circuits; FastF1 session summaries 2018+; one tidy `entries.parquet` (one row per driver per race) | Row counts per season match known race/entry counts; schema test; traps 1–8 handled or flagged |
| **WP2** Evaluation harness | `src/pitlane/evaluation/`: split definitions, walk-forward runner, all metrics, bootstrap CIs, run writer, leaderboard updater | Metrics unit-tested on hand-computed toy races; M00 grid baseline runs end-to-end on dev |
| **WP3** Features | `src/pitlane/features/`: rolling/expanding/EWMA form, circuit history, weekend signals | Leakage truncation test passes |
| **WP4** Baselines | M00, M01, M02 on dev | Results in leaderboard; grid baseline numbers sanity-checked by hand on 2 races |
| **WP5** Latent model | M03 with the three switches; parameter-recovery test on simulated data | Recovers known driver/car strengths on synthetic data; all 8 cells run on dev |
| **WP6** GBM factorial | M04, 8 cells on dev | All 8 cells in leaderboard |
| **WP7** Freeze + test | Freeze configs, tag the commit, run everything once on 2024–2025 and 2026 | Tagged commit; test results written; no config edits after tag |
| **WP8** Analysis + report | Stats, figures from §5, `docs/report.md` answering the three questions | Every claim in the report traces to a file in `results/` |
| **WP9** Extensions | See §7 | Per extension |

**Build the harness before any model (WP2 before WP4).** A trustworthy scoreboard first; otherwise
every later number is suspect.

**Tuning budget per variant:** ≤ 30 configurations, chosen on dev mean Spearman ρ.

---

## 7. Extensions (after WP8, in rough priority order)

| ID | Extension | Why it is worth doing |
|----|-----------|----------------------|
| X1 | **Forecast horizons**: pre-weekend (no session data) vs. post-practice vs. post-qualifying | Shows where the decomposition matters most — likely before qualifying, when history is all there is |
| X2 | **Regime-aware dynamics**: inflate uncertainty / reset car strength faster at regulation changes (2022, 2026) | Directly attacks gap G2; 2026 is a live test |
| X3 | **DNF model**: separate reliability/incident hazard per driver and car, combined with the pace model | DNFs are the biggest error source; separates "slow" from "unlucky/fragile" |
| X4 | **Circuit clustering**: learn circuit types from data instead of the hand-made table | Makes the circuit effect less arbitrary |
| X5 | **Bayesian version** of M03 (PyMC or NumPyro on CPU, small data) | Credible intervals on driver/car strengths; honest uncertainty |
| X6 | **Conformal prediction** intervals on finishing position | Distribution-free coverage guarantee |
| X7 | **Wet-weather driver effect** | Classic "driver vs car" question |
| X8 | **Rookie cold start** using junior-series results as a prior | Fixes a known weak spot |
| X9 | **Season simulation**: Monte Carlo championship probabilities | Natural downstream use of calibrated race probabilities |
| X10 | **Market benchmark**: compare against bookmaker odds if a clean source exists | The toughest real-world baseline |
| X11 | Dashboard / what-if tool | Last, and only on top of validated models |

---

## 8. Risks

| Risk | Mitigation |
|------|-----------|
| Grid order is so strong that all models tie with it | Report X1 horizons; pre-weekend forecasting is where modelling can add value. A tie is still a reportable result. |
| Driver and car effects poorly identified (few team changes) | Parameter-recovery test on synthetic data; regularisation; face-validity checks in §5 |
| Too few test races for significance | Pre-declared primary metrics; paired tests; also report pooled dev+test walk-forward as secondary |
| Overfitting to the dev block through repeated tries | Fixed tuning budget; test block run once; log every attempt including failures |
| 2026 season incomplete at freeze | Use completed races; state the count |
| API rate limits / outages | Cache raw responses; ingestion is resumable |

---

## 9. Prior work to read and cite (verify each reference before citing)

- Phillips (2014) — driver performance adjusted for team and competition effects.
- Bell, Smith, Sabel & Jones (2016) — multilevel modelling of driver vs. constructor performance.
- Henderson & Kirrane (2018) — time-weighted Plackett–Luce models for forecasting F1 results.
- van Kesteren & Bergkamp (2023) — Bayesian disentangling of driver skill and constructor advantage.
- The deck's sources: Prakoso et al. (2026), F1 Race Intelligence, PitWall Intel, SilverSouls predictor, FastF1.

The deck already says the three ideas are not unprecedented; these papers are why. Our contribution
is the controlled, out-of-time test of their combination against strong conventional baselines.
