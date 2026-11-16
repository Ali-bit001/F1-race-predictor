# PitLane — Research Plan 3: race pace, car development and a clean test

Follows [`PLAN.md`](PLAN.md) and [`PLAN2.md`](PLAN2.md). Everything not changed here still applies: CPU only,
no neural networks, explainable predictions, walk-forward evaluation, paired per-race statistics, no
temporal leakage, the working rules in `CLAUDE.md`, and the run conventions in
`.claude/skills/pitlane-run/SKILL.md`.

**Amendment 1 (2026-11-16, before any plan-3 code or run).** Eight further ideas were reviewed; the
verdict on each is in §2.1. They add A4, A5, M14, M15 and M16, extend M10, add the agent roster in §10
and renumber the work packages to R0–R13.

**Amendment 2 (2026-11-16, before any plan-3 code or run).** Exact-position accuracy becomes a
headline metric with starting-grid order as the baseline to beat; the evidence and the review of six
further ideas are in §2.2. Adds M17–M21, extends M09 and M14, adds the attribution protocol in §5
(every change is tested on its own), and renumbers the work packages to R0–R17.

---

## 1. Why a third plan

Plan 2 result (`results/plan2/REPORT.md`, frozen config `0d1b7c074d80`):

| Finding | Evidence |
|---|---|
| Extra weekend information did not improve the finishing order | Test 2024–25: qualifying order 0.701, plan-1 `latent_111` 0.703, plan-2 latent with all new information 0.701 (Spearman ρ). No information group passed the decision rule |
| The simulator improved probabilities, not order | Ranked probability score on test 2024–25: 0.1218 (plan 1) to 0.1093 (M08 S3); ρ unchanged at 0.702 |
| Race pace is the one measured gap in order accuracy | On 22 races of 2023, finishers ranked by their median race lap score 0.830 against the model's 0.757 (gap 0.073). Practice pace did not capture it |
| Retirements are mostly luck | Knowing them would add 0.197 ρ, but the retirement model's AUC is 0.55 on dev and its dev gain over a constant rate has an interval spanning 0 |
| Modelling matters most before qualifying | Test 2026, history only: all factors on 0.567 against 0.374 off; with practice 0.655; qualifying order 0.710 |
| The test blocks are used up | 2024–2026 have been scored in plan 1 and again in plan 2 |

**Research question 3:** does information about *race* pace, as distinct from one-lap pace, and about car
development improve the predicted finishing order beyond qualifying order, on races nobody has scored?

The three plan-1 factors stay in every model (cell `111` is the base), and the plan-2 simulator stays
the probability layer on top.

---

## 2. What plan 3 tests

| # | Question | Tested by | Priority |
|---|---|---|---|
| Q1 | Is a car's race pace relative to its qualifying pace persistent from race to race? | Persistence analysis (A3) | Gate, run first |
| Q2 | Does a race-pace offset learned from earlier races improve the order? | Race-pace model (M09) | **Primary hypothesis** |
| Q3 | Do declared car upgrades predict changes in car strength? | Upgrade signal (M10) | Secondary |
| Q4 | Does a likelihood that ignores or down-weights the noisy back of the field fit better? | Likelihood variants (M11) | Secondary, cheap |
| Q5 | Before qualifying, is "predict qualifying, then the race" better than predicting the race directly? | Two-stage Friday forecast (M12) | Secondary |
| Q6 | Does averaging the latent model and LightGBM help? | Ensemble (M05, reserved in plan 1) | Secondary, cheap |
| Q7 | How do we compare with the betting market? | Odds collection (M13) | Data collection only |
| Q8 | Do hyper-parameters tuned once on 2021–23 go stale, and how does accuracy move season by season? | Rolling-origin evaluation (A4) | Secondary, cheap |
| Q9 | Are the estimated driver and car strengths believable, and is every team shown? | Strength audit (A5) | **Run first**, no new data |
| Q10 | Do measurable car traits (engine supplier, straight-line speed against cornering) predict where a car is strong? | Car traits (M14) | Secondary |
| Q11 | Does a junior-series record give a better starting estimate for a rookie? | Rookie prior (M15) | Low; scored on rookies only |
| Q12 | Do better noise models improve the simulator's probabilities? | Simulator extensions (M16) | Secondary, scored on RPS, not ρ |

### 2.1 Review of the ideas added in Amendment 1

| # | Idea as proposed | Verdict | What goes into the plan |
|---|---|---|---|
| 1 | Train through one season, test its last races, continue to the next; do not train to 2023 and test on 2025 | **Already done, more strictly.** The harness refits before *every* race on all earlier races (`walk_forward`, `refit_every=1` in final runs), so a 2025 race is predicted by a model fitted through the race before it. Only the *hyper-parameters* (half-lives, penalties) were fixed on 2021–23 | A4: report every season 2015–2026 split into early and late races, and test re-tuning the hyper-parameters each season on the seasons before it |
| 2 | Relate car specifications to win rate | **Mostly not feasible.** The specifications that decide pace (downforce, drag, power) are secret; published ones (weight, wheelbase) barely differ. Win rate is a poor target (few cars ever win) | M14: two traits that *are* observable: engine supplier, and a straight-line against cornering profile from timing data. Target is the existing car strength, not wins |
| 3 | Verstappen is ranked badly | **Partly right.** Checked in stored walk-forward strengths: the dynamic model ranks him 1st from 2022 on (mean rank 1.8 in 2022, 1.2 in 2023, 1.0 in 2025–26), but 9th–11th in 2017–18 and 3.6 in 2021; the static model never has him above 3rd. The final chart ranks him 1st of 30. Real defects found on the way: a driver with **one** race (Hirakawa) is ranked 2nd and another (Camara) last, and no uncertainty is shown. Plan 2 never re-checked or plotted strengths | A5 |
| 4 | Some teams are missing from the analysis | **Correct, plotting bug.** `report.py` keeps the 8 lineages with the most rows, which drops McLaren (536 rows against 537), Haas, Cadillac, Manor and Caterham. The model itself fits all 13 | A5: every lineage in every plot and table |
| 5 | Use sub-agents (Sonnet 5.5) | Done outside the plan: definitions in `.claude/agents/` | §10 |
| 6 | Use F3 and F2 history for drivers | **Small.** It can only matter for rookies in their first races (about 2–5 drivers a season), and after qualifying the grid slot already says most of it. Not enough rookies (about 25 since 2014) to measure an effect on overall ρ | M15: a hand-made table, scored on rookies' first 10 races and before qualifying. No scraping |
| 7 | NLP on documents | **Mostly no.** About 1,000 short FIA documents is too few to learn text features, and news text was already rejected. The FIA forms have fixed fields, so keyword rules get the content without a language model | M10 extended: component category and stated reason, by keyword rules |
| 8 | More advanced stochastic processes for the simulator | **Useful for probabilities, not for order.** ρ is limited by information, not by the sampler; plan 2 showed the simulator moves RPS only. Lap-level processes (Hawkes incident clustering, semi-Markov position chains) need the lap simulator and stay deferred | M16: four race-level extensions taken from the horse-racing and rating literature |

### 2.2 Exact-position accuracy and the ideas added in Amendment 2

A target of 25% exact positions before the race was proposed. Measured on existing files before
planning anything (dev 2021–23, 66 races, 1,311 entries, unless stated):

| Check | Exact-position rate |
|---|---|
| Starting-grid order | 16.3% |
| Plan-2 simulator S3 (sorted by its score) | 13.6% |
| Model order, told in advance who retires and where they are classified | 28.0% (18.2% on finishers) |
| Grid order by finishing band: places 1–3 / 4–6 / 7–10 / 11–15 / 16+ | 35.9 / 19.2 / 12.1 / 12.1 / 10.3% |
| Running order after lap 1 (2023 only, 22 races; grid 14.2% on the same races) | 18.6% |
| Running order at 25% / 50% / 75% of the distance (2023 only) | 22.0 / 31.9 / 42.5% |
| Standard error of the per-race mean on dev | about 1.1 points |

Consequences: 25% before the race is at the level of an oracle that knows every retirement, so it is
not adopted as a pre-race target. The current model is *below* grid order on this metric. 25% is
crossed during the race, between a quarter and half distance. Differences of about one point are noise.

**Targets set instead:** before the race, beat grid order on exact-position rate on dev with a paired
interval excluding 0 (a realistic level is 17–19%); in the race, reach 25% by quarter distance.

| # | Question | Tested by | Priority |
|---|---|---|---|
| Q13 | Does choosing the predicted order to maximise expected exact hits beat sorting by score? | Metric-matched decoding (M17) | **Run early**, no new data |
| Q14 | Does a pace threshold for passing improve on a linear pace effect? | Overtaking threshold (M18) | Secondary |
| Q15 | How fast does accuracy grow once the race is running? | In-race forecast (M19) | Secondary, reported separately from pre-race results |
| Q16 | Do lap-level driver traits and pit-crew speed add anything? | Race-craft and pit crew (M20) | Secondary, small expected effect |
| Q17 | Does a small neural network do better than the current models on the same inputs? | Control (M21) | Optional, needs a rule change |

| Idea as proposed | Verdict | What goes into the plan |
|---|---|---|
| A small neural network or another deep-learning method | **Not expected to help.** There are about 5,500 rows and 260 races; large models work because of very large data. LightGBM, already a flexible non-linear learner, did not beat the linear latent model or qualifying order, which says the limit is information, not model flexibility. It also conflicts with the "no neural networks" rule and weakens the exact decomposition | M21: one cheap control (scikit-learn MLP, CPU, no new dependency) so the question is settled by a number. Runs only if the rule in `CLAUDE.md` is relaxed for this control |
| Telemetry to judge drivers against an "optimal" decision | **Not as proposed.** The optimum needs a car model, which is not public. Lap time is already the sum of everything in the telemetry, and teammate qualifying gap (same car) is already a feature. Volume is large (roughly half a million samples per car-session set) and, if available at all, 2023 onward only (OpenF1 `car_data`, **unverified**) | M20: three driver traits that lap times do give: lap-time consistency, tyre degradation against the teammate, places gained on lap 1 |
| Pit-stop times | **Available, small.** Jolpica lists each stop with a duration (believed to be total pit-lane time, **verify**). Typical gaps between crews are a few tenths of a second; what costs places is the occasional slow stop | M20: crew median time loss and slow-stop rate, from earlier races. The stops are needed anyway to split stints for the corrected pace |
| Decision-matched prediction, overtaking threshold, in-race forecast, corrected pace, lap-1 gain, learned car × circuit factors (proposed in review) | Adopted | M17, M18, M19, M09 extension, M20, M14 part 3 |
| Test every change on its own | Adopted; it was only partly true before (ladders inside each method) | §5 attribution protocol |

### Decided against, with reasons

| Idea | Decision | Reason |
|---|---|---|
| News and social-media scraping, sentiment features | Not built | No study found showing it helps in F1 (evidence is from football and hockey, with modest gains). Qualifying already reflects most of what news reports. Article dates are unreliable, which is a leakage risk, and text features need a language model. The FIA upgrade documents (M10) are the dated, official, structured version of the same signal |
| Lap-by-lap race simulator (tyres, pit stops, caution periods) | Deferred to plan 4, gated | Only worth its cost if M09 shows that race pace adds order accuracy. The TUM `race-simulation` project (Heilmeier et al. 2020) is the template |
| Bayesian (MCMC) version of the latent model | Not built | van Kesteren and Bergkamp (2023) use the same rank-ordered logit structure; it would add credible intervals, not accuracy |
| Better retirement prediction as a route to order accuracy | Not pursued | Plan 2 shows the predictable part is small. M07 is kept as is |
| Betting odds as a model input | Not used | Odds are a benchmark. Using them as a feature would make the comparison circular |
| Car specification sheets, win rate as the target (idea 2) | Not built | Pace-relevant specifications are not public; see M14 for the observable substitute |
| Learned text features or a language model on documents (idea 7) | Not built | Too few documents; keyword rules on the fixed FIA fields are enough (M10) |
| Scraping full F2/F3 result histories (idea 6) | Not built | About 25 rookies in the data; a hand table of their last junior season is enough to test the idea (M15) |
| Hawkes, semi-Markov or other lap-level processes (idea 8) | Deferred with the lap simulator | They model events inside a race, which the race-level simulator does not represent |
| A neural network as the main model, or a sequence model on laps or telemetry | Not built | Too little data at race level; lap-level deep models need a GPU and give up the decomposition. Only the M21 control is allowed |
| Raw telemetry features; "optimal decision" scoring of drivers | Not built | No public car model to define the optimum; lap times carry the usable part (M20) |
| Volatility (GARCH), diffusion and option-pricing models, Mallows or spin-type ranking models | Not built | They describe the size of the uncertainty, not reduce it; the part that helps probabilities is already in M16 |
| 25% exact positions as a pre-race target | Not adopted | At oracle level (§2.2) |

---

## 3. Data additions

| Source | What | Status | Notes |
|---|---|---|---|
| Jolpica laps | Race lap times, 2014–2022 | Verified in plan 2 (2023 fetched) | About 12 requests per race and a limit of about 500 requests per hour: roughly 180 races, 4 to 5 hours, cached and resumable. Run it in the background first |
| OpenF1 | Race lap times and stints, 2023–2026 | Practice endpoint verified; **verify the race sessions before relying on it** | Falls back to Jolpica laps if not usable |
| TUM Formula 1 timing database | Lap times 2014–2019 | **Unverified** (seen only in a search listing) | Optional accelerator for the Jolpica download; check the licence |
| Jolpica pit stops | Lap and duration of each pit stop, all seasons | **Verify** the endpoint, its coverage and what "duration" measures | About one request per race. Used to split stints (corrected pace, M09) and for crew features (M20) |
| FIA "Car Presentation Submissions" PDFs | New components each team declares before first practice, per event | 2024 documents seen; **verify 2022–2023 are still downloadable** | Rule exists since 2022. Parsing needs a PDF text library (`pypdf`, light); if parsing is unreliable, fall back to a hand-made table in `data/external/upgrades.csv` |
| Bookmaker or prediction-market prices | Win and podium prices before each race | No free historical archive found | Collect going forward only. **Check the source's terms of use first** |
| Engine supplier per team per season | Hand table `data/external/power_units.csv` (about 130 rows) | To be written by hand | All seasons; works or customer team flagged |
| OpenF1 lap fields | Speed-trap and sector times per lap, 2023–2026 | **Verify the field names and coverage** | Qualifying and practice sessions only, so the trait is known before the race |
| Junior-series record of rookies | Hand table `data/external/rookies.csv`: last junior series, final championship position, seasons spent there, age at debut | To be written by hand (about 25 drivers) | Each row needs a source link; values are known before the debut |

### New features (all per driver per race, all from earlier races or from before the race)

| Group | Features |
|---|---|
| Race pace of earlier races | Per driver per past race: median clean lap (lap 1, in-laps, out-laps and laps above 107% of the race median dropped; at least 10 clean laps) as a % gap to the best median in that race |
| Corrected race pace (variant) | Per past race, a least-squares fit `lap time = driver pace + fuel term × laps remaining + tyre term × stint age`, on clean laps; the driver pace replaces the median lap |
| Driver race-craft | From earlier races, decayed and shrunk: lap-time spread on clean corrected laps relative to the teammate; tyre-degradation slope relative to the teammate; places gained on lap 1 relative to grid slot |
| Pit crew | Per car lineage from earlier races: median pit-lane time relative to the field at that race; share of stops slower than the field median by 3 s or more |
| In-race state (M19 only) | At lap k: running position, gap to the car ahead, pit stops made, laps completed. Uses laps 1..k only |
| Race-minus-qualifying offset | `rq = race pace gap − qualifying gap` for each past race. Time-decayed, shrunk means per car lineage (`con_rq`) and per driver (`drv_rq`), with the half-lives already used in plan 1 |
| Offset × overtaking | `con_rq × (1 − stick)`: the offset should matter more where the grid is less sticky (plan-2 circuit stickiness) |
| Upgrades | Components declared by the team at this event; decayed sum over recent events; both relative to the field mean at that event. Split by keyword rules into category (floor, wings, bodywork, other) and stated reason (performance, circuit-specific, reliability) |
| Car traits | Engine supplier (grouping); straight-line index and slow-corner index from earlier events' qualifying and practice laps, each as a decayed mean relative to the field (2023 onward) |
| Trait × circuit | Car index × the circuit's own index (share of the lap at full throttle, estimated from earlier years' timing data) |
| Rookie record | For drivers with fewer than 10 F1 starts: final position in the last junior season, seasons spent there |
| Predicted qualifying | Out-of-sample qualifying position predicted from history and practice (M12 only) |

**Leakage test:** extend `tests/test_leakage_plan2.py` to the race-pace features: blanking a race's laps and
result must not change that race's features.

---

## 4. Analyses and methodologies

### A5 — Strength audit (run first; no new data)
The explainability claim rests on the strengths being believable, and plan 2 did not re-check them.

1. **Every team.** All 13 car lineages in every plot and table; remove the "largest 8" filter.
2. **Uncertainty.** Each driver and car strength gets an interval from the curvature of the fitted
   objective (Laplace approximation). Drivers with fewer than 10 starts are shown separately.
3. **Shrinkage check.** Find out why one-race drivers sit at the extremes (penalty too weak, or a bug)
   and fix the cause, not the chart. Any change to the penalty is a tuned hyper-parameter (dev only).
4. **What `a` means.** With qualifying in the model, `a` is a driver's race-day gain beyond their grid
   slot, not overall skill. Report a second table from the history-only model (H0), where `a` carries
   overall skill, and say which is which.
5. **Named checks.** Season-by-season rank of the eventual champion and of the champion's teammate;
   teammate and car correlations (plan 1: +0.54 and +0.92) recomputed per season.
6. **Identification.** A driver who never changes team is separated from the car only through
   teammates. List drivers by the number of distinct teammates and teams; flag weakly identified ones.

Output: `results/plan3/strengths.csv` (every driver and lineage, every season, with intervals).
If step 3 finds a bug that changes predictions, plan-1 and plan-2 numbers are re-stated next to the
originals, never overwritten.

### A4 — Rolling-origin evaluation
The models already refit before every race. What stays fixed is the hyper-parameters.

1. **Season table.** ρ for qualifying order, `latent_111` and `gbm_100` for every season 2015–2026,
   split into the first and second half of the season. Seasons before 2021 are labelled "not clean":
   the hyper-parameters were tuned on later data.
2. **Rolling re-tune.** For each season S, tune on the three seasons before S (same budget) and
   predict S. Compare with the fixed hyper-parameters on dev seasons 2021–2023.
3. **Decision.** If rolling re-tune passes the decision rule on dev, the final model's
   hyper-parameters are tuned on the three seasons before the freeze; otherwise they stay as tuned on
   2021–23. Either way the choice is made before the freeze.

### A3 — Is the race-pace offset persistent? (gate, run first)
On the development block, once laps for 2014–2023 are cached:

1. **Extended pace oracle.** Repeat the plan-2 oracle on all 66 development races instead of 22.
2. **Persistence.** Correlate each car's offset `rq` in a race with its decayed mean over earlier races;
   the same per driver. Report by season.
3. **Split of the oracle gap.** How much of the gap between the model and the oracle is explained by
   the car-level offset, the driver-level offset, and the race-specific remainder.

**Gate:** M09 is built only if the extended oracle gap is at least 0.03 ρ on finishers **and** the
car-level persistence correlation is at least 0.2 on dev. If the gate fails, that is the answer to Q2,
it is reported as a null result, and work moves to M10–M12.

### M09 — Race-pace model (primary)
The plan-2 latent model (cell `111`, plan-1 weekend signals) plus `con_rq`, `drv_rq` and the stickiness
interaction in the additive weekend term, ridge-penalised as in plan 2. The same columns go to LightGBM.
Ladder: reference → + car offset → + driver offset → + stickiness interaction.
Each prediction still decomposes exactly, with a new named part: "race pace versus qualifying pace".
Pace definition is tested as its own switch: median clean lap against the fuel- and tyre-corrected fit
(§3). The persistence numbers of A3 are reported for both.

### M17 — Metric-matched decoding (no new data)
The simulator gives, for each driver, the probability of each finishing position. Three ways of turning
that into one predicted order are compared on the same simulations:

| Decoder | Rule | Best for |
|---|---|---|
| D0 | Sort by score (current) | — |
| D1 | Sort by expected position | Position error, ρ |
| D2 | The assignment of drivers to positions with the largest sum of probabilities (`scipy.optimize.linear_sum_assignment`) | Exact-position rate |

Nothing is fitted, so there is nothing to tune. Reported for the plan-2 simulator first, then for
every later model. Each metric is reported with the decoder meant for it and with D0.

### M18 — Overtaking threshold
A pace advantage only turns into a pass above some size, which differs by circuit. The linear term
`con_rq × (1 − stick)` is compared with a dead-band version: the pace term between a car and the car
ahead on the grid counts only above a threshold `d · stick[circuit]`, with one fitted `d`. If A3 fails
and there is no pace offset, the same dead-band is applied to the model's strength difference.

### M19 — In-race forecast (reported separately)
Horizons H4 (after lap 1), H5 (25% of the distance) and H6 (50%). Model: the frozen pre-race score
plus the in-race state features, in the latent model's additive term and in LightGBM. Baseline at each
horizon: the running order itself. Needs laps for all seasons (R0). These numbers are never mixed
with or compared against pre-race results in one table without the horizon stated.

### M20 — Race-craft and pit crew
Four features entering the weekend term one at a time: driver consistency, driver tyre degradation,
driver lap-1 gain, pit-crew time loss (with slow-stop rate). Before any model run, each gets the same
persistence check as A3 (correlation of a race's value with the decayed mean of earlier races); a
feature with persistence below 0.2 on dev is reported as not persistent and not run.

### M21 — Neural-network control (optional)
`sklearn.neural_network.MLPRegressor`, one or two hidden layers of at most 32 units, early stopping,
five seeds averaged, same inputs and target as `gbm_100`, same tuning budget, dev only. It is a
control, not a candidate: it cannot enter the final model, because it has no exact decomposition.
**Runs only after the "no neural networks" rule in `CLAUDE.md` is relaxed for this one control.**

### M10 — Upgrade signal
Upgrade features enter the same weekend term. Evaluated at two horizons, because qualifying should
already reflect an upgrade that works: H2 (with qualifying) and H0/H1 (before qualifying), where the
expected effect is larger. Covers 2022 onward only.
Ladder: total count → + category → + stated reason. Categories and reasons come from keyword rules on
the form's fixed fields; the rule list is committed and spot-checked against documents by hand.

### M11 — Likelihood variants (latent model, cell `111`)
Three switches, each with an analytic gradient and a finite-difference test:

| Variant | Definition | Source of the idea |
|---|---|---|
| Finishers-only | Retirees are removed from the ranking likelihood (treated as censored); M07 handles them in the simulator | Our own finding that retirements are mostly luck |
| Truncated top-k | Only the first k places contribute (k = 10) | Henderson and Kirrane (2018) |
| Reverse | The order is built from last place upward | Extended Plackett–Luce paper (arXiv 2002.05953) |

### M12 — Two-stage Friday forecast
Stage 1 predicts qualifying position from history and practice. Stage 2 feeds the predicted position
into the frozen H2 model. Compared with the direct pre-qualifying model (plan-2 H1) on races with
practice data (2023 onward). Also reports how well qualifying itself is predicted.

### M05 — Ensemble
Average of the within-race ranks of the latent model and LightGBM (equal weights, no tuning), and
LightGBM with monotone constraints so that a better qualifying position or grid slot never predicts
a worse finish.

### M14 — Car traits
Two parts, tested separately.

- **Engine supplier pooling (all seasons).** Car strength gets a shared supplier term:
  `b[car] = e[supplier, season] + own part`, both penalised. Expected to matter mainly at a regulation
  change, when there is no history for the new cars (2026; 2014 is burn-in). Dev 2021–23 contains one
  chassis rule change (2022) and no engine change, so dev can show little; this is stated in advance.
- **Trait × circuit (2023 onward).** Replace or supplement the hand-labelled car × circuit-type term
  `g` with car straight-line index × circuit full-throttle share. Only one dev season has the data,
  so this is exploratory and cannot enter the final model on dev evidence alone.
- **Learned car × circuit factors (all seasons).** One or two latent factors per car-season and per
  circuit, fitted to the residuals of earlier races (a low-rank interaction, as in factor models),
  compared with the hand-labelled circuit types. About 24 races per car-season, so little is expected.

### M15 — Rookie prior
The driver penalty currently pulls every new driver toward the same value. Here a rookie's starting
value is `c · (junior record score)`, with one coefficient `c` fitted with the model. Scored only on
rookies' first 10 starts, at H0/H1 and H2. With about 25 rookies the interval will be wide; the
result is reported as descriptive unless it is clear.

### M16 — Simulator extensions (probabilities)
Added to the plan-2 simulator one at a time on top of S3, each fitted on dev by RPS as in plan 2.

| Step | Change | Why | Source of the idea |
|---|---|---|---|
| S4 Position-dependent discount | The Plackett–Luce temperature differs by stage (1st, 2nd, 3rd, rest), fitted | The plain model is known to over-state favourites' chances of the minor places | Harville (1973); Henery (1981); Stern (1990); Lo and Bacon-Shone; Benter (1994) |
| S5 Normal performance draws with team shocks | Finishing order from `u_i + τ·η_team + σ_i·ε_i`: a shock shared by teammates, and a spread `σ` that may differ by experience and by wet or dry | Teammates' results move together (strategy, car problems); Plackett–Luce assumes independence | Thurstone-type ranking models |
| S6 Parameter uncertainty | Each simulated race first draws strengths from their Laplace interval (A5), wider at the start of a season and after a rule change | The simulator treats estimated strengths as exact, which over-states confidence early in a season | State-space rating models (Glickman's dynamic paired comparison; TrueSkill Through Time) |
| S7 Mean-reverting season carry-over | Car strength carried to a new season as `φ · last season + noise`, `φ` fitted, with larger noise at a rule change | Tests whether fields converge under stable rules; replaces one fixed decay | Ornstein–Uhlenbeck / AR(1) process |

S5 also needs a joint metric, since a team shock changes pairs, not single drivers: log-loss of "both
cars of a team score points". S7 changes the latent model itself and is scored on ρ as well as RPS.

### M13 — Market benchmark (collection only)
A small script stores the available win and podium prices before each race from now on. No result is
expected inside plan 3; once 20 or more races are collected, the simulator's probabilities are scored
against the market's with log-loss.

### Final model
Whichever of M09–M11, M14 (supplier pooling, learned factors), M15, M18 and M20 pass the decision rule are
combined into one pre-race model, decoded with M17;
the simulator (M08 plus the M16 steps that pass) is re-fitted on top of it using dev only; the
hyper-parameter window follows the A4 decision; and that combination is frozen.

---

## 5. Evaluation

### Splits
| Block | Seasons | Use in plan 3 |
|---|---|---|
| Burn-in | 2014–2020 | Fit only |
| Development | 2021–2023 | **All decisions**: tuning, the gate, which variants enter the final model |
| Test — stable, regime shift | 2024–2025, 2026 rounds 1–20 | Third look. Reported for continuity and labelled as such; **not used for any decision** |
| Fresh holdout | Every race run after the plan-3 freeze | **The only clean test.** Reported separately, with its race count |

The plan-2 frozen models and qualifying order are scored on the same fresh races as references, so
the comparison between plans is also clean. With few fresh races the intervals will be wide; the
holdout keeps growing through 2027 and the report is regenerated by re-running the script.

### Metrics
- **Primary for order:** mean per-race Spearman ρ and position MAE; for M09 also ρ on finishers, because
  that is where the oracle gap was measured.
- **Primary for probabilities:** ranked probability score and log-loss of P(points finish).
- **Headline for exact positions (Amendment 2):** exact-position rate, with starting-grid order as the
  baseline, reported overall, on finishers, and for places 1–3, 4–10 and 11+.
- **Also reported:** within ±1 and ±2, top-10 and top-5 overlap (as in plan 2).

### Attribution protocol: every change is tested on its own
A "component" is one switch: one feature group, one likelihood variant, one decoder, one simulator
step, one pace definition. For every component:

1. **Alone.** Reference + this component only, tuned separately with the same budget, against the
   reference on the same dev races with the same seeds. The reference is `latent_111` (and `gbm_100`
   for the tree family) for order components, plan-2 S3 for simulator components, D0 for decoders.
   The reference is re-run in the same harness, not read from an old file.
2. **Leave-one-out.** After the final model is assembled, each included component is removed in turn.
3. **Interactions** only for pairs declared here: pace offset × overtaking threshold (M09 × M18),
   parameter uncertainty × decoder (S6 × D2), finishers-only likelihood × retirement model (M11 × M07).
4. **Record.** One run folder and one leaderboard row per component, one commit per component, and one
   row in `results/plan3/attribution.csv`: component, gain alone (ρ, exact-position rate, RPS) with
   95% intervals, leave-one-out gain, fresh-holdout gain, verdict.

No result is reported for a bundle of components without the per-component rows beside it. A
component that helps alone but not in leave-one-out is reported as redundant, not as a gain.

### Decision rules (fixed before any run)
- A variant "adds value" if its paired per-race gain on dev has a 95% bootstrap CI excluding 0. Because
  the old test blocks are no longer independent, the second condition of plan 2 (same sign on test
  2024–25) is reported but replaced, as confirmation, by the fresh holdout.
- **Primary hypothesis:** M09 against the plan-1 `latent_111` reference. Everything else (M10–M12, M05,
  M14–M16, A4 re-tune) is secondary. After Amendment 1 there are about fifteen secondary comparisons,
  so one or two intervals excluding 0 are expected by chance alone: a secondary result counts as a
  finding only if it also holds on the fresh holdout, and is otherwise reported as "dev only".
- M16 is judged on RPS and log-loss, not ρ. M17 is judged on exact-position rate against grid order.
- A component enters the final model only if its gain *alone* passes the rule; leave-one-out then
  decides whether it stays. After Amendment 2 there are about thirty components, so two or three
  chance passes are expected: the fresh-holdout condition above applies to all of them.
- Same tuning budget per variant (≤ 30 configurations), every variant tuned separately.
- The freeze is recorded with a config hash **and a git tag**, with the date and the last race included.

---

## 6. Work packages

In order. Each ends with a commit, a `LOG.md` entry and its acceptance check.

| WP | Deliverable | Acceptance check |
|---|---|---|
| **R0** Lap and pit-stop ingestion | Cached race laps and pit stops 2014–2026 (Jolpica, OpenF1), resumable, started first because it takes hours | Files per season match the race counts; failures listed |
| **R1** A5 strength audit | All lineages in every plot; intervals; shrinkage cause found; `strengths.csv` | No driver with fewer than 10 starts in the top or bottom 3 without a stated reason; plan-1 numbers reproduced or the difference explained |
| **R2** Attribution harness + M17 | `attribution.csv` writer; decoders D0–D2 on the plan-2 simulator | D0 reproduces the plan-2 exact-position rate (13.6% on dev); grid order 16.3% reproduced |
| **R3** A4 rolling-origin | Season table 2015–2026; rolling re-tune against fixed on dev | Re-tune decision written to `LOG.md` |
| **R4** Race-pace features + A3 | Per-race pace table (median and corrected), offsets, leakage test, the three A3 numbers | Leakage test passes; gate decision written to `LOG.md` |
| **R5** M11 likelihood variants | Three switches with gradient tests; runs on dev | Plan-1 numbers reproduced exactly with all switches off |
| **R6** M09 | Ladder on dev, both families (only if the gate passes) | All rungs in the leaderboard |
| **R7** M18 | Dead-band against linear on dev | Threshold 0 reproduces the linear version exactly |
| **R8** M20 | Persistence check, then each of the four features alone | Persistence numbers logged before any model run |
| **R9** M10 | FIA document download and parser (or hand table), keyword rules; runs at H2 and H0/H1 on dev | Parsed counts and categories spot-checked by hand against three documents |
| **R10** M12 and M05 | Two-stage forecast and ensemble on dev | In the leaderboard |
| **R11** M14 and M15 | Hand tables, supplier pooling, trait × circuit, learned factors, rookie prior | Hand tables have a source per row; leakage test covers the trait features |
| **R12** M16 | Steps S4–S7 on dev, one at a time | Simulator with all new steps off reproduces plan-2 S3 exactly |
| **R13** M19 | In-race forecasts at H4–H6 on dev | Leakage test: changing laps after lap k does not change the lap-k forecast |
| **R14** Final model + freeze | Combine what passed alone, leave-one-out, re-fit the simulator on dev, freeze, tag | `attribution.csv` complete; tag exists; no config edits after it |
| **R15** Score | Old test blocks (third look) and the fresh holdout | Fresh-holdout race count stated |
| **R16** Report | `results/plan3/REPORT.md` and figures answering Q1–Q17, all teams shown, attribution table | Every claim traces to a file in `results/` |
| **R17** M13 | Price-collection script, run before each race | First snapshot stored |

M21 (neural-network control) is not a numbered package; it runs after R6 only if the rule is relaxed.
R1–R3 need no new data and run while R0 downloads. If time runs short, cut in this order:
M21, M15, M14 trait × circuit and learned factors, M16 S7, M12, M20.

New method folders: `methodologies/A3_pace_persistence/`, `A4_rolling_origin/`, `A5_strength_audit/`,
`M09_race_pace/`, `M10_upgrades/`, `M11_likelihood_variants/`, `M12_two_stage/`, `M05_ensemble/`,
`M13_market_benchmark/`, `M14_car_traits/`, `M15_rookie_prior/`, `M16_simulator_extensions/`,
`M17_decoding/`, `M18_overtaking_threshold/`, `M19_in_race/`, `M20_racecraft_pitcrew/`, `M21_nn_control/`.
Plan-3 cross-method outputs go to `results/plan3/`; earlier outputs are not overwritten.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| The race-pace offset is not persistent | A3 checks this before M09 is built; a failed gate is a reported result |
| Lap download is slow or interrupted | Cached and resumable; R1–R3, R5 and R9–R12 do not depend on it |
| About thirty components, each tested alone, is a lot of runs | One at a time with `refit_every` above 1 while tuning; the cut order in §6; chance passes handled by the fresh-holdout rule |
| The corrected-pace fit is unstable in short or interrupted races | Minimum laps per stint; fall back to the median lap for that race and count how often |
| In-race results are mistaken for pre-race accuracy | Separate tables and a horizon label on every number |
| The exact-position target invites tuning to the metric | Decoders have no fitted parameters; the baseline (grid order) and the targets are fixed in §2.2 |
| Median race lap is distorted by strategy, traffic and caution periods | Clean-lap filter; report sensitivity to the 107% threshold; tyre-corrected pace is left to the gated lap-level simulator |
| FIA documents missing for 2022–23 or hard to parse | Hand-made table fallback; M10 is secondary |
| Upgrades are already reflected in qualifying | Evaluated before qualifying as well as after |
| Too few fresh races for a conclusion | State the count and interval; keep scoring through 2027; do not fall back on the used test blocks for claims |
| Many variants, small effects | One primary hypothesis; fixed budget; every attempt logged, failures included |
| Odds source terms of use | Checked before any collection; M13 is dropped if not permitted |
| The plan has grown from 7 to 12 questions | One primary hypothesis; the cut order in §6; secondary results need the fresh holdout to count |
| The audit changes the penalty and so changes earlier numbers | Earlier numbers are re-stated beside the originals; the change is tuned on dev only |
| Rolling re-tune uses 2024–26 as tuning data for the final model | That is legitimate only because those seasons are no longer a test; the fresh holdout stays untouched |
| Hand tables contain errors or hindsight | A source link per row; only values known before the event; a second pass by the tester agent |
| Car traits exist for one dev season | Trait × circuit is labelled exploratory and cannot enter the final model on dev evidence alone |
| Agents report success that did not happen | The skeptic agent re-derives every reported number from files in `results/`; the main session reads diffs before committing |

## 8. Expected outcome

Stated before running. Order accuracy: 0.00–0.02 ρ from M09 if the gate passes, with the largest
effect on finishers; about 0.005 from the ensemble; little from upgrades at H2 and a visible effect
before qualifying. The likelihood variants are expected to change ρ on finishers more than overall ρ.
The most likely single outcome is that the offset is only weakly persistent and qualifying order
remains level with every model on overall ρ.

Amendment 1 items: the audit (A5) is expected to change the charts and the ranking of low-sample
drivers, and ρ by less than 0.003. Rolling re-tune: within ±0.005. Supplier pooling: nothing on dev,
possibly visible in the first races of 2026 (already seen, so not a claim). Rookie prior: not
measurable on overall ρ. M16: a further 1–3% lower RPS, mostly from S4 and S6; no change in ρ.

Amendment 2 items. Exact-position rate before the race: D2 decoding brings the simulator from 13.6% to
about grid level (16–17%) on dev; with pace and the threshold, 17–19% at best; 25% is not expected.
In-race: about 20% after lap 1 and 25% or more at quarter distance. Corrected pace: persistence a
little higher than the median lap. M20: each feature under 0.003 ρ; lap-1 gain is the likeliest to
persist. M21: within noise of `gbm_100`, probably slightly worse. Failure for M17 is D2 not beating
grid order on dev; that would mean the simulator's position probabilities carry less than the grid.

## 9. Sources (seen as search summaries on 2026-11-15; read each before citing)

- Henderson and Kirrane (2018), *A comparison of truncated and time-weighted Plackett–Luce models for probabilistic forecasting of Formula One results*, Bayesian Analysis 13(2).
- *On Bayesian inference for the Extended Plackett–Luce model*, arXiv 2002.05953.
- van Kesteren and Bergkamp (2023), *Bayesian analysis of Formula One race results: disentangling driver skill and constructor advantage*, Journal of Quantitative Analysis in Sports 19(4). (`PLAN.md` §9 names the second author incorrectly; corrected there.)
- Heilmeier, Graf, Betz and Lienkamp (2020), *Application of Monte Carlo methods to consider probabilistic effects in a race simulation for circuit motorsport*, Applied Sciences 10(12); TUMFTM `race-simulation` repository and timing database.
- FIA, "Car Presentation Submissions" documents per event.
- Public repositories noted for ideas only: MonashDeepNeuron/F1-Zoom, 2fst4u/f1predictor (two-stage), MallamTeja/f1_models (monotone constraints).

Added in Amendment 1, **cited from memory and not looked up; the researcher agent confirms each before use**:
- Harville (1973), *Assigning probabilities to the outcomes of multi-entry competitions*, JASA.
- Henery (1981), *Permutation probabilities as models for horse races*, JRSS B.
- Stern (1990), *Models for distributions on permutations*, JASA.
- Lo and Bacon-Shone, work on approximating ordering probabilities with a discount parameter; Benter (1994), *Computer based horse race handicapping and wagering systems*.
- Glickman (1999), *Parameter estimation in large dynamic paired comparison experiments*; Dangauthier et al. (2007), *TrueSkill Through Time*.

Added in Amendment 2, **from memory, not looked up**:
- Grinsztajn, Oyallon and Varoquaux (2022), *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS.
- Fuel- and tyre-corrected lap-time model: as used in Heilmeier et al. (above); the exact form there has not been read.

## 10. Agents

Definitions are in `.claude/agents/` and all four run on Sonnet 5.5 (`model: sonnet`). The main session
plans, decides and commits; it is whatever model the user has selected.

| Agent | Does | Does not |
|---|---|---|
| `pitlane-researcher` | Reads the sources and data endpoints marked "verify" in §3 and §9; reports what each actually says or returns | Write project code |
| `pitlane-coder` | Implements one work package to its acceptance check | Change splits, metrics, decision rules or this plan |
| `pitlane-tester` | Writes and runs tests; leakage and reproduction checks; second pass on hand tables | Edit `src/` to make a test pass |
| `pitlane-skeptic` | Re-derives every number in a log entry or report from files in `results/`; looks for the weakest point of each claim | Edit anything |

Order per work package: researcher (only if something is unverified) → coder → tester → skeptic →
main session reviews and commits.
