# PitLane — Research Plan 2: weekend information and race simulation

Follows [`PLAN.md`](PLAN.md). Everything in plan 1 that is not changed here still applies: CPU only,
no neural networks, explainable predictions, walk-forward evaluation, paired per-race statistics,
no temporal leakage, and the working rules in `CLAUDE.md`.

---

## 1. Why a second plan

Plan 1 result (`results/REPORT.md`, frozen config `5980a1f6ddba`):

| Finding | Evidence |
|---|---|
| All trained models tie with qualifying order | Test 2024–25: qualifying 0.701, proposed `latent_111` 0.703, best model 0.712 (Spearman ρ) |
| Only the dynamic factor helped, slightly | Latent A→B: +0.010 ρ, p = 0.04; separation +0.003; ranking +0.001 |
| Retirements are a large share of the error | `latent_111`: 0.703 on all starters, 0.780 on finishers; 0.757 in races with ≤10% DNFs, 0.549 with >20% |
| The weekend signal dominates each prediction | Weekend term ≈ 0.5–1.1 against 0.0–0.17 for driver and car |

Reading: the models are not short of modelling power. They are short of information that qualifying
does not already contain, and they treat retirements as ordinary slow finishes.

**Research question 2:** which pre-race information, and which race structure, add predictive value
beyond qualifying order?

The three factors of plan 1 (dynamic, driver–car separation, structured ranking) stay in the study:
they are re-tested at each forecast horizon in §4.2, where qualifying no longer masks them.

---

## 2. What plan 2 tests

| # | Question | Tested by |
|---|---|---|
| Q1 | How much accuracy is left to gain at all? | Ceiling analysis (A1) |
| Q2 | Do the three plan-1 factors matter more before qualifying is known? | Horizon analysis (A2) |
| Q3 | Do practice sessions, richer qualifying data and sprint results add information beyond qualifying position? | Information ladder (M06) |
| Q4 | Does modelling retirements separately from pace improve order and probabilities? | DNF hazard model (M07) |
| Q5 | Does a race-level Monte Carlo simulation with circuit-specific grid stickiness beat the plain ranking model? | Race simulator (M08) |
| Q6 | Does weather known before the race change how much the grid should be trusted? | Weather features inside M06 and M08 |

**Qualifying and practice data are mandatory inputs of the main model.** The main plan-2 model
(information set H3 below) uses history, practice, qualifying, grid and sprint results together.
Models without qualifying or practice exist only as diagnostic rungs.

### Decided against

| Idea | Decision | Reason |
|---|---|---|
| Lap-by-lap race simulator | Not built | Needs tyre, pit-stop and overtaking parameters that cannot be calibrated from the data we have |
| Hidden Markov model for car form | Not built | Car form changes continuously; plan 1's dynamic factor gained only +0.010 ρ, and qualifying already measures current form. A two-regime noise mixture in M08 covers the "race state" idea |
| Full F2 / junior-series ingestion | Reduced to an optional table (P2-X1) | It only informs a handful of rookies per season, and only until their first qualifying sessions |

---

## 3. Data additions

| Source | What | Notes |
|---|---|---|
| FastF1, 2018+ | Practice laps (FP1–FP3): lap time, compound, stint, tyre life, track-status flag | Load laps only (`telemetry=False, messages=False`). About 700 sessions; the first download takes hours. Cache on disk, make ingestion resumable, throttle on rate limits |
| FastF1, 2018+ | Session weather for practice, qualifying and sprint: rainfall flag, track and air temperature | Known before the race, so not leakage |
| FastF1 or Jolpica | Sprint qualifying and sprint race results | Deferred in plan 1; all sprint sessions finish before the Grand Prix |
| Jolpica (already cached) | Q1, Q2, Q3 lap times | Plan 1 used only position and the Q1 gap |
| Forecast source (verify first) | Race-day rain forecast as issued before the race | Check whether Open-Meteo's historical forecast archive covers the circuits and years. If it does not, race weather is used only in the oracle analysis of A1 |

`fastf1` is in `requirements.txt` but is not installed yet.

### New features (all per driver per race, all known before the race)

| Group | Features |
|---|---|
| Practice pace | Best-lap gap to session best (per session, in %); long-run pace: median of clean laps in stints of ≥5 laps, minus the session's median for that compound; laps completed |
| Practice availability | Indicator per missing session (pre-2018, sprint weekends, cancelled sessions). Missing values are 0 plus the indicator |
| Qualifying, richer | Best lap gap to pole in % over Q1–Q3; session reached; gap to teammate; position (as in plan 1) |
| Sprint | Sprint finishing position, sprint grid, positions gained in the sprint; indicator for sprint weekend |
| Weather | Wet qualifying flag (rain or intermediate/wet tyres); wet practice flags; race rain forecast if available; mismatch flag: qualifying wet and race expected dry, or the reverse |
| Circuit | Grid stickiness: walk-forward, shrunk correlation between grid and finish at that circuit in earlier years; historical DNF rate at the circuit |

Known limitation: practice fuel loads and engine modes are unknown, so practice pace is noisy.
Tyre-degradation slopes are left out of the first version for the same reason.

**Leakage test:** extend `tests/test_leakage.py` to the new features. Add one test per horizon
that an information set contains no column from a later session.

---

## 4. Analyses and methodologies

### A1 — Ceiling analysis (run first, existing data only)
Three numbers on the dev block, for `latent_111` and qualifying order:

1. **Self-consistency ceiling:** sample finishing orders from the calibrated model and score the
   model against its own samples. This is the ρ a correct model would get given its own noise level.
2. **Retirement oracle:** tell the model who retires (placed last, in retirement order). The gain is
   the most a perfect DNF model could add.
3. **Race-pace oracle:** rank finishers by their median race lap time. The gap to the model is what
   better pace information could add; the gap to 1.0 is strategy, incidents and luck.

Oracles use race data on purpose. They are labelled as oracles and never enter a leaderboard.

### A2 — Horizon analysis (plan 1 extension X1)
Four information sets, each a strict superset of the one before:

| Set | Contains | Role |
|---|---|---|
| H0 | History only | Pre-weekend forecast |
| H1 | H0 + practice | After practice |
| H2 | H0 + qualifying position, grid, Q1 gap | Plan 1's information; the reference |
| H3 | H2 + practice, richer qualifying, sprint, weather | **Main plan-2 model** |

Run the plan-1 2×2×2 factorial of Family L at H0 (existing code, weekend inputs switched off) and
report the three main effects next to the H2 effects from plan 1. H1 needs the FastF1 data and runs
after P2.

### M06 — Weekend-information models
Family L and Family G from plan 1, at cell `111` and cell `000`, with the weekend term extended.

- Family L: new signals enter the additive weekend term, so the breakdown stays exact. The
  qualifying coefficient is allowed to shrink when the mismatch flag is on:
  `β_quali · (1 − δ · mismatch)`, δ fitted.
- Family G: the same columns as features.

**Information ladder**, starting from H2 and adding one group at a time: richer qualifying →
practice pace → sprint → weather. Also report leave-one-group-out from H3, because the order of a
ladder can hide redundancy.

### M07 — DNF hazard model
Logistic regression for "this entry is not classified", fitted walk-forward.
Inputs: driver's decayed incident rate, car lineage's decayed mechanical rate, circuit DNF rate,
grid position band, wet flag, first season of new regulations, laps completed in practice.
Coefficients are the explanation. Scored by Brier and log-loss against a constant-rate baseline.

### M08 — Race-level Monte Carlo simulator
One draw of a race (5 000 draws per race, fixed seeds):

1. **Retirements:** each entry retires with its M07 probability. Retired entries go behind the
   finishers in random order.
2. **Pace order of the survivors:** M06 score plus Gumbel noise, which is Plackett–Luce sampling.
3. **Grid stickiness:** the grid term's weight is `β_grid · (1 + κ · stick_c)` for circuit *c*,
   κ fitted on dev.
4. **Race regime:** the noise scale is a two-component mixture (normal race, disrupted race); the
   disruption probability depends on the circuit's safety-car history and the wet flags.

Outputs: the full driver × position probability matrix; point prediction = expected finishing
position. Each step is a switch, so the structure ladder is: M06 alone → + retirements → + grid
stickiness → + regime mixture.

Explanation per driver: the M06 additive breakdown, plus retirement probability, plus the shift in
expected position caused by each simulator step.

### Optional (only after P8)

| ID | Item |
|---|---|
| P2-X1 | Rookie prior from junior-series results: a small hand-made table in `data/external/`, scored on rookies' first five races only |
| P2-X2 | Race-by-race random-walk car strength (state-space) in place of per-season car effects |
| P2-X3 | Wet-weather driver effect (plan 1 X7), if at least 15 wet races exist in burn-in + dev |

---

## 5. Evaluation

### Splits
| Block | Seasons | Use in plan 2 |
|---|---|---|
| Burn-in | 2014–2020 | Fit only. Practice features exist from 2018, so their coefficients are learned on 2018–2020 |
| Development | 2021–2023 | All tuning and all model selection |
| Test — stable | 2024–2025 | **Already scored once in plan 1.** Scored once more, after the plan-2 freeze |
| Test — regime shift | 2026, rounds 1–20 | Same: second look |
| Fresh holdout | Races after 2026 round 20 | The only data nobody has seen. Reported separately, however few races it is |

The second look is stated in the report next to every test number. Nothing may be tuned on a test
block; the plan-1 test results may not be used to choose between plan-2 variants.

### Metrics
- **Primary for order:** mean per-race Spearman ρ and position MAE (unchanged).
- **Primary for probabilities** (M07, M08): log-loss of P(points finish) and the ranked probability
  score over the position distribution.
- **Added to the harness:** top-10 overlap, top-5 overlap, exact-position rate, within ±1 and ±2.
- **Subsets**, each reported with its race or entry count: first five rounds of a season, rookies,
  drivers in their first season with a team, wet-affected weekends, sprint weekends.

### Decision rules (fixed before any run)
- An information group or simulator step "adds value" if its paired per-race gain on dev has a 95%
  bootstrap CI excluding 0 **and** the gain has the same sign on test 2024–25.
- **Gate after A1:** if the race-pace oracle beats `latent_111` by less than 0.03 ρ on finishers,
  order accuracy is treated as saturated: probabilities become the headline metric, and M06 is
  limited to one tuning round.
- Same tuning budget per variant (≤ 30 configurations), as in plan 1.

---

## 6. Work packages

In order. Each ends with a commit, a `LOG.md` entry and its acceptance check.

| WP | Deliverable | Acceptance check |
|---|---|---|
| **P0** Harness | New metrics and subset reports; an `info_set` switch (H0–H3) in the feature builder | Unit tests on toy races; plan-1 numbers reproduced exactly at H2 |
| **P1** A1 + A2 at H0 | Ceiling numbers; factorial at H0; short section in the report | Gate decision written to `LOG.md` |
| **P2** Ingestion | FastF1 practice laps, session weather, sprint results; Q1–Q3 times; forecast source checked | Session counts per season match the calendar; missing sessions listed, not silently dropped |
| **P3** Features | Feature groups of §3 | Extended leakage test and horizon tests pass |
| **P4** M06 | Information ladder and leave-one-out on dev, both families | All rungs in the leaderboard |
| **P5** M07 | DNF hazard model on dev | Beats or ties the constant-rate baseline; calibration plot |
| **P6** M08 | Simulator with its four switches on dev | Probability matrix rows and columns sum to 1; structure ladder in the leaderboard |
| **P7** Freeze + test | Freeze configs, tag the commit, run test blocks and fresh holdout once | Tag exists; no config edits after it |
| **P8** Report | `results/plan2/REPORT.md` and figures answering Q1–Q6 | Every claim traces to a file in `results/` |

New method folders: `methodologies/M06_weekend_info/`, `M07_dnf_hazard/`, `M08_race_monte_carlo/`.
Plan-2 cross-method outputs go to `results/plan2/`; plan-1 outputs are not overwritten.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Practice pace is too noisy to add anything | Report it as a null result; leave-one-out shows it directly |
| FastF1 download is slow or rate-limited | Resumable cached ingestion; P0 and P1 do not depend on it |
| No usable forecast archive | Use only wet flags from sessions before the race; race weather stays in the oracle analysis |
| Test blocks were already seen | Second look stated everywhere; fresh holdout reported separately; decision rules fixed in §5 |
| Few wet races and few rookies | Subset results reported with counts and CIs; no claims from subsets under 10 races |
| Simulator adds parameters without adding accuracy | Each step is a switch with its own paired test; steps that fail the rule are dropped from the final model |
| Overfitting dev through many variants | Fixed budget; every attempt logged, failures included |

## 8. Expected outcome

Stated before running, so the result can be judged against it: gains in order accuracy are expected
to be small (0.00–0.02 ρ at H3), larger gains are expected in probability scores from M07 and M08,
and the three plan-1 factors are expected to show larger effects at H0 than at H2.

---

## 9. Amendment 1 (2026-11-15, written before the plan-2 run, after probing the data sources)

These are changes forced by data access. Research questions, metrics and decision rules are unchanged.

| Planned in sections 3–5 | Amendment | Consequence |
|---|---|---|
| FastF1 practice laps and session weather from 2018 | FastF1's timing API answers HTTP 401 without an F1 account login. Practice laps and session weather come from **OpenF1** (no login), which starts in **2023** | Practice features exist for 2023–2026 only (earlier races carry `fp_missing = 1`). The development block has practice data for one season (2023, 22 races), so practice groups are judged on dev-2023 plus the test blocks, and their dev interval is wide by construction. The decision rule is applied as written to those races |
| Race-day rain forecast, "verify first" | Verified: Open-Meteo's historical-forecast archive returns hourly precipitation from 2020 (probabilities only from 2024). Used: forecast precipitation in the 2 h race window, from 2020; ERA5 observed precipitation on the qualifying day, all seasons. A window of ≥ 0.5 mm counts as wet | Forecast flags are missing before 2020 (`wx_missing = 1`) |
| Race-pace oracle on the whole development block | Jolpica allows about 500 requests per hour and a race is about 12 pages, so race laps were fetched for 2023 only | The pace-oracle gate is evaluated on 22 races |
| Weather and sprint flags as features | In the additive latent model a race-wide flag is constant within a race and cannot change a race order. In family L these enter only as interactions with qualifying position (`x_q_x_*`); family G (trees) gets the raw flags | None for the decision rule |
| Fresh holdout | The cached 2026 data ends at round 20, so the holdout contains no races yet. The runner scores any races after round 20 automatically when re-run after re-ingesting | Reported as "no races yet" until then |
| Model-run labels | Plan-2 runs are written to `results/plan2/leaderboard.csv` (plan-1 files untouched); how runs are recorded is documented in `.claude/skills/pitlane-run/SKILL.md` | — |
