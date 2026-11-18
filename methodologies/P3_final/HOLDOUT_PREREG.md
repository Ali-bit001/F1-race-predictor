# Plan-3 fresh-holdout pre-registration (R14, written 2026-11-18, before 2026 R21)

This file fixes, before any race after 2026 round 20 (race_id 271) is run, **what the holdout scores, against what, and what counts
as confirmation**. It is part of the freeze tag; its SHA-256 is in `FREEZE.md`. Nothing here may change after the tag. A later
change is a new pre-registration, and races run before that change cannot count for it.

## 1. Holdout
- Every race with race_id > 271, in date order, as it is run (2026 R21 onward, into 2027 and later).
- A race is scored at H2 if it has a classified result. A race that enters the data with an incomplete classification is scored on
  the classified rows, as in plans 1–2. No race is removed after its outcome is seen.

## 2. What is scored (decides the skeptic's B1)
- **Point order (rho, MAE, exact rate, within ±k, the place bands): the walk-forward *score order*.** It comes from the harness path in
  `src/pitlane/plan3/final.py` as run by `methodologies/P3_final/run.py`:
  - every race r is predicted from a model fitted on all rows before r, with `frozen_config.yaml` hyper-parameters;
  - the universe is the full one and the warm start is the harness one;
  - ties are broken without using outcomes.

  This is the path that produced every dev headline number (rho 0.6632, exact 0.1512).
- **Probabilities (RPS, points log-loss): the S3 matrix** with the frozen re-fitted theta (tau 1.7, dnf_scale 1.25, kappa 0, p0 0.05,
  s_dis 1.5, a_vol 1.0, a_rain 2). It is drawn on the harness scores with the harness seeds (`seed + 10000 + i`, seed 0, i = the race index in date order within the holdout, 0 for the first holdout race) and the M07
  retirement model with C = 0.02.
- **Descriptive only:**
  - the simulator D0 order (sort by -E[position]);
  - the output of `predict.py`, which is the live pre-race tool and differs from the harness by up to 6.9e-4 in score. A
    `predict.py` file may be saved before a race as a public record; it is not what the holdout scores.

## 3. References (all frozen, all run through the same harness on the same races)
| Name | Definition |
|---|---|
| `latent_111` | plan-1 frozen M03 cell 111 (`configs/frozen.yaml`), lam 32, half-life 50 |
| `final` | the frozen plan-3 model: `latent_111` + M11 reverse + M09 car pace history (lam_x 3.0) |
| `final_minus_reverse` | `final` with reverse off (= the stored M09 history configuration) |
| `reverse_only` | `final` with the pace column off (= `latent_111` + reverse, the stored M11 reverse configuration). Named second model |
| `plan2_H2` | plan-2 frozen H2 model (`configs/frozen_plan2.yaml`) |
| `grid` | grid order |

H1: plan 3 has **no H1 hypothesis**. H1 is the plan-2 frozen H1 model unchanged, scored descriptively against `plan2_H1`. The
2023 H1 diagnostic with the components switched on carries no weight and must not be used to switch H1 later. The M10 H1 test
is pre-registered separately in `methodologies/M10_upgrades/README.md`.

## 4. Primary tests (three, at H2)
| # | Claim | Contrast (candidate minus reference) | Metric |
|---|---|---|---|
| T1 | the plan-3 model beats plan 1 | `final` − `latent_111` | Spearman rho, all classified |
| T2 | reverse helps | `final` − `final_minus_reverse` | MAE (lower is better; gain = reference − candidate) |
| T3 | the M09 car pace column helps | `final` − `reverse_only` | Spearman rho, all classified |

- **Test:** paired per-race bootstrap, 10,000 resamples, seed 0, 95% interval. This is the same code as `attribution.csv` on dev.
- **Confirmed:** the interval excludes 0 in the stated direction after Holm correction over T1–T3 (familywise 0.05). An
  uncorrected pass, or the same sign without the interval, is reported as **not confirmed**. Nothing is "promising".
- **Minimum race count: 20 holdout races** before any of T1–T3 gets a verdict. Before that, every holdout number is printed as
  "no verdict (n < 20)", with its interval.
- **Read-outs:** the verdict is read once, at the first point where n ≥ 20, and again at n ≥ 40 if the data exist. The later read
  supersedes the earlier one; there is no stopping when a result looks good.
- If T3 is not confirmed, M09 is reported as "not confirmed on holdout". The pre-stated reading is that its dev gain is car form
  (the R6 control), not race pace. `reverse_only` is then the better-supported model, but the frozen model is not swapped.

## 5. Secondary, reported with intervals, no verdict weight
- **Exact rate against grid order** (Amendment 2 target) for `final`, `reverse_only` and `latent_111`. The target counts as met
  only if the interval excludes 0 at n ≥ 20.
- The other metrics of the dev tables: finishers' rho and MAE, exact by place band, within ±1/±2/±3, and RPS and points log-loss
  of `final` against `latent_111` through its own S3 re-fit.
- `final` against `plan2_H2`.
- Per season.

## 6. Data and 2027 inputs (decided now, applied mechanically)
- Holdout rows are appended by the existing fetch and build code. **Rows for race_id ≤ 271 must not change**; the R15 script
  checks this against the hashes in `FREEZE.md` (restricted to those rows) and stops if they differ.
- **2027 entry list:** taken from the first classified 2027 race's Jolpica entry list. Before that race, it is taken from the FIA
  entry list as published.
- **Lineage:** a 2027 team keeps the lineage of its constructor id. A new constructor id that is a rebrand of an existing entry
  (same entry and factory, new name) joins that entry's lineage, as for the rebrand chains in `data/external/lineage.csv`. Any
  other new entry gets a new lineage with no history (pace column 0, as for Cadillac 2026).
- **Circuit types:** `data/external/circuits.csv` types are analyst judgement, not a measured rule. A new circuit gets a type
  (street / downforce / power, as in that file), with the source "analyst judgement", written down before its first session.
- Each addition is committed before the first session of the race it is needed for. Existing rows are never edited.
- If a mechanical step cannot be done without a judgement (a case the rules above do not cover), the race is scored with the
  judgement written in LOG.md before the outcome is fetched.
