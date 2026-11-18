# best_models — the picked model per stage

Evidence and caveats: `results/MODEL_COMPARISON.md`. **Nothing here is confirmed on the fresh holdout (n = 0).**

The YAML files here are byte-identical copies, for reference only. The originals are authoritative and the code reads
the originals; `P3_final/predict.py` also checks its own `frozen_config.yaml` against the hash in `FREEZE.md`.
Editing a copy here changes nothing.

| Stage | Model | Copy here | Original (authoritative) | SHA-256 |
|---|---|---|---|---|
| Before qualifying (H1) | `h1_latent_111`, plan-2 frozen | `h1_prequali__frozen_plan2.yaml` (section `H1`) | `configs/frozen_plan2.yaml` | `4965a18d…7057` |
| After qualifying (H2) | `final` = `latent_111` + M11 reverse + M09 pace history (lam_x 3.0), with the S3 simulator for probabilities | `h2_postquali__p3_frozen_config.yaml` | `methodologies/P3_final/frozen_config.yaml` | `24ef86b6…be00` |
| In-race H4 / H5 | M19 two-stage `gbm_s2` (`latent_s2` is close) | `inrace__m19_config.yaml` | `methodologies/M19_inrace/config.yaml` | `3bbe3717…728e` |
| In-race H6 | running order (no model needed; M19 gain is marginal) | — | — | — |
| Exact-position hits, any stage after qualifying | grid order / running order | — | — | — |

Alternative at H2: `reverse_only` (`final` with the pace column off) is level with `final` and slightly ahead on 2024–26
(−0.0036 [−0.0108, +0.0037], not significant). It is the pre-stated fallback if holdout test T3 fails.

## Commands (run from the repo root)
```bash
# Before and after qualifying, one race (H1, H2 or both). Without a live entries builder, pass --entries for a future race.
PYTHONPATH=src python methodologies/P3_final/predict.py --season 2026 --round 21 --horizon both [--entries PATH] [--out PATH] [--no-sim]

# In-race (H4/H5/H6) study on the development block (no live mode exists yet)
PYTHONPATH=src python methodologies/M19_inrace/run.py [--config methodologies/M19_inrace/config.yaml] [--quick]

# Holdout scoring (prints "no verdict (n < 20)" until 20 holdout races exist)
PYTHONPATH=src python methodologies/P3_final/score.py
```

Limits: no live entries builder for H1/H2; M19 has no live in-race predictor (study code only) and has never been
checked outside 2021–23.
