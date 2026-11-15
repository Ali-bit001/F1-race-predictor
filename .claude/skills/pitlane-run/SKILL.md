---
name: pitlane-run
description: How to implement, run and record an experiment in the PitLane F1 repo — where results go, how runs are named and logged, how a full study is launched detached, and how results are committed and pushed. Use whenever you add a methodology, run a plan (PLAN.md / PLAN2.md), or are asked to "run the train and test commands and exit".
---

# PitLane: run and record an experiment

The study is a sequence of plans (`PLAN.md`, `PLAN2.md`, ...). The plan is the specification: implement it as written,
and if the data or the world forces a change, **write the change into the plan first** (an "amendment"), then code it.
Rules that never bend are in `CLAUDE.md` (CPU only, no leakage, test blocks locked, honest reporting).

## 1. Before running anything
1. `python -m pytest -q` must pass. It includes the leakage tests (`tests/test_leakage*.py`): features for race R must
   not change when R's own outcome is removed. A new feature family gets its own leakage test.
2. New model options must be covered by a gradient or recovery test when they touch a likelihood.
3. Changed shared code (metrics, harness, latent model)? Re-run a frozen plan-1 config and confirm the numbers did not move
   (plan 2 checks this inside the report: "Reproduction check").
4. Smoke test the whole script with `--quick` (2-config grids, one test split, no commit) before the real launch.

## 2. How results are recorded (every run, no exceptions)
`pitlane.evaluation.harness.save_run` writes, for each model x split:

```
methodologies/<ID>_<name>/results/<YYYYMMDD-HHMM>_<label>_<split>/
    config.yaml            full config + git commit hash + split
    metrics.json           mean of the per-race metrics
    per_race_metrics.csv   one row per race (the unit of analysis; all paired tests use this)
    predictions.parquet    per-entry score, p_win, p_podium, p_points (plan 2 also p_dnf)
```
and appends one row to a leaderboard CSV: `results/leaderboard.csv` (plan 1), `results/plan2/leaderboard.csv` (plan 2),
`results/plan2/leaderboard_dnf.csv` (the retirement model, whose metrics are different).
Tuning grids are saved next to the runs as `methodologies/<ID>/results/tuning_<name>.csv`.
Every method has a card `methodologies/<ID>_<name>/README.md` (copy `_TEMPLATE`) and a row in `methodologies/README.md`.

Cross-method outputs: `results/REPORT.md` + `results/figures/` (plan 1), `results/plan2/REPORT.md` + `results/plan2/figures/` (plan 2).
Reports are generated only from saved runs, so each number traces back to a file; a failing report section prints its error
inside the document instead of aborting the run.

`experiments/LOG.md` is append-only: add an entry for every run worth remembering, **including failures and dead ends**
(`finalize*.py` appends an automatic entry with the headline numbers and the deviations from the plan).

## 3. Protocol that must hold
- Walk-forward: to predict race r use only races before r; final runs refit before every race (`refit_every=1`);
  tuning may use `refit_every=4`.
- Tune on the development block (2021-2023) only, <= 30 configurations per variant; then freeze the configs to
  `configs/frozen*.yaml` with a hash, and only then score the test blocks. If the test blocks were already scored in an earlier
  plan, say "second look" next to every test number and keep untouched races (the fresh holdout) separate.
- Paired per-race bootstrap CIs and Wilcoxon tests; report CIs, not only means; subsets under 10 races are not interpreted.
- A null or negative result is reported as such. Never describe a variant as better without the paired interval.

## 4. Launching a full run and exiting (what the user means by "run it and exit")
The runner scripts are resumable (tuning lives in the frozen file, finished runs in the leaderboard) and finish by committing
and pushing the results themselves, so launch detached and do not wait:

```powershell
cd <repo>
New-Item -ItemType Directory -Force logs | Out-Null
Start-Process -FilePath python -ArgumentList "-u","scripts\run_plan2.py" -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput logs\run_plan2.out -RedirectStandardError logs\run_plan2.err -WindowStyle Hidden
```
(`scripts/run_all.py` is the plan-1 equivalent.) Logs are in `logs/` (git-ignored). Commit and push the **code** first, so that the
results commit that the run makes at the end sits on top of the code that produced them.

The final stage (`scripts/finalize*.py`) does: append the LOG entry -> `git add results methodologies configs experiments` ->
commit -> `git push -u origin HEAD`. Commit messages end with the co-author trailer given in the session's attribution reminder.

Tell the user, when launching: what runs, where the results will appear (`results/plan2/REPORT.md`, figures, leaderboards),
that the run pushes by itself, and which data sources had to be substituted.

## 5. Data and environment notes (learned the hard way)
- Jolpica (Ergast successor): ~500 requests/hour, 100 rows/page; cache everything under `data/raw/` (git-ignored).
- FastF1 timing now needs an F1 login (HTTP 401) -> practice laps and session weather come from OpenF1 (2023+ only).
  Open-Meteo gives weather (ERA5 for observed days, historical-forecast archive from 2020).
- Windows: PowerShell `Set-Content -Encoding utf8` writes a BOM (breaks TOML); use the Write tool for files. PowerShell blocks
  `Remove-Item` with wildcards on data directories; use Git Bash `rm`. Bash heredocs containing quotes can fail; write scripts to
  the scratchpad directory with the Write tool instead.
- Row order of `entries*.parquet` is part of the contract (the harness needs a RangeIndex sorted by race_id, pos; cached latent
  strengths are aligned by row). Merges must preserve it; `build_entries2` asserts this.
