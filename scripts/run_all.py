"""Run the whole study end to end: ingest -> features -> tune on dev -> freeze -> test once -> report.

    python scripts/run_all.py            # full run
    python scripts/run_all.py --quick    # 2-config grids, test_shift only (smoke test)

Resumable: tuning results live in configs/frozen.yaml and finished runs in results/leaderboard.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd
from joblib import Parallel, delayed

from pitlane import pipeline as P
from pitlane.config import CONFIGS, METHODS, PROCESSED, RESULTS, load_yaml, save_yaml
from pitlane.evaluation.harness import save_run, walk_forward
from pitlane.models.baselines import HEURISTICS
from pitlane.models.registry import build_model

os.environ.setdefault("OMP_NUM_THREADS", "1")
FROZEN = CONFIGS / "frozen.yaml"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _run(cfg, entries, split):
    model = build_model(cfg, entries)
    return cfg, *walk_forward(entries, model, split, refit_every=1)


def run_many(cfgs: list[dict], entries: pd.DataFrame, split: str, label_of, method_of, n_jobs: int) -> None:
    outs = Parallel(n_jobs=n_jobs)(delayed(_run)(c, entries, split) for c in cfgs)
    for cfg, preds, per_race in outs:
        m = method_of(cfg)
        save_run(METHODS / P.METHOD_DIR[m], label_of(cfg), cfg, preds, per_race, split, m, cfg.get("cell", ""))
        log(f"  saved {m} {label_of(cfg)} {split}: rho={per_race['spearman'].mean():.3f} mae={per_race['mae'].mean():.2f}")


def method_of(cfg: dict) -> str:
    if cfg["kind"] == "heuristic":
        return "M00"
    if cfg["kind"] == "rf":
        return "M01"
    return P.method_of(cfg["kind"], cfg["cell"])


def label_of(cfg: dict) -> str:
    if cfg["kind"] == "heuristic":
        return cfg["which"]
    return f"{cfg['kind']}_{cfg['cell']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    n_jobs = max(1, min(6, (os.cpu_count() or 2) - 1))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(exist_ok=True)
    (RESULTS / "_stage").mkdir(parents=True, exist_ok=True)
    test_splits = ["test_shift"] if args.quick else ["test_stable", "test_shift"]

    if not (PROCESSED / "entries.parquet").exists():
        log("ingesting data")
        from pitlane.data.ingest import main as ingest
        ingest()
    entries = P.load_entries()
    log(f"entries {entries.shape}, seasons {entries.season.min()}-{entries.season.max()}")

    frozen = load_yaml(FROZEN) if FROZEN.exists() else {"M03": {}, "M04": {}, "M01rf": None}
    def commit_frozen():
        save_yaml(frozen, FROZEN)

    # ---- M03: tune on dev ------------------------------------------------------------------
    for cell in P.CELLS:
        if cell in frozen["M03"]:
            continue
        cfg, tab = P.tune("latent", cell, entries, "dev", refit_every=4, n_jobs=n_jobs, quick=args.quick)
        frozen["M03"][cell] = cfg
        d = METHODS / P.METHOD_DIR["M03"] / "results"; d.mkdir(parents=True, exist_ok=True)
        tab.to_csv(d / f"tuning_{cell}.csv", index=False)
        commit_frozen()
        log(f"M03 tuned {cell}: {cfg}")

    # ---- Walk-forward latent strengths for the F2-on cells of Family G -----------------------
    entries = P.add_latent_strengths(entries, frozen)
    log("latent strengths ready")

    # ---- M01 (RF + GBM regression), M02, M04: tune on dev ------------------------------------
    if frozen["M01rf"] is None:
        grid = [dict(kind="rf", cell="000", min_samples_leaf=m) for m in (10, 20, 40)][:2 if args.quick else 3]
        res = Parallel(n_jobs=n_jobs)(delayed(P._eval)(c, entries, "dev", 4) for c in grid)
        frozen["M01rf"] = max(res, key=lambda r: r["spearman"])["cfg"]
        commit_frozen()
        log(f"RF tuned: {frozen['M01rf']}")
    for cell in P.CELLS:
        if cell in frozen["M04"]:
            continue
        cfg, tab = P.tune("gbm", cell, entries, "dev", refit_every=4, n_jobs=n_jobs, quick=args.quick)
        frozen["M04"][cell] = cfg
        d = METHODS / P.METHOD_DIR[P.method_of("gbm", cell)] / "results"; d.mkdir(parents=True, exist_ok=True)
        tab.to_csv(d / f"tuning_{cell}.csv", index=False)
        commit_frozen()
        log(f"G tuned {cell}: {cfg}")
    frozen["frozen_at"] = datetime.now().isoformat(timespec="seconds")
    frozen["hash"] = P.sha({k: frozen[k] for k in ("M03", "M04", "M01rf")})
    commit_frozen()
    log(f"configs frozen, hash {frozen['hash']}")

    # ---- Final dev runs (refit before every race) + the one-shot test runs ------------------
    heur = [dict(kind="heuristic", cell="000", which=w) for w in HEURISTICS]
    cfgs = heur + [frozen["M01rf"]] + list(frozen["M03"].values()) + list(frozen["M04"].values())
    lb = RESULTS / "leaderboard.csv"
    for split in ["dev"] + test_splits:
        if lb.exists() and ((pd.read_csv(lb)["split"] == split).sum() >= len(cfgs)):
            log(f"{split}: already complete, skipping")
            continue
        log(f"running {len(cfgs)} models on {split}")
        run_many(cfgs, entries, split, label_of, method_of, n_jobs)

    log(f"experiments finished in {(time.time() - t0) / 60:.1f} min")
    if not args.no_report:
        from pitlane.report import make_report
        make_report()
        if not args.quick:
            import runpy
            runpy.run_path(os.path.join(os.path.dirname(__file__), "finalize.py"), run_name="__main__")
    log("done")


if __name__ == "__main__":
    main()
