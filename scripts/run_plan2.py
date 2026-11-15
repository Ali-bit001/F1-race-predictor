"""Run plan 2 end to end (PLAN2.md section 6): ingest -> features -> A1 ceiling -> tune on dev -> dev runs ->
M07 -> M08 fit -> freeze -> test runs -> report -> commit and push.

    python scripts/run_plan2.py            # full run
    python scripts/run_plan2.py --quick    # 2-config grids, dev + test_shift only (smoke test, no commit)

Resumable: tuning results live in configs/frozen_plan2.yaml, finished runs in results/plan2/leaderboard*.csv.
"""
from __future__ import annotations

import argparse
import os
import runpy
import time
from datetime import datetime

import numpy as np
import pandas as pd

from pitlane import pipeline as P
from pitlane.config import CONFIGS, METHODS, PLAN2, RESULTS, load_yaml, save_yaml
from pitlane.evaluation.harness import save_run, split_races
from pitlane.features.weekend import GROUP_ORDER, g_cols, l_cols
from pitlane.plan2 import dnf as D
from pitlane.plan2 import pipeline2 as Q
from pitlane.plan2 import simulate as S
from pitlane.plan2.ceiling import run_ceiling

os.environ.setdefault("OMP_NUM_THREADS", "1")
FROZEN2 = CONFIGS / "frozen_plan2.yaml"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def sim_inputs(entries: pd.DataFrame, split: str) -> pd.DataFrame | None:
    """M06 score + M07 p_dnf + circuit/weather columns for the counted races of a split."""
    m06 = Q.load_run_preds("m06_L_r4_111", split)
    if m06 is None or not len(m06):
        return None
    lb = pd.read_csv(D.LB_DNF)
    row = lb[(lb["split"] == split)]
    if row.empty:
        return None
    dn = pd.read_parquet(D.ROOT / row.iloc[-1]["run_dir"] / "predictions.parquet")[["race_id", "driverId", "p_dnf"]]
    cols = ["race_id", "driverId", "grid_norm", "stick", "circ_vol", "rain_fc"]
    return m06.merge(dn, on=["race_id", "driverId"]).merge(entries[cols], on=["race_id", "driverId"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--skip-ingest", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    n_jobs = max(1, min(6, (os.cpu_count() or 2) - 1))
    PLAN2.mkdir(parents=True, exist_ok=True)
    tests = ["test_shift"] if args.quick else ["test_stable", "test_shift", "fresh"]

    # ---- P2/P3: data and features -------------------------------------------------------------
    if not args.skip_ingest and not args.quick:
        log("ingesting plan-2 data (cached; resumable)")
        from pitlane.data import ingest2
        ingest2.main()
    entries = Q.load_all(rebuild=True)       # features are cheap to rebuild and must see the freshest ingested data
    log(f"entries {entries.shape}")
    frozen1 = load_yaml(CONFIGS / "frozen.yaml")
    lb1 = pd.read_csv(RESULTS / "leaderboard.csv")

    # ---- P1: A1 ceiling analysis (existing plan-1 dev predictions) ------------------------------
    ceil = run_ceiling(lb1)
    log(f"A1 ceiling: {ceil}")
    lamx = Q.GATE_LAMX if ceil["saturated"] else Q.LAMX_GRID

    # ---- tuning on dev (<= 30 configurations per variant) ---------------------------------------
    frozen2 = load_yaml(FROZEN2) if FROZEN2.exists() else {}
    for k in ("H0", "H1", "M06L", "M06R", "M06N", "M06G", "M07", "M08"):
        frozen2.setdefault(k, {})
    prac = {"base_x": False, "x_cols": l_cols(["practice"]), "lam_x": 1.0}
    h3l = {"base_x": True, "x_cols": l_cols(GROUP_ORDER)}
    h3g = {"extra_cols": g_cols(GROUP_ORDER + ["circuit"])}
    stages = ([("H0", "A2", c, lambda c=c: Q.lat_grid(c, {"base_x": False})) for c in Q.CELLS] +
              [("H1", "A2", c, lambda c=c: Q.lat_grid(c, prac)) for c in ("000", "111")] +
              [("M06L", "M06", c, lambda c=c: Q.lat_grid(c, h3l, lamx)) for c in ("000", "111")] +
              [(key, "M06", name, lambda c=c, g=g: Q.lat_grid(c, {"base_x": True, "x_cols": l_cols(g)}, lamx))
               for key, name, c, g in Q.m06_variants()] +
              [("M06G", "M06", c, lambda c=c: Q.gbm_grid(c, h3g)) for c in ("000", "111")])
    for key, method, cell, grid in stages:
        if cell in frozen2[key]:
            continue
        cfg, tab = Q.tune_pick(grid(), entries, n_jobs, args.quick)
        frozen2[key][cell] = cfg
        Q.save_tuning(method, f"{key}_{cell}", tab)
        save_yaml(frozen2, FROZEN2)
        log(f"tuned {key} {cell}: {cfg}")
    if "C" not in frozen2["M07"]:
        C, tab = D.tune_C(entries)
        frozen2["M07"] = {"C": C}
        Q.save_tuning("M07", "C", tab)
        save_yaml(frozen2, FROZEN2)
        log(f"M07 tuned: C={C}")

    # ---- dev runs, M07 dev, M08 fit (all development block only) --------------------------------
    items = Q.h_items(frozen2) + Q.m06_items(frozen1, frozen2)
    log("dev runs")
    Q.run_items(items, entries, ["dev"], n_jobs, log)

    def m07(split: str) -> None:
        have = pd.read_csv(D.LB_DNF) if D.LB_DNF.exists() else pd.DataFrame(columns=["split"])
        if (have["split"] == split).any():
            return
        preds, per, coef = D.walk_dnf(entries, split, frozen2["M07"]["C"])
        if len(preds):
            D.save_dnf_run(preds, per, coef, split, frozen2["M07"]["C"])
            log(f"M07 {split}: logloss {per['logloss'].mean():.4f} vs const {per['logloss_const'].mean():.4f}, "
                f"auc {per['auc'].mean():.3f}")
    m07("dev")

    dev = entries[entries["season"].between(2021, 2023)]
    consts = dict(stick_ref=float(dev["stick"].mean()), vol_mu=float(dev["circ_vol"].mean()),
                  vol_sd=float(dev["circ_vol"].std()))
    inp = sim_inputs(entries, "dev")
    for v in S.SWITCHES:
        if v not in frozen2["M08"]:
            theta, rps = S.fit_theta(S.prepare(inp, consts), v)
            frozen2["M08"][v] = {**{k: float(x) for k, x in theta.items()}, **consts, "fit_rps": float(rps)}
            save_yaml(frozen2, FROZEN2)
            log(f"M08 {v} fitted on dev: {frozen2['M08'][v]}")
    frozen2["frozen_at"] = datetime.now().isoformat(timespec="seconds")
    frozen2["hash"] = P.sha({k: frozen2[k] for k in ("H0", "H1", "M06L", "M06R", "M06N", "M06G", "M07", "M08")})
    save_yaml(frozen2, FROZEN2)
    log(f"plan-2 configs frozen, hash {frozen2['hash']}")

    # ---- test runs (one shot) -------------------------------------------------------------------
    log(f"test runs: {tests}")
    Q.run_items(items + Q.ref_items(frozen1), entries, tests, n_jobs, log)
    for s in tests:
        m07(s)

    def m08(split: str) -> None:
        have = Q.done_pairs()
        inp_s = sim_inputs(entries, split)
        if inp_s is None or inp_s.empty:
            return
        races = S.prepare(inp_s, consts)
        for v in S.SWITCHES:
            if (f"m08_{v}", split) in have:
                continue
            th = {k: frozen2["M08"][v][k] for k in S.DEFAULT}
            preds, per = S.evaluate(races, th, v)
            save_run(METHODS / Q.METHOD_DIR2["M08"], f"m08_{v}", {"kind": "m08", "cell": v, "theta": th, "consts": consts,
                                                                  "switches": S.SWITCHES[v]},
                     preds, per, split, "M08", v, leaderboard=Q.LB2)
            log(f"  M08 {v} {split}: rho={per['spearman'].mean():.3f} rps={per['rps'].mean():.4f} "
                f"logloss_points={per['logloss_points'].mean():.4f}")
    for s in ["dev"] + tests:
        m08(s)
    log(f"experiments finished in {(time.time() - t0) / 60:.1f} min")

    if not args.no_report:
        from pitlane.plan2.report2 import make_report2
        try:
            make_report2(ceil)
        except Exception as e:          # never lose the results because the report failed
            import traceback
            log("report failed: " + repr(e) + " | " + traceback.format_exc().replace("\n", " | "))
        if not args.quick:
            runpy.run_path(os.path.join(os.path.dirname(__file__), "finalize2.py"), run_name="__main__")
    log("done")


if __name__ == "__main__":
    main()
