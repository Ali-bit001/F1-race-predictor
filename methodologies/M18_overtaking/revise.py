"""M18 revision after referee review (R7). Dev only. Writes nothing to leaderboard.csv.

    python methodologies/M18_overtaking/revise.py

1. Scores d = 0 vs d = 0.03 (and the descriptive d = 0.5, 1.5) on the STORED matrices by grid-slot binary log-loss and
   by exact-slot log-loss (paired, 95% CI over races, 10,000 resamples, seed 0). Positive gain = lower log-loss.
2. Writes run folders for the two descriptive large-d runs (traceability only; no leaderboard or attribution row).
"""
from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pitlane.config import METHODS, ROOT, load_yaml
from pitlane.plan2 import pipeline2 as Q
from pitlane.plan2 import simulate as S
from pitlane.plan3 import calibration as CAL
from pitlane.plan3 import decode as DEC
from pitlane.plan3 import overtake as OT

OUT = METHODS / "M18_overtaking"
EPS = (1e-3, 1e-4)


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, METHODS / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def per_race_logloss(mats, races, keys, kind: str, eps: float) -> np.ndarray:
    out = []
    for P, r, k in zip(mats, races, keys):
        pos = np.asarray(r["pos"], float)
        if kind == "gridslot":      # binary: finishes in its grid slot or not
            slot = CAL.grid_slots(r["g"]["grid"].to_numpy(float), k)
            p = np.clip(P[np.arange(len(slot)), slot - 1], eps, 1 - eps)
            y = (pos == slot).astype(float)
            out.append(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()))
        else:                       # exact slot: -log P(realised place)
            idx = pos.astype(int) - 1
            p = np.clip(P[np.arange(len(idx)), idx], eps, 1.0)
            out.append(float(-np.log(p).mean()))
    return np.array(out)


def boot(diff: np.ndarray, n_boot=10000, seed=0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    m = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    run = _load("m18_run", "M18_overtaking/run.py")
    m17 = run._m17()
    cfg = load_yaml(OUT / "config.yaml")
    frozen = load_yaml(ROOT / cfg["frozen_path"])
    theta = {k: frozen["M08"][cfg["simulator"]][k] for k in S.DEFAULT}
    consts = {k: frozen["M08"][cfg["simulator"]][k] for k in ("stick_ref", "vol_mu", "vol_sd")}
    races = S.prepare(m17.sim_inputs(Q.load_all(), "dev", cfg["m06_label"]), consts)
    keys = m17.driver_keys(races)
    ds = [0.0, 0.03] + list(cfg["show_d"])
    mats = {d: OT.simulate_matrices_db(races, theta, d, cfg["simulator"], cfg["draws"], cfg["seed"]) for d in ds}

    rows = []
    for kind in ("gridslot", "exact_slot"):
        for eps in EPS:
            base = per_race_logloss(mats[0.0], races, keys, kind, eps)
            for d in ds[1:]:
                g, lo, hi = boot(base - per_race_logloss(mats[d], races, keys, kind, eps))
                rows.append(dict(objective=f"{kind}_logloss", eps=eps, d=d, mean_ll_d0=base.mean(), gain_vs_d0=g, lo=lo, hi=hi))
    res = pd.DataFrame(rows)
    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    res.to_csv(OUT / "results" / f"{stamp}_logloss_objectives_dev.csv", index=False)
    pd.set_option("display.width", 200)
    print("LOG-LOSS ON STORED MATRICES (gain = d0 minus d, positive = d better; 95% CI over 66 races)")
    print(res.round(4).to_string(index=False))

    gid = run.git_id()
    for d in cfg["show_d"]:           # run folders only; no leaderboard / attribution rows
        for kind in ("D0", "D2"):
            _, preds, per = run.table_for(m17, races, mats[d], kind)
            rid, _ = run.save(f"descriptive_d{d}_{kind}", cfg, theta, d, preds, per, stamp, gid, mats[d], races)
            print("wrote", rid)


if __name__ == "__main__":
    main()
