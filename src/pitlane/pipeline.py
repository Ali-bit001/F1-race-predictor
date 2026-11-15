"""Orchestration: tune on dev -> freeze -> run once on the locked test blocks (PLAN.md WP4-WP7).

Used by scripts/run_all.py. Every stage is resumable: finished stages leave artifacts in
results/_stage/ and are skipped on re-run.
"""
from __future__ import annotations

import itertools
import json
import os
import hashlib
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from pitlane.config import CONFIGS, METHODS, PROCESSED, RESULTS, load_yaml, save_yaml
from pitlane.evaluation.harness import save_run, summarize, walk_forward
from pitlane.features.build import add_features
from pitlane.models.baselines import HEURISTICS
from pitlane.models.latent import walk_strengths
from pitlane.models.registry import build_model

STAGE = RESULTS / "_stage"
CELLS = [f"{a}{b}{c}" for a, b, c in itertools.product("01", repeat=3)]
LAM_GRID = [0.5, 2.0, 8.0, 32.0]
HL_GRID = [8, 20, 50]
GBM_GRID = [dict(num_leaves=7, n_estimators=150, learning_rate=0.05, min_child_samples=40),
            dict(num_leaves=15, n_estimators=150, learning_rate=0.05, min_child_samples=40),
            dict(num_leaves=7, n_estimators=300, learning_rate=0.03, min_child_samples=80),
            dict(num_leaves=4, n_estimators=200, learning_rate=0.05, min_child_samples=40)]
GBM_HL_GRID = [8, 16, 32]


def method_of(kind: str, cell: str) -> str:
    if kind == "latent":
        return "M03"
    return {"000": "M01", "001": "M02"}.get(cell, "M04")


METHOD_DIR = {"M00": "M00_heuristics", "M01": "M01_direct_gbm", "M02": "M02_gbm_ranking",
              "M03": "M03_latent_strength", "M04": "M04_gbm_factorial", "M01rf": "M01_direct_gbm"}


def load_entries(quick: bool = False) -> pd.DataFrame:
    cache = PROCESSED / "entries_feat.parquet"
    if cache.exists() and not quick:
        return pd.read_parquet(cache)
    raw = pd.read_parquet(PROCESSED / "entries.parquet")
    df = add_features(raw)
    df.to_parquet(cache, index=False)
    return df


def _eval(cfg: dict, entries: pd.DataFrame, split: str, refit_every: int) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    model = build_model(cfg, entries)
    _, per_race = walk_forward(entries, model, split, refit_every=refit_every, n_samples=200)
    return {"cfg": cfg, **summarize(per_race)}


def grid_for(kind: str, cell: str) -> list[dict]:
    dyn = cell[0] == "1"
    out = []
    if kind == "latent":
        for lam in LAM_GRID:
            for hl in (HL_GRID if dyn else [None]):
                c = {"kind": kind, "cell": cell, "lam": lam}
                if hl:
                    c["halflife"] = hl
                out.append(c)
    else:
        for g in GBM_GRID:
            for hl in (GBM_HL_GRID if dyn else [None]):
                c = {"kind": kind, "cell": cell, **g}
                if hl:
                    c["halflife"] = hl
                out.append(c)
    return out


def tune(kind: str, cell: str, entries: pd.DataFrame, split: str, refit_every: int, n_jobs: int,
         quick: bool) -> tuple[dict, pd.DataFrame]:
    grid = grid_for(kind, cell)
    if quick:
        grid = grid[:2]
    res = Parallel(n_jobs=n_jobs)(delayed(_eval)(c, entries, split, refit_every) for c in grid)
    tab = pd.DataFrame([{**r["cfg"], **{k: v for k, v in r.items() if k != "cfg"}} for r in res])
    return res[int(tab["spearman"].idxmax())]["cfg"], tab


def run_and_save(cfg: dict, entries: pd.DataFrame, split: str, method: str, label: str, refit_every: int = 1):
    model = build_model(cfg, entries)
    preds, per_race = walk_forward(entries, model, split, refit_every=refit_every)
    save_run(METHODS / METHOD_DIR[method], label, cfg, preds, per_race, split, method, cfg.get("cell", ""))
    return per_race


def add_latent_strengths(entries: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Walk-forward latent strengths for the F2-on cells of Family G. Static strengths use the tuned
    '010' config, dynamic ones the tuned '110' config (Gaussian likelihood: keeps F3 out of F2)."""
    cache = PROCESSED / "latent_strengths.parquet"
    if cache.exists():
        s = pd.read_parquet(cache)
    else:
        first = int(entries.loc[entries["season"] == 2015, "race_id"].min())
        parts = {}
        for tag, cell in (("lat0", "010"), ("lat1", "110")):
            cfg = {k: v for k, v in frozen["M03"][cell].items() if k not in ("kind", "cell")}
            dyn, dec, rank = cell[0] == "1", True, False
            parts[tag] = walk_strengths(entries, dict(dynamic=dyn, decomp=dec, ranking=rank, **cfg), first)
        s = pd.concat({k: v.add_prefix(k + "_") for k, v in parts.items()}, axis=1)
        s.columns = [c[1] for c in s.columns]
        s.to_parquet(cache, index=False)
    return pd.concat([entries.drop(columns=[c for c in s.columns if c in entries.columns]), s], axis=1)


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]
