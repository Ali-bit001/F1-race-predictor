"""Plan-2 orchestration helpers (PLAN2.md section 6). Used by scripts/run_plan2.py; every stage is resumable.

Runs are written exactly like plan-1 runs (see .claude/skills/pitlane-run/SKILL.md): a folder per run under
methodologies/<ID>/results/<run_id>/ plus one row in results/plan2/leaderboard.csv.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pitlane import pipeline as P
from pitlane.config import METHODS, PLAN2, ROOT, load_yaml
from pitlane.evaluation.harness import save_run, split_races, walk_forward
from pitlane.features.weekend import GROUP_ORDER, g_cols, l_cols, load_entries2
from pitlane.models.registry import build_model

LB2 = PLAN2 / "leaderboard.csv"
METHOD_DIR2 = {"A2": "A2_horizon", "M06": "M06_weekend_info", "M07": "M07_dnf_hazard", "M08": "M08_race_monte_carlo",
               **P.METHOD_DIR}
ALL_SPLITS = ["dev", "test_stable", "test_shift", "fresh"]
CELLS = P.CELLS
LAMX_GRID = [0.3, 3.0]
GATE_LAMX = [1.0]       # one tuning round if A1 says order accuracy is saturated


def load_all(rebuild: bool = False) -> pd.DataFrame:
    """Plan-2 entries (plan-1 features + weekend + history + scaled signals) with plan-1 latent strengths."""
    frozen1 = load_yaml(ROOT / "configs" / "frozen.yaml")
    return P.add_latent_strengths(load_entries2(rebuild), frozen1)


# ------------------------------------------------------------------------------------ grids and tuning
def lat_grid(cell: str, extra: dict, lamx: list[float] | None = None) -> list[dict]:
    out = []
    for c in P.grid_for("latent", cell):
        for lx in (lamx or [None]):
            out.append({**c, **extra, **({"lam_x": lx} if lx is not None else {})})
    return out


def gbm_grid(cell: str, extra: dict) -> list[dict]:
    return [{**c, **extra} for c in P.grid_for("gbm", cell)]


def tune_pick(grid: list[dict], entries: pd.DataFrame, n_jobs: int, quick: bool) -> tuple[dict, pd.DataFrame]:
    grid = grid[:2] if quick else grid
    res = Parallel(n_jobs=n_jobs)(delayed(P._eval)(c, entries, "dev", 4) for c in grid)
    flat = lambda c: {k: (str(v) if isinstance(v, list) else v) for k, v in c.items()}
    tab = pd.DataFrame([{**flat(r["cfg"]), **{k: v for k, v in r.items() if k != "cfg"}} for r in res])
    return res[int(tab["spearman"].idxmax())]["cfg"], tab


def save_tuning(method: str, name: str, tab: pd.DataFrame) -> None:
    d = METHODS / METHOD_DIR2[method] / "results"
    d.mkdir(parents=True, exist_ok=True)
    tab.to_csv(d / f"tuning_{name}.csv", index=False)


# ------------------------------------------------------------------------------------ variants
RUNGS = [[], GROUP_ORDER[:1], GROUP_ORDER[:2], GROUP_ORDER[:3], GROUP_ORDER]   # H2, +qrich, +practice, +sprint, +weather = H3


def m06_variants() -> list[tuple[str, str, str, list[str]]]:
    """(frozen key, name, cell, groups) for every M06 latent variant that is tuned separately.
    r0 (H2) is the plan-1 frozen model and r4 (H3) is the M06L model, so neither is listed here."""
    out = [("M06R", f"r{r}_{cell}", cell, RUNGS[r]) for cell in ("000", "111") for r in (1, 2, 3)]
    out += [("M06N", f"no_{g}_111", "111", [x for x in GROUP_ORDER if x != g]) for g in GROUP_ORDER]
    return out


def m06_items(frozen1: dict, frozen2: dict) -> list[dict]:
    """Every variant is tuned on dev with the same grid (PLAN2.md section 2.6: same budget per variant)."""
    items = []
    for cell in ("000", "111"):
        items.append(dict(label=f"m06_L_r0_{cell}", method="M06", cfg=dict(frozen1["M03"][cell])))
        for r in (1, 2, 3):
            items.append(dict(label=f"m06_L_r{r}_{cell}", method="M06", cfg=frozen2["M06R"][f"r{r}_{cell}"]))
        items.append(dict(label=f"m06_L_r4_{cell}", method="M06", cfg=frozen2["M06L"][cell]))
    for g in GROUP_ORDER:
        items.append(dict(label=f"m06_L_no_{g}_111", method="M06", cfg=frozen2["M06N"][f"no_{g}_111"]))
    for cell in ("000", "111"):
        items.append(dict(label=f"m06_G_h3_{cell}", method="M06", cfg=frozen2["M06G"][cell]))
    return items


def h_items(frozen2: dict) -> list[dict]:
    items = [dict(label=f"h0_latent_{c}", method="A2", cfg=frozen2["H0"][c]) for c in CELLS]
    items += [dict(label=f"h1_latent_{c}", method="A2", cfg=frozen2["H1"][c]) for c in ("000", "111")]
    return items


def ref_items(frozen1: dict) -> list[dict]:
    """Plan-1 frozen models, re-run only on the fresh holdout (they have not seen it)."""
    from pitlane.models.baselines import HEURISTICS
    items = [dict(label=w, method="M00", cfg=dict(kind="heuristic", cell="000", which=w), splits=["fresh"])
             for w in ("grid", "quali", "standings")]
    cfgs = [frozen1["M01rf"], frozen1["M03"]["000"], frozen1["M03"]["111"], frozen1["M04"]["000"],
            frozen1["M04"]["001"], frozen1["M04"]["111"]]
    for c in cfgs:
        label = "rf_000" if c["kind"] == "rf" else f"{c['kind']}_{c['cell']}"
        items.append(dict(label=label, method="M01" if c["kind"] == "rf" else P.method_of(c["kind"], c["cell"]),
                          cfg=c, splits=["fresh"]))
    return items


# ------------------------------------------------------------------------------------ running
def _run(cfg: dict, entries: pd.DataFrame, split: str):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    model = build_model(cfg, entries)
    return walk_forward(entries, model, split, refit_every=1, posdist=True)


def done_pairs() -> set[tuple[str, str]]:
    if not LB2.exists():
        return set()
    d = pd.read_csv(LB2)
    return set(zip(d["label"], d["split"]))


def run_items(items: list[dict], entries: pd.DataFrame, splits: list[str], n_jobs: int, log=print) -> None:
    have = done_pairs()
    avail = {s: len(split_races(entries, s)[1]) > 0 for s in splits}
    jobs = []
    for it in items:
        for s in it.get("splits", splits):
            if s in splits and avail.get(s, False) and (it["label"], s) not in have:
                jobs.append((it, s))
    if not jobs:
        log(f"  nothing to run for {splits}")
        return
    log(f"  running {len(jobs)} model/split jobs on {splits}")
    outs = Parallel(n_jobs=n_jobs)(delayed(_run)(it["cfg"], entries, s) for it, s in jobs)
    for (it, s), (preds, per) in zip(jobs, outs):
        save_run(METHODS / METHOD_DIR2[it["method"]], it["label"], it["cfg"], preds, per, s, it["method"],
                 it["cfg"].get("cell", ""), leaderboard=LB2)
        log(f"  saved {it['method']} {it['label']} {s}: rho={per['spearman'].mean():.3f} mae={per['mae'].mean():.2f}")


def load_run_preds(label: str, split: str) -> pd.DataFrame | None:
    if not LB2.exists():
        return None
    d = pd.read_csv(LB2)
    m = d[(d["label"] == label) & (d["split"] == split)]
    return None if m.empty else pd.read_parquet(ROOT / m.iloc[-1]["run_dir"] / "predictions.parquet")
