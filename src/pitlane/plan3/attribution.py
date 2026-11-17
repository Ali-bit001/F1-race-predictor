"""Attribution harness (PLAN3.md section 5, "Attribution protocol").

Every component (feature group, likelihood variant, decoder, simulator step, pace definition) is compared with
its reference on the same races: the paired per-race gain and a 95% bootstrap CI that resamples races
(fixed seed). Results are upserted into results/plan3/attribution.csv, one row per (component, reference).

Conventions
- `pos` is the derived finishing order of ALL entries (1..n): classified finishers at their official place,
  retirees behind them in the order they retired (ingest.py). Non-classified drivers therefore have an actual
  position and are scored like everyone else in the "overall" rows; "finishers" rows use classified entries only.
- Predicted order is a permutation 1..n of all entries. Exact = predicted place == actual place.
- Gains are signed so that POSITIVE = candidate better: candidate - reference for higher-is-better metrics,
  reference - candidate for MAE, RPS and log-loss.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from pitlane.config import RESULTS
from pitlane.evaluation.metrics import pred_rank, race_metrics

ATTRIBUTION_CSV = RESULTS / "plan3" / "attribution.csv"
HIGHER_BETTER = {"spearman": True, "spearman_fin": True, "mae": False, "mae_fin": False,
                 "exact": True, "exact_fin": True, "exact_1_3": True, "exact_4_10": True, "exact_11p": True,
                 "within1": True, "within2": True, "rps": False, "logloss_points": False}
METRICS = list(HIGHER_BETTER)  # fixed: these are the columns of the append-only attribution.csv
HIGHER_BETTER["within3"] = True  # reported only (PLAN3.md section 5); pass metrics=[..., "within3"] to paired_gain
BANDS = {"exact_1_3": (1, 3), "exact_4_10": (4, 10), "exact_11p": (11, 10_000)}
N_BOOT = 10_000
SEED = 0


def exact_by_band(rk: np.ndarray, pos: np.ndarray, classified: np.ndarray) -> dict[str, float]:
    """Exact-position rates: on classified entries, and by ACTUAL place band (share of entries whose actual
    place lies in the band that were predicted at exactly that place)."""
    hit = rk == pos
    out = {"exact_fin": float(hit[classified].mean()) if classified.any() else np.nan}
    for name, (lo, hi) in BANDS.items():
        m = (pos >= lo) & (pos <= hi)
        out[name] = float(hit[m].mean()) if m.any() else np.nan
    return out


def race_table(preds: pd.DataFrame, score_col: str = "score", rps: dict | None = None) -> pd.DataFrame:
    """Per-race metric table from per-entry predictions (columns race_id, pos, classified, score [, p_points]).
    Higher score = predicted to finish better. `rps` optionally maps race_id -> RPS (needs the position matrix)."""
    rows = []
    for rid, g in preds.groupby("race_id", sort=True):
        pos, cls = g["pos"].to_numpy(float), g["classified"].to_numpy(bool)
        u = g[score_col].to_numpy(float)
        m = race_metrics(pos, u, cls)
        m.update(exact_by_band(pred_rank(u), pos, cls))
        if "p_points" in g:
            q = np.clip(g["p_points"].to_numpy(float), 1e-4, 1 - 1e-4)
            y = (pos <= 10).astype(float)
            m["logloss_points"] = float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
        if rps is not None and rid in rps:
            m["rps"] = float(rps[rid])
        m.update(race_id=rid, season=int(g["season"].iloc[0]) if "season" in g else -1, n=len(g))
        rows.append(m)
    return pd.DataFrame(rows)


def paired_gain(ref: pd.DataFrame, cand: pd.DataFrame, metrics: list[str] | None = None,
                n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Paired per-race gain (positive = candidate better) with a 95% percentile bootstrap CI over races.
    Both tables need `race_id`; only races present in both are used. A metric missing in either table, or
    NaN for a race, is skipped for that race (n reports how many races were used)."""
    m = ref.merge(cand, on="race_id", suffixes=("_r", "_c"))
    out = []
    for k in (metrics or METRICS):
        if f"{k}_r" not in m or f"{k}_c" not in m:
            continue
        rng = np.random.default_rng([seed, zlib.crc32(k.encode())])   # one stream per metric: CI independent of order
        sgn = 1.0 if HIGHER_BETTER[k] else -1.0
        d = sgn * (m[f"{k}_c"].to_numpy(float) - m[f"{k}_r"].to_numpy(float))
        d = d[~np.isnan(d)]
        if len(d) == 0:
            continue
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        bm = d[idx].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        out.append(dict(metric=k, ref_mean=float(np.nanmean(m[f"{k}_r"])), cand_mean=float(np.nanmean(m[f"{k}_c"])),
                        gain=float(d.mean()), lo=float(lo), hi=float(hi), n=len(d)))
    return pd.DataFrame(out)


def verdict(gain: pd.Series | dict) -> str:
    """Section 5 rule: CI excluding 0 on dev. Positive = candidate better."""
    if gain["lo"] > 0:
        return "passes alone"
    if gain["hi"] < 0:
        return "worse"
    return "no measurable difference"


def attribution_row(component: str, reference: str, gains: pd.DataFrame, verdict_metric: str, run_id: str,
                    n_races: int, declared: str = "yes") -> dict:
    """`declared` = is the component declared in PLAN3 (yes/no). Off-plan diagnostics are marked "no": no verdict
    from them is used."""
    row: dict = {"component": component, "reference": reference, "declared": declared}
    for _, g in gains.iterrows():
        row[f"gain_{g['metric']}"] = g["gain"]
        row[f"lo_{g['metric']}"] = g["lo"]
        row[f"hi_{g['metric']}"] = g["hi"]
    row.update(loo_gain="", fresh_holdout_gain="", verdict_metric=verdict_metric,
               verdict=verdict(gains[gains["metric"] == verdict_metric].iloc[0]), n_races=n_races, run_id=run_id)
    return row


def upsert_attribution(rows: list[dict], path: Path = ATTRIBUTION_CSV) -> pd.DataFrame:
    """Read-modify-write right before finishing. Key = (component, reference). A row that already exists keeps
    its leave-one-out and fresh-holdout cells unless the new row supplies them."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, dtype={"loo_gain": object, "fresh_holdout_gain": object})
        for i, r in new.iterrows():
            m = (old["component"] == r["component"]) & (old["reference"] == r["reference"])
            if m.any():
                for c in ("loo_gain", "fresh_holdout_gain"):
                    if r[c] in ("", None) or (isinstance(r[c], float) and np.isnan(r[c])):
                        new.loc[i, c] = old.loc[m, c].iloc[0]
        keep = old.merge(new[["component", "reference"]], on=["component", "reference"], how="left", indicator=True)
        old = old[(keep["_merge"] == "left_only").to_numpy()]
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new
