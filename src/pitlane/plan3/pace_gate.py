"""Statistics for the A3 gate (PLAN3.md section 4, A3): pace oracle, persistence correlation, split of the gap.

All functions are pure. Oracles use race data on purpose and are never features.
Bootstrap: resample RACES with replacement (10,000 resamples, seed 0 unless stated).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

N_BOOT = 10_000
SEED = 0
MIN_ORACLE_DRIVERS = 8      # as plan 2: a race enters the oracle with at least 8 classified drivers having a pace


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = stats.rankdata(a), stats.rankdata(b)
    if len(ra) < 3 or np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_mean(d: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Mean and 95% percentile bootstrap CI of per-race values (NaN dropped)."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bm = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


# ------------------------------------------------------------------------------------ plan-2 oracle (repeat)
def plan2_pace(laps_race: pd.DataFrame, drivers: list[str]) -> dict[str, float]:
    """Exactly `pitlane.plan2.ceiling.pace_oracle_rank` on a dataframe: median lap per driver over laps >= 2 and
    <= 107% of the race median (pit laps kept), at least 5 laps."""
    d = laps_race[laps_race["lap"] >= 2]
    d = d[d["lap_time_s"] <= 1.07 * d["lap_time_s"].median()]
    med = d.groupby("driverId")["lap_time_s"].agg(["median", "size"])
    return {x: float(med.loc[x, "median"]) for x in drivers if x in med.index and med.loc[x, "size"] >= 5}


def plan2_rho(pace: dict[str, float], fin: pd.DataFrame) -> float:
    """Plan-2 convention: rank by pace, Pearson against the raw classified positions of the drivers with a pace."""
    sub = fin[fin["driverId"].isin(pace)]
    if len(sub) < MIN_ORACLE_DRIVERS:
        return np.nan
    t = sub["driverId"].map(pace).to_numpy()
    pr = stats.rankdata(t)                       # pred_rank(-t) in plan 2 = rank by time ascending
    pos = sub["pos"].to_numpy(float)
    return float(np.corrcoef(pr, pos)[0, 1]) if np.std(pr) and np.std(pos) else np.nan


# ------------------------------------------------------------------------------------ matched oracle
def matched_rhos(g: pd.DataFrame, value: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """On the classified drivers with a valid pace value (one race): Spearman of the finishing order with the
    order by `value` (ascending = faster) and with the model's order (descending score)."""
    m = g["classified"].to_numpy(bool) & valid
    if m.sum() < MIN_ORACLE_DRIVERS:
        return {"n": int(m.sum()), "oracle": np.nan, "model": np.nan}
    pos = g["pos"].to_numpy(float)[m]
    return {"n": int(m.sum()), "oracle": spearman(-value[m], -pos),
            "model": spearman(g["score"].to_numpy(float)[m], -pos)}


# ------------------------------------------------------------------------------------ persistence
def pooled_corr(df: pd.DataFrame, x: str, y: str, race: str = "race_id", n_boot: int = N_BOOT,
                seed: int = SEED) -> dict[str, float]:
    """Pooled Pearson correlation of x (predicted by earlier races) with y (this race) over all rows, with a 95%
    CI from resampling races. Uses per-race sufficient statistics, so the bootstrap is exact and fast."""
    d = df[[race, x, y]].dropna()
    if len(d) < 5:
        return dict(r=np.nan, lo=np.nan, hi=np.nan, n_obs=len(d), n_races=d[race].nunique())
    g = d.assign(xx=d[x] ** 2, yy=d[y] ** 2, xy=d[x] * d[y], one=1.0).groupby(race)[[x, y, "xx", "yy", "xy", "one"]].sum()
    s = g.to_numpy()

    def corr(t: np.ndarray) -> np.ndarray:
        sx, sy, sxx, syy, sxy, n = (t[..., i] for i in range(6))
        with np.errstate(invalid="ignore", divide="ignore"):
            return (n * sxy - sx * sy) / np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))

    r = float(corr(s.sum(0)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    bm = corr(s[idx].sum(1))
    lo, hi = np.nanpercentile(bm, [2.5, 97.5])
    return dict(r=r, lo=float(lo), hi=float(hi), n_obs=int(len(d)), n_races=int(len(g)))
