"""Plan-3 race-pace features (PLAN3.md section 3, R4).

Two layers, both pure functions of their inputs so that they can be leakage-tested:

1. `build_pace_table(laps, pits)` — one row per (season, round, driver) from that race's own laps and pit stops
   only. Nothing here looks at another race. It is an OUTCOME of the race (used as a target of the history
   features below and as an oracle in A3), never a feature of the same race.
   * median clean lap: lap 1, in-laps, out-laps and laps above 107% of the race median dropped; at least
     `min_clean` clean laps; reported as a % gap to the best median in that race (`med_gap_pct`).
   * corrected pace: per race, least squares on the clean laps of eligible drivers
         lap time = driver pace + f * laps remaining + t * stint age
     (f, t common to the race; stint age = laps since the driver's last stop, from pit-stop laps, so it is
     defined the same way in every season). The driver pace is evaluated at the race-average fuel load and
     tyre age, then turned into a % gap (`corr_gap_pct`). If the fit is not identified (no pit data, fuel
     and age almost collinear, too few drivers) the race falls back to the median lap and `corr_fallback` is set.
2. `rq_table` / `rq_history_features` — race-minus-qualifying offset `rq = race gap % - qualifying gap %` and its
   shrunk, time-decayed means over EARLIER races only (per car lineage `con_rq`, per driver `drv_rq`), with the
   same machinery (`_ewm_prior`, race ordinal as time, one pseudo-observation of the prior) as the plan-1
   history features. The prior is 0 because the offset is centred on the field median of each race.

Laps of 2024-2026 come from OpenF1 only, where lap-1 times are not interchangeable with Jolpica's; lap 1 is
dropped here, and `build_pace_table` takes one source at a time (`pitlane.data.ingest3.dedupe_laps`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitlane.features.build import _ewm_prior

THRESH = 1.07           # clean-lap threshold, share of the race median lap
MIN_CLEAN = 10          # clean laps needed for a driver's pace in a race
MAX_ABS_CORR = 0.98     # corrected fit is skipped if demeaned fuel and tyre-age columns are this collinear
MIN_FIT_DRIVERS = 8
HALFLIVES = (8, 16, 32, 50)     # races; 32 = frozen gbm_100, 50 = frozen latent_111
KEY = ["season", "round", "driverId"]
RACE = ["season", "round"]


# ------------------------------------------------------------------------------------ laps -> clean laps
def mark_laps(laps: pd.DataFrame, pits: pd.DataFrame, thresh: float = THRESH) -> pd.DataFrame:
    """Add race_med, in_lap, out_lap, stint_age, rem, clean (section 3 definition) and clean_p2 (plan-2 filter:
    lap >= 2 and <= 107% of the race median, pit laps kept). `laps` needs season, round, driverId, lap, lap_time_s;
    `pits` needs season, round, driverId, lap (the lap of the stop = the in-lap). One source per race."""
    d = laps[KEY + ["lap", "lap_time_s"]].copy()
    d = d[np.isfinite(d["lap_time_s"]) & (d["lap_time_s"] > 0)].reset_index(drop=True)
    med = d[d["lap"] >= 2].groupby(RACE)["lap_time_s"].median().rename("race_med")
    d = d.join(med, on=RACE)
    d["race_laps"] = d.groupby(RACE)["lap"].transform("max")
    d["rem"] = d["race_laps"] - d["lap"]
    pit_idx = pd.MultiIndex.from_frame(pits[KEY + ["lap"]].drop_duplicates())
    d["in_lap"] = pd.MultiIndex.from_frame(d[KEY + ["lap"]]).isin(pit_idx)
    prev = d[KEY + ["lap"]].assign(lap=d["lap"] - 1)
    d["out_lap"] = pd.MultiIndex.from_frame(prev).isin(pit_idx)
    d["stint_age"] = _stint_age(d, pits)
    under = d["lap_time_s"] <= thresh * d["race_med"]
    d["clean_p2"] = (d["lap"] >= 2) & under
    d["clean"] = d["clean_p2"] & ~d["in_lap"] & ~d["out_lap"]
    return d


def _stint_age(d: pd.DataFrame, pits: pd.DataFrame) -> np.ndarray:
    """Laps since the stint started: lap - (last stop lap before this lap) - 1; first stint starts on lap 1."""
    pit_laps = {k: np.sort(g["lap"].unique()) for k, g in pits.groupby(KEY)}
    age = np.empty(len(d))
    for k, ix in d.groupby(KEY).indices.items():
        lap = d["lap"].to_numpy()[ix]
        pl = pit_laps.get(k)
        if pl is None:
            age[ix] = lap - 1
        else:
            n = np.searchsorted(pl, lap, side="left")          # stops strictly before this lap
            last = np.where(n > 0, pl[np.maximum(n - 1, 0)], 0)
            age[ix] = lap - last - 1
    return age


# ------------------------------------------------------------------------------------ one race
def _gap_pct(x: pd.Series) -> pd.Series:
    return (x / x.min() - 1.0) * 100.0


def corrected_pace(c: pd.DataFrame) -> tuple[pd.Series | None, float, float, str]:
    """Fuel- and tyre-corrected driver pace from the clean laps `c` of eligible drivers (one race).
    Returns (pace in seconds at race-average conditions or None, fuel coef, tyre coef, reason if not fitted)."""
    drv = c["driverId"].nunique()
    if drv < MIN_FIT_DRIVERS:
        return None, np.nan, np.nan, "too few drivers"
    y, rem, age = c["lap_time_s"].to_numpy(float), c["rem"].to_numpy(float), c["stint_age"].to_numpy(float)
    g = c["driverId"]
    ym, rm, am = (pd.Series(v, index=c.index).groupby(g).transform("mean").to_numpy() for v in (y, rem, age))
    r_d, a_d, y_d = rem - rm, age - am, y - ym
    if np.std(r_d) < 1e-9 or np.std(a_d) < 1e-9:
        return None, np.nan, np.nan, "no variation"
    if abs(np.corrcoef(r_d, a_d)[0, 1]) > MAX_ABS_CORR:
        return None, np.nan, np.nan, "collinear"
    (f, t), *_ = np.linalg.lstsq(np.column_stack([r_d, a_d]), y_d, rcond=None)
    adj = pd.Series(y - f * rem - t * age, index=c.index).groupby(g).mean()      # driver pace at zero fuel, age 0
    return adj + f * rem.mean() + t * age.mean(), float(f), float(t), ""


def _race_rows(g: pd.DataFrame, has_pit: bool, min_clean: int) -> pd.DataFrame:
    c = g[g["clean"]]
    n_clean = c.groupby("driverId").size()
    elig = n_clean[n_clean >= min_clean].index
    out = pd.DataFrame({"n_laps": g.groupby("driverId").size()})
    out["n_clean"] = n_clean.reindex(out.index).fillna(0).astype(int)
    ce = c[c["driverId"].isin(elig)]
    med = ce.groupby("driverId")["lap_time_s"].median()
    out["med_lap_s"] = med.reindex(out.index)
    out["med_gap_pct"] = _gap_pct(med).reindex(out.index) if len(med) else np.nan
    pace, f, t, why = (None, np.nan, np.nan, "no pit data") if not has_pit else corrected_pace(ce)
    out["corr_fallback"] = pace is None
    out["corr_reason"] = why
    out["fuel_coef"], out["tyre_coef"] = f, t
    cl = med if pace is None else pace
    out["corr_lap_s"] = cl.reindex(out.index) if len(cl) else np.nan
    out["corr_gap_pct"] = _gap_pct(cl).reindex(out.index) if len(cl) else np.nan
    return out


def build_pace_table(laps: pd.DataFrame, pits: pd.DataFrame, thresh: float = THRESH,
                     min_clean: int = MIN_CLEAN) -> pd.DataFrame:
    """Per (season, round, driverId): n_laps, n_clean, med_lap_s, med_gap_pct, corr_lap_s, corr_gap_pct and the
    race-level corr_fallback / corr_reason / fuel_coef / tyre_coef. Each race uses only its own laps and stops."""
    m = mark_laps(laps, pits, thresh)
    have_pit = set(map(tuple, pits[RACE].drop_duplicates().to_numpy()))
    parts = []
    for (s, r), g in m.groupby(RACE):
        p = _race_rows(g, (s, r) in have_pit, min_clean)
        p.insert(0, "round", r)
        p.insert(0, "season", s)
        parts.append(p.reset_index())
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------------------------ rq and its history
def rq_table(pace: pd.DataFrame, entries: pd.DataFrame, gap_col: str = "med_gap_pct") -> pd.DataFrame:
    """Race-minus-qualifying offset per entry. `entries` needs race_id, season, round, driverId, lineage and
    q_best_gap (a share; qualifying gap to pole). Columns added: q_gap_pct, race_gap_pct, rq (race minus
    qualifying gap, % points), rq_c (rq minus the race median of rq: field-relative), car_rq (mean rq_c of the
    car's drivers in the race) and rq_tm (rq_c minus car_rq; NaN for a car with one valid driver).
    Entries without a valid qualifying gap or race pace have NaN offsets."""
    e = entries[["race_id", "season", "round", "driverId", "lineage", "q_best_gap"]].merge(
        pace[KEY + [gap_col]], on=KEY, how="left")
    e["q_gap_pct"] = e["q_best_gap"] * 100.0
    e["race_gap_pct"] = e[gap_col]
    e["rq"] = e["race_gap_pct"] - e["q_gap_pct"]
    e["rq_c"] = e["rq"] - e.groupby("race_id")["rq"].transform("median")
    grp = e.groupby(["race_id", "lineage"])["rq_c"]
    e["car_rq"] = grp.transform("mean")
    e["car_n"] = grp.transform("count")
    e["rq_tm"] = np.where(e["car_n"] >= 2, e["rq_c"] - e["car_rq"], np.nan)
    return e.drop(columns=["q_best_gap"]).sort_values(["race_id", "driverId"]).reset_index(drop=True)


def _decayed(df: pd.DataFrame, keys: list[str], col: str, h: float | None) -> tuple[pd.Series, pd.Series]:
    """Shrunk decayed mean (prior 0) and the number of valid earlier observations, per row, from earlier
    elements of each group only. `df` must be sorted by race_id within each group."""
    mean = pd.Series(np.nan, index=df.index)
    cnt = pd.Series(0, index=df.index)
    for _, g in df.groupby(keys, sort=False):
        v = g[col].to_numpy(float)
        mean.loc[g.index] = _ewm_prior(g["race_id"].to_numpy(float), v, h, 0.0)
        cnt.loc[g.index] = np.concatenate([[0], np.cumsum(~np.isnan(v))[:-1]])
    return mean, cnt


def rq_history_features(rq: pd.DataFrame, halflives: tuple[int, ...] = HALFLIVES,
                        season_half: int | None = 50) -> pd.DataFrame:
    """Per (race_id, driverId): con_rq_ew{h} (car lineage), drv_rq_ew{h} (driver, includes the car),
    drvtm_rq_ew{h} (driver relative to the teammate) — each a decayed mean over EARLIER races only —, and the
    counts of earlier valid observations (con_n, drv_n, drvtm_n). `con_rq_ssn` is the car mean restricted to
    earlier races of the same season (half-life `season_half`). Features for race r depend only on rows with
    race_id < r (tested by blanking a race)."""
    cr = (rq.groupby(["lineage", "race_id"], as_index=False)
            .agg(season=("season", "first"), car_rq=("rq_c", "mean")).sort_values(["lineage", "race_id"]).reset_index(drop=True))
    dr = rq.sort_values(["driverId", "race_id"]).reset_index(drop=True)
    out_c = cr[["lineage", "race_id"]].copy()
    out_d = dr[["race_id", "driverId"]].copy()
    for h in halflives:
        out_c[f"con_rq_ew{h}"], out_c["con_n"] = _decayed(cr, ["lineage"], "car_rq", h)
        out_d[f"drv_rq_ew{h}"], out_d["drv_n"] = _decayed(dr, ["driverId"], "rq_c", h)
        out_d[f"drvtm_rq_ew{h}"], out_d["drvtm_n"] = _decayed(dr, ["driverId"], "rq_tm", h)
    if season_half is not None:
        out_c["con_rq_ssn"], out_c["con_n_ssn"] = _decayed(cr, ["lineage", "season"], "car_rq", season_half)
    res = rq[["race_id", "driverId", "lineage"]].merge(out_c, on=["lineage", "race_id"], how="left")
    return res.merge(out_d, on=["race_id", "driverId"], how="left").drop(columns="lineage")
