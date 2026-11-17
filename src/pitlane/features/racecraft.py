"""Plan-3 race-craft and pit-crew features (PLAN3.md section 4, M20; definitions in methodologies/M20_racecraft/README.md).

Two layers, both pure functions so that they can be leakage-tested:

1. Race values (one row per unit and race, from that race's own laps / stops / grid only; outcomes of the race,
   used as persistence targets and as inputs of the history means, never as features of the same race):
   * `driver_lap_values`  cons (robust residual spread) and deg (pooled within-stint slope), driver level
   * `lap1_values`        lap-1 gain adjusted for grid slot, driver level
   * `crew_values`        pit-lane time loss and slow-stop rate, TEAM (car lineage) level
   Each `*_c` column is field-relative (race median, or mean for the rate).
2. `history_values` - shrunk, time-decayed mean of the `*_c` value over EARLIER races of the same unit
   (`pace._decayed`: race_id as time, one pseudo-observation of prior 0), plus the count of earlier valid values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitlane.features.pace import KEY, RACE, _decayed

MIN_STINT_LAPS = 5      # clean laps needed in a stint to enter cons / deg
MIN_DRIVER_LAPS = 10    # clean laps (in qualifying stints) needed for a driver's cons / deg in a race
MAD_SCALE = 1.4826
MIN_LAP1_DRIVERS = 8
RED_FLAG_MARGIN_S = 20.0
MAX_STOP_S = 60.0       # absolute cap applied first: in red-flag races most stops are long and inflate the race median
SLOW_S = 3.0
MIN_CREW_STOPS = 2


# ------------------------------------------------------------------------------------ cons and deg
def _stint_fit(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Within-stint demeaned (Sxy, Sxx, residuals of the stint's own OLS line)."""
    xd, yd = x - x.mean(), y - y.mean()
    sxx = float((xd ** 2).sum())
    sxy = float((xd * yd).sum())
    slope = sxy / sxx if sxx > 0 else 0.0
    return sxy, sxx, yd - slope * xd


def driver_lap_values(m: pd.DataFrame, min_stint: int = MIN_STINT_LAPS, min_laps: int = MIN_DRIVER_LAPS) -> pd.DataFrame:
    """cons, deg (and cons_c, deg_c) per (season, round, driverId) from the marked laps `m`
    (`pace.mark_laps` output; needs clean, stint_age, lap, lap_time_s). Stint id = lap - stint_age. Lap times are
    expressed in % of the driver's median clean lap of the race. cons = 1.4826 * median |residual| over stints with
    >= min_stint clean laps; deg = sum Sxy / sum Sxx over the same stints. Drivers with < min_laps such laps are
    dropped."""
    c = m[m["clean"]].copy()
    c["stint"] = (c["lap"] - c["stint_age"]).astype(int)
    rows = []
    for (s, r, d), g in c.groupby(KEY):
        y = g["lap_time_s"].to_numpy(float)
        g = g.assign(ypct=100.0 * y / np.median(y))
        res, sxy, sxx, n = [], 0.0, 0.0, 0
        for _, st in g.groupby("stint"):
            if len(st) < min_stint:
                continue
            a, b, e = _stint_fit(st["ypct"].to_numpy(float), st["lap"].to_numpy(float))
            sxy, sxx, n = sxy + a, sxx + b, n + len(st)
            res.append(e)
        if n < min_laps or sxx <= 0:
            continue
        rows.append(dict(season=s, round=r, driverId=d, n_stint_laps=n,
                         cons=float(MAD_SCALE * np.median(np.abs(np.concatenate(res)))), deg=sxy / sxx))
    out = pd.DataFrame(rows, columns=["season", "round", "driverId", "n_stint_laps", "cons", "deg"])
    return field_relative(out, ["cons", "deg"])


def field_relative(df: pd.DataFrame, cols: list[str], how: str = "median") -> pd.DataFrame:
    """Add `<col>_c` = value minus the race median (or mean) over the units present in the race."""
    out = df.copy()
    for col in cols:
        out[f"{col}_c"] = out[col] - out.groupby(RACE)[col].transform(how)
    return out


# ------------------------------------------------------------------------------------ lap 1
def lap1_positions(laps: pd.DataFrame) -> pd.DataFrame:
    """Position after lap 1 per (season, round, driverId): the stored `position` of lap 1 where the race has one,
    else the rank of the lap-1 time among drivers with a lap-1 time. Column `p1_source` says which."""
    l1 = laps[laps["lap"] == 1][KEY + ["position", "lap_time_s"]].copy()
    l1["rank_time"] = l1.groupby(RACE)["lap_time_s"].rank(method="first")
    has = l1.groupby(RACE)["position"].transform(lambda x: x.notna().all())
    l1["p1"] = np.where(has, l1["position"], l1["rank_time"])
    l1["p1_source"] = np.where(has, "stored", "time_rank")
    return l1[KEY + ["p1", "rank_time", "p1_source"]]


def lap1_values(pos1: pd.DataFrame, entries: pd.DataFrame, min_drivers: int = MIN_LAP1_DRIVERS) -> pd.DataFrame:
    """gain = grid - p1 and the grid-adjusted `lap1_c` (residual of a per-race OLS of gain on grid). Pit-lane
    starters are excluded; races with fewer than `min_drivers` drivers get no values. `entries`: season, round,
    driverId, grid, pit_start."""
    e = entries[entries["pit_start"] == 0][KEY + ["grid"]]
    d = pos1.merge(e, on=KEY, how="inner")
    d["gain"] = d["grid"] - d["p1"]
    parts = []
    for _, g in d.groupby(RACE):
        if len(g) < min_drivers or g["grid"].nunique() < 2:
            continue
        x, y = g["grid"].to_numpy(float), g["gain"].to_numpy(float)
        b = np.cov(x, y, bias=True)[0, 1] / np.var(x)
        parts.append(g.assign(lap1_c=y - y.mean() - b * (x - x.mean())))
    return pd.concat(parts, ignore_index=True) if parts else d.assign(lap1_c=np.nan).iloc[0:0]


# ------------------------------------------------------------------------------------ pit crew
def valid_stops(pits: pd.DataFrame, margin: float = RED_FLAG_MARGIN_S, max_s: float = MAX_STOP_S) -> pd.DataFrame:
    """Stops with a finite duration that are not red-flag / repair stops: first stops longer than `max_s` are
    dropped (a red flag makes most stops of a race long, so the median alone would not catch them; amendment 1 of
    the card), then stops longer than the race median of the remainder + margin. Race-wide medians are fine here: the
    value of race r is only used for races after r. Adds `loss_s` = duration - race median of the kept stops."""
    p = pits[KEY + ["lap", "duration_s"]].dropna(subset=["duration_s"]).copy()
    p = p[(p["duration_s"] > 0) & (p["duration_s"] <= max_s)]
    med_all = p.groupby(RACE)["duration_s"].transform("median")
    p = p[p["duration_s"] <= med_all + margin].copy()
    p["loss_s"] = p["duration_s"] - p.groupby(RACE)["duration_s"].transform("median")
    return p


def crew_values(pits: pd.DataFrame, entries: pd.DataFrame, margin: float = RED_FLAG_MARGIN_S,
                slow_s: float = SLOW_S, min_stops: int = MIN_CREW_STOPS, max_s: float = MAX_STOP_S) -> pd.DataFrame:
    """Per (season, round, lineage): n_stops, crew_loss (median time loss, s), crew_slow (share of stops with loss >
    slow_s) and the field-relative crew_loss_c (minus race median over teams), crew_slow_c (minus race mean).
    Teams with < min_stops kept stops get no value. `entries`: season, round, driverId, lineage."""
    p = valid_stops(pits, margin, max_s).merge(entries[KEY + ["lineage"]], on=KEY, how="inner")
    p["slow"] = (p["loss_s"] > slow_s).astype(float)
    g = (p.groupby(RACE + ["lineage"]).agg(n_stops=("loss_s", "size"), crew_loss=("loss_s", "median"),
                                           crew_slow=("slow", "mean")).reset_index())
    g = g[g["n_stops"] >= min_stops]
    g = field_relative(g, ["crew_loss"], "median")
    return field_relative(g, ["crew_slow"], "mean")


# ------------------------------------------------------------------------------------ history and teammate view
def teammate_relative(df: pd.DataFrame, col: str, team: pd.DataFrame) -> pd.Series:
    """`col` minus the mean of the team's eligible drivers in that race (NaN if the driver is alone). `team`:
    season, round, driverId, lineage."""
    m = df[KEY + [col]].merge(team[KEY + ["lineage"]], on=KEY, how="left")
    grp = m.groupby(RACE + ["lineage"])[col]
    mean, n = grp.transform("mean"), grp.transform("count")
    return pd.Series(np.where(n >= 2, m[col] - mean, np.nan), index=df.index)


def history_values(df: pd.DataFrame, unit: str, col: str, halflives: tuple[int, ...]) -> pd.DataFrame:
    """For every row, `<col>_h{h}` = shrunk decayed mean of `col` over EARLIER races of the same `unit`, and
    `<col>_n` = count of earlier valid values. `df` needs race_id and `unit`; it is sorted here. Row r depends only
    on rows with race_id < r (tested by truncation)."""
    d = df.sort_values([unit, "race_id"]).reset_index(drop=True)
    out = d[["race_id", unit]].copy()
    for h in halflives:
        out[f"{col}_h{h}"], out[f"{col}_n"] = _decayed(d, [unit], col, h)
    return out
