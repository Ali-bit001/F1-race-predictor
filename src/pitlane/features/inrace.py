"""In-race state features for the M19 in-race forecast (PLAN3.md section 3 "In-race state", section 4 M19).

Everything here is a function of laps 1..k and pit stops on laps <= k of ONE race, plus the list of entered drivers
(known before the race). Rows of the lap and pit tables with lap > k are dropped on entry to `race_state`, so changing
them cannot change the output (leakage test: tests for R13).

Running order at the end of lap k (documented derivation, works for every season including 2023 where the lap table
has no position column):
  * runners = entries with a lap-k record; ordered by cumulative race time through lap k (sum of their laps 1..k).
    Jolpica lap 1 is the time from the start to the line, so the grid stagger is included. On 2021-2022, where Jolpica
    also stores the position, this order agrees with the stored one (checked in the run, see data_checks.csv).
  * entries that have no lap-k record (retired, or never completed lap 1) are known to be out at lap k. They are
    placed behind all runners, ordered by laps completed (more = better), ties by driverId (outcome-free).
Pit stops: `red_flag_suspect` stored in the pit table is NOT used (it compares each stop with the median of the whole race,
which includes stops after lap k). The flag is recomputed here from the stops on laps <= k only: a stop is suspect if its
duration exceeds the median duration of those stops by more than `red_flag_margin_s` (configs/ingest3.yaml, pits.red_flag_margin_s).
Race distance L (which sets the horizon lap) is the SCHEDULED distance known before the race, see `scheduled_distance`.
Sources are never mixed inside a race: the caller passes the lap table of one source (Jolpica for 2014-2023).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

GAP_CAP_S = 60.0
FEATURE_COLS = ["ir_pos", "ir_gap", "ir_pits", "ir_out", "ir_def"]
STATE_COLS = ["driverId", "laps_done", "cum_s", "run_pos", "gap_ahead_s", "pits_made", "out"]


def horizon_lap(total_laps: int, frac: float | None, available_laps: int | None = None) -> int:
    """Lap k of a horizon: frac=None -> after lap 1 (H4); else ceil(frac * race distance), at least 1.
    `total_laps` is the SCHEDULED race distance (known before the race). If the race stopped before that lap
    (`available_laps` = last lap that exists in the lap table, e.g. 2021 Belgian GP: 1), the last available lap <= the
    horizon lap is used instead."""
    k = 1 if frac is None else int(min(max(1, math.ceil(frac * total_laps)), max(1, total_laps)))
    if available_laps is not None:
        k = max(1, min(k, int(available_laps)))
    return k


def stops_before(pits: pd.DataFrame | None, k: int, red_flag_margin_s: float = 20.0) -> pd.DataFrame:
    """Stops made on laps <= k that are not suspected red-flag stops. The flag is computed from these stops only
    (median of their durations + margin); the stored `red_flag_suspect` column is ignored. NaN durations are never suspect."""
    if pits is None or not len(pits):
        return pd.DataFrame({"driverId": [], "lap": []})
    pp = pits[pits["lap"] <= k]
    if "duration_s" in pp and len(pp):
        dur = pp["duration_s"].astype(float)
        sus = dur > dur.median() + red_flag_margin_s       # NaN median/duration -> False
        pp = pp[~sus.to_numpy()]
    return pp


def race_state(laps: pd.DataFrame, pits: pd.DataFrame | None, drivers: list[str], k: int,
               red_flag_margin_s: float = 20.0) -> pd.DataFrame:
    """State of one race at the end of lap k, one row per entered driver (order of `drivers`).

    laps: columns driverId, lap, lap_time_s (one source). pits: driverId, lap, duration_s (may be None/empty); a stored
    `red_flag_suspect` column is ignored (see module docstring). Only rows with lap <= k are used.
    """
    lp = laps[laps["lap"] <= k]
    g = lp.groupby("driverId")
    done = g["lap"].max().reindex(drivers).fillna(0).astype(int)
    cum = g["lap_time_s"].sum(min_count=1).reindex(drivers)
    st = pd.DataFrame({"driverId": drivers, "laps_done": done.to_numpy(), "cum_s": cum.to_numpy()})
    st["out"] = (st["laps_done"] < k).astype(int)
    if pits is not None and len(pits):
        pp = stops_before(pits, k, red_flag_margin_s)
        n_stops = pp.groupby("driverId").size().reindex(drivers).fillna(0).to_numpy()
    else:
        n_stops = np.zeros(len(drivers))
    st["pits_made"] = n_stops.astype(float)
    # ordering: runners by cumulative time, then retirees by laps done (desc); every tie by driverId
    run = st["out"] == 0
    st["_t"] = np.where(run, st["cum_s"].fillna(np.inf), 0.0)
    st["_d"] = np.where(run, 0, -st["laps_done"])
    st = st.sort_values(["out", "_d", "_t", "driverId"], kind="mergesort").reset_index(drop=True)
    st["run_pos"] = np.arange(1, len(st) + 1)
    gap = st["cum_s"].diff()
    gap[(st["out"] == 1) | (st["run_pos"] == 1)] = 0.0
    prev_out = st["out"].shift(1, fill_value=0)
    gap[(prev_out == 1)] = 0.0
    st["gap_ahead_s"] = gap.fillna(0.0)
    return st.drop(columns=["_t", "_d"])


def state_features(st: pd.DataFrame, k: int) -> pd.DataFrame:
    """Model inputs from a `race_state` table (all scaled so that a ridge penalty is comparable to the weekend signals)."""
    n = len(st)
    runners = st["out"] == 0
    med = float(st.loc[runners, "pits_made"].median()) if runners.any() else 0.0
    f = pd.DataFrame({"driverId": st["driverId"]})
    f["ir_pos"] = (st["run_pos"] - 1) / max(n - 1, 1) - 0.5          # centred running position, lower = ahead
    f["ir_gap"] = np.minimum(st["gap_ahead_s"], GAP_CAP_S) / 10.0     # gap to the car ahead, tens of seconds
    f["ir_pits"] = st["pits_made"] - med                              # stops made relative to the median runner
    f["ir_out"] = st["out"].astype(float)                             # known to be out at lap k
    f["ir_def"] = np.where(st["out"] == 1, (k - st["laps_done"]) / max(k, 1), 0.0)   # share of the k laps not completed
    return f


def build_state_table(entries: pd.DataFrame, laps: pd.DataFrame, pits: pd.DataFrame | None,
                      k_by_race: dict[int, int], red_flag_margin_s: float = 20.0) -> pd.DataFrame:
    """State and features for every (race_id, driverId) of races in `k_by_race`. `entries` needs race_id, season, round,
    driverId. `laps`/`pits` carry season, round. Returns a frame with race_id, driverId, k, STATE_COLS and FEATURE_COLS."""
    lap_g = dict(tuple(laps.groupby(["season", "round"])))
    pit_g = dict(tuple(pits.groupby(["season", "round"]))) if pits is not None else {}
    out = []
    for rid, e in entries.groupby("race_id", sort=True):
        if rid not in k_by_race:
            continue
        key = (int(e["season"].iloc[0]), int(e["round"].iloc[0]))
        if key not in lap_g:
            continue
        k = int(k_by_race[rid])
        st = race_state(lap_g[key], pit_g.get(key), e["driverId"].tolist(), k, red_flag_margin_s)
        ft = state_features(st, k)
        t = st.merge(ft, on="driverId")
        t.insert(0, "k", k)
        t.insert(0, "race_id", rid)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def race_distance(laps: pd.DataFrame) -> dict[tuple[int, int], int]:
    """Last lap number in the lap table per (season, round). This is what the race actually ran, NOT a pre-race quantity:
    it differs from the scheduled distance for races that were shortened (2021 R12, 2022 R7, R17, R18, 2023 R14 in dev).
    Used only for races that are over before the one being forecast (training) and as `available_laps`."""
    return {k: int(v) for k, v in laps.groupby(["season", "round"])["lap"].max().items()}


def scheduled_distance(laps: pd.DataFrame, table: pd.DataFrame) -> dict[tuple[int, int], int]:
    """Race distance used to place the horizons. Races listed in `table` (columns season, round, scheduled_laps; the dev
    block, data/external/scheduled_laps.csv) use the scheduled distance announced before the race; every other race (2014-2020,
    used only as training history of the in-race coefficients, so already run) uses the last lap in its lap table."""
    out = race_distance(laps)
    for r in table.itertuples():
        out[(int(r.season), int(r.round))] = int(r.scheduled_laps)
    return out


def attach(entries: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Entries (same row order, same index) with the FEATURE_COLS of `table` merged in; NaN for races without laps."""
    cols = ["race_id", "driverId"] + FEATURE_COLS
    m = entries[["race_id", "driverId"]].merge(table[cols], on=["race_id", "driverId"], how="left")
    assert len(m) == len(entries)
    out = entries.copy()
    for c in FEATURE_COLS:
        out[c] = m[c].to_numpy()
    return out
