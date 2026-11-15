"""Plan-2 weekend and history features (PLAN2.md section 3).

Weekend features (qualifying detail, sprint, practice, weather) are measured *before* the race starts,
so they can never carry race-day information. History features (incident / mechanical rates, circuit
grid stickiness, circuit DNF rate) are built from earlier races only and are covered by
tests/test_leakage_plan2.py.

Scaling constants below are fixed by hand from the typical size of each raw signal (never from race outcomes
and never fitted), so no outcome information flows from the test blocks.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from pitlane.config import PROCESSED, RAW
from pitlane.data.ingest import _t, fetch_pages
from pitlane.data.ingest2 import RAW2, openf1_sessions
from pitlane.features.build import _series_feature

NEW_REGS = (2014, 2017, 2022, 2026)
WET_MM = 0.5            # >= 0.5 mm in the 3 h window counts as wet
FC_FROM = 2020          # Open-Meteo historical-forecast archive is only trusted from this season
INCIDENT = re.compile(r"collision|accident|spun|damage|puncture|debris|crash|fatal", re.I)
KEY = ["season", "round", "driverId"]


# ------------------------------------------------------------------------------------ qualifying
def qualifying_rich() -> pd.DataFrame:
    rows = []
    for season in range(2014, 2027):
        for race in fetch_pages("qualifying", season):
            recs = []
            for x in race["rows"]:
                q = [_t(x.get(k)) for k in ("Q1", "Q2", "Q3")]
                have = [i for i, v in enumerate(q) if not np.isnan(v)]
                recs.append(dict(season=season, round=int(race["round"]), driverId=x["Driver"]["driverId"],
                                 constructorId=x["Constructor"]["constructorId"], q=q,
                                 best=min([q[i] for i in have]) if have else np.nan,
                                 reach=(max(have) + 1) if have else np.nan,
                                 seg=q[max(have)] if have else np.nan))
            d = pd.DataFrame(recs)
            if d.empty:
                continue
            d["q_best_gap"] = d["best"] / d["best"].min() - 1
            seg_best = {k: np.nanmin([r[k] for r in d["q"]]) if not all(np.isnan(r[k]) for r in d["q"]) else np.nan
                        for k in range(3)}
            d["q_seg_gap"] = [(s / seg_best[int(r) - 1] - 1) if not np.isnan(r) else np.nan
                              for s, r in zip(d["seg"], d["reach"])]
            d["q_reach"] = d["reach"]
            tm = []
            for i, r in d.iterrows():
                mates = d[(d["constructorId"] == r["constructorId"]) & (d["driverId"] != r["driverId"])]["q_best_gap"]
                tm.append(r["q_best_gap"] - mates.mean() if mates.notna().any() and not np.isnan(r["q_best_gap"]) else np.nan)
            d["q_tm_gap"] = tm
            rows.append(d[KEY + ["q_best_gap", "q_seg_gap", "q_reach", "q_tm_gap"]])
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------------ sprint
def sprint_features() -> pd.DataFrame:
    rows = []
    for season in range(2021, 2027):
        for race in fetch_pages("sprint", season):
            n = len(race["rows"])
            for x in race["rows"]:
                if not str(x.get("position", "")).isdigit():
                    continue
                pos, grid = int(x["position"]), int(x["grid"]) if str(x.get("grid", "0")).isdigit() else 0
                grid = grid if grid > 0 else n
                rows.append(dict(season=season, round=int(race["round"]), driverId=x["Driver"]["driverId"],
                                 sprint_pos=(pos - 1) / max(n - 1, 1), sprint_gain=(grid - pos) / max(n - 1, 1)))
    return pd.DataFrame(rows)


def sprint_weekends() -> set[tuple[int, int]]:
    return {(s, int(r["round"])) for s in range(2021, 2027) for r in fetch_pages("sprint", s) if r["rows"]}


# ------------------------------------------------------------------------------------ practice
def session_practice_metrics(rows: list) -> pd.DataFrame:
    """rows: [driver_number, lap_number, lap_duration_s | None, is_pit_out_lap]. Returns one row per driver:
    gap_best (best lap / session best - 1), long_gap (long-run median lap / field median - 1, NaN if no
    stint of >= 5 clean laps), laps_rel (laps completed / field median)."""
    df = pd.DataFrame(rows, columns=["dn", "lap", "dur", "pit_out"]).sort_values(["dn", "lap"])
    df["valid"] = df["dur"].notna() & (df["dur"] > 0)
    session_best = df.loc[df["valid"] & (df["pit_out"] == 0), "dur"].min()
    out = []
    for dn, g in df.groupby("dn"):
        g = g.reset_index(drop=True)
        stint = g["pit_out"].cumsum()
        is_in = g["pit_out"].shift(-1, fill_value=0) == 1          # in-lap: the lap before a pit-out lap
        ok = g["valid"] & (g["pit_out"] == 0)
        best = g.loc[ok, "dur"].min() if ok.any() else np.nan
        clean = ok & ~is_in & (g["dur"] <= 1.07 * session_best)
        longs = []
        for _, gs in g[clean].groupby(stint[clean]):
            if len(gs) >= 5:
                longs += gs["dur"].tolist()
        out.append(dict(dn=dn, best=best, long=float(np.median(longs)) if longs else np.nan, laps=int(g["valid"].sum())))
    d = pd.DataFrame(out)
    d["gap_best"] = d["best"] / session_best - 1
    lg = d["long"].dropna()
    d["long_gap"] = d["long"] / lg.median() - 1 if len(lg) >= 5 else np.nan
    d["laps_rel"] = d["laps"] / max(d["laps"].median(), 1)
    return d[["dn", "gap_best", "long_gap", "laps_rel"]]


def _driver_maps() -> dict[int, dict[str, str]]:
    """season -> {code or surname: driverId} from the cached Jolpica results."""
    out: dict[int, dict[str, str]] = {}
    for season in range(2023, 2027):
        m: dict[str, str] = {}
        for race in fetch_pages("results", season):
            for x in race["rows"]:
                d = x["Driver"]
                m[d.get("code", "").upper()] = d["driverId"]
                m[d["familyName"].upper()] = d["driverId"]
        out[season] = m
    return out


def practice_features(entries: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, round, driverId) for races with OpenF1 practice data (2023+)."""
    races = entries.groupby(["season", "round"], as_index=False)["date"].first()
    races["date"] = pd.to_datetime(races["date"])
    maps = _driver_maps()
    rows = []
    for year in (2023, 2024, 2025, 2026):
        sess = openf1_sessions(year) or []
        by_m: dict[int, list[dict]] = {}
        for s in sess:
            by_m.setdefault(s["meeting_key"], []).append(s)
        ry = races[races["season"] == year]
        for mk, ss in by_m.items():
            race_s = [s for s in ss if s["session_name"] == "Race"]
            if not race_s:
                continue
            d0 = pd.Timestamp(datetime.fromisoformat(race_s[0]["date_start"]).astimezone(timezone.utc)).tz_localize(None).normalize()
            near = ry[(ry["date"] - d0).abs() <= pd.Timedelta(days=1)]
            if near.empty:
                continue
            rnd = int(near.iloc[0]["round"])
            dfile = RAW2 / "openf1" / f"drivers_{mk}.json"
            if not dfile.exists():
                continue
            nums = {}
            for r in json.loads(dfile.read_text(encoding="utf-8")):
                did = maps[year].get(str(r.get("name_acronym", "")).upper()) or maps[year].get(str(r.get("last_name", "")).upper())
                if did:
                    nums[r["driver_number"]] = did
            wfile = RAW2 / "openf1" / f"weather_{mk}.json"
            wx = json.loads(wfile.read_text(encoding="utf-8")) if wfile.exists() else {}
            per_session, any_rain = [], 0
            for s in ss:
                if s["session_type"] != "Practice":
                    continue
                lf = RAW2 / "openf1" / f"laps_{s['session_key']}.json"
                if not lf.exists():
                    continue
                laps = json.loads(lf.read_text(encoding="utf-8"))
                if len(laps) < 50:
                    continue
                if wx.get(str(s["session_key"]), {}).get("rain_frac", 0.0) > 0.2:
                    any_rain = 1
                    continue                       # wet practice: pace gaps are not comparable
                m = session_practice_metrics(laps)
                m["driverId"] = m["dn"].map(nums)
                per_session.append(m.dropna(subset=["driverId"]))
            if not per_session and not any_rain:
                continue
            if per_session:
                agg = (pd.concat(per_session).groupby("driverId")
                       .agg(fp_gap=("gap_best", "mean"), fp_long=("long_gap", "mean"), fp_laps=("laps_rel", "mean"))
                       .reset_index())
            else:
                agg = pd.DataFrame(columns=["driverId", "fp_gap", "fp_long", "fp_laps"])
            agg["season"], agg["round"], agg["rain_fp"] = year, rnd, any_rain
            rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=KEY + ["fp_gap", "fp_long", "fp_laps", "rain_fp"])


# ------------------------------------------------------------------------------------ weather
def weather_features() -> pd.DataFrame:
    rows = []
    for season in range(2014, 2027):
        for race in fetch_pages("results", season):
            f = RAW2 / "weather" / f"wx_{season}_{race['round']}.json"
            w = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
            sat = w.get("sat_mm")
            fc = w.get("fc_mm") if season >= FC_FROM else None
            rows.append(dict(season=season, round=int(race["round"]),
                             sat_mm=np.nan if sat is None else sat, fc_mm=np.nan if fc is None else fc))
    d = pd.DataFrame(rows)
    d["wet_q"] = np.where(d["sat_mm"].isna(), np.nan, (d["sat_mm"] >= WET_MM).astype(float))
    d["rain_fc"] = np.where(d["fc_mm"].isna(), np.nan, (d["fc_mm"] >= WET_MM).astype(float))
    d["wx_missing"] = d["rain_fc"].isna().astype(float)
    d["mismatch"] = np.where(d["wet_q"].notna() & d["rain_fc"].notna(), (d["wet_q"] != d["rain_fc"]).astype(float), 0.0)
    return d


# ------------------------------------------------------------------------------------ history
def _shrunk_by_circuit(rl: pd.DataFrame, col: str, default: float, pseudo: float = 2.0) -> np.ndarray:
    """Expanding mean of a race-level quantity per circuit, shrunk toward the running global mean.
    Row i uses races strictly before i (race i's own value is added afterwards)."""
    out = np.full(len(rl), np.nan)
    gs, gn = 0.0, 0
    state: dict[str, tuple[float, int]] = {}
    for i, (c, v) in enumerate(zip(rl["circuitId"], rl[col])):
        glob = gs / gn if gn else default
        s, n = state.get(c, (0.0, 0))
        out[i] = (s + pseudo * glob) / (n + pseudo)
        if not np.isnan(v):
            state[c] = (s + v, n + 1)
            gs, gn = gs + v, gn + 1
    return out


def add_history2(df: pd.DataFrame) -> pd.DataFrame:
    """Leak-safe history features. `df` must be sorted by (race_id, pos) with a RangeIndex."""
    df = df.copy()
    status = df["status"].fillna("")
    inc = status.str.contains(INCIDENT) & (df["dnf"] == 1)
    df["inc"] = np.where(df["dnf"].isna(), np.nan, inc.astype(float))
    df["mech"] = np.where(df["dnf"].isna(), np.nan, ((df["dnf"] == 1) & ~inc).astype(float))
    for h in (16,):
        df[f"drv_inc_ew{h}"] = _series_feature(df, ["driverId"], "inc", h, prior=0.06)
    cr = (df.groupby(["lineage", "race_id"], as_index=False)
            .agg(race_t=("race_t", "first"), cmech=("mech", "mean")).sort_values(["lineage", "race_id"]))
    cr["con_mech_ew16"] = _series_feature(cr.reset_index(drop=True), ["lineage"], "cmech", 16, prior=0.07).to_numpy()
    df = df.merge(cr[["lineage", "race_id", "con_mech_ew16"]], on=["lineage", "race_id"], how="left")

    def rho(g: pd.DataFrame) -> float:
        a, b = g["grid"].to_numpy(float), g["pos"].to_numpy(float)
        if np.isnan(b).any() or len(g) < 5 or np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    recs = []
    for rid, g in df.groupby("race_id", sort=True):
        done = not g["pos"].isna().any()
        recs.append(dict(race_id=rid, circuitId=g["circuitId"].iloc[0], rho=rho(g),
                         vol=float((g["grid"] - g["pos"]).abs().mean() / len(g)) if done else np.nan,
                         cdnf=float(g["dnf"].mean()) if not g["dnf"].isna().any() else np.nan))
    rl = pd.DataFrame(recs)
    rl["stick"] = _shrunk_by_circuit(rl, "rho", 0.6)
    rl["circ_vol"] = _shrunk_by_circuit(rl, "vol", 0.15)
    rl["circ_dnf"] = _shrunk_by_circuit(rl, "cdnf", 0.15)
    df = df.merge(rl[["race_id", "stick", "circ_vol", "circ_dnf"]], on="race_id", how="left")
    df["newregs"] = df["season"].isin(NEW_REGS).astype(float)
    df = df.sort_values(["driverId", "race_id"]).reset_index(drop=True)
    prev = df.groupby("driverId")["lineage"].shift(1)
    df["new_team"] = ((prev.notna()) & (prev != df["lineage"])).astype(float)
    first = df.groupby("driverId")["season"].transform("min")
    df["rookie"] = ((df.groupby("driverId").cumcount() < 20) & (first > df["season"].min())).astype(float)
    return df.sort_values(["race_id", "pos"], kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------------------------ scaling
def make_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-scaled, centred, NaN-free columns `x_*` for the additive latent weekend term (fixed constants).
    Race-wide flags are constant within a race and so cannot change a race order on their own; in the
    latent model they only act through interactions with the qualifying position (`x_q_x_*`)."""
    q = df["quali_norm"] - 0.5
    x = pd.DataFrame(index=df.index)
    x["x_q_best_gap"] = df["q_best_gap"].clip(0, 0.10).fillna(0.10) * 10 - 0.3
    x["x_q_seg_gap"] = df["q_seg_gap"].clip(0, 0.05).fillna(0.05) * 20 - 0.2
    x["x_q_reach"] = df["q_reach"].fillna(1.0) - 2.0
    x["x_q_tm_gap"] = df["q_tm_gap"].clip(-0.05, 0.05).fillna(0.0) * 20
    x["x_fp_gap"] = np.where(df["fp_gap"].notna(), df["fp_gap"].clip(0, 0.05) * 20 - 0.25, 0.0)     # typical gap ~1.3%
    x["x_fp_long"] = np.where(df["fp_long"].notna(), df["fp_long"].clip(-0.05, 0.05) * 20, 0.0)
    x["x_fp_laps"] = np.where(df["fp_laps"].notna(), (df["fp_laps"] - 1).clip(-1, 1) * 3, 0.0)
    x["x_sprint_pos"] = np.where(df["sprint_pos"].notna(), df["sprint_pos"] - 0.5, 0.0)
    x["x_sprint_gain"] = np.where(df["sprint_gain"].notna(), df["sprint_gain"].clip(-0.5, 0.5), 0.0)
    x["x_q_x_mismatch"] = q * df["mismatch"].fillna(0.0)
    x["x_q_x_rainfc"] = q * df["rain_fc"].fillna(0.0)
    x["x_q_x_wetq"] = q * df["wet_q"].fillna(0.0)
    return pd.concat([df, x], axis=1)


GROUPS_L = {   # latent weekend-term columns per information group
    "qrich": ["x_q_best_gap", "x_q_seg_gap", "x_q_reach", "x_q_tm_gap"],
    "practice": ["x_fp_gap", "x_fp_long", "x_fp_laps"],
    "sprint": ["x_sprint_pos", "x_sprint_gain"],
    "weather": ["x_q_x_mismatch", "x_q_x_rainfc", "x_q_x_wetq"],
}
GROUPS_G = {   # raw columns for tree models (NaN allowed)
    "qrich": ["q_best_gap", "q_seg_gap", "q_reach", "q_tm_gap"],
    "practice": ["fp_gap", "fp_long", "fp_laps", "fp_missing"],
    "sprint": ["sprint_flag", "sprint_pos", "sprint_gain"],
    "weather": ["wet_q", "rain_fc", "mismatch", "rain_fp", "wx_missing"],
    "circuit": ["stick", "circ_dnf", "circ_vol"],
}
GROUP_ORDER = ["qrich", "practice", "sprint", "weather"]


def l_cols(groups: list[str]) -> list[str]:
    return [c for g in GROUP_ORDER if g in groups for c in GROUPS_L[g]]


def g_cols(groups: list[str]) -> list[str]:
    return [c for g in GROUP_ORDER + ["circuit"] if g in groups for c in GROUPS_G[g]]


# ------------------------------------------------------------------------------------ master builder
def build_entries2(base: pd.DataFrame) -> pd.DataFrame:
    """base = plan-1 entries with features (pipeline.load_entries()). Returns it plus the plan-2 columns,
    in the same row order (the harness and the cached latent strengths rely on it)."""
    df = base.copy()
    for part in (qualifying_rich(), sprint_features(), practice_features(base)):
        df = df.merge(part.drop_duplicates(KEY), on=KEY, how="left")
    sw = sprint_weekends()
    df["sprint_flag"] = [float((s, r) in sw) for s, r in zip(df["season"], df["round"])]
    df = df.merge(weather_features(), on=["season", "round"], how="left")
    have_fp = df.groupby("race_id")["fp_gap"].transform(lambda s: s.notna().any())
    df["fp_missing"] = (~have_fp).astype(float)
    df["rain_fp"] = df["rain_fp"].fillna(0.0)
    df = add_history2(df)
    assert (df[["race_id", "driverId"]].to_numpy() == base[["race_id", "driverId"]].to_numpy()).all(), "row order changed"
    return make_signals(df)


def load_entries2(rebuild: bool = False) -> pd.DataFrame:
    from pitlane.pipeline import load_entries
    cache = PROCESSED / "entries2.parquet"
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache)
    df = build_entries2(load_entries())
    df.to_parquet(cache, index=False)
    return df
