"""PLAN3 R8 phase 1 / M20 race-craft persistence: leakage, hand tables, stored numbers, split discipline.

Sections: (1) history_values on hand tables and the truncation / row-order tests, (2) hand tables for each raw
per-race feature (cons, deg, lap-1, crew), (3) stored persistence numbers reproduced by an independent
implementation, (4) mechanical decisions, README table, no model / leaderboard rows, dev-only filters,
(5) lap-1 position rule, data checks.
Real-data tests skip if the git-ignored inputs are missing; the full 2014-2023 recompute is marked `slow`.
"""
from __future__ import annotations

import importlib.util
import json
import re

import numpy as np
import pandas as pd
import pytest

from pitlane.config import PROCESSED, ROOT, load_yaml
from pitlane.features import pace as P
from pitlane.features import racecraft as RC

METH = ROOT / "methodologies/M20_racecraft"
FINAL = "20261117-0243_m20_persistence_dev"
FIRST = "20261117-0242_m20_persistence_dev"
RUN_DIR, FIRST_DIR = METH / "results" / FINAL, METH / "results" / FIRST
RES = ROOT / "results/plan3"
CSV, SUMMARY = RES / "m20_persistence.csv", RES / "m20_summary.json"
LAPS, PITS = ROOT / "data/interim/plan3/laps.parquet", ROOT / "data/interim/plan3/pits.parquet"
ENT = PROCESSED / "entries2.parquet"
REAL = LAPS.exists() and PITS.exists() and ENT.exists()
STORED = RUN_DIR.exists() and CSV.exists() and SUMMARY.exists()
needs_real = pytest.mark.skipif(not REAL, reason="plan-3 laps / entries not on disk")
needs_stored = pytest.mark.skipif(not STORED, reason="M20 outputs not on disk")
CFG = load_yaml(METH / "config.yaml")
HLS = tuple(CFG["halflives"])
WINS = "winsorised_1_99 (in-sample bounds)"


def _load_run():
    spec = importlib.util.spec_from_file_location("m20_run_under_test", METH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUN = _load_run()
FEATURES = RUN.FEATURES   # feature -> (value col, unit col, level, teammate-relative)


# ====================================================================================== helpers
def _hist(tabs: dict, feat: str) -> pd.DataFrame:
    col, unit, level, _ = FEATURES[feat]
    tab = tabs[level].dropna(subset=[col])
    return RC.history_values(tab.rename(columns={col: "yv"}), unit, "yv", HLS)


def _sorted(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.sort_values(keys).reset_index(drop=True)


def indep_history(df: pd.DataFrame, col: str, unit: str, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed form of the shrunk decayed mean (prior 0, one pseudo observation) and the count of earlier valid
    values, per row of df (any row order). Written from the definition, not from `_ewm_prior`."""
    t, v, u = df["race_id"].to_numpy(float), df[col].to_numpy(float), df[unit].to_numpy()
    mean, cnt = np.zeros(len(df)), np.zeros(len(df), int)
    for k in pd.unique(u):
        ix = np.flatnonzero(u == k)
        for i in ix:
            j = ix[(t[ix] < t[i]) & ~np.isnan(v[ix])]
            w = 0.5 ** ((t[i] - t[j]) / h)
            mean[i] = (w * v[j]).sum() / (1.0 + w.sum())
            cnt[i] = len(j)
    return mean, cnt


def indep_pooled(x: np.ndarray, y: np.ndarray, race: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """Pooled Pearson r with a race-bootstrap percentile CI (resample the sorted race list with
    rng.integers, 10,000 draws, seed 0 - the declared procedure)."""
    r = float(np.corrcoef(x, y)[0, 1])
    races = np.unique(race)
    st = np.array([[x[race == k].sum(), y[race == k].sum(), (x[race == k] ** 2).sum(), (y[race == k] ** 2).sum(),
                    (x[race == k] * y[race == k]).sum(), (race == k).sum()] for k in races], float)
    idx = np.random.default_rng(seed).integers(0, len(races), size=(n_boot, len(races)))
    s = st[idx].sum(1)
    n, sx, sy, sxx, syy, sxy = s[:, 5], s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]
    b = (n * sxy - sx * sy) / np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return dict(r=r, lo=float(lo), hi=float(hi), n_obs=len(x), n_races=len(races))


def indep_persistence(tab: pd.DataFrame, ycol: str, unit: str, h: int, seasons=None, min_obs: int = 5) -> dict:
    t = tab.dropna(subset=[ycol]).reset_index(drop=True)
    m, c = indep_history(t, ycol, unit, h)
    keep = c >= min_obs
    if seasons is not None:
        keep &= t["season"].isin(seasons).to_numpy()
    return indep_pooled(m[keep], t[ycol].to_numpy(float)[keep], t["race_id"].to_numpy()[keep])


# ====================================================================================== stored fixtures
@pytest.fixture(scope="module")
def stored():
    drv = pd.read_parquet(RUN_DIR / "predictions.parquet")
    crew = pd.read_parquet(RUN_DIR / "crew_values.parquet")
    return dict(driver=drv, team=crew)


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(CSV)


@pytest.fixture(scope="module")
def summary():
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


# ====================================================================================== (1) history + truncation
def test_history_values_hand_table():
    # one unit, half-life 1 race; hand values: row0 prior 0, row1 .5/1.5, row2 .25/1.25 (NaN skipped), row3 1.625/1.625
    df = pd.DataFrame(dict(race_id=[0, 1, 2, 3], unit=["a"] * 4, v=[1.0, np.nan, 3.0, 5.0]))
    out = RC.history_values(df, "unit", "v", (1,))
    np.testing.assert_allclose(out["v_h1"].to_numpy(), [0.0, 1 / 3, 0.2, 1.0], rtol=0, atol=1e-15)
    assert out["v_n"].tolist() == [0, 1, 1, 2]


def test_history_values_is_independent_of_own_and_later_values():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(dict(race_id=np.arange(30), unit="a", v=rng.normal(size=30)))
    base = RC.history_values(df, "unit", "v", HLS)
    for r in (5, 17, 29):
        alt = df.copy()
        alt.loc[alt["race_id"] >= r, "v"] = rng.normal(size=(alt["race_id"] >= r).sum()) * 50
        o = RC.history_values(alt, "unit", "v", HLS)
        pd.testing.assert_frame_equal(base[base["race_id"] <= r].reset_index(drop=True),
                                      o[o["race_id"] <= r].reset_index(drop=True), check_exact=True)
        blank = df.copy()
        blank.loc[blank["race_id"] >= r, "v"] = np.nan
        o = RC.history_values(blank, "unit", "v", HLS)
        pd.testing.assert_frame_equal(base[base["race_id"] <= r].reset_index(drop=True),
                                      o[o["race_id"] <= r].reset_index(drop=True), check_exact=True)
    ctrl = df.copy()
    ctrl.loc[ctrl["race_id"] == 3, "v"] += 10          # control: an EARLIER race does move later rows
    o = RC.history_values(ctrl, "unit", "v", HLS)
    assert (o["v_h50"].to_numpy()[4:] != base["v_h50"].to_numpy()[4:]).all()
    assert (o["v_h50"].to_numpy()[:4] == base["v_h50"].to_numpy()[:4]).all()


def test_history_values_row_order_independent():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(dict(race_id=np.tile(np.arange(20), 4), unit=np.repeat(list("abcd"), 20), v=rng.normal(size=80)))
    df.loc[rng.choice(80, 12, replace=False), "v"] = np.nan
    a = _sorted(RC.history_values(df, "unit", "v", HLS), ["unit", "race_id"])
    b = _sorted(RC.history_values(df.sample(frac=1, random_state=3), "unit", "v", HLS), ["unit", "race_id"])
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_history_matches_closed_form():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(dict(race_id=np.tile(np.arange(0, 60, 3), 3), unit=np.repeat(list("abc"), 20), v=rng.normal(size=60)))
    df.loc[[2, 9, 33], "v"] = np.nan
    out = _sorted(RC.history_values(df, "unit", "v", (8, 50)), ["unit", "race_id"])
    d = _sorted(df, ["unit", "race_id"])
    for h in (8, 50):
        m, c = indep_history(d, "v", "unit", h)
        np.testing.assert_allclose(out[f"v_h{h}"].to_numpy(), m, rtol=0, atol=1e-12)
        assert (out["v_n"].to_numpy() == c).all()


@pytest.fixture(scope="module")
def raw():
    """Seasons 2020-2022 of the real Jolpica laps / pits and the entries (small enough to rebuild several times)."""
    laps, pits = pd.read_parquet(LAPS), pd.read_parquet(PITS)
    laps = laps[(laps["source"] == "jolpica") & laps["season"].between(2020, 2022)].reset_index(drop=True)
    pits = pits[(pits["source"] == "jolpica") & pits["season"].between(2020, 2022)].reset_index(drop=True)
    ent = pd.read_parquet(ENT)
    ent = ent[ent["season"].between(2020, 2022)].reset_index(drop=True)
    return laps, pits, ent


def _values(laps, pits, ent) -> dict:
    return RUN.race_values(laps, pits, ent, CFG)


@pytest.fixture(scope="module")
def base_vals(raw):
    return _values(*raw)


def _rid(ent: pd.DataFrame, season: int, rnd: int) -> int:
    return int(ent[(ent["season"] == season) & (ent["round"] == rnd)]["race_id"].iloc[0])


def _race_mask(df: pd.DataFrame, ent: pd.DataFrame, pred) -> pd.Series:
    ids = ent[["season", "round", "race_id"]].drop_duplicates()
    ids = ids[pred(ids["race_id"])][["season", "round"]]
    hit = df.merge(ids.assign(_hit=True), on=["season", "round"], how="left")["_hit"]
    return (hit == True).to_numpy()  # noqa: E712


def _alter(raw, pred, seed: int = 0):
    """Perturb laps, pits and entries of every race with race_id satisfying pred (positions, times, durations,
    grid, pit-lane starts)."""
    laps, pits, ent = (x.copy() for x in raw)
    rng = np.random.default_rng(seed)
    ml, mp, me = _race_mask(laps, ent, pred), _race_mask(pits, ent, pred), pred(ent["race_id"]).to_numpy()
    laps.loc[ml, "lap_time_s"] = laps.loc[ml, "lap_time_s"] * rng.uniform(0.9, 1.3, ml.sum())
    laps.loc[ml, "position"] = rng.permutation(laps.loc[ml, "position"].to_numpy())
    pits.loc[mp, "duration_s"] = rng.uniform(15, 40, mp.sum())
    ent.loc[me, "grid"] = rng.permutation(ent.loc[me, "grid"].to_numpy())
    ent.loc[me, "pit_start"] = (rng.random(me.sum()) < 0.1).astype(ent["pit_start"].dtype)
    return laps, pits, ent


def _drop(raw, pred):
    laps, pits, ent = raw
    ml, mp, me = _race_mask(laps, ent, pred), _race_mask(pits, ent, pred), pred(ent["race_id"]).to_numpy()
    return laps[~ml].reset_index(drop=True), pits[~mp].reset_index(drop=True), ent[~me].reset_index(drop=True)


VALCOLS = {"driver": ["cons", "deg", "cons_c", "deg_c", "gain", "lap1_c", "p1"],
           "team": ["n_stops", "crew_loss", "crew_slow", "crew_loss_c", "crew_slow_c"]}
UNIT = {"driver": ["driverId"], "team": ["lineage"]}


def _cmp_values(a: dict, b: dict, level: str, keep) -> None:
    cols = ["season", "round", "race_id"] + UNIT[level] + VALCOLS[level]
    x = _sorted(a[level][keep(a[level]["race_id"])][cols], ["race_id"] + UNIT[level])
    y = _sorted(b[level][keep(b[level]["race_id"])][cols], ["race_id"] + UNIT[level])
    pd.testing.assert_frame_equal(x, y, check_exact=True)


def _cmp_hist(a: dict, b: dict, feat: str, keep) -> None:
    """History rows exist only for units whose OWN race value is valid, so row presence of the altered race itself
    may differ (that is selection on the race's own outcome, not a leak into the value). Rows before the cut must
    be identical in presence and value; rows of the cut race are compared on the units present in both."""
    unit = FEATURES[feat][1]
    x, y = _hist(a, feat), _hist(b, feat)
    x = _sorted(x[keep(x["race_id"])], ["race_id", unit])
    y = _sorted(y[keep(y["race_id"])], ["race_id", unit])
    assert len(x) > 0
    last = x["race_id"].max()
    pd.testing.assert_frame_equal(x[x["race_id"] < last].reset_index(drop=True),
                                  y[y["race_id"] < last].reset_index(drop=True), check_exact=True)
    cx, cy = x[x["race_id"] == last], y[y["race_id"] == last]
    j = cx.merge(cy, on=["race_id", unit], suffixes=("_a", "_b"))
    assert len(j) >= 0.5 * len(cx)
    for c in [c for c in cx.columns if c.startswith("yv_")]:
        assert (j[f"{c}_a"].to_numpy() == j[f"{c}_b"].to_numpy()).all(), c


@needs_real
@pytest.mark.parametrize("feat", list(FEATURES))
@pytest.mark.parametrize("scenario", ["altered", "removed_later", "removed_own_and_later"])
def test_history_feature_of_race_r_unchanged_by_r_and_later(raw, base_vals, feat, scenario):
    """Bitwise: blanking / changing / removing race r and later (raw laps, stops, grid) leaves race-r history
    features, and every earlier race value, unchanged. r = 2022 round 10 (history from 2020-2021 and 2022 R1-9)."""
    r = _rid(raw[2], 2022, 10)
    pred = lambda s: s >= r
    mod = _alter(raw, pred) if scenario == "altered" else _drop(raw, pred if scenario == "removed_own_and_later"
                                                                  else (lambda s: s > r))
    alt = _values(*mod)
    level = FEATURES[feat][2]
    _cmp_values(base_vals, alt, level, lambda s: s < r)
    # the race r history row uses only earlier races; for removal of r itself the row cannot exist, compare < r
    upto = (lambda s: s < r) if scenario == "removed_own_and_later" else (lambda s: s <= r)
    _cmp_hist(base_vals, alt, feat, upto)


@needs_real
@pytest.mark.parametrize("feat", list(FEATURES))
def test_control_altering_an_earlier_race_moves_later_history_only(raw, base_vals, feat):
    r0 = _rid(raw[2], 2021, 5)
    alt = _values(*_alter(raw, lambda s: s == r0, seed=5))
    level = FEATURES[feat][2]
    # other races' own values do not change
    _cmp_values(base_vals, alt, level, lambda s: s < r0)
    _cmp_values(base_vals, alt, level, lambda s: s > r0)
    unit = FEATURES[feat][1]
    x, y = _hist(base_vals, feat), _hist(alt, feat)
    x, y = _sorted(x, ["race_id", unit]), _sorted(y, ["race_id", unit])
    assert len(x) == len(y)
    before = x["race_id"] <= r0
    pd.testing.assert_frame_equal(x[before].reset_index(drop=True), y[before].reset_index(drop=True), check_exact=True)
    moved = (x.loc[~before, "yv_h50"].to_numpy() != y.loc[~before, "yv_h50"].to_numpy())
    assert moved.sum() > 0, "control has no power: altering race r0 moved no later history value"


@needs_real
def test_race_values_row_order_independent(raw, base_vals):
    laps, pits, ent = (x.sample(frac=1, random_state=11).reset_index(drop=True) for x in raw)
    sh = _values(laps, pits, ent)
    for level in ("driver", "team"):
        cols = ["season", "round", "race_id"] + UNIT[level] + VALCOLS[level]
        a = _sorted(base_vals[level][cols], ["race_id"] + UNIT[level])
        b = _sorted(sh[level][cols], ["race_id"] + UNIT[level])
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)


# ====================================================================================== (2) hand tables
def _marked(rows: list[tuple]) -> pd.DataFrame:
    """rows: (driver, lap, stint_age, lap_time_s); every row clean, one race."""
    d = pd.DataFrame(rows, columns=["driverId", "lap", "stint_age", "lap_time_s"])
    d["season"], d["round"], d["clean"] = 2022, 1, True
    return d


def _stint(drv, first_lap, times):
    return [(drv, first_lap + i, i + 1, t) for i, t in enumerate(times)]


def test_cons_and_deg_hand_table():
    # A: two linear stints, slopes .1 and .3 (equal Sxx=10) -> pooled deg .2, residuals 0 -> cons 0; plus a 4-lap
    #    stint at exactly 100.0 (below the 5-lap minimum) that must be ignored; driver median is exactly 100.0.
    # B: residual pattern .1*[1,-2,0,2,-1] orthogonal to lap and constant -> slope 0, median |res| .1 -> cons .14826
    # C: 9 qualifying laps (5 + a 4-lap stint) -> below the 10-lap minimum, dropped.
    a = (_stint("A", 2, [99.8, 99.9, 100.0, 100.1, 100.2]) + _stint("A", 12, [99.4, 99.7, 100.0, 100.3, 100.6])
         + _stint("A", 30, [100.0] * 4))
    res = [100.1, 99.8, 100.0, 100.2, 99.9]
    b = _stint("B", 2, res) + _stint("B", 12, res)
    c = _stint("C", 2, res) + _stint("C", 12, [100.0] * 4)
    out = RC.driver_lap_values(_marked(a + b + c)).set_index("driverId")
    assert set(out.index) == {"A", "B"}
    assert out.loc["A", "n_stint_laps"] == 10 and out.loc["B", "n_stint_laps"] == 10
    assert out.loc["A", "deg"] == pytest.approx(0.2, abs=1e-12)
    assert out.loc["A", "cons"] == pytest.approx(0.0, abs=1e-10)
    assert out.loc["B", "deg"] == pytest.approx(0.0, abs=1e-12)
    assert out.loc["B", "cons"] == pytest.approx(1.4826 * 0.1, abs=1e-12)
    # field relative: median over the two eligible drivers (= their mean)
    assert out.loc["A", "cons_c"] == pytest.approx(-1.4826 * 0.05, abs=1e-12)
    assert out.loc["B", "cons_c"] == pytest.approx(1.4826 * 0.05, abs=1e-12)
    assert out.loc["A", "deg_c"] == pytest.approx(0.1, abs=1e-12)
    assert out.loc["B", "deg_c"] == pytest.approx(-0.1, abs=1e-12)


def test_deg_uses_within_stint_slope_not_pooled_level_shift():
    # stint 1 flat at 99, stint 2 flat at 101 (median exactly 100): a pooled slope across stints would be > 0
    rows = _stint("A", 2, [99.0] * 5) + _stint("A", 12, [101.0] * 5) + _stint("B", 2, [100.0] * 10)
    out = RC.driver_lap_values(_marked(rows)).set_index("driverId")
    assert out.loc["A", "deg"] == pytest.approx(0.0, abs=1e-12)
    # B is a single 10-lap stint with stint_age 1..10 (flat) -> deg 0, cons 0
    assert out.loc["B", "deg"] == pytest.approx(0.0, abs=1e-12)


def test_cons_deg_net_of_fuel_stint_id_from_lap_minus_stint_age():
    # an out-lap-removed gap inside a stint keeps the same stint id (lap - stint_age); two laps ids -> 2 stints
    rows = [("A", 2, 1, 100.0), ("A", 3, 2, 100.1), ("A", 5, 4, 100.3), ("A", 6, 5, 100.4), ("A", 7, 6, 100.5),
            ("A", 12, 1, 100.0), ("A", 13, 2, 100.1), ("A", 14, 3, 100.2), ("A", 15, 4, 100.3), ("A", 16, 5, 100.4)]
    out = RC.driver_lap_values(_marked(rows)).set_index("driverId")
    assert out.loc["A", "n_stint_laps"] == 10     # lap 4 missing (not clean) but laps 2,3,5,6,7 are one 5-lap stint
    assert out.loc["A", "deg"] == pytest.approx(0.1 / 100.25 * 100, rel=0.05)   # slope ~0.1 s/lap in % of median
    assert out.loc["A", "cons"] == pytest.approx(0.0, abs=1e-9)


def test_cons_deg_other_races_do_not_enter():
    rows = _stint("A", 2, [99.8, 99.9, 100.0, 100.1, 100.2]) * 2
    one = _marked(_stint("A", 2, [99.8, 99.9, 100.0, 100.1, 100.2]) + _stint("A", 12, [99.8, 99.9, 100.0, 100.1, 100.2]))
    two = pd.concat([one, one.assign(round=2, lap_time_s=one["lap_time_s"] * 1.5)])
    a = RC.driver_lap_values(one)
    b = RC.driver_lap_values(two)
    pd.testing.assert_frame_equal(a, b[b["round"] == 1].reset_index(drop=True)[a.columns], check_exact=True)


def _entries(spec: dict[str, str]) -> pd.DataFrame:
    d = pd.DataFrame(dict(driverId=list(spec), lineage=list(spec.values())))
    return d.assign(season=2022, round=1)


def _pits(rows: list[tuple]) -> pd.DataFrame:
    d = pd.DataFrame(rows, columns=["driverId", "lap", "duration_s"])
    return d.assign(season=2022, round=1)


CREW_ENT = _entries({"a": "T1", "b": "T1", "c": "T2", "d": "T3", "e": "T3", "f": "T4"})
CREW_STOPS = _pits([("a", 10, 22.0), ("a", 30, 23.0), ("b", 12, 22.5), ("c", 11, 21.0), ("c", 33, 26.5),
                    ("d", 14, 24.0), ("d", 35, 21.5), ("e", 15, 22.0), ("f", 20, 22.25),
                    ("a", 40, 90.0),            # > 60 s: absolute cap
                    ("b", 41, 45.0),            # <= 60 but > median(10 stops incl. it) + 20: repair stop
                    ("c", 42, np.nan),          # no duration
                    ("d", 43, 0.0)])            # non-positive


def test_valid_stops_hand_table():
    kept = RC.valid_stops(CREW_STOPS)
    assert sorted(kept["duration_s"]) == sorted([22.0, 23.0, 22.5, 21.0, 26.5, 24.0, 21.5, 22.0, 22.25])
    # kept median 22.25: loss = duration - 22.25 (NOT minus the median that still contains the 45 s stop, 22.375)
    got = dict(zip(zip(kept["driverId"], kept["lap"]), kept["loss_s"]))
    assert got[("a", 10)] == pytest.approx(-0.25) and got[("c", 33)] == pytest.approx(4.25)
    assert got[("f", 20)] == pytest.approx(0.0)


def test_crew_values_hand_table():
    out = RC.crew_values(CREW_STOPS, CREW_ENT).set_index("lineage")
    assert set(out.index) == {"T1", "T2", "T3"}          # T4 has 1 kept stop: no value
    assert out["n_stops"].to_dict() == {"T1": 3, "T2": 2, "T3": 3}
    # losses: T1 -.25,.75,.25 ; T2 -1.25,4.25 ; T3 1.75,-.75,-.25
    assert out["crew_loss"].to_dict() == pytest.approx({"T1": 0.25, "T2": 1.5, "T3": -0.25}, abs=1e-12)
    assert out["crew_slow"].to_dict() == pytest.approx({"T1": 0.0, "T2": 0.5, "T3": 0.0}, abs=1e-12)
    # field-relative: median over teams .25 ; mean slow 1/6
    assert out["crew_loss_c"].to_dict() == pytest.approx({"T1": 0.0, "T2": 1.25, "T3": -0.5}, abs=1e-12)
    assert out["crew_slow_c"].to_dict() == pytest.approx({"T1": -1 / 6, "T2": 1 / 3, "T3": -1 / 6}, abs=1e-12)


def test_crew_slow_threshold_is_strict_and_min_stops_is_two():
    ent = _entries({"a": "T1", "b": "T2", "c": "T3"})
    # kept median 22.0; losses T1: 0, 3.0 ; T2: 3.0001, -1 ; T3: 0 only (one stop)
    p = _pits([("a", 1, 22.0), ("a", 2, 25.0), ("b", 1, 25.0001), ("b", 2, 21.0), ("c", 1, 22.0),
               ("c", 3, 22.0), ("b", 3, 22.0), ("a", 4, 21.0), ("c", 5, 22.0)])
    out = RC.crew_values(p, ent).set_index("lineage")
    kept_med = np.median([22.0, 25.0, 25.0001, 21.0, 22.0, 22.0, 22.0, 21.0, 22.0])
    assert kept_med == 22.0
    assert out.loc["T1", "crew_slow"] == 0.0                         # loss exactly 3.0 is not slow (> 3.0)
    assert out.loc["T2", "crew_slow"] == pytest.approx(1 / 3)       # 3.0001 is
    only_one = _pits([("a", 1, 22.0), ("a", 2, 22.0), ("b", 1, 22.0)])
    assert set(RC.crew_values(only_one, _entries({"a": "T1", "b": "T2"}))["lineage"]) == {"T1"}


def test_red_flag_cap_applied_before_median_rule():
    # red-flag race: five stops of 70-90 s and three normal ones. Without the cap the median is 72.5 and the
    # median + 20 rule keeps everything; with the cap only the 3 normal stops remain and the median is 23.0.
    p = _pits([("a", 1, 80.0), ("a", 2, 85.0), ("b", 1, 90.0), ("b", 2, 75.0), ("c", 1, 70.0),
               ("c", 2, 22.0), ("d", 1, 23.0), ("d", 2, 24.0)])
    kept = RC.valid_stops(p)
    assert sorted(kept["duration_s"]) == [22.0, 23.0, 24.0]
    assert kept["loss_s"].tolist() == pytest.approx([-1.0, 0.0, 1.0]) or sorted(kept["loss_s"]) == pytest.approx([-1.0, 0.0, 1.0])
    # the rule alone (cap disabled) would keep all eight
    assert len(RC.valid_stops(p, max_s=1e9)) == 8
    # median + 20 rule on the remainder: median of {22,22,23,23,x} is 23 for x >= 23, so the limit is 43.0 inclusive
    base = [("a", 1, 22.0), ("a", 2, 22.0), ("b", 1, 23.0), ("b", 2, 23.0)]
    assert 43.0 in RC.valid_stops(_pits(base + [("c", 1, 43.0)]))["duration_s"].tolist()
    assert 43.5 not in RC.valid_stops(_pits(base + [("c", 1, 43.5)]))["duration_s"].tolist()
    assert 60.0 not in RC.valid_stops(_pits(base + [("c", 1, 60.5)]))["duration_s"].tolist()


def test_crew_values_race_with_only_long_stops_is_empty_not_an_error():
    p = _pits([("a", 1, 100.0), ("a", 2, 200.0), ("b", 1, 90.0)])
    out = RC.crew_values(p, _entries({"a": "T1", "b": "T2"}))
    assert len(out) == 0


def test_crew_values_use_only_their_own_race():
    two = pd.concat([CREW_STOPS, CREW_STOPS.assign(round=2, duration_s=CREW_STOPS["duration_s"] * 1.3)])
    ent2 = pd.concat([CREW_ENT, CREW_ENT.assign(round=2)])
    a = RC.crew_values(CREW_STOPS, CREW_ENT)
    b = RC.crew_values(two, ent2)
    pd.testing.assert_frame_equal(a, b[b["round"] == 1].reset_index(drop=True)[a.columns], check_exact=True)


def _l1_entries(grid, pit_start=None, season=2022, rnd=1):
    n = len(grid)
    return pd.DataFrame(dict(season=season, round=rnd, driverId=[f"d{i}" for i in range(n)], grid=grid,
                             pit_start=pit_start if pit_start is not None else [0] * n))


def _l1_pos(p1, season=2022, rnd=1):
    return pd.DataFrame(dict(season=season, round=rnd, driverId=[f"d{i}" for i in range(len(p1))], p1=p1))


def test_lap1_hand_table_slot_adjusted():
    grid = np.arange(1, 11)
    p1 = [1, 2, 4, 3, 5, 6, 7, 8, 9, 10]            # d2 loses a place, d3 gains one
    out = RC.lap1_values(_l1_pos(p1), _l1_entries(grid.tolist()))
    gain = grid - np.array(p1)
    assert out["gain"].tolist() == gain.tolist()
    b = ((grid - 5.5) * (gain - gain.mean())).sum() / ((grid - 5.5) ** 2).sum()    # 0.1 / 8.25
    assert b == pytest.approx(0.1 / 8.25)
    np.testing.assert_allclose(out["lap1_c"].to_numpy(), gain - gain.mean() - b * (grid - 5.5), atol=1e-12)
    assert out["lap1_c"].sum() == pytest.approx(0.0, abs=1e-12)
    assert (out["lap1_c"] * (out["grid"] - out["grid"].mean())).sum() == pytest.approx(0.0, abs=1e-12)


def test_lap1_constant_gain_slot_adjusted_is_zero():
    grid = list(range(1, 11))
    out = RC.lap1_values(_l1_pos([g - 2 if g > 2 else g for g in grid]), _l1_entries(grid))
    # gain 0 for grid 1,2 and 2 for the rest is NOT linear; use an exactly linear case instead
    out = RC.lap1_values(_l1_pos([5] * 10), _l1_entries(grid))     # gain = grid - 5, linear in grid
    np.testing.assert_allclose(out["lap1_c"].to_numpy(), 0.0, atol=1e-12)


def test_lap1_exclusions():
    grid = list(range(1, 11)) + [0]
    ent = _l1_entries(grid, pit_start=[0] * 10 + [1])               # d10 starts from the pit lane
    p1 = list(range(1, 11)) + [11]
    out = RC.lap1_values(_l1_pos(p1), ent)
    assert "d10" not in set(out["driverId"]) and len(out) == 10
    # the pit-lane starter does not influence the others
    ref = RC.lap1_values(_l1_pos(list(range(1, 11))), _l1_entries(list(range(1, 11))))
    pd.testing.assert_frame_equal(out.reset_index(drop=True), ref, check_exact=True)
    # driver with no lap-1 record (retired on lap 1) is simply absent
    out2 = RC.lap1_values(_l1_pos(p1[:9]), _l1_entries(list(range(1, 11))))
    assert len(out2) == 9 and "d9" not in set(out2["driverId"])
    # fewer than 8 drivers -> no values; one distinct grid slot -> no values
    assert len(RC.lap1_values(_l1_pos([1, 2, 3, 4, 5, 6, 7]), _l1_entries(list(range(1, 8))))) == 0
    assert len(RC.lap1_values(_l1_pos(list(range(1, 11))), _l1_entries([3] * 10))) == 0
    assert len(RC.lap1_values(_l1_pos([1, 2, 3, 4, 5, 6, 7, 8]), _l1_entries(list(range(1, 9))))) == 8   # 8 is enough


def test_lap1_positions_rule_hand_table():
    rows = []
    for (rnd, pos, tm) in [(1, [3, 1, 2], [91.0, 88.0, 89.5]),                 # stored positions used (differ from time rank)
                           (2, [np.nan] * 3, [90.0, 88.0, 89.5]),             # no stored position: time rank
                           (3, [1, np.nan, 3], [90.0, 88.0, np.nan]),         # partly missing: time rank, NaN time stays NaN
                           (4, [1, 2, 3], [90.0, 88.0, 89.5])]:
        for i in range(3):
            rows.append(dict(season=2023, round=rnd, driverId=f"d{i}", lap=1, position=pos[i], lap_time_s=tm[i]))
    rows.append(dict(season=2023, round=1, driverId="d0", lap=2, position=9, lap_time_s=50.0))   # ignored: lap 2
    out = RC.lap1_positions(pd.DataFrame(rows)).set_index(["round", "driverId"])
    assert out.loc[1, "p1"].tolist() == [3, 1, 2] and set(out.loc[1, "p1_source"]) == {"stored"}
    assert out.loc[2, "p1"].tolist() == [3, 1, 2] and set(out.loc[2, "p1_source"]) == {"time_rank"}
    assert out.loc[3, "p1"].iloc[[0, 1]].tolist() == [2, 1] and np.isnan(out.loc[3, "p1"].iloc[2])
    assert out.loc[4, "p1"].tolist() == [1, 2, 3] and out.loc[4, "rank_time"].tolist() == [3, 1, 2]


def test_teammate_relative_hand_table():
    df = pd.DataFrame(dict(season=2022, round=1, driverId=["a", "b", "c", "d", "e"], v=[1.0, 3.0, 5.0, 2.0, 8.0]))
    team = pd.DataFrame(dict(season=2022, round=1, driverId=["a", "b", "c", "d", "e"], lineage=["X", "X", "Y", "Z", "Z"]))
    out = RC.teammate_relative(df, "v", team)
    assert out.iloc[[0, 1, 3, 4]].tolist() == [-1.0, 1.0, -3.0, 3.0]
    assert np.isnan(out.iloc[2])                    # alone in its team
    # a teammate with no eligible value (row absent) leaves the driver alone -> NaN
    out2 = RC.teammate_relative(df.iloc[[0, 2, 3, 4]], "v", team)
    assert np.isnan(out2.iloc[0])


# ====================================================================================== (3) stored numbers
README_NUMBERS = {   # feature: (r, lo, hi, n_obs, n_races) as printed in the card / task
    "consistency": (0.256, 0.153, 0.355, 1187, 65),
    "tyre_degradation": (0.121, 0.058, 0.177, 1187, 65),
    "lap1_gain": (0.163, 0.107, 0.220, 1222, 66),
    "crew_time_loss": (0.186, 0.102, 0.282, 603, 65),
    "crew_slow_stop_rate": (0.119, 0.036, 0.204, 603, 65),
}


def _dev(table, feat, **kw):
    q = table[(table["feature"] == feat) & (table["variant"] == kw.get("variant", "declared"))
              & (table["halflife"] == kw.get("h", 50)) & (table["scope"] == kw.get("scope", "dev"))]
    assert len(q) == 1
    return q.iloc[0]


@needs_stored
@pytest.mark.parametrize("feat", list(FEATURES))
def test_stored_declared_numbers_match_card(table, summary, feat):
    r, lo, hi, n, k = README_NUMBERS[feat]
    row = _dev(table, feat)
    assert round(row["r"], 3) == r and round(row["lo"], 3) == lo and round(row["hi"], 3) == hi
    assert (int(row["n_obs"]), int(row["n_races"])) == (n, k)
    d = summary["decision"][feat]
    for k in ("r", "lo", "hi"):      # json and csv round trips differ in the last digit only
        assert d["persistence" if k == "r" else k] == pytest.approx(row[k], abs=1e-12)


@needs_stored
@pytest.mark.parametrize("feat", list(FEATURES))
def test_declared_dev_persistence_reproduced_independently(stored, table, feat):
    """Independent history (closed form), np.corrcoef and a re-implemented race bootstrap (10,000, seed 0)."""
    col, unit, level, _ = FEATURES[feat]
    res = indep_persistence(stored[level].dropna(subset=[col]), col, unit, 50, seasons=[2021, 2022, 2023])
    row = _dev(table, feat)
    assert res["n_obs"] == row["n_obs"] and res["n_races"] == row["n_races"]
    assert res["r"] == pytest.approx(row["r"], abs=1e-9)
    assert res["lo"] == pytest.approx(row["lo"], abs=1e-9) and res["hi"] == pytest.approx(row["hi"], abs=1e-9)


@needs_stored
@pytest.mark.parametrize("feat", list(FEATURES))
def test_by_season_burn_in_and_halflife_rows_reproduced(stored, table, feat):
    col, unit, level, _ = FEATURES[feat]
    tab = stored[level].dropna(subset=[col])
    scopes = {"dev": [2021, 2022, 2023], "burn_in_2014_2020": list(range(2014, 2021)),
              **{str(s): [s] for s in range(2014, 2024)}}
    for h in (8, 16, 32, 50):
        m, c = indep_history(tab.reset_index(drop=True), col, unit, h)
        t = tab.reset_index(drop=True)
        for scope, seasons in scopes.items():
            if scope.isdigit() and int(scope) < 2021 and h != 50:
                assert table[(table.feature == feat) & (table.variant == "declared") & (table.halflife == h)
                             & (table.scope == scope)].empty
                continue
            keep = (c >= 5) & t["season"].isin(seasons).to_numpy()
            r = float(np.corrcoef(m[keep], t[col].to_numpy(float)[keep])[0, 1])
            row = _dev(table, feat, h=h, scope=scope)
            assert row["r"] == pytest.approx(r, abs=1e-9), (feat, h, scope)
            assert row["n_obs"] == keep.sum() and row["n_races"] == t.loc[keep, "race_id"].nunique()
            assert row["block"] == ("burn-in (not a decision block)" if scope.startswith("burn") or
                                    (scope.isdigit() and int(scope) < 2021) else "dev")


@needs_stored
def test_burn_in_and_dev_blocks_are_separate_in_csv(table):
    assert set(table["scope"]) == {"dev", "burn_in_2014_2020"} | {str(s) for s in range(2014, 2024)}
    assert set(table["halflife"]) == {8, 16, 32, 50}
    # per (feature, variant): 4 half-lives x (dev + burn-in + 3 dev seasons) = 20, plus 7 burn-in seasons at hl 50
    assert table.groupby(["feature", "variant"]).size().eq(27).all()
    assert len(table) == 13 * 27        # 3 variants for the 3 driver features, 2 for the 2 crew features
    assert table["r"].notna().all()


@needs_stored
@pytest.mark.parametrize("feat", ["consistency", "tyre_degradation", "lap1_gain"])
def test_teammate_relative_variant_reproduced(stored, table, feat):
    col, unit, level, tm = FEATURES[feat]
    assert tm
    tab = stored[level].dropna(subset=[col]).reset_index(drop=True)
    g = tab.groupby(["season", "round", "lineage"])[col]
    tab["y"] = np.where(g.transform("count") >= 2, tab[col] - g.transform("mean"), np.nan)
    res = indep_persistence(tab, "y", unit, 50, seasons=[2021, 2022, 2023])
    row = _dev(table, feat, variant="teammate_relative")
    assert res["r"] == pytest.approx(row["r"], abs=1e-9) and res["n_obs"] == row["n_obs"]
    assert not table[(table.feature == "crew_time_loss") & (table.variant == "teammate_relative")].shape[0]
    assert round(_dev(table, "consistency", variant="teammate_relative")["r"], 3) == 0.091
    assert round(_dev(table, "lap1_gain", variant="teammate_relative")["r"], 3) == 0.046


@needs_stored
@pytest.mark.parametrize("feat", list(FEATURES))
def test_winsorised_variant_reproduced(stored, table, feat):
    col, unit, level, _ = FEATURES[feat]
    tab = stored[level].dropna(subset=[col]).reset_index(drop=True)
    lo, hi = np.quantile(tab[col], [0.01, 0.99])
    tab["y"] = tab[col].clip(lo, hi)
    res = indep_persistence(tab, "y", unit, 50, seasons=[2021, 2022, 2023])
    row = _dev(table, feat, variant=WINS)
    assert res["r"] == pytest.approx(row["r"], abs=1e-9) and res["n_obs"] == row["n_obs"]


@needs_stored
def test_sensitivity_claims_in_card(table):
    """Card: winsorised crew loss .195, hl32 crew loss .185, burn-in crew loss .312; no sensitivity version of a
    failing feature reaches 0.2 on the declared block (hl 50, dev) EXCEPT as noted: this test reports it."""
    assert round(_dev(table, "crew_time_loss", variant=WINS)["r"], 3) == 0.195
    assert round(_dev(table, "crew_time_loss", h=32)["r"], 3) == 0.185
    assert round(_dev(table, "crew_time_loss", scope="burn_in_2014_2020")["r"], 3) == 0.312
    sens = table[(table["variant"] != "declared") & (table["halflife"] == 50) & (table["scope"] == "dev")]
    declared = table[(table["variant"] == "declared") & (table["halflife"] == 50) & (table["scope"] == "dev")].set_index("feature")
    for _, r in sens.iterrows():
        if declared.loc[r["feature"], "r"] < 0.2:
            assert r["r"] < 0.2, f"sensitivity row would rescue {r['feature']} ({r['variant']}): {r['r']}"


@needs_stored
def test_superseded_run_numbers_in_card(table):
    """Card: first run crew time-loss persistence -0.010 (declared) and 0.178 (winsorised), from red-flag
    values of hundreds of seconds. Recomputed from the superseded run's stored crew values."""
    first = pd.read_parquet(FIRST_DIR / "crew_values.parquet")
    assert first["crew_loss"].abs().max() > 100          # the artefact is really in the superseded file
    d = indep_persistence(first, "crew_loss_c", "lineage", 50, seasons=[2021, 2022, 2023])
    assert round(d["r"], 3) == -0.010
    t = first.dropna(subset=["crew_loss_c"]).reset_index(drop=True)
    lo, hi = np.quantile(t["crew_loss_c"], [0.01, 0.99])
    t["y"] = t["crew_loss_c"].clip(lo, hi)
    assert round(indep_persistence(t, "y", "lineage", 50, seasons=[2021, 2022, 2023])["r"], 3) == 0.178
    final = pd.read_parquet(RUN_DIR / "crew_values.parquet")
    assert final["crew_loss"].abs().max() < RC.MAX_STOP_S


@needs_stored
def test_driver_values_identical_between_first_and_final_run():
    a = pd.read_parquet(FIRST_DIR / "predictions.parquet")
    b = pd.read_parquet(RUN_DIR / "predictions.parquet")
    pd.testing.assert_frame_equal(a, b, check_exact=True)


# ====================================================================================== (4) decisions, no model, split
@needs_stored
def test_decisions_follow_mechanically_from_threshold(table, summary):
    thr = CFG["persistence_min"]
    assert thr == 0.2 and CFG["halflife_primary"] == 50
    for feat, d in summary["decision"].items():
        row = _dev(table, feat)
        assert d["persistent"] == bool(row["r"] >= thr)
        assert d["ci_straddles_threshold"] == bool(row["lo"] < thr <= row["hi"])
    assert {f for f, d in summary["decision"].items() if d["persistent"]} == {"consistency"}
    assert set(summary["decision"]) == set(FEATURES)


def test_decision_rule_in_run_uses_point_estimate_only():
    src = (METH / "run.py").read_text(encoding="utf-8")
    assert re.search(r'persistent=bool\(r\["r"\] >= cfg\["persistence_min"\]\)', src)
    assert 'declared' in src and 'halflife"] == hp' in src


@needs_stored
def test_readme_table_matches_summary(summary):
    text = (METH / "README.md").read_text(encoding="utf-8")
    names = {"Driver consistency": "consistency", "Driver tyre degradation": "tyre_degradation",
             "Driver lap-1 gain": "lap1_gain", "Pit-crew time loss": "crew_time_loss",
             "Pit-crew slow-stop rate": "crew_slow_stop_rate"}
    found = 0
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] in names and len(cells) == 7:
            feat = names[cells[0]]
            d = summary["decision"][feat]
            assert float(cells[2]) == round(d["persistence"], 3)
            assert cells[3] == f"[{d['lo']:.3f}, {d['hi']:.3f}]"
            assert int(cells[4]) == d["n_obs"] and int(cells[5]) == d["n_races"]
            assert cells[6].startswith("persistent" if d["persistent"] else "not persistent"), line
            assert ("straddles" in cells[6]) == d["ci_straddles_threshold"], line
            found += 1
    assert found == 5
    assert FINAL in text and "0242" in text and "superseded" in text


@needs_stored
def test_run_folders_complete_and_summary_points_to_final(summary):
    assert summary["run_id"] == FINAL
    for d in (RUN_DIR, FIRST_DIR):
        for f in ("config.yaml", "metrics.json", "per_race_metrics.csv", "predictions.parquet", "crew_values.parquet"):
            assert (d / f).exists(), (d.name, f)
    cfg = load_yaml(RUN_DIR / "config.yaml")
    assert cfg["run_id"] == FINAL and cfg.get("git_commit")
    for k, v in CFG.items():
        assert cfg[k] == v, k
    pd.testing.assert_frame_equal(pd.read_csv(RUN_DIR / "per_race_metrics.csv"), pd.read_csv(CSV), check_exact=True)
    m = json.loads((RUN_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert m["decision"] == summary["decision"] and m["checks"] == summary["checks"]


def test_no_model_and_no_leaderboard_rows_for_m20():
    for f in ("results/leaderboard.csv", "results/plan3/leaderboard.csv", "results/plan3/attribution.csv"):
        p = ROOT / f
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore").lower()
            assert "m20" not in txt and "racecraft" not in txt and "race-craft" not in txt, f
    for f in ("src/pitlane/features/racecraft.py", "methodologies/M20_racecraft/run.py"):
        src = (ROOT / f).read_text(encoding="utf-8")
        for bad in ("lightgbm", "sklearn", "pitlane.models", "pitlane.weekend", "fit_", ".fit(", "leaderboard.csv"):
            assert bad not in src, (f, bad)
    for d in (RUN_DIR, FIRST_DIR):
        if d.exists():
            cols = set(pd.read_parquet(d / "predictions.parquet").columns)    # file name is required; content is race values
            assert not [c for c in cols if re.search(r"pred|prob|score|rho|rank_pred", c)], cols
            assert not list(d.glob("*.pkl")) and not list(d.glob("*model*"))
    assert CFG["phase"] == 1 and CFG["split"] == "dev"


def test_dev_only_filters_in_code_and_config():
    assert CFG["dev_seasons"] == [2021, 2022, 2023] and max(CFG["dev_seasons"]) == 2023
    assert CFG["burn_in_seasons"] == list(range(2014, 2021))
    for f in ("src/pitlane/features/racecraft.py", "methodologies/M20_racecraft/run.py", "methodologies/M20_racecraft/config.yaml"):
        txt = (ROOT / f).read_text(encoding="utf-8")
        assert not re.search(r"\b202[4-9]\b", txt), f
        assert "holdout" not in txt.lower() and "fresh" not in txt.lower(), f
    src = (METH / "run.py").read_text(encoding="utf-8")
    assert 'max(cfg["dev_seasons"]) == 2023' in src
    assert src.count('["season"] <= last') >= 3        # laps, pits, entries


@needs_real
def test_loaded_inputs_contain_no_season_after_2023():
    laps, pits, ent = RUN.load(CFG)
    assert laps["season"].max() == 2023 and pits["season"].max() == 2023 and ent["season"].max() == 2023
    assert set(laps["source"]) == {"jolpica"} and set(pits["source"]) == {"jolpica"}


@needs_stored
def test_stored_values_and_csv_are_dev_and_burn_in_only(stored, table):
    for t in stored.values():
        assert t["season"].between(2014, 2023).all()
    assert set(table["scope"]).isdisjoint({"2024", "2025", "2026"})


def test_fixed_constants_are_declared_not_fitted():
    """Hand-declared constants; the winsor bounds are the only data-dependent number and are labelled in-sample."""
    assert (RC.SLOW_S, RC.RED_FLAG_MARGIN_S, RC.MAX_STOP_S, RC.MIN_CREW_STOPS) == (3.0, 20.0, 60.0, 2)
    assert (RC.MIN_STINT_LAPS, RC.MIN_DRIVER_LAPS, RC.MIN_LAP1_DRIVERS) == (5, 10, 8)
    for k, v in dict(red_flag_margin_s=20.0, max_stop_s=60.0, slow_stop_s=3.0, min_crew_stops=2, min_stint_laps=5,
                     min_driver_laps=10, min_lap1_drivers=8, thresh=1.07).items():
        assert CFG[k] == v
    ing = ROOT / "configs/ingest3.yaml"
    if ing.exists():
        assert load_yaml(ing)["pits"]["red_flag_margin_s"] == CFG["red_flag_margin_s"]


# ====================================================================================== (5) lap 1 and data checks
@pytest.fixture(scope="module")
def all_raw():
    laps, pits = pd.read_parquet(LAPS), pd.read_parquet(PITS)
    laps = laps[(laps["source"] == "jolpica") & (laps["season"] <= 2023)].reset_index(drop=True)
    pits = pits[(pits["source"] == "jolpica") & (pits["season"] <= 2023)].reset_index(drop=True)
    ent = pd.read_parquet(ENT)
    return laps, pits, ent[ent["season"] <= 2023].reset_index(drop=True)


@needs_real
def test_lap1_time_rank_agrees_with_stored_positions_2021_22(all_raw):
    laps = all_raw[0]
    p = RC.lap1_positions(laps)
    chk = p[p["season"].between(2021, 2022)]
    assert set(chk["p1_source"]) == {"stored"}
    assert (chk["p1"] == chk["rank_time"]).all()
    # the rule that the 2023 values rely on, on every season with stored positions
    st = p[p["p1_source"] == "stored"]
    assert (st["p1"] == st["rank_time"]).mean() == 1.0
    assert set(p[p["season"] == 2023]["p1_source"]) == {"time_rank"}
    assert laps[(laps["season"] == 2023)]["position"].isna().all()
    assert laps[(laps["season"] < 2023) & (laps["lap"] == 1)]["position"].notna().all()


@needs_real
def test_lap1_2023_ranks_are_a_permutation_and_resemble_grid_order(all_raw):
    laps, _, ent = all_raw
    p = RC.lap1_positions(laps)
    p23 = p[p["season"] == 2023]
    for _, g in p23.groupby("round"):
        assert sorted(g["p1"]) == list(range(1, len(g) + 1))
    m = p.merge(ent[["season", "round", "driverId", "grid", "pit_start"]], on=["season", "round", "driverId"])
    m = m[m["pit_start"] == 0]
    sp = m.groupby(["season", "round"]).apply(lambda g: g["p1"].corr(g["grid"], method="spearman"), include_groups=False)
    assert abs(sp.loc[2023].mean() - sp.loc[2022].mean()) < 0.1
    assert sp.loc[2023].min() > 0.3


@needs_stored
def test_stored_summary_lap1_agreement_and_counts(summary):
    c = summary["checks"]
    assert c["lap1_stored_vs_time_rank_agreement_2021_22"] == {"2021": 1.0, "2022": 1.0}
    assert c["dev_races_total"] == 66


@needs_stored
def test_data_checks_stored_values(stored, summary):
    drv, crew = stored["driver"], stored["team"]
    ent = pd.read_parquet(ENT) if ENT.exists() else None
    # duplicates
    assert not drv.duplicated(["season", "round", "driverId"]).any()
    assert not crew.duplicated(["season", "round", "lineage"]).any()
    # missing values: value columns are NaN together or not at all
    assert drv["cons"].isna().equals(drv["deg"].isna()) and drv["cons"].isna().equals(drv["cons_c"].isna())
    assert drv["gain"].isna().equals(drv["lap1_c"].isna())
    assert crew[["n_stops", "crew_loss", "crew_slow", "crew_loss_c", "crew_slow_c"]].notna().all().all()
    # ranges
    assert (drv["cons"].dropna() >= 0).all() and drv["cons"].max() < 10
    assert drv["deg"].dropna().abs().max() < 2.0              # % of lap per lap
    assert drv["n_stint_laps"].dropna().ge(RC.MIN_DRIVER_LAPS).all()
    assert crew["n_stops"].ge(RC.MIN_CREW_STOPS).all()
    assert crew["crew_slow"].between(0, 1).all()
    assert crew["crew_loss"].abs().max() < RC.MAX_STOP_S
    assert crew["crew_loss"].abs().max() < RC.MAX_STOP_S - 22 + RC.RED_FLAG_MARGIN_S + 1    # cannot exceed the cap rules
    # field-relative values: lap-1 residuals sum to zero per race, cons/deg/crew medians zero per race
    l1 = drv.dropna(subset=["lap1_c"])
    assert l1.groupby(["season", "round"])["lap1_c"].sum().abs().max() < 1e-9
    assert drv.dropna(subset=["cons_c"]).groupby(["season", "round"])["cons_c"].median().abs().max() < 1e-12
    assert drv.dropna(subset=["deg_c"]).groupby(["season", "round"])["deg_c"].median().abs().max() < 1e-12
    assert crew.groupby(["season", "round"])["crew_loss_c"].median().abs().max() < 1e-12
    assert crew.groupby(["season", "round"])["crew_slow_c"].mean().abs().max() < 1e-12
    # rows per race <= field size; rows per season against race counts
    if ent is not None:
        ent = ent[ent["season"] <= 2023]
        nrace = ent.groupby("season")["race_id"].nunique()
        field = ent.groupby(["season", "round"]).size()
        per_race = drv.groupby(["season", "round"]).size()
        # two lap-table drivers are missing from entries2 (2019 R18 norris, 2023 R19 alonso): documented, not hidden
        over = per_race[per_race > field.reindex(per_race.index)]
        assert set(over.index) == {(2019, 18), (2023, 19)}
        assert drv.groupby("season")["round"].nunique().le(nrace).all()
        assert crew.groupby("season")["round"].nunique().le(nrace).all()
        assert crew.groupby(["season", "round"]).size().le(11).all()      # at most 11 teams
        assert drv["race_id"].notna().all() and drv["race_id"].between(0, ent["race_id"].max()).all()
        # races without values (summary says one dev race lacks cons/deg and crew values)
        assert summary["checks"]["dev_races_with_values"]["consistency"] == 65
        have = set(zip(drv.dropna(subset=["cons"])["season"], drv.dropna(subset=["cons"])["round"]))
        miss = [k for k in set(zip(ent["season"], ent["round"])) if k[0] >= 2021 and k not in have]
        assert miss == [(2021, 12)]                       # the Belgian 2021 race (one lap, no clean laps)
        assert (2021, 12) not in set(zip(crew["season"], crew["round"]))
        # rows without lineage are drivers that have laps but no entry row: they cannot enter crew or lap-1 values
        nol = drv[drv["lineage"].isna()]
        assert set(nol["driverId"]) == {"norris", "alonso"} and nol["gain"].isna().all()


@needs_real
def test_stop_counts_in_summary_reproduce(all_raw, summary):
    pits = all_raw[1]
    kept = RC.valid_stops(pits, CFG["red_flag_margin_s"], CFG["max_stop_s"])
    n_all = int(pits["duration_s"].notna().sum())
    assert n_all == summary["checks"]["stops_with_duration"] == 7528
    assert n_all - len(kept) == summary["checks"]["stops_dropped_red_flag_or_repair"] == 497
    assert not pits.duplicated(["season", "round", "driverId", "stop"]).any()
    assert kept["duration_s"].le(RC.MAX_STOP_S).all() and kept["loss_s"].between(-15, RC.RED_FLAG_MARGIN_S + 2).all()
    # a stop faster than ~13 s would be a data error
    assert pits["duration_s"].dropna().min() > 10


@needs_real
@pytest.mark.slow
def test_stored_race_values_reproduce_from_raw(all_raw, stored):
    laps, pits, ent = all_raw
    v = RUN.race_values(laps, pits, ent, CFG)
    for level in ("driver", "team"):
        cols = ["season", "round", "race_id"] + UNIT[level] + VALCOLS[level]
        a = _sorted(v[level][cols], ["race_id"] + UNIT[level])
        b = _sorted(stored[level][cols], ["race_id"] + UNIT[level])
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10, atol=1e-12)
