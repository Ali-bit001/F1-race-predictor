"""R13 / M19 in-race forecast (v2: two-stage main model, scheduled distance, flag from stops <= k): independent checks.

Sections: hand tables (running order, features, pit stops); gradient of the stage-2 likelihood; leakage (state, features and
forecasts on real dev races); horizon lap and scheduled-distance table; data checks; reproduction with the switch off; stored result files
(v1 runs `m19_*` are superseded but still on the leaderboard; v2 runs are `m19v2_*`).

Tests named test_ACCEPTANCE_* are acceptance tests of the work package; a failure there is a finding, not a test bug.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
from io import StringIO

import numpy as np
import pandas as pd
import pytest

import pitlane.models.stage2 as S2
from pitlane.config import ROOT, load_yaml
from pitlane.evaluation.harness import split_races, walk_forward
from pitlane.features import inrace as I
from pitlane.models.registry import build_model
from pitlane.plan3 import attribution as A
from pitlane.plan3 import runutil as U

M19 = ROOT / "methodologies" / "M19_inrace"
RES = M19 / "results"
LAPS = ROOT / "data" / "interim" / "plan3" / "laps.parquet"
PITS = ROOT / "data" / "interim" / "plan3" / "pits.parquet"
SCHED = ROOT / "data" / "external" / "scheduled_laps.csv"
LB3 = ROOT / "results" / "plan3" / "leaderboard.csv"
needs_laps = pytest.mark.skipif(not (LAPS.exists() and PITS.exists() and SCHED.exists()), reason="lap/pit/scheduled tables missing (regenerable)")
needs_res = pytest.mark.skipif(not (RES / "verdicts.csv").exists(), reason="M19 results missing")
HZ = {"H4": None, "H5": 0.25, "H6": 0.5}
V1 = ("running_order", "latent", "gbm")
V2 = ("running_order", "latent_s2", "gbm_s2", "latent_joint", "gbm_joint")
CELL = {"running_order": "000", "latent": "111", "gbm": "100", "latent_s2": "111", "gbm_s2": "100", "latent_joint": "111", "gbm_joint": "100"}
VERDICT_METRICS = ["spearman", "mae", "exact", "exact_fin", "exact_1_3", "exact_4_10", "exact_11p", "spearman_fin", "mae_fin"]
COLS = ["ir_pos", "ir_gap", "ir_pits", "ir_out", "ir_def"]
SHORT_RACES = {149: ((2021, 12), 44, 1), 166: ((2022, 7), 77, 64), 176: ((2022, 17), 61, 59), 177: ((2022, 18), 53, 28)}
MONZA = 195      # 2023 R14: 53 scheduled, 51 run (aborted start)


# ====================================================================== hand tables
def _laps(rows):
    out = []
    for d, ts in rows:
        out += [dict(driverId=d, lap=i + 1, lap_time_s=float(t)) for i, t in enumerate(ts)]
    return pd.DataFrame(out)


def _pits(rows):
    """rows: (driverId, lap, duration_s[, stored_flag])."""
    return pd.DataFrame([dict(driverId=r[0], lap=r[1], duration_s=r[2], **({"red_flag_suspect": r[3]} if len(r) > 3 else {})) for r in rows],
                        columns=["driverId", "lap", "duration_s"] + (["red_flag_suspect"] if rows and len(rows[0]) > 3 else []))


def test_horizon_lap_hand_values():
    assert I.horizon_lap(56, None) == 1 and I.horizon_lap(1, None) == 1
    assert I.horizon_lap(56, 0.25) == 14 and I.horizon_lap(57, 0.25) == 15 and I.horizon_lap(57, 0.5) == 29
    assert I.horizon_lap(1, 0.5) == 1 and I.horizon_lap(1, 0.25) == 1
    assert I.horizon_lap(50, 1.0) == 50 and I.horizon_lap(50, 2.0) == 50
    assert I.horizon_lap(0, 0.5) == 1
    # available_laps: the horizon is capped at the last lap that exists, never below lap 1, and does nothing when the race ran long enough
    assert I.horizon_lap(44, 0.5, 1) == 1 and I.horizon_lap(44, 0.25, 1) == 1 and I.horizon_lap(44, None, 1) == 1
    assert I.horizon_lap(78, 0.5, 64) == 39 and I.horizon_lap(53, 0.5, 28) == 27 and I.horizon_lap(53, 0.5, 20) == 20
    assert I.horizon_lap(53, 0.5, 0) == 1
    assert all(I.horizon_lap(D, None) == 1 for D in (1, 28, 44, 78))


def test_running_order_by_cumulative_time_not_by_last_lap_time():
    laps = _laps([("a", [100, 100, 100]), ("b", [98, 99, 120]), ("c", [101, 101, 99.5])])
    st = I.race_state(laps, None, ["a", "b", "c"], 3)
    assert st.set_index("driverId")["cum_s"].to_dict() == {"a": 300.0, "b": 317.0, "c": 301.5}
    assert st["driverId"].tolist() == ["a", "c", "b"] and st["run_pos"].tolist() == [1, 2, 3]
    assert st["gap_ahead_s"].tolist() == [0.0, 1.5, pytest.approx(15.5)]
    assert I.race_state(laps, None, ["a", "b", "c"], 2)["driverId"].tolist() == ["b", "a", "c"]


def test_exact_ties_are_broken_by_driver_id_not_by_input_order():
    laps = _laps([("zed", [90, 90]), ("amy", [90, 90]), ("bob", [89, 91])])
    for order in (["zed", "amy", "bob"], ["bob", "amy", "zed"], ["amy", "zed", "bob"]):
        assert I.race_state(laps, None, order, 2)["driverId"].tolist() == ["amy", "bob", "zed"], order
    assert I.race_state(laps.sample(frac=1.0, random_state=1), None, ["zed", "amy", "bob"], 2)["driverId"].tolist() == ["amy", "bob", "zed"]


def test_lapped_car_with_a_lap_k_record_stays_a_runner_behind_the_cars_on_the_lead_lap():
    laps = _laps([("a", [90] * 5), ("b", [91] * 5), ("slow", [120] * 5)])
    st = I.race_state(laps, None, ["slow", "a", "b"], 5).set_index("driverId")
    assert st.loc["slow", "out"] == 0 and st.loc["slow", "run_pos"] == 3
    assert st.loc["slow", "gap_ahead_s"] == pytest.approx(600 - 455) and st.loc["a", "gap_ahead_s"] == 0.0


def test_retirements_before_k_sit_behind_all_runners_ordered_by_laps_then_driver_id():
    laps = _laps([("r1", [90] * 6), ("r2", [95] * 6), ("q4", [80] * 4), ("p4", [85] * 4), ("o2", [70] * 2), ("n0", [])])
    st = I.race_state(laps, None, ["n0", "o2", "p4", "q4", "r2", "r1"], 6)
    assert st["driverId"].tolist() == ["r1", "r2", "p4", "q4", "o2", "n0"]          # 4-lap tie: p4 before q4 although q4 is faster
    assert st["out"].tolist() == [0, 0, 1, 1, 1, 1] and st["laps_done"].tolist() == [6, 6, 4, 4, 2, 0]
    assert st["run_pos"].tolist() == [1, 2, 3, 4, 5, 6]
    assert st["gap_ahead_s"].tolist() == [0.0, 30.0, 0.0, 0.0, 0.0, 0.0]
    assert st[st["driverId"] == "n0"]["cum_s"].isna().all()
    f = I.state_features(st, 6).set_index("driverId")
    assert f["ir_out"].to_dict() == {"r1": 0.0, "r2": 0.0, "p4": 1.0, "q4": 1.0, "o2": 1.0, "n0": 1.0}
    assert f["ir_def"].to_dict() == {"r1": 0.0, "r2": 0.0, "p4": pytest.approx(2 / 6), "q4": pytest.approx(2 / 6), "o2": pytest.approx(4 / 6), "n0": 1.0}


def test_a_car_that_retires_after_k_is_a_runner_at_k_and_laps_after_k_are_ignored():
    laps = _laps([("a", [90] * 8), ("b", [91] * 3)])
    assert I.race_state(laps, None, ["a", "b"], 3)["out"].tolist() == [0, 0]
    assert I.race_state(laps, None, ["a", "b"], 4)["out"].tolist() == [0, 1]


def test_driver_without_entry_and_driver_in_entry_but_not_in_laps():
    st = I.race_state(_laps([("a", [90] * 3), ("ghost", [10] * 3)]), None, ["a", "b"], 3)
    assert st["driverId"].tolist() == ["a", "b"] and st["out"].tolist() == [0, 1]


def test_pit_flag_is_computed_from_stops_up_to_k_only_and_the_stored_flag_is_ignored():
    laps = _laps([(d, [90] * 12) for d in "abcdef"])
    base = [("a", 3, 24.0), ("b", 4, 25.0), ("c", 5, 23.0), ("d", 6, 24.5), ("e", 7, 120.0)]       # e: a long stop (red flag)
    # k = 8: median of the 5 stops = 24.5, e is 95.5 s above -> suspect, not counted
    st = I.race_state(laps, _pits(base), list("abcdef"), 8).set_index("driverId")
    assert st["pits_made"].to_dict() == {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 0.0, "f": 0.0}
    # stops after k, whatever their duration, cannot move the median or add a stop
    for dur in (2.0, 24.0, 4000.0):
        extra = base + [(d, 9 + i, dur) for i, d in enumerate("abcdef")] + [("f", 11, dur)]
        s2 = I.race_state(laps, _pits(extra), list("abcdef"), 8).set_index("driverId")
        assert s2["pits_made"].to_dict() == st["pits_made"].to_dict(), dur
    # a stop on lap k counts, on lap k+1 does not
    assert I.race_state(laps, _pits([("a", 8, 24.0)]), list("abcdef"), 8).set_index("driverId").loc["a", "pits_made"] == 1.0
    assert I.race_state(laps, _pits([("a", 9, 24.0)]), list("abcdef"), 8).set_index("driverId").loc["a", "pits_made"] == 0.0
    # the stored flag is ignored in both directions
    st_flag = I.race_state(laps, _pits([(d, l, u, True) for d, l, u in base[:4]]), list("abcdef"), 8).set_index("driverId")
    assert st_flag["pits_made"].to_dict() == {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 0.0, "f": 0.0}
    st_noflag = I.race_state(laps, _pits([(d, l, u, False) for d, l, u in base]), list("abcdef"), 8).set_index("driverId")
    assert st_noflag["pits_made"]["e"] == 0.0
    # margin is a parameter: with a huge margin nothing is suspect
    assert I.race_state(laps, _pits(base), list("abcdef"), 8, red_flag_margin_s=1000.0).set_index("driverId").loc["e", "pits_made"] == 1.0
    # boundary: duration exactly median + margin is NOT suspect (strict >)
    on = [("a", 3, 24.0), ("b", 4, 24.0), ("c", 5, 44.0)]               # median 24, margin 20 -> 44 is not > 44
    assert I.race_state(laps, _pits(on), list("abcdef"), 8).set_index("driverId").loc["c", "pits_made"] == 1.0
    # None / empty / NaN durations
    for p in (None, _pits([])):
        assert (I.race_state(laps, p, list("abcdef"), 8)["pits_made"] == 0).all()
    nan = I.race_state(laps, _pits([("a", 3, np.nan), ("b", 4, 24.0)]), list("abcdef"), 8).set_index("driverId")
    assert nan.loc["a", "pits_made"] == 1.0 and nan.loc["b", "pits_made"] == 1.0


def test_state_features_hand_values_and_ranges():
    laps = _laps([("a", [90] * 4), ("b", [90.5] * 4), ("c", [200] * 4), ("d", [90] * 1)])
    pits = _pits([("a", 2, 24.0), ("b", 2, 24.0), ("b", 3, 24.0), ("c", 1, 24.0)])
    st = I.race_state(laps, pits, ["a", "b", "c", "d"], 4)
    f = I.state_features(st, 4).set_index("driverId")
    assert f["ir_pos"].to_dict() == {"a": -0.5, "b": pytest.approx(-0.5 + 1 / 3), "c": pytest.approx(-0.5 + 2 / 3), "d": 0.5}
    assert f["ir_pits"].to_dict() == {"a": 0.0, "b": 1.0, "c": 0.0, "d": -1.0}          # median stops of the 3 runners (1, 2, 1) = 1
    assert f["ir_gap"]["a"] == 0.0 and f["ir_gap"]["b"] == pytest.approx(0.2)
    assert f["ir_gap"]["c"] == pytest.approx(6.0)                                         # 438 s gap capped at 60 s -> 6.0
    assert f["ir_gap"]["d"] == 0.0 and f["ir_out"]["d"] == 1.0 and f["ir_def"]["d"] == pytest.approx(0.75)
    one = I.state_features(I.race_state(_laps([("a", [90])]), None, ["a"], 1), 1)
    assert one.loc[0, "ir_pos"] == -0.5 and np.isfinite(one[COLS].to_numpy(float)).all()


def test_state_does_not_depend_on_row_order_of_laps_pits_or_driver_list():
    rng = np.random.default_rng(3)
    names = [f"d{i:02d}" for i in range(12)]
    laps = _laps([(d, np.round(rng.uniform(88, 95, 10), 0)) for d in names])
    laps = laps[~((laps["driverId"] == "d03") & (laps["lap"] > 6))]
    pits = _pits([(d, int(rng.integers(1, 10)), float(rng.choice([22, 24, 26, 90]))) for d in names for _ in range(2)])
    base = I.race_state(laps, pits, names, 8)
    for s in range(4):
        sh = I.race_state(laps.sample(frac=1.0, random_state=s), pits.sample(frac=1.0, random_state=s), list(rng.permutation(names)), 8)
        pd.testing.assert_frame_equal(base, sh, check_exact=True)


def test_missing_lap_time_inside_the_first_k_laps_documents_current_behaviour():
    """KNOWN WEAKNESS (0 such drivers in 2021-2023, see data_checks.csv): a NaN lap time is skipped by the sum, so the car's cumulative
    time is understated and it is ranked AHEAD of cars that are really in front of it. If this is changed, update this test."""
    st = I.race_state(_laps([("a", [90, 90, 90]), ("b", [100, np.nan, 100])]), None, ["a", "b"], 3)
    assert st.set_index("driverId").loc["b", "cum_s"] == 200.0 and st["driverId"].tolist() == ["b", "a"]


def test_attach_keeps_row_order_and_gives_nan_for_races_without_laps():
    ent = pd.DataFrame({"race_id": [1, 1, 2, 2, 3], "season": 2021, "round": [1, 1, 2, 2, 3], "driverId": ["b", "a", "a", "b", "a"]})
    laps = pd.concat([_laps([("a", [90] * 3), ("b", [91] * 3)]).assign(season=2021, round=1),
                      _laps([("a", [92] * 3), ("b", [90] * 3)]).assign(season=2021, round=2)])
    table = I.build_state_table(ent, laps, None, {1: 3, 2: 3, 3: 3})
    assert set(table["race_id"]) == {1, 2}
    out = I.attach(ent, table)
    assert out["driverId"].tolist() == ent["driverId"].tolist() and out.index.equals(ent.index)
    assert out.loc[[0, 1], "ir_pos"].tolist() == [0.5, -0.5] and out.loc[[2, 3], "ir_pos"].tolist() == [0.5, -0.5]
    assert out.loc[4, COLS].isna().all() and not out.loc[:3, COLS].isna().any().any()
    t1 = I.build_state_table(ent, laps, None, {1: 1})
    assert set(t1["race_id"]) == {1} and (t1["k"] == 1).all()


def test_scheduled_distance_overrides_table_races_only():
    laps = pd.concat([_laps([("a", [90] * 10)]).assign(season=2021, round=1), _laps([("a", [90] * 7)]).assign(season=2020, round=1)])
    sd = I.scheduled_distance(laps, pd.DataFrame({"season": [2021], "round": [1], "scheduled_laps": [44]}))
    assert sd == {(2021, 1): 44, (2020, 1): 7} and I.race_distance(laps) == {(2021, 1): 10, (2020, 1): 7}


# ====================================================================== stage-2 models: gradient, switches off, no test-row outcome use
def _s2_frame(n_races=14, n=7, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(1, n_races + 1):
        for p in range(1, n + 1):                                        # rows in finishing order inside a race, as in the harness
            rows.append(dict(race_id=r, driverId=f"d{p}", pos=p, y=(p - 0.5) / n, classified=True,
                             sc=float(-p + rng.normal(0, 1.5)), x1=float(rng.normal()), x2=float(rng.normal())))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("fix_c", [False, True])
def test_stage2_latent_analytic_gradient_equals_finite_differences(fix_c, monkeypatch):
    df = _s2_frame()
    cap = {}

    def fake_min(fun, x0, args=(), jac=None, method=None, options=None):
        cap.update(fun=fun, x0=np.array(x0), args=args)
        return type("R", (), {"x": np.array(x0)})()

    monkeypatch.setattr(S2, "minimize", fake_min)
    m = S2.Stage2Latent("sc", ["x1", "x2"], lam_x=3.0, fix_c=fix_c, halflife=5.0)
    m.fit_predict(df[df["race_id"] <= 12], df[df["race_id"] == 13])
    f, x0, args = cap["fun"], cap["x0"], cap["args"]
    rng = np.random.default_rng(1)
    for _ in range(3):
        th = x0 + rng.normal(0, 0.7, len(x0))
        val, g = f(th, *args)
        eps = 1e-6
        fd = np.array([(f(th + eps * e, *args)[0] - f(th - eps * e, *args)[0]) / (2 * eps) for e in np.eye(len(th))])
        assert np.allclose(g, fd, rtol=1e-5, atol=1e-7), (g, fd)
    # the ridge is on the in-race coefficients (lam_x) and tiny on c; with fix_c there is no c at all
    assert m.names == (["x1", "x2"] if fix_c else ["c", "x1", "x2"]) and m.pen[-2:].tolist() == [3.0, 3.0]
    if not fix_c:
        assert m.pen[0] == 1e-4


def test_stage2_latent_with_no_inrace_columns_and_c_fixed_returns_the_pre_race_score_exactly():
    df = _s2_frame()
    out = S2.Stage2Latent("sc", [], fix_c=True).fit_predict(df[df["race_id"] < 10], df[df["race_id"] == 10])
    assert np.array_equal(out, df.loc[df["race_id"] == 10, "sc"].to_numpy())


def test_stage2_ignores_the_outcome_columns_of_the_test_race_and_of_later_races():
    df = _s2_frame()
    tr, te = df[df["race_id"] < 12], df[df["race_id"] == 12]
    alt = te.copy()
    alt["pos"], alt["y"], alt["classified"] = np.random.default_rng(5).permutation(alt["pos"].to_numpy()), 0.123, False
    for mk in (lambda: S2.Stage2Latent("sc", ["x1", "x2"], lam_x=1.0, halflife=5.0), lambda: S2.Stage2GBM("sc", ["x1", "x2"], n_estimators=20, min_child_samples=5)):
        a, b = mk().fit_predict(tr, te), mk().fit_predict(tr, alt)
        assert np.array_equal(a, b)
    # control: the training races DO matter
    tr2 = tr.copy()
    tr2["x1"] = -tr2["x1"]
    tr2["y"] = 1 - tr2["y"]
    assert not np.array_equal(S2.Stage2GBM("sc", ["x1", "x2"], n_estimators=20, min_child_samples=5).fit_predict(tr, te),
                              S2.Stage2GBM("sc", ["x1", "x2"], n_estimators=20, min_child_samples=5).fit_predict(tr2, te))


def test_stage2_min_race_drops_early_training_races():
    df = _s2_frame()
    tr, te = df[df["race_id"] < 12], df[df["race_id"] == 12]
    m = S2.Stage2Latent("sc", ["x1"], min_race=5)
    m.fit_predict(tr, te)
    assert m.last_n_train_races == 7
    bad = tr.copy()
    bad.loc[bad["race_id"] < 5, "x1"] = 99.0                                      # races below min_race must not matter
    m2 = S2.Stage2Latent("sc", ["x1"], min_race=5)
    assert np.array_equal(m2.fit_predict(bad, te), S2.Stage2Latent("sc", ["x1"], min_race=5).fit_predict(tr, te))


# ====================================================================== real data fixtures
@pytest.fixture(scope="module")
def cfg():
    return load_yaml(M19 / "config.yaml")


@pytest.fixture(scope="module")
def data(cfg):
    from pitlane.plan2 import pipeline2 as Q
    entries = Q.load_all()
    laps = pd.read_parquet(LAPS)
    pits = pd.read_parquet(PITS)
    laps = laps[(laps["source"] == cfg["source"]) & (laps["season"] <= 2023)].reset_index(drop=True)
    pits = pits[(pits["source"] == cfg["source"]) & (pits["season"] <= 2023)].reset_index(drop=True)
    _, counted = split_races(entries, "dev")
    sched = pd.read_csv(SCHED)
    margin = float(load_yaml(ROOT / cfg["ingest_config"])["pits"]["red_flag_margin_s"])
    return dict(entries=entries, laps=laps, pits=pits, counted=counted, sched=sched, margin=margin,
                avail=I.race_distance(laps), dist=I.scheduled_distance(laps, sched))


def _ks(d, frac):
    """Horizon lap per race exactly as run.py builds it: scheduled distance, capped at the last lap in the table."""
    out = {}
    for rid, g in d["entries"].groupby("race_id"):
        key = (int(g["season"].iloc[0]), int(g["round"].iloc[0]))
        if key in d["dist"]:
            out[int(rid)] = I.horizon_lap(d["dist"][key], frac, d["avail"][key])
    return out


def _key(entries, rid):
    g = entries[entries["race_id"] == rid]
    return int(g["season"].iloc[0]), int(g["round"].iloc[0])


def _frame(d, laps, pits, ks):
    ent = d["entries"]
    table = I.build_state_table(ent[ent["season"] <= 2023], laps, pits, ks, d["margin"])
    return I.attach(ent, table), table


def _alter(laps, pits, ks_key, r_key, mode, seed=0):
    """Change laps and pit stops AFTER lap k: of race r (after its k), of earlier races (after their own k), of later races (all).
    mode: random | delete | shift. Pit durations are replaced as well (the flag now comes from them)."""
    rng = np.random.default_rng(seed)

    def thr(df):
        kk = pd.DataFrame([dict(season=s, round=r, thr=(k if (s, r) <= r_key else 0)) for (s, r), k in ks_key.items()])
        t = df[["season", "round"]].merge(kk, on=["season", "round"], how="left")["thr"].fillna(10 ** 6).to_numpy()
        return pd.Series(t, index=df.index)
    late, plate = laps["lap"] > thr(laps), pits["lap"] > thr(pits)
    L, P = laps.copy(), pits.copy()
    if mode == "random":
        L.loc[late, "lap_time_s"] = rng.uniform(60, 200, int(late.sum()))
        P.loc[plate, "duration_s"] = rng.choice([2.0, 22.0, 400.0, 3000.0], int(plate.sum()))
        P.loc[plate, "red_flag_suspect"] = rng.random(int(plate.sum())) < 0.5
        P.loc[plate, "lap"] = pits.loc[plate, "lap"] + rng.integers(0, 5, int(plate.sum()))      # stays after k
    elif mode == "delete":
        L, P = L[~late].reset_index(drop=True), P[~plate].reset_index(drop=True)
    elif mode == "shift":
        L.loc[late, "lap"] = L.loc[late, "lap"] + 3
        P.loc[plate, "lap"] = P.loc[plate, "lap"] + 3
    else:
        raise ValueError(mode)
    return L, P


STATE_RACES = {"H4": (150, 166, 177, 190, 203), "H5": (140, 166, 177, 193, 202), "H6": (145, 166, 177, 189, 200)}


@needs_laps
@pytest.mark.parametrize("h", ["H4", "H5", "H6"])
@pytest.mark.parametrize("mode", ["random", "delete", "shift"])
def test_state_and_features_of_race_r_unchanged_when_laps_and_pits_after_k_are_altered(data, h, mode):
    d = data
    ks_all = _ks(d, HZ[h])
    ks_key = {_key(d["entries"], r): k for r, k in ks_all.items()}
    for r in STATE_RACES[h]:
        sub = {x: ks_all[x] for x in [y for y in ks_all if y < r][-4:] + [r]}
        _, base_t = _frame(d, d["laps"], d["pits"], sub)
        L, P = _alter(d["laps"], d["pits"], ks_key, _key(d["entries"], r), mode, seed=r)
        _, t = _frame(d, L, P, sub)
        for x in sub:                                                                    # race r and earlier races (their own k)
            a, b = base_t[base_t["race_id"] == x].reset_index(drop=True), t[t["race_id"] == x].reset_index(drop=True)
            assert len(a) > 0
            pd.testing.assert_frame_equal(a, b, check_exact=True)
    # the alteration is not a no-op
    L, P = _alter(d["laps"], d["pits"], ks_key, _key(d["entries"], STATE_RACES[h][0]), mode, seed=1)
    assert not (L.reset_index(drop=True).equals(d["laps"]) and P.reset_index(drop=True).equals(d["pits"]))


def _pit_perturbation_diff(d, h, variant):
    """Race ids whose lap-k pits_made changes when raw pit durations after k are replaced (flag recomputed whole-race, as ingestion did)."""
    ks = _ks(d, HZ[h])
    base_p = d["pits"].copy()
    P = d["pits"].copy()
    race_of = P[["season", "round"]].merge(d["entries"].groupby(["season", "round"])["race_id"].first().reset_index(), on=["season", "round"], how="left")["race_id"]
    thr = pd.Series([ks.get(r, 0) for r in race_of], index=P.index)
    late = P["lap"] > thr
    assert late.sum() > 0
    P.loc[late, "duration_s"] = 400.0 if variant == "huge" else 2.0
    for q in (base_p, P):
        q["red_flag_suspect"] = q["duration_s"] > q.groupby(["season", "round"])["duration_s"].transform("median") + 20.0
    ent = d["entries"][d["entries"]["season"] <= 2023]
    a = I.build_state_table(ent, d["laps"], base_p, ks, d["margin"])
    b = I.build_state_table(ent, d["laps"], P, ks, d["margin"])
    a, b = a[a["race_id"].isin(d["counted"])].reset_index(drop=True), b[b["race_id"].isin(d["counted"])].reset_index(drop=True)
    return a, b, sorted(set(a.loc[a["pits_made"].to_numpy() != b["pits_made"].to_numpy(), "race_id"]))


@needs_laps
@pytest.mark.parametrize("h", ["H4", "H5", "H6"])
@pytest.mark.parametrize("variant", ["huge", "tiny"])
def test_ACCEPTANCE_ir_pits_unchanged_when_raw_pit_durations_after_k_are_perturbed(data, h, variant):
    """ACCEPTANCE (the v1 leak): raw pit durations after lap k are replaced (400 s or 2 s), for ALL dev races at once, and the stored
    red-flag column is recomputed the way ingestion did it (race-wide median + 20 s) and passed along. The lap-k state of every dev
    race must not change. v1 failed this (3-6 races); it must pass now that the flag is built from stops <= k."""
    a, b, bad = _pit_perturbation_diff(data, h, variant)
    assert not bad, f"{h}/{variant}: pits_made at lap k changes in {len(bad)} dev races: {bad}"
    pd.testing.assert_frame_equal(a, b, check_exact=True)


@needs_laps
def test_the_pit_perturbation_test_does_detect_the_v1_behaviour(data, monkeypatch):
    """Test of the test: with race_state restored to v1 (trust the stored whole-race flag) the perturbation changes pits_made in several races."""
    def v1_stops(pits, k, red_flag_margin_s=20.0):
        if pits is None or not len(pits):
            return pd.DataFrame({"driverId": [], "lap": []})
        return pits[(pits["lap"] <= k) & ~pits["red_flag_suspect"].astype(object).eq(True)]
    monkeypatch.setattr(I, "stops_before", v1_stops)
    seen = {}
    for h in ("H4", "H5", "H6"):
        seen[h] = sum(len(_pit_perturbation_diff(data, h, v)[2]) for v in ("huge", "tiny"))
    assert all(v > 0 for v in seen.values()), seen


@needs_laps
@pytest.mark.parametrize("h", ["H4", "H5", "H6"])
def test_control_changing_a_lap_at_or_before_k_changes_the_state(data, h):
    d = data
    ks = _ks(d, HZ[h])
    r = 185
    key, k = _key(d["entries"], r), ks[r]
    sub = {r: k}
    _, base_t = _frame(d, d["laps"], d["pits"], sub)
    a = base_t[base_t["race_id"] == r].set_index("driverId")
    lead = a.sort_values("run_pos").index[0]
    for lap in sorted({1, max(1, k // 2), k}):
        L = d["laps"].copy()
        L.loc[(L["season"] == key[0]) & (L["round"] == key[1]) & (L["lap"] == lap) & (L["driverId"] == lead), "lap_time_s"] += 500.0
        _, t = _frame(d, L, d["pits"], sub)
        b = t[t["race_id"] == r].set_index("driverId")
        assert b.loc[lead, "cum_s"] == a.loc[lead, "cum_s"] + 500.0 and b.loc[lead, "run_pos"] > a.loc[lead, "run_pos"], (h, lap)
        assert not np.array_equal(a.loc[b.index, "ir_pos"].to_numpy(), b["ir_pos"].to_numpy())
    new = dict(driverId=lead, lap=k, stop=9.0, time_of_day=None, duration_s=24.0, red_flag_suspect=False, season=key[0], round=key[1], source="jolpica")
    def g(P):
        t = _frame(d, d["laps"], P, sub)[1]
        return t[(t["race_id"] == r) & (t["driverId"] == lead)]["pits_made"].iloc[0]
    p0 = g(d["pits"])
    assert g(pd.concat([d["pits"], pd.DataFrame([new])], ignore_index=True)) == p0 + 1
    assert g(pd.concat([d["pits"], pd.DataFrame([dict(new, lap=k + 1)])], ignore_index=True)) == p0
    # control for the new flag: a very long stop on lap <= k is not counted, the same stop after k leaves the count alone
    if h != "H4":       # at H4 (k = 1) the median of the stops on lap 1 is mostly that one stop itself, so nothing can look long
        assert g(pd.concat([d["pits"], pd.DataFrame([dict(new, duration_s=900.0)])], ignore_index=True)) == p0


# ---------------------------------------------------------------- forecasts
FORECAST = {"H4": 175, "H5": 185, "H6": 195}


def _proxy_scores(eh):
    """Stand-in for the frozen pre-race score of stage 1 (which never reads laps): a pre-race-only function of grid and qualifying."""
    e = eh.copy()
    e["pre_lat"] = -(0.6 * e["grid_norm"].fillna(0.5) + 0.4 * e["quali_norm"].fillna(0.5))
    e["pre_gbm"] = -(0.3 * e["grid_norm"].fillna(0.5) + 0.7 * e["quali_norm"].fillna(0.5)) + 0.05 * e["drv_ew16"].fillna(0.0)
    return e


def _forecast(cfg, d, h, eh, r):
    rows = np.flatnonzero(eh["race_id"].to_numpy() == r)
    first = int(rows.min())
    fz = load_yaml(ROOT / cfg["frozen_path"])
    lat, gbm = dict(fz[cfg["latent"]["key"]][cfg["latent"]["cell"]]), dict(fz[cfg["gbm"]["key"]][cfg["gbm"]["cell"]])
    rc = json.loads((RES / "reproduction_check.json").read_text(encoding="utf-8"))["lam_x_chosen"][h]
    e = _proxy_scores(eh)
    keep = {k: gbm[k] for k in ("num_leaves", "n_estimators", "learning_rate", "min_child_samples", "halflife") if k in gbm}
    models = {
        "latent_joint": build_model(dict(lat, x_cols=cfg["feature_cols"], lam_x=rc["joint"]), e),
        "gbm_joint": build_model(dict(gbm, extra_cols=cfg["feature_cols"]), e),
        "latent_s2": S2.Stage2Latent("pre_lat", cfg["feature_cols"], lam_x=rc["two_stage"], halflife=float(lat["halflife"]),
                                     fix_c=bool(cfg["stage2"]["fix_c"]), min_race=int(cfg["min_train_race"])),
        "gbm_s2": S2.Stage2GBM("pre_gbm", cfg["feature_cols"], min_race=int(cfg["min_train_race"]), **keep),
    }
    return {n: np.asarray(m.fit_predict(e.iloc[:first], e.iloc[rows]), dtype=float) for n, m in models.items()}


@needs_laps
@needs_res
@pytest.mark.slow
@pytest.mark.parametrize("h", ["H4", "H5", "H6"])
def test_ACCEPTANCE_forecast_of_race_r_bitwise_unchanged_by_laps_and_pits_after_k_and_by_all_later_races(cfg, data, h):
    """Acceptance (PLAN3 R13): random / deleted / shifted laps and pit stops after lap k of race r, after lap k_j of every earlier race,
    and ALL laps and pits of every later race leave all four forecasts of race r (latent and gbm, two-stage and joint) bit for bit
    unchanged. REDUCED: the two-stage models take a proxy pre-race score (grid/qualifying based) instead of the frozen walk-forward
    latent_111/gbm_100 score, because stage 1 never reads laps and a real 150-race walk-forward per case is too slow; the joint
    variants use the real frozen configurations."""
    d = data
    r = FORECAST[h]
    ks = _ks(d, HZ[h])
    ks_key = {_key(d["entries"], x): k for x, k in ks.items()}
    base_eh, _ = _frame(d, d["laps"], d["pits"], ks)
    base = _forecast(cfg, d, h, base_eh, r)
    assert all(np.isfinite(v).all() for v in base.values()) and len(base) == 4
    for mode in ("random", "delete", "shift"):
        L, P = _alter(d["laps"], d["pits"], ks_key, _key(d["entries"], r), mode, seed=7)
        eh, _ = _frame(d, L, P, ks)
        rr = base_eh["race_id"] == r
        assert np.array_equal(base_eh.loc[rr, COLS].to_numpy(float), eh.loc[rr, COLS].to_numpy(float), equal_nan=True)
        if mode != "shift":
            later = base_eh["race_id"] > r
            assert not np.array_equal(base_eh.loc[later, COLS].to_numpy(float), eh.loc[later, COLS].to_numpy(float), equal_nan=True)
        got = _forecast(cfg, d, h, eh, r)
        for name in base:
            assert np.array_equal(base[name], got[name]), (h, mode, name, float(np.abs(base[name] - got[name]).max()))


@needs_laps
@needs_res
@pytest.mark.slow
@pytest.mark.parametrize("h", ["H4", "H6"])
def test_control_forecast_reacts_to_laps_up_to_k_of_race_r_and_to_earlier_races(cfg, data, h):
    d = data
    r = FORECAST[h]
    ks = _ks(d, HZ[h])
    base_eh, _ = _frame(d, d["laps"], d["pits"], ks)
    base = _forecast(cfg, d, h, base_eh, r)
    key = _key(d["entries"], r)
    lead = base_eh[base_eh["race_id"] == r].sort_values("ir_pos")["driverId"].iloc[0]
    L = d["laps"].copy()
    L.loc[(L["season"] == key[0]) & (L["round"] == key[1]) & (L["lap"] == ks[r]) & (L["driverId"] == lead), "lap_time_s"] += 500.0
    got = _forecast(cfg, d, h, _frame(d, L, d["pits"], ks)[0], r)
    for name in base:
        assert not np.array_equal(base[name], got[name]), (h, name, "race-r lap <= k")
    x = r - 3
    kx = _key(d["entries"], x)
    L2 = d["laps"].copy()
    m = (L2["season"] == kx[0]) & (L2["round"] == kx[1]) & (L2["lap"] <= ks[x])
    L2.loc[m, "lap_time_s"] = np.random.default_rng(0).uniform(80, 120, int(m.sum()))
    got2 = _forecast(cfg, d, h, _frame(d, L2, d["pits"], ks)[0], r)
    for name in ("latent_joint", "latent_s2"):
        assert not np.array_equal(base[name], got2[name]), name


# ====================================================================== horizon lap and scheduled distance
@needs_laps
def test_ACCEPTANCE_horizon_k_does_not_depend_on_laps_after_k(data):
    """Acceptance: deleting every lap after k (and every lap of later races) of each dev race leaves its horizon lap unchanged when
    k is rebuilt exactly as run.py builds it (scheduled distance from the table, capped at the last lap that exists)."""
    d = data
    for h in ("H5", "H6"):
        ks = _ks(d, HZ[h])
        for r in d["counted"]:
            key = _key(d["entries"], r)
            lp = d["laps"][(d["laps"]["season"] == key[0]) & (d["laps"]["round"] == key[1])]
            cut = lp[lp["lap"] <= ks[r]]
            dist2 = I.scheduled_distance(cut, d["sched"])
            avail2 = I.race_distance(cut)
            assert dist2[key] == d["dist"][key]
            assert I.horizon_lap(dist2[key], HZ[h], avail2[key]) == ks[r], (h, r, key)


@needs_laps
def test_h4_horizon_is_lap_one_everywhere_and_k_is_inside_the_race(data):
    d = data
    for h in HZ:
        ks = _ks(d, HZ[h])
        for r in d["counted"]:
            assert r in ks and 1 <= ks[r] <= d["avail"][_key(d["entries"], r)]
            if h == "H4":
                assert ks[r] == 1


@needs_laps
def test_horizon_laps_of_the_five_shortened_or_cut_dev_races(data):
    """Hand values from the scheduled distance: 149 Belgium (44, 1 lap run) is capped to lap 1 at H5/H6; 166 Monaco (78), 176 Singapore
    (61), 177 Japan (53) and 195 Monza (51 used) are NOT capped (the race ran past the horizon) and use the scheduled distance."""
    d = data
    want = {149: (1, 1), 166: (20, 39), 176: (16, 31), 177: (14, 27), MONZA: (13, 26)}
    for r, (k5, k6) in want.items():
        assert _ks(d, 0.25)[r] == k5 and _ks(d, 0.5)[r] == k6, r
    for rid, (key, sched, real) in SHORT_RACES.items():
        assert _key(d["entries"], rid) == key and rid in d["counted"] and d["avail"][key] == real and d["dist"][key] == sched
    assert _key(d["entries"], MONZA) == (2023, 14) and d["avail"][(2023, 14)] == 51 == d["dist"][(2023, 14)]
    # with the old realised-distance rule the same races would have had other k at H5 and H6
    changed = sum(int(I.horizon_lap(real, f) != I.horizon_lap(sched, f)) for _, sched, real in SHORT_RACES.values() for f in (0.25, 0.5))
    assert changed == 8
    cap = pd.read_csv(RES / "capped_horizons.csv")
    assert cap[["horizon", "season", "round", "nominal_k", "used_k"]].values.tolist() == [["H5", 2021, 12, 11, 1], ["H6", 2021, 12, 22, 1]]


@needs_laps
def test_degenerate_race_149_has_one_racing_lap_and_the_same_state_at_every_horizon(data):
    d = data
    key = _key(d["entries"], 149)
    ll = d["laps"][(d["laps"]["season"] == 2021) & (d["laps"]["round"] == 12)]
    assert key == (2021, 12) and ll["lap"].max() == 1 and ll["driverId"].nunique() == 20
    assert {_ks(d, f)[149] for f in HZ.values()} == {1}


@needs_laps
def test_scheduled_laps_table_rows_values_and_provenance(data):
    s, d = data["sched"], data
    dev_keys = {_key(d["entries"], r) for r in d["counted"]}
    assert len(s) == 66 and not s.duplicated(["season", "round"]).any() and set(zip(s["season"], s["round"])) == dev_keys
    assert s[["scheduled_laps", "laps_in_table"]].notna().all().all() and (s["scheduled_laps"] > 0).all()
    assert (s["scheduled_laps"] >= s["laps_in_table"]).all()
    for r in s.itertuples():                                                         # laps_in_table is the real table, not a typed number
        assert d["avail"][(r.season, r.round)] == r.laps_in_table, (r.season, r.round)
    differ = s[s["scheduled_laps"] != s["laps_in_table"]]
    assert sorted(zip(differ["season"], differ["round"])) == [(2021, 12), (2022, 7), (2022, 17), (2022, 18)]
    assert s["note"].notna().all() and (s["note"].str.len() > 10).all()
    same = s[s["scheduled_laps"] == s["laps_in_table"]]
    assert same["note"].str.contains("ran to its scheduled distance|aborted start").all()
    # spot checks against well-known race distances (from memory of the official records, unverified here)
    spot = {(2021, 1): 56, (2021, 12): 44, (2022, 7): 77, (2022, 18): 53, (2021, 22): 58, (2023, 6): 78, (2023, 22): 58, (2023, 21): 50}
    got = s.set_index(["season", "round"])["scheduled_laps"].to_dict()
    for k, v in spot.items():
        assert got[k] == v, k


@needs_laps
def test_ACCEPTANCE_scheduled_laps_table_has_a_source_column():
    """Hand tables in data/external need a source for every row (CLAUDE.md / test brief). config.yaml says 'table with a source per
    row', but the file has only a free-text `note`. FAILS until a source (URL or document name) is added per row."""
    s = pd.read_csv(SCHED)
    src = [c for c in s.columns if "source" in c.lower() or "url" in c.lower()]
    assert src, f"no source column in {list(s.columns)}"
    assert s[src[0]].notna().all() and (s[src[0]].astype(str).str.len() > 3).all()


@needs_laps
def test_ACCEPTANCE_every_scheduled_laps_row_has_an_independent_source():
    """Hand-table rule: every row has a source that is not the very lap table it is meant to be checked against. 63 rows cite a
    Wikipedia race article; 3 rows (2021 R2, 2021 R22, 2022 R4) cite only 'lap table (Jolpica)' because the infobox could not be parsed,
    i.e. their scheduled distance is unverified and circular. FAILS until those three are sourced (or declared unverified)."""
    s = pd.read_csv(SCHED)
    src = s["source"].fillna("").astype(str)
    assert (src.str.len() > 10).all(), "missing source"
    circular = s[src.str.lower().str.startswith("lap table")]
    assert circular.empty, f"rows without an independent source: {sorted(zip(circular['season'], circular['round']))}"


@needs_laps
def test_scheduled_laps_cut_race_rows_cite_a_scheduled_distance_and_all_other_rows_agree_with_the_lap_table(data):
    s = data["sched"].set_index(["season", "round"])
    for key, (sched, real) in {(2021, 12): (44, 1), (2022, 7): (77, 64), (2022, 17): (61, 59), (2022, 18): (53, 28)}.items():
        r = s.loc[key]
        assert r["scheduled_laps"] == sched and r["laps_in_table"] == real == data["avail"][key]
        assert str(sched) in r["source"] and "wikipedia" in r["source"].lower()
    assert s.loc[(2023, 14), "scheduled_laps"] == s.loc[(2023, 14), "laps_in_table"] == 51 == data["avail"][(2023, 14)]
    rest = s.drop(index=[(2021, 12), (2022, 7), (2022, 17), (2022, 18), (2023, 14)])
    assert len(rest) == 61 and (rest["scheduled_laps"] == rest["laps_in_table"]).all()
    assert all(data["avail"][k] == v for k, v in rest["scheduled_laps"].items())


@needs_laps
def test_monza_2023_distance_is_a_post_start_fact_documented():
    """2023 R14: 51 is entered as the 'pre-race' distance because the start was aborted (53 -> 51). That is known only after the start
    procedure, i.e. after the event began. It changes k only at H5 (13 vs 14) and H6 (26 vs 27); H4 is unaffected. Reported, not a failure."""
    assert I.horizon_lap(51, 0.25) == 13 and I.horizon_lap(53, 0.25) == 14 and I.horizon_lap(51, 0.5) == 26 and I.horizon_lap(53, 0.5) == 27
    assert I.horizon_lap(51, None) == I.horizon_lap(53, None) == 1


def _stored_flag_from_prefix(pits: pd.DataFrame, k: int, margin: float = 20.0) -> pd.Series:
    early = pits[pits["lap"] <= k]
    return early["duration_s"] > early["duration_s"].median() + margin


@needs_laps
def test_stored_pit_flag_is_the_whole_race_median_rule_and_v1_counts(data):
    """Premise and size of the v1 leak (now fixed in src): the stored flag equals duration > race-wide median + 20 s, and recomputing it
    from stops <= k changes the per-driver stop counts in 4 (H4), 5 (H5), 3 (H6) of the 66 dev races (v1 horizons: 4/6/3; v2 horizons: 4/5/3). Data-only; src ignores the flag."""
    d = data
    p = d["pits"]
    whole = p["duration_s"] > p.groupby(["season", "round"])["duration_s"].transform("median") + 20.0
    assert (whole.to_numpy() == p["red_flag_suspect"].astype(object).eq(True).to_numpy()).all()
    out = {}
    for h, f in HZ.items():
        ks = _ks(d, f)
        n = 0
        for rid in d["counted"]:
            key = _key(d["entries"], rid)
            g = p[(p["season"] == key[0]) & (p["round"] == key[1])]
            early = g[g["lap"] <= ks[rid]]
            if early.empty:
                continue
            old = early[~early["red_flag_suspect"].astype(object).eq(True)].groupby("driverId").size()
            new = early[~_stored_flag_from_prefix(g, ks[rid])].groupby("driverId").size()
            idx = old.index.union(new.index)
            n += int(not old.reindex(idx).fillna(0).equals(new.reindex(idx).fillna(0)))
        out[h] = n
    assert out == {"H4": 4, "H5": 5, "H6": 3}, out


# ====================================================================== data checks
@needs_laps
def test_lap_and_pit_tables_row_counts_duplicates_values(data):
    d = data
    laps, pits, ent = d["laps"], d["pits"], d["entries"]
    dev = laps[laps["season"].between(2021, 2023)]
    assert dev.groupby("season")["round"].nunique().tolist() == [22, 22, 22]
    assert not laps.duplicated(["season", "round", "driverId", "lap"]).any()
    assert dev["lap_time_s"].notna().all() and (dev["lap_time_s"] > 30).all() and (dev["lap_time_s"] < 3600 * 3).all()
    assert (dev["lap"] >= 1).all() and dev["lap"].dtype.kind == "i"
    g = dev.groupby(["season", "round", "driverId"])["lap"]
    assert (g.max() == g.count()).all()
    e = ent[ent["season"].between(2021, 2023)][["season", "round", "driverId"]].drop_duplicates()
    m = dev[["season", "round", "driverId"]].drop_duplicates().merge(e, how="left", indicator=True)
    # one entry-table gap (pre-existing, not M19): Alonso has 47 laps in the Mexico 2023 lap table but no entry row (19 entrants)
    assert m[m["_merge"] != "both"][["season", "round", "driverId"]].values.tolist() == [[2023, 19, "alonso"]]
    per = dev.groupby(["season", "round"])["lap"].max()
    assert all(per[k] == d["avail"][k] for k in per.index)
    pdv = pits[pits["season"].between(2021, 2023)]
    assert not pdv.duplicated(["season", "round", "driverId", "lap", "stop"]).any()
    assert (pdv["lap"] >= 1).all() and (pdv["duration_s"].dropna() > 0).all()
    assert pdv[pdv["duration_s"].isna()][["season", "round", "driverId", "lap"]].values.tolist() == [[2021, pdv[pdv["driverId"] == "mazepin"]["round"].iloc[0], "mazepin", 27]] or pdv["duration_s"].isna().sum() == 1
    last = dev.groupby(["season", "round", "driverId"])["lap"].max().rename("last").reset_index()
    mm = pdv.merge(last, on=["season", "round", "driverId"], how="left")
    assert mm["last"].notna().all() and (mm["lap"] <= mm["last"] + 1).all()


@needs_laps
def test_derived_running_order_matches_stored_position_2021_22_independent_recomputation(data):
    d = data
    ent, laps = d["entries"], d["laps"]
    stored = pd.read_csv(RES / "data_checks.csv")
    got = {}
    for (s, r), g in ent[ent["season"].isin([2021, 2022])].groupby(["season", "round"]):
        ll = laps[(laps["season"] == s) & (laps["round"] == r)]
        for h, f in HZ.items():
            k = I.horizon_lap(d["dist"][(s, r)], f, int(ll["lap"].max()))
            cum = ll[ll["lap"] <= k].groupby("driverId")["lap_time_s"].sum()
            at_k = ll[ll["lap"] == k].set_index("driverId")["position"]
            order = sorted(at_k.index, key=lambda x: (cum[x], x))
            mine = pd.Series({x: i + 1 for i, x in enumerate(order)})
            got[(s, r, h)] = float((mine.reindex(at_k.index).to_numpy() == at_k.rank().to_numpy()).mean())
    ag = stored[stored["check"].str.startswith("agree_stored_position_")]
    assert len(ag) == 44 * 3 == len(got)
    for _, row in ag.iterrows():
        assert got[(row["season"], row["round"], row["check"][-2:])] == pytest.approx(row["value"], abs=1e-12)
    below = {k: v for k, v in got.items() if v < 1.0}
    assert set(below) == {(2022, 18, "H5"), (2022, 18, "H6")}, below
    assert below[(2022, 18, "H5")] == pytest.approx(15 / 18) and below[(2022, 18, "H6")] == pytest.approx(13 / 18)
    assert np.mean(list(got.values())) > 0.99


@needs_res
@needs_laps
def test_data_checks_csv_consistent_with_lap_tables(data):
    d = data
    dc = pd.read_csv(RES / "data_checks.csv")
    ent = d["entries"]
    assert {"scheduled_distance_laps", "last_lap_in_table"} <= set(dc["check"])
    for chk, ref in (("scheduled_distance_laps", d["dist"]), ("last_lap_in_table", d["avail"])):
        s = dc[dc["check"] == chk].set_index(["season", "round"])["value"]
        assert len(s) == 204
        for key, v in ref.items():
            if key in s.index:
                assert s[key] == v, (chk, key)
    no_lap = dc[(dc["check"] == "entries_without_any_lap") & (dc["season"] >= 2021)]
    for _, row in no_lap.iterrows():
        g = ent[(ent["season"] == row["season"]) & (ent["round"] == row["round"])]
        ll = d["laps"][(d["laps"]["season"] == row["season"]) & (d["laps"]["round"] == row["round"])]
        assert row["value"] == len(set(g["driverId"]) - set(ll["driverId"]))
    assert (dc[(dc["check"] == "drivers_with_lap_gaps_or_nan") & (dc["season"] >= 2021)]["value"] == 0).all()
    assert (dc["check"] == "no_laps").sum() == 0


def _state_csv(prefix, h, m):
    return pd.read_csv(_dir(prefix, h, m) / "state_table_dev.csv")


@needs_res
@needs_laps
def test_v2_state_tables_of_all_fifteen_runs_match_recomputation_and_have_sane_values(data):
    d = data
    for h, f in HZ.items():
        ks = _ks(d, f)
        _, table = _frame(d, d["laps"], d["pits"], ks)
        table = table[table["race_id"].isin(d["counted"])]
        a = table.sort_values(["race_id", "driverId"]).reset_index(drop=True)
        for model in V2:
            st = _state_csv("m19v2", h, model)
            assert len(st) == 1311 and st["race_id"].nunique() == 66 and not st.duplicated(["race_id", "driverId"]).any()
            b = st.sort_values(["race_id", "driverId"]).reset_index(drop=True)
            for c in ["k", "laps_done", "out", "run_pos"]:
                assert (a[c].to_numpy() == b[c].to_numpy()).all(), (h, model, c)
            for c in ["pits_made", "gap_ahead_s"] + COLS:
                assert np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), atol=1e-9, rtol=0), (h, model, c)
            assert np.allclose(a["cum_s"].to_numpy(float), b["cum_s"].to_numpy(float), atol=1e-9, equal_nan=True)
        assert not table[COLS].isna().any().any()
        assert table["ir_pos"].between(-0.5, 0.5).all() and table["ir_gap"].between(0, 6).all()
        assert table["ir_out"].isin([0.0, 1.0]).all() and table["ir_def"].between(0, 1).all()
        assert ((table["ir_out"] == 0) == (table["ir_def"] == 0)).all()
        for rid, g in table.groupby("race_id"):
            assert sorted(g["run_pos"]) == list(range(1, len(g) + 1)) and (g["laps_done"] <= g["k"]).all()
            runners = g[g["out"] == 0].sort_values("run_pos")
            assert (np.diff(runners["cum_s"].to_numpy()) >= 0).all()
            if (g["out"] == 1).any():
                assert g[g["out"] == 1]["run_pos"].min() > runners["run_pos"].max()
        if f is None:
            assert (table["k"] == 1).all()
        # the config of every v2 run records the k used
        for model in V2:
            c = load_yaml(_dir("m19v2", h, model) / "config.yaml")
            assert {int(k): int(v) for k, v in c["k_dev_races"].items()} == {r: ks[r] for r in d["counted"]}
            assert c["source"] == "jolpica" and list(c["feature_cols"]) == COLS and c.get("split", "dev") == "dev"


# ====================================================================== switches off reproduce plan 1
@needs_laps
@pytest.mark.slow
@pytest.mark.parametrize("tag,key", [("latent_111", "latent"), ("gbm_100", "gbm")])
def test_inrace_switch_off_reproduces_stored_plan1_dev_runs_exactly(cfg, data, tag, key):
    """Frame with the five in-race columns attached (H5), model built WITHOUT them: scores equal the stored plan-1 dev run."""
    d = data
    eh, _ = _frame(d, d["laps"], d["pits"], _ks(d, 0.25))
    assert set(COLS) <= set(eh.columns)
    fz = load_yaml(ROOT / cfg["frozen_path"])
    c = dict(fz[cfg[key]["key"]][cfg[key]["cell"]])
    p, m = walk_forward(eh, build_model(c, eh), "dev", refit_every=1)
    old_pm, old_pr = U.stored_run(cfg["plan1_leaderboard"], tag)
    mm = p.merge(old_pr, on=["race_id", "driverId"], suffixes=("_n", "_o"))
    assert len(mm) == len(p) == len(old_pr) == 1311
    assert float((mm["score_n"] - mm["score_o"]).abs().max()) == 0.0, tag
    dm, n = U.max_diff(m, old_pm)
    assert n == 66 and dm <= 1e-12


@needs_res
def test_reproduction_check_json_values_and_lam_x_choice_is_the_argmax_of_the_tuning_table(cfg):
    rc = json.loads((RES / "reproduction_check.json").read_text(encoding="utf-8"))
    rep = rc["reproduction"]
    for tag in ("latent_111", "gbm_100"):
        r = rep[tag]
        assert r["max_abs_diff_scores"] == 0.0 and r["max_abs_diff_per_race_metrics"] <= 1e-12 and r["races"] == 66 and r["entries"] == 1311
    assert rep["walk_scores_gbm_100_vs_stored_dev"]["max_abs_diff_scores"] == 0.0
    # NOT bit-exact: the stage-1 walk-forward score of latent_111 differs from the stored plan-1 dev score at the 1e-12 level
    assert 0.0 < rep["walk_scores_latent_111_vs_stored_dev"]["max_abs_diff_scores"] <= 1e-11 and rep["walk_scores_latent_111_vs_stored_dev"]["entries"] == 1311
    assert rep["stage2_latent_no_inrace_columns_fixed_c"]["max_abs_diff_scores_vs_pre_race"] == 0.0
    tu = pd.read_csv(RES / "tuning_lam_x.csv")
    for (h, var), g in tu.groupby(["horizon", "variant"]):
        want = g.loc[g["spearman"].idxmax(), "lam_x"]
        assert rc["lam_x_chosen"][h][var] == want
        model = "latent_s2" if var == "two_stage" else "latent_joint"
        c = load_yaml(_dir("m19v2", h, model) / "config.yaml")["model_config"]
        assert c["lam_x"] == want
        if var == "joint":
            assert c["x_cols"] == COLS
    for h in HZ:
        assert "lam_x" not in load_yaml(_dir("m19v2", h, "gbm_s2") / "config.yaml")["model_config"]          # GBM: frozen, nothing tuned
        assert "lam_x" not in load_yaml(_dir("m19v2", h, "gbm_joint") / "config.yaml")["model_config"]
    assert rc["capped_horizons"]["H4"] == [] and rc["capped_horizons"]["H5"] == [[2021, 12, 11, 1]]


# ====================================================================== tuning budget and split discipline
@needs_res
def test_tuning_used_at_most_30_configs_on_dev_only(cfg):
    tu = pd.read_csv(RES / "tuning_lam_x.csv")
    assert len(tu) == 18 <= 30 and len(tu) == len(cfg["tune_lam_x"]) * 3 * 2
    assert sorted(tu["horizon"].unique()) == ["H4", "H5", "H6"] and sorted(tu["variant"].unique()) == ["joint", "two_stage"]
    assert not tu.duplicated(["horizon", "variant", "lam_x"]).any()
    assert cfg["split"] == "dev" and cfg["bootstrap"] == {"n_boot": 10000, "seed": 0} and cfg["tie_break"] == "driverId"
    assert cfg["tune_refit_every"] == 4 and cfg["refit_every"] == 1 and cfg["min_train_race"] == 20


def test_run_code_scores_only_the_dev_split_and_never_names_test_or_fresh():
    for rel in ("methodologies/M19_inrace/run.py", "src/pitlane/features/inrace.py", "src/pitlane/models/stage2.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                if fn in ("walk_forward", "split_races"):
                    strs = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                    assert strs == ["dev"], (rel, fn)
        consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert not [c for c in consts if len(c) < 40 and re.fullmatch(r"(test|fresh|final|holdout)", c)], rel
    run = (M19 / "run.py").read_text(encoding="utf-8")
    assert re.search(r'laps\["season"\] <= 2023', run) and re.search(r'pits\["season"\] <= 2023', run) and 'assert cfg["split"] == "dev"' in run
    assert "last_race = int(counted[-1])" in run and "walk_scores(entries, lat_cfg, last_race)" in run      # stage-1 scores only up to the last DEV race
    c = (M19 / "config.yaml").read_text(encoding="utf-8")
    assert "fresh" not in c.lower() and "2024" not in c


@needs_laps
def test_no_lap_or_pit_after_2023_is_loaded_or_used_in_the_state_tables(data):
    assert data["laps"]["season"].max() == 2023 and data["pits"]["season"].max() == 2023
    for h in HZ:
        assert set(_state_csv("m19v2", h, "latent_s2")["race_id"]) == set(data["counted"])


# ====================================================================== stored runs
def _dir(prefix, h, m):
    c = sorted(RES.glob(f"*_{prefix}_{h.lower()}_{m}_dev"))
    assert len(c) == 1, (prefix, h, m, c)
    return c[0]


def _all_runs():
    return [("m19", h, m) for h in HZ for m in V1] + [("m19v2", h, m) for h in HZ for m in V2]


@needs_res
@pytest.mark.parametrize("prefix,h,m", _all_runs(), ids=lambda x: str(x))
def test_per_race_metrics_reproduce_from_predictions(prefix, h, m, data):
    d = _dir(prefix, h, m)
    pr = pd.read_parquet(d / "predictions.parquet")
    pm = pd.read_csv(d / "per_race_metrics.csv").set_index("race_id")
    of = pd.read_csv(d / "outcome_free_table.csv").set_index("race_id")
    assert len(pr) == 1311 and len(pm) == 66 and not pr.duplicated(["race_id", "driverId"]).any() and pr["score"].notna().all()
    assert pr["season"].between(2021, 2023).all() and sorted(pr["race_id"].unique()) == sorted(data["counted"]) == sorted(pm.index)
    pd.testing.assert_frame_equal(pm, of, check_exact=True)
    t = A.race_table(U.outcome_free(pr, "driverId")).set_index("race_id")
    for c in [c for c in t.columns if c in pm.columns]:
        assert np.allclose(t.loc[pm.index, c].to_numpy(float), pm[c].to_numpy(float), atol=1e-12, rtol=0, equal_nan=True), (prefix, h, m, c)
    ent = data["entries"].set_index(["race_id", "driverId"])
    j = pr.set_index(["race_id", "driverId"])
    assert (ent.loc[j.index, "pos"].to_numpy() == j["pos"].to_numpy()).all() and (ent.loc[j.index, "classified"].to_numpy() == j["classified"].to_numpy()).all()
    mj = pd.read_json(d / "metrics.json", typ="series")
    for c in ("spearman", "mae", "exact", "spearman_fin", "mae_fin"):
        assert mj[c] == pytest.approx(pm[c].mean(), abs=1e-12)
    assert load_yaml(d / "config.yaml").get("split", "dev") == "dev"


@needs_res
def test_v2_running_order_scores_are_minus_the_running_position_and_unique_per_race():
    for h in HZ:
        d = _dir("m19v2", h, "running_order")
        pr = pd.read_parquet(d / "predictions.parquet")
        st = pd.read_csv(d / "state_table_dev.csv")
        mm = pr.merge(st[["race_id", "driverId", "run_pos"]], on=["race_id", "driverId"])
        assert len(mm) == 1311 and (mm["score"] == -mm["run_pos"].astype(float)).all()
        assert not pr.duplicated(["race_id", "score"]).any()
        assert mm.groupby("race_id")["run_pos"].apply(lambda s: sorted(s) == list(range(1, len(s) + 1))).all()


@needs_res
def test_tie_break_is_driver_id_not_row_order_on_stored_predictions():
    for m in ("latent_s2", "gbm_s2"):
        pr = pd.read_parquet(_dir("m19v2", "H5", m) / "predictions.parquet")
        a = U.outcome_free(pr, "driverId").set_index(["race_id", "driverId"])["score"]
        b = U.outcome_free(pr.sample(frac=1.0, random_state=0), "driverId").set_index(["race_id", "driverId"])["score"].reindex(a.index)
        assert np.array_equal(a.to_numpy(), b.to_numpy())
    x = pd.DataFrame({"race_id": 1, "driverId": ["z", "a", "m"], "score": [1.0, 1.0, 0.0]})
    o = U.outcome_free(x, "driverId").set_index("driverId")["score"]
    assert o["a"] > o["z"] > o["m"]


# ====================================================================== paired gains and verdicts
def _tab(prefix, h, m):
    return A.race_table(U.outcome_free(pd.read_parquet(_dir(prefix, h, m) / "predictions.parquet"), "driverId"))


def _gain(ref, cand, ids=None):
    if ids is not None:
        ref, cand = ref[ref["race_id"].isin(ids)], cand[cand["race_id"].isin(ids)]
    g = A.paired_gain(ref, cand, VERDICT_METRICS, n_boot=10000, seed=0)
    g["verdict"] = ["passes alone" if r["lo"] > 0 else ("worse" if r["hi"] < 0 else "no measurable difference") for _, r in g.iterrows()]
    return g


@needs_res
def test_v2_paired_gains_file_reproduces_from_stored_predictions_seed0_10000_resamples(data):
    pg = pd.read_csv(RES / "paired_gains_dev.csv")
    counted = data["counted"]
    season_of = data["entries"].groupby("race_id")["season"].first()
    short4, monza = [149, 166, 176, 177], [MONZA]
    subsets = {"all_66": None, "excl_4_shortened": [r for r in counted if r not in short4],
               "excl_4_shortened_and_monza": [r for r in counted if r not in short4 + monza]}
    for s in (2021, 2022, 2023):
        subsets[f"season_{s}"] = [r for r in counted if season_of[r] == s]
    exp = []
    for h in HZ:
        t = {m: _tab("m19v2", h, m) for m in V2}
        for model in ("latent_s2", "gbm_s2", "latent_joint", "gbm_joint"):
            for tag, ids in subsets.items():
                exp.append(_gain(t["running_order"], t[model], ids).assign(horizon=h, model=model, subset=tag, reference="running_order"))
        for fam in ("latent", "gbm"):
            exp.append(_gain(t[f"{fam}_s2"], t[f"{fam}_joint"]).assign(horizon=h, model=f"{fam}_joint_vs_two_stage", subset="all_66", reference=f"{fam}_s2"))
        exp.append(_gain(t["latent_s2"], t["gbm_s2"]).assign(horizon=h, model="gbm_vs_latent_two_stage", subset="all_66", reference="latent_s2"))
    exp = pd.concat(exp, ignore_index=True)
    assert len(pg) == len(exp) == 729
    key = ["horizon", "model", "subset", "metric"]
    a, b = pg.set_index(key).sort_index(), exp.set_index(key).sort_index()
    assert a.index.equals(b.index)
    for c in ("ref_mean", "cand_mean", "gain", "lo", "hi", "n"):
        assert np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), atol=1e-12, rtol=0), c
    assert (a["verdict"] == b["verdict"]).all() and (a["reference"] == b["reference"]).all()
    assert pg[pg["subset"] == "all_66"]["n"].eq(66).all() and pg[pg["subset"] == "excl_4_shortened"]["n"].eq(62).all()
    assert pg[pg["subset"] == "excl_4_shortened_and_monza"]["n"].eq(61).all()
    assert (pg["lo"] <= pg["gain"] + 1e-12).all() and (pg["gain"] <= pg["hi"] + 1e-12).all()


@needs_res
def test_v2_verdict_files_are_slices_of_the_gain_table_and_follow_the_ci_rule():
    pg = pd.read_csv(RES / "paired_gains_dev.csv")
    cols = ["horizon", "model", "metric", "ref_mean", "cand_mean", "gain", "lo", "hi", "n", "verdict"]
    for fname, models in (("verdicts.csv", ["latent_s2", "gbm_s2"]), ("verdicts_joint_variant.csv", ["latent_joint", "gbm_joint"])):
        v = pd.read_csv(RES / fname)
        sl = pg[(pg["subset"] == "all_66") & (pg["reference"] == "running_order") & pg["model"].isin(models)]
        assert len(v) == len(sl) == 54 and set(v["model"]) == set(models)
        pd.testing.assert_frame_equal(v[cols].reset_index(drop=True), sl[cols].reset_index(drop=True), check_exact=True)
        for _, r in v.iterrows():
            assert r["verdict"] == ("passes alone" if r["lo"] > 0 else "worse" if r["hi"] < 0 else "no measurable difference")
    v = pd.read_csv(RES / "verdicts.csv")
    for h in HZ:                                        # direct: gain = mean paired difference of the stored per-race metrics
        ref = pd.read_csv(_dir("m19v2", h, "running_order") / "per_race_metrics.csv").set_index("race_id")
        for model in ("latent_s2", "gbm_s2"):
            cand = pd.read_csv(_dir("m19v2", h, model) / "per_race_metrics.csv").set_index("race_id")
            for metric, sign in (("spearman", 1), ("mae", -1), ("exact", 1), ("mae_fin", -1)):
                row = v[(v["horizon"] == h) & (v["model"] == model) & (v["metric"] == metric)].iloc[0]
                assert row["gain"] == pytest.approx(float((sign * (cand[metric] - ref.loc[cand.index, metric])).mean()), abs=1e-12)


# ====================================================================== tuning rerun (H4, joint) and leaderboard
@needs_laps
@needs_res
@pytest.mark.slow
def test_tuning_table_rows_reproduce_for_h4_joint(cfg, data):
    """Re-run the H4 joint tuning (3 lam_x, refit every 4, dev walk). NOT re-run: the two-stage tuning (needs 200 walk-forward stage-1 fits)."""
    d = data
    eh, _ = _frame(d, d["laps"], d["pits"], _ks(d, None))
    fz = load_yaml(ROOT / cfg["frozen_path"])
    lat = dict(fz[cfg["latent"]["key"]][cfg["latent"]["cell"]])
    tu = pd.read_csv(RES / "tuning_lam_x.csv")
    tu = tu[(tu["horizon"] == "H4") & (tu["variant"] == "joint")].set_index("lam_x")
    for lx in cfg["tune_lam_x"]:
        _, per = walk_forward(eh, build_model(dict(lat, x_cols=COLS, lam_x=lx), eh), "dev", refit_every=cfg["tune_refit_every"])
        for c in ("spearman", "mae", "exact"):
            assert per[c].mean() == pytest.approx(tu.loc[lx, c], abs=1e-9), (lx, c)


def _norm(t: str) -> str:
    return t.replace("\r\n", "\n")


@needs_res
def test_leaderboard_existing_rows_identical_to_head_and_all_m19_rows_present_once():
    rel = "results/plan3/leaderboard.csv"
    try:
        old = _norm(subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT, text=True))
    except Exception as e:                                                       # pragma: no cover
        pytest.skip(f"git show failed: {e}")
    o, c = pd.read_csv(StringIO(old), dtype=str), pd.read_csv(StringIO(_norm((ROOT / rel).read_text(encoding="utf-8"))), dtype=str)
    assert list(o.columns) == list(c.columns)
    pd.testing.assert_frame_equal(c.iloc[:len(o)].reset_index(drop=True), o.reset_index(drop=True), check_exact=True)     # old rows byte for byte
    assert not c.duplicated("run_id").any()
    new = c.iloc[len(o):]
    assert new["method"].eq("M19").all(), "rows appended since HEAD must all be M19 rows"
    m19 = c[c["method"] == "M19"]
    assert len(m19) == 39 and m19["split"].eq("dev").all()                       # 9 v1 (superseded) + 15 v2 (superseded by v2b, identical numbers) + 15 v2b
    want = {(f"m19_{h.lower()}_{m}", str(int(CELL[m]))) for h in HZ for m in V1} | {(f"m19v2_{h.lower()}_{m}", str(int(CELL[m]))) for h in HZ for m in V2} | {(f"m19v2b_{h.lower()}_{m}", str(int(CELL[m]))) for h in HZ for m in V2}
    # NB the leaderboard stores the baseline cell "000" as 0 (leading zeros lost), compared as integers
    assert set(zip(m19["label"], m19["cell"].astype(int).astype(str))) == want
    assert len(new) in (30, 39)                                                  # 30 once the v1 rows are committed (R4 commit)


@needs_res
def test_leaderboard_m19_rows_match_their_run_folders():
    lb = pd.read_csv(LB3, dtype={"run_id": str, "cell": str})
    m19 = lb[lb["method"] == "M19"]
    assert len(m19) == 39
    for _, r in m19.iterrows():
        d = ROOT / r["run_dir"].replace("\\", "/")
        assert d.exists() and d.name == r["run_id"]
        pm = pd.read_csv(d / "per_race_metrics.csv")
        assert r["n_races"] == len(pm) == 66
        for c in ("spearman", "mae", "exact", "spearman_fin", "mae_fin", "logloss_points"):
            if c in pm and pd.notna(r[c]):
                assert r[c] == pytest.approx(pm[c].mean(), abs=1e-12), (r["run_id"], c)
        assert load_yaml(d / "config.yaml").get("split", "dev") == "dev"
    for rel in ("results/leaderboard.csv", "results/plan2/leaderboard.csv"):
        assert not (pd.read_csv(ROOT / rel, dtype=str)["method"] == "M19").any(), rel


@needs_laps
@needs_res
@pytest.mark.slow
@pytest.mark.parametrize("tag,key,tol", [("latent_111", "latent", 1e-9), ("gbm_100", "gbm", 0.0)])
def test_walk_forward_pre_race_scores_equal_stored_plan1_dev_scores(cfg, data, tag, key, tol):
    """Stage-1 scores from stage2.walk_scores (refit before every race on all earlier races, one warm-started model object) equal the
    stored plan-1 dev scores: latent to 1e-9 (the stored json shows ~4e-12, i.e. not bit-exact), gbm exactly. Also: stage-2 with no
    in-race columns and c fixed to 1 returns that score exactly through the harness."""
    d = data
    ent = d["entries"]
    fz = load_yaml(ROOT / cfg["frozen_path"])
    c = dict(fz[cfg[key]["key"]][cfg[key]["cell"]])
    last = int(d["counted"][-1])
    sc = S2.walk_scores(ent, c, last)
    dev = ent["race_id"].isin(d["counted"]).to_numpy()
    new = ent.loc[dev, ["race_id", "driverId"]].assign(new=sc[dev])
    _, old = U.stored_run(cfg["plan1_leaderboard"], tag)
    m = new.merge(old[["race_id", "driverId", "score"]], on=["race_id", "driverId"])
    assert len(m) == 1311 and m["new"].notna().all()
    assert float((m["new"] - m["score"]).abs().max()) <= tol
    if key == "latent":
        e2 = ent.assign(pre_lat=sc)
        p, _ = walk_forward(e2, S2.Stage2Latent("pre_lat", [], fix_c=True, min_race=int(cfg["min_train_race"])), "dev", refit_every=1)
        mm = p.merge(e2[["race_id", "driverId", "pre_lat"]], on=["race_id", "driverId"])
        assert len(mm) == 1311 and (mm["score"] == mm["pre_lat"]).all()
        assert np.isnan(sc[ent["race_id"].to_numpy() > last]).all()
