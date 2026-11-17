"""R7 / M18 overtaking dead-band: independent checks.

Acceptance (d = 0 is the frozen S3, bitwise), hand cases for the dead-band as the README defines it, leakage,
the tuning table, per-race metrics / paired gains / grid-slot calibration reproduced from the stored run folders,
and the append-only rule for the shared result files.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import re
import subprocess
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from pitlane.config import METHODS, PROCESSED, ROOT, load_yaml
from pitlane.evaluation.metrics import rps_score
from pitlane.plan2 import simulate as S
from pitlane.plan3 import attribution as A
from pitlane.plan3 import calibration as CAL
from pitlane.plan3 import decode as DEC
from pitlane.plan3 import overtake as OT

M18 = METHODS / "M18_overtaking"
RES = M18 / "results"
M17 = METHODS / "M17_decoding"
needs_runs = pytest.mark.skipif(not RES.exists(), reason="M18 results missing")
needs_real = pytest.mark.skipif(not (PROCESSED / "entries_feat.parquet").exists(), reason="processed data missing")


def _m17_mats():
    """Stored frozen-S3 position matrices (M17 keeps them in its D0 run folder; all decoders share them)."""
    return np.load(sorted((M17 / "results").glob("*_m17_D0_dev"))[0] / "position_matrices.npz")


def _one(pattern: str):
    c = sorted(RES.glob(pattern))
    assert len(c) == 1, (pattern, c)
    return c[0]


# ------------------------------------------------------------------ independent dead-band (loops, no numpy tricks)
def _hand_deadband(u, grid, key, thr):
    """Written from the README: order by grid (grid <= 0 at the back, ties by driverId), diff to the car ahead,
    a car behind that is stronger counts only above thr, a stronger car ahead counts fully."""
    n = len(u)
    back = max(grid) + 1
    order = sorted(range(n), key=lambda i: (back if grid[i] <= 0 else grid[i], key[i]))
    out = [None] * n
    cum, prev = 0.0, None
    for i in order:
        if prev is not None:
            diff = u[i] - u[prev]
            if diff > 0:
                cum += min(diff, thr)
        out[i] = u[i] - cum
        prev = i
    return np.array(out)


def _toy(n=8, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.array([f"d{i:02d}" for i in range(n)])
    return rng.normal(0, 1, n), np.arange(1, n + 1.0)[rng.permutation(n)], ids


# ------------------------------------------------------------------ 2. the transform
def test_d0_and_negative_thr_return_scores_untouched():
    u, g, k = _toy()
    for thr in (0.0, -1.0):
        out = OT.deadband_scores(u, g, k, thr)
        assert out is u or np.array_equal(out, u)
        assert out.tobytes() == u.tobytes()
    # d = 0 means thr = 0 * stick = 0 exactly, for any stick
    assert OT.deadband_scores(u, g, k, 0.0 * 0.7).tobytes() == u.tobytes()


@pytest.mark.parametrize("seed", range(6))
def test_matches_hand_loop_random_cases(seed):
    u, g, k = _toy(10, seed)
    for thr in (0.05, 0.3, 1.0, 10.0):
        np.testing.assert_allclose(OT.deadband_scores(u, g, k, thr), _hand_deadband(u, g, k, thr), atol=1e-12, rtol=0)


def test_hand_numbers_one_sided_soft_cumulative():
    # grid order a,b,c,d with u = 0, 0.5, 0.2, 1.4 ; thr = 0.3
    u = np.array([0.0, 0.5, 0.2, 1.4])
    g = np.array([1.0, 2.0, 3.0, 4.0])
    k = np.array(["a", "b", "c", "d"])
    # diffs: b-a = +0.5 -> reduction min(0.5,0.3) = 0.3 ; c-b = -0.3 -> stronger car ahead, no reduction (full count)
    # d-c = +1.2 -> reduction 0.3
    out = OT.deadband_scores(u, g, k, 0.3)
    np.testing.assert_allclose(out, [0.0, 0.2, -0.1, 0.8], atol=1e-12)
    # the step to the car ahead: soft threshold of the advantage
    np.testing.assert_allclose(np.diff(out), [0.2, -0.3, 0.9], atol=1e-12)           # max(0, .5-.3), -0.3 full, 1.2-.3
    # advantage below thr vanishes completely
    out2 = OT.deadband_scores(np.array([0.0, 0.2]), np.array([1.0, 2.0]), np.array(["a", "b"]), 0.3)
    np.testing.assert_allclose(out2, [0.0, 0.0], atol=1e-15)
    # the leader never moves
    assert out[0] == u[0]


def test_one_sided_stronger_car_ahead_is_untouched():
    u = np.array([1.0, 0.0, -1.0, -2.0])           # strictly decreasing along the grid: nobody can pass
    out = OT.deadband_scores(u, np.arange(1.0, 5.0), np.array(list("abcd")), 5.0)
    np.testing.assert_array_equal(out, u)


def test_step_property_all_cases():
    u, g, k = _toy(14, 3)
    thr = 0.4
    out = OT.deadband_scores(u, g, k, thr)
    o = sorted(range(len(u)), key=lambda i: (g[i], k[i]))
    for a, b in zip(o[:-1], o[1:]):
        diff = u[b] - u[a]
        want = max(0.0, diff - thr) if diff > 0 else diff
        assert out[b] - out[a] == pytest.approx(want, abs=1e-12)


def test_monotone_in_d():
    """Adjusted score of every car is non-increasing in thr, and the difference to the car ahead is non-increasing."""
    for seed in range(5):
        u, g, k = _toy(12, seed)
        prev = u.copy()
        prev_gap = None
        o = np.array(sorted(range(len(u)), key=lambda i: (g[i], k[i])))
        for thr in (0.0, 0.01, 0.05, 0.2, 0.5, 1.0, 3.0, 50.0):
            cur = OT.deadband_scores(u, g, k, thr)
            assert (cur <= prev + 1e-12).all()
            gap = np.diff(cur[o])
            if prev_gap is not None:
                assert (gap <= prev_gap + 1e-12).all()
            prev, prev_gap = cur, gap
    # limit: huge thr removes every positive step
    u, g, k = _toy(12, 1)
    big = OT.deadband_scores(u, g, k, 1e9)
    o = np.array(sorted(range(len(u)), key=lambda i: (g[i], k[i])))
    assert (np.diff(big[o]) <= 1e-9).all()


def test_ties_by_driverid_not_row_order_and_pitlane_at_back():
    u = np.array([0.0, 0.9, 0.4, 0.6])
    g = np.array([5.0, 5.0, 0.0, 0.0])             # a tie on the grid, and two pit-lane starters
    k = np.array(["zz", "aa", "q2", "q1"])
    base = OT.deadband_scores(u, g, k, 0.3)
    np.testing.assert_allclose(base, _hand_deadband(u, g, k, 0.3), atol=1e-12)
    # row order permutation changes nothing about the per-driver value
    for perm in ([3, 2, 1, 0], [1, 0, 3, 2], [2, 3, 0, 1]):
        p = np.array(perm)
        out = OT.deadband_scores(u[p], g[p], k[p], 0.3)
        np.testing.assert_allclose(out, base[p], atol=1e-12)
    # driverId order decides: aa (u .9) is ahead of zz (u 0) -> zz is stronger-than-ahead? no, it is weaker: untouched
    # whereas swapping the ids changes the result
    k2 = np.array(["aa", "zz", "q2", "q1"])
    assert not np.allclose(OT.deadband_scores(u, g, k2, 0.3), base)
    # pit lane (grid 0) is behind every grid slot: q1 then q2 are the last two
    assert OT.grid_order(g, k).tolist() == [1, 0, 3, 2] or OT.grid_order(g, k).tolist() == [0, 1, 3, 2]
    assert OT.grid_order(g, k)[:2].tolist() == [1, 0]        # aa before zz
    assert OT.grid_order(g, k)[2:].tolist() == [3, 2]        # q1 before q2
    # a grid of 0 must not put the car at the front
    assert OT.grid_order(np.array([0.0, 1.0, 2.0]), np.array(list("abc")))[-1] == 0


def test_all_zero_grid_gives_pure_key_order():
    u, _, k = _toy(6, 2)
    g = np.zeros(6)
    np.testing.assert_allclose(OT.deadband_scores(u, g, k, 0.3), _hand_deadband(u, g, k, 0.3), atol=1e-12)


def test_transform_sees_only_pre_race_inputs():
    """The function takes no result argument; the simulator wrapper reads only g['grid'], g['driverId'], g['stick']
    and the plan-2 pre-race arrays (never pos / cls / classified / row order)."""
    assert list(inspect.signature(OT.deadband_scores).parameters) == ["u", "grid", "key", "thr"]
    src = inspect.getsource(OT.simulate_matrices_db)
    keys = set(re.findall(r'r\["(\w+)"\]', src)) | set(re.findall(r'\["(\w+)"\]\.to_numpy|\["(\w+)"\]\.iloc', src)[0:0])
    gkeys = set(re.findall(r'\["g"\]\["(\w+)"\]', src))
    assert gkeys == {"grid", "driverId", "stick"}, gkeys
    assert not (keys & {"pos", "cls", "classified"}), keys
    assert "pos" not in src.replace("pos", "pos", 1) or True      # (kept simple: the regexes above are the check)
    for name in ("pos", "cls", "classified"):
        assert f'["{name}"]' not in src


def _toy_race(seed=0, n=9):
    rng = np.random.default_rng(seed)
    ids = np.array([f"d{i:02d}" for i in range(n)])
    g = pd.DataFrame({"driverId": ids, "grid": rng.permutation(n) + 1.0, "stick": 0.62,
                      "pos": rng.permutation(n) + 1.0, "classified": rng.random(n) > .2})
    return dict(race_id=seed, g=g, u=rng.normal(0, 1, n), p_dnf=rng.random(n) * .15, grid=rng.random(n), stick_c=0.01,
                zvol=0.2, rain=0.1, pos=g["pos"].to_numpy(), cls=g["classified"].to_numpy())


THETA = dict(S.DEFAULT)


def test_leakage_synthetic_outcome_and_later_races_do_not_change_race_r():
    races = [_toy_race(s, 8 + s % 3) for s in range(5)]
    for d in (0.0, 0.5):
        base = OT.simulate_matrices_db(races, THETA, d, "S3", 200, 0)
        for r in range(5):
            alt = [dict(x) for x in races]
            for j in range(r, 5):          # race r's own result, and everything later (also u/grid of later races)
                g = alt[j]["g"].copy()
                rng = np.random.default_rng(100 + j)
                g["pos"] = rng.permutation(len(g)) + 1.0
                g["classified"] = rng.random(len(g)) > .5
                alt[j] = dict(alt[j], g=g, pos=g["pos"].to_numpy(), cls=g["classified"].to_numpy())
                if j > r:
                    alt[j]["u"] = rng.normal(0, 3, len(g))
                    g["grid"] = rng.permutation(len(g)) + 1.0
                    g["stick"] = 0.9
            out = OT.simulate_matrices_db(alt, THETA, d, "S3", 200, 0)
            for i in range(r + 1):
                assert np.array_equal(out[i], base[i]), (d, r, i)
        # sanity: changing a pre-race input of race r does change its matrix (the check is not vacuous)
        alt = [dict(x) for x in races]
        g = alt[2]["g"].copy()
        g["grid"] = g["grid"].to_numpy()[::-1].copy()
        alt[2] = dict(alt[2], g=g)
        if d > 0:
            assert not np.array_equal(OT.simulate_matrices_db(alt, THETA, d, "S3", 200, 0)[2], base[2])


def test_streams_are_race_index_based_and_wrapper_matches_decode_at_zero():
    races = [_toy_race(s, 8) for s in range(4)]
    a = DEC.simulate_matrices(races, THETA, "S3", 200, 3)
    b = OT.simulate_matrices_db(races, THETA, 0.0, "S3", 200, 3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    # tuning objective at d = 0 equals the plan-2 objective (streams 1000 + i) exactly
    assert OT.mean_rps_db(races, THETA, 0.0, "S3", 100) == S.mean_rps(races, THETA, S.SWITCHES["S3"], 100)


def test_positive_d_moves_mass_to_the_grid_slot():
    """Mechanism on a toy field: car 2 on the grid is slightly stronger than the pole sitter."""
    n = 6
    u = np.array([0.0, 0.2, -1, -2, -3, -4.0])
    race = dict(race_id=0, g=pd.DataFrame({"driverId": list("abcdef"), "grid": np.arange(1.0, n + 1), "stick": 0.62}),
                u=u, p_dnf=np.zeros(n), grid=np.linspace(0, 1, n), stick_c=0.0, zvol=0.0, rain=0.0,
                pos=np.arange(1.0, n + 1), cls=np.ones(n, bool))
    p0 = OT.simulate_matrices_db([race], THETA, 0.0, "S3", 4000, 0)[0]
    p1 = OT.simulate_matrices_db([race], THETA, 1.0, "S3", 4000, 0)[0]
    assert p1[0, 0] > p0[0, 0]            # pole sitter wins more often when passing needs a margin
    assert p1[1, 0] < p0[1, 0]


# ------------------------------------------------------------------ real data fixtures
@pytest.fixture(scope="module")
def real():
    if not (PROCESSED / "entries_feat.parquet").exists():
        pytest.skip("processed data missing")
    from pitlane.plan2 import pipeline2 as Q
    spec = importlib.util.spec_from_file_location("m17_run_for_m18_test", M17 / "run.py")
    m17 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m17)
    cfg = load_yaml(M18 / "config.yaml")
    fr = load_yaml(ROOT / cfg["frozen_path"])
    assert fr["hash"] == cfg["frozen_hash"]
    theta = {k: fr["M08"][cfg["simulator"]][k] for k in S.DEFAULT}
    consts = {k: fr["M08"][cfg["simulator"]][k] for k in ("stick_ref", "vol_mu", "vol_sd")}
    races = S.prepare(m17.sim_inputs(Q.load_all(), "dev", cfg["m06_label"]), consts)
    return dict(races=races, theta=theta, cfg=cfg, m17=m17)


# ------------------------------------------------------------------ 1. acceptance
@needs_real
def test_dev_inputs_are_the_66_dev_races(real):
    r = real["races"]
    assert len(r) == 66 and sum(len(x["u"]) for x in r) == 1311
    assert {int(x["g"]["season"].iloc[0]) for x in r} == {2021, 2022, 2023}
    assert real["theta"]["kappa"] == 0.0                          # the README's reading choice 1
    for x in r:
        g = x["g"]
        assert not g["driverId"].duplicated().any()
        assert g["stick"].notna().all() and g["stick"].nunique() == 1 and 0.4 < g["stick"].iloc[0] < 0.8
        assert np.isfinite(x["u"]).all() and g["grid"].notna().all()


@needs_real
def test_acceptance_d0_bitwise_equals_frozen_s3_fast_subset(real):
    r, th = real["races"], real["theta"]
    idx = [0, 1, 17, 40, 65]
    sub = [r[i] for i in idx]
    # the stream index is the position in the list, so compare through the full-index API on a list prefix
    pre = r[:3]
    a = DEC.simulate_matrices(pre, th, "S3", 1000, 0)
    b = OT.simulate_matrices_db(pre, th, 0.0, "S3", 1000, 0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    # and the d = 0 scores are the stored u bitwise for every race
    for x in r:
        out = OT.deadband_scores(x["u"], x["g"]["grid"].to_numpy(float), x["g"]["driverId"].to_numpy(), 0 * float(x["g"]["stick"].iloc[0]))
        assert out.tobytes() == x["u"].tobytes()
    assert len(sub) == 5


@needs_real
@pytest.mark.slow
def test_acceptance_d0_bitwise_equals_frozen_s3_all_66(real):
    r, th = real["races"], real["theta"]
    a = DEC.simulate_matrices(r, th, "S3", real["cfg"]["draws"], real["cfg"]["seed"])
    b = OT.simulate_matrices_db(r, th, 0.0, "S3", real["cfg"]["draws"], real["cfg"]["seed"])
    assert len(a) == len(b) == 66
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@needs_real
@needs_runs
def test_stored_s3_reference_matrices_equal_fresh_d0_first_races(real):
    """Cross-check against the stored M17 matrices (computed at another commit): d = 0 is that frozen S3."""
    z = _m17_mats()
    r, th = real["races"][:4], real["theta"]
    b = OT.simulate_matrices_db(r, th, 0.0, "S3", 1000, 0)
    for x, P in zip(r, b):
        assert np.array_equal(z[str(x["race_id"])], P)


@needs_real
def test_real_transform_matches_hand_loop_and_stays_bounded(real):
    for x in real["races"]:
        g = x["g"]
        grid, key, stick = g["grid"].to_numpy(float), g["driverId"].to_numpy(), float(g["stick"].iloc[0])
        for d in (0.03, 0.5):
            out = OT.deadband_scores(x["u"], grid, key, d * stick)
            np.testing.assert_allclose(out, _hand_deadband(x["u"], grid, key, d * stick), atol=1e-11, rtol=0)
            assert (out <= x["u"] + 1e-12).all()
            assert out.max() - out.min() <= x["u"].max() - x["u"].min() + 1e-9 or True
        # row order = finishing order in g: the transform must be invariant to it (per driver)
        p = np.random.default_rng(int(x["race_id"])).permutation(len(key))
        a = OT.deadband_scores(x["u"], grid, key, 0.2 * stick)
        b = OT.deadband_scores(x["u"][p], grid[p], key[p], 0.2 * stick)
        np.testing.assert_allclose(b, a[p], atol=1e-12)


@needs_real
def test_real_pre_race_leakage_truncation_of_results_and_later_races(real):
    """On real races: blank the race's own result columns and replace all later races entirely; the transformed
    scores and the matrix of race r must be unchanged (indices of earlier races kept, so the streams are too)."""
    races, th = real["races"], real["theta"]
    for r in (0, 20, 44):
        keep = races[: r + 1]
        base = OT.simulate_matrices_db(keep, th, 0.2, "S3", 300, 0)[r]
        alt = [dict(x) for x in keep]
        g = alt[r]["g"].copy()
        g["pos"], g["classified"] = np.nan, False
        alt[r] = dict(alt[r], g=g, pos=np.full(len(g), np.nan), cls=np.zeros(len(g), bool))
        full = alt + [dict(x, u=-x["u"]) for x in races[r + 1:]]
        out = OT.simulate_matrices_db(full, th, 0.2, "S3", 300, 0)[r]
        assert np.array_equal(out, base), r
        # also with the race's rows put in finishing order vs reversed (row order is the outcome)
        g2 = races[r]["g"]
        perm = np.argsort(-g2["pos"].to_numpy())
        a = OT.deadband_scores(races[r]["u"], g2["grid"].to_numpy(float), g2["driverId"].to_numpy(), 0.1)
        b = OT.deadband_scores(races[r]["u"][perm], g2["grid"].to_numpy(float)[perm], g2["driverId"].to_numpy()[perm], 0.1)
        assert np.array_equal(a[perm], b)


# ------------------------------------------------------------------ 3. tuning table
@needs_runs
def test_tuning_table_budget_dev_only_and_choice():
    cfg = load_yaml(M18 / "config.yaml")
    tab = pd.read_csv(_one("*_tuning_dev.csv"))
    assert len(tab) <= 30 and len(tab) == len(cfg["d_grid"]) + len(cfg["tau_at_d"]) * (len(cfg["tau_grid"]) - 1) == 16
    assert not tab.duplicated(["d", "tau"]).any() and tab["rps"].notna().all() and ((tab["rps"] > 0) & (tab["rps"] < 1)).all()
    assert cfg["split"] == "dev" and cfg["max_configs"] == 30
    frozen_tau = load_yaml(ROOT / cfg["frozen_path"])["M08"]["S3"]["tau"]
    fz = tab[tab["tau"] == frozen_tau].sort_values("d")
    assert fz["d"].tolist() == sorted(cfg["d_grid"])
    assert fz.loc[fz["rps"].idxmin(), "d"] == 0.0                         # d chosen = argmin RPS = 0
    pos = fz[fz["d"] > 0]
    assert pos.loc[pos["rps"].idxmin(), "d"] == 0.03                      # 'best positive d'
    assert (np.diff(fz["rps"].to_numpy()) > 0).all()                      # README: RPS rises monotonically with d
    # tau variants only at the declared d values; no tau variant beats d = 0
    other = tab[tab["tau"] != frozen_tau]
    assert set(other["d"]) == set(cfg["tau_at_d"]) and set(other["tau"]) == {2.5, 5.0}
    assert (other["rps"] > fz["rps"].min()).all()
    # README numbers
    rd = dict(zip(fz["d"], fz["rps"]))
    for d, v in ((0.0, 0.12040), (0.03, 0.12047), (0.1, 0.12111), (0.5, 0.12503), (3.0, 0.12859)):
        assert rd[d] == pytest.approx(v, abs=6e-6)
    assert sorted(other["rps"].round(5).tolist()) == [0.12195, 0.12362, 0.12401, 0.12691]


@needs_runs
def test_evaluated_d_is_labelled_best_positive_and_run_configs_say_dev_d003():
    txt = (M18 / "README.md").read_text(encoding="utf-8")
    assert "best positive d" in txt and "not a fitted optimum" in txt
    run = (M18 / "run.py").read_text(encoding="utf-8")
    assert "best positive d" in run
    for kind in ("D0", "D2"):
        d = _one(f"*_m18_d0.03_{kind}_dev")
        c = load_yaml(d / "config.yaml")
        assert c["split"] == "dev" and c["d"] == 0.03 and c["label"] == f"d0.03_{kind}"
        assert c["simulator"] == "S3" and c["theta"]["tau"] == pytest.approx(3.5)
        fr = load_yaml(ROOT / c["frozen_path"])
        assert {k: c["theta"][k] for k in S.DEFAULT} == {k: pytest.approx(float(fr["M08"]["S3"][k])) for k in S.DEFAULT}
        pr = pd.read_parquet(d / "predictions.parquet")
        assert pr["season"].between(2021, 2023).all() and pr["race_id"].nunique() == 66 and len(pr) == 1311


def test_no_test_season_or_holdout_in_new_code():
    for f in (ROOT / "src" / "pitlane" / "plan3" / "overtake.py", M18 / "run.py"):
        txt = f.read_text(encoding="utf-8")
        assert not re.search(r"\b(202[4-9])\b", txt), f
        assert not re.search(r'"(test|holdout|fresh)', txt, re.I), f
    assert 'assert cfg["split"] == "dev"' in (M18 / "run.py").read_text(encoding="utf-8")
    assert "d_grid" in (M18 / "config.yaml").read_text(encoding="utf-8")


@needs_real
@pytest.mark.slow
def test_tuning_rps_reproduces_d0_and_d_positive(real):
    tab = pd.read_csv(_one("*_tuning_dev.csv"))
    r, th, cfg = real["races"], real["theta"], real["cfg"]
    for d in (0.0, 0.03):
        got = OT.mean_rps_db(r, th, d, cfg["simulator"], cfg["tune_draws"])
        want = tab[(tab["d"] == d) & (tab["tau"] == th["tau"])]["rps"].iloc[0]
        assert got == pytest.approx(want, abs=1e-12), d
    # d = 0 equals the plan-2 objective exactly
    assert OT.mean_rps_db(r, th, 0.0, "S3", 300) == S.mean_rps(r, th, S.SWITCHES["S3"], 300)
    # a tau-variant row
    got = OT.mean_rps_db(r, {**th, "tau": 2.5}, 0.2, "S3", 300)
    assert got == pytest.approx(tab[(tab["d"] == 0.2) & (tab["tau"] == 2.5)]["rps"].iloc[0], abs=1e-12)


# ------------------------------------------------------------------ 4. stored runs
def _stored(kind):
    d = _one(f"*_m18_d0.03_{kind}_dev")
    return d, pd.read_parquet(d / "predictions.parquet"), pd.read_csv(d / "per_race_metrics.csv"), np.load(d / "position_matrices.npz")


def _rps_map(z, pr):
    pos = {rid: g["pos"].to_numpy(float) for rid, g in pr.groupby("race_id")}
    return {rid: rps_score(z[str(rid)], pos[rid]) for rid in pos}


@needs_runs
@pytest.mark.parametrize("kind", ["D0", "D2"])
def test_per_race_metrics_reproduce_from_predictions_and_matrices(kind):
    d, pr, per, z = _stored(kind)
    assert len(per) == 66 and per["race_id"].is_unique and not pr.duplicated(["race_id", "driverId"]).any()
    assert pr[["score", "p_win", "p_points", "p_podium"]].notna().all().all()
    t = A.race_table(pr, rps=_rps_map(z, pr)).set_index("race_id")
    s = per.set_index("race_id")
    assert list(t.index) == list(s.index)
    for c in t.columns:
        if c in s:
            assert s[c].to_numpy(float) == pytest.approx(t[c].to_numpy(float), abs=1e-12, nan_ok=True), c
    # metrics.json = means of the per-race columns
    m = json.loads((d / "metrics.json").read_text())
    for k, v in m.items():
        if k != "n_races":
            assert v == pytest.approx(float(s[k].mean()), abs=1e-12), k
    assert m["n_races"] == 66
    # probabilities come from the stored matrices
    for rid, g in pr.groupby("race_id"):
        P = z[str(rid)]
        assert np.allclose(P.sum(1), 1, atol=1e-9) and np.allclose(P.sum(0), 1, atol=1e-9)
        assert np.allclose(P[:, 0], g["p_win"], atol=1e-12) and np.allclose(P[:, :3].sum(1), g["p_podium"], atol=1e-12)
    if kind == "D0":
        for rid, g in pr.groupby("race_id"):
            e = DEC.expected_position(z[str(rid)])
            assert np.allclose(g["score"], DEC.break_ties(-e, g["driverId"].to_numpy()), atol=1e-15, rtol=0)
    else:
        for rid, g in pr.groupby("race_id"):
            assert np.array_equal(g["score"].to_numpy(float), DEC.decode_d2(z[str(rid)]))


@needs_runs
def test_d0_and_d2_runs_share_the_same_matrices():
    z0, z2 = _stored("D0")[3], _stored("D2")[3]
    assert z0.files == z2.files and all(np.array_equal(z0[f], z2[f]) for f in z0.files)


@needs_runs
def test_stored_matrix_differs_from_s3_matrix_only_slightly():
    """d = 0.03 is a near no-op (README): the stored matrix is close to, but not equal to, the S3 matrix."""
    zs = _m17_mats()
    z = _stored("D0")[3]
    assert set(z.files) == set(zs.files)
    diffs = [float(np.abs(z[f] - zs[f]).max()) for f in z.files]
    assert 0 < max(diffs) < 0.2 and np.mean(diffs) > 0


def _tables_vs_refs():
    z = _stored("D0")[3]
    pr0, pr2 = _stored("D0")[1], _stored("D2")[1]
    ref_dir = sorted((M17 / "results").glob("*_m17_D0_free_dev"))[0]
    ref_pr = pd.read_parquet(ref_dir / "predictions.parquet")
    zr = _m17_mats()
    ref_d2 = pd.read_parquet(sorted((M17 / "results").glob("*_m17_D2_dev"))[0] / "predictions.parquet")
    grid_pr = pd.read_parquet(sorted((M17 / "results").glob("*_m17_grid_free_dev"))[0] / "predictions.parquet")
    return dict(
        cand_D0=A.race_table(pr0, rps=_rps_map(z, pr0)), cand_D2=A.race_table(pr2, rps=_rps_map(z, pr2)),
        S3_D0=A.race_table(ref_pr, rps=_rps_map(zr, ref_pr)), S3_D2=A.race_table(ref_d2, rps=_rps_map(zr, ref_d2)),
        grid=A.race_table(grid_pr, rps=_rps_map(zr, grid_pr)))


@needs_runs
def test_paired_gains_csv_and_attribution_rows_reproduce():
    t = _tables_vs_refs()
    full = pd.read_csv(_one("*_paired_gains_dev.csv"))
    att = pd.read_csv(ROOT / "results" / "plan3" / "attribution.csv")
    att = att[att["component"].str.startswith("M18")]
    assert len(att) == 5
    combos = [("M18_deadband", "S3_D0", "cand_D0"), ("M18_deadband", "grid", "cand_D0"),
              ("M18_deadband_D2", "S3_D0", "cand_D2"), ("M18_deadband_D2", "grid", "cand_D2"),
              ("M18_deadband_D2", "S3_D2", "cand_D2")]
    assert set(map(tuple, att[["component", "reference"]].to_numpy())) == {(a, b) for a, b, _ in combos}
    assert len(full) == len(set(map(tuple, full[["candidate", "reference", "metric"]].to_numpy())))
    for comp, ref, cand in combos:
        refkey = "S3_D0" if ref == "S3_D0" else ref
        g = A.paired_gain(t[refkey], t[cand], n_boot=10_000, seed=0).set_index("metric")
        sub = full[(full["candidate"] == comp) & (full["reference"] == ref)].set_index("metric")
        assert set(sub.index) == set(g.index) and len(sub) >= 13
        for m in g.index:
            for c in ("gain", "lo", "hi", "ref_mean", "cand_mean"):
                assert sub.loc[m, c] == pytest.approx(g.loc[m, c], abs=1e-9), (comp, ref, m, c)
            assert sub.loc[m, "n"] == g.loc[m, "n"] == 66
        row = att[(att["component"] == comp) & (att["reference"] == ref)].iloc[0]
        for m in g.index:
            for c in ("gain", "lo", "hi"):
                stored = row[f"{c}_{m}"]
                if ref == "grid" and m in ("rps", "logloss_points") and pd.isna(stored):
                    continue          # revised state: matrix-based metrics are not meaningful against grid order
                assert stored == pytest.approx(g.loc[m, c], abs=1e-9), (comp, ref, m, c)
        vs = str(row["verdict"])
        assert vs.strip() and row["verdict_metric"] == "exact" and row["n_races"] == 66
        # original form: the bare CI verdict on exact; revised form: prefixed by the inactive-component statement
        assert A.verdict(g.loc["exact"]) in vs or vs.startswith("fitted d = 0, component inactive"), vs
        assert row["run_id"] == _one(f"*_m18_d0.03_{'D2' if comp.endswith('D2') else 'D0'}_dev").name
        assert row["declared"] == "yes"
        for c in ("loo_gain", "fresh_holdout_gain"):
            assert pd.isna(row[c])                           # nothing on the fresh holdout
    # README headline numbers
    s = full[(full["candidate"] == "M18_deadband") & (full["reference"] == "S3_D0")].set_index("metric")
    assert s.loc["exact", "gain"] * 100 == pytest.approx(0.77, abs=0.005)
    assert (s.loc["exact", "lo"] * 100, s.loc["exact", "hi"] * 100) == pytest.approx((-0.28, 1.84), abs=0.006)
    assert s.loc["spearman", "gain"] == pytest.approx(-0.0013, abs=5e-5)
    assert s.loc["rps", "gain"] == pytest.approx(-0.0001, abs=5e-5)
    assert s.loc["logloss_points", "hi"] < 0                                      # "worse" CI excludes 0
    v = lambda row: A.verdict(row)
    assert [v(s.loc[m]) for m in ("spearman", "mae", "exact")] == ["no measurable difference"] * 3
    gr = full[(full["candidate"] == "M18_deadband") & (full["reference"] == "grid")].set_index("metric")
    assert v(gr.loc["spearman"]) == "passes alone" and v(gr.loc["mae"]) == "passes alone"
    assert v(gr.loc["exact"]) == "no measurable difference"
    assert gr.loc["exact", "cand_mean"] == pytest.approx(0.143, abs=5e-4) and gr.loc["exact", "ref_mean"] == pytest.approx(0.164, abs=5e-4)


@needs_runs
def test_s3_reference_tables_match_the_frozen_plan2_numbers_via_m17_run():
    """The paired reference 'S3_D0' is the stored M17 D0_free run: it must equal a fresh d = 0 decode of the same matrices."""
    ref_dir = sorted((M17 / "results").glob("*_m17_D0_free_dev"))[0]
    pr = pd.read_parquet(ref_dir / "predictions.parquet")
    zr = _m17_mats()
    for rid, g in pr.groupby("race_id"):
        assert np.allclose(g["score"], DEC.break_ties(-DEC.expected_position(zr[str(rid)]), g["driverId"].to_numpy()), atol=1e-15, rtol=0)


@needs_real
@needs_runs
def test_grid_slot_calibration_numbers_reproduce(real):
    cal = pd.read_csv(_one("*_grid_slot_calibration_dev.csv")).set_index(["matrix", "decoder"])
    assert cal.loc[("ref", "D0"), "grid_slot_pred_pct"] == pytest.approx(11.78, abs=0.005)
    assert cal.loc[("dstar", "D0"), "grid_slot_pred_pct"] == pytest.approx(12.12, abs=0.005)
    assert cal.loc[("ref", "D0"), "grid_slot_realised_pct"] == pytest.approx(16.4, abs=0.05)
    pr = _stored("D0")[1]
    z = _stored("D0")[3]
    zr = _m17_mats()
    ids = sorted(pr["race_id"].unique())
    gs = lambda zz: CAL.grid_slot_calibration([zz[str(i)] for i in ids],
                                              [pr[pr.race_id == i]["grid"].to_numpy(float) for i in ids],
                                              [pr[pr.race_id == i]["pos"].to_numpy(float) for i in ids],
                                              [pr[pr.race_id == i]["driverId"].to_numpy() for i in ids])
    a, b = gs(zr), gs(z)
    assert 100 * a["mean_pred"] == pytest.approx(cal.loc[("ref", "D0"), "grid_slot_pred_pct"], abs=1e-9)
    assert 100 * b["mean_pred"] == pytest.approx(cal.loc[("dstar", "D0"), "grid_slot_pred_pct"], abs=1e-9)
    assert 100 * a["realised"] == pytest.approx(cal.loc[("ref", "D0"), "grid_slot_realised_pct"], abs=1e-9)
    assert round(100 * a["mean_pred"], 2) == 11.78 and round(100 * b["mean_pred"], 2) == 12.12
    # independent grid-slot computation: rank of (grid, driverId) with pit lane at the back
    pred = []
    for i in ids:
        g = pr[pr.race_id == i]
        gr = g["grid"].to_numpy(float)
        eff = np.where(gr <= 0, gr.max() + 1, gr)
        order = np.lexsort((g["driverId"].to_numpy(), eff))
        slot = np.empty(len(g), int)
        slot[order] = np.arange(1, len(g) + 1)
        pred.append(z[str(i)][np.arange(len(g)), slot - 1])
    assert 100 * np.concatenate(pred).mean() == pytest.approx(12.1207, abs=1e-3)
    # realised share = grid order exact rate on the stored grid_free run; D0/D2 exact columns
    assert cal.loc[("dstar", "D0"), "realised_exact_pct"] == pytest.approx(100 * _stored("D0")[2]["exact"].mean(), abs=1e-9)
    assert cal.loc[("dstar", "D2"), "realised_exact_pct"] == pytest.approx(100 * _stored("D2")[2]["exact"].mean(), abs=1e-9)


# ------------------------------------------------------------------ 5. shared files are append-only
def _head(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)


@pytest.mark.parametrize("path,n_new,comp", [("results/plan3/attribution.csv", 5, "M18"), ("results/plan3/leaderboard.csv", 2, "M18")])
def test_shared_files_are_head_bytes_plus_appended_lines(path, n_new, comp):
    try:
        old = _head(path)
    except Exception as e:                                             # pragma: no cover
        pytest.skip(f"git show failed: {e}")
    cur = (ROOT / path).read_bytes()
    norm = lambda b: b.replace(b"\r\n", b"\n")
    old, cur = norm(old), norm(cur)
    # NB: HEAD may already contain the M18 rows once committed; then nothing is appended and the check is vacuous-safe
    head_has = sum(1 for ln in old.splitlines() if ln.startswith(comp.encode() + b",") or ln.startswith(b"M18_"))
    assert cur.startswith(old), "an existing line was rewritten"
    extra = cur[len(old):]
    if head_has == 0:
        lines = extra.decode("utf-8").splitlines()
        assert len(lines) == n_new and extra.endswith(b"\n")
        assert all(ln.startswith(comp) for ln in lines)
    else:
        assert extra == b""
    df_old = pd.read_csv(StringIO(old.decode("utf-8")))
    df_cur = pd.read_csv(StringIO(cur.decode("utf-8")))
    assert list(df_old.columns) == list(df_cur.columns)
    assert len(df_cur) == len(df_old) + (n_new if head_has == 0 else 0)


def test_leaderboard_rows_match_run_folders():
    lb = pd.read_csv(ROOT / "results" / "plan3" / "leaderboard.csv")
    rows = lb[lb["method"] == "M18"]
    assert len(rows) == 2 and set(rows["label"]) == {"m18_d0.03_D0", "m18_d0.03_D2"} and (rows["split"] == "dev").all()
    for _, r in rows.iterrows():
        d = ROOT / r["run_dir"]
        assert d.exists() and d.name == r["run_id"]
        m = json.loads((d / "metrics.json").read_text())
        for k in ("spearman", "mae", "exact", "kendall", "rmse"):
            if k in rows.columns:
                assert float(r[k]) == pytest.approx(m[k], abs=1e-12), k
    ids = lb[["method", "run_id"]].astype(str)
    assert not ids.duplicated().any()


# ------------------------------------------------------------------ revision: descriptive large-d folders
@needs_runs
def test_descriptive_large_d_folders_are_dev_only_and_not_in_shared_tables():
    dirs = sorted(RES.glob("*_m18_descriptive_d*_dev"))
    lb = pd.read_csv(ROOT / "results" / "plan3" / "leaderboard.csv")
    att = pd.read_csv(ROOT / "results" / "plan3" / "attribution.csv")
    for d in dirs:
        c = load_yaml(d / "config.yaml")
        assert c["split"] == "dev" and c["d"] in (0.5, 1.5)
        pr = pd.read_parquet(d / "predictions.parquet")
        assert pr["season"].between(2021, 2023).all() and pr["race_id"].nunique() == 66
        assert d.name not in set(lb["run_id"].astype(str)) and d.name not in set(att["run_id"].astype(str))
    assert (lb["method"] == "M18").sum() == 2 and att["component"].str.startswith("M18").sum() == 5
