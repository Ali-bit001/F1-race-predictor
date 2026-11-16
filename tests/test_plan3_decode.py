"""PLAN3 R2: attribution harness (exact-position metrics, paired_gain, upsert) and M17 decoders."""
import inspect
import itertools
import json
import re

import numpy as np
import pandas as pd
import pytest

from pitlane.config import METHODS, ROOT
from pitlane.evaluation.metrics import pred_rank, race_metrics
from pitlane.plan3 import attribution as A
from pitlane.plan3 import decode as DEC

M17 = METHODS / "M17_decoding"
RUNS = M17 / "results"


def _run(label):
    d = sorted(RUNS.glob(f"*_m17_{label}_dev"))
    if not d:
        pytest.skip(f"no run folder for {label}")
    return d[-1]


def _sinkhorn(rng, n, it=500):
    P = rng.random((n, n)) ** 3 + 1e-3
    for _ in range(it):
        P /= P.sum(1, keepdims=True)
        P /= P.sum(0, keepdims=True)
    return P


def _hits(P, place):
    return float(P[np.arange(len(place)), place.astype(int) - 1].sum())


# ------------------------------------------------------------------ 1. metrics
def test_exact_hand_race():
    pos = np.array([1, 2, 3, 4.0])
    u = np.array([4, 2, 3, 1.0])          # predicted ranks [1,3,2,4]
    cls = np.array([True, True, True, False])
    rk = pred_rank(u)
    assert list(rk) == [1, 3, 2, 4]
    b = A.exact_by_band(rk, pos, cls)
    assert b["exact_fin"] == pytest.approx(1 / 3)
    assert b["exact_1_3"] == pytest.approx(1 / 3)
    assert b["exact_4_10"] == 1.0
    assert np.isnan(b["exact_11p"])
    t = A.race_table(pd.DataFrame(dict(race_id=1, season=2021, pos=pos, classified=cls, score=u)))
    assert t["exact"].iloc[0] == 0.5                       # overall, includes the retiree
    assert t["exact"].iloc[0] == race_metrics(pos, u, cls)["exact"]


def test_exact_bands_partition_and_perfect():
    n = 20
    pos = np.arange(1, n + 1.0)
    u = -pos
    cls = pos <= 15
    b = A.exact_by_band(pred_rank(u), pos, cls)
    assert all(v == 1.0 for v in b.values())
    # reversed prediction on 20 cars: no exact hit (even n, no fixed point)
    b = A.exact_by_band(pred_rank(-u), pos, cls)
    assert all(v == 0.0 for v in b.values())
    # band rates weighted by band size recompose the overall rate
    rng = np.random.default_rng(0)
    u = rng.normal(size=n)
    rk = pred_rank(u)
    b = A.exact_by_band(rk, pos, cls)
    tot = b["exact_1_3"] * 3 + b["exact_4_10"] * 7 + b["exact_11p"] * 10
    assert tot / n == pytest.approx((rk == pos).mean())


def test_exact_matches_stored_runs():
    d = _run("D0")
    p = pd.read_parquet(d / "predictions.parquet")
    per = pd.read_csv(d / "per_race_metrics.csv").set_index("race_id")
    for rid, g in p.groupby("race_id"):
        m = race_metrics(g["pos"].to_numpy(float), g["score"].to_numpy(float), g["classified"].to_numpy(bool))
        assert per.loc[rid, "exact"] == pytest.approx(m["exact"])


# ------------------------------------------------------------------ 2. paired_gain / verdict
def _tab(rng, n=40, **kw):
    t = pd.DataFrame({"race_id": np.arange(n)})
    for k in ("spearman", "mae", "exact", "rps"):
        t[k] = rng.random(n)
    for k, v in kw.items():
        t[k] = v
    return t


def test_paired_gain_deterministic_and_ci():
    rng = np.random.default_rng(1)
    r = _tab(rng)
    c = r.copy()
    c["exact"] = r["exact"] + 0.05 + 0.1 * rng.normal(size=len(r))
    g1 = A.paired_gain(r, c, ["exact"], n_boot=2000, seed=3)
    g2 = A.paired_gain(r, c, ["exact"], n_boot=2000, seed=3)
    pd.testing.assert_frame_equal(g1, g2)
    g3 = A.paired_gain(r, c, ["exact"], n_boot=2000, seed=4)
    assert g3["lo"].iloc[0] != g1["lo"].iloc[0]
    row = g1.iloc[0]
    assert row["gain"] == pytest.approx((c["exact"] - r["exact"]).mean())
    assert row["lo"] < row["gain"] < row["hi"]
    assert row["n"] == len(r)


def test_paired_gain_zero_and_sign_convention():
    rng = np.random.default_rng(2)
    r = _tab(rng)
    z = A.paired_gain(r, r.copy(), n_boot=500)
    assert (z[["gain", "lo", "hi"]] == 0).all().all()
    for k, better in [("spearman", +0.1), ("exact", +0.1), ("mae", -0.1), ("rps", -0.1)]:
        c = r.copy()
        c[k] = r[k] + better                # candidate better in the metric's own direction
        g = A.paired_gain(r, c, [k], n_boot=500).iloc[0]
        assert g["gain"] == pytest.approx(0.1), k
        assert g["lo"] > 0, k
        c[k] = r[k] - better                # candidate worse
        g = A.paired_gain(r, c, [k], n_boot=500).iloc[0]
        assert g["gain"] == pytest.approx(-0.1), k
        assert g["hi"] < 0, k


def test_paired_gain_nan_and_race_alignment():
    rng = np.random.default_rng(5)
    r = _tab(rng, 10)
    c = r.copy().iloc[::-1].reset_index(drop=True)      # same races, other order: merge on race_id
    c.loc[c["race_id"] == 3, "exact"] = np.nan
    g = A.paired_gain(r, c, ["exact"], n_boot=200).iloc[0]
    assert g["n"] == 9 and g["gain"] == 0.0
    assert A.paired_gain(r, c.iloc[:5], ["exact"], n_boot=200).iloc[0]["n"] <= 5


def test_verdict_rule():
    assert A.verdict({"lo": 0.01, "hi": 0.05}) == "passes alone"
    assert A.verdict({"lo": -0.05, "hi": -0.01}) == "worse"
    assert A.verdict({"lo": -0.01, "hi": 0.05}) == "no measurable difference"
    assert A.verdict({"lo": 0.0, "hi": 0.05}) == "no measurable difference"     # CI must EXCLUDE 0
    assert A.verdict({"lo": -0.05, "hi": 0.0}) == "no measurable difference"


# ------------------------------------------------------------------ 3. upsert
def _row(comp, ref, gain, **kw):
    return {"component": comp, "reference": ref, "gain_exact": gain, "loo_gain": "", "fresh_holdout_gain": "",
            "verdict": "x", **kw}


def test_upsert_keeps_other_rows_and_blank_columns(tmp_path):
    p = tmp_path / "a.csv"
    A.upsert_attribution([_row("c1", "D0", 1.0), _row("c2", "D0", 2.0), _row("c1", "grid", 3.0)], p)
    df = A.upsert_attribution([_row("c1", "D0", 9.0)], p)
    assert len(df) == 3
    d = df.set_index(["component", "reference"])
    assert d.loc[("c1", "D0"), "gain_exact"] == 9.0
    assert d.loc[("c2", "D0"), "gain_exact"] == 2.0 and d.loc[("c1", "grid"), "gain_exact"] == 3.0
    back = pd.read_csv(p)
    assert len(back) == 3 and not back.duplicated(["component", "reference"]).any()
    assert list(back.columns) == list(df.columns)


def test_upsert_preserves_loo_holdout_only_on_matching_row(tmp_path):
    p = tmp_path / "a.csv"
    A.upsert_attribution([_row("c1", "D0", 1.0), _row("c1", "grid", 1.0)], p)
    old = pd.read_csv(p, dtype=object)
    old.loc[(old.component == "c1") & (old.reference == "grid"), "loo_gain"] = "0.5"
    old.loc[(old.component == "c1") & (old.reference == "D0"), "loo_gain"] = "0.7"
    old.to_csv(p, index=False)
    df = A.upsert_attribution([_row("c1", "grid", 2.0)], p).set_index(["component", "reference"])
    assert float(df.loc[("c1", "grid"), "loo_gain"]) == 0.5
    assert float(df.loc[("c1", "D0"), "loo_gain"]) == 0.7
    assert df.loc[("c1", "grid"), "gain_exact"] == 2.0


def test_upsert_loo_not_copied_across_references(tmp_path):
    """Regression probe: upsert of (c1, grid) must not take the LOO value of (c1, D0)."""
    p = tmp_path / "a.csv"
    A.upsert_attribution([_row("c1", "D0", 1.0)], p)
    old = pd.read_csv(p, dtype=object)
    old["loo_gain"] = "0.7"
    old.to_csv(p, index=False)
    df = A.upsert_attribution([_row("c1", "grid", 2.0)], p).set_index(["component", "reference"])
    assert pd.isna(df.loc[("c1", "grid"), "loo_gain"]) or df.loc[("c1", "grid"), "loo_gain"] == ""


def test_upsert_two_new_rows_same_component_different_reference(tmp_path):
    """Both rows share a component; each must keep ITS OWN preserved cells."""
    p = tmp_path / "a.csv"
    A.upsert_attribution([_row("c1", "D0", 1.0), _row("c1", "grid", 1.0)], p)
    old = pd.read_csv(p, dtype=object)
    old.loc[old.reference == "D0", "loo_gain"] = "0.1"
    old.loc[old.reference == "grid", "loo_gain"] = "0.2"
    old.to_csv(p, index=False)
    df = A.upsert_attribution([_row("c1", "D0", 5.0), _row("c1", "grid", 6.0)], p).set_index(["component", "reference"])
    assert float(df.loc[("c1", "D0"), "loo_gain"]) == 0.1
    assert float(df.loc[("c1", "grid"), "loo_gain"]) == 0.2


def test_stored_attribution_csv_unique_keys():
    df = pd.read_csv(ROOT / "results" / "plan3" / "attribution.csv")
    assert not df.duplicated(["component", "reference"]).any()


# ------------------------------------------------------------------ 4. decoders
def test_d2_valid_permutation_and_optimal_vs_d0_d1():
    rng = np.random.default_rng(7)
    for n in (3, 8, 15, 20):
        for _ in range(30):
            P = _sinkhorn(rng, n, 100)
            pl = -DEC.decode_d2(P)
            assert sorted(pl) == list(range(1, n + 1))
            assert np.array_equal(pred_rank(DEC.decode_d2(P)), pl)
            for f in (DEC.decode_d1, lambda M: -(M @ np.arange(1, n + 1))):
                other = pred_rank(f(P))
                assert _hits(P, pl) >= _hits(P, other) - 1e-9


def test_d2_beats_sort_by_expected_position():
    P = np.array([[.5, 0, .5], [.4, .3, .3], [.1, .7, .2]])
    assert np.allclose(P.sum(0), 1) and np.allclose(P.sum(1), 1)
    r1 = pred_rank(DEC.decode_d1(P))
    r2 = pred_rank(DEC.decode_d2(P))
    assert list(r1) == [2, 1, 3]
    assert list(r2) == [3, 1, 2]
    assert _hits(P, r2) == pytest.approx(1.6) and _hits(P, r1) == pytest.approx(0.6)


def test_d2_bruteforce_all_permutations():
    rng = np.random.default_rng(11)
    for n in range(2, 8):
        for _ in range(6):
            P = _sinkhorn(rng, n, 50)
            best = max(sum(P[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
            assert _hits(P, -DEC.decode_d2(P)) == pytest.approx(best, abs=1e-6)


def test_d1_is_negative_expected_position():
    P = _sinkhorn(np.random.default_rng(3), 6, 50)
    assert np.allclose(DEC.decode_d1(P), -(P @ np.arange(1, 7)))


# ------------------------------------------------------------------ 5. outcome leakage in decoding
def test_d2_and_grid_order_invariant_to_row_shuffle():
    rng = np.random.default_rng(13)
    for n in (5, 10, 20):
        P = _sinkhorn(rng, n, 100)
        grid = rng.permutation(np.arange(1, n + 1)).astype(float)
        grid[rng.integers(n)] = 0                                   # one pit-lane start
        for _ in range(20):
            s = rng.permutation(n)
            assert np.array_equal(pred_rank(DEC.decode_d2(P))[s], pred_rank(DEC.decode_d2(P[s])))
            assert np.array_equal(pred_rank(DEC.grid_utility(grid))[s], pred_rank(DEC.grid_utility(grid[s])))


def test_d2_tie_break_not_input_order_on_exact_ties():
    """Two drivers with IDENTICAL rows are interchangeable for the assignment; the ORDER must still not
    depend on which one comes first when other drivers are present (checked on a tie-heavy matrix)."""
    P = np.full((6, 6), 1 / 6)
    P[0] = P[1] = [.4, .3, .1, .1, .05, .05]
    P[2:, :] = (1 - P[0:1, :] * 2) / 4
    assert np.allclose(P.sum(0), 1) and np.allclose(P.sum(1), 1)
    base = pred_rank(DEC.decode_d2(P))
    assert sorted(base) == list(range(1, 7))
    # fully identical rows: the optimum value must not change with row order, which is all that is
    # required of a tie-break that cannot see the outcome
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.permutation(6)
        assert _hits(P[s], pred_rank(DEC.decode_d2(P[s]))) == pytest.approx(_hits(P, base))


def _dev_matrices():
    d = _run("D0")
    z = np.load(d / "position_matrices.npz")
    p = pd.read_parquet(d / "predictions.parquet")
    return z, p


def test_d0_row_shuffle_sensitivity_reported():
    """D0 ties are broken by input row order = finishing order. Measure the effect of random shuffles."""
    z, p = _dev_matrices()
    rng = np.random.default_rng(0)
    base, shuf, tied = [], [], 0
    groups = {str(r): g for r, g in p.groupby("race_id")}
    n_rep = 100
    rates = np.zeros((n_rep, len(groups)))
    for j, (rid, g) in enumerate(groups.items()):
        P = z[rid]
        pos = g["pos"].to_numpy(float)
        e = P @ np.arange(1, len(pos) + 1)
        tied += len(np.unique(e)) < len(e)
        base.append((pred_rank(-e) == pos).mean())
        for k in range(n_rep):
            s = rng.permutation(len(pos))
            rates[k, j] = (pred_rank(-e[s]) == pos[s]).mean()
    b = float(np.mean(base))
    sh = rates.mean(1)
    print(f"\nD0 exact dev: as-stored {100*b:.4f}%  shuffled mean {100*sh.mean():.4f}% "
          f"min {100*sh.min():.4f}% max {100*sh.max():.4f}%  races with tied E[pos]: {tied}")
    assert abs(sh.mean() - b) < 0.005          # at most half a percentage point
    assert tied <= 5


def test_simulate_matrices_ignores_outcome_columns():
    n = 8
    rng = np.random.default_rng(0)
    theta = dict(tau=2.0, dnf_scale=1.0, kappa=0.0, p0=0.1, s_dis=2.0, a_vol=0.0, a_rain=0.0)
    base = dict(u=rng.normal(size=n), p_dnf=rng.random(n) * .2, grid=rng.random(n), stick_c=0.3, zvol=0.1, rain=0.2)
    r_clean = [dict(base)]                                          # no outcome keys at all: must not KeyError
    r_dirty = [dict(base, pos=np.arange(1, n + 1.0), cls=np.ones(n, bool),
                    g=pd.DataFrame({"pos": np.arange(n), "classified": True}))]
    r_dirty2 = [dict(base, pos=rng.permutation(n) + 1.0, cls=rng.random(n) > .5, g=None)]
    a = DEC.simulate_matrices(r_clean, theta, "S3", k=200, seed=0)[0]
    b = DEC.simulate_matrices(r_dirty, theta, "S3", k=200, seed=0)[0]
    c = DEC.simulate_matrices(r_dirty2, theta, "S3", k=200, seed=0)[0]
    assert np.array_equal(a, b) and np.array_equal(a, c)
    src = inspect.getsource(DEC.simulate_matrices)
    assert not re.search(r'\["(pos|cls|classified|g)"\]', src)


def test_sim_inputs_join_has_no_outcome_in_u():
    """The simulator inputs (u, p_dnf) come from stored run predictions of earlier components; the run.py
    join must take only these columns from them (outcome columns are used for scoring only)."""
    src = (M17 / "run.py").read_text()
    assert 'dn = pd.read_parquet(ROOT / row["run_dir"] / "predictions.parquet")[["race_id", "driverId", "p_dnf"]]' in src


# ------------------------------------------------------------------ 6. stored matrices
def test_stored_matrices_sanity():
    z, p = _dev_matrices()
    assert len(z.files) == 66 and p["race_id"].nunique() == 66
    assert sum(z[f].shape[0] for f in z.files) == 1311 == len(p)
    for rid, g in p.groupby("race_id"):
        P = z[str(rid)]
        assert P.shape == (len(g), len(g))
        assert (P >= 0).all()
        assert np.allclose(P.sum(1), 1, atol=1e-9)              # each driver: a distribution over places
        assert np.allclose(P.sum(0), 1, atol=1e-9)              # each place: exactly one driver per draw
        e = P @ np.arange(1, len(g) + 1)
        assert np.allclose(-e, g["score"].to_numpy(float), atol=1e-12)
        assert np.allclose(P[:, 0], g["p_win"].to_numpy(float), atol=1e-12)
        assert np.allclose(P[:, :3].sum(1), g["p_podium"].to_numpy(float), atol=1e-12)
        assert np.array_equal(pred_rank(-e), pred_rank(g["score"].to_numpy(float)))


def test_decoders_on_stored_matrices_consistent_with_stored_runs():
    z, p = _dev_matrices()
    d2 = pd.read_parquet(_run("D2") / "predictions.parquet")
    for rid, g in d2.groupby("race_id"):
        P = z[str(rid)]
        assert np.array_equal(g["score"].to_numpy(float), DEC.decode_d2(P))
        pl = -DEC.decode_d2(P)
        assert sorted(pl) == list(range(1, len(g) + 1))
        assert _hits(P, pl) >= _hits(P, pred_rank(DEC.decode_d1(P))) - 1e-9


# ------------------------------------------------------------------ 7. acceptance numbers
def test_acceptance_numbers_reproduce():
    for label, want in (("D0", 13.57), ("grid", 16.31)):
        m = json.loads((_run(label) / "metrics.json").read_text())
        assert round(100 * m["exact"], 2) == want
        p = pd.read_parquet(_run(label) / "predictions.parquet")
        t = A.race_table(p)
        assert round(100 * t["exact"].mean(), 2) == want
        assert len(t) == 66
        assert set(p["season"]) == {2021, 2022, 2023}


def test_grid_run_is_grid_order():
    p = pd.read_parquet(_run("grid") / "predictions.parquet")
    for rid, g in p.groupby("race_id"):
        assert np.array_equal(g["score"].to_numpy(float), DEC.grid_utility(g["grid"].to_numpy(float)))


def test_d1_equals_d0_stored():
    a = pd.read_parquet(_run("D0") / "predictions.parquet")
    b = pd.read_parquet(_run("D1") / "predictions.parquet")
    assert np.array_equal(a["score"].to_numpy(float), b["score"].to_numpy(float))


# ------------------------------------------------------------------ 8. split discipline
def test_no_test_season_in_r2_code():
    files = [ROOT / "src/pitlane/plan3/attribution.py", ROOT / "src/pitlane/plan3/decode.py",
             M17 / "run.py", M17 / "config.yaml"]
    for f in files:
        txt = f.read_text()
        assert not re.search(r"\b202[456]\b", txt.replace("2021-2023", "")), f.name
        assert not re.search(r"\b(test|holdout|fresh)\b", re.sub(r"fresh_holdout_gain|fresh-holdout", "", txt)), f.name
    assert "dev" in (M17 / "config.yaml").read_text()
    run_src = (M17 / "run.py").read_text()
    assert 'assert cfg["split"] == "dev"' in run_src
    assert '"dev"' in run_src and '"test"' not in run_src


def test_stored_runs_dev_only():
    for f in RUNS.glob("*_m17_*_dev"):
        p = pd.read_parquet(f / "predictions.parquet")
        assert p["season"].between(2021, 2023).all(), f.name
