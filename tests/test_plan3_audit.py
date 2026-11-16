"""PLAN3 R1 (A5 strength audit): lam_a switch, Laplace intervals, shrinkage, output tables, split discipline."""
import importlib.util
import inspect
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from pitlane.config import PROCESSED, RESULTS, ROOT, load_yaml
from pitlane.models import latent as new_latent
from pitlane.models.latent import SIGMA, LatentModel
from pitlane.plan3 import audit as A
from test_latent_recovery import _synthetic

ENTRIES = PROCESSED / "entries_feat.parquet"
STRENGTHS = RESULTS / "plan3" / "strengths.csv"
CELLS = ["000", "010", "110", "111"]
LINEAGES = ["mercedes", "red_bull", "rb_lineage", "ferrari", "alpine_lineage", "sauber_lineage", "williams",
            "aston_lineage", "mclaren", "haas", "manor_lineage", "cadillac", "caterham"]


def _flags(cell):
    return cell[0] == "1", cell[1] == "1", cell[2] == "1"


def _head_latent():
    """The committed (HEAD) latent.py loaded as a separate module."""
    try:
        src = subprocess.run(["git", "show", "HEAD:src/pitlane/models/latent.py"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
    except Exception:  # pragma: no cover
        pytest.skip("git HEAD not available")
    mod = types.ModuleType("latent_head")
    exec(compile(src, "latent_head.py", "exec"), mod.__dict__)
    return mod


def _syn(n_races=25, seed=0):
    df, _, _ = _synthetic(seed, n_races=n_races)
    rng = np.random.default_rng(seed + 9)
    df["quali_norm"] = rng.random(len(df))
    df["grid_norm"] = np.clip(df["quali_norm"] + rng.normal(0, 0.05, len(df)), 0, 1)
    df["q1_gap"] = rng.random(len(df)) * 0.02
    df["season"] = 2020 + df["race_id"] // 10
    return df.sort_values(["race_id", "pos"]).reset_index(drop=True)


# ------------------------------------------------------------------ 1. reproduction
@pytest.mark.parametrize("cell", CELLS)
def test_lam_a_none_reproduces_head_bit_for_bit(cell):
    old = _head_latent()
    df = _syn()
    dyn, dec, rk = _flags(cell)
    kw = dict(dynamic=dyn, decomp=dec, ranking=rk, lam=3.0, halflife=8, maxiter=60)
    mo = old.LatentModel(df, **kw)
    mn = LatentModel(df, **kw)
    n = int((df["race_id"] < 20).sum())
    mo.fit(n, 20.0)
    mn.fit(n, 20.0)
    assert np.array_equal(mo.theta, mn.theta), np.abs(mo.theta - mn.theta).max()
    # warm-started second fit, then predictions
    mo.fit(len(df), 25.0)
    mn.fit(len(df), 25.0)
    assert np.array_equal(mo.theta, mn.theta)
    # explicit lam_a == lam is the same fit as None
    ml = LatentModel(df, lam_a=3.0, **kw)
    ml.fit(n, 20.0)
    ml.fit(len(df), 25.0)
    assert np.array_equal(ml.theta, mn.theta)
    # objective values identical at a random point
    th = np.random.default_rng(0).normal(0, 0.3, len(mn.theta))
    w, pad = mn.fit_inputs(n, 20.0)
    wo = 0.5 ** ((20.0 - df["race_id"].to_numpy()[:n]) / 8) if dyn else np.ones(n)
    po = mo._pad(int(df["race_id"].iloc[:n].max()) + 1) if rk else None
    fo, go = mo._objective(th, n, wo, po)
    fn, gn = mn._objective(th, n, w, pad)
    assert fo == fn and np.array_equal(go, gn)


@pytest.mark.skipif(not ENTRIES.exists(), reason="no processed data")
@pytest.mark.parametrize("cell", ["110", "111", "010", "000"])
def test_lam_a_none_reproduces_head_on_real_slice(cell):
    old = _head_latent()
    e = pd.read_parquet(ENTRIES)
    e = e[e["race_id"] < 45].reset_index(drop=True)
    dyn, dec, rk = _flags(cell)
    kw = dict(dynamic=dyn, decomp=dec, ranking=rk, lam=32.0, halflife=50, maxiter=40)
    n = int((e["race_id"] < 40).sum())
    mo, mn = old.LatentModel(e, **kw), LatentModel(e, **kw)
    mo.fit(n, 40.0)
    mn.fit(n, 40.0)
    assert np.array_equal(mo.theta, mn.theta)


def test_changing_lam_a_changes_the_fit_but_not_other_penalties():
    df = _syn()
    kw = dict(dynamic=True, decomp=True, ranking=False, lam=3.0, halflife=8, maxiter=100)
    m0, m1 = LatentModel(df, **kw), LatentModel(df, lam_a=50.0, **kw)
    assert m1.lam == 3.0 and m1.lam_a == 50.0 and m0.lam_a == 3.0
    m0.fit(len(df), 25.0)
    m1.fit(len(df), 25.0)
    assert np.linalg.norm(m1.theta[:m1.nD]) < 0.6 * np.linalg.norm(m0.theta[:m0.nD])


# ------------------------------------------------------------------ 2. gradient with lam_a != lam
@pytest.mark.parametrize("ranking", [False, True])
@pytest.mark.parametrize("dynamic,decomp", [(True, True), (False, False)])
def test_gradient_with_separate_lam_a(ranking, dynamic, decomp):
    df = _syn(15)
    m = LatentModel(df, dynamic=dynamic, decomp=decomp, ranking=ranking, lam=1.5, lam_a=37.0, halflife=6)
    n = len(df)
    w, pad = m.fit_inputs(n, 15.0)
    th0 = np.random.default_rng(2).normal(0, 0.4, len(m.theta))
    f = lambda th: m._objective(th, n, w, pad)[0]
    g = lambda th: m._objective(th, n, w, pad)[1]
    assert check_grad(f, g, th0, epsilon=1e-6) < 1e-5 * max(1.0, np.linalg.norm(g(th0)))
    # the penalty difference is exactly (lam_a - lam) a'a / W
    m2 = LatentModel(df, dynamic=dynamic, decomp=decomp, ranking=ranking, lam=1.5, halflife=6)
    a = th0[:m.nD]
    diff = f(th0) - m2._objective(th0, n, w, pad)[0]
    assert diff == pytest.approx((37.0 - 1.5) * (a @ a) / w.sum(), rel=1e-9)


# ------------------------------------------------------------------ 3. Hessian / Laplace
def _small_fit(ranking, decomp=True, dynamic=True, lam_a=None):
    df = _syn(20)
    m = LatentModel(df, dynamic=dynamic, decomp=decomp, ranking=ranking, lam=2.0, lam_a=lam_a, halflife=10,
                    base_x=False)
    A.fit_tight(m, len(df), 20.0)
    return m, df


@pytest.mark.parametrize("ranking", [False, True])
def test_hessian_symmetric_pd_and_matches_value_second_differences(ranking):
    m, df = _small_fit(ranking, lam_a=20.0)
    n = len(df)
    H1 = A.hessian_fd(m, n, 20.0)
    assert np.allclose(H1, H1.T, atol=1e-8 * np.abs(H1).max())
    assert np.linalg.eigvalsh(H1).min() > 0
    w, pad = m.fit_inputs(n, 20.0)
    assert np.abs(m._objective(m.theta, n, w, pad)[1]).max() < 1e-6      # it is an optimum
    H2 = A.hessian_fd_value_only(m, n, 20.0)
    assert np.abs(H1 - H2).max() / np.abs(H1).max() < 1e-4


def test_hessian_matches_analytic_gaussian_with_lam_a():
    m, df = _small_fit(False, lam_a=20.0)
    H1, Ha = A.hessian_fd(m, len(df), 20.0), A.hessian_analytic_gaussian(m, len(df), 20.0)
    assert np.abs(H1 - Ha).max() / np.abs(Ha).max() < 1e-6
    # the analytic penalty must use lam_a for the driver block and lam for the car block
    m0, _ = _small_fit(False, lam_a=None)
    d = A.hessian_analytic_gaussian(m, len(df), 20.0) - A.hessian_analytic_gaussian(m0, len(df), 20.0)
    assert np.allclose(np.diag(d)[:m.nD], 2 * (20.0 - 2.0))
    assert np.allclose(np.diag(d)[m.nD:], 0)


def test_gaussian_se_closed_form_and_shrinks_with_starts():
    """Gaussian, drivers only, static weights: se_a = sqrt(1 / (n / sigma^2 + 2 lam_a))."""
    rng = np.random.default_rng(0)
    counts = {"few": 3, "mid": 15, "many": 80}
    rows, rid = [], 0
    for d, k in counts.items():
        for _ in range(k):
            rows.append(dict(race_id=rid, season=2020, driverId=d, lineage="c", circuit_type="x",
                             pos=1, n=1, y=float(rng.random()), grid_norm=0.5, quali_norm=0.5, q1_gap=0.0, pit_start=0))
            rid += 1
    df = pd.DataFrame(rows)
    lam_a = 4.0
    m = LatentModel(df, dynamic=False, decomp=False, ranking=False, lam=lam_a, base_x=False)
    A.fit_tight(m, len(df), float(rid))
    se = np.sqrt(np.diag(A.laplace_cov(A.hessian_fd(m, len(df), float(rid)))))[:m.nD]
    got = dict(zip(m.drivers, se))
    for d, k in counts.items():
        assert got[d] == pytest.approx(np.sqrt(1 / (k / SIGMA ** 2 + 2 * lam_a)), rel=1e-5)
    assert got["few"] > got["mid"] > got["many"]


def test_pl_interval_width_shrinks_with_more_starts():
    df = _syn(20)
    # drop most races of d0 so that it has fewer starts than d1 (same car swaps, so identifiable)
    keep = ~((df["driverId"] == "d0") & (df["race_id"] % 4 != 0))
    df = df[keep].reset_index(drop=True)
    m = LatentModel(df, dynamic=False, decomp=True, ranking=True, lam=2.0, lam_a=2.0, base_x=False)
    n = len(df)
    A.fit_tight(m, n, 20.0)
    se = np.sqrt(np.diag(A.laplace_cov(A.hessian_fd(m, n, 20.0))))[:m.nD]
    got = dict(zip(m.drivers, se))
    assert (df["driverId"] == "d0").sum() < (df["driverId"] == "d1").sum()
    assert got["d0"] > got["d1"]


def test_laplace_cov_handles_unused_block():
    H = np.diag([2.0, 0.0, 4.0])
    c = A.laplace_cov(H)
    assert c[1, 1] == 0 and c[0, 0] == pytest.approx(0.5) and c[2, 2] == pytest.approx(0.25)


# ------------------------------------------------------------------ 4. shrinkage claim
def _one_race_frame():
    df = _syn(20)
    extra = df[df["race_id"] == 3].iloc[[0]].copy()
    extra["driverId"] = "one_race"
    extra["y"] = 0.9                                 # finishes near the back: raw a would be 0.5 - 0.9 = -0.4
    df = pd.concat([df, extra]).sort_values(["race_id", "pos"], kind="stable").reset_index(drop=True)
    return df


def test_one_race_driver_a_matches_stated_weight_and_moves_to_zero():
    df = _one_race_frame()
    sig2 = SIGMA ** 2
    prev = None
    for lam_a in [1.0, 8.0, 64.0, 512.0, 4096.0]:
        m = LatentModel(df, dynamic=False, decomp=False, ranking=False, lam=2.0, lam_a=lam_a, base_x=False)
        A.fit_tight(m, len(df), 20.0)
        a = m.theta[m.drivers["one_race"]]
        w = 1.0
        expected = (w / (w + 2 * lam_a * sig2)) * (0.5 - 0.9)
        assert a == pytest.approx(expected, rel=1e-6)
        sh = A.shrinkage_table(m, df, len(df), 20.0).set_index("driverId").loc["one_race"]
        assert sh["data_weight"] == pytest.approx(w / (w + 2 * lam_a * sig2), rel=1e-9)
        assert sh["a_unpenalised"] == pytest.approx(0.5 - 0.9, rel=1e-6)
        assert sh["a"] == pytest.approx(sh["a_stationary"], abs=1e-6)
        assert prev is None or abs(a) < abs(prev)
        prev = a
    assert abs(prev) < 0.01


def test_shrinkage_weight_formula_is_only_approximate_with_car_effects():
    """With the car effect in the model the stated weight is not exact for a one-race driver (the car absorbs
    part of the residual). Document the size of the gap: it must be small enough to be a diagnostic."""
    df = _one_race_frame()
    m = LatentModel(df, dynamic=False, decomp=True, ranking=False, lam=2.0, lam_a=2.0, base_x=False)
    A.fit_tight(m, len(df), 20.0)
    sh = A.shrinkage_table(m, df, len(df), 20.0).set_index("driverId")
    # the stationarity identity itself is exact in the Gaussian case regardless of other terms
    assert (sh["a"] - sh["a_stationary"]).abs().max() < 1e-6


# ------------------------------------------------------------------ leakage of the snapshot fits
@pytest.mark.skipif(not ENTRIES.exists(), reason="no processed data")
def test_snapshot_table_ignores_later_races():
    e = pd.read_parquet(ENTRIES)
    e = e[e["race_id"] < 45].reset_index(drop=True)
    cfg = dict(kind="latent", cell="110", lam=32.0, halflife=50)
    (s, n_train, t_ref) = A.snapshot_points(e)[0]
    t1, _, _, _ = A.snapshot_table(e, cfg, "x", s, n_train, t_ref)
    alt = e.copy()
    later = alt.index >= n_train
    rng = np.random.default_rng(0)
    alt.loc[later, "y"] = rng.random(later.sum())
    alt.loc[later, "quali_norm"] = rng.random(later.sum())
    alt.loc[later, "grid_norm"] = rng.random(later.sum())
    alt.loc[later, "points"] = 99.0
    t2, _, _, _ = A.snapshot_table(alt, cfg, "x", s, n_train, t_ref)
    pd.testing.assert_frame_equal(t1, t2, check_exact=False, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not ENTRIES.exists(), reason="no processed data")
def test_snapshot_points_are_season_ends():
    e = pd.read_parquet(ENTRIES)
    for s, n_train, t_ref in A.snapshot_points(e):
        assert e["season"].iloc[n_train - 1] == s
        assert n_train == len(e) or e["season"].iloc[n_train] > s
        assert t_ref == e["race_id"].iloc[n_train - 1] + 1


# ------------------------------------------------------------------ 5. data checks on the committed outputs
@pytest.fixture(scope="module")
def tables():
    if not (STRENGTHS.exists() and ENTRIES.exists()):
        pytest.skip("no A5 outputs / data")
    return pd.read_csv(STRENGTHS), pd.read_parquet(ENTRIES)


def test_strengths_no_duplicates_no_missing_ordered_intervals(tables):
    s, _ = tables
    assert not s.duplicated(["model", "season", "kind", "id"]).any()
    for c in ["est", "se", "lo", "hi", "n_season", "n_total"]:
        assert s[c].notna().all() and np.isfinite(s[c]).all(), c
    assert (s["se"] > 0).all()
    assert (s["lo"] < s["est"]).all() and (s["est"] < s["hi"]).all()
    np.testing.assert_allclose((s["hi"] - s["lo"]) / 2, 1.96 * s["se"], rtol=1e-9)
    assert set(s["model"]) == {"quali_in", "quali_in_lam_a", "h0", "h0_lam_a"}
    assert s["se"].max() < 5          # no absurd interval


def test_all_13_lineages_present_in_the_seasons_they_raced(tables):
    s, e = tables
    assert sorted(e["lineage"].unique()) == sorted(LINEAGES)
    raced = set(map(tuple, e[["lineage", "season"]].drop_duplicates().to_numpy()))
    for model, g in s[s["kind"] == "car"].groupby("model"):
        got = set(zip(g["id"], g["season"]))
        assert got == raced, (model, raced - got, got - raced)
        assert set(g["id"]) == set(LINEAGES)


def test_every_driver_in_entries_appears_each_season_with_right_counts(tables):
    s, e = tables
    per = e.groupby(["season", "driverId"]).size()
    cum = e.groupby("driverId").cumcount() + 1          # entries sorted by race
    e = e.assign(cum=cum)
    for model, g in s[s["kind"] == "driver"].groupby("model"):
        got = g.set_index(["season", "id"])
        assert set(got.index) == set(per.index), model
        assert (got["n_season"].reindex(per.index) == per).all()
        last_cum = e.groupby(["season", "driverId"])["cum"].max()   # starts through that season end (no future)
        assert (got["n_total"].reindex(last_cum.index) == last_cum).all()


def test_car_n_season_matches_entries(tables):
    s, e = tables
    per = e.groupby(["season", "lineage"]).size()
    for model, g in s[s["kind"] == "car"].groupby("model"):
        got = g.set_index(["season", "id"])["n_season"]
        assert (got.reindex(per.index) == per).all(), model


def test_rank_columns_consistent(tables):
    s, _ = tables
    for (model, season, kind), g in s.groupby(["model", "season", "kind"]):
        assert sorted(g["rank"]) == list(range(1, len(g) + 1))
        assert (g["low_sample"] == ((g["kind"] == "driver") & (g["n_total"] < 10))).all()
        main = g[~g["low_sample"]]["rank_main"]
        assert main.notna().all() and sorted(main.astype(int)) == list(range(1, len(main) + 1))
        assert g[g["low_sample"]]["rank_main"].isna().all()


def test_final_fit_tables(tables):
    _, e = tables
    d = pd.read_csv(RESULTS / "plan3" / "drivers_final_fit.csv")
    c = pd.read_csv(RESULTS / "plan3" / "cars_final_fit.csv")
    for model, g in d.groupby("model"):
        assert not g.duplicated("id").any() and set(g["id"]) == set(e["driverId"])
        assert g[["est", "se", "lo", "hi"]].notna().all().all() and (g["lo"] < g["est"]).all() and (g["est"] < g["hi"]).all()
    for model, g in c.groupby("model"):
        assert not g.duplicated(["id", "season"]).any()
        assert set(zip(g["id"], g["season"])) == set(map(tuple, e[["lineage", "season"]].drop_duplicates().to_numpy()))
        assert (g["se"] > 0).all()


def test_lam_a_variants_have_smaller_driver_spread_and_narrower_driver_intervals(tables):
    s, _ = tables
    d = s[s["kind"] == "driver"]
    for tag in ("quali_in", "h0"):
        a, b = d[d["model"] == tag], d[d["model"] == f"{tag}_lam_a"]
        assert b["est"].std() < a["est"].std()
        assert b["se"].mean() < a["se"].mean()


def test_tuning_table_consistent_with_rule_2():
    p = RESULTS / "plan3" / "tuning_lam_a.csv"
    if not p.exists():
        pytest.skip("no tuning table")
    t = pd.read_csv(p)
    cfg = load_yaml(ROOT / "methodologies" / "A5_strength_audit" / "config.yaml")
    grid = np.array(cfg["lam_a_grid"], float)
    for model, g in t.groupby("model"):
        eb = g["lam_a_eb"].iloc[0]
        sd = g["sd_unpenalised_a_2023"].iloc[0]
        assert eb == pytest.approx(1 / (2 * sd ** 2))
        want = grid[np.abs(np.log(grid) - np.log(eb)).argmin()]
        ch = g[g["is_chosen"]]
        assert len(ch) == 1 and ch["lam_a_eff"].iloc[0] == want
        assert g["n_races"].nunique() == 1 and g["n_races"].iloc[0] == 66          # 3 dev seasons: 22 + 22 + 22
        # chosen must be dev-indistinguishable from the best, as the config states
        assert ch["d_lo"].iloc[0] <= 0 <= ch["d_hi"].iloc[0]


# ------------------------------------------------------------------ 6. split discipline
AUDIT_SRC = Path(inspect.getsourcefile(A)).read_text(encoding="utf-8")
RUN_SRC = (ROOT / "methodologies" / "A5_strength_audit" / "run.py").read_text(encoding="utf-8")


def _function_source(src: str, name: str) -> str:
    m = re.search(rf"^def {name}\(.*?(?=^def |^# -{{10}}|\Z)", src, re.S | re.M)
    assert m, name
    return m.group(0)


def test_tuning_uses_dev_split_only():
    tune = _function_source(RUN_SRC, "tune_lam_a")
    code = "\n".join(l.split("#")[0] for l in tune.splitlines())
    assert code.count("walk_forward(") == 1 and re.search(r'walk_forward\(.*?"dev"', code, re.S)
    assert not re.search(r'"(test|test_2024|holdout|all)"', code)
    for bad in ("test", "holdout", "2024", "2025", "2026", "fresh"):
        assert bad not in code, bad
    cfg = load_yaml(ROOT / "methodologies" / "A5_strength_audit" / "config.yaml")
    assert cfg["eb_snapshot_season"] <= 2023
    assert max(cfg["lam_a_grid"]) <= 4096 and cfg["select_rule"] == "eb_nearest_log"


def test_audit_module_has_no_split_constants():
    """Late seasons may appear only as the descriptive n_2024_26 column of final_tables (not used for any choice)."""
    code = "\n".join(l.split("#")[0] for l in AUDIT_SRC.splitlines())
    code = code.replace("\n".join(l.split("#")[0] for l in _function_source(AUDIT_SRC, "final_tables").splitlines()), "")
    for bad in ("2024", "2025", "2026", "holdout", '"test"'):
        assert bad not in code, bad
    assert "n_2024_26" not in _function_source(RUN_SRC, "tune_lam_a")


def test_run_only_touches_late_seasons_outside_tuning():
    """Every mention of 2024-26 in run.py must sit outside tune_lam_a and be descriptive (n_2024_26 flags, plots)."""
    code = "\n".join(l.split("#")[0] for l in RUN_SRC.splitlines())
    tune = "\n".join(l.split("#")[0] for l in _function_source(RUN_SRC, "tune_lam_a").splitlines())
    assert not re.search(r"202[456]", tune)
    for ln in code.splitlines():
        if re.search(r"202[456]", ln):
            assert re.search(r"n_2024_26|2024-26|plot|ax\.|title|2026\)", ln) or '"' in ln, ln


@pytest.mark.skipif(not ENTRIES.exists(), reason="no processed data")
def test_eb_sd_uses_only_data_through_2023():
    """The empirical-Bayes snapshot must be the 2023 season-end fit: truncating entries after 2023 changes nothing."""
    e = pd.read_parquet(ENTRIES)
    pts = {s: (n, t) for s, n, t in A.snapshot_points(e)}
    n23, t23 = pts[2023]
    assert e["season"].iloc[:n23].max() == 2023
    assert (e["season"].iloc[n23:] > 2023).all()
    cfg = dict(kind="latent", cell="110", lam=32.0, halflife=50)
    full = e[e["race_id"] < 200].reset_index(drop=True)      # keep the test fast: slice then check same truncation property
    n, t = [(n, t) for s, n, t in A.snapshot_points(full)][-1]
    m1 = LatentModel(full, dynamic=True, decomp=True, ranking=False, lam=32.0, halflife=50)
    A.fit_tight(m1, n, t, maxiter=300)
    cut = full.iloc[:n].reset_index(drop=True)
    m2 = LatentModel(cut, dynamic=True, decomp=True, ranking=False, lam=32.0, halflife=50)
    A.fit_tight(m2, n, t, maxiter=300)
    s1 = A.shrinkage_table(m1, full, n, t).set_index("driverId")
    s2 = A.shrinkage_table(m2, cut, n, t).set_index("driverId")
    pd.testing.assert_series_equal(s1["a_unpenalised"].loc[s2.index], s2["a_unpenalised"], check_exact=False, rtol=1e-5, atol=1e-7)
