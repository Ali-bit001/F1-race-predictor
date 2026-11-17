"""PLAN3 R11 / M14 part 1 (engine-supplier pooling, b[car] = e[supplier, season] + own part): independent checks.

Sections: (1) finite-difference gradient over every parameter (a, b, g, beta, e) at two lam_e values, (2) off switch (lam_e = inf is
LatentModel bit for bit; the stored reference run equals the stored plan-1 latent_111 dev run), (3) leakage (scrambling race r and
later outcomes and later seasons' suppliers leaves race r's score unchanged, cold and warm-started; power controls; truncation at
the exact optimum), (4) supplier lookup (by supplier value, season-level rows), (5) e is exactly 0 before its first race,
(6) stored candidate and tuning table reproduced fresh, (7) paired gains / by season / exact-vs-grid / attribution row re-derived,
(8) shared files append-only, (9) split discipline, data checks and README numbers, (10) optimiser convergence measurements.
Real-data tests skip when the git-ignored processed data are missing. Later packages may append rows to the shared files.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from pitlane import pipeline as P
from pitlane.config import PROCESSED, ROOT, SPLITS, load_yaml
from pitlane.evaluation.harness import split_races, walk_forward
from pitlane.evaluation.metrics import race_metrics
from pitlane.models.latent import LatentModel
from pitlane.models.latent_supplier import LatentSupplierModel
from pitlane.models.registry import build_model
from pitlane.plan3 import attribution as A
from pitlane.plan3 import decode as DEC
from pitlane.plan3 import runutil as U

METH = ROOT / "methodologies/M14_car_traits"
RES = METH / "results"
CFG = load_yaml(METH / "config.yaml")
STAMP = "20261117-1617"
POST_STAMP = "20261117-1854"
OLD_POST_STAMP = "20261117-1828"      # superseded (look-ahead defect), kept on disk
POST_LABEL = "m14_supplier_pool_carry_total_posthoc"
POST_COMPONENT = "M14_supplier_pool_carry_total_posthoc"
POST_BASE = "0473890"                 # HEAD when the revision-2 post-hoc run was made
BASE = "3ad8d53"                      # commit the package was built on
LABELS = {"latent_111": "m14_ref_latent_111", "M14_supplier_pool_latent_111": "m14_supplier_pool_latent_111"}
needs_real = pytest.mark.skipif(not ((PROCESSED / "entries_feat.parquet").exists() or (PROCESSED / "entries.parquet").exists()),
                                reason="processed data missing (regenerable)")
needs_runs = pytest.mark.skipif(not (RES / "paired_gains_dev.csv").exists(), reason="M14 results missing")


def _load_run():
    spec = importlib.util.spec_from_file_location("m14_run_under_test", METH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_run()
LAT = load_yaml(ROOT / CFG["frozen_path"])[CFG["latent_ref"]["key"]][CFG["latent_ref"]["cell"]]
KW = {k: v for k, v in LAT.items() if k not in ("kind", "cell")}


def _dir(label: str):
    c = sorted(RES.glob(f"{POST_STAMP if label == POST_LABEL else STAMP}_{label}_dev"))
    assert len(c) == 1, (label, c)
    return c[0]


def sup_model(u: pd.DataFrame, lam_e) -> LatentSupplierModel:
    return LatentSupplierModel(u, dynamic=True, decomp=True, ranking=True, lam_e=lam_e, **KW)


def _rid(ent: pd.DataFrame, season: int, rnd: int) -> int:
    ids = ent[["season", "round", "race_id"]].drop_duplicates()
    return int(ids[(ids["season"] == season) & (ids["round"] == rnd)]["race_id"].iloc[0])


@pytest.fixture(scope="module")
def ent() -> pd.DataFrame:
    return R.load_universe(CFG)


# ====================================================================================== (1) gradient
def _fd_report(m: LatentSupplierModel, n_train: int, t_ref: float, seed: int):
    w_row, pad = m.fit_inputs(n_train, t_ref)
    th = np.random.default_rng(seed).normal(0, 0.4, len(m.theta))
    g = m._objective(th, n_train, w_row, pad)[1]
    num = np.empty_like(g)
    for i in range(len(th)):
        d = np.zeros_like(th)
        d[i] = 1e-6
        num[i] = (m._objective(th + d, n_train, w_row, pad)[0] - m._objective(th - d, n_train, w_row, pad)[0]) / 2e-6
    return g, num


@needs_real
@pytest.mark.parametrize("lam_e,last_season,seed", [(0.7, 2021, 11), (25.0, 2021, 12), (3.0, 2016, 13)])
def test_gradient_matches_finite_differences_over_all_parameters_including_e(ent, lam_e, last_season, seed):
    """Random non-zero point (e too), decay weights on, several seasons so the carry-over links cars across seasons."""
    u = ent[ent["season"] <= last_season].reset_index(drop=True)
    m = sup_model(u, lam_e)
    rid = int(u["race_id"].max()) - 3
    g, num = _fd_report(m, int(m.race_rows[rid].min()), float(rid), seed)
    assert m.nE >= 8 and len(g) == m.nbase + m.nE
    err = np.abs(g - num) / np.maximum(1.0, np.abs(num))
    assert err.max() < 1e-7, err.max()
    e_idx = np.arange(m.nbase, len(g))
    assert np.abs(g[e_idx]).max() > 1e-3 and np.abs(num[e_idx]).max() > 1e-3          # e coordinates are not vacuous
    assert err[e_idx].max() < 1e-7


@needs_real
def test_gradient_wrt_e_changes_with_lam_e_and_e_penalty_enters_the_value(ent):
    u = ent[ent["season"] <= 2016].reset_index(drop=True)
    rid = int(u["race_id"].max()) - 2
    a, b = sup_model(u, 1.0), sup_model(u, 50.0)
    n = int(a.race_rows[rid].min())
    w, pad = a.fit_inputs(n, float(rid))
    th = np.random.default_rng(3).normal(0, 0.4, len(a.theta))
    fa, ga = a._objective(th, n, w, pad)
    fb, gb = b._objective(th, n, w, pad)
    e = th[a.nbase:]
    assert fb - fa == pytest.approx((50.0 - 1.0) * float(e @ e) / w.sum(), rel=1e-9)         # value: only the e ridge differs
    assert np.allclose(ga[:a.nbase], gb[:a.nbase], atol=1e-13)                                 # base coordinates: same gradient
    assert np.allclose(gb[a.nbase:] - ga[a.nbase:], 2 * (50.0 - 1.0) * e / w.sum(), atol=1e-13)


@needs_real
def test_utility_is_the_sum_of_named_components_including_e(ent):
    u = ent[ent["season"] <= 2016].reset_index(drop=True)
    m = sup_model(u, 5.0)
    m.theta = np.random.default_rng(5).normal(0, 0.3, len(m.theta))
    rows = m.race_rows[int(u["race_id"].max())]
    c = m.components(rows)
    np.testing.assert_allclose(c[["a_driver", "b_car", "g_circuit", "e_supplier", "weekend"]].sum(axis=1), c["u"], atol=1e-13)
    np.testing.assert_allclose(c["e_supplier"], m._e(m.theta)[m.row_e[rows]], atol=0)
    off = sup_model(u, float("inf"))
    assert "e_supplier" not in off.components(rows).columns and off.supplier_effects().empty and len(off.theta) == off.nbase


# ====================================================================================== (2) off switch and references
@needs_real
@pytest.mark.parametrize("off", [float("inf"), None])
def test_lam_e_off_equals_latent_model_bitwise_over_the_whole_dev_walk(ent, off):
    """Warm-started refit before every race (15 warm-up + 66 dev races): every score bit for bit; also through build_model."""
    ref, _ = walk_forward(ent, build_model(dict(LAT), ent), "dev", refit_every=1)
    cfg = dict(LAT, kind="latent_supplier", lam_e=off)
    p, _ = walk_forward(ent, build_model(cfg, ent), "dev", refit_every=1)
    assert len(p) == len(ref) == 1311
    assert np.array_equal(p["score"].to_numpy(), ref["score"].to_numpy())
    assert np.array_equal(p["p_win"].to_numpy(), ref["p_win"].to_numpy())


@needs_real
def test_off_model_has_no_e_parameters_and_same_theta_layout_as_latent_model(ent):
    m = sup_model(ent, float("inf"))
    b = LatentModel(ent, dynamic=True, decomp=True, ranking=True, **KW)
    assert m.nE == 0 and not m.on and m.theta.shape == b.theta.shape and m.nbase == len(b.theta)
    on = sup_model(ent, 10.0)
    assert on.on and on.theta.shape[0] == len(b.theta) + on.nE and (on.theta[on.nbase:] == 0).all()
    with pytest.raises(ValueError):
        LatentSupplierModel(ent, dynamic=False, decomp=True, ranking=True, lam_e=1.0, **KW)
    with pytest.raises(ValueError):
        LatentSupplierModel(ent.drop(columns="supplier"), dynamic=True, decomp=True, ranking=True, lam_e=1.0, **KW)


@needs_runs
def test_stored_reference_run_equals_stored_plan1_latent_111_dev_run():
    d = _dir(LABELS["latent_111"])
    old_pm, old_pr = U.stored_run(CFG["plan1_leaderboard"], "latent_111")
    pm, pr = pd.read_csv(d / "per_race_metrics.csv"), pd.read_parquet(d / "predictions.parquet")
    diff, n = U.max_diff(pm, old_pm)
    assert n == 66 and diff < 1e-12, diff
    sd, ne = U.score_diff(pr, old_pr)
    assert ne == 1311 and sd <= 1e-12, sd
    assert list(pm["race_id"]) == list(old_pm["race_id"])
    rep = json.loads((RES / "reproduction_check.json").read_text(encoding="utf-8"))
    assert rep["latent_111"]["races"] == 66 and rep["latent_111"]["entries"] == 1311 and rep["latent_111"]["max_abs_diff_scores"] == 0.0
    assert rep["off_arm_vs_latent_111_same_harness"] == dict(max_abs_diff_per_race_metrics=0.0, max_abs_diff_scores=0.0)
    cfg = load_yaml(d / "config.yaml")
    for k, v in LAT.items():
        assert cfg[k] == v, k
    assert cfg["kind"] == "latent" and "lam_e" not in cfg


@needs_real
@needs_runs
def test_stored_reference_run_equals_a_fresh_latent_model_run(ent):
    p, _ = walk_forward(ent, build_model(dict(LAT), ent), "dev", refit_every=1)
    st = pd.read_parquet(_dir(LABELS["latent_111"]) / "predictions.parquet")
    d, n = U.score_diff(p, st)
    assert n == 1311 and d <= 1e-12


# ====================================================================================== (3) leakage
OUTCOME_COLS = ["y", "pos", "finish_order", "dnf", "classified", "points", "status", "positionText"]
PRE_COLS = ["q1_gap", "grid_norm", "quali_norm", "grid", "quali_pos"]
RACES = [(2022, 1), (2021, 12), (2023, 15)]


def _reorder(ent: pd.DataFrame, race_ids, seed: int, reverse: bool = False) -> pd.DataFrame:
    """Reorder the rows INSIDE each of the given races (the Plackett-Luce fit reads the finishing order from the row order of a
    race: universe rows are sorted by finishing position), keeping races in place. reverse=True: exact reversal of the order."""
    rng = np.random.default_rng(seed)
    order = np.arange(len(ent))
    for r, ix in ent.groupby("race_id").indices.items():
        if r in race_ids:
            order[ix] = ix[::-1] if reverse else rng.permutation(ix)
    return ent.iloc[order].reset_index(drop=True)


def _scramble(ent: pd.DataFrame, rid: int, seed: int, suppliers: bool = True) -> pd.DataFrame:
    """Everything that is an outcome of race r or later is destroyed: the finishing ORDER (row order inside each race, which is what the
    ranking likelihood reads) is shuffled, and the outcome columns are permuted across those rows too. Pre-race columns of races after
    r are permuted; the supplier of every later SEASON is replaced by a new label. Race r's own pre-race inputs (grid, qualifying,
    supplier of its season) are untouched."""
    ids = set(int(x) for x in ent["race_id"].unique() if x >= rid)
    e2 = _reorder(ent, ids, seed)
    rng = np.random.default_rng(seed + 1)
    late = (e2["race_id"] >= rid).to_numpy()
    perm = rng.permutation(np.flatnonzero(late))
    for c in OUTCOME_COLS:
        if c in e2:
            e2.loc[late, c] = e2[c].to_numpy()[perm]
    after = (e2["race_id"] > rid).to_numpy()
    perm2 = rng.permutation(np.flatnonzero(after))
    for c in PRE_COLS:
        if c in e2:
            e2.loc[after, c] = e2[c].to_numpy()[perm2]
    if suppliers:
        season = int(ent.loc[ent["race_id"] == rid, "season"].iloc[0])
        e2.loc[e2["season"] > season, "supplier"] = "scrambled"
    return e2


def _cold(u: pd.DataFrame, rid: int, lam_e: float) -> pd.Series:
    m = sup_model(u, lam_e)
    rows = m.race_rows[rid]
    s = m.fit_predict(u.iloc[:int(rows.min())], u.iloc[rows])
    return pd.Series(s, index=u.iloc[rows]["driverId"].to_numpy()).sort_index()


def _warm(u: pd.DataFrame, rid: int, lam_e: float) -> pd.Series:
    """Scores of race rid after the same warm-started sequence of refits that walk_forward(refit_every=1) performs."""
    m = sup_model(u, lam_e)
    warm, counted = split_races(u, "dev")
    out = None
    for r in warm + counted:
        rows = m.race_rows[r]
        s = m.fit_predict(u.iloc[:int(rows.min())], u.iloc[rows])
        if r == rid:
            out = pd.Series(s, index=u.iloc[rows]["driverId"].to_numpy()).sort_index()
            break
    assert out is not None
    return out


def _gap(a: pd.Series, b: pd.Series) -> float:
    assert list(a.index) == list(b.index)
    return float(np.abs(a.to_numpy() - b.to_numpy()).max())


@needs_real
@pytest.mark.parametrize("season,rnd", RACES)
@pytest.mark.parametrize("lam_e", [3.0, 100.0])
def test_scores_of_race_r_unchanged_when_race_r_and_later_outcomes_and_later_suppliers_are_scrambled(ent, season, rnd, lam_e):
    rid = _rid(ent, season, rnd)
    base = _cold(ent, rid, lam_e)
    got = _cold(_scramble(ent, rid, rid), rid, lam_e)
    assert len(base) >= 18 and _gap(got, base) <= 1e-12, _gap(got, base)
    # same through the warm-started walk path used by run.py (bitwise expected; tolerance 1e-12)
    wb, wg = _warm(ent, rid, lam_e), _warm(_scramble(ent, rid, rid + 1), rid, lam_e)
    assert _gap(wb, wg) <= 1e-12, _gap(wb, wg)


@needs_real
@pytest.mark.parametrize("season,rnd", RACES)
def test_power_controls_an_earlier_race_or_a_same_season_supplier_moves_the_score(ent, season, rnd):
    """If these did not move the score, the unchanged-score test above would prove nothing."""
    rid = _rid(ent, season, rnd)
    base = _cold(ent, rid, 3.0)
    # (a) reverse the finishing order of the race just before r, and shuffle the order of the three races before it
    assert _gap(_cold(_reorder(ent, {rid - 1}, 0, reverse=True), rid, 3.0), base) > 1e-4
    assert _gap(_cold(_reorder(ent, {rid - 3, rid - 2, rid - 1}, 5), rid, 3.0), base) > 1e-4
    # (b) the SUPPLIER of the season of r: only informative once the season has races, i.e. not at round 1 (e = 0 there)
    sea = int(ent.loc[ent["race_id"] == rid, "season"].iloc[0])
    e3 = ent.copy()
    e3.loc[(e3["season"] == sea) & (e3["constructorId"] == "williams"), "supplier"] = "ferrari"
    e3.loc[(e3["season"] == sea) & (e3["constructorId"] == "haas"), "supplier"] = "mercedes"
    d = _gap(_cold(e3, rid, 3.0), base)
    if rnd == 1:
        assert d <= 1e-12            # first race of the season: its season's levels are all 0 whatever the labels are
    else:
        assert d > 1e-4, d


@needs_real
def test_the_ranking_fit_reads_row_order_not_the_outcome_columns(ent):
    """Documents why _scramble reorders rows: permuting y / pos / finish_order / classified of an EARLIER race changes nothing in the
    Plackett-Luce fit (bitwise), reordering its rows does. A leakage check that only permutes those columns cannot detect leakage."""
    rid = _rid(ent, 2023, 15)
    base = _cold(ent, rid, 3.0)
    e2 = ent.copy()
    prev = (e2["race_id"] == rid - 1).to_numpy()
    for c in ("y", "pos", "finish_order", "classified", "dnf", "points"):
        e2.loc[prev, c] = np.roll(e2.loc[prev, c].to_numpy(), 7)
    assert _gap(_cold(e2, rid, 3.0), base) == 0.0
    assert _gap(_cold(_reorder(ent, {rid - 1}, 0, reverse=True), rid, 3.0), base) > 1e-4
    # the universe rows of every race are in finishing order (pos ascending), which is what the fit relies on
    assert all((g["pos"].to_numpy() == np.sort(g["pos"].to_numpy())).all() for _, g in ent.groupby("race_id"))


@needs_real
@pytest.mark.parametrize("season,rnd", RACES)
def test_truncated_universe_gives_the_same_score_at_the_exact_optimum(ent, season, rnd):
    """The objective is strictly convex (ridge on every block): fitted to tight tolerance, deleting every row after race r (and so
    every later driver, car and level) must not change race r's score. The loose default tolerance (pgtol 1e-5) leaves ~1e-3."""
    rid = _rid(ent, season, rnd)

    def tight(u, lam_e):
        m = sup_model(u, lam_e) if lam_e is not None else LatentModel(u, dynamic=True, decomp=True, ranking=True, **KW)
        rows = m.race_rows[rid]
        first = int(rows.min())
        w, pad = m.fit_inputs(first, float(rid))
        res = minimize(m._objective, m.theta, args=(first, w, pad), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=20000, maxfun=100000, gtol=1e-12, ftol=1e-16, maxcor=40))
        m.theta = res.x
        return m.components(rows)["u"].to_numpy()

    cut = ent[ent["race_id"] <= rid].reset_index(drop=True)
    for lam_e in (3.0, None):                       # None = plain LatentModel as a control for the method itself
        a, b = tight(ent, lam_e), tight(cut, lam_e)
        assert np.abs(a - b).max() < 1e-6, (lam_e, np.abs(a - b).max())


@needs_real
def test_supplier_of_race_r_comes_from_its_own_season_row_not_from_later_rows(ent):
    """Changing the table row of a LATER season cannot change an earlier universe's supplier column."""
    pu = pd.read_csv(ROOT / CFG["power_units"])
    pu2 = pu.copy()
    pu2.loc[pu2["season"] >= 2022, "supplier"] = "zzz"
    e0 = P.load_entries()
    e0 = e0[e0["season"] <= CFG["universe_last_season"]].reset_index(drop=True)
    alt = e0.merge(pu2[["season", "constructorId", "supplier"]], on=["season", "constructorId"], how="left")
    assert (alt.loc[alt["season"] <= 2021, "supplier"].to_numpy() == ent.loc[ent["season"] <= 2021, "supplier"].to_numpy()).all()
    assert (alt.loc[alt["season"] >= 2022, "supplier"] == "zzz").all()


# ====================================================================================== (4) supplier lookup
@needs_real
def test_universe_adds_only_the_supplier_column_and_matches_the_table_row_of_the_season(ent):
    e0 = P.load_entries()
    e0 = e0[e0["season"] <= CFG["universe_last_season"]].reset_index(drop=True)
    assert len(ent) == len(e0) and ent.drop(columns="supplier").equals(e0)
    pu = pd.read_csv(ROOT / CFG["power_units"])
    assert not pu.duplicated(["season", "constructorId"]).any()
    key = {(r.season, r.constructorId): r.supplier for r in pu.itertuples()}
    want = [key[(s, c)] for s, c in zip(ent["season"], ent["constructorId"])]
    assert list(ent["supplier"]) == want and ent["supplier"].notna().all()
    # season-level: the same constructor changes supplier between seasons (mclaren renault -> mercedes at 2021, alpine -> mercedes 2026)
    mc = ent[ent["constructorId"] == "mclaren"].groupby("season")["supplier"].agg(lambda s: set(s))
    assert mc[2020] == {"renault"} and mc[2021] == {"mercedes"} and mc[2019] == {"renault"}
    assert (ent.groupby(["season", "constructorId"])["supplier"].nunique() == 1).all()


@needs_real
def test_pooling_is_by_supplier_value_not_constructor_id(ent):
    m = sup_model(ent, 10.0)
    lev = m.sup_levels
    inv = {i: k for k, i in m.car.items()}                              # car index -> (lineage, season)

    def level(lineage: str, season: int) -> int:
        return int(m.car_e[m.car[(lineage, season)]])

    for s in (2021, 2022, 2023):
        # one level per supplier-season: ferrari, honda, mercedes, renault
        assert {sup for (sup, y) in lev if y == s} == {"ferrari", "honda", "mercedes", "renault"}
        # Ferrari customers pool with the works team under different constructorIds / lineages
        assert level("sauber_lineage", s) == level("haas", s) == level("ferrari", s) == lev[("ferrari", s)]
        assert level("williams", s) == level("aston_lineage", s) == level("mercedes", s) == lev[("mercedes", s)]
        assert level("alpine_lineage", s) == lev[("renault", s)]
        # honda (RBPT-badged in 2022) pools Red Bull with the sister team (constructorId alphatauri)
        assert level("red_bull", s) == level("rb_lineage", s) == lev[("honda", s)]
    assert level("mclaren", 2020) == lev[("renault", 2020)] and level("mclaren", 2021) == lev[("mercedes", 2021)]
    assert level("sauber_lineage", 2026) == lev[("audi", 2026)] and level("sauber_lineage", 2025) == lev[("ferrari", 2025)]
    assert level("rb_lineage", 2026) == level("red_bull", 2026) == lev[("red_bull_powertrains", 2026)]
    # levels are never shared across seasons (no carry-over): same supplier, different level
    assert lev[("ferrari", 2021)] != lev[("ferrari", 2022)]
    # every car-season has exactly one level and levels are used by at least one car
    assert set(m.car_e.tolist()) == set(lev.values()) and m.nE == len(lev)
    # the lineage of a constructor whose id changed (alfa -> sauber, 2023 -> 2024) stays on one supplier value
    assert level("sauber_lineage", 2023) == lev[("ferrari", 2023)] and level("sauber_lineage", 2024) == lev[("ferrari", 2024)]


@needs_real
def test_rbpt_2022_is_honda_in_the_table_and_the_model(ent):
    pu = pd.read_csv(ROOT / CFG["power_units"])
    r22 = pu[pu["season"] == 2022].set_index("constructorId")
    assert r22.loc["red_bull", "supplier"] == "honda" and r22.loc["alphatauri", "supplier"] == "honda"
    assert "RBPT" in r22.loc["red_bull", "notes"]
    assert set(ent.loc[(ent["season"] == 2022) & ent["constructorId"].isin(["red_bull", "alphatauri"]), "supplier"]) == {"honda"}
    assert not (pu["supplier"] == "rbpt").any()


# ====================================================================================== (5) e exactly 0 before its first race
@needs_real
def test_e_is_exactly_zero_for_supplier_seasons_before_their_first_race(ent):
    m = sup_model(ent, 3.0)
    first = int(m.race_rows[_rid(ent, 2022, 1)].min())
    m.fit(first, float(_rid(ent, 2022, 1)))
    eff = m.supplier_effects()
    assert (eff.loc[eff["season"] >= 2022, "e"] == 0.0).all()                      # exactly, not approximately
    assert (eff.loc[eff["season"] == 2021, "e"] != 0.0).all() and (eff["e"].abs() > 1e-6).sum() >= 4
    # one race later the 2022 levels have moved (non-vacuous), 2023+ still exactly 0
    nxt = int(m.race_rows[_rid(ent, 2022, 2)].min())
    m.fit(nxt, float(_rid(ent, 2022, 2)))
    eff = m.supplier_effects()
    assert (eff.loc[(eff["season"] == 2022), "e"].abs() > 0).all() and (eff.loc[eff["season"] >= 2023, "e"] == 0.0).all()
    # before any race at all: everything 0
    m0 = sup_model(ent, 3.0)
    m0.fit(int(m0.race_rows[1].min()), 1.0)
    # race 0 only: levels of 2014 may move; every later season must be exactly 0
    assert (m0.supplier_effects().query("season >= 2015")["e"] == 0.0).all()


@needs_real
def test_a_supplier_season_that_enters_mid_season_stays_at_zero_until_its_first_race(ent):
    """Synthetic: McLaren's rows of 2022 rounds 1-4 are deleted and McLaren gets a new supplier label for 2022; the new level has no rows
    before round 5, so it must be exactly 0 in a fit before round 5 and move in a fit after."""
    u = ent[ent["season"] <= 2022].copy()
    r1, r4 = _rid(ent, 2022, 1), _rid(ent, 2022, 4)
    u = u[~((u["constructorId"] == "mclaren") & (u["race_id"] >= r1) & (u["race_id"] <= r4))].reset_index(drop=True)
    u.loc[(u["season"] == 2022) & (u["constructorId"] == "mclaren"), "supplier"] = "newcomer"
    m = sup_model(u, 3.0)
    lv = m.sup_levels[("newcomer", 2022)]
    before = int(m.race_rows[r4].min())
    m.fit(before, float(r4))
    assert m.theta[m.nbase + lv] == 0.0
    assert (m.supplier_effects().query("season == 2022 and supplier != 'newcomer'")["e"] != 0.0).all()
    r5 = _rid(ent, 2022, 5)
    after = int(m.race_rows[r5].min()) + 2 * len(m.race_rows[r5])        # includes round 5 and 6
    m.fit(after, float(r5 + 2))
    assert m.theta[m.nbase + lv] != 0.0


# ====================================================================================== (6) stored runs reproduce
def _cand_cfg(lam_e: float) -> dict:
    return dict(LAT, kind="latent_supplier", lam_e=lam_e)


@needs_real
@needs_runs
def test_stored_candidate_run_equals_a_fresh_run_at_lam_e_100_all_66_races(ent):
    summ = json.loads((RES / "tuning_summary.json").read_text(encoding="utf-8"))
    cfg = load_yaml(_dir(LABELS["M14_supplier_pool_latent_111"]) / "config.yaml")
    assert cfg["lam_e"] == summ["lam_e_chosen"] == 100.0 and cfg["kind"] == "latent_supplier"
    p, per = walk_forward(ent, build_model(_cand_cfg(100.0), ent), "dev", refit_every=1)
    st = pd.read_parquet(_dir(LABELS["M14_supplier_pool_latent_111"]) / "predictions.parquet")
    d, n = U.score_diff(p, st)
    assert n == len(st) == 1311 and st["race_id"].nunique() == 66 and d <= 1e-9, d
    pm = pd.read_csv(_dir(LABELS["M14_supplier_pool_latent_111"]) / "per_race_metrics.csv")
    for c in ("spearman", "mae", "exact"):
        assert pm[c].to_numpy(float) == pytest.approx(per[c].to_numpy(float), abs=1e-9)
    # the candidate differs from the reference (non-vacuous)
    ref = pd.read_parquet(_dir(LABELS["latent_111"]) / "predictions.parquet")
    assert U.score_diff(st, ref)[0] > 1e-3


@needs_real
@needs_runs
def test_stored_candidate_first_races_from_a_manual_warm_start_loop(ent):
    """The loop run.py's harness performs, written out: first 4 dev races (15 warm-up fits, then the counted ones)."""
    m = sup_model(ent, 100.0)
    warm, counted = split_races(ent, "dev")
    st = pd.read_parquet(_dir(LABELS["M14_supplier_pool_latent_111"]) / "predictions.parquet").set_index(["race_id", "driverId"])["score"]
    for r in warm + counted[:4]:
        rows = m.race_rows[r]
        s = m.fit_predict(ent.iloc[:int(rows.min())], ent.iloc[rows])
        if r in counted[:4]:
            want = st.loc[[(r, d) for d in ent.iloc[rows]["driverId"]]].to_numpy()
            assert np.abs(s - want).max() <= 1e-9


@needs_real
@needs_runs
def test_tuning_table_reproduced_by_refit_every_4_and_the_pick_is_its_argmax(ent):
    t = pd.read_csv(RES / "tuning_lam_e.csv")
    summ = json.loads((RES / "tuning_summary.json").read_text(encoding="utf-8"))
    assert len(t) == summ["configurations_evaluated"] == 6 <= 30 and summ["budget"] == 30
    assert list(t["lam_e"]) == CFG["lam_e_grid"] == summ["grid"] and (t["n_races"] == 66).all() and CFG["tune_refit_every"] == 4
    assert (t["spearman"] == t["spearman"].max()).sum() == 1 and float(t.loc[t["spearman"].idxmax(), "lam_e"]) == summ["lam_e_chosen"]
    assert t[["spearman", "mae", "exact", "ref_spearman"]].notna().all().all() and not t.duplicated("lam_e").any()
    _, mref = walk_forward(ent, build_model(dict(LAT), ent), "dev", refit_every=4)
    assert t["ref_spearman"].nunique() == 1 and float(t["ref_spearman"].iloc[0]) == pytest.approx(float(mref["spearman"].mean()), abs=1e-12)
    for _, row in t.iterrows():
        _, per = walk_forward(ent, build_model(_cand_cfg(float(row["lam_e"])), ent), "dev", refit_every=4)
        assert len(per) == 66
        for c in ("spearman", "mae", "exact"):
            assert float(per[c].mean()) == pytest.approx(float(row[c]), abs=1e-9), (row["lam_e"], c)
    # the tuned value is chosen on dev rho only; its margin over the reference at refit 4 is tiny (reported, not asserted as a finding)
    pick = t[t["lam_e"] == summ["lam_e_chosen"]].iloc[0]
    assert abs(pick["spearman"] - pick["ref_spearman"]) < 1e-3


# ====================================================================================== (7) paired gains and attribution row
@pytest.fixture(scope="module")
def tabs():
    out = {}
    for k, label in LABELS.items():
        out[k] = A.race_table(U.outcome_free(pd.read_parquet(_dir(label) / "predictions.parquet"), CFG["tie_break"]))
    pr = pd.read_parquet(_dir(LABELS["latent_111"]) / "predictions.parquet")
    gp = pr[["race_id", "season", "driverId", "pos", "grid", "classified"]].copy()
    gp["score"] = np.concatenate([DEC.grid_utility(g["grid"].to_numpy(float), g["driverId"].to_numpy())
                                  for _, g in gp.groupby("race_id", sort=False)])
    out["grid"] = A.race_table(gp)
    out["post"] = A.race_table(U.outcome_free(pd.read_parquet(_dir(POST_LABEL) / "predictions.parquet"), CFG["tie_break"]))
    return out


REF, CAND = "latent_111", "M14_supplier_pool_latent_111"
SEAS_METRICS = ["spearman", "mae", "exact", "spearman_fin", "mae_fin"]


@needs_runs
def test_run_folders_are_complete_dev_only_and_agree_with_predictions(tabs):
    for k, label in LABELS.items():
        d = _dir(label)
        pr = pd.read_parquet(d / "predictions.parquet")
        assert pr["season"].between(2021, 2023).all() and pr["race_id"].nunique() == 66 and len(pr) == 1311
        assert not pr.duplicated(["race_id", "driverId"]).any() and pr["score"].notna().all()
        assert pr.groupby("season")["race_id"].nunique().to_dict() == {2021: 22, 2022: 22, 2023: 22}
        for f in ("config.yaml", "metrics.json", "per_race_metrics.csv", "predictions.parquet"):
            assert (d / f).exists(), (label, f)
        assert load_yaml(d / "config.yaml").get("git_commit") and load_yaml(d / "config.yaml")["split"] == "dev"
        pm = pd.read_csv(d / "per_race_metrics.csv")
        assert len(pm) == 66 and pm["season"].between(2021, 2023).all() and not pm.duplicated("race_id").any()
        stored = pd.read_csv(d / "outcome_free_table.csv")
        assert set(tabs[k].columns) - set(stored.columns) <= {"within3"}  # metrics added after these runs were stored (PLAN3 section 5, 2026-11-17); recomputed but not stored
        pd.testing.assert_frame_equal(stored.reset_index(drop=True), tabs[k][stored.columns].reset_index(drop=True), check_exact=False, rtol=0, atol=1e-12)
        t = tabs[k].set_index("race_id")
        pm = pm.set_index("race_id")
        for c in ("spearman", "mae", "exact"):
            assert pm[c].to_numpy(float) == pytest.approx(t.loc[pm.index, c].to_numpy(float), abs=1e-9, nan_ok=True), (k, c)
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        assert m["n_races"] == 66 and m["spearman"] == pytest.approx(float(pm["spearman"].mean()), abs=1e-12)


@needs_runs
def test_paired_gains_dev_rederived(tabs):
    nb = CFG["bootstrap"]
    assert nb == {"n_boot": 10000, "seed": 0} and CFG["tie_break"] == "driverId"
    got = A.paired_gain(tabs[REF], tabs[CAND], n_boot=nb["n_boot"], seed=nb["seed"])
    st = pd.read_csv(RES / "paired_gains_dev.csv")
    assert (st["component"] == "M14_supplier_pool").all() and (st["reference"] == "latent_111").all()
    assert list(st["metric"]) == list(got["metric"]) and len(st) == 12
    for c in ("ref_mean", "cand_mean", "gain", "lo", "hi"):
        assert st[c].to_numpy(float) == pytest.approx(got[c].to_numpy(float), abs=1e-12, nan_ok=True), c
    assert list(st["n"]) == list(got["n"]) and set(st["n"]) == {66}
    assert list(st["verdict"]) == [A.verdict(r) for _, r in got.iterrows()]
    # gain = mean of the per-race difference in the stored per_race_metrics.csv (sign: positive = better)
    a = pd.read_csv(_dir(LABELS[REF]) / "per_race_metrics.csv").set_index("race_id")
    b = pd.read_csv(_dir(LABELS[CAND]) / "per_race_metrics.csv").set_index("race_id")
    g = got.set_index("metric")
    assert g.loc["spearman", "gain"] == pytest.approx(float((b["spearman"] - a["spearman"]).mean()), abs=1e-9)
    assert g.loc["mae", "gain"] == pytest.approx(float((a["mae"] - b["mae"]).mean()), abs=1e-9)
    assert (g["lo"] <= g["gain"] + 1e-12).all() and (g["gain"] <= g["hi"] + 1e-12).all()
    # means.csv agrees with the tables
    mn = pd.read_csv(RES / "means.csv", index_col=0)
    for k in ("latent_111", "M14_supplier_pool_latent_111", "grid"):
        for c in mn.columns:
            if c in tabs[k] and not np.isnan(mn.loc[k, c]):
                assert float(tabs[k][c].mean()) == pytest.approx(float(mn.loc[k, c]), abs=1e-12), (k, c)


@needs_runs
def test_paired_gains_by_season_rederived(tabs):
    st = pd.read_csv(RES / "paired_gains_by_season.csv")
    assert len(st) == 15
    for s in (2021, 2022, 2023):
        a, b = tabs[REF][tabs[REF]["season"] == s], tabs[CAND][tabs[CAND]["season"] == s]
        got = A.paired_gain(a, b, SEAS_METRICS, n_boot=10000, seed=0)
        ss = st[st["season"] == s].reset_index(drop=True)
        assert list(ss["metric"]) == list(got["metric"]) and (ss["n"] == 22).all() and (got["n"] == 22).all()
        for c in ("ref_mean", "cand_mean", "gain", "lo", "hi"):
            assert ss[c].to_numpy(float) == pytest.approx(got[c].to_numpy(float), abs=1e-12), (s, c)
    # exactly two of the 15 by-season intervals exclude 0, both 2022 on finishers (README must not say "one")
    ex = st[(st["lo"] > 0) | (st["hi"] < 0)]
    assert sorted(zip(ex["season"], ex["metric"])) == [(2022, "mae_fin"), (2022, "spearman_fin")] and (ex["gain"] < 0).all()


@needs_runs
def test_exact_vs_grid_rederived(tabs):
    st = pd.read_csv(RES / "exact_vs_grid.csv")
    assert len(st) == 4
    for cand, key in (("M14_supplier_pool", CAND), ("latent_111 (reference)", REF)):
        got = A.paired_gain(tabs["grid"], tabs[key], ["exact", "exact_fin"], n_boot=10000, seed=0)
        s = st[st["candidate"] == cand].reset_index(drop=True)
        assert list(s["metric"]) == ["exact", "exact_fin"]
        for c in ("ref_mean", "cand_mean", "gain", "lo", "hi"):
            assert s[c].to_numpy(float) == pytest.approx(got[c].to_numpy(float), abs=1e-12), (cand, c)
        assert list(s["verdict"]) == [A.verdict(r) for _, r in got.iterrows()] and (s["n"] == 66).all()
    # the grid reference is the pure starting-grid order with the driverId tie-break: independent re-derivation of its exact rate
    pr = pd.read_parquet(_dir(LABELS[REF]) / "predictions.parquet")
    hit = []
    for _, g in pr.groupby("race_id"):
        order = g.assign(k=np.where(g["grid"] <= 0, g["grid"].max() + 1, g["grid"])).sort_values(["k", "driverId"])
        pred_place = pd.Series(np.arange(1, len(g) + 1), index=order.index)
        hit.append(float((pred_place.loc[g.index].to_numpy() == g["pos"].to_numpy()).mean()))
    assert float(np.mean(hit)) == pytest.approx(float(tabs["grid"]["exact"].mean()), abs=1e-12)


@needs_runs
def test_attribution_row_rederived(tabs):
    att = pd.read_csv(ROOT / "results/plan3/attribution.csv")
    rows = att[att["component"].str.startswith("M14")]
    assert len(rows) == 2 and list(rows["component"]) == ["M14_supplier_pool", POST_COMPONENT]
    row = rows.iloc[0]
    assert row["component"] == "M14_supplier_pool" and row["reference"] == "latent_111" and row["declared"] == "yes"
    got = A.paired_gain(tabs[REF], tabs[CAND], n_boot=10000, seed=0).set_index("metric")
    n = 0
    for m in got.index:
        if f"gain_{m}" in row.index and not pd.isna(row[f"gain_{m}"]):
            n += 1
            for pre in ("gain", "lo", "hi"):
                assert float(row[f"{pre}_{m}"]) == pytest.approx(float(got.loc[m, pre]), abs=1e-12), (m, pre)
    assert n == 12
    assert row["verdict_metric"] == "spearman" and row["verdict"] == A.verdict(got.loc["spearman"]) == "no measurable difference"
    assert int(row["n_races"]) == 66 and row["run_id"] == _dir(LABELS[CAND]).name
    assert pd.isna(row["loo_gain"]) and pd.isna(row["fresh_holdout_gain"])
    assert pd.isna(row["gain_rps"]) and pd.isna(row["lo_rps"])              # no RPS invented for a model with no simulator run


@needs_runs
def test_posthoc_attribution_row_rederived(tabs):
    att = pd.read_csv(ROOT / "results/plan3/attribution.csv")
    row = att[att["component"] == POST_COMPONENT]
    assert len(row) == 1
    row = row.iloc[0]
    assert row["reference"] == "latent_111" and row["declared"] == "no"              # post hoc: not declared
    got = A.paired_gain(tabs[REF], tabs["post"], n_boot=10000, seed=0).set_index("metric")
    n = 0
    for m in got.index:
        if f"gain_{m}" in row.index and not pd.isna(row[f"gain_{m}"]):
            n += 1
            for pre in ("gain", "lo", "hi"):
                assert float(row[f"{pre}_{m}"]) == pytest.approx(float(got.loc[m, pre]), abs=1e-12), (m, pre)
    assert n == 12
    assert row["verdict_metric"] == "spearman" and row["verdict"] == A.verdict(got.loc["spearman"]) == "worse"
    assert int(row["n_races"]) == 66 and row["run_id"] == _dir(POST_LABEL).name
    assert pd.isna(row["loo_gain"]) and pd.isna(row["fresh_holdout_gain"]) and pd.isna(row["gain_rps"])
    # the posthoc csv is the same computation
    st = pd.read_csv(RES / "paired_gains_dev_posthoc.csv")
    assert (st["component"] == POST_COMPONENT).all() and list(st["metric"]) == list(got.reset_index()["metric"]) and len(st) == 12
    for c in ("ref_mean", "cand_mean", "gain", "lo", "hi"):
        assert st[c].to_numpy(float) == pytest.approx(got.reset_index()[c].to_numpy(float), abs=1e-12, nan_ok=True), c
    assert list(st["verdict"]) == [A.verdict(r) for _, r in got.reset_index().iterrows()]


# ====================================================================================== (8) shared files
def _git_bytes(rev: str, rel: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", f"{rev}:{rel}"], cwd=ROOT, stderr=subprocess.DEVNULL)
    except Exception as e:                                              # pragma: no cover
        pytest.skip(f"git show failed: {e}")


@pytest.mark.parametrize("rel,prefix,n_m14", [("results/plan3/attribution.csv", "M14", 2), ("results/plan3/leaderboard.csv", "M14,", 3)])
@pytest.mark.parametrize("rev", ["HEAD", BASE])
def test_shared_files_earlier_lines_are_a_byte_prefix_and_m14_adds_exactly_its_rows(rel, prefix, n_m14, rev):
    old = _git_bytes(rev, rel)
    cur = (ROOT / rel).read_bytes()
    assert cur.startswith(old), f"a line present at {rev} was rewritten (byte comparison)"
    assert b"\r\n" not in cur or b"\r\n" in old
    lines = cur.decode("utf-8").splitlines()
    mine = [ln for ln in lines if ln.startswith(prefix)]
    if rev == BASE:
        assert not [ln for ln in old.decode("utf-8").splitlines() if ln.startswith("M14")]
    assert len(mine) == n_m14, (rel, len(mine))
    c = pd.read_csv(StringIO(cur.decode("utf-8")), dtype=str)
    o = pd.read_csv(StringIO(old.decode("utf-8")), dtype=str)
    assert list(o.columns) == list(c.columns)
    pd.testing.assert_frame_equal(c.iloc[:len(o)].reset_index(drop=True), o.reset_index(drop=True), check_exact=True)
    assert len(c) == len(lines) - 1
    if "component" in c.columns:
        assert not c.duplicated(["component", "reference"]).any()           # no repeated names anywhere in the file
    else:
        assert not c.duplicated("label").any() and not c.duplicated("run_id").any()
        assert (c["method"] == "M14").sum() == 3 and set(c.loc[c["method"] == "M14", "label"]) == set(LABELS.values()) | {POST_LABEL}


@needs_runs
def test_leaderboard_m14_rows_match_their_run_folders():
    lb = pd.read_csv(ROOT / "results/plan3/leaderboard.csv", dtype={"cell": str})
    rows = lb[lb["method"] == "M14"]
    assert len(rows) == 3 and set(rows["label"]) == set(LABELS.values()) | {POST_LABEL} and (rows["split"] == "dev").all() and (rows["cell"] == "111").all()
    for _, r in rows.iterrows():
        d = ROOT / r["run_dir"].replace("\\", "/")
        assert d.exists() and d.name == r["run_id"] and d.name.endswith(f"_{r['label']}_dev") and r["n_races"] == 66
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        pm = pd.read_csv(d / "per_race_metrics.csv")
        for c in ("spearman", "mae", "exact", "logloss_points"):
            assert float(r[c]) == pytest.approx(m[c], abs=1e-12) and m[c] == pytest.approx(float(pm[c].mean()), abs=1e-12)
    assert not (RES / "_lb_tmp.csv").exists()
    for rel in ("results/leaderboard.csv", "results/plan2/leaderboard.csv"):
        assert not (pd.read_csv(ROOT / rel, dtype=str)["method"] == "M14").any(), rel


# ====================================================================================== (9) split discipline, data, README
def _int_constants(path) -> set[int]:
    return {n.value for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.Constant)
            and isinstance(n.value, int) and not isinstance(n.value, bool)}


def test_code_has_no_test_season_or_holdout_constant_and_run_asserts_dev_only():
    for f in (METH / "run.py", ROOT / "src/pitlane/models/latent_supplier.py"):
        assert not {c for c in _int_constants(f) if 2024 <= c <= 2029}, f
        assert "fresh" not in f.read_text(encoding="utf-8").lower(), f
    src = (METH / "run.py").read_text(encoding="utf-8")
    assert 'assert cfg["split"] == "dev"' in src and CFG["split"] == "dev"
    assert 'max(int(p["season"].max()) for _, p, _ in runs.values()) <= 2023' in src
    txt = (METH / "config.yaml").read_text(encoding="utf-8")
    # the only season >= 2024 literal in the config is the universe horizon (rows are loaded, never fitted or scored)
    assert re.findall(r"\b202[4-9]\b", txt) == ["2026"] and CFG["universe_last_season"] == 2026 and "fresh" not in txt.lower()
    # tuned quantity: dev mean rho only
    assert 'tune.loc[tune["spearman"].idxmax(), "lam_e"]' in src and CFG["verdict_metric"] == "spearman"


@needs_real
def test_no_race_of_2024_or_later_is_fitted_or_scored_in_the_dev_walk(ent):
    """Instrument fit_predict: every training slice ends before the target chunk, targets are all <= 2023, and no 2024+ row is used."""
    seen = []

    class Spy(LatentSupplierModel):
        def fit_predict(self, train, test):
            seen.append((int(train["season"].max()), int(test["season"].max()), int(train["race_id"].max()), int(test["race_id"].min())))
            assert int(self.row_season[:len(train)].max()) <= 2023
            return super().fit_predict(train, test)

    spy = Spy(ent, dynamic=True, decomp=True, ranking=True, lam_e=100.0, **KW)
    _, per = walk_forward(ent, spy, "dev", refit_every=1)
    assert len(seen) == 81 and max(s[0] for s in seen) <= 2023 and max(s[1] for s in seen) <= 2023
    assert all(s[2] < s[3] for s in seen)
    assert per["season"].between(2021, 2023).all() and len(per) == 66
    t = pd.read_csv(RES / "tuning_lam_e.csv")
    assert (t["n_races"] == 66).all()
    for k, label in LABELS.items():
        assert pd.read_csv(_dir(label) / "per_race_metrics.csv")["season"].max() <= SPLITS["dev"][1]


@needs_real
def test_universe_data_checks(ent):
    assert not ent.duplicated(["race_id", "driverId"]).any() and ent["supplier"].notna().all()
    per = ent.groupby("season")["race_id"].nunique()
    assert per.loc[2021] == per.loc[2022] == per.loc[2023] == 22
    dev = ent[ent["season"].between(2021, 2023)]
    assert len(dev) == 1311 and dev["race_id"].nunique() == 66
    assert (ent["race_id"].diff().fillna(0) >= 0).all()                     # sorted by race: row order = training order
    assert set(ent["supplier"]) <= {"mercedes", "ferrari", "renault", "honda", "red_bull_powertrains", "audi"}


def test_power_unit_table_rows_have_a_source_and_dev_rows_are_season_start_facts():
    pu = pd.read_csv(ROOT / CFG["power_units"], dtype=str)
    assert pu["source"].notna().all() and pu["source"].str.startswith("https://").all()
    assert not pu.duplicated(["season", "constructorId"]).any() and pu["supplier"].notna().all() and pu["works"].isin(["0", "1"]).all()
    # one season-year source per row: the URL names the same season as the row
    assert (pu["source"].str.extract(r"/(\d{4})_Formula")[0] == pu["season"]).all()
    # dev rows recalled from F1 history (UNVERIFIED against a source): 10 teams x 3 seasons
    want = {"alfa": "ferrari", "alphatauri": "honda", "alpine": "renault", "aston_martin": "mercedes", "ferrari": "ferrari",
            "haas": "ferrari", "mclaren": "mercedes", "mercedes": "mercedes", "red_bull": "honda", "williams": "mercedes"}
    d = pu[pu["season"].isin(["2021", "2022", "2023"])]
    assert len(d) == 30 and all(want[r.constructorId] == r.supplier for r in d.itertuples())
    assert len(pu) == 133


@needs_runs
def test_stored_checks_json_agree_with_the_thresholds_and_the_readme():
    chk = json.loads((RES / "checks.json").read_text(encoding="utf-8"))
    for v in chk["gradient"].values():
        assert v["max_rel_err"] < 1e-8 and v["max_abs_grad_e"] > 1e-2 and v["n_e"] == 11
    assert set(chk["gradient"]) == {"lam_e=3.0", "lam_e=32.0"}
    assert chk["off_switch"]["bitwise_equal_off"] is True and chk["off_switch"]["max_abs_score_diff_off"] == 0.0
    lk = chk["leakage"]
    assert lk["max_abs_diff_scrambled_outcomes_and_later_suppliers"] < 1e-12 and len(lk["races"]) == 6
    assert lk["max_abs_diff_truncated_universe"] < 2e-3 and lk["max_abs_diff_truncated_universe_plain_latent_model"] < 2e-3


def _flat(txt: str) -> str:
    return re.sub(r"\s+", " ", txt)


@needs_runs
def test_readme_numbers_match_the_csvs_and_json(tabs):
    readme = _flat((METH / "README.md").read_text(encoding="utf-8"))
    g = pd.read_csv(RES / "paired_gains_dev.csv").set_index("metric")

    def fmt(r, nd=4):
        return f"{r['gain']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]"
    for text in (f"rho {fmt(g.loc['spearman'])}", f"rho on finishers {fmt(g.loc['spearman_fin'])}",
                 f"MAE (lower is better, sign flipped so positive = better) {fmt(g.loc['mae'])}", f"exact rate {fmt(g.loc['exact'])}"):
        assert text in readme, text
    sb = pd.read_csv(RES / "paired_gains_by_season.csv")
    s = sb[sb["metric"] == "spearman"].set_index("season")
    assert "By season rho: " + ", ".join(f"{y} {fmt(s.loc[y])}" for y in (2021, 2022, 2023)) in readme
    f22 = sb[(sb["season"] == 2022)].set_index("metric")
    assert f"2022 rho on finishers {fmt(f22.loc['spearman_fin'])}" in readme and f"MAE on finishers {fmt(f22.loc['mae_fin'])}" in readme
    vg = pd.read_csv(RES / "exact_vs_grid.csv")
    c1, c2 = vg[(vg["candidate"] == "M14_supplier_pool") & (vg["metric"] == "exact")].iloc[0], vg[(vg["candidate"] == "latent_111 (reference)") & (vg["metric"] == "exact")].iloc[0]
    assert (f"(0.1638): candidate {c1['cand_mean']:.4f}, gap {fmt(c1)}; latent_111 {c2['cand_mean']:.4f}, gap {fmt(c2)}") in readme
    t = pd.read_csv(RES / "tuning_lam_e.csv")
    seq = ", ".join(f"{l:g} -> {x:.4f}" for l, x in zip(t["lam_e"], t["spearman"]))
    assert f"{seq}; reference at refit_every 4: {t['ref_spearman'].iloc[0]:.4f}" in readme
    mn = pd.read_csv(RES / "means.csv", index_col=0)
    assert (f"| m14_ref_latent_111 | dev | {mn.loc['latent_111', 'spearman']:.4f} | {mn.loc['latent_111', 'mae']:.3f} | {mn.loc['latent_111', 'exact']:.4f} |") in readme
    c = mn.loc["M14_supplier_pool_latent_111"]
    assert f"| {c['spearman']:.4f} | {c['mae']:.3f} | {c['exact']:.4f} |" in readme
    ds = json.loads((RES / "descriptive.json").read_text(encoding="utf-8"))
    e = pd.DataFrame(ds["e_end_of_dev"])
    assert e[e["season"].between(2021, 2023)]["e"].abs().max() == pytest.approx(0.0662, abs=1e-12) and "all |e| <= 0.066" in readme
    vs = {r["season"]: r for r in ds["variance_share_end_of_dev"]}
    assert "2021 %.1f%%, 2022 %.1f%%, 2023 %.1f%%" % tuple(100 * vs[y]["share_e"] for y in (2021, 2022, 2023)) in readme
    assert "-6.5%, +9.5%, +13.0%" in readme and tuple(round(100 * vs[y]["share_cov"], 1) for y in vs) == (-6.5, 9.5, 13.0)
    rep = json.loads((RES / "reproduction_check.json").read_text(encoding="utf-8"))
    assert f"{rep['latent_111']['max_abs_diff_per_race_metrics']:.1e}" == "8.9e-16" and "8.9e-16" in readme


@needs_real
@needs_runs
def test_descriptive_fitted_e_reproduced(ent):
    ds = json.loads((RES / "descriptive.json").read_text(encoding="utf-8"))
    dev = R.dev_race_ids(ent)
    last = int(dev[-1])
    n_end = int(ent.index[ent["race_id"] == last].max()) + 1
    m = R.fit_e_at(ent, LAT, 100.0, n_end, float(last + 1))
    eff = m.supplier_effects()
    eff = eff[eff["season"].between(2021, 2023)].set_index(["supplier", "season"])["e"]
    for r in ds["e_end_of_dev"]:
        if 2021 <= r["season"] <= 2023:
            assert eff.loc[(r["supplier"], r["season"])] == pytest.approx(r["e"], abs=6e-5)
    vs = R.variance_share(m, [2021, 2022, 2023])
    for a, b in zip(vs, ds["variance_share_end_of_dev"]):
        assert a["share_e"] == pytest.approx(b["share_e"], abs=1e-6) and a["n_cars"] == b["n_cars"] == 10


PRESTATEMENT_DECLARED_SHA256 = "fa83bd7dd77a60cad7425e0b24ca1b446d9a34fe7cfb742f8db740d64e4737c1"   # pinned 2026-11-17 (after the post-hoc append)


def _prestatement_parts() -> tuple[str, str]:
    txt = (METH / "prestatement.md").read_text(encoding="utf-8").replace(chr(13) + chr(10), chr(10))
    heads = [m.start() for m in re.finditer(r"^# POST-HOC variant", txt, flags=re.M)]
    assert len(heads) == 1, heads
    return txt[:heads[0]], txt[heads[0]:]


def test_prestatement_declared_part_is_unchanged_and_the_posthoc_section_states_its_rule():
    """Content check, NOT a proof of order in time: the declared text (everything before the POST-HOC heading) must hash to the value pinned
    when this test was written. That shows the declared text is unchanged from now on; it cannot show that it was written before the
    first run (mtimes are reset by a checkout, and the section was appended after the declared model's dev result was seen)."""
    import hashlib
    declared, post = _prestatement_parts()
    assert hashlib.sha256(declared.encode("utf-8")).hexdigest() == PRESTATEMENT_DECLARED_SHA256
    assert "Expected gain on dev: about 0 rho" in declared and "Failure:" in declared and "[0.3, 1, 3, 10, 32, 100]" in declared
    assert [0.3, 1, 3, 10, 32, 100] == CFG["lam_e_grid"] and "6 configurations, budget 30" in declared
    assert "POST-HOC" not in declared and "post hoc" not in declared.lower()
    # the section appended later states the rule: label, expected gain, failure condition, no decision weight
    low = post.lower()
    assert "post hoc" in low and "written before the variant was run" in low
    assert "Expected gain on dev: about 0 rho" in post and "Failure: the paired per-race rho interval" in post and "includes 0" in post
    assert "no decision weight" in low
    assert "lam_e grid [1, 3, 10, 32, 100, 300]" in post and "6 configurations" in post and "12 of 30" in post
    assert [1.0, 3.0, 10.0, 32.0, 100.0, 300.0] == CFG["posthoc_variant"]["lam_e_grid"] and CFG["posthoc_variant"]["lam_b"] == 32.0 == CFG["posthoc_variant"]["lam_b"]
    # revision 2: a dated correction note is appended BELOW the post-hoc section; the section above it is not rewritten
    i_post, i_corr = post.index("# POST-HOC variant"), post.index("# Correction note, dated 2026-11-17 (revision 2")
    assert i_post < i_corr and post.count("# Correction note") == 1
    corr = post[i_corr:]
    assert "lam_e = inf" in corr and "Look-ahead defect, fixed" in corr and "no new configuration, no extra budget" in corr
    assert "`lam_b` is written as 32.0 (= lam)" in corr and "marked superseded" in corr
    assert "the limit `lam_e -> inf` is therefore" in post[:i_corr]          # the (wrong) original sentence is kept, corrected only by the note


# ====================================================================================== (10) optimiser convergence
def _walk_with_diag(ent: pd.DataFrame, model, maxiter: int, record: set[int]):
    warm, counted = split_races(ent, "dev")
    rho, diag = [], {}
    for r in warm + counted:
        rows = model.race_rows[r]
        first = int(rows.min())
        w, pad = model.fit_inputs(first, float(r))
        res = minimize(model._objective, model.theta, args=(first, w, pad), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
        model.theta = res.x
        if r in counted:
            d = ent.iloc[rows]
            rho.append(race_metrics(d["pos"].to_numpy(), model.components(rows)["u"].to_numpy(), d["classified"].to_numpy())["spearman"])
        diag[r] = dict(status=int(res.status), nit=int(res.nit), gmax=float(np.abs(res.jac).max()))
    return float(np.mean(rho)), diag


@needs_real
def test_lbfgs_converges_before_maxiter_150_in_the_warm_walk_and_low_lam_e_loss_is_not_an_optimiser_failure(ent):
    """Measured (not assumed): final projected-gradient max-norm, status and iteration count at three dev races for lam_e 0.3, 100 and the
    plain reference; dev mean rho at maxiter 150 equals dev mean rho at maxiter 3000, so the rho drop at small lam_e is not truncation."""
    dev = R.dev_race_ids(ent)
    picks = {int(dev[0]), int(dev[22]), int(dev[50])}
    makers = {"ref": lambda: LatentModel(ent, dynamic=True, decomp=True, ranking=True, **KW),
              "lam_e=0.3": lambda: sup_model(ent, 0.3), "lam_e=100": lambda: sup_model(ent, 100.0)}
    rho = {}
    for name, mk in makers.items():
        r150, d150 = _walk_with_diag(ent, mk(), 150, picks)
        r3000, d3000 = _walk_with_diag(ent, mk(), 3000, picks)
        assert r150 == r3000, (name, r150, r3000)
        assert all(v["status"] == 0 and v["nit"] < 150 and v["gmax"] < 1e-4 for v in d150.values()), name       # every fit, not only 3
        for p in picks:
            assert d150[p]["status"] == 0 and d150[p]["gmax"] < 1e-5
        rho[name] = r150
    assert rho["lam_e=0.3"] < rho["ref"] - 0.02 and abs(rho["lam_e=100"] - rho["ref"]) < 1e-3


# ====================================================================================== (11) R11 revision: pinning, carry_on_total (post hoc)
def ck_model(u: pd.DataFrame, lam_e, lam_b=None) -> LatentSupplierModel:
    return LatentSupplierModel(u, dynamic=True, decomp=True, ranking=True, lam_e=lam_e, carry_on_total=True, lam_b=lam_b, **KW)


def _pin_setup(ent: pd.DataFrame):
    """Universe up to 2018; training stops before 2017 round 3, so every 2017 level with rows is active and all 2018 levels are inactive."""
    u = ent[ent["season"] <= 2018].reset_index(drop=True)
    rid = _rid(ent, 2017, 3)
    return u, rid


@needs_real
@pytest.mark.parametrize("lam_e,lam_b,seed", [(3.0, None, 21), (32.0, 5.0, 22), (0.7, 32.0, 23), (100.0, 0.5, 24)])
def test_carry_on_total_gradient_matches_finite_differences_and_pinned_levels_have_gradient_exactly_zero(ent, lam_e, lam_b, seed):
    u, rid = _pin_setup(ent)
    m = ck_model(u, lam_e, lam_b)
    assert m.carry_total and m.lam_b == (m.lam if lam_b is None else lam_b) and m.lam_b > 0
    n = int(m.race_rows[rid].min())
    act = m._active(n)
    assert 0 < act.sum() < m.nE and (~act).sum() >= 3                           # both kinds of level exist
    w, pad = m.fit_inputs(n, float(rid))
    th = np.random.default_rng(seed).normal(0, 0.4, len(m.theta))                 # random point; pinned coordinates are non-zero too
    f, g = m._objective(th, n, w, pad)
    num = np.empty_like(g)
    for i in range(len(th)):
        d = np.zeros_like(th)
        d[i] = 1e-6
        num[i] = (m._objective(th + d, n, w, pad)[0] - m._objective(th - d, n, w, pad)[0]) / 2e-6
    err = np.abs(g - num) / np.maximum(1.0, np.abs(num))
    assert err.max() < 1e-7, (err.max(), int(err.argmax()), m.nbase)
    e_idx = m.nbase + np.arange(m.nE)
    assert np.abs(g[e_idx[act]]).max() > 1e-3 and err[e_idx[act]].max() < 1e-7        # active e coordinates: non-vacuous and exact
    assert (g[e_idx[~act]] == 0.0).all()                                                # pinned: analytic gradient exactly 0
    assert np.abs(num[e_idx[~act]]).max() < 1e-7                                        # and the value really does not depend on them
    th2 = th.copy()
    th2[e_idx[~act]] += 3.0
    assert m._objective(th2, n, w, pad)[0] == f
    # the same holds for the default (carry_on_total False) model: pinned gradient exactly 0
    m0 = sup_model(u, lam_e)
    assert (m0._objective(th, n, w, pad)[1][e_idx[~act]] == 0.0).all()


@needs_real
@pytest.mark.parametrize("lam_b", [None, 7.0])
def test_carry_on_total_value_differs_from_the_plain_model_by_the_stated_penalty_terms(ent, lam_b):
    """Independent value check for the fixed objective (revision 2), written from the docstring: carry-on-total minus plain =
    [lam (|dT|^2 over cars WITH a training row - |d|^2 over all cars) + lam_b |b|^2 over cars with a training row] / W, with
    d = b - rho b_prev, dT = T - rho T_prev, T = b + e (e of levels without a training row = 0). Cars without a row have no term."""
    u, rid = _pin_setup(ent)
    a, c = sup_model(u, 10.0), ck_model(u, 10.0, lam_b)
    n = int(a.race_rows[rid].min())
    w, pad = a.fit_inputs(n, float(rid))
    th = np.random.default_rng(9).normal(0, 0.4, len(a.theta))
    fa, fc = a._objective(th, n, w, pad)[0], c._objective(th, n, w, pad)[0]
    b = a._split(th)[1]
    e = np.where(a._active(n), th[a.nbase:], 0.0)
    T = b + e[a.car_e]
    prev, has = a.car_prev, a.car_prev >= 0
    car_has_row = np.zeros(a.nB, dtype=bool)
    car_has_row[a.row_b[:n]] = True
    assert 0 < car_has_row.sum() < a.nB and (~car_has_row).sum() >= 8            # later-season cars exist, with no row
    d = b - np.where(has, a.rho * b[np.where(has, prev, 0)], 0.0)
    dT = T - np.where(has, a.rho * T[np.where(has, prev, 0)], 0.0)
    lb = c.lam if lam_b is None else lam_b
    want = (a.lam * ((dT[car_has_row] ** 2).sum() - d @ d) + lb * (b[car_has_row] ** 2).sum()) / w.sum()
    assert has[car_has_row].sum() > 10 and abs((dT[car_has_row] ** 2).sum() - d @ d) > 1e-3
    assert fc - fa == pytest.approx(want, rel=1e-9, abs=1e-12)


@needs_real
def test_carry_on_total_with_lam_e_off_is_the_plain_latent_model_and_ignores_lam_b(ent):
    """Documented behaviour (design note for the report): lam_e = inf removes the e block AND the carry-on-total/lam_b terms, so the
    off arm is the reference exactly and is NOT 'carry-over on b plus an extra ridge on b' (the limit lam_e -> inf of the on-arm)."""
    u = ent[ent["season"] <= 2016].reset_index(drop=True)
    ref = LatentModel(u, dynamic=True, decomp=True, ranking=True, **KW)
    off = ck_model(u, float("inf"), 32.0)
    rid = int(u["race_id"].max())
    n = int(ref.race_rows[rid].min())
    w, pad = ref.fit_inputs(n, float(rid))
    th = np.random.default_rng(1).normal(0, 0.3, len(ref.theta))
    f0, g0 = ref._objective(th, n, w, pad)
    f1, g1 = off._objective(th, n, w, pad)
    assert f0 == f1 and np.array_equal(g0, g1)


@needs_real
def test_carry_on_total_fit_resets_stale_warm_start_of_unseen_levels(ent):
    m = ck_model(ent, 32.0)
    r1 = _rid(ent, 2022, 1)
    n = int(m.race_rows[r1].min())
    m.theta[m.nbase:] = 0.7                                   # a stale warm start must be reset for unseen levels
    m.fit(n, float(r1))
    eff = m.supplier_effects()
    assert (eff.loc[eff["season"] >= 2022, "e"] == 0.0).all() and (eff.loc[eff["season"] == 2021, "e"] != 0.0).all()


@needs_real
@needs_runs
def test_default_supplier_model_with_explicit_off_switches_reproduces_the_stored_declared_run(ent):
    """carry_on_total=False, lam_b=0 (explicit) at the chosen lam_e = 100, refit before every race: per-race metrics against the stored declared
    run <= 1e-12, scores <= 1e-12 (pinning changed nothing there). lam_b is ignored when carry_on_total is False."""
    cfg = dict(LAT, kind="latent_supplier", lam_e=100.0, carry_on_total=False, lam_b=0.0)
    p, per = walk_forward(ent, build_model(cfg, ent), "dev", refit_every=1)
    d = _dir(LABELS["M14_supplier_pool_latent_111"])
    st = pd.read_parquet(d / "predictions.parquet")
    sd, n = U.score_diff(p, st)
    assert n == len(st) == 1311 and sd <= 1e-12, sd
    pm = pd.read_csv(d / "per_race_metrics.csv")
    diff, nn = U.max_diff(per, pm)
    assert nn == 66 and diff <= 1e-12, diff
    default = LatentSupplierModel(ent, dynamic=True, decomp=True, ranking=True, lam_e=100.0, **KW)
    assert default.carry_total is False and default.lam_b == default.lam
    n0 = int(default.race_rows[_rid(ent, 2022, 5)].min())
    w, pad = default.fit_inputs(n0, 165.0)
    th = np.random.default_rng(2).normal(0, 0.3, len(default.theta))
    zero = LatentSupplierModel(ent, dynamic=True, decomp=True, ranking=True, lam_e=100.0, lam_b=0.0, **KW)
    a, b = default._objective(th, n0, w, pad), zero._objective(th, n0, w, pad)
    assert a[0] == b[0] and np.array_equal(a[1], b[1])


def _ck_cold(u: pd.DataFrame, rid: int, lam_e: float) -> pd.Series:
    m = ck_model(u, lam_e)
    rows = m.race_rows[rid]
    s = m.fit_predict(u.iloc[:int(rows.min())], u.iloc[rows])
    return pd.Series(s, index=u.iloc[rows]["driverId"].to_numpy()).sort_index()


@needs_real
@pytest.mark.parametrize("season,rnd", RACES)
def test_carry_on_total_scores_of_race_r_unchanged_by_scrambling_race_r_and_later(ent, season, rnd):
    rid = _rid(ent, season, rnd)
    base = _ck_cold(ent, rid, 32.0)
    got = _ck_cold(_scramble(ent, rid, rid), rid, 32.0)
    assert len(base) >= 18 and _gap(got, base) <= 1e-12, _gap(got, base)
    # power: the previous race's finishing order does move the score
    assert _gap(_ck_cold(_reorder(ent, {rid - 1}, 0, reverse=True), rid, 32.0), base) > 1e-4


@needs_real
@pytest.mark.parametrize("season,rnd", RACES + [(2021, 1), (2023, 1)])
def test_carry_on_total_truncated_universe_gives_the_same_score_at_the_exact_optimum(ent, season, rnd):
    """Kept as originally written (tolerance 1e-6), and extended to first races of seasons (2021 R1, 2022 R1, 2023 R1)."""
    rid = _rid(ent, season, rnd)

    def tight(u):
        m, rows, _ = _tight_fit(ck_model(u, 32.0), rid)
        return m.components(rows)["u"].to_numpy()

    a, b = tight(ent), tight(ent[ent["race_id"] <= rid].reset_index(drop=True))
    assert np.abs(a - b).max() < 1e-6, np.abs(a - b).max()


# ---- stored post-hoc run folder
@needs_runs
def test_posthoc_run_folder_is_dev_only_complete_and_rederives_from_predictions(tabs):
    d = _dir(POST_LABEL)
    for f in ("config.yaml", "metrics.json", "per_race_metrics.csv", "predictions.parquet", "outcome_free_table.csv"):
        assert (d / f).exists(), f
    pr = pd.read_parquet(d / "predictions.parquet")
    assert pr["season"].between(2021, 2023).all() and pr["race_id"].nunique() == 66 and len(pr) == 1311
    assert not pr.duplicated(["race_id", "driverId"]).any() and pr["score"].notna().all()
    assert pr.groupby("season")["race_id"].nunique().to_dict() == {2021: 22, 2022: 22, 2023: 22}
    ref = pd.read_parquet(_dir(LABELS["latent_111"]) / "predictions.parquet")
    assert list(pr[["race_id", "driverId"]].itertuples(index=False)) == list(ref[["race_id", "driverId"]].itertuples(index=False))
    assert (pr["pos"].to_numpy() == ref["pos"].to_numpy()).all() and (pr["grid"].to_numpy() == ref["grid"].to_numpy()).all()
    pm = pd.read_csv(d / "per_race_metrics.csv")
    assert len(pm) == 66 and pm["season"].between(2021, 2023).all() and not pm.duplicated("race_id").any()
    t = tabs["post"].set_index("race_id")
    pmi = pm.set_index("race_id")
    for c in ("spearman", "mae", "exact", "logloss_points", "spearman_fin", "mae_fin"):
        assert pmi[c].to_numpy(float) == pytest.approx(t.loc[pmi.index, c].to_numpy(float), abs=1e-9, nan_ok=True), c
    stored = pd.read_csv(d / "outcome_free_table.csv")
    pd.testing.assert_frame_equal(stored.reset_index(drop=True), tabs["post"][stored.columns].reset_index(drop=True),
                                  check_exact=False, rtol=0, atol=1e-12)
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    assert m["n_races"] == 66 and m["spearman"] == pytest.approx(float(pm["spearman"].mean()), abs=1e-12)
    # config
    cfg = load_yaml(d / "config.yaml")
    ph = json.loads((RES / "posthoc_summary.json").read_text(encoding="utf-8"))
    assert cfg["kind"] == "latent_supplier" and cfg["carry_on_total"] is True and cfg["split"] == "dev"
    tune = pd.read_csv(RES / "tuning_lam_e_posthoc.csv")
    assert cfg["lam_e"] == ph["lam_e_chosen"] == float(tune.loc[tune["spearman"].idxmax(), "lam_e"]) == 1.0 and cfg["lam"] == 32.0
    assert cfg["lam_b"] == 32.0 == cfg["lam"] == CFG["posthoc_variant"]["lam_b"] and "lam_b = lam" in cfg["frozen_hyperparameters"]
    assert "post hoc" in cfg["status"].lower() and cfg["component"] == POST_COMPONENT
    for k, v in LAT.items():
        if k not in ("kind", "cell"):
            assert cfg[k] == v, k
    assert cfg["cell"] == "111" and cfg["git_commit"].startswith(POST_BASE)
    assert ph["configurations_posthoc"] == 6 and ph["configurations_total"] == 12 <= ph["budget"] == 30
    # the superseded first run is kept on disk, not referenced by the shared csv files, and its aggregates are renamed
    old = sorted(RES.glob(f"{OLD_POST_STAMP}_{POST_LABEL}_dev"))
    assert len(old) == 1 and old[0] != d
    assert load_yaml(old[0] / "config.yaml")["lam_e"] == 32.0 and load_yaml(old[0] / "config.yaml")["lam_b"] is None
    for stem in ("paired_gains_dev_posthoc", "paired_gains_by_season_posthoc", "exact_vs_grid_posthoc", "convergence_posthoc", "tuning_lam_e_posthoc",
                 "tuning_per_race_posthoc"):
        assert (RES / f"{stem}_superseded_lookahead.csv").exists(), stem
    assert json.loads((RES / "posthoc_summary_superseded_lookahead.json").read_text(encoding="utf-8"))["lam_e_chosen"] == 32.0
    for rel in ("results/plan3/attribution.csv", "results/plan3/leaderboard.csv"):
        assert OLD_POST_STAMP not in (ROOT / rel).read_text(encoding="utf-8"), rel


@needs_real
@needs_runs
def test_posthoc_run_equals_a_fresh_run_of_the_variant(ent):
    cfg0 = load_yaml(_dir(POST_LABEL) / "config.yaml")
    cfg = dict(LAT, kind="latent_supplier", lam_e=cfg0["lam_e"], carry_on_total=True, lam_b=cfg0["lam_b"])
    p, per = walk_forward(ent, build_model(cfg, ent), "dev", refit_every=1)
    d = _dir(POST_LABEL)
    st = pd.read_parquet(d / "predictions.parquet")
    sd, n = U.score_diff(p, st)
    assert n == len(st) == 1311 and sd <= 1e-9, sd
    pm = pd.read_csv(d / "per_race_metrics.csv")
    diff, nn = U.max_diff(per, pm)
    assert nn == 66 and diff <= 1e-9, diff
    assert U.score_diff(st, pd.read_parquet(_dir(LABELS["latent_111"]) / "predictions.parquet"))[0] > 1e-3     # non-vacuous
    # and the walk on a universe cut at the dev horizon (no 2024+ lineage at all) gives the same scores: the future entry list is irrelevant
    cut = ent[ent["season"] <= 2023].reset_index(drop=True)
    pc, _ = walk_forward(cut, build_model(cfg, cut), "dev", refit_every=1)
    assert U.score_diff(pc, st)[0] < 1e-6, U.score_diff(pc, st)[0]


@needs_runs
def test_posthoc_paired_gains_by_season_and_exact_vs_grid_rederived(tabs):
    st = pd.read_csv(RES / "paired_gains_by_season_posthoc.csv")
    assert len(st) == 15
    for s in (2021, 2022, 2023):
        a, b = tabs[REF][tabs[REF]["season"] == s], tabs["post"][tabs["post"]["season"] == s]
        got = A.paired_gain(a, b, SEAS_METRICS, n_boot=10000, seed=0)
        ss = st[st["season"] == s].reset_index(drop=True)
        assert list(ss["metric"]) == list(got["metric"]) and (ss["n"] == 22).all()
        for c in ("ref_mean", "cand_mean", "gain", "lo", "hi"):
            assert ss[c].to_numpy(float) == pytest.approx(got[c].to_numpy(float), abs=1e-12), (s, c)
    vg = pd.read_csv(RES / "exact_vs_grid_posthoc.csv")
    assert len(vg) == 4
    for cand, key in (("post-hoc carry_on_total", "post"), ("latent_111 (reference)", REF)):
        got = A.paired_gain(tabs["grid"], tabs[key], ["exact", "exact_fin"], n_boot=10000, seed=0)
        s = vg[vg["candidate"] == cand].reset_index(drop=True)
        assert list(s["metric"]) == ["exact", "exact_fin"]
        for c in ("ref_mean", "cand_mean", "gain", "lo", "hi"):
            assert s[c].to_numpy(float) == pytest.approx(got[c].to_numpy(float), abs=1e-12), (cand, c)
        assert list(s["verdict"]) == [A.verdict(r) for _, r in got.iterrows()]


# ---- tuning per-race files and convergence files
@needs_runs
@pytest.mark.parametrize("name,tab,per", [("declared", "tuning_lam_e.csv", "tuning_per_race_declared.csv"),
                                           ("posthoc", "tuning_lam_e_posthoc.csv", "tuning_per_race_posthoc.csv")])
def test_tuning_per_race_csv_means_reproduce_the_tuning_table_and_the_pick(name, tab, per):
    t = pd.read_csv(RES / tab)
    pr = pd.read_csv(RES / per)
    summ_path = RES / ("tuning_summary.json" if name == "declared" else "posthoc_summary.json")
    chosen = json.loads(summ_path.read_text(encoding="utf-8"))["lam_e_chosen"]
    fin = np.isfinite(pr["lam_e"].astype(float))
    ref, cand = pr[~fin], pr[fin]
    assert ref["race_id"].nunique() == len(ref) == 66 and ref["season"].between(2021, 2023).all()
    grid = CFG["lam_e_grid"] if name == "declared" else CFG["posthoc_variant"]["lam_e_grid"]
    assert list(t["lam_e"]) == grid and sorted(set(cand["lam_e"].astype(float))) == sorted(grid)
    assert not pr.duplicated(["lam_e", "race_id"]).any()
    for lam, g in cand.groupby("lam_e"):
        assert len(g) == 66 and sorted(g["race_id"]) == sorted(ref["race_id"])
        row = t[t["lam_e"] == float(lam)].iloc[0]
        for c in ("spearman", "mae", "exact"):
            assert float(g[c].mean()) == pytest.approx(float(row[c]), abs=1e-12), (lam, c)
        assert float(ref["spearman"].mean()) == pytest.approx(float(row["ref_spearman"]), abs=1e-12)
    assert float(t.loc[t["spearman"].idxmax(), "lam_e"]) == chosen
    means = cand.groupby("lam_e")["spearman"].mean()
    assert float(means.idxmax()) == chosen and (means == means.max()).sum() == 1
    assert (t["n_races"] == 66).all() and t[["spearman", "mae", "exact"]].notna().all().all()


@needs_runs
def test_declared_tuning_rederivation_is_identical_to_the_original_table():
    t = pd.read_csv(RES / "tuning_lam_e.csv")
    pr = pd.read_csv(RES / "tuning_per_race_declared.csv")
    m = pr[np.isfinite(pr["lam_e"].astype(float))].groupby("lam_e")[["spearman", "mae", "exact"]].mean().reset_index()
    assert list(m["lam_e"]) == list(t["lam_e"])
    assert np.abs(m["spearman"].to_numpy() - t["spearman"].to_numpy()).max() <= 1e-12       # README: max |rho diff| 0.0
    summ = json.loads((RES / "posthoc_summary.json").read_text(encoding="utf-8"))
    assert summ["rederived_declared_tuning_max_abs_rho_diff"] == 0.0 and summ["reference_rerun_vs_stored_max_abs_per_race_metric_diff"] < 1e-15


def _conv(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / f"convergence_{name}.csv")


@needs_runs
def test_convergence_csvs_rederive_the_readme_convergence_numbers():
    readme = _flat((METH / "README.md").read_text(encoding="utf-8"))
    dec, ph = _conv("declared"), _conv("posthoc")
    assert len(dec) == len(ph) == 7 and (dec["n_fits"] == 21).all() and (ph["n_fits"] == 21).all()
    assert -(-81 // CFG["tune_refit_every"]) == 21                           # 15 warm-up + 66 dev races at refit_every 4
    assert list(dec["lam_e"].iloc[1:]) == CFG["lam_e_grid"] and list(ph["lam_e"].iloc[1:]) == CFG["posthoc_variant"]["lam_e_grid"]
    for d in (dec, ph):
        assert (d["share_hit_maxiter"] == 0).all() and (d["share_converged_status0"] == 1).all() and (d["max_n_iter"] < CFG["convergence_maxiter"]).all()
    assert dec.iloc[0].drop("arm").astype(float).tolist() == ph.iloc[0].drop("arm").astype(float).tolist()   # same reference arm
    assert "21 fits each, status 0 in all, none reaches 150 iterations (max 40, median 23-28)" in readme
    assert int(dec["max_n_iter"].max()) == 40 and dec["median_n_iter"].min() == 23 and dec["median_n_iter"].max() == 28
    assert "final gradient inf-norm median 8e-6, max 3e-5" in readme
    assert round(float(dec["median_grad_inf"].median()), 6) == 8e-6 and float(dec["max_grad_inf"].max()) == pytest.approx(3e-5, rel=0.05)
    assert "86-100% of fits end under it, none above 3e-5" in readme
    assert round(100 * dec["share_grad_inf_below_1e5"].min()) == 86 and dec["share_grad_inf_below_1e5"].max() == 1.0
    assert "All fits converge (status 0, none at maxiter; max final gradient inf-norm 9.3e-5, 24-57% of fits under 1e-5)" in readme
    assert round(float(ph["max_grad_inf"].max()), 6) == 9.3e-5 and round(100 * ph["share_grad_inf_below_1e5"].iloc[1:].min()) == 24
    assert round(100 * ph["share_grad_inf_below_1e5"].iloc[1:].max()) == 57
    sp = json.loads((RES / "solver_probe.json").read_text(encoding="utf-8"))
    assert "The 0.33 maximum score difference at lam_e = 1e8" in readme and "with maxiter 2000 it is still 0.27" in readme
    assert round(sp["maxiter_150"]["max_abs_score_diff_lam_e_1e8_vs_reference"], 2) == 0.33
    assert round(sp["maxiter_2000"]["max_abs_score_diff_lam_e_1e8_vs_reference"], 2) == 0.27
    big = sp["maxiter_150"]["lam_e_1e8"]
    assert "median 1e-2, max 0.27, against 1e-5 for the reference; median 8 iterations at maxiter 150 and 1 at maxiter 2000" in readme
    assert round(big["median_grad_inf"], 2) == 0.01 and round(big["max_grad_inf"], 2) == 0.27 and big["median_n_iter"] == 8
    assert sp["maxiter_2000"]["lam_e_1e8"]["median_n_iter"] == 1 and sp["maxiter_150"]["reference"]["n_fits"] == 6
    assert sp["reference_maxiter_150_vs_2000_max_abs_score_diff"] == 0.0


@needs_runs
def test_readme_posthoc_numbers_match_the_result_files():
    readme = _flat((METH / "README.md").read_text(encoding="utf-8"))
    g = pd.read_csv(RES / "paired_gains_dev_posthoc.csv").set_index("metric")

    def fmt(r, nd=4):
        return f"{r['gain']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]"
    for text in (f"rho {fmt(g.loc['spearman'])}", f"rho on finishers {fmt(g.loc['spearman_fin'])}",
                 f"MAE (sign flipped) {fmt(g.loc['mae'])}", f"exact rate {fmt(g.loc['exact'])}"):
        assert text in readme, text
    assert f"{g.loc['spearman', 'gain']:+.4f}" == "-0.0155" and g.loc["spearman", "hi"] < 0 and "Attribution verdict: worse" in readme
    sb = pd.read_csv(RES / "paired_gains_by_season_posthoc.csv")
    s = sb[sb["metric"] == "spearman"].set_index("season")
    assert "By season rho: " + ", ".join(f"{y} {fmt(s.loc[y])}" for y in (2021, 2022, 2023)) in readme
    vg = pd.read_csv(RES / "exact_vs_grid_posthoc.csv")
    c1 = vg[(vg["candidate"] == "post-hoc carry_on_total") & (vg["metric"] == "exact")].iloc[0]
    assert f"(0.1638): variant {c1['cand_mean']:.4f}, gap {fmt(c1)}, now excluding 0 on the wrong side" in readme and f"{c1['ref_mean']:.4f}" == "0.1638"
    assert c1["hi"] < 0
    t = pd.read_csv(RES / "tuning_lam_e_posthoc.csv")
    assert "Dev rho at refit_every 4, new run: " + ", ".join(f"{x:.4f}" for x in t["spearman"]) + " (lam_e 1 ... 300)" in readme
    assert [f"{x:.4f}" for x in t["spearman"]] == ["0.6302", "0.6296", "0.6294", "0.6299", "0.6292", "0.6292"]
    assert f"against {t['ref_spearman'].iloc[0]:.4f} for the reference: flat (range 0.001) and about 0.015 below the reference" in readme
    assert round(t["spearman"].max() - t["spearman"].min(), 3) == 0.001 and f"{t['ref_spearman'].iloc[0] - t['spearman'].max():.3f}" == "0.015"
    assert "**lam_e = 1 chosen**" in readme and float(t.loc[t["spearman"].idxmax(), "lam_e"]) == 1.0
    s1, s32 = float(t.loc[t["lam_e"] == 1.0, "spearman"].iloc[0]), float(t.loc[t["lam_e"] == 32.0, "spearman"].iloc[0])
    assert round(s1 - s32, 4) == 0.0003 and "32, which is 0.0003 behind here" in readme        # the 32 comparison in the README is right
    ps = json.loads((RES / "posthoc_summary.json").read_text(encoding="utf-8"))
    vs = {r["season"]: r for r in ps["variance_share_end_of_dev"]}
    ys = (2021, 2022, 2023)
    assert "e takes 76%, 83% and 81% of var(T) in 2021-2023" in readme
    assert tuple(round(100 * vs[y]["share_e"]) for y in ys) == (76, 83, 81)                  # (the coordinator's message said 76/84/81: 2022 is 83.46)
    assert "the own part takes 22%, 15%, 17%" in readme and tuple(round(100 * vs[y]["share_own"]) for y in ys) == (22, 15, 17)
    assert tuple(round(100 * vs[y]["share_cov"], 1) for y in ys) == (1.4, 2.0, 2.3)
    assert "the covariance term is small (1-2%)" in readme                                    # 2023 is 2.3%: loose but not wrong
    assert "Total variance of T is small (0.004-0.017)" in readme and round(min(vs[y]["var_total"] for y in ys), 3) == 0.004 and round(max(vs[y]["var_total"] for y in ys), 3) == 0.017
    for y in ys:
        assert vs[y]["share_e"] + vs[y]["share_own"] + vs[y]["share_cov"] == pytest.approx(1.0, abs=1e-9)
    # the old-run comparison quoted in the README is the superseded csv
    og = pd.read_csv(RES / "paired_gains_dev_posthoc_superseded_lookahead.csv").set_index("metric")
    ob = pd.read_csv(RES / "paired_gains_by_season_posthoc_superseded_lookahead.csv")
    osr = ob[ob["metric"] == "spearman"].set_index("season")
    assert (f"(Old run for comparison: rho {fmt(og.loc['spearman'])}; rho on finishers {og.loc['spearman_fin', 'gain']:+.4f}; MAE {og.loc['mae', 'gain']:+.4f}; "
            f"exact {og.loc['exact', 'gain']:+.4f}; by season {osr.loc[2021, 'gain']:+.4f}, {osr.loc[2022, 'gain']:+.4f}, {osr.loc[2023, 'gain']:+.4f}.)") in readme
    ovg = pd.read_csv(RES / "exact_vs_grid_posthoc_superseded_lookahead.csv")
    o1 = ovg[(ovg["candidate"] == "post-hoc carry_on_total") & (ovg["metric"] == "exact")].iloc[0]
    assert f"the old run's gap was {fmt(o1)}" in readme
    osh = json.loads((RES / "posthoc_summary_superseded_lookahead.json").read_text(encoding="utf-8"))["variance_share_end_of_dev"]
    assert "(old run 24%, 38%, 32%; declared model 1-2%)" in readme and tuple(round(100 * r["share_e"]) for r in osh) == (24, 38, 32)
    # reproduction numbers quoted for the declared arm after the fix
    assert "max |per-race metric diff| 1.1e-16" in readme and "re-run reproduces at 8.9e-16" in readme
    assert f"{ps['reference_rerun_vs_stored_max_abs_per_race_metric_diff']:.1e}" == "8.9e-16" and ps["rederived_declared_tuning_max_abs_rho_diff"] == 0.0


@needs_runs
def test_readme_margin_of_the_posthoc_pick_over_the_runner_up():
    t = pd.read_csv(RES / "tuning_lam_e_posthoc.csv").sort_values("spearman", ascending=False)
    assert list(t["lam_e"].iloc[:2]) == [1.0, 32.0]                    # runner-up is lam_e = 32, not 3
    margin = float(t["spearman"].iloc[0] - t["spearman"].iloc[1])
    readme = _flat((METH / "README.md").read_text(encoding="utf-8"))
    assert round(margin, 4) == 0.0003 and "highest by 0.0003" in readme


@needs_runs
def test_readme_declared_descriptive_numbers_and_season_interval_statement():
    readme = _flat((METH / "README.md").read_text(encoding="utf-8"))
    ds = json.loads((RES / "descriptive.json").read_text(encoding="utf-8"))
    vs = {r["season"]: r for r in ds["variance_share_end_of_dev"]}
    ys = (2021, 2022, 2023)
    assert "2021 0.8%, 2022 1.5%, 2023 2.3%" in readme and tuple(round(100 * vs[y]["share_e"], 1) for y in ys) == (0.8, 1.5, 2.3)
    assert "-6.5%, +9.5%, +13.0%" in readme and tuple(round(100 * vs[y]["share_cov"], 1) for y in ys) == (-6.5, 9.5, 13.0)
    ub = tuple(round(100 * (vs[y]["share_e"] + vs[y]["share_cov"]), 1) for y in (2023, 2022, 2021))
    half = tuple(round(100 * (vs[y]["share_e"] + vs[y]["share_cov"] / 2), 1) for y in (2023, 2022, 2021))
    assert ub == (15.3, 11.0, -5.7) and "2023 15.3%, 2022 11.0%, 2021 -5.7%" in readme
    assert half == (8.8, 6.3, -2.4)                      # exact value for 2021 is -2.446%: rounds to -2.4 (README prints -2.5, see next test)
    for y in ys:
        assert vs[y]["share_e"] + vs[y]["share_own"] + vs[y]["share_cov"] == pytest.approx(1.0, abs=1e-9)
    e = pd.DataFrame(ds["e_end_of_dev"]).set_index(["supplier", "season"])["e"]
    assert "Honda 2023 +0.066, Ferrari 2022 -0.034, Mercedes 2022 +0.029" in readme
    assert (round(e[("honda", 2023)], 3), round(e[("ferrari", 2022)], 3), round(e[("mercedes", 2022)], 3)) == (0.066, -0.034, 0.029)
    a5 = pd.DataFrame(ds["e_after_2022_round5"]).set_index("supplier")["e"]
    assert "e Ferrari +0.010, Honda -0.028, Mercedes +0.024, Renault -0.006; e share 1.7%" in readme
    # descriptive.json stores 4 decimals, so Ferrari +0.0095 is an exact half-way case (the README rounds half up: +0.010)
    for k, v in (("ferrari", 0.010), ("honda", -0.028), ("mercedes", 0.024), ("renault", -0.006)):
        assert abs(a5[k] - v) <= 0.0005 + 1e-12, k
    assert round(100 * ds["variance_share_after_2022_round5"][0]["share_e"], 1) == 1.7
    assert "rho difference 0, 0, -0.0045, 0, +0.015 (mean +0.002)" in readme
    dd = [r["d_spearman"] for r in ds["early_2022"]]
    assert dd == [0.0, 0.0, -0.0045, 0.0, 0.015] and round(ds["early_2022_mean_d_spearman"], 3) == 0.002
    sb = pd.read_csv(RES / "paired_gains_by_season.csv")
    ex = sb[(sb["lo"] > 0) | (sb["hi"] < 0)]
    assert len(ex) == 2 and "Two of the 15 season intervals exclude 0, both on the wrong side" in readme
    for _, r in ex.iterrows():
        assert f"{r['gain']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" in readme
    ck = json.loads((RES / "checks.json").read_text(encoding="utf-8"))
    assert f"{ck['leakage']['max_abs_diff_scrambled_outcomes_and_later_suppliers']:.1e}" == "1.6e-14" and "max abs diff 1.6e-14" in readme
    assert f"{ck['leakage']['max_abs_diff_truncated_universe']:.1e}" == "1.1e-03"
    assert f"{ck['leakage']['max_abs_diff_truncated_universe_plain_latent_model']:.1e}" == "1.0e-03"
    assert "changes scores by 1.1e-3" in readme and "(1.0e-3)" in readme


@needs_runs
def test_posthoc_run_code_stamp_and_config_consistency():
    """The stamp `<head>+dirty-<sha1 of the USED files>` was computed from the files as they were at run time. config.yaml and run.py were
    edited afterwards (mtimes later than the run folder), and the files were never committed, so the stamped bytes are not recoverable.
    We (a) try the exact hash on the current files (reported, not asserted), (b) check the run's recorded config against the current config."""
    import hashlib
    import yaml
    cfg = load_yaml(_dir(POST_LABEL) / "config.yaml")
    m = re.fullmatch(r"([0-9a-f]{7,40})\+dirty-([0-9a-f]{12})", cfg["git_commit"])
    assert m and m.group(1) == POST_BASE
    assert subprocess.check_output(["git", "merge-base", "--is-ancestor", POST_BASE, "HEAD"], cwd=ROOT) == b""          # the stamped base is in history
    h = hashlib.sha1()
    for f in R.USED:
        h.update(f.encode())
        h.update((ROOT / f).read_bytes())
    assert h.hexdigest()[:12] == m.group(2) == "7c679b54449f"          # revision 2: the stamped files are exactly the current files
    ph = CFG["posthoc_variant"]
    assert ph["name"] == "carry_on_total" and ph["component"] == cfg["component"]
    assert cfg["lam_e"] in ph["lam_e_grid"] and cfg["lam_b"] == ph["lam_b"] and cfg["carry_on_total"] is True
    txt = (METH / "config.yaml").read_text(encoding="utf-8")
    nocom = "\n".join(re.sub(r"\s+#.*$", "", ln) for ln in txt.splitlines() if not ln.lstrip().startswith("#"))
    assert yaml.safe_load(nocom) == yaml.safe_load(txt)                       # comments carry no data in the current file


@needs_runs
def test_readme_equal_split_numbers_2021_value():
    readme = _flat((METH / "README.md").read_text(encoding="utf-8"))
    ds = json.loads((RES / "descriptive.json").read_text(encoding="utf-8"))
    vs = {r["season"]: r for r in ds["variance_share_end_of_dev"]}
    half = tuple(round(100 * (vs[y]["share_e"] + vs[y]["share_cov"] / 2), 1) for y in (2023, 2022, 2021))
    assert "gives " + ", ".join(f"{x:.1f}%" for x in half[:2]) + f" and {half[2]:.1f}%" in readme


def _tight_fit(m, rid):
    rows = m.race_rows[rid]
    first = int(rows.min())
    w, pad = m.fit_inputs(first, float(rid))
    m.pin_inactive(first)
    res = minimize(m._objective, m.theta, args=(first, w, pad), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=20000, maxfun=100000, gtol=1e-12, ftol=1e-16, maxcor=40))
    m.theta = res.x
    assert res.status == 0 and np.abs(res.jac).max() < 1e-7
    return m, rows, first


@needs_real
@pytest.mark.parametrize("lam_b", [None, 1e-9])
def test_regression_truncation_gap_with_lam_b_equal_lam_is_now_below_1e_9(ent, lam_b):
    """Regression for the look-ahead defect found in revision 1 (gap 5.7e-3 at 2022 R1 with lam_b = lam; 0.05 elsewhere). With cars that
    have no training row pinned, deleting every later row leaves race r's scores unchanged at the exact optimum, for lam_b = lam AND a
    negligible lam_b."""
    rid = _rid(ent, 2022, 1)
    cut = ent[ent["race_id"] <= rid].reset_index(drop=True)
    ma, rows_a, _ = _tight_fit(ck_model(ent, 32.0, lam_b), rid)
    mb, rows_b, _ = _tight_fit(ck_model(cut, 32.0, lam_b), rid)
    gap = np.abs(ma.components(rows_a)["u"].to_numpy() - mb.components(rows_b)["u"].to_numpy()).max()
    assert gap < (1e-9 if lam_b is None else 1e-6), gap          # lam_b = 1e-9 leaves a nearly flat direction: optimiser-limited, still 4 orders below the old gap


@needs_real
def test_pinned_car_value_equals_rho_times_previous_car_and_the_plain_model_agrees(ent):
    """At the first race of 2022 every 2022 car has no training row. Its predicted b must be rho * b of last season's car (0 without one);
    the plain LatentModel's optimum gives exactly that value to a car without data (carry-over penalty alone), which is what the fix claims."""
    rid = _rid(ent, 2022, 1)
    m, rows, first = _tight_fit(ck_model(ent, 32.0), rid)
    ref = LatentModel(ent, dynamic=True, decomp=True, ranking=True, **KW)
    w, pad = ref.fit_inputs(first, float(rid))
    res = minimize(ref._objective, ref.theta, args=(first, w, pad), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=20000, maxfun=100000, gtol=1e-12, ftol=1e-16, maxcor=40))
    ref.theta = res.x
    car_has_row = np.zeros(m.nB, dtype=bool)
    car_has_row[m.row_b[:first]] = True
    cars = m.row_b[rows]
    assert not car_has_row[cars].any()                                                       # all cars of the race are unseen (two drivers share a car)
    b_raw = m._split(m.theta)[1]
    comp = m.components(rows)
    want = np.array([m.rho * b_raw[m.car_prev[c]] if m.car_prev[c] >= 0 else 0.0 for c in cars])
    assert (m.car_prev[cars] >= 0).sum() >= 15 and np.abs(want).max() > 1e-3
    np.testing.assert_allclose(comp["b_car"].to_numpy(), want, atol=1e-14, rtol=0)
    # the rule is the one the plain LatentModel's own optimum applies to a car without data (its carry-over penalty alone)
    rb = ref._split(ref.theta)[1]
    ref_want = np.array([ref.rho * rb[ref.car_prev[c]] if ref.car_prev[c] >= 0 else 0.0 for c in cars])
    np.testing.assert_allclose(ref.components(rows)["b_car"].to_numpy(), ref_want, atol=1e-6)
    assert (comp["e_supplier"].to_numpy() == 0.0).all()                                   # its supplier level has no row either
    np.testing.assert_allclose(comp[["a_driver", "b_car", "g_circuit", "e_supplier", "weekend"]].sum(axis=1), comp["u"], atol=1e-13)
    # an unseen car whose predecessor is itself unseen chains through the substitution (2 seasons ahead): value rho^2 * b of the car 2 seasons back
    u2 = ent[ent["season"] <= 2018].reset_index(drop=True)
    m2 = ck_model(u2, 32.0)
    n2 = int(m2.race_rows[_rid(ent, 2017, 3)].min())
    m2.theta = np.random.default_rng(3).normal(0, 0.3, len(m2.theta))
    m2.pin_inactive(n2)
    bp = m2._b_pinned()
    raw = m2._split(m2.theta)[1]
    seen = np.zeros(m2.nB, dtype=bool)
    seen[m2.row_b[:n2]] = True
    for c in np.flatnonzero(~seen):
        pv = m2.car_prev[c]
        assert bp[c] == (m2.rho * bp[pv] if pv >= 0 else 0.0)
    assert (bp[seen] == raw[seen]).all() and (~seen).sum() >= 8


@needs_real
@pytest.mark.parametrize("variant", ["carry_on_total", "declared"])
def test_car_without_training_row_has_gradient_exactly_zero_and_no_influence_on_the_value(ent, variant):
    """Revision 2: in the carry_on_total variant a b level without a training row has gradient exactly 0 at a random point (a data-free
    coordinate with no term), and the value is invariant to it. (Declared variant: not claimed; reported by the second parametrisation.)"""
    u, rid = _pin_setup(ent)
    m = ck_model(u, 32.0) if variant == "carry_on_total" else sup_model(u, 32.0)
    n = int(m.race_rows[rid].min())
    w, pad = m.fit_inputs(n, float(rid))
    th = np.random.default_rng(31).normal(0, 0.4, len(m.theta))
    f, g = m._objective(th, n, w, pad)
    seen = np.zeros(m.nB, dtype=bool)
    seen[m.row_b[:n]] = True
    bidx = m.nD + np.flatnonzero(~seen)
    assert len(bidx) >= 8 and seen.sum() > 10
    if variant == "declared":
        # the declared path keeps the plan-1 carry-over for unseen cars: b of an unseen car with a seen predecessor is pulled by the
        # penalty, so a non-zero gradient is expected there; documented, not asserted to be zero
        assert np.abs(g[bidx]).max() > 0
        return
    assert np.abs(g[bidx]).max() < 1e-15, np.abs(g[bidx]).max()           # zero up to round-off (exactly 0 is NOT achieved: see the xfail test below)
    th2 = th.copy()
    th2[bidx] += 2.5
    f2, g2 = m._objective(th2, n, w, pad)
    assert f2 == f or abs(f2 - f) < 1e-12
    # gradient of every other coordinate is unchanged too (no coupling through the pinned cars)
    keep = np.ones(len(th), dtype=bool)
    keep[bidx] = False
    np.testing.assert_allclose(g2[keep], g[keep], atol=1e-12, rtol=0)


@needs_real
@pytest.mark.parametrize("season,rnd", [(2021, 1), (2021, 12), (2022, 1), (2023, 1), (2023, 15)])
def test_fit_with_and_without_later_season_lineages_is_identical_at_the_optimum(ent, season, rnd):
    """Pinned cars change no other level's optimum: fit the full universe (2014-2026 lineages) and the universe cut at the end of the target
    season's PREVIOUS race; every fitted utility of every training row and every supplier level with data agree."""
    rid = _rid(ent, season, rnd)
    cut = ent[ent["race_id"] < rid].reset_index(drop=True)                    # no row of race r or later, hence no later lineage at all
    full, rows, first = _tight_fit(ck_model(ent, 32.0), rid)
    small = ck_model(cut, 32.0)
    small.pin_inactive(len(cut))
    w, pad = small.fit_inputs(len(cut), float(rid))
    res = minimize(small._objective, small.theta, args=(len(cut), w, pad), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=20000, maxfun=100000, gtol=1e-12, ftol=1e-16, maxcor=40))
    small.theta = res.x
    assert full.nB > small.nB and full.nE > small.nE                           # later cars and levels really are absent in the cut
    tr = np.arange(max(0, first - 60), first)
    np.testing.assert_allclose(full.components(tr)["u"].to_numpy(), small.components(tr)["u"].to_numpy(), atol=1e-7, rtol=0)
    ef = full.supplier_effects().set_index(["supplier", "season"])["e"]
    es = small.supplier_effects().set_index(["supplier", "season"])["e"]
    assert np.abs(ef.loc[es.index].to_numpy() - es.to_numpy()).max() < 1e-7


@pytest.mark.xfail(strict=True, reason="Claimed 'gradient exactly 0' for a b level without a training row; measured 4e-18 (round-off in grad*W - pgb). "
                                       "Harmless in practice (value invariant, L-BFGS does not move it), but not exact. Remove when the code masks the coordinate.")
@needs_real
def test_car_without_training_row_gradient_is_bitwise_zero(ent):
    u, rid = _pin_setup(ent)
    m = ck_model(u, 32.0)
    n = int(m.race_rows[rid].min())
    w, pad = m.fit_inputs(n, float(rid))
    th = np.random.default_rng(31).normal(0, 0.4, len(m.theta))
    g = m._objective(th, n, w, pad)[1]
    seen = np.zeros(m.nB, dtype=bool)
    seen[m.row_b[:n]] = True
    assert (g[m.nD + np.flatnonzero(~seen)] == 0.0).all()
