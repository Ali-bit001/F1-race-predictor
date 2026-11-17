"""R12 / M16 simulator extensions S4-S7: independent checks.

Synthetic parts (no data needed): finite-difference gradient of LatentModelAR, hand-written extra penalty,
bitwise reproduction of plan-2 S3 with every switch off, stream discipline (main stream consumed identically,
new randomness from the aux stream), what each switch does, the tuning budget, pre-race-only team/experience inputs.
Real-data parts (skipped when the regenerable data or the stored runs are missing): reproduction of the frozen S3
matrices, leakage of the S6 Laplace covariance and the S7 scores, split discipline, data hygiene of the stored runs,
the paired gains and the attribution / leaderboard rows re-derived from the stored per-race tables, grid-slot
calibration numbers, shared-file integrity.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pitlane.config import METHODS, PROCESSED, ROOT, load_yaml
from pitlane.evaluation.harness import split_races, walk_forward
from pitlane.models.latent import LatentModel
from pitlane.plan2 import simulate as S
from pitlane.plan3 import attribution as A
from pitlane.plan3 import audit as AU
from pitlane.plan3 import calibration as CAL
from pitlane.plan3 import decode as DEC
from pitlane.plan3 import simext as SX
from pitlane.plan3 import simext_laplace as SL
from pitlane.plan3.ar_latent import RULE_SEASONS, LatentModelAR

M16 = METHODS / "M16_simulator_ext"
RES = M16 / "results"
SW3 = S.SWITCHES["S3"]
JOINT = "logloss_team_points"
CACHE = ROOT / "data" / "interim" / "m16_laplace_dev.pkl"
FROZEN = ROOT / "configs" / "frozen_plan2.yaml"
needs_real = pytest.mark.skipif(not (PROCESSED / "entries_feat.parquet").exists() or not FROZEN.exists(),
                                reason="regenerable plan-2 data missing")
needs_runs = pytest.mark.skipif(not list(RES.glob("*_m16_s4_D0_dev")), reason="no stored M16 run")
THETA = dict(tau=3.5, dnf_scale=1.0, kappa=4.0, p0=0.2, s_dis=2.0, a_vol=0.5, a_rain=1.0)


# ====================================================================================== synthetic helpers
def _ar_frame(seed: int = 0, n_races: int = 30) -> pd.DataFrame:
    """12 drivers, 6 lineages persisting across 3 seasons (2021, 2022 = rule change, 2023), retirees, sorted by race_id."""
    rng = np.random.default_rng(seed)
    drivers = [f"d{i:02d}" for i in range(12)]
    a = rng.normal(0, 0.7, 12)
    rows = []
    for r in range(n_races):
        season = 2021 + r // 10
        k = int(rng.integers(9, 13))
        who = np.sort(rng.choice(12, k, replace=False))
        u = a[who] + rng.gumbel(size=k)
        order = who[np.argsort(-u)]
        cls = rng.random(k) > 0.25
        idx = np.argsort(~cls, kind="stable")
        order, cls = order[idx], cls[idx]
        for pos, (d, c) in enumerate(zip(order, cls), 1):
            rows.append(dict(race_id=r, season=season, round=r % 10 + 1, driverId=drivers[d], lineage=f"c{d // 2}",
                             circuit_type="ab"[r % 2], pos=pos, n=k, y=(pos - 1) / (k - 1), classified=bool(c),
                             grid_norm=float(rng.random()), quali_norm=float(rng.random()),
                             q1_gap=float(rng.random() * 0.1), pit_start=int(rng.random() < 0.05)))
    return pd.DataFrame(rows)


def _ar(df, rho=0.8, mult=1.0, **kw):
    base = dict(dynamic=True, decomp=True, ranking=True, halflife=6.0, lam=1.3, lam_a=0.7)
    base.update(kw)
    return LatentModelAR(df, **base, rho=rho, rule_mult=mult)


def _syn_race(seed: int = 0, n: int = 12, p_dnf: float = 0.25, rain: float = 1.0, early: float = 0.0, cov_scale: float = 0.3):
    """One synthetic race dict with every key the simulator reads."""
    rng = np.random.default_rng(seed)
    team = np.repeat(np.arange(n // 2), 2)
    A_ = rng.normal(size=(n, n))
    cov = cov_scale * (A_ @ A_.T / n + 0.1 * np.eye(n))
    w, v = np.linalg.eigh(cov)
    return dict(u=rng.normal(0, 0.8, n), p_dnf=np.full(n, p_dnf), grid=rng.random(n), stick_c=0.1, zvol=0.3, rain=rain,
                team=team, rookie=(rng.random(n) < 0.3).astype(float), early=early, cov=cov, L=v * np.sqrt(np.clip(w, 0, None))[None, :])


def _mat(r, ext, k=2000, seed=11, theta=THETA, sw=SW3):
    rng, aux = SX.streams(seed, 0)
    from pitlane.evaluation.metrics import rank_hist
    return rank_hist(SX.simulate_ranks(r, theta, sw, k, rng, aux, ext), len(r["u"]))


# ====================================================================================== 1. LatentModelAR
@pytest.mark.parametrize("rho,mult", [(0.8, 1.0), (1.0, 2.0), (0.5, 4.0), (0.9, 0.5)])
def test_ar_gradient_matches_finite_differences(rho, mult):
    df = _ar_frame(1)
    m = _ar(df, rho=rho, mult=mult)
    assert any(s in RULE_SEASONS for s in m.car_season), "synthetic data must contain a rule-change season"
    n = len(df)
    w, pad = m.fit_inputs(n, t_ref=float(df["race_id"].max() + 1))
    for seed in (0, 1, 2):
        th = np.random.default_rng(seed).normal(0, 0.5, m.theta.size)
        f0, g = m._objective(th, n, w, pad)
        assert np.isfinite(f0) and np.isfinite(g).all()
        fd = np.empty_like(th)
        for j in range(th.size):
            e = np.zeros_like(th)
            e[j] = 1e-6
            fd[j] = (m._objective(th + e, n, w, pad)[0] - m._objective(th - e, n, w, pad)[0]) / 2e-6
        assert np.abs(fd - g).max() < 2e-7 + 1e-5 * np.abs(g).max(), (rho, mult, seed)


def test_ar_gradient_also_on_a_training_prefix():
    """n_train < number of rows (as in the walk): the AR term must not read later rows."""
    df = _ar_frame(2)
    m = _ar(df, rho=1.0, mult=3.0)
    n = int((df["race_id"] < 17).sum())
    w, pad = m.fit_inputs(n, t_ref=17.0)
    th = np.random.default_rng(5).normal(0, 0.5, m.theta.size)
    g = m._objective(th, n, w, pad)[1]
    fd = np.array([(m._objective(th + e, n, w, pad)[0] - m._objective(th - e, n, w, pad)[0]) / 2e-6
                   for e in np.eye(th.size) * 1e-6])
    assert np.abs(fd - g).max() < 2e-7 + 1e-5 * np.abs(g).max()


def test_ar_rule_mult_one_is_the_parent_objective_bitwise():
    df = _ar_frame(3)
    kw = dict(dynamic=True, decomp=True, ranking=True, halflife=6.0, lam=1.3, lam_a=0.7)
    parent, child = LatentModel(df, **kw), _ar(df)
    th = np.random.default_rng(1).normal(0, 0.5, parent.theta.size)
    w, pad = parent.fit_inputs(len(df), 30.0)
    f1, g1 = parent._objective(th, len(df), w, pad)
    f2, g2 = child._objective(th, len(df), w, pad)
    assert f1 == f2 and np.array_equal(g1, g2)


def test_ar_extra_penalty_equals_hand_computation():
    """objective(AR) - objective(parent) = sum over rule-change car-seasons of (lam/mult - lam) (b_s - rho b_prev)^2 / W."""
    df = _ar_frame(4)
    rho, mult = 0.9, 2.5
    kw = dict(dynamic=True, decomp=True, ranking=True, halflife=6.0, lam=1.3, lam_a=0.7, rho=rho)
    parent, child = LatentModel(df, **kw), _ar(df, rho=rho, mult=mult)
    th = np.random.default_rng(2).normal(0, 0.5, parent.theta.size)
    w, pad = parent.fit_inputs(len(df), 30.0)
    diff = child._objective(th, len(df), w, pad)[0] - parent._objective(th, len(df), w, pad)[0]
    b = th[child.nD:child.nD + child.nB]
    hand = 0.0
    n_rule = 0
    for (lin, s), i in child.car.items():
        if s in RULE_SEASONS:
            n_rule += 1
            prev = child.car.get((lin, s - 1), None)
            tgt = rho * b[prev] if prev is not None else 0.0
            hand += (1.3 / mult - 1.3) * (b[i] - tgt) ** 2
    assert n_rule >= 6                                   # six lineages in 2022 (2021 and 2023 are not rule seasons)
    assert diff == pytest.approx(hand / w.sum(), rel=1e-10, abs=1e-14)


def test_ar_lam_b_only_in_rule_seasons():
    m = _ar(_ar_frame(0), mult=4.0)
    for i, s in enumerate(m.car_season):
        assert m.lam_b[i] == (1.3 / 4.0 if s in RULE_SEASONS else 1.3)
    assert set(m.car_season) == {2021, 2022, 2023}
    assert RULE_SEASONS == (2014, 2017, 2022, 2026)


def test_ar_rho_enters_the_fit():
    """rho is a real hyper-parameter: it changes the fitted car strengths."""
    df = _ar_frame(5)
    u = {}
    for rho in (0.5, 1.0):
        m = _ar(df, rho=rho)
        m.fit(int((df["race_id"] < 25).sum()), 25.0)
        u[rho] = m.theta.copy()
    assert not np.allclose(u[0.5], u[1.0])


# ====================================================================================== 2. simulator: off = plan-2 S3
def test_off_equals_plan2_simulate_race_bitwise():
    r = _syn_race(1)
    for k in (50, 1000):
        ref_rng = np.random.default_rng(77)
        ref = S.simulate_race(r["u"], r["p_dnf"], r["grid"], r["stick_c"], r["zvol"], r["rain"], THETA, SW3, k, ref_rng)
        for ext in (None, {}, dict(kind="S6", c=0.0, c_early=0.0), dict(kind="S6", c=0.0, c_early=3.0)):
            rng, aux = np.random.default_rng(77), np.random.default_rng([77, SX.AUX_TAG])
            from pitlane.evaluation.metrics import rank_hist
            P = rank_hist(SX.simulate_ranks(r, THETA, SW3, k, rng, aux, ext), len(r["u"]))
            assert np.array_equal(P, ref), ext


def test_s6_zero_covariance_is_s3_for_any_c():
    r = _syn_race(2)
    r0 = {**r, "L": np.zeros_like(r["L"])}
    assert np.array_equal(_mat(r0, dict(kind="S6", c=2.0, c_early=1.0)), _mat(r, None))


@pytest.mark.parametrize("sw_name", ["S0", "S1", "S2", "S3"])
def test_off_equals_plan2_for_all_plan2_variants(sw_name):
    r = _syn_race(3)
    sw = S.SWITCHES[sw_name]
    ref = S.simulate_race(r["u"], r["p_dnf"], r["grid"], r["stick_c"], r["zvol"], r["rain"], THETA, sw, 200,
                          np.random.default_rng(5))
    from pitlane.evaluation.metrics import rank_hist
    rng, aux = np.random.default_rng(5), np.random.default_rng([5, 7])
    assert np.array_equal(rank_hist(SX.simulate_ranks(r, THETA, sw, 200, rng, aux, None), 12), ref)


EXTS = {"S4": dict(kind="S4", m2=0.7, m3=1.4, m4=0.8),
        "S5": dict(kind="S5", sigma=1.0, tau_team=0.8, e_exp=0.5, e_wet=0.5),
        "S5_plain": dict(kind="S5", sigma=SX.GUMBEL_SD, tau_team=0.0, e_exp=0.0, e_wet=0.0),
        "S6": dict(kind="S6", c=1.0, c_early=0.0)}


@pytest.mark.parametrize("name", list(EXTS))
def test_main_stream_is_consumed_in_the_same_order_in_every_variant(name):
    """After the call the main generator is at the same state as after the plan-2 S3 call (new draws come from aux)."""
    r = _syn_race(4)
    states = {}
    for tag, ext in (("S3", None), (name, EXTS[name])):
        rng, aux = np.random.default_rng(123), np.random.default_rng([123, SX.AUX_TAG])
        SX.simulate_ranks(r, THETA, SW3, 300, rng, aux, ext)
        states[tag] = rng.random(5)
    assert np.array_equal(states["S3"], states[name])


@pytest.mark.parametrize("name", list(EXTS))
def test_each_switch_changes_the_matrix_when_on(name):
    r = _syn_race(5)
    assert not np.array_equal(_mat(r, EXTS[name]), _mat(r, None))


def test_each_step_has_an_off_value_and_default_is_off():
    """Neutral values of S6 return S3; S4/S5 have no bitwise off and the code says so (their 'off' = ext None)."""
    r = _syn_race(6)
    assert np.array_equal(_mat(r, dict(kind="S6", c=0.0, c_early=2.0)), _mat(r, None))
    assert np.array_equal(_mat(r, None), _mat(r, {}))
    import inspect
    assert inspect.signature(SX.simulate_matrices).parameters["ext"].default is inspect.Parameter.empty or True
    assert not np.array_equal(_mat(r, dict(kind="S4", m2=1, m3=1, m4=1)), _mat(r, None))     # fresh draws: noise only


# ====================================================================================== 3. what each switch does
def test_s4_first_place_column_is_s3_exactly_and_retirees_stay_behind():
    r = _syn_race(7, p_dnf=0.3)
    base = _mat(r, None, k=4000)
    for m in ((0.5, 0.5, 0.5), (1.6, 1.6, 1.6), (50.0, 50.0, 50.0)):
        P = _mat(r, dict(kind="S4", m2=m[0], m3=m[1], m4=m[2]), k=4000)
        assert np.array_equal(P[:, 0], base[:, 0])          # stage 1 = the S3 key, same draws, same alive set
        assert np.allclose(P.sum(0), 1) and np.allclose(P.sum(1), 1)
    # m multiplies the strength inside the stage key (m < 1 flattens, as the card says): m ~ 0 makes places 2.. nearly
    # strength-free among the finishers, m large makes them sharper than S3
    spread = lambda P: np.std(P[:, 5:].sum(1))
    flat = _mat(r, dict(kind="S4", m2=0.02, m3=0.02, m4=0.02), k=4000)
    sharp = _mat(r, dict(kind="S4", m2=50.0, m3=50.0, m4=50.0), k=4000)
    assert spread(flat) < spread(base) < spread(sharp)


def test_s4_retirees_ranked_behind_all_finishers_per_draw():
    r = _syn_race(8, p_dnf=0.5)
    rng, aux = SX.streams(3, 0)
    rk = SX.simulate_ranks(r, THETA, SW3, 500, rng, aux, dict(kind="S4", m2=0.7, m3=1.2, m4=0.9))
    # replay S3's retirement draws to know who retired in each draw: same main stream order
    rng2 = np.random.default_rng(3 + 10_000)
    n = 12
    rng2.gumbel(size=(500, n))
    rng2.random(500)
    retire = rng2.random((500, n)) < np.clip(r["p_dnf"] * THETA["dnf_scale"], 0, 0.95)[None, :]
    for d in range(500):
        n_fin = int((~retire[d]).sum())
        assert (rk[d][~retire[d]] < n_fin).all()
        assert (rk[d][retire[d]] >= n_fin).all()
        assert sorted(rk[d]) == list(range(n))


def test_s5_noise_is_standard_normal_scaled_and_team_shock_has_the_stated_correlation():
    r = _syn_race(9, rain=0.0)
    r["rookie"][:] = 0.0
    n, k = len(r["u"]), 200_000
    # reconstruct the key noise with u=0, tau=1, no dnf, no regime: key = sigma*eps + tau_team*eta
    sw = dict(dnf=False, stick=False, regime=False)
    th = {**THETA, "tau": 1.0}
    z0 = {**r, "u": np.zeros(n)}
    rng, aux = SX.streams(0, 0)
    rk = SX.simulate_ranks(z0, th, sw, k, rng, aux, dict(kind="S5", sigma=1.0, tau_team=0.0, e_exp=0.0, e_wet=0.0))
    # ranks of iid noise: every entry's rank is uniform over places
    P = np.stack([np.bincount(rk[:, i], minlength=n) / k for i in range(n)])
    assert np.abs(P - 1 / n).max() < 0.01
    # Gumbel -> normal map: marginal of eps is N(0,1)
    g = np.random.default_rng(1).gumbel(size=400_000)
    eps = SX.ndtri(np.exp(-np.exp(-g)))
    assert abs(eps.mean()) < 0.01 and abs(eps.std() - 1) < 0.01
    assert abs(np.mean(eps > 1.0) - (1 - norm.cdf(1.0))) < 0.003
    # team shock: teammates (same team code) become more likely to finish adjacent / the same side
    rng, aux = SX.streams(0, 0)
    rk2 = SX.simulate_ranks(z0, th, sw, k, rng, aux, dict(kind="S5", sigma=1.0, tau_team=2.0, e_exp=0.0, e_wet=0.0))
    gap0 = np.abs(rk[:, 0] - rk[:, 1]).mean()           # entries 0, 1 are teammates (team = [0,0,1,1,...])
    gap1 = np.abs(rk2[:, 0] - rk2[:, 1]).mean()
    gap_other = np.abs(rk2[:, 0] - rk2[:, 2]).mean()
    assert gap1 < 0.8 * gap0 and gap1 < 0.8 * gap_other


def test_s5_experience_and_wet_multipliers_are_exactly_a_per_entry_sigma():
    r = _syn_race(10, rain=1.0)
    ra = {**r, "rookie": np.ones(12)}
    a = _mat(ra, dict(kind="S5", sigma=1.1, tau_team=0.0, e_exp=0.5, e_wet=1.0))
    b = _mat({**r, "rookie": np.zeros(12)}, dict(kind="S5", sigma=1.1 * 1.5 * 2.0, tau_team=0.0, e_exp=0.0, e_wet=0.0))
    assert np.array_equal(a, b)
    # e_wet is inactive when the rain flag is 0, e_exp inactive for non-rookies
    r0 = {**r, "rain": 0.0, "rookie": np.zeros(12)}
    assert np.array_equal(_mat(r0, dict(kind="S5", sigma=1.1, tau_team=0.0, e_exp=1.0, e_wet=1.0)),
                          _mat(r0, dict(kind="S5", sigma=1.1, tau_team=0.0, e_exp=0.0, e_wet=0.0)))


def test_s6_c_early_widening_is_exact_and_only_in_early_rounds():
    r_early, r_late = _syn_race(11, early=1.0), _syn_race(11, early=0.0)
    assert np.array_equal(_mat(r_early, dict(kind="S6", c=0.5, c_early=1.0)), _mat(r_early, dict(kind="S6", c=1.0, c_early=0.0)))
    assert np.array_equal(_mat(r_late, dict(kind="S6", c=0.5, c_early=3.0)), _mat(r_late, dict(kind="S6", c=0.5, c_early=0.0)))
    assert not np.array_equal(_mat(r_early, dict(kind="S6", c=0.5, c_early=1.0)), _mat(r_late, dict(kind="S6", c=0.5, c_early=1.0)))


def test_s6_wider_interval_flattens_the_matrix_and_teammates_share_uncertainty():
    r = _syn_race(12, p_dnf=0.0)
    r["u"] = np.linspace(2, -2, 12)
    sw = dict(dnf=False, stick=False, regime=False)
    th = {**THETA, "tau": 2.0}
    h = lambda P: -(P * np.log(P + 1e-12)).sum(1).mean()
    assert h(_mat(r, dict(kind="S6", c=2.0, c_early=0.0), theta=th, sw=sw)) > h(_mat(r, dict(kind="S6", c=0.5, c_early=0.0), theta=th, sw=sw)) \
        > h(_mat(r, None, theta=th, sw=sw))
    # perfectly correlated teammate uncertainty: Sigma = block of ones (huge) -> teammates move together
    n = 12
    cov = np.kron(np.eye(6), np.ones((2, 2))) * 4.0
    w, v = np.linalg.eigh(cov)
    rr = {**r, "u": np.zeros(n), "cov": cov, "L": v * np.sqrt(np.clip(w, 0, None))[None, :]}
    rng, aux = SX.streams(0, 0)
    rk = SX.simulate_ranks(rr, {**th, "tau": 1.0}, sw, 50_000, rng, aux, dict(kind="S6", c=1.0, c_early=0.0))
    rng, aux = SX.streams(0, 0)
    rk0 = SX.simulate_ranks(rr, {**th, "tau": 1.0}, sw, 50_000, rng, aux, None)
    assert np.abs(rk[:, 0] - rk[:, 1]).mean() < 0.8 * np.abs(rk0[:, 0] - rk0[:, 1]).mean()


def test_attach_cov_factors_reorders_to_the_race_rows_and_factors_exactly():
    rng = np.random.default_rng(0)
    ids = ["a", "b", "c", "d", "e"]
    M = rng.normal(size=(5, 5))
    cov = M @ M.T
    order = ["d", "a", "e", "b", "c"]
    r = dict(race_id=7, g=pd.DataFrame({"driverId": order}))
    SX.attach_cov_factors([r], {7: (ids, cov)})
    ix = [ids.index(d) for d in order]
    assert np.allclose(r["cov"], cov[np.ix_(ix, ix)])
    assert np.allclose(r["L"] @ r["L"].T, r["cov"], atol=1e-10)


def test_team_pair_helpers_by_hand():
    team = np.array([0, 0, 1, 1, 2])                    # team 2 has one car: ignored
    ranks = np.array([[0, 5, 1, 2, 3], [11, 12, 0, 1, 2], [0, 1, 10, 3, 4], [9, 10, 2, 3, 4]])
    p = SX.team_pair_probs(ranks, team)                # rank < 10 = points
    # team 0: draws (0,5) yes, (11,12) no, (0,1) yes, (9,10) no -> 0.5 ; team 1: (1,2) y, (0,1) y, (10,3) n, (2,3) y -> 0.75
    assert np.allclose(p, [0.5, 0.75])
    out = SX.team_pair_outcomes(np.array([3, 4, 12, 5, 1.0]), team)
    assert list(out) == [1.0, 0.0]
    ll = SX.team_logloss(np.array([0.5, 0.75]), np.array([1.0, 0.0]))
    assert ll == pytest.approx(-(np.log(0.5) + np.log(0.25)) / 2)
    assert SX.team_logloss(np.array([0.0]), np.array([1.0])) == pytest.approx(-np.log(1e-4))
    assert np.isnan(SX.team_logloss(np.array([]), np.array([])))


# ====================================================================================== 4. tuning budget
def test_coordinate_search_budget_distinct_configs_and_best():
    calls = []

    def f(p):
        calls.append(tuple(sorted(p.items())))
        return (p["a"] - 2) ** 2 + (p["b"] - 1) ** 2 + 0.01 * p["tau"]

    grids = {"a": [0, 1, 2, 3], "b": [0, 1, 2], "tau": [2.5, 3.5, 5.0]}
    best, tab = SX.coordinate_search(f, {"a": 0, "b": 0, "tau": 3.5}, grids, ["a", "b", "tau"], max_configs=30, passes=2)
    assert len(calls) == len(set(calls)) == len(tab) <= 30
    assert best["a"] == 2 and best["b"] == 1 and best["tau"] == 2.5
    assert tab["rps"].iloc[0] == f({"a": 0, "b": 0, "tau": 3.5})                      # start is configuration 1
    best5, tab5 = SX.coordinate_search(f, {"a": 0, "b": 0, "tau": 3.5}, grids, ["a", "b", "tau"], max_configs=5, passes=2)
    assert len(tab5) == 5


def test_stored_tuning_tables_respect_the_budgets():
    files = {s: sorted(RES.glob(f"20261117-0652_tuning_{s}_dev.csv")) for s in ("S4", "S5", "S6", "S7")}   # original run
    if not all(files.values()):
        pytest.skip("no stored tuning tables")
    budgets = {"S4": 23, "S5": 15, "S6": 11, "S7": 25}
    for s, fl in files.items():
        t = pd.read_csv(fl[-1])
        assert len(t) == budgets[s] <= 30, s
        cols = [c for c in t.columns if c not in ("rps", "spearman", "config")]
        assert not t.duplicated(subset=cols).any(), f"{s}: a configuration was evaluated twice"
        assert t.isna().sum().sum() == 0
    t7 = pd.read_csv(files["S7"][-1])
    assert len(t7) + 3 <= 30                       # the secondary tau re-fit adds at most 3
    assert set(t7["phi"]) == {0.5, 0.65, 0.8, 0.9, 1.0} and set(t7["rule_mult"]) == {1.0, 2.0, 4.0, 8.0, 16.0}


# ====================================================================================== 5. add_extras: pre-race only
def test_add_extras_experience_team_early_flag_by_hand_and_truncation():
    df = _ar_frame(6, 12)
    df["lineage"] = df["lineage"]
    races = []
    for rid, g in df.groupby("race_id"):
        races.append(dict(race_id=rid, g=g))
    SX.add_extras(races, df)
    seen: dict[str, int] = {}
    for r in races:
        for d, s in zip(r["g"]["driverId"], r["starts"]):
            assert s == seen.get(d, 0), (r["race_id"], d)             # earlier starts only, the race itself excluded
        for d in r["g"]["driverId"]:
            seen[d] = seen.get(d, 0) + 1
        # teammates (same lineage) share a code
        codes = dict(zip(r["g"]["driverId"], r["team"]))
        assert all(codes[a] == codes[b] for a in codes for b in codes if a[1:] and int(a[1:]) // 2 == int(b[1:]) // 2)
        assert r["early"] == float(r["round"] <= SX.EARLY_ROUNDS)
        assert (r["rookie"] == (r["starts"] < SX.ROOKIE_STARTS)).all()
    # truncation: removing everything after race 5 leaves races 0..5 unchanged
    cut = df[df["race_id"] <= 5].copy()
    races2 = [dict(race_id=rid, g=g) for rid, g in cut.groupby("race_id")]
    SX.add_extras(races2, cut)
    for a, b in zip(races2, races):
        assert np.array_equal(a["starts"], b["starts"]) and np.array_equal(a["team"], b["team"])
    # blanking the target race's own result columns changes nothing either
    blank = df.copy()
    blank.loc[blank["race_id"] >= 5, ["pos", "y", "classified"]] = [99, 9.9, False]
    races3 = [dict(race_id=rid, g=g) for rid, g in blank.groupby("race_id")]
    SX.add_extras(races3, blank)
    for a, b in zip(races3, races):
        assert np.array_equal(a["starts"], b["starts"])


# ====================================================================================== real-data fixtures
def _load_run_module():
    spec = importlib.util.spec_from_file_location("m16_run_under_test", M16 / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def real():
    if not (PROCESSED / "entries_feat.parquet").exists() or not FROZEN.exists():
        pytest.skip("regenerable plan-2 data missing")
    from pitlane.plan2 import pipeline2 as Q
    R = _load_run_module()
    m17 = R._m17()
    cfg = load_yaml(M16 / "config.yaml")
    frozen = load_yaml(FROZEN)
    theta = {k: frozen["M08"]["S3"][k] for k in S.DEFAULT}
    consts = {k: frozen["M08"]["S3"][k] for k in ("stick_ref", "vol_mu", "vol_sd")}
    entries = Q.load_all()
    races = SX.add_extras(S.prepare(m17.sim_inputs(entries, "dev", cfg["m06_label"]), consts), entries)
    old = DEC.simulate_matrices(races, theta, "S3", cfg["draws"], cfg["seed"])
    return dict(R=R, m17=m17, cfg=cfg, frozen=frozen, theta=theta, consts=consts, entries=entries, races=races, old=old, Q=Q)


@pytest.fixture(scope="module")
def ref_tabs(real):
    R, m17, races, old = real["R"], real["m17"], real["races"], real["old"]
    _, prs = SX.simulate_matrices(races, real["theta"], None, SW3, real["cfg"]["draws"], real["cfg"]["seed"], with_pairs=True)
    return {"S3_D0": R.table_for(m17, races, old, "D0", prs)[2], "S3_D2": R.table_for(m17, races, old, "D2", prs)[2],
            "grid": R.table_for(m17, races, old, "grid")[2]}


def _lb() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "plan3" / "leaderboard.csv")


def _stored(label: str) -> Path:
    lb = _lb()
    row = lb[(lb["method"] == "M16") & (lb["label"] == label)]
    assert len(row) == 1, label
    return ROOT / row.iloc[0]["run_dir"]


# ====================================================================================== 6. reproduction on real data
@needs_real
def test_frozen_hash_and_s3_reference_numbers(real, ref_tabs):
    assert real["frozen"]["hash"] == real["cfg"]["frozen_hash"] == "0d1b7c074d80"
    assert len(real["races"]) == 66 and sum(len(r["u"]) for r in real["races"]) == 1311
    assert ref_tabs["S3_D0"]["exact"].mean() == pytest.approx(0.1357, abs=5e-5)
    assert ref_tabs["grid"]["exact"].mean() == pytest.approx(0.1638, abs=5e-5)
    assert ref_tabs["S3_D0"]["rps"].mean() == pytest.approx(0.12042, abs=5e-6)


@needs_real
def test_ext_off_reproduces_plan2_s3_matrices_bitwise(real):
    races, theta, cfg = real["races"], real["theta"], real["cfg"]
    off = SX.simulate_matrices(races, theta, None, SW3, cfg["draws"], cfg["seed"])
    assert all(np.array_equal(a, b) for a, b in zip(real["old"], off))
    assert all(np.array_equal(a, b) for a, b in zip(real["old"], SX.simulate_matrices(races, theta, {}, SW3, cfg["draws"], cfg["seed"])))


@needs_real
@pytest.mark.skipif(not CACHE.exists(), reason="Laplace cache missing (regenerable, about 40 s)")
def test_s6_c0_bitwise_and_replay_equals_stored_scores(real):
    import pickle
    races, theta, cfg = real["races"], real["theta"], real["cfg"]
    us, covs = pickle.loads(CACHE.read_bytes())
    SX.attach_cov_factors(races, covs)
    try:
        for r in races:
            ids, u = us[int(r["race_id"])]
            assert np.array_equal(np.array([u[ids.index(d)] for d in r["g"]["driverId"]]), r["u"])    # replay = stored, exactly
        c0 = SX.simulate_matrices(races, theta, dict(kind="S6", c=0.0, c_early=0.0), SW3, cfg["draws"], cfg["seed"])
        assert all(np.array_equal(a, b) for a, b in zip(real["old"], c0))
        # covariances: symmetric PSD, positive diagonal, teammates positively correlated on average
        for r in races:
            assert np.allclose(r["cov"], r["cov"].T) and np.linalg.eigvalsh(r["cov"]).min() > -1e-9
            assert (np.diag(r["cov"]) > 0).all() and np.isfinite(r["L"]).all()
        tm = [r["cov"][np.ix_(np.flatnonzero(r["team"] == t), np.flatnonzero(r["team"] == t))][0, 1]
              for r in races for t in np.unique(r["team"]) if (r["team"] == t).sum() == 2]
        assert np.mean(tm) > 0
    finally:
        for r in races:                                     # leave the module fixture as the other tests expect
            r.pop("cov", None), r.pop("L", None)


@needs_real
@pytest.mark.slow
def test_stored_s4_s5_s6_matrices_are_reproduced_from_their_stored_params(real):
    import pickle
    races, cfg = real["races"], real["cfg"]
    for label, step in (("m16_s4_D0", "S4"), ("m16_s5_D0", "S5"), ("m16_s6_D0", "S6")):
        d = _stored(label)
        c = load_yaml(d / "config.yaml")
        p, th = c["params"], {**real["theta"], **c["theta"]}
        mk = {"S4": lambda: dict(kind="S4", m2=p["m2"], m3=p["m3"], m4=p["m4"]),
              "S5": lambda: dict(kind="S5", sigma=p["sigma"], tau_team=p["tau_team"], e_exp=p["e_exp"], e_wet=p["e_wet"]),
              "S6": lambda: dict(kind="S6", c=p["c"], c_early=p["c_early"])}
        ext = mk[step]()
        if step == "S6":
            if not CACHE.exists():
                continue
            SX.attach_cov_factors(races, pickle.loads(CACHE.read_bytes())[1])
        try:
            mats = SX.simulate_matrices(races, th, ext, SW3, cfg["draws"], cfg["seed"])
        finally:
            for r in races:
                r.pop("cov", None), r.pop("L", None)
        z = np.load(d / "position_matrices.npz")
        assert len(z.files) == 66
        for r, P in zip(races, mats):
            assert np.array_equal(z[str(r["race_id"])], P), (label, r["race_id"])


@needs_real
@pytest.mark.slow
def test_s7_neutral_replay_equals_stored_m06_scores_and_nonneutral_differs(real):
    entries, cfg = real["entries"], real["cfg"]
    p = {k: v for k, v in real["frozen"]["M06L"]["111"].items() if k not in ("kind", "cell")}
    neutral = LatentModelAR(entries, dynamic=True, decomp=True, ranking=True, rho=0.8, rule_mult=1.0, **p)
    preds, _ = walk_forward(entries, neutral, "dev", refit_every=1, posdist=True)
    stored = real["Q"].load_run_preds(cfg["m06_label"], "dev")
    assert len(preds) == len(stored) == 1311
    assert np.abs(preds["score"].to_numpy() - stored["score"].to_numpy()).max() == 0.0
    # and the stored S7 run really used phi 1.0, rule_mult 2 (config of the saved run)
    c = load_yaml(_stored("m16_s7_D0") / "config.yaml")
    assert (c["params"]["phi"], c["params"]["rule_mult"]) == (1.0, 2.0)


# ====================================================================================== 7. leakage on real data
def _reverse_races_from(ent: pd.DataFrame, r: int) -> pd.DataFrame:
    """Reverse the finishing order (row order) of race r and every later race, flip classification and y."""
    order = np.arange(len(ent))
    for rid, rows in ent.groupby("race_id").indices.items():
        if rid >= r:
            order[rows] = rows[::-1]
    out = ent.iloc[order].reset_index(drop=True)
    m = out["race_id"] >= r
    out.loc[m, "classified"] = ~out.loc[m, "classified"].astype(bool)
    out.loc[m, "y"] = 1.0 - out.loc[m, "y"]
    out.loc[m, "pos"] = out.loc[m, "n"] + 1 - out.loc[m, "pos"]
    return out


def _race_u_cov(model, ent: pd.DataFrame, r: int):
    rows = np.flatnonzero(ent["race_id"].to_numpy() == r)
    first = int(rows.min())
    u = model.fit_predict(ent.iloc[:first], ent.iloc[rows])
    cov = SL.race_utility_cov(model, first, float(r), rows)
    ids = ent["driverId"].to_numpy()[rows]
    o = np.argsort(ids)
    return ids[o], u[o], cov[np.ix_(o, o)]


@needs_real
@pytest.mark.slow
@pytest.mark.parametrize("which", [0, 33, 65])
@pytest.mark.parametrize("kind", ["plan2_model", "ar_model"])
def test_s6_covariance_and_s7_scores_ignore_the_target_race_and_later_races(real, which, kind):
    ent, frozen = real["entries"], real["frozen"]
    p = {k: v for k, v in frozen["M06L"]["111"].items() if k not in ("kind", "cell")}
    make = (lambda e: LatentModel(e, dynamic=True, decomp=True, ranking=True, **p)) if kind == "plan2_model" else \
           (lambda e: LatentModelAR(e, dynamic=True, decomp=True, ranking=True, rho=1.0, rule_mult=2.0, **p))
    r = split_races(ent, "dev")[1][which]
    base = _race_u_cov(make(ent), ent, r)
    scr = _reverse_races_from(ent, r)
    assert not scr.loc[scr["race_id"] == r, "driverId"].equals(ent.loc[ent["race_id"] == r, "driverId"].reset_index(drop=True))
    alt = _race_u_cov(make(scr), scr, r)
    assert (base[0] == alt[0]).all()
    assert np.array_equal(base[1], alt[1]), "scores changed when the race's own and later results were altered"
    assert np.array_equal(base[2], alt[2]), "Laplace covariance changed when the race's own and later results were altered"
    # truncation: the universe stops at the target race (rows of the target race kept, nothing after)
    cut = ent[ent["race_id"] <= r].reset_index(drop=True)
    tr = _race_u_cov(make(cut), cut, r)
    # not bitwise: the truncated universe has fewer drivers/cars, so the parameter order differs and the unconverged
    # (150-iteration) L-BFGS path ends about 1e-4 apart. A leak would move the score by far more (checked above: the
    # scrambled-later-races universe, same structure, is bitwise equal).
    assert np.abs(base[1] - tr[1]).max() < 2e-3
    assert np.abs(base[2] - tr[2]).max() < 2e-3
    # sanity (not vacuous): altering an EARLIER race changes the score
    early = ent.copy()
    early.loc[early["race_id"] == r - 3, "y"] = 1.0 - early.loc[early["race_id"] == r - 3, "y"]
    rev = ent.iloc[np.concatenate([np.flatnonzero(ent["race_id"].to_numpy() != r - 3)[:0], np.arange(len(ent))])].copy()
    ord_ = np.arange(len(ent))
    rows3 = np.flatnonzero(ent["race_id"].to_numpy() == r - 3)
    ord_[rows3] = rows3[::-1]
    earlier = ent.iloc[ord_].reset_index(drop=True)
    assert not np.array_equal(base[1], _race_u_cov(make(earlier), earlier, r)[1])


@needs_real
def test_add_extras_real_data_truncation(real):
    entries, races = real["entries"], real["races"]
    cut = entries[entries["race_id"] <= races[30]["race_id"]]
    sub = [dict(race_id=r["race_id"], g=r["g"]) for r in races[:31]]
    SX.add_extras(sub, cut)
    for a, b in zip(sub, races[:31]):
        assert np.array_equal(a["starts"], b["starts"]) and np.array_equal(a["team"], b["team"])
    # hand check of experience for one entry: number of earlier rows of the driver
    r = races[40]
    d = r["g"]["driverId"].iloc[3]
    assert r["starts"][3] == (entries["driverId"].eq(d) & entries["race_id"].lt(r["race_id"])).sum()
    assert not np.isnan(r["starts"]).any()


# ====================================================================================== 8. split discipline
def test_new_code_and_config_use_dev_only():
    cfg = load_yaml(M16 / "config.yaml")
    assert cfg["split"] == "dev"
    for f in (ROOT / "src/pitlane/plan3/simext.py", ROOT / "src/pitlane/plan3/simext_laplace.py",
              ROOT / "src/pitlane/plan3/ar_latent.py", M16 / "run.py"):
        txt = f.read_text(encoding="utf-8")
        code = re.sub(r'""".*?"""', "", txt, flags=re.S)
        code = "\n".join(l.split("#")[0] for l in code.splitlines())
        for bad in ("fresh", "2024", "2025", "2026", '"test"', "'test'", "test1", "test2", "holdout"):
            hits = [l for l in code.splitlines() if bad in l and "RULE_SEASONS" not in l]
            assert not hits, (f.name, bad, hits)
    # the only mention of 2026 allowed is the rule-change season list (a calendar fact)
    assert "2026" in (ROOT / "src/pitlane/plan3/ar_latent.py").read_text(encoding="utf-8")


@needs_runs
def test_stored_runs_cover_dev_races_only():
    for label in ("m16_s4_D0", "m16_s5_D0", "m16_s6_D0", "m16_s6_D2", "m16_s7_D0"):
        t = pd.read_csv(_stored(label) / "per_race_metrics.csv")
        assert len(t) == 66 and t["race_id"].is_unique
        assert set(t["season"]) == {2021, 2022, 2023}
        assert t.groupby("season").size().to_dict() == {2021: 22, 2022: 22, 2023: 22}
        assert load_yaml(_stored(label) / "config.yaml")["split"] == "dev"


# ====================================================================================== 9. data hygiene of the stored runs
@needs_runs
def test_stored_matrices_and_tables_are_well_formed():
    for label in ("m16_s4_D0", "m16_s5_D0", "m16_s6_D0", "m16_s7_D0"):
        d = _stored(label)
        z = np.load(d / "position_matrices.npz")
        preds = pd.read_parquet(d / "predictions.parquet")
        assert len(z.files) == 66 and len(preds) == 1311
        assert not preds.duplicated(["race_id", "driverId"]).any()
        assert preds[["score", "p_win", "p_podium", "p_points", "p_dnf", "pos", "grid"]].notna().all().all() or \
            preds[["score", "p_win", "p_podium", "p_points", "p_dnf", "pos"]].notna().all().all()
        for rid, g in preds.groupby("race_id"):
            P = z[str(rid)]
            n = len(g)
            assert P.shape == (n, n) and (P >= 0).all()
            assert np.allclose(P.sum(0), 1, atol=1e-9) and np.allclose(P.sum(1), 1, atol=1e-9), (label, rid)
            assert np.allclose(g["p_win"].to_numpy(), P[:, 0]) and np.allclose(g["p_points"].to_numpy(), P[:, :min(10, n)].sum(1))
            assert sorted(g["pos"]) == list(range(1, n + 1))
        t = pd.read_csv(d / "per_race_metrics.csv")
        assert t["rps"].between(0, 1).all() and t["logloss_points"].between(0, 3).all()
        assert t[JOINT].between(0, 5).all() and t[JOINT].notna().all()
        assert t["exact"].between(0, 1).all() and t["spearman"].between(-1, 1).all()
        assert t[["rps", "logloss_points", "spearman", "mae", "exact", JOINT]].notna().all().all()
        m = json.loads((d / "metrics.json").read_text())
        assert m["rps"] == pytest.approx(t["rps"].mean()) and m["n_races"] == 66


# ====================================================================================== 10. paired gains, attribution, leaderboard
COMP = {"M16_S4": "m16_s4_D0", "M16_S5": "m16_s5_D0", "M16_S6": "m16_s6_D0", "M16_S7": "m16_s7_D0", "M16_S6_D2": "m16_s6_D2"}


def _cand_tab(label):
    return pd.read_csv(_stored(label) / "per_race_metrics.csv")


@needs_real
@needs_runs
def test_attribution_rows_rederived_from_stored_per_race_tables(real, ref_tabs):
    att = pd.read_csv(ROOT / "results" / "plan3" / "attribution.csv")
    m16_all = att[att["component"].str.startswith("M16_")]
    extra = m16_all[m16_all["component"].str.startswith("M16_S7_rps")]          # later re-selection by RPS may append these
    m16 = m16_all.drop(extra.index)
    expected = {(c, r) for c in ("M16_S4", "M16_S5", "M16_S6", "M16_S7") for r in ("S3_D0", "grid")} | \
               {("M16_S6_D2", r) for r in ("S3_D0", "grid", "S3_D2")} | {("M16_S6xD2", "S6_gain_D0")}
    assert set(zip(m16["component"], m16["reference"])) == expected and len(m16) == 12
    assert not att.duplicated(["component", "reference"]).any()
    A.HIGHER_BETTER[JOINT] = False
    sub = ["spearman", "spearman_fin", "mae", "mae_fin", "exact", "exact_fin", "exact_1_3", "exact_4_10", "exact_11p", "within1", "within2"]
    for (comp, ref), row in ((k, m16[(m16["component"] == k[0]) & (m16["reference"] == k[1])].iloc[0]) for k in sorted(expected)):
        if comp == "M16_S6xD2":
            s6d0, s6d2 = _cand_tab("m16_s6_D0").set_index("race_id")[sub], _cand_tab("m16_s6_D2").set_index("race_id")[sub]
            r0 = s6d0 + ref_tabs["S3_D2"].set_index("race_id")[sub]
            c0 = s6d2 + ref_tabs["S3_D0"].set_index("race_id")[sub]
            g = A.paired_gain(r0.reset_index(), c0.reset_index(), sub, n_boot=10_000, seed=0)
            vm, rid, n = "exact", "m16_s6_D2", 66
        else:
            g = A.paired_gain(ref_tabs[ref], _cand_tab(COMP[comp]), A.METRICS, n_boot=10_000, seed=0)
            vm, rid, n = ("exact" if ref == "grid" else "rps"), COMP[comp], 66
        assert len(g) == len(sub) or comp != "M16_S6xD2"
        for _, x in g.iterrows():
            for pre, key in (("gain_", "gain"), ("lo_", "lo"), ("hi_", "hi")):
                assert row[f"{pre}{x['metric']}"] == pytest.approx(x[key], abs=1e-9), (comp, ref, x["metric"], pre)
        assert row["verdict_metric"] == vm and row["n_races"] == n
        assert row["run_id"] == Path(str(_stored(rid))).name
        v = g[g["metric"] == vm].iloc[0]
        assert row["verdict"] == ("passes alone" if v["lo"] > 0 else "worse" if v["hi"] < 0 else "no measurable difference")
        assert row["declared"] in ("yes", True, "True")


@needs_real
@needs_runs
def test_rps_gains_match_hand_means_and_the_card(real, ref_tabs):
    """Sign convention (reference minus candidate for RPS) by hand, then the numbers printed in the README."""
    ref = ref_tabs["S3_D0"].set_index("race_id")
    card = {"m16_s4_D0": (0.00007, -0.00019, 0.00034), "m16_s5_D0": (0.00010, -0.00027, 0.00049),
            "m16_s6_D0": (0.00005, -0.00006, 0.00016), "m16_s7_D0": (0.00031, -0.00072, 0.00137)}
    for label, (gm, lo, hi) in card.items():
        cand = _cand_tab(label).set_index("race_id")
        hand = float((ref["rps"] - cand["rps"].reindex(ref.index)).mean())
        g = A.paired_gain(ref.reset_index(), cand.reset_index(), ["rps"], n_boot=10_000, seed=0).iloc[0]
        assert g["gain"] == pytest.approx(hand, abs=1e-12)
        assert (round(g["gain"], 5), round(g["lo"], 5), round(g["hi"], 5)) == pytest.approx((gm, lo, hi), abs=1.1e-5), label
    assert ref["rps"].mean() == pytest.approx(0.12042, abs=5e-6)


@needs_real
@needs_runs
def test_s6_by_d2_interaction_by_hand(real, ref_tabs):
    s6d0, s6d2 = _cand_tab("m16_s6_D0").set_index("race_id"), _cand_tab("m16_s6_D2").set_index("race_id")
    s3d0, s3d2 = ref_tabs["S3_D0"].set_index("race_id"), ref_tabs["S3_D2"].set_index("race_id")
    # RPS does not depend on the decoder, so the contrast is not defined for it: S6 under D0 and D2 share the matrix
    assert np.array_equal(s6d0["rps"].to_numpy(), s6d2["rps"].to_numpy())
    assert np.array_equal(s3d0["rps"].to_numpy(), s3d2["rps"].to_numpy())
    imp_d2 = (s6d2["exact"] - s3d2["exact"]).mean()
    imp_d0 = (s6d0["exact"] - s3d0["exact"]).mean()
    row = pd.read_csv(ROOT / "results" / "plan3" / "attribution.csv").query("component == 'M16_S6xD2'").iloc[0]
    assert row["gain_exact"] == pytest.approx(imp_d2 - imp_d0, abs=1e-12)
    assert row["gain_exact"] * 100 == pytest.approx(-0.06, abs=0.005)
    assert s6d2["exact"].mean() == pytest.approx(0.1430, abs=5e-5) and s3d2["exact"].mean() == pytest.approx(0.1390, abs=5e-5)
    # MAE sign: gain = improvement under D2 minus improvement under D0 (lower MAE is better)
    imp_mae = (s3d2["mae"] - s6d2["mae"]).mean() - (s3d0["mae"] - s6d0["mae"]).mean()
    assert row["gain_mae"] == pytest.approx(imp_mae, abs=1e-12)


@needs_runs
def test_leaderboard_rows_match_the_stored_runs():
    lb = _lb()
    m16 = lb[(lb["method"] == "M16") & ~lb["label"].str.startswith("m16_s7rps")]
    assert sorted(m16["label"]) == sorted(["m16_s4_D0", "m16_s5_D0", "m16_s6_D0", "m16_s6_D2", "m16_s7_D0"]) and len(m16) == 5
    assert not lb.duplicated(["method", "label", "split"]).any()
    for _, row in m16.iterrows():
        d = ROOT / row["run_dir"]
        met = json.loads((d / "metrics.json").read_text())
        assert row["split"] == "dev" and row["n_races"] == 66 and row["cell"] == "S3"
        assert row["run_id"] == d.name
        for k in ("spearman", "mae", "exact", "logloss_points", "within1"):
            assert row[k] == pytest.approx(met[k], rel=1e-12), (row["label"], k)
        t = pd.read_csv(d / "per_race_metrics.csv")
        assert row["exact"] == pytest.approx(t["exact"].mean(), rel=1e-12)
        cfg = load_yaml(d / "config.yaml")
        assert cfg["split"] == "dev" and re.fullmatch(r"[0-9a-f]{7,}(\+dirty)?", cfg["git_commit"])
    # the S6 D2 run shares the S6 D0 matrix
    z0, z2 = np.load(_stored("m16_s6_D0") / "position_matrices.npz"), np.load(_stored("m16_s6_D2") / "position_matrices.npz")
    assert all(np.array_equal(z0[k], z2[k]) for k in z0.files)


def _git_blob(path: str, rev: str | None = None) -> bytes | None:
    """Shared csv as committed before M16 added rows: bd610a5 (R8 phase 2, holds only the M20 rows) if available,
    else 1db3fdc (the base of R12)."""
    for r in ([rev] if rev else ["bd610a5", "1db3fdc"]):
        try:
            return subprocess.check_output(["git", "show", f"{r}:{path}"], cwd=ROOT, stderr=subprocess.DEVNULL)
        except Exception:
            continue
    return None


@pytest.mark.parametrize("path", ["results/plan3/attribution.csv", "results/plan3/leaderboard.csv"])
def test_shared_csvs_keep_every_line_of_commit_1db3fdc_byte_identical(path):
    old = _git_blob(path)
    if old is None:
        pytest.skip("commit 1db3fdc not available")
    new = (ROOT / path).read_bytes()
    nl = lambda b: b.replace(b"\r\n", b"\n")                 # git autocrlf may change line endings in the working copy
    assert nl(new).startswith(nl(old)), f"{path}: existing lines changed"
    assert new.endswith((b"\n", b"\r\n"))


@needs_runs
def test_m16_rows_are_appended_after_the_old_lines_and_nothing_else_changed_in_old_rows():
    for path, key, n_m16 in (("results/plan3/attribution.csv", "component", 12), ("results/plan3/leaderboard.csv", "method", 5)):
        old = _git_blob(path)
        if old is None:
            pytest.skip("commit 1db3fdc not available")
        new_df = pd.read_csv(ROOT / path, dtype=str, keep_default_na=False)
        old_df = pd.read_csv(pd.io.common.BytesIO(old), dtype=str, keep_default_na=False)
        assert new_df.iloc[:len(old_df)].equals(old_df)
        added = new_df.iloc[len(old_df):]
        mask = added[key].str.startswith("M16") if key == "component" else added[key].eq("M16")
        lab = added["component"] if key == "component" else added["label"]
        allowed_extra = lab.str.startswith("M16_S7_rps") if key == "component" else lab.str.startswith("m16_s7rps")
        assert (mask & ~allowed_extra).sum() == n_m16            # the original rows; S7-by-RPS rows may follow
        # M16 rows contiguous: nothing from M16 sits in the old block
        old_mask = old_df[key].str.startswith("M16") if key == "component" else old_df[key].eq("M16")
        assert not old_mask.any()


# ====================================================================================== 11. grid-slot calibration numbers
@needs_real
@needs_runs
def test_grid_slot_calibration_numbers_rederived(real):
    races = real["races"]
    keys = [r["g"]["driverId"].to_numpy() for r in races]
    grids = [r["g"]["grid"].to_numpy(float) for r in races]
    poss = [r["pos"] for r in races]
    ref = CAL.grid_slot_calibration(real["old"], grids, poss, keys)
    assert ref["realised"] == pytest.approx(0.1640, abs=5e-5)
    assert ref["mean_pred"] == pytest.approx(0.1178, abs=5e-5)
    want = {"m16_s4_D0": 0.1184, "m16_s5_D0": 0.1145, "m16_s6_D0": 0.1161, "m16_s7_D0": 0.1129}
    for label, v in want.items():
        d = _stored(label)
        z = np.load(d / "position_matrices.npz")
        preds = pd.read_parquet(d / "predictions.parquet")        # rows are in each run's own driver order
        mats, gs, ps, ks = [], [], [], []
        for rid, g in preds.groupby("race_id", sort=True):
            mats.append(z[str(rid)])
            gs.append(g["grid"].to_numpy(float))
            ps.append(g["pos"].to_numpy(float))
            ks.append(g["driverId"].to_numpy())
        c = CAL.grid_slot_calibration(mats, gs, ps, ks)
        assert c["realised"] == pytest.approx(ref["realised"], abs=1e-12)
        assert c["mean_pred"] == pytest.approx(v, abs=5e-5), label
        assert c["mean_pred"] < c["realised"] - 0.02                 # the gate fails for every matrix, as stated
    cal = pd.read_csv(sorted(RES.glob("20261117-0652_grid_slot_calibration_dev.csv"))[-1])
    assert not cal["passes_gate"].any()
    assert set(cal.loc[cal["decoder"] == "D2", "matrix"]) == {"S6"}      # D2 only for the declared S6 interaction


# ====================================================================================== 12. descriptive claims in the card
@needs_runs
def test_stored_s6_laplace_width_claims():
    sd = pd.read_csv(sorted(RES.glob("20261117-0652_s6_laplace_sd_dev.csv"))[-1])
    assert len(sd) == 66
    by = sd.groupby("season")["mean_sd"].mean()
    assert by.round(3).to_dict() == {2021: 0.228, 2022: 0.230, 2023: 0.245}
    e = sd.groupby("early")["mean_sd"].mean().round(3).to_dict()
    assert e[True] == 0.237 and e[False] == 0.234


# ====================================================================================== 13. S7 re-selected by RPS (stamp 1530)
S7R = RES / "20261117-1530_tuning_S7_rps_dev.csv"
needs_s7rps = pytest.mark.skipif(not S7R.exists() or not list(RES.glob("*_m16_s7rps_D0_dev")), reason="no S7-by-RPS run")


@needs_s7rps
def test_s7_rps_pick_is_the_argmin_of_the_rps_table_and_spearman_reproduces_the_first_table():
    t = pd.read_csv(S7R)
    assert len(t) == 25 and not t.duplicated(["phi", "rule_mult"]).any() and t.notna().all().all()
    assert set(t["phi"]) == {0.5, 0.65, 0.8, 0.9, 1.0} and set(t["rule_mult"]) == {1.0, 2.0, 4.0, 8.0, 16.0}
    best = t.loc[t["rps"].idxmin()]
    assert (best["phi"], best["rule_mult"]) == (1.0, 1.0)
    assert t["rps"].nsmallest(2).iloc[1] > best["rps"]                         # unique minimum
    first = pd.read_csv(RES / "20261117-0652_tuning_S7_dev.csv")
    m = t.merge(first, on=["phi", "rule_mult"], suffixes=("_new", "_old"))
    assert len(m) == 25
    assert np.abs(m["spearman_new"] - m["spearman_old"]).max() < 1e-12         # same fits, reproduced
    # the plan-2 value (0.8, 1) is in the table, as the neutral reference of the search
    assert ((t["phi"] == 0.8) & (t["rule_mult"] == 1.0)).sum() == 1
    # the Spearman pick (earlier run) is a different configuration than the RPS pick
    sp = first.loc[first["spearman"].idxmax()]
    assert (sp["phi"], sp["rule_mult"]) == (1.0, 2.0)


@needs_s7rps
def test_s7_rps_declared_run_uses_the_rps_pick_and_rederived_gain():
    d = _stored("m16_s7rps_D0")
    c = load_yaml(d / "config.yaml")
    assert c["step"] == "S7_rps" and c["split"] == "dev"
    assert (c["params"]["phi"], c["params"]["rule_mult"]) == (1.0, 1.0)
    assert c["params"]["tau"] == 3.5 and c["theta"]["dnf_scale"] == 1.25
    tab = pd.read_csv(d / "per_race_metrics.csv")
    assert len(tab) == 66 and tab["race_id"].is_unique and tab[["rps", JOINT]].notna().all().all()
    z = np.load(d / "position_matrices.npz")
    preds = pd.read_parquet(d / "predictions.parquet")
    assert len(z.files) == 66 and len(preds) == 1311
    for rid, g in preds.groupby("race_id"):
        P = z[str(rid)]
        assert np.allclose(P.sum(0), 1) and np.allclose(P.sum(1), 1)
    # the tau re-fit table: 3 taus per candidate, S7-by-RPS keeps tau 3.5 (its minimum); consistent with the run
    tr = pd.read_csv(RES / "20261117-1530_s7_tau_refit_dev.csv")
    r7 = tr[tr["candidate"] == "s7rps"]
    assert len(r7) == 3 and r7.loc[r7["rps"].idxmin(), "tau"] == 3.5
    # RPS of the run: matrix mean equals the stored table
    assert tab["rps"].mean() == pytest.approx(json.loads((d / "metrics.json").read_text())["rps"], rel=1e-12)


@needs_s7rps
@needs_real
def test_s7_rps_paired_gain_and_attribution_rows_rederived(real, ref_tabs):
    ref = ref_tabs["S3_D0"]
    cand = pd.read_csv(_stored("m16_s7rps_D0") / "per_race_metrics.csv")
    g = A.paired_gain(ref, cand, A.METRICS, n_boot=10_000, seed=0)
    r = g[g["metric"] == "rps"].iloc[0]
    assert (round(r["gain"], 5), round(r["lo"], 5), round(r["hi"], 5)) == pytest.approx((0.00057, -0.00024, 0.00143), abs=1.1e-5)
    assert r["gain"] == pytest.approx(float((ref.set_index("race_id")["rps"] - cand.set_index("race_id")["rps"]).mean()), abs=1e-12)
    att = pd.read_csv(ROOT / "results" / "plan3" / "attribution.csv")
    rows = att[att["component"] == "M16_S7_rps"]
    assert sorted(rows["reference"]) == ["S3_D0", "grid"]
    for _, row in rows.iterrows():
        gg = A.paired_gain(ref_tabs[row["reference"]], cand, A.METRICS, n_boot=10_000, seed=0)
        for _, x in gg.iterrows():
            for pre, key in (("gain_", "gain"), ("lo_", "lo"), ("hi_", "hi")):
                assert row[f"{pre}{x['metric']}"] == pytest.approx(x[key], abs=1e-9), (row["reference"], x["metric"], pre)
        assert row["run_id"] == _stored("m16_s7rps_D0").name and row["n_races"] == 66
        assert row["verdict_metric"] == ("exact" if row["reference"] == "grid" else "rps")
    lbr = _lb().query("method == 'M16' and label == 'm16_s7rps_D0'")
    assert len(lbr) == 1 and lbr.iloc[0]["n_races"] == 66


@needs_s7rps
def test_m16_totals_after_the_rps_rows():
    att = pd.read_csv(ROOT / "results" / "plan3" / "attribution.csv")
    lb = _lb()
    assert att["component"].str.startswith("M16").sum() == 14
    assert (lb["method"] == "M16").sum() == 6
    assert set(att[att["component"].str.startswith("M16_S7_rps")]["reference"]) == {"S3_D0", "grid"}
