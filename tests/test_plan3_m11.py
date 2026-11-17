"""R5 / M11 likelihood variants of the latent model (finishers_only, topk, reverse): independent checks.

Hand-written Plackett-Luce terms, finite differences of the full penalised objective, equality with the HEAD
version of latent.py when all switches are off, leakage of the classification flag, parameter recovery,
and the stored result files of methodologies/M11_likelihood_variants.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitlane.config import PROCESSED, ROOT, load_yaml
from pitlane.evaluation.harness import walk_forward
from pitlane.evaluation.metrics import race_metrics
from pitlane.models.latent import LatentModel
from pitlane.models.registry import build_model
from pitlane.plan3 import attribution as A
from pitlane.plan3 import decode as DEC

M11 = ROOT / "methodologies" / "M11_likelihood_variants"
RES = M11 / "results"
REAL = PROCESSED / "entries_feat.parquet"
needs_real = pytest.mark.skipif(not REAL.exists(), reason="data/processed/entries_feat.parquet missing (regenerable)")

SWITCHES = {
    "finishers": dict(finishers_only=True),
    "top10": dict(topk=10),
    "reverse": dict(reverse=True),
    "finishers+reverse": dict(finishers_only=True, reverse=True),
}
VARIANT_RUNS = {  # attribution component -> (run folder suffix, switches)
    "M11_finishers": ("m11_latent_111_finishers_dev", dict(finishers_only=True)),
    "M11_topk10": ("m11_latent_111_top10_dev", dict(topk=10)),
    "M11_reverse": ("m11_latent_111_reverse_dev", dict(reverse=True)),
}


# ------------------------------------------------------------------ synthetic data
def _frame(seed: int = 0, n_races: int = 24) -> pd.DataFrame:
    """12 drivers, 6 lineages, two seasons, varying field size, retirees (also one ahead of a classified entry,
    one race with a single classified entry, one with none), sorted by race_id, pos."""
    rng = np.random.default_rng(seed)
    drivers = [f"d{i:02d}" for i in range(12)]
    a = rng.normal(0, 0.7, 12)
    rows = []
    for r in range(n_races):
        season = 2020 if r < n_races // 2 else 2021
        k = int(rng.integers(8, 13))
        who = np.sort(rng.choice(12, k, replace=False))
        u = a[who] + rng.gumbel(size=k)
        order = who[np.argsort(-u)]
        cls = rng.random(k) > 0.25
        if r == 5:      # retiree ahead of a classified entry
            cls[:] = True
            cls[2] = False
        if r == 7:      # a single classified entry
            cls[:] = False
            cls[0] = True
        if r == 8:      # nobody classified
            cls[:] = False
        # classified first (as the ingest does) except in race 5
        if r != 5:
            idx = np.argsort(~cls, kind="stable")
            order, cls = order[idx], cls[idx]
        for pos, (d, c) in enumerate(zip(order, cls), 1):
            rows.append(dict(race_id=r, season=season, driverId=drivers[d], lineage=f"c{(d // 2 + (r // 8)) % 6}",
                             circuit_type="ab"[r % 2], pos=pos, n=k, y=(pos - 1) / (k - 1), classified=bool(c),
                             grid_norm=float(rng.random()), quali_norm=float(rng.random()),
                             q1_gap=float(rng.random() * 0.1), pit_start=int(rng.random() < 0.05),
                             x_extra=float(rng.normal())))
    return pd.DataFrame(rows)


def _model(df, **kw):
    base = dict(dynamic=True, decomp=True, ranking=True, halflife=6.0, lam=1.3, lam_a=0.7, lam_x=0.5, x_cols=["x_extra"])
    base.update(kw)
    return LatentModel(df, **base)


# ------------------------------------------------------------------ 1. finite differences
@pytest.mark.parametrize("name", list(SWITCHES))
def test_gradient_matches_finite_differences_full_objective(name):
    df = _frame(1)
    m = _model(df, **SWITCHES[name])
    n = len(df)
    w, pad = m.fit_inputs(n, t_ref=float(df["race_id"].max() + 1))
    assert len(pad) == 3 if name != "reverse" else len(pad) == 2
    assert w.min() < 0.5 * w.max(), "time-decay weights are not exercised"
    for seed in (0, 1, 2):
        th = np.random.default_rng(seed).normal(0, 0.4, m.theta.size)
        if seed == 2:                       # exact ties in strength: two drivers share the same effect
            th[1] = th[0]
        f0, g = m._objective(th, n, w, pad)
        assert np.isfinite(f0) and np.isfinite(g).all()
        fd = np.empty_like(th)
        for j in range(th.size):
            e = np.zeros_like(th)
            e[j] = 1e-6
            fd[j] = (m._objective(th + e, n, w, pad)[0] - m._objective(th - e, n, w, pad)[0]) / 2e-6
        err = np.abs(fd - g).max()
        assert err < 2e-7 + 1e-5 * np.abs(g).max(), (name, seed, err)


def test_switches_change_the_objective():
    """Sanity so that the gradient tests above are not vacuous: every switch gives a different loss."""
    df = _frame(1)
    th = np.random.default_rng(0).normal(0, 0.4, _model(df).theta.size)
    vals = {}
    for name in ["off", *SWITCHES]:
        m = _model(df, **({} if name == "off" else SWITCHES[name]))
        w, pad = m.fit_inputs(len(df), 25.0)
        vals[name] = m._objective(th, len(df), w, pad)[0]
    assert len({round(v, 9) for v in vals.values()}) == len(vals), vals


# ------------------------------------------------------------------ 2. hand-computed likelihood
def _pl(us):
    """Plackett-Luce log-likelihood of the list `us` (strengths, in finishing order), written out term by term."""
    ll = 0.0
    for j in range(len(us)):
        ll += us[j] - np.log(sum(np.exp(us[k]) for k in range(j, len(us))))
    return ll


def _hand(us, cls, mode, k=None):
    us, cls = list(us), list(cls)
    if mode == "off":
        return _pl(us)
    if mode == "finishers":
        return _pl([u for u, c in zip(us, cls) if c])
    if mode == "reverse":
        return _pl([-u for u in reversed(us)])
    if mode == "finishers+reverse":
        return _pl([-u for u, c in zip(reversed(us), reversed(cls)) if c])
    if mode == "topk":
        return sum(us[j] - np.log(sum(np.exp(us[i]) for i in range(j, len(us)))) for j in range(min(k, len(us))))
    raise ValueError(mode)


def _tiny():
    """Race 0: 5 drivers, retiree in slot 3 (ahead of a classified entry); race 1: 4 drivers, retiree last."""
    rows = []
    for r, (ds, cl) in enumerate([(["a", "b", "c", "d", "e"], [1, 1, 0, 1, 1]), (["b", "f", "g", "h"], [1, 1, 1, 0])]):
        for p, (d, c) in enumerate(zip(ds, cl), 1):
            rows.append(dict(race_id=r, season=2020, driverId=d, lineage="L", circuit_type="x", pos=p, n=len(ds),
                             y=(p - 1) / (len(ds) - 1), classified=bool(c), grid_norm=.5, quali_norm=.5, q1_gap=0., pit_start=0))
    return pd.DataFrame(rows)


def _tiny_model(**kw):
    df = _tiny()
    m = LatentModel(df, dynamic=False, decomp=False, ranking=True, base_x=False, lam_a=0.0, **kw)
    # row weights 2.0 (race 0) and 0.5 (race 1), so W != number of rows
    w = np.where(df["race_id"].to_numpy() == 0, 2.0, 0.5)
    pad = m._pad(2)
    return df, m, w, pad


def _hand_total(df, a_by_driver, mode, k=None):
    tot = 0.0
    for r, wr in ((0, 2.0), (1, 0.5)):
        g = df[df["race_id"] == r]
        tot += wr * _hand([a_by_driver[d] for d in g["driverId"]], g["classified"], mode, k)
    return tot


@pytest.mark.parametrize("mode,kw,k", [("off", {}, None), ("finishers", dict(finishers_only=True), None),
                                       ("topk", dict(topk=3), 3), ("topk", dict(topk=1), 1),
                                       ("topk", dict(topk=10), 10),
                                       ("reverse", dict(reverse=True), None),
                                       ("finishers+reverse", dict(finishers_only=True, reverse=True), None)])
def test_hand_computed_likelihood_and_gradient(mode, kw, k):
    df, m, w, pad = _tiny_model(**kw)
    a = {d: v for d, v in zip(sorted(df["driverId"].unique()), np.random.default_rng(3).normal(0, 0.8, m.nD))}
    th = np.array([a[d] for d in sorted(m.drivers, key=m.drivers.get)])
    W = w.sum()
    f, g = m._objective(th, len(df), w, pad)
    assert f * W == pytest.approx(-_hand_total(df, a, mode, k), abs=1e-12, rel=1e-12)
    # gradient: central differences of the hand-written likelihood, independent of the model code
    names = sorted(m.drivers, key=m.drivers.get)
    for j, d in enumerate(names):
        up, dn = dict(a), dict(a)
        up[d] += 1e-6
        dn[d] -= 1e-6
        fd = -(_hand_total(df, up, mode, k) - _hand_total(df, dn, mode, k)) / 2e-6 / W
        assert g[j] == pytest.approx(fd, abs=1e-8), (mode, d)


def test_topk_structure_only_first_k_stages_and_full_choice_sets():
    """k=2 of a 5-entry race: swapping entries in slots 3..5 leaves the likelihood alone (they stay in the choice
    sets as a set), swapping slots 2 and 3 changes it, and the value is exactly the first two PL terms."""
    u = np.array([0.4, -0.2, 0.9, 0.1, -0.5])
    two = _hand(u, [1] * 5, "topk", 2)
    manual = (u[0] - np.log(np.exp(u).sum())) + (u[1] - np.log(np.exp(u[1:]).sum()))
    assert two == pytest.approx(manual, abs=1e-14)
    assert _hand(u[[0, 1, 4, 2, 3]], [1] * 5, "topk", 2) == pytest.approx(two, abs=1e-14)
    assert _hand(u[[0, 2, 1, 3, 4]], [1] * 5, "topk", 2) != pytest.approx(two, abs=1e-6)
    # the model agrees on the same structure: a 5-driver race, topk=2
    rows = [dict(race_id=0, season=2020, driverId=f"p{i}", lineage="L", circuit_type="x", pos=i + 1, n=5, y=i / 4,
                 classified=True, grid_norm=.5, quali_norm=.5, q1_gap=0., pit_start=0) for i in range(5)]
    df = pd.DataFrame(rows)
    m = LatentModel(df, dynamic=False, decomp=False, ranking=True, base_x=False, lam_a=0.0, topk=2)
    th = np.array([u[int(d[1:])] for d in sorted(m.drivers, key=m.drivers.get)])
    f, _ = m._objective(th, 5, np.ones(5), m._pad(1))
    assert f * 5 == pytest.approx(-manual, abs=1e-12)
    _, mask, stage = m._pad(1)
    assert mask.all() and stage[0].tolist() == [True, True, False, False, False]


def test_finishers_only_retirees_are_in_no_stage_and_no_choice_set():
    """Code: retirees are removed from both the stages and the choice sets (full censoring). The method card says the
    same ('no stage of their own and not in anyone's choice set'). Checked through the mask and through the
    gradient of a driver who only ever appears as a retiree (it must be exactly zero)."""
    df, m, w, pad = _tiny_model(finishers_only=True)
    idx, mask, stage = pad
    assert (mask == stage).all()
    cls = df["classified"].to_numpy()
    assert not cls[idx[mask]].tolist().count(False), "a retiree is in a choice set"
    assert mask.sum() == cls.sum()
    # drivers c (race 0) and h (race 1) are retirees appearing once
    th = np.random.default_rng(5).normal(0, 0.8, m.nD)
    _, g = m._objective(th, len(df), w, pad)
    for d in ("c", "h"):
        assert g[m.drivers[d]] == 0.0
    m_off = LatentModel(df, dynamic=False, decomp=False, ranking=True, base_x=False, lam_a=0.0)
    _, g_off = m_off._objective(th, len(df), w, m_off._pad(2))
    assert g_off[m.drivers["c"]] != 0.0
    # removing the retiree's u cannot matter: change the value of its strength, loss stays the same
    th2 = th.copy()
    th2[m.drivers["c"]] += 3.0
    assert m._objective(th2, len(df), w, pad)[0] == m._objective(th, len(df), w, pad)[0]


def test_reverse_equals_standard_pl_on_reversed_order_with_negated_strengths():
    """Run the standard (all-off) model on the reversed frame with negated strengths: same loss, mirrored gradient."""
    df = _frame(4)
    rev = df.iloc[::-1].copy()
    rev["pos"] = rev.groupby("race_id").cumcount() + 1
    rev = rev.sort_values(["race_id", "pos"]).reset_index(drop=True)
    kw = dict(dynamic=False, decomp=False, ranking=True, base_x=False, lam_a=0.0)
    df = df.reset_index(drop=True)
    mr = LatentModel(df, reverse=True, **kw)
    ms = LatentModel(rev, **kw)
    assert mr.drivers == ms.drivers
    th = np.random.default_rng(0).normal(0, 0.6, mr.theta.size)
    w = np.ones(len(df))
    f1, g1 = mr._objective(th, len(df), w, mr._pad(24))
    f2, g2 = ms._objective(-th, len(rev), np.ones(len(rev)), ms._pad(24))
    assert f1 == pytest.approx(f2, abs=1e-12, rel=1e-12)
    assert np.allclose(g1, -g2, atol=1e-12)


# ------------------------------------------------------------------ 3. all switches off = HEAD
@pytest.fixture(scope="module")
def head_latent(tmp_path_factory):
    try:
        src = subprocess.check_output(["git", "show", "c3b4fbc:src/pitlane/models/latent.py"], cwd=ROOT, text=True)
    except Exception as e:                                           # pragma: no cover
        pytest.skip(f"git show failed: {e}")
    d = tmp_path_factory.mktemp("head_latent")
    p = d / "latent_head.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("latent_head", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["latent_head"] = mod
    spec.loader.exec_module(mod)
    return mod


def _compare_to_head(head, df, n_train, t_ref, **kw):
    new, old = LatentModel(df, **kw), head.LatentModel(df, **kw)
    w1, p1 = new.fit_inputs(n_train, t_ref)
    w2, p2 = old.fit_inputs(n_train, t_ref)
    assert (w1 == w2).all()
    if kw["ranking"]:
        assert len(p1) == 2 and (p1[0] == p2[0]).all() and (p1[1] == p2[1]).all()
    th = np.random.default_rng(0).normal(0, 0.4, new.theta.size)
    f1, g1 = new._objective(th, n_train, w1, p1)
    f2, g2 = old._objective(th, n_train, w2, p2)
    assert f1 == f2 and (g1 == g2).all()
    new.fit(n_train, t_ref)
    old.fit(n_train, t_ref)
    assert (new.theta == old.theta).all()


@pytest.mark.parametrize("cell", ["111", "110", "011", "101"])
@pytest.mark.parametrize("lam_a", [None, 9.0])
def test_switches_off_identical_to_head_synthetic(head_latent, cell, lam_a):
    df = _frame(2)
    n = int(np.flatnonzero(df["race_id"].to_numpy() == 18)[0])     # a race boundary
    dyn, dec, rk = (c == "1" for c in cell)
    _compare_to_head(head_latent, df, n, float(df["race_id"].iloc[n]), dynamic=dyn, decomp=dec, ranking=rk,
                     halflife=10.0, lam=2.0, lam_a=lam_a, x_cols=["x_extra"])


@needs_real
@pytest.mark.parametrize("cell,lam_a", [("111", None), ("110", None), ("111", 64.0), ("110", 64.0)])
def test_switches_off_identical_to_head_real(head_latent, cell, lam_a):
    e = pd.read_parquet(REAL)
    n = int((e["race_id"] < 205).sum())
    dyn, dec, rk = (c == "1" for c in cell)
    _compare_to_head(head_latent, e, n, 205.0, dynamic=dyn, decomp=dec, ranking=rk, halflife=50, lam=32.0, lam_a=lam_a)


def test_degenerate_switches_equal_all_off():
    """topk larger than every field and finishers_only on all-classified data are the plan-1 likelihood
    (different code path, so equal to rounding only)."""
    df = _frame(6)
    df["classified"] = True
    th = np.random.default_rng(1).normal(0, 0.4, _model(df).theta.size)
    ref = _model(df)
    w, p = ref.fit_inputs(len(df), 25.0)
    f0, g0 = ref._objective(th, len(df), w, p)
    for kw in (dict(topk=99), dict(finishers_only=True), dict(finishers_only=True, topk=99)):
        m = _model(df, **kw)
        f, g = m._objective(th, len(df), w, m.fit_inputs(len(df), 25.0)[1])
        assert f == pytest.approx(f0, rel=1e-13) and np.allclose(g, g0, atol=1e-13), kw


# ------------------------------------------------------------------ 4. invalid combinations
@pytest.mark.parametrize("kw", [dict(finishers_only=True), dict(topk=10), dict(reverse=True),
                                dict(finishers_only=True, reverse=True)])
def test_switch_without_ranking_raises(kw):
    with pytest.raises(ValueError):
        LatentModel(_frame(0), dynamic=True, decomp=True, ranking=False, **kw)


def test_reverse_with_topk_and_bad_k_raise():
    df = _frame(0)
    with pytest.raises(ValueError):
        LatentModel(df, dynamic=True, decomp=True, ranking=True, reverse=True, topk=5)
    with pytest.raises(ValueError):
        LatentModel(df, dynamic=True, decomp=True, ranking=True, topk=0)
    LatentModel(df, dynamic=True, decomp=True, ranking=False)      # no switch, no ranking: still fine


# ------------------------------------------------------------------ 5. leakage of `classified`
def _scramble_from(df: pd.DataFrame, r0: int, seed: int) -> pd.DataFrame:
    """Re-order the rows of races >= r0 at random and flip `classified` at random (a different 'result'); the
    pre-race columns stay attached to their driver."""
    rng = np.random.default_rng(seed)
    parts = [df[df["race_id"] < r0]]
    for r, g in df[df["race_id"] >= r0].groupby("race_id"):
        g = g.iloc[rng.permutation(len(g))].copy()
        g["classified"] = rng.random(len(g)) > 0.5
        g["pos"] = np.arange(1, len(g) + 1)
        g["y"] = (g["pos"] - 1) / max(len(g) - 1, 1)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _predict(df, r, **kw):
    m = _model(df, **kw)
    first = int(np.flatnonzero(df["race_id"].to_numpy() == r)[0])
    test = df[df["race_id"] == r]
    s = m.fit_predict(df.iloc[:first], test)
    return pd.Series(s, index=test["driverId"].to_numpy()).sort_index()


@pytest.mark.parametrize("name", list(SWITCHES))
@pytest.mark.parametrize("r", [9, 15])
def test_classified_of_predicted_and_later_races_is_never_read(name, r):
    df = _frame(3)
    a = _predict(df, r, **SWITCHES[name])
    b = _predict(_scramble_from(df, r, 11), r, **SWITCHES[name])
    assert (a.index == b.index).all() and (a.to_numpy() == b.to_numpy()).all()
    # sanity: scrambling the PAST does change the finishers-only prediction (the flag is used for training races)
    if name.startswith("finishers"):
        past = df.copy()
        past.loc[past["race_id"] == 2, "classified"] = ~past.loc[past["race_id"] == 2, "classified"]
        c = _predict(past, r, **SWITCHES[name])
        assert not np.array_equal(a.to_numpy(), c.to_numpy())


def test_padded_matrix_only_holds_training_rows():
    df = _frame(3)
    m = _model(df, finishers_only=True)
    first = int(np.flatnonzero(df["race_id"].to_numpy() == 15)[0])
    w, pad = m.fit_inputs(first, 15.0)
    idx, mask, stage = pad
    assert idx[mask].max() < first and idx.shape[0] == 15


@needs_real
@pytest.mark.parametrize("name", ["finishers", "reverse"])
def test_classified_leakage_real_data(name):
    e = pd.read_parquet(REAL)
    r = int(e.loc[e["season"] == 2022, "race_id"].min()) + 5
    keep = e[e["race_id"] <= r + 3].reset_index(drop=True)
    kw = dict(dynamic=True, decomp=True, ranking=True, halflife=50, lam=32.0)

    def pred(d):
        m = LatentModel(d, **kw, **SWITCHES[name])
        first = int(np.flatnonzero(d["race_id"].to_numpy() == r)[0])
        t = d[d["race_id"] == r]
        return pd.Series(m.fit_predict(d.iloc[:first], t), index=t["driverId"].to_numpy()).sort_index()

    a = pred(keep)
    b = pred(_scramble_from(keep, r, 5))
    assert (a.to_numpy() == b.to_numpy()).all()


# ------------------------------------------------------------------ 6. recovery and sign
def _simulate_reverse(seed=0, n_races=300, n_drv=10):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.8, n_drv)
    rows = []
    for r in range(n_races):
        left = list(range(n_drv))
        last_first = []
        while left:        # reverse PL: the LAST place is drawn first, probability ~ exp(-u)
            p = np.exp(-a[left])
            p /= p.sum()
            pick = int(rng.choice(len(left), p=p))
            last_first.append(left.pop(pick))
        order = last_first[::-1]
        for pos, d in enumerate(order, 1):
            rows.append(dict(race_id=r, season=2020, driverId=f"d{d:02d}", lineage="L", circuit_type="x", pos=pos,
                             n=n_drv, y=(pos - 1) / (n_drv - 1), classified=True, grid_norm=.5, quali_norm=.5,
                             q1_gap=0., pit_start=0))
    return pd.DataFrame(rows), a


def test_reverse_recovers_strengths_and_has_the_right_sign():
    df, a = _simulate_reverse()
    kw = dict(dynamic=False, decomp=False, ranking=True, base_x=False, lam_a=0.5, maxiter=300)
    m = LatentModel(df, reverse=True, **kw)
    m.fit(len(df), t_ref=300.0)
    a_hat = m.theta[[m.drivers[f"d{i:02d}"] for i in range(10)]]
    assert np.corrcoef(a_hat, a)[0, 1] > 0.97
    assert np.abs(a_hat - a_hat.mean() - (a - a.mean())).max() < 0.35        # loose, finite sample, ridge shrinkage
    # likelihood is higher at the truth than at the negated truth
    th = np.zeros(m.theta.size)
    th[[m.drivers[f"d{i:02d}"] for i in range(10)]] = a
    w = np.ones(len(df))
    pad = m._pad(300)
    assert m._objective(th, len(df), w, pad)[0] < m._objective(-th, len(df), w, pad)[0]
    # prediction: stronger driver = better predicted place (score descending); take a fresh 'race' of all drivers
    test = df[df["race_id"] == 299]
    d2 = pd.concat([df[df["race_id"] < 299], test]).reset_index(drop=True)
    m2 = LatentModel(d2, reverse=True, **kw)
    s = m2.fit_predict(d2[d2["race_id"] < 299], d2[d2["race_id"] == 299])
    names = d2.loc[d2["race_id"] == 299, "driverId"].map(lambda x: int(x[1:])).to_numpy()
    rho = pd.Series(s).corr(pd.Series(a[names]), method="spearman")
    assert rho > 0.9
    best, worst = names[np.argmax(s)], names[np.argmin(s)]
    assert a[best] > a[worst]
    pred_rank = pd.Series(s).rank(ascending=False).to_numpy()
    assert pred_rank[np.argmax(a[names])] <= 2


def test_forward_and_reverse_fit_agree_on_sign_for_forward_data():
    """Data from the ordinary PL: the reverse fit must still put strong drivers on top (same sign convention)."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.8, 10)
    rows = []
    for r in range(300):
        order = np.argsort(-(a + rng.gumbel(size=10)))
        for pos, d in enumerate(order, 1):
            rows.append(dict(race_id=r, season=2020, driverId=f"d{d:02d}", lineage="L", circuit_type="x", pos=pos, n=10,
                             y=(pos - 1) / 9, classified=True, grid_norm=.5, quali_norm=.5, q1_gap=0., pit_start=0))
    df = pd.DataFrame(rows)
    for kw in (dict(reverse=True), dict(), dict(topk=5), dict(finishers_only=True)):
        m = LatentModel(df, dynamic=False, decomp=False, ranking=True, base_x=False, lam_a=0.5, maxiter=300, **kw)
        m.fit(len(df), 300.0)
        a_hat = m.theta[[m.drivers[f"d{i:02d}"] for i in range(10)]]
        assert np.corrcoef(a_hat, a)[0, 1] > 0.9, kw


# ------------------------------------------------------------------ 7. stored results
def _run_dir(suffix: str) -> Path:
    c = sorted(RES.glob(f"*_{suffix}"))
    assert len(c) == 1, (suffix, c)
    return c[0]


needs_results = pytest.mark.skipif(not RES.exists(), reason="M11 results missing")


def _plan1_dev_file(label: str = "latent_111") -> pd.DataFrame:
    lb = pd.read_csv(ROOT / "results" / "leaderboard.csv")
    row = lb[(lb["label"] == label) & (lb["split"] == "dev")].iloc[-1]
    return pd.read_csv(ROOT / row["run_dir"] / "per_race_metrics.csv")


@needs_results
def test_reference_run_equals_stored_plan1_latent_111_dev_exactly():
    new = pd.read_csv(_run_dir("m11_ref_latent_111_dev") / "per_race_metrics.csv")
    old = _plan1_dev_file()
    # the plan-1 file predates the columns top5, exact, within1, within2 (added in plan 2): compare all shared ones
    shared = [c for c in old.columns if c in new.columns]
    assert shared == list(old.columns), "a plan-1 column is missing from the M11 reference"
    assert len(new) == len(old) == 66
    pd.testing.assert_frame_equal(new[shared], old[shared], check_exact=True)


@needs_results
@pytest.mark.parametrize("suffix", ["m11_ref_latent_111_dev", *[v[0] for v in VARIANT_RUNS.values()]])
def test_per_race_metrics_equal_rescore_of_predictions(suffix):
    d = _run_dir(suffix)
    pr = pd.read_parquet(d / "predictions.parquet")
    stored = pd.read_csv(d / "per_race_metrics.csv").set_index("race_id")
    assert sorted(pr["race_id"].unique()) == sorted(stored.index) and len(stored) == 66 and len(pr) == 1311
    assert not pr.duplicated(["race_id", "driverId"]).any() and pr["score"].notna().all()
    for rid, g in pr.groupby("race_id"):
        m = race_metrics(g["pos"].to_numpy(), g["score"].to_numpy(), g["classified"].to_numpy())
        assert len(g) == stored.loc[rid, "n"]
        for k, v in m.items():
            assert stored.loc[rid, k] == pytest.approx(v, abs=1e-12, nan_ok=True), (suffix, rid, k)


@needs_results
@pytest.mark.parametrize("comp", list(VARIANT_RUNS))
def test_run_configs_carry_the_declared_switches(comp):
    suffix, sw = VARIANT_RUNS[comp]
    cfg = load_yaml(_run_dir(suffix) / "config.yaml")
    ref = load_yaml(_run_dir("m11_ref_latent_111_dev") / "config.yaml")
    for k, v in sw.items():
        assert cfg[k] == v
    other = {"finishers_only", "topk", "reverse"} - set(sw)
    assert not any(cfg.get(k) for k in other)
    for k in ("kind", "cell", "lam", "halflife"):
        assert cfg[k] == ref[k]
    assert not any(ref.get(k) for k in ("finishers_only", "topk", "reverse"))
    frozen = {k: v for k, v in load_yaml(ROOT / "configs" / "frozen.yaml")["M03"]["111"].items() if k not in ("kind", "cell", "label")}
    assert (ref["lam"], ref["halflife"]) == (frozen["lam"], frozen["halflife"])


def _outcome_free(preds: pd.DataFrame) -> pd.DataFrame:
    out = preds.copy()
    sc = np.empty(len(out))
    for _, ix in out.groupby("race_id").indices.items():
        sc[ix] = DEC.break_ties(out["score"].to_numpy(float)[ix], out["driverId"].to_numpy()[ix])
    out["score"] = sc
    return out


def _tables():
    return {k: A.race_table(_outcome_free(pd.read_parquet(_run_dir(s) / "predictions.parquet")))
            for k, s in [("ref", "m11_ref_latent_111_dev"), *[(c, v[0]) for c, v in VARIANT_RUNS.items()]]}


@needs_results
def test_attribution_m11_rows_reproduce_from_run_folders():
    att = pd.read_csv(ROOT / "results" / "plan3" / "attribution.csv")
    tabs = _tables()
    for comp, (suffix, _) in VARIANT_RUNS.items():
        row = att[(att["component"] == comp) & (att["reference"] == "latent_111")]
        assert len(row) == 1, comp
        row = row.iloc[0]
        assert row["run_id"] == _run_dir(suffix).name and row["n_races"] == 66
        g = A.paired_gain(tabs["ref"], tabs[comp], n_boot=10000, seed=0)
        seen = 0
        for _, r in g.iterrows():
            for key, val in (("gain", r["gain"]), ("lo", r["lo"]), ("hi", r["hi"])):
                stored = row[f"{key}_{r['metric']}"]
                assert stored == pytest.approx(val, abs=1e-12), (comp, r["metric"], key)
                seen += 1
        assert seen >= 3 * 10
        # metrics that were not recomputable (rps) must be empty in the stored row, not stale numbers
        for m in set(A.METRICS) - set(g["metric"]):
            assert pd.isna(row[f"gain_{m}"]), (comp, m)
        # per-metric verdict string recomputed from the paired gains: 95% CI excludes 0 -> gain / worse, else none
        gm = g.set_index("metric")
        word = lambda m: "gain" if gm.loc[m, "lo"] > 0 else ("worse" if gm.loc[m, "hi"] < 0 else "none")
        recomputed = "; ".join(f"{lab}: {word(m)}" for m, lab in (("spearman", "rho"), ("mae", "MAE"), ("exact", "exact")))
        assert row["verdict"] == recomputed + "; dev only", (comp, row["verdict"], recomputed)
        assert json.loads((RES / "verdict_strings_dev.json").read_text(encoding="utf-8"))[comp] == recomputed
        pm = pd.read_csv(RES / "verdicts_per_metric_dev.csv")
        pm = pm[pm["component"] == comp].set_index("metric")
        assert list(pm.index) == ["spearman", "mae", "exact"]
        for m in pm.index:
            assert pm.loc[m, "verdict"] == word(m)
            for key in ("gain", "lo", "hi"):
                assert pm.loc[m, key] == pytest.approx(gm.loc[m, key], abs=1e-12)


@needs_results
def test_attribution_m17_rows_unchanged_against_head():
    cur = pd.read_csv(ROOT / "results" / "plan3" / "attribution.csv", dtype=object)
    old_txt = subprocess.check_output(["git", "show", "c3b4fbc:results/plan3/attribution.csv"], cwd=ROOT, text=True)
    from io import StringIO
    old = pd.read_csv(StringIO(old_txt), dtype=object)
    key = ["component", "reference"]
    cur_m17 = cur[cur["component"].str.startswith("M17")].set_index(key).sort_index()
    old_m17 = old[old["component"].str.startswith("M17")].set_index(key).sort_index()
    assert list(cur_m17.index) == list(old_m17.index) and len(old_m17) >= 6
    assert list(cur.columns) == list(old.columns)
    for c in old_m17.columns:
        a, b = cur_m17[c], old_m17[c]
        na, nb = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        both_num = na.notna() & nb.notna()
        assert (na.isna() == nb.isna()).all() or (a.isna() == b.isna()).all(), c
        assert ((na[both_num] - nb[both_num]).abs() <= 1e-12).all(), c
        txt = ~(na.notna() | nb.notna())
        assert (a[txt].fillna("") == b[txt].fillna("")).all(), c
    # the component set is exactly HEAD's (the M17 rows) plus the explicitly listed new components, nothing else
    assert set(old["component"]) == {"M17_D2", "M17_grid", "M17_D0u", "M17_D0"}
    expected_new = {"M11_finishers", "M11_topk10", "M11_reverse", "M05_rank_avg", "M05_gbm_monotone", "M12_two_stage",
                    "M11_finishers_x_M07"}
    added = set(cur["component"]) - set(old["component"])
    assert expected_new <= added
    # later work packages append their own components; none may reuse an M17 or M11/M05/M12 name
    assert all(not c.startswith(("M17", "M11", "M05", "M12")) for c in added - expected_new), added - expected_new
    assert set(old["component"]) <= set(cur["component"])
    assert not cur.duplicated(key).any()
    # every old HEAD row (all of them, not only M17 by prefix) is still there unchanged
    assert len(cur[cur["component"].isin(old["component"])]) == len(old)


@needs_results
def test_no_tied_scores_and_no_dependence_on_row_order():
    rng = np.random.default_rng(0)
    for suffix in ["m11_ref_latent_111_dev", *[v[0] for v in VARIANT_RUNS.values()]]:
        pr = pd.read_parquet(_run_dir(suffix) / "predictions.parquet")
        assert all(g["score"].nunique() == len(g) for _, g in pr.groupby("race_id")), suffix
        base = A.race_table(_outcome_free(pr)).drop(columns=[])
        shuf = pr.iloc[rng.permutation(len(pr))].sort_values("race_id", kind="stable")
        alt = A.race_table(_outcome_free(shuf))
        pd.testing.assert_frame_equal(base.reset_index(drop=True), alt.reset_index(drop=True), check_exact=False, rtol=0, atol=1e-12)


# ------------------------------------------------------------------ data claims of the method card
@needs_real
@needs_results
def test_card_data_claims():
    e = pd.read_parquet(REAL)
    assert e["classified"].notna().all()
    last_dev = int(e.loc[e["season"] <= 2023, "race_id"].max())
    first_dev = int(e.loc[e["season"] >= 2021, "race_id"].min())
    sh = pd.read_csv(RES / "descriptive_stage_shares.csv")
    dev = e[(e["race_id"] >= first_dev) & (e["race_id"] <= last_dev)]
    up = e[e["race_id"] <= last_dev]
    assert (len(dev), int((~dev["classified"]).sum()), dev["race_id"].nunique()) == (1311, 157, 66)
    assert (len(up), int((~up["classified"]).sum()), up["race_id"].nunique()) == (4119, 641, 204)
    assert sh["entries"].tolist() == [1311, 4119] and sh["retirees"].tolist() == [157, 641]
    # "1 of the 204 races has a retiree ahead of a classified entry"
    bad = 0
    for _, g in up.groupby("race_id"):
        c = g["classified"].to_numpy()
        bad += bool((~c[:-1] & c[1:]).any())
    assert bad == 1
    # top-10 share: stages removed = entries beyond place 10
    assert sh.loc[0, "topk_removed_share"] == pytest.approx(1 - dev.groupby("race_id").size().clip(upper=10).sum() / len(dev))
    # rows per race and per season: row count = entries; no duplicate driver within a race; ordering by pos
    assert not dev.duplicated(["race_id", "driverId"]).any()
    for _, g in up.groupby("race_id"):
        assert g["pos"].tolist() == list(range(1, len(g) + 1))


# ------------------------------------------------------------------ 8. re-run from scratch
@needs_real
@needs_results
@pytest.mark.slow
def test_reverse_walk_forward_rerun_reproduces_stored_predictions():
    e = pd.read_parquet(REAL)
    d = _run_dir("m11_latent_111_reverse_dev")
    cfg = {k: v for k, v in load_yaml(d / "config.yaml").items()
           if k in ("kind", "cell", "lam", "halflife", "reverse")}
    assert cfg["reverse"] is True
    model = build_model(cfg, e)
    p, m = walk_forward(e, model, "dev", refit_every=1)
    old = pd.read_parquet(d / "predictions.parquet")
    mm = p.merge(old, on=["race_id", "driverId"], suffixes=("_n", "_o"))
    assert len(mm) == len(old) == len(p) == 1311
    assert np.abs(mm["score_n"] - mm["score_o"]).max() <= 1e-9
    for c in ("p_win", "p_podium", "p_points"):
        assert np.abs(mm[f"{c}_n"] - mm[f"{c}_o"]).max() <= 1e-9
    stored = pd.read_csv(d / "per_race_metrics.csv")
    pd.testing.assert_frame_equal(m.reset_index(drop=True)[stored.columns], stored, check_exact=False, rtol=1e-9, atol=1e-9)


# ------------------------------------------------------------------ 9. revision: multiplicity, grid comparison, tuning, mechanism, gradient
def _holm(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm step-down written independently: sort ascending, compare p_(i) with alpha/(m-i), stop at the first failure."""
    m = len(p)
    rej = np.zeros(m, bool)
    for i, j in enumerate(np.argsort(p, kind="stable")):
        if p[j] <= alpha / (m - i):
            rej[j] = True
        else:
            break
    return rej


@needs_results
def test_multiplicity_table_recomputed_from_the_intervals():
    from scipy.stats import norm
    mult = pd.read_csv(RES / "multiplicity_dev.csv")
    full = pd.read_csv(RES / "paired_gains_dev.csv")
    assert len(mult) == 36 and len(full) == 36
    assert mult.groupby("component").size().to_dict() == {"M11_finishers": 12, "M11_reverse": 12, "M11_topk10": 12}
    assert not mult.duplicated(["component", "metric"]).any() and mult[["gain", "lo", "hi"]].notna().all().all()
    # the 36 intervals are the stored paired gains, unchanged
    mm = mult.merge(full, on=["component", "metric"], suffixes=("", "_f"))
    assert len(mm) == 36
    for c in ("gain", "lo", "hi"):
        assert np.abs(mm[c] - mm[f"{c}_f"]).max() <= 1e-12
    # p from the interval: SE = width / (2 * 1.96), z = gain / SE, two-sided normal
    se = (mult["hi"] - mult["lo"]) / (2 * norm.ppf(0.975))
    p = 2 * norm.sf(np.abs(mult["gain"] / se))
    assert np.allclose(mult["se_from_ci"], se, rtol=1e-6)
    assert np.allclose(mult["p_normal"], p, rtol=1e-5, atol=1e-15)
    assert ((mult["lo"] > 0) | (mult["hi"] < 0)).tolist() == mult["ci_excludes_0"].tolist()
    # Holm: rank, threshold alpha/(m-rank+1), rejections
    rank = pd.Series(p).rank(method="first").astype(int)
    assert mult["holm_rank"].tolist() == rank.tolist()
    assert np.allclose(mult["holm_threshold"], 0.05 / (36 - rank + 1), rtol=1e-12)
    assert mult["holm_reject"].tolist() == _holm(np.asarray(p)).tolist()
    assert not (mult["holm_reject"] & ~mult["ci_excludes_0"]).any()
    # the card: 10 of 36 exclude 0, 4 survive; the survivors are named
    assert int(mult["ci_excludes_0"].sum()) == 10 and int(mult["holm_reject"].sum()) == 4
    surv = set(map(tuple, mult.loc[mult["holm_reject"], ["component", "metric"]].to_numpy()))
    assert surv == {("M11_reverse", "exact_1_3"), ("M11_reverse", "logloss_points"), ("M11_reverse", "mae_fin"),
                    ("M11_finishers", "mae_fin")}
    # step-down property: every rejected p is smaller than every non-rejected p
    assert mult.loc[mult["holm_reject"], "p_normal"].max() < mult.loc[~mult["holm_reject"], "p_normal"].min()


def _grid_scores(pr: pd.DataFrame) -> np.ndarray:
    """Starting-grid order, written independently: pit-lane starts (grid 0) at the back, ties by driverId (not by row
    order, which is the finishing order). Utility = -predicted place."""
    sc = np.empty(len(pr))
    for _, ix in pr.groupby("race_id").indices.items():
        g = pr["grid"].to_numpy(float)[ix]
        eff = np.where(g <= 0, g.max() + 1, g)
        order = np.lexsort((pr["driverId"].to_numpy()[ix], eff))
        place = np.empty(len(ix))
        place[order] = np.arange(1, len(ix) + 1)
        sc[ix] = -place
    return sc


def _grid_rank_table(pr: pd.DataFrame) -> pd.DataFrame:
    return A.race_table(pr.assign(score=_grid_scores(pr)))


@needs_results
def test_vs_grid_table_reproduces_from_predictions():
    vg = pd.read_csv(RES / "vs_grid_dev.csv")
    runs = {"latent_111": "m11_ref_latent_111_dev", **{c: v[0] for c, v in VARIANT_RUNS.items()}}
    metr = ["exact", "exact_fin", "exact_1_3", "exact_4_10", "exact_11p", "spearman", "mae"]
    assert len(vg) == 4 * 4 * 7 and not vg.duplicated(["run", "scope", "metric"]).any()
    for run, suf in runs.items():
        pr = pd.read_parquet(_run_dir(suf) / "predictions.parquet")
        gt = _grid_rank_table(pr)
        # grid order is independent of row order: shuffle the rows, same table
        sh = pr.sample(frac=1.0, random_state=3).sort_values("race_id", kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(gt.reset_index(drop=True), _grid_rank_table(sh).reset_index(drop=True),
                                      check_exact=False, atol=1e-12)
        t = A.race_table(_outcome_free(pr))
        for scope in ["all", "2021", "2022", "2023"]:
            a = gt if scope == "all" else gt[gt["season"] == int(scope)]
            b = t if scope == "all" else t[t["season"] == int(scope)]
            g = A.paired_gain(a, b, metr, n_boot=10000, seed=0).set_index("metric")
            for m in metr:
                row = vg[(vg["run"] == run) & (vg["scope"] == scope) & (vg["metric"] == m)]
                assert len(row) == 1, (run, scope, m)
                row = row.iloc[0]
                k = 100.0 if m.startswith("exact") else 1.0
                for key, col in (("cand_mean", "run_mean"), ("ref_mean", "grid_mean"), ("gain", "gain_vs_grid"), ("lo", "lo"), ("hi", "hi")):
                    assert row[col] == pytest.approx(k * g.loc[m, key], abs=1e-9), (run, scope, m, col)
                assert row["n_races"] == g.loc[m, "n"] == (66 if scope == "all" else 22)
                word = "gain" if row["lo"] > 0 else ("worse" if row["hi"] < 0 else "none")
                assert row["verdict"] == word
        # exact rate of grid order without the library: share of entries whose grid place equals the finishing place
        sc = _grid_scores(pr)
        per_race = [float(np.mean(-sc[ix] == pr["pos"].to_numpy()[ix])) for _, ix in pr.groupby("race_id").indices.items()]
        stored = vg[(vg["run"] == run) & (vg["scope"] == "all") & (vg["metric"] == "exact")]["grid_mean"].iloc[0]
        assert stored == pytest.approx(100 * np.mean(per_race), abs=1e-9)


@needs_results
def test_vs_grid_matches_the_m17_grid_free_run_folder():
    """Cross-check against the stored M17 'grid_free' run (a different code path, same definition)."""
    d = sorted((ROOT / "methodologies" / "M17_decoding" / "results").glob("*_m17_grid_free_dev"))
    if len(d) != 1:
        pytest.skip("M17 grid_free run folder not found")
    st = pd.read_csv(d[0] / "per_race_metrics.csv").set_index("race_id")
    pr = pd.read_parquet(_run_dir("m11_ref_latent_111_dev") / "predictions.parquet")
    mine = _grid_rank_table(pr).set_index("race_id")
    for c in ("spearman", "mae", "exact"):
        assert np.abs(mine[c] - st.loc[mine.index, c]).max() <= 1e-12, c


@needs_results
def test_fitted_components_has_converged_flags_and_they_are_true():
    fc = pd.read_csv(RES / "fitted_components.csv")
    flag = [c for c in fc.columns if "converge" in c.lower()]
    assert flag, f"fitted_components.csv has no convergence flag column (columns: {list(fc.columns)[:8]}...); the card claims 'converged'"
    assert fc[flag[0]].astype(bool).all()
    assert len(fc) == 8 and set(fc["run"]) == {"latent_111", "M11_finishers", "M11_topk10", "M11_reverse"}


@needs_real
@needs_results
@pytest.mark.slow
def test_card_convergence_claim_projected_gradient_below_1e5():
    """The revised card says the first cold fits reach max |grad| < 1e-5 except finishers-only before 2022 (4.5e-5),
    which was refitted to 9e-9. Refit cold before the first 2022 race at the frozen setting with the default
    LatentModel.fit and check the gradient (no bounds, so projected = plain) against fitted_components.csv."""
    e = pd.read_parquet(REAL)
    frozen = {k: v for k, v in load_yaml(ROOT / "configs" / "frozen.yaml")["M03"]["111"].items() if k not in ("kind", "cell", "label")}
    rid = int(e.loc[e["season"] == 2022, "race_id"].min())
    frozen = {k: v for k, v in frozen.items() if k not in ("kind", "cell", "label")}
    bad = {}
    for sw in ({}, dict(finishers_only=True), dict(topk=10), dict(reverse=True)):
        m = LatentModel(e, dynamic=True, decomp=True, ranking=True, maxiter=3000, **frozen, **sw)
        n = int(m.race_rows[rid].min())
        m.fit(n, t_ref=float(rid))
        w, pad = m.fit_inputs(n, float(rid))
        g = m._objective(m.theta, n, w, pad)[1]
        bad[str(sw)] = float(np.abs(g).max())
    fc = pd.read_csv(RES / "fitted_components.csv")
    fc = fc[fc["fit_before_race_id"] == rid].set_index("run")
    names = dict(zip(map(str, ({}, dict(finishers_only=True), dict(topk=10), dict(reverse=True))),
                     ("latent_111", "M11_finishers", "M11_topk10", "M11_reverse")))
    for k, v in bad.items():
        row = fc.loc[names[k]]
        assert v == pytest.approx(row["first_fit_max_abs_grad"], rel=1e-3), (k, v)
        assert bool(row["first_fit_converged"]) == (v < 1e-5), (k, v)
        assert row["max_abs_grad"] < 1e-5 and bool(row["converged"])
    assert bad[str(dict(finishers_only=True))] > 1e-5  # the one fit the card says needed a refit


@needs_results
def test_tuned_runs_chose_a_plan1_grid_configuration_with_at_most_30_configs():
    from pitlane import pipeline as P
    grid = pd.read_csv(RES / "tuning_grid_dev.csv")
    ids = json.loads((RES / "tuned_run_ids.json").read_text(encoding="utf-8"))
    assert set(ids) == {"latent_111", "M11_finishers", "M11_topk10", "M11_reverse"}
    plan1 = set(product(P.LAM_GRID, P.HL_GRID))
    assert (P.LAM_GRID, P.HL_GRID) == ([0.5, 2.0, 8.0, 32.0], [8, 20, 50])       # the plan-1 grid itself, pinned
    for comp, rid in ids.items():
        g = grid[grid["component"] == comp]
        assert len(g) <= 30 and len(g) == len(plan1) == 12
        assert set(zip(g["lam"].astype(float), g["halflife"].astype(int))) == plan1
        assert (g["n_races"] == 66).all() and not g.duplicated(["lam", "halflife"]).any()
        assert g["chosen_by_rho"].sum() == 1
        ch = g[g["chosen_by_rho"]].iloc[0]
        assert ch["spearman"] == g["spearman"].max()                 # selection criterion: mean rho on dev
        cfg = load_yaml(RES / rid / "config.yaml")
        assert (float(cfg["lam"]), int(cfg["halflife"])) == (float(ch["lam"]), int(ch["halflife"]))
        assert (float(cfg["lam"]), int(cfg["halflife"])) in plan1
        assert cfg["split"] == "dev"
        assert "dev" in str(cfg["tuned_on"]) and "test" not in str(cfg["tuned_on"]).lower()
        pm = pd.read_csv(RES / rid / "per_race_metrics.csv")
        assert len(pm) == 66 and pm["spearman"].mean() == pytest.approx(ch["spearman"], abs=1e-12)
        assert pm["mae"].mean() == pytest.approx(ch["mae"], abs=1e-12) and pm["exact"].mean() == pytest.approx(ch["exact"], abs=1e-12)
        pr = pd.read_parquet(RES / rid / "predictions.parquet")
        assert pr["season"].between(2021, 2023).all()
        assert sorted(pr["race_id"].unique()) == sorted(pm["race_id"])
    # the frozen-setting row of the reference reproduces the plan-1 run (same footing)
    ref = grid[(grid["component"] == "latent_111") & grid["is_frozen_setting"]].iloc[0]
    assert (ref["lam"], ref["halflife"]) == (32.0, 50)
    assert ref["spearman"] == pytest.approx(_plan1_dev_file()["spearman"].mean(), abs=1e-12)


@needs_results
def test_gradient_check_csv_max_relative_error_below_1e5():
    gc = pd.read_csv(RES / "gradient_check.csv")
    assert {"off", "finishers", "top10", "reverse"} <= set(gc["variant"])
    assert (gc["coords_checked"] >= 40).all() and (gc["n_params"] > 100).all()
    assert gc["max_rel_error"].notna().all() and (gc["max_rel_error"] < 1e-5).all(), gc


@needs_real
@needs_results
def test_gradient_check_csv_spot_recheck_on_real_window():
    """Recompute the finishers_only row's setting (training window and parameter count) and its gradient error."""
    gc = pd.read_csv(RES / "gradient_check.csv").set_index("variant")
    e = pd.read_parquet(REAL)
    frozen = {k: v for k, v in load_yaml(ROOT / "configs" / "frozen.yaml")["M03"]["111"].items() if k not in ("kind", "cell", "label")}
    rid = int(e.loc[e["season"] == 2021, "race_id"].min())
    m = LatentModel(e, dynamic=True, decomp=True, ranking=True, finishers_only=True, **frozen)
    n = int(m.race_rows[rid].min())
    assert n == gc.loc["finishers", "n_train_rows"] and m.theta.size == gc.loc["finishers", "n_params"]
    w, pad = m.fit_inputs(n, float(rid))
    th = np.random.default_rng(0).normal(0, 0.3, m.theta.size)
    _, g = m._objective(th, n, w, pad)
    errs = []
    for j in np.random.default_rng(1).choice(th.size, 40, replace=False):
        ee = np.zeros_like(th)
        ee[j] = 1e-6
        fd = (m._objective(th + ee, n, w, pad)[0] - m._objective(th - ee, n, w, pad)[0]) / 2e-6
        errs.append(abs(fd - g[j]) / max(abs(g[j]), 1e-3 * np.abs(g).max()))
    assert max(errs) < 1e-5, max(errs)
    assert max(errs) == pytest.approx(gc.loc["finishers", "max_rel_error"], rel=1e-3)
