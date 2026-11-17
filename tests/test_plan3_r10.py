"""R10 / M05 ensemble and M12 two-stage: independent checks.

Monotone constraints, rank averaging, truncation/leakage of the qualifying-history features and of the two-stage
Friday forecast, and the stored result files of methodologies/M05_ensemble and methodologies/M12_two_stage.
"""
from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitlane.config import PROCESSED, ROOT, load_yaml
from pitlane.evaluation.harness import walk_forward
from pitlane.evaluation.metrics import pred_rank, race_metrics
from pitlane.features.build import gbm_feature_cols
from pitlane.models.ensemble import MONOTONE_UP, MonotoneGBM, RankAverageModel, rank_average_score
from pitlane.models.gbm import GBMModel
from pitlane.models.registry import build_model, cell_flags
from pitlane.models.two_stage import (HALFLIVES_Q, STAGE1_HIST, STAGE1_PRACTICE, QualiStage1, TwoStageH2,
                                      add_quali_history, rank_within)
from pitlane.plan3 import attribution as A
from pitlane.plan3 import runutil as U

M05 = ROOT / "methodologies" / "M05_ensemble" / "results"
M12 = ROOT / "methodologies" / "M12_two_stage" / "results"
REAL = PROCESSED / "entries_feat.parquet"
REAL2 = PROCESSED / "entries2.parquet"
needs_real = pytest.mark.skipif(not (REAL.exists() and REAL2.exists()), reason="processed data missing (regenerable)")
needs_m05 = pytest.mark.skipif(not M05.exists(), reason="M05 results missing")
needs_m12 = pytest.mark.skipif(not M12.exists(), reason="M12 results missing")
LB3 = ROOT / "results" / "plan3" / "leaderboard.csv"
ATT = ROOT / "results" / "plan3" / "attribution.csv"


# ============================================================== rank averaging, hand tables
def test_rank_average_hand_table_with_a_tie_broken_by_key():
    keys = np.array(["a", "b", "c", "d"])
    m1 = np.array([4.0, 3.0, 2.0, 1.0])        # ranks 1 2 3 4
    m2 = np.array([1.0, 2.0, 4.0, 3.0])        # ranks 4 3 1 2
    # mean ranks: a 2.5, b 2.5, c 2.0, d 3.0 -> c first, a/b tie (key a first), d last
    u = rank_average_score([m1, m2], keys)
    assert pred_rank(u).tolist() == [2, 3, 1, 4]
    # no key: the utility itself is exactly minus the mean rank
    assert rank_average_score([m1, m2], None).tolist() == [-2.5, -2.5, -2.0, -3.0]
    assert rank_average_score([m1], None).tolist() == [-1.0, -2.0, -3.0, -4.0]


def test_rank_average_member_tie_is_broken_by_key_not_by_row_order():
    keys = np.array(["x", "y", "z"])
    s = np.array([1.0, 1.0, 2.0])
    assert pred_rank(rank_average_score([s], keys)).tolist() == [2, 3, 1]          # z best, then x before y
    perm = [1, 0, 2]                                                                 # swap rows x and y
    assert pred_rank(rank_average_score([s[perm]], keys[perm])).tolist() == [3, 2, 1]   # y row now first but still ranks 3rd


def test_rank_average_equals_mean_of_ranks_random_and_row_order_free():
    rng = np.random.default_rng(0)
    for n in (5, 12, 20):
        keys = np.array([f"d{i:02d}" for i in range(n)])
        sc = [rng.normal(size=n).round(1) for _ in range(3)]          # rounding creates many ties
        u = rank_average_score(sc, keys)
        ranks = []
        for s in sc:                                                    # hand ranking: higher score first, ties by smaller key
            order = sorted(range(n), key=lambda i: (-s[i], keys[i]))
            r = np.empty(n)
            r[order] = np.arange(1, n + 1)
            ranks.append(r)
        mean = np.mean(ranks, axis=0)
        expected_order = sorted(range(n), key=lambda i: (mean[i], keys[i]))
        got = pred_rank(u)
        assert [int(got[i]) for i in expected_order] == list(range(1, n + 1))
        for _ in range(5):                                              # any row order gives the same rank per key
            p = rng.permutation(n)
            u2 = rank_average_score([s[p] for s in sc], keys[p])
            assert pred_rank(u2)[np.argsort(p)].tolist() == got.tolist()


class _Col:
    """Stub member: its score for a row is a given column of the test frame (ignores training data, records it)."""

    def __init__(self, col: str):
        self.col, self.seen = col, []

    def fit_predict(self, train, test):
        self.seen.append(len(train))
        return test[self.col].to_numpy(float)


def _two_races() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for r in (1, 2):
        for i in range(8):
            rows.append(dict(race_id=r, driverId=f"d{i}", s1=float(rng.integers(0, 4)), s2=float(rng.integers(0, 4))))
    return pd.DataFrame(rows)


def test_rank_average_model_is_mean_of_member_ranks_per_race():
    df = _two_races()
    mem = [_Col("s1"), _Col("s2")]
    out = RankAverageModel(mem).fit_predict(df.iloc[:0], df)
    assert mem[0].seen == mem[1].seen == [0]                          # both fitted on the same training rows
    for r, g in df.groupby("race_id"):
        ix = g.index.to_numpy()
        want = rank_average_score([g["s1"].to_numpy(), g["s2"].to_numpy()], g["driverId"].to_numpy())
        assert np.array_equal(out[ix], want)
        # hand check of the mean rank order on this race
        rk = []
        for c in ("s1", "s2"):
            order = sorted(range(len(g)), key=lambda i: (-g[c].iloc[i], g["driverId"].iloc[i]))
            x = np.empty(len(g))
            x[order] = np.arange(1, len(g) + 1)
            rk.append(x)
        mean = np.mean(rk, axis=0)
        final = pred_rank(out[ix])
        for i in range(len(g)):
            for j in range(len(g)):
                if mean[i] < mean[j]:
                    assert final[i] < final[j]


def test_rank_average_model_has_no_dependence_on_row_order():
    df = _two_races()
    base = pd.Series(RankAverageModel([_Col("s1"), _Col("s2")]).fit_predict(df.iloc[:0], df), index=list(zip(df["race_id"], df["driverId"])))
    rng = np.random.default_rng(7)
    for _ in range(5):
        sh = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
        o = pd.Series(RankAverageModel([_Col("s1"), _Col("s2")]).fit_predict(sh.iloc[:0], sh), index=list(zip(sh["race_id"], sh["driverId"])))
        o = o.reindex(base.index)
        assert np.array_equal(o.to_numpy(), base.to_numpy())


# ============================================================== MonotoneGBM
def _gbm_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = gbm_feature_cols(False, False)
    X = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    X["grid_norm"] = rng.random(n)
    X["quali_norm"] = rng.random(n)
    # adversarial target: decreasing in grid_norm in one half of the data, in quali_norm in the other half, plus noise
    half = rng.random(n) < 0.5
    X["y"] = np.where(half, 1 - X["grid_norm"], 0.5 * np.sin(8 * X["quali_norm"]) - 0.5 * X["quali_norm"]) + 0.2 * rng.normal(size=n)
    X["race_id"] = np.sort(rng.integers(0, 30, n))
    X["n"] = 20
    X["pos"] = 1
    return X


def _sweep_min_step(model, base: pd.DataFrame, col: str, grid=np.linspace(0, 1, 41)) -> float:
    worst = 0.0
    for i in range(len(base)):
        d = pd.concat([base.iloc[[i]]] * len(grid), ignore_index=True)
        d[col] = grid
        y = model.model.predict(d[model.cols].astype(float))
        worst = min(worst, float(np.diff(y).min()))
    return worst


KW = dict(dynamic=False, decomp=False, ranking=False, n_estimators=60, num_leaves=7, min_child_samples=10)


def test_monotone_constraint_vector_is_exactly_the_two_columns():
    m = MonotoneGBM(**KW)
    assert MONOTONE_UP == ("grid_norm", "quali_norm")
    assert [c for c, k in zip(m.cols, m.constraints) if k == 1] == ["grid_norm", "quali_norm"]
    assert sum(k != 0 for k in m.constraints) == 2 and len(m.constraints) == len(m.cols)
    assert m.params["monotone_constraints"] == m.constraints
    assert "monotone_constraints" not in GBMModel(**KW).params          # the base class is untouched


def test_monotone_gbm_predicted_y_never_decreases_along_a_sweep_and_control_does():
    tr = _gbm_frame(600, 0)
    te = _gbm_frame(40, 1)
    mono, free = MonotoneGBM(**KW), GBMModel(**KW)
    s_mono = mono.fit_predict(tr, te)
    free.fit_predict(tr, te)
    for col in ("grid_norm", "quali_norm"):
        assert _sweep_min_step(mono, te, col) >= -1e-12, col
        assert _sweep_min_step(free, te, col) < -1e-3, f"control: the unconstrained model should violate {col}"
    # score = -y, so the score never INCREASES when the grid slot gets worse
    d = pd.concat([te.iloc[[0]]] * 21, ignore_index=True)
    d["grid_norm"] = np.linspace(0, 1, 21)
    assert (np.diff(-mono.model.predict(d[mono.cols])) <= 1e-12).all()
    assert np.allclose(s_mono, -mono.model.predict(te[mono.cols]))


def test_monotone_gbm_other_columns_stay_free():
    tr = _gbm_frame(600, 3)
    te = _gbm_frame(40, 4)
    mono = MonotoneGBM(**KW)
    mono.fit_predict(tr, te)
    other = [c for c in mono.cols if c not in MONOTONE_UP]
    # a target that depends on a free column: the monotone model must still use it (in both directions)
    tr2 = tr.copy()
    tr2["y"] = np.sin(2 * tr2[other[0]]) + 0.1 * np.random.default_rng(0).normal(size=len(tr2))
    m2 = MonotoneGBM(**KW)
    m2.fit_predict(tr2, te)
    d = pd.concat([te.iloc[[0]]] * 25, ignore_index=True)
    d[other[0]] = np.linspace(-2, 2, 25)
    st = np.diff(m2.model.predict(d[m2.cols]))
    assert (st > 1e-3).any() and (st < -1e-3).any()


def test_monotone_gbm_rejects_ranking_and_missing_columns():
    with pytest.raises(ValueError):
        MonotoneGBM(dynamic=False, decomp=False, ranking=True)
    with pytest.raises(ValueError):
        MonotoneGBM(constrained=("no_such_col",), **KW)


# ============================================================== add_quali_history: truncation
def _quali_frame(n_races: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        who = np.sort(rng.choice(10, int(rng.integers(8, 11)), replace=False))
        qn = rng.permutation(len(who)) / (len(who) - 1)
        for d, q in zip(who, qn):
            rows.append(dict(race_id=r, season=2020 + r // 15, driverId=f"d{d}", lineage=f"c{(d // 2 + r // 10) % 5}", quali_norm=float(q),
                             y=float(rng.random()), pos=1))
    return pd.DataFrame(rows)


QH_COLS = [f"drv_q_ew{h}" for h in HALFLIVES_Q] + [f"con_q_ew{h}" for h in HALFLIVES_Q]


def _qh(df):
    return add_quali_history(df).set_index(["race_id", "driverId"]).sort_index()


@pytest.mark.parametrize("r", [3, 12, 22])
def test_quali_history_for_race_r_unchanged_when_race_r_and_later_races_are_altered(r):
    df = _quali_frame()
    base = _qh(df)
    rng = np.random.default_rng(r)
    alt = df.copy()
    late = alt["race_id"] >= r                                  # race r itself and everything later
    alt.loc[late, "quali_norm"] = rng.random(int(late.sum()))
    alt.loc[late, "y"] = rng.random(int(late.sum()))
    other = _qh(alt)
    idx = base.index[base.index.get_level_values(0) == r]
    assert np.array_equal(base.loc[idx, QH_COLS].to_numpy(), other.loc[idx, QH_COLS].to_numpy())
    # all earlier races too
    idx2 = base.index[base.index.get_level_values(0) <= r]
    assert np.array_equal(base.loc[idx2, QH_COLS].to_numpy(), other.loc[idx2, QH_COLS].to_numpy())
    # cutting the frame at race r (later races removed, race r quali blanked) gives the same values for race r
    cut = df[df["race_id"] <= r].copy()
    cut.loc[cut["race_id"] == r, "quali_norm"] = np.nan
    c = _qh(cut)
    assert np.array_equal(base.loc[idx, QH_COLS].to_numpy(), c.loc[idx, QH_COLS].to_numpy())
    # control: the features are not constants; they DO react to an earlier race
    alt2 = df.copy()
    alt2.loc[alt2["race_id"] == r - 2, "quali_norm"] = 1.0 - alt2.loc[alt2["race_id"] == r - 2, "quali_norm"]
    assert not np.allclose(base.loc[idx, QH_COLS].to_numpy(), _qh(alt2).loc[idx, QH_COLS].to_numpy())


def test_quali_history_does_not_depend_on_row_order_within_a_race_and_keeps_input_order():
    df = _quali_frame()
    out = add_quali_history(df)
    assert (out["race_id"].to_numpy() == df["race_id"].to_numpy()).all() and (out["driverId"].to_numpy() == df["driverId"].to_numpy()).all()
    rng = np.random.default_rng(0)
    parts = [g.iloc[rng.permutation(len(g))] for _, g in df.groupby("race_id", sort=True)]
    sh = pd.concat(parts, ignore_index=True)
    a, b = _qh(df), _qh(sh)
    assert np.allclose(a[QH_COLS].to_numpy(), b[QH_COLS].to_numpy(), atol=1e-15)
    # first appearance gets the prior (nothing earlier)
    first = out.groupby("driverId").head(1)
    assert first["drv_q_ew8"].nunique() == 1 and first["drv_q_ew32"].nunique() == 1


def test_quali_history_input_columns_are_pre_race_only():
    assert not {"y", "pos", "classified", "finish_order", "points", "dnf"} & set(STAGE1_HIST + STAGE1_PRACTICE)
    # drv_ew16 / con_ew16 are plan-1 form features (earlier races only); quali columns are the new history
    assert set(STAGE1_HIST) == {"drv_q_ew8", "drv_q_ew32", "con_q_ew8", "con_q_ew32", "drv_ew16", "con_ew16"}


@needs_real
@pytest.mark.parametrize("r", [150, 185, 202])
def test_quali_history_truncation_real_data(r):
    from pitlane.plan2 import pipeline2 as Q
    e = Q.load_all()
    base = add_quali_history(e).set_index(["race_id", "driverId"]).sort_index()
    alt = e.copy()
    rng = np.random.default_rng(1)
    late = alt["race_id"] >= r
    alt.loc[late, "quali_norm"] = rng.random(int(late.sum()))
    other = add_quali_history(alt).set_index(["race_id", "driverId"]).sort_index()
    idx = base.index[base.index.get_level_values(0) == r]
    assert np.array_equal(base.loc[idx, QH_COLS].to_numpy(), other.loc[idx, QH_COLS].to_numpy())


# ============================================================== two-stage: leakage of the Friday forecast
@pytest.fixture(scope="module")
def real_entries():
    from pitlane.plan2 import pipeline2 as Q
    return add_quali_history(Q.load_all(), HALFLIVES_Q)


def _h2():
    return load_yaml(ROOT / "configs" / "frozen.yaml")["M03"]["111"]


def _randomise_race(e: pd.DataFrame, r: int, seed: int, from_r_on: bool = True) -> pd.DataFrame:
    """Replace race r's (and later races') real qualifying, grid, Q1 gap, pit-start and result columns by random values."""
    rng = np.random.default_rng(seed)
    out = e.copy()
    m = (out["race_id"] >= r) if from_r_on else (out["race_id"] == r)
    k = int(m.sum())
    out.loc[m, "quali_norm"] = rng.random(k)
    out.loc[m, "grid_norm"] = rng.random(k)
    out.loc[m, "q1_gap"] = rng.random(k) * 0.2
    out.loc[m, "pit_start"] = (rng.random(k) < 0.2).astype(int)
    out.loc[m, "quali_pos"] = rng.integers(1, 21, k).astype(float)
    out.loc[m, "grid"] = rng.integers(0, 21, k).astype(float)
    out.loc[m, "y"] = rng.random(k)
    out.loc[m, "pos"] = rng.integers(1, 21, k)
    out.loc[m, "classified"] = rng.random(k) < 0.5
    return out


STAGE_COLS = STAGE1_HIST + STAGE1_PRACTICE


@needs_real
@pytest.mark.parametrize("r", [175, 190])          # one 2022 race without practice, one 2023 race with practice
def test_stage1_prediction_ignores_the_races_own_quali_grid_results(real_entries, r):
    e = real_entries
    first = int(np.flatnonzero(e["race_id"].to_numpy() == r)[0])
    alt = _randomise_race(e, r, 5)
    # the history features of race r (computed from earlier races) are untouched by the randomisation
    alt_h = add_quali_history(alt.drop(columns=[c for c in QH_COLS]), HALFLIVES_Q)
    assert np.array_equal(e.loc[e["race_id"] == r, QH_COLS].to_numpy(), alt_h.loc[alt_h["race_id"] == r, QH_COLS].to_numpy())
    a = QualiStage1(STAGE_COLS).fit(e.iloc[:first]).predict(e[e["race_id"] == r])
    b = QualiStage1(STAGE_COLS).fit(alt_h.iloc[:first]).predict(alt_h[alt_h["race_id"] == r])
    assert np.array_equal(a, b)
    # control: randomising an EARLIER race changes it
    early = e.copy()
    early.loc[early["race_id"] == r - 3, "quali_norm"] = 1 - early.loc[early["race_id"] == r - 3, "quali_norm"]
    early = add_quali_history(early.drop(columns=QH_COLS), HALFLIVES_Q)
    c = QualiStage1(STAGE_COLS).fit(early.iloc[:first]).predict(early[early["race_id"] == r])
    assert not np.allclose(a, c)


def _two_stage_scores(e: pd.DataFrame, r: int, shuffle_seed: int | None = None):
    first = int(np.flatnonzero(e["race_id"].to_numpy() == r)[0])
    m = TwoStageH2(e, _h2(), STAGE_COLS)
    test = e[e["race_id"] == r]
    if shuffle_seed is not None:
        test = test.iloc[np.random.default_rng(shuffle_seed).permutation(len(test))]
    s = m.fit_predict(e.iloc[:first], test)
    return pd.Series(s, index=test["driverId"].to_numpy()).sort_index(), m


@needs_real
@pytest.mark.slow
@pytest.mark.parametrize("r", [175, 190])
def test_two_stage_prediction_ignores_the_races_own_real_qualifying_and_results(real_entries, r):
    a, ma = _two_stage_scores(real_entries, r)
    alt = _randomise_race(real_entries.drop(columns=[]), r, 9)
    b, mb = _two_stage_scores(alt, r)
    assert (a.index == b.index).all() and np.array_equal(a.to_numpy(), b.to_numpy())
    la, lb = ma.log[0].sort_values("driverId"), mb.log[0].sort_values("driverId")
    assert np.array_equal(la["q1_raw"].to_numpy(), lb["q1_raw"].to_numpy()) and np.array_equal(la["q1_rank"].to_numpy(), lb["q1_rank"].to_numpy())
    # the real qualifying of race r is only carried in the log for scoring, never fed to stage 2
    assert not np.array_equal(la["quali_pos"].to_numpy(), lb["quali_pos"].to_numpy())
    # same with only race r randomised (later races intact): the later races cannot matter either
    c, _ = _two_stage_scores(_randomise_race(real_entries, r, 11, from_r_on=False), r)
    assert np.array_equal(a.to_numpy(), c.to_numpy())
    # control: the prediction is not a constant function; randomising the earlier races changes it
    early = _randomise_race(real_entries, r - 4, 3, from_r_on=False)
    d, _ = _two_stage_scores(early, r)
    assert not np.allclose(a.to_numpy(), d.to_numpy())


@needs_real
@pytest.mark.slow
def test_two_stage_scores_do_not_depend_on_row_order_of_the_test_frame(real_entries):
    a, _ = _two_stage_scores(real_entries, 190)
    for seed in (1, 2):
        b, _ = _two_stage_scores(real_entries, 190, shuffle_seed=seed)
        assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12, rtol=0)


@needs_real
@pytest.mark.slow
def test_two_stage_restores_the_universe_after_prediction(real_entries):
    """Stage 2 overwrites the test rows' inputs while predicting and must restore the real values (later refits train on them)."""
    e = real_entries
    r = 190
    first = int(np.flatnonzero(e["race_id"].to_numpy() == r)[0])
    m = TwoStageH2(e, _h2(), STAGE_COLS)
    before = m.latent.row_x.copy()
    m.fit_predict(e.iloc[:first], e[e["race_id"] == r])
    assert np.array_equal(before, m.latent.row_x)


def test_rank_within_is_row_order_free_and_one_is_best():
    v = np.array([0.3, 0.1, 0.1, 0.9])
    k = np.array(["c", "b", "a", "d"])
    r = rank_within(v, k)
    assert r.tolist() == [3, 2, 1, 4]                       # smallest value = rank 1; tie between b and a goes to a
    p = np.array([2, 0, 3, 1])
    assert rank_within(v[p], k[p])[np.argsort(p)].tolist() == r.tolist()


# ============================================================== stored results
def _dir(root: Path, suffix: str) -> Path:
    c = sorted(root.glob(f"*_{suffix}"))
    assert len(c) == 1, (suffix, c)
    return c[0]


M05_RUNS = {"latent_111": "m05_ref_latent_111_dev", "gbm_100": "m05_ref_gbm_100_dev", "M05_rank_avg": "m05_rank_avg_dev",
            "M05_gbm_monotone": "m05_gbm_monotone_dev"}
M12_RUNS = {"h1": "m12_ref_h1_latent_111_dev", "two_stage": "m12_two_stage_dev"}


def _all_runs():
    return [(M05, s) for s in M05_RUNS.values()] + [(M12, s) for s in M12_RUNS.values()]


@needs_m05
@needs_m12
@pytest.mark.parametrize("root,suffix", _all_runs(), ids=lambda x: x if isinstance(x, str) else "")
def test_per_race_metrics_reproduce_from_predictions(root, suffix):
    d = _dir(root, suffix)
    pr = pd.read_parquet(d / "predictions.parquet")
    st = pd.read_csv(d / "per_race_metrics.csv").set_index("race_id")
    assert len(pr) == 1311 and len(st) == 66 and sorted(pr["race_id"].unique()) == sorted(st.index)
    assert not pr.duplicated(["race_id", "driverId"]).any() and pr["score"].notna().all()
    assert pr["season"].between(2021, 2023).all()
    for rid, g in pr.groupby("race_id"):
        m = race_metrics(g["pos"].to_numpy(), g["score"].to_numpy(), g["classified"].to_numpy())
        assert len(g) == st.loc[rid, "n"]
        for k, v in m.items():
            if k == "within3" and k not in st.columns:  # metrics added after these runs were stored (PLAN3 section 5, 2026-11-17); recomputed but not stored
                continue
            assert st.loc[rid, k] == pytest.approx(v, abs=1e-12, nan_ok=True), (suffix, rid, k)


def _table(root, suffix):
    return A.race_table(U.outcome_free(pd.read_parquet(_dir(root, suffix) / "predictions.parquet"), "driverId"))


def _check_attribution_row(comp, ref, ref_tab, cand_tab, run_dir_name, n_races):
    att = pd.read_csv(ATT)
    row = att[(att["component"] == comp) & (att["reference"] == ref)]
    assert len(row) == 1, (comp, ref)
    row = row.iloc[0]
    g = A.paired_gain(ref_tab, cand_tab, n_boot=10000, seed=0)
    assert len(g) >= 10
    for _, r in g.iterrows():
        for key in ("gain", "lo", "hi"):
            assert row[f"{key}_{r['metric']}"] == pytest.approx(r[key], abs=1e-12), (comp, ref, r["metric"], key)
    for m in set(A.METRICS) - set(g["metric"]):
        assert pd.isna(row[f"gain_{m}"]), (comp, m)
    assert row["verdict_metric"] == "spearman" and row["verdict"] == A.verdict(g[g["metric"] == "spearman"].iloc[0])
    assert row["run_id"] == run_dir_name and row["n_races"] == n_races and row["declared"] == "yes"


@needs_m05
def test_attribution_m05_rows_reproduce_from_run_folders():
    tabs = {k: _table(M05, s) for k, s in M05_RUNS.items()}
    for comp, ref in (("M05_rank_avg", "latent_111"), ("M05_rank_avg", "gbm_100"), ("M05_gbm_monotone", "gbm_100")):
        _check_attribution_row(comp, ref, tabs[ref], tabs[comp], _dir(M05, M05_RUNS[comp]).name, 66)


@needs_m05
def test_m05_paired_gains_and_verdict_files_match_the_attribution_rows():
    pg = pd.read_csv(M05 / "paired_gains_dev.csv")
    assert set(zip(pg["component"], pg["reference"])) == {("M05_rank_avg", "latent_111"), ("M05_rank_avg", "gbm_100"), ("M05_gbm_monotone", "gbm_100")}
    att = pd.read_csv(ATT)
    for (comp, ref), g in pg.groupby(["component", "reference"]):
        row = att[(att["component"] == comp) & (att["reference"] == ref)].iloc[0]
        for _, r in g.iterrows():
            assert row[f"gain_{r['metric']}"] == pytest.approx(r["gain"], abs=1e-12)
    ver = pd.read_csv(M05 / "verdicts.csv")
    for _, r in ver.iterrows():
        w = "passes alone" if r["lo"] > 0 else ("worse" if r["hi"] < 0 else "no measurable difference")
        assert r["verdict"] == w


def _prac_ids_dev(e: pd.DataFrame) -> list[int]:
    first = e.groupby("race_id")["season"].first()
    dev = first[(first >= 2021) & (first <= 2023)].index
    has = e.groupby("race_id")["fp_gap"].apply(lambda s: bool(s.notna().any()))
    return [int(r) for r in dev if has[r]]


@needs_real
def test_m12_comparison_uses_exactly_the_21_dev_races_with_practice_and_not_race_193():
    from pitlane.plan2 import pipeline2 as Q
    e = Q.load_all()
    ids = _prac_ids_dev(e)
    assert len(ids) == 21 and 193 not in ids
    first = e.groupby("race_id")["season"].first()
    assert all(first[i] == 2023 for i in ids)
    dev2023 = [int(r) for r in first[first == 2023].index]
    assert len(dev2023) == 22 and set(dev2023) - set(ids) == {193}
    assert e.loc[e["race_id"] == 193, "fp_gap"].isna().all()             # why 193 is out: no practice data at all
    for r in ids:
        assert e.loc[e["race_id"] == r, "fp_gap"].notna().any()
    chk = pd.read_json(M12 / "reproduction_check.json", typ="series") if (M12 / "reproduction_check.json").exists() else None
    assert chk is not None and list(chk["practice_race_ids"]) == ids and chk["practice_races"] == 21


@needs_m12
@needs_real
def test_attribution_m12_row_and_paired_gain_file_reproduce_from_run_folders():
    from pitlane.plan2 import pipeline2 as Q
    ids = _prac_ids_dev(Q.load_all())
    sub = lambda t: t[t["race_id"].isin(ids)].reset_index(drop=True)
    ts, h1 = sub(_table(M12, M12_RUNS["two_stage"])), sub(_table(M12, M12_RUNS["h1"]))
    assert len(ts) == len(h1) == 21 and 193 not in set(ts["race_id"])
    _check_attribution_row("M12_two_stage", "plan2_H1_latent_111", h1, ts, _dir(M12, M12_RUNS["two_stage"]).name, 21)
    pg = pd.read_csv(M12 / "paired_gains_dev.csv")
    g = A.paired_gain(h1, ts, n_boot=10000, seed=0).set_index("metric")
    p = pg[pg["subset"] == "practice_2023"].set_index("metric")
    assert (p["n"] == 21).all()
    for m in g.index:
        for key in ("gain", "lo", "hi"):
            assert p.loc[m, key] == pytest.approx(g.loc[m, key], abs=1e-12), (m, key)
    assert set(pg.groupby("subset")["n"].first()) == {21, 45, 66}


def _stored_plan(lb: str, label: str):
    return U.stored_run(lb, label)


@needs_m05
@needs_m12
def test_reference_runs_equal_the_stored_plan1_and_plan2_dev_runs():
    for root, suffix, lb, label in ((M05, M05_RUNS["latent_111"], "results/leaderboard.csv", "latent_111"),
                                    (M05, M05_RUNS["gbm_100"], "results/leaderboard.csv", "gbm_100"),
                                    (M12, M12_RUNS["h1"], "results/plan2/leaderboard.csv", "h1_latent_111")):
        d = _dir(root, suffix)
        new_pm, new_pr = pd.read_csv(d / "per_race_metrics.csv"), pd.read_parquet(d / "predictions.parquet")
        old_pm, old_pr = _stored_plan(lb, label)
        shared = [c for c in old_pm.columns if c in new_pm.columns]
        assert {"spearman", "mae", "spearman_fin"} <= set(shared)   # plan-1 file predates the exact-rate columns
        assert len(new_pm) == len(old_pm) == 66
        a, b = new_pm.set_index("race_id")[[c for c in shared if c != "race_id"]], old_pm.set_index("race_id")[[c for c in shared if c != "race_id"]]
        assert np.nanmax(np.abs(a.astype(float) - b.loc[a.index].astype(float)).to_numpy()) <= 1e-12, label
        mm = new_pr.merge(old_pr, on=["race_id", "driverId"], suffixes=("_n", "_o"))
        assert len(mm) == len(new_pr) == len(old_pr) == 1311, label
        assert (mm["score_n"] - mm["score_o"]).abs().max() <= 1e-12, label


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n")


@needs_m05
@needs_m12
def test_leaderboards_are_append_only_against_head():
    for rel in ("results/plan3/leaderboard.csv", "results/leaderboard.csv", "results/plan2/leaderboard.csv"):
        try:
            old = _norm(subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT, text=True))
        except Exception as e:                                                  # pragma: no cover
            pytest.skip(f"git show failed: {e}")
        cur = _norm((ROOT / rel).read_text(encoding="utf-8"))
        o, c = pd.read_csv(StringIO(old), dtype=str), pd.read_csv(StringIO(cur), dtype=str)
        assert list(o.columns) == list(c.columns), rel
        assert len(c) >= len(o), rel
        pd.testing.assert_frame_equal(c.iloc[:len(o)].reset_index(drop=True), o.reset_index(drop=True), check_exact=True)   # old rows byte-for-byte
        assert not c.duplicated("run_id").any(), rel
        if rel != "results/plan3/leaderboard.csv":
            assert len(c) == len(o), f"{rel}: plan-1/plan-2 leaderboards must not gain rows from R10"


@needs_m05
@needs_m12
def test_new_leaderboard_rows_match_their_run_folders():
    lb = pd.read_csv(LB3, dtype={"run_id": str})
    want = {**{f"m05_{k}": v for k, v in {"ref_latent_111": M05_RUNS["latent_111"], "ref_gbm_100": M05_RUNS["gbm_100"],
                                           "rank_avg": M05_RUNS["M05_rank_avg"], "gbm_monotone": M05_RUNS["M05_gbm_monotone"]}.items()}}
    new = lb[lb["method"].isin(["M05", "M12"])]
    assert len(new) == 6 and new["split"].eq("dev").all() and not new.duplicated("run_id").any()
    for _, r in new.iterrows():
        d = ROOT / r["run_dir"]
        assert d.exists() and d.name == r["run_id"]
        pm = pd.read_csv(d / "per_race_metrics.csv")
        assert r["n_races"] == len(pm) == 66
        for c in ("spearman", "mae", "exact", "spearman_fin", "mae_fin", "logloss_points"):
            assert r[c] == pytest.approx(pm[c].mean(), abs=1e-12), (r["run_id"], c)
        cfg = load_yaml(d / "config.yaml")
        assert cfg.get("split", "dev") == "dev"


@needs_m05
def test_m05_monotone_check_json_and_card_numbers():
    import json
    chk = json.loads((M05 / "monotone_check.json").read_text(encoding="utf-8"))
    assert all(v["entries_with_decreasing_y"] == 0 for v in chk["M05_gbm_monotone"].values())
    assert chk["gbm_100"]["grid_norm"]["entries_with_decreasing_y"] > 0          # control: the unconstrained model does violate
    ag = json.loads((M05 / "member_agreement.json").read_text(encoding="utf-8"))
    lt, gb = (pd.read_parquet(_dir(M05, M05_RUNS[k]) / "predictions.parquet") for k in ("latent_111", "gbm_100"))
    rho = []
    for rid, a in lt.groupby("race_id"):
        b = gb[gb["race_id"] == rid].set_index("driverId").loc[a["driverId"]]
        rho.append(np.corrcoef(a["score"].rank().to_numpy(), b["score"].rank().to_numpy())[0, 1])
    assert ag["races"] == 66 and ag["mean_rank_corr_latent_vs_gbm"] == pytest.approx(np.mean(rho), abs=1e-12)


@needs_m05
def test_m05_rank_average_predictions_equal_rank_average_of_stored_member_scores():
    """The stored M05 rank-average run is the equal-weight rank average of the stored latent and gbm runs (hand re-derivation)."""
    lt, gb, ra = (pd.read_parquet(_dir(M05, M05_RUNS[k]) / "predictions.parquet") for k in ("latent_111", "gbm_100", "M05_rank_avg"))
    for rid, g in ra.groupby("race_id"):
        a = lt[lt["race_id"] == rid].set_index("driverId").loc[g["driverId"], "score"].to_numpy()
        b = gb[gb["race_id"] == rid].set_index("driverId").loc[g["driverId"], "score"].to_numpy()
        want = pred_rank(rank_average_score([a, b], g["driverId"].to_numpy()))
        got = pred_rank(U.outcome_free(g, "driverId")["score"].to_numpy())
        assert np.array_equal(want, got), rid


# ============================================================== slow: re-run the first dev races from scratch
FIRST_RACES = 3


def _quick(entries: pd.DataFrame, model):
    first = entries.groupby("race_id")["season"].first()
    dev = first[(first >= 2021) & (first <= 2023)].index[:FIRST_RACES]
    run = entries[entries["race_id"] <= dev.max()].reset_index(drop=True)
    p, m = walk_forward(run, model(run), "dev", refit_every=1)
    return p[p["race_id"].isin(dev)], m[m["race_id"].isin(dev)]


def _same_scores(p, d, tol=1e-9):
    old = pd.read_parquet(d / "predictions.parquet")
    mm = p.merge(old, on=["race_id", "driverId"], suffixes=("_n", "_o"))
    assert len(mm) == len(p) > 0
    assert (mm["score_n"] - mm["score_o"]).abs().max() <= tol


@needs_real
@needs_m05
@pytest.mark.slow
def test_rerun_first_dev_races_rank_average_and_monotone_reproduce_stored_scores():
    from pitlane import pipeline as P
    e = P.load_entries()
    fz = load_yaml(ROOT / "configs" / "frozen.yaml")
    lat, gbm = fz["M03"]["111"], fz["M04"]["100"]
    cfg = load_yaml(ROOT / "methodologies" / "M05_ensemble" / "config.yaml")
    p, _ = _quick(e, lambda run: RankAverageModel([build_model(lat, run), build_model(gbm, run)], "driverId"))
    _same_scores(p, _dir(M05, M05_RUNS["M05_rank_avg"]))
    dyn, dec, rk = cell_flags(gbm["cell"])
    p, _ = _quick(e, lambda run: MonotoneGBM(dynamic=dyn, decomp=dec, ranking=rk, constrained=tuple(cfg["monotone"]["increasing_y"]),
                                             **{k: v for k, v in gbm.items() if k not in ("kind", "cell")}))
    _same_scores(p, _dir(M05, M05_RUNS["M05_gbm_monotone"]))


@needs_real
@needs_m12
@pytest.mark.slow
def test_rerun_two_stage_full_dev_walk_reproduces_stored_scores():
    """Full dev walk-forward on the FULL universe (the stored run's setup). A universe cut at the third dev race does NOT
    reproduce the stored scores to 1e-9 (max diff about 3.5e-4, see test below): the L-BFGS fit is capped at maxiter and
    its path depends on the number of parameters in the universe."""
    from pitlane.plan2 import pipeline2 as Q
    e = add_quali_history(Q.load_all(), HALFLIVES_Q)
    cfg = load_yaml(ROOT / "methodologies" / "M12_two_stage" / "config.yaml")
    s1 = cfg["stage1"]
    p, m = walk_forward(e, TwoStageH2(e, _h2(), s1["history_cols"] + s1["practice_cols"], s1["alpha"]), "dev", refit_every=1)
    d = _dir(M12, M12_RUNS["two_stage"])
    old = pd.read_parquet(d / "predictions.parquet")
    mm = p.merge(old, on=["race_id", "driverId"], suffixes=("_n", "_o"))
    assert len(mm) == len(old) == len(p) == 1311
    assert (mm["score_n"] - mm["score_o"]).abs().max() <= 1e-9
    stored = pd.read_csv(d / "per_race_metrics.csv")
    pd.testing.assert_frame_equal(m.reset_index(drop=True)[stored.columns], stored, check_exact=False, rtol=1e-9, atol=1e-9)


@needs_real
@needs_m12
@pytest.mark.slow
def test_two_stage_scores_depend_on_universe_size_only_through_the_optimiser_not_through_future_data():
    """Documents the 3.5e-4 effect: cutting the universe changes the scores slightly (fewer unused parameters), while changing
    the VALUES of every later row (same universe size) changes nothing, bit for bit (leakage test above)."""
    from pitlane.plan2 import pipeline2 as Q
    e = add_quali_history(Q.load_all(), HALFLIVES_Q)
    cfg = load_yaml(ROOT / "methodologies" / "M12_two_stage" / "config.yaml")
    s1 = cfg["stage1"]
    cols = s1["history_cols"] + s1["practice_cols"]
    first = e.groupby("race_id")["season"].first()
    dev = first[(first >= 2021) & (first <= 2023)].index[:2]
    r = int(dev[-1])
    alt = _randomise_race(e, int(dev[-1]) + 1, 21)           # every row after the second dev race gets random values
    # direct comparison on the two first dev races, same sequence of refits (warm starts), full-size universes
    def run(u):
        m = TwoStageH2(u, _h2(), cols, s1["alpha"])
        sc = {}
        for rid in [int(x) for x in sorted(u["race_id"].unique()) if x <= r]:
            n = int(np.flatnonzero(u["race_id"].to_numpy() == rid)[0])
            if n == 0:
                continue
            t = u[u["race_id"] == rid]
            sc[rid] = pd.Series(m.fit_predict(u.iloc[:n], t), index=t["driverId"].to_numpy()).sort_index()
        return sc
    a, b = run(e), run(alt)
    assert a.keys() == b.keys() and r in a
    for k in a:
        assert np.array_equal(a[k].to_numpy(), b[k].to_numpy()), k


@needs_real
@needs_m05
@pytest.mark.slow
def test_monotone_sweep_on_real_data_last_dev_race():
    from pitlane import pipeline as P
    e = P.load_entries()
    gbm = load_yaml(ROOT / "configs" / "frozen.yaml")["M04"]["100"]
    dyn, dec, rk = cell_flags(gbm["cell"])
    first = e.groupby("race_id")["season"].first()
    rid = int(first[first <= 2023].index.max())
    n = int(np.flatnonzero(e["race_id"].to_numpy() == rid)[0])
    te = e[e["race_id"] == rid]
    m = MonotoneGBM(dynamic=dyn, decomp=dec, ranking=rk, **{k: v for k, v in gbm.items() if k not in ("kind", "cell")})
    m.fit_predict(e.iloc[:n], te)
    for col in ("grid_norm", "quali_norm"):
        assert _sweep_min_step(m, te, col) >= -1e-12, col
    free = build_model(gbm, e)
    free.fit_predict(e.iloc[:n], te)
    assert min(_sweep_min_step(free, te, c) for c in ("grid_norm", "quali_norm")) < -1e-6      # control
