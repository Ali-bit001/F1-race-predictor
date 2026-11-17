"""PLAN3 R8 phase 2 / M20 consistency in the model (cons_x in latent_111's weekend term and in gbm_100): independent checks.

Sections: (1) leakage of the cons_x history (truncation and scrambling of race r and later), (2) the join checked by brute force,
(3) the scaling constant (seasons <= 2020 only), (4) reproduction (re-run references vs stored plan-1 runs; zero column = reference),
(5) paired gains and attribution rows re-derived with the shared harness, (6) shared files are append-only (pinned commit),
(7) split discipline and data checks.
Real-data tests skip when the git-ignored inputs are missing. M16 (and later) rows appended to the shared files are allowed.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from pitlane import pipeline as P
from pitlane.config import PROCESSED, ROOT, SPLITS, load_yaml
from pitlane.evaluation.harness import walk_forward
from pitlane.features import racecraft as RC
from pitlane.models.registry import build_model
from pitlane.plan3 import attribution as A
from pitlane.plan3 import runutil as U

METH = ROOT / "methodologies/M20_racecraft"
RES = METH / "results"
CFG = load_yaml(METH / "config_phase2.yaml")
PHASE1_COMMIT = "1db3fdc"          # R8 phase 1: shared result files as phase 2 found them
STAMP = "20261117-0645"
LABELS = {"latent_111": "m20p2_ref_latent_111", "gbm_100": "m20p2_ref_gbm_100",
          "M20_consistency_latent_111": "m20p2_cons_latent_111", "M20_consistency_gbm_100": "m20p2_cons_gbm_100",
          "M20_consistency_tm_latent_111": "m20p2_posthoc_tm_latent_111", "M20_consistency_tm_gbm_100": "m20p2_posthoc_tm_gbm_100"}
needs_runs = pytest.mark.skipif(not (RES / "phase2_paired_gains_dev.csv").exists(), reason="M20 phase-2 results missing")
needs_real = pytest.mark.skipif(not ((PROCESSED / "entries2.parquet").exists() and (ROOT / "data/interim/plan3/laps.parquet").exists()),
                                reason="processed data / lap tables missing (regenerable)")


def _load_run2():
    spec = importlib.util.spec_from_file_location("m20_phase2_run_under_test", METH / "run_phase2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R2 = _load_run2()


def _dir(label: str):
    c = sorted(RES.glob(f"{STAMP}_{label}_dev"))
    assert len(c) == 1, (label, c)
    return c[0]


# ====================================================================================== fixtures
@pytest.fixture(scope="module")
def world():
    """Raw entries (all seasons), phase-1 race values (2014-2023) and entries <= 2023, built once (~15 s)."""
    entries0 = P.load_entries()
    vals, ent_le = R2.race_values_dev(CFG)
    return dict(entries0=entries0, vals=vals, ent_le=ent_le)


@pytest.fixture(scope="module")
def featured(world):
    ent, info = R2.with_features(world["entries0"], world["vals"], world["ent_le"], CFG)
    return ent, info


def _dev_race_ids(entries: pd.DataFrame) -> np.ndarray:
    first = entries.groupby("race_id")["season"].first()
    return first[(first >= SPLITS["dev"][0]) & (first <= SPLITS["dev"][1])].index.to_numpy()


def _hist(entries, vals, col="cons_c"):
    return R2.history_column(entries, vals, col, CFG["halflife"])


# ====================================================================================== (1) leakage
@needs_real
@pytest.mark.parametrize("pos_in_dev", [0, 30, 60])
def test_cons_x_of_race_r_unchanged_when_race_r_and_later_values_are_scrambled(world, pos_in_dev):
    """Scramble every cons_c value of race r and all later races (also swapping which driver of the race they belong to): the feature of every
    entry of race <= r is bit for bit unchanged, including race r itself."""
    e0, vals = world["entries0"], world["vals"]
    r = int(_dev_race_ids(e0)[pos_in_dev])
    base = _hist(e0, vals)
    rng = np.random.default_rng(pos_in_dev)
    alt = vals.copy()
    late = (alt["race_id"] >= r).to_numpy()
    alt.loc[late, "cons_c"] = rng.normal(0, 7.0, late.sum())
    for _, ix in alt[late].groupby("race_id").groups.items():            # swap which driver owns which value, inside each race
        alt.loc[ix, "driverId"] = rng.permutation(alt.loc[ix, "driverId"].to_numpy())
    got = _hist(e0, alt)
    m = (e0["race_id"] <= r).to_numpy()
    assert m.sum() > 100 and base[m].notna().all()
    assert np.array_equal(base[m].to_numpy(), got[m].to_numpy())
    # control (power): scrambling an EARLIER race moves later entries and leaves earlier ones alone
    r0 = int(_dev_race_ids(e0)[pos_in_dev]) - 3
    alt2 = vals.copy()
    sel = (alt2["race_id"] == r0).to_numpy()
    assert sel.sum() > 0
    alt2.loc[sel, "cons_c"] = alt2.loc[sel, "cons_c"] + 50.0
    c = _hist(e0, alt2)
    assert np.array_equal(base[(e0["race_id"] <= r0).to_numpy()].to_numpy(), c[(e0["race_id"] <= r0).to_numpy()].to_numpy())
    later = ((e0["race_id"] > r0) & (e0["race_id"] <= r0 + 10)).to_numpy()
    assert (base[later].to_numpy() != c[later].to_numpy()).mean() > 0.3


@needs_real
@pytest.mark.parametrize("pos_in_dev", [0, 30, 60])
def test_cons_x_of_race_r_unchanged_when_values_from_race_r_onward_are_removed(world, pos_in_dev):
    """Truncation: drop every value row with race_id >= r. One sentinel value row of a far later race is kept, because
    history_column masks races beyond the last valued race (NaN) and without the sentinel race r itself would be NaN and the
    comparison vacuous (see run_phase2.leak_check, which compares with nanmax and so cannot see race r)."""
    e0, vals = world["entries0"], world["vals"]
    r = int(_dev_race_ids(e0)[pos_in_dev])
    base = _hist(e0, vals)
    cut = vals[vals["race_id"] < r]
    far = vals[vals["race_id"] == vals["race_id"].max()].iloc[[0]].copy()
    far["driverId"] = "zz_sentinel_driver"
    got = _hist(e0, pd.concat([cut, far], ignore_index=True))
    m = (e0["race_id"] <= r).to_numpy()
    assert got[m].notna().all()
    assert np.array_equal(base[m].to_numpy(), got[m].to_numpy())
    # and the literal truncation (no sentinel) leaves races < r unchanged, race r falls to NaN: nothing of race r is used
    lit = _hist(e0, cut)
    below = (e0["race_id"] < r).to_numpy()
    assert np.array_equal(base[below].to_numpy(), lit[below].to_numpy())


@needs_real
def test_known_weakness_run_phase2_leak_check_is_blind_to_race_r_itself(world):
    """Documented finding, not a failure of the feature: leak_check blanks race r and later with NaN, then compares with nanmax;
    the rows of race r become NaN (last valued race = r - 1) and are ignored, so the run's own check covers races < r only.
    The feature itself is clean (tests above); this test pins the behaviour so a change is noticed."""
    e0, vals = world["entries0"], world["vals"]
    r = int(_dev_race_ids(e0)[30])
    cut = _hist(e0, vals.assign(cons_c=vals["cons_c"].where(vals["race_id"] < r)))
    assert cut[(e0["race_id"] == r).to_numpy()].isna().all()
    assert R2.leak_check(e0, vals, r, CFG["halflife"]) == 0.0


@needs_real
def test_scale_constant_does_not_depend_on_dev_or_later_values(world):
    """Scrambling all values of seasons >= 2021 leaves scale_sd and every feature of seasons <= 2020 unchanged."""
    e0, vals, ent_le = world["entries0"], world["vals"], world["ent_le"]
    ent, info = R2.with_features(e0, vals, ent_le, CFG)
    rng = np.random.default_rng(3)
    alt = vals.copy()
    late = (alt["season"] >= 2021).to_numpy()
    alt.loc[late, "cons_c"] = rng.normal(0, 9.0, late.sum())
    ent2, info2 = R2.with_features(e0, alt, ent_le, CFG)
    assert info2["cons_x"]["scale_sd"] == info["cons_x"]["scale_sd"]
    old = (e0["season"] <= 2020).to_numpy()
    assert np.array_equal(ent.loc[old, "cons_x"].to_numpy(), ent2.loc[old, "cons_x"].to_numpy(), equal_nan=True)
    assert not np.array_equal(ent.loc[~old & (e0["season"] <= 2023).to_numpy(), "cons_x"].to_numpy(),
                              ent2.loc[~old & (e0["season"] <= 2023).to_numpy(), "cons_x"].to_numpy())


# ====================================================================================== (2) the join, by brute force
def _brute(vals: pd.DataFrame, driver: str, race: int, hl: float) -> float:
    """Shrunk decayed mean of the driver's valid values of races < race: sum(w v) / (1 + sum(w)), w = 0.5 ** ((race - s) / hl)."""
    v = vals[(vals["driverId"] == driver) & (vals["race_id"] < race)].dropna(subset=["cons_c"])
    w = 0.5 ** ((race - v["race_id"].to_numpy(float)) / hl)
    return float((w * v["cons_c"].to_numpy(float)).sum() / (1.0 + w.sum()))


@needs_real
def test_history_column_equals_brute_force_on_a_sample_of_entries(world):
    e0, vals = world["entries0"], world["vals"]
    h = _hist(e0, vals)
    assert len(h) == len(e0) and h.index.equals(e0.index)
    dev = e0[e0["season"].between(2021, 2023)]
    rng = np.random.default_rng(0)
    pick = list(rng.choice(dev.index.to_numpy(), 60, replace=False))
    early = e0[e0["season"] <= 2016]
    pick += list(rng.choice(early.index.to_numpy(), 20, replace=False))
    for i in pick:
        want = _brute(vals, e0.loc[i, "driverId"], int(e0.loc[i, "race_id"]), CFG["halflife"])
        assert h.loc[i] == pytest.approx(want, abs=1e-12), (i, e0.loc[i, ["race_id", "driverId"]].tolist())


@needs_real
def test_entries_without_valid_earlier_values_get_zero_prior_not_nan(world):
    """Documented in run_phase2 (shrunk decayed mean, prior 0, race_id as time): no valid earlier value -> exactly 0.0, which is
    the field-median consistency (cons_c is field-relative). Every driver of the first race has no history; so do dev rookies."""
    e0, vals = world["entries0"], world["vals"]
    h = _hist(e0, vals)
    first = e0["race_id"] == e0["race_id"].min()
    assert (h[first.to_numpy()] == 0.0).all()
    ever = vals.dropna(subset=["cons_c"])[["driverId", "race_id"]].groupby("driverId")["race_id"].min()
    dev = e0[e0["season"].between(2021, 2023)]
    nohist = dev[dev.apply(lambda r: (r["driverId"] not in ever.index) or ever[r["driverId"]] >= r["race_id"], axis=1)]
    assert len(nohist) > 0
    assert (h.loc[nohist.index] == 0.0).all() and h.loc[nohist.index].notna().all()
    # and ONLY those entries are exactly zero apart from rounding coincidences: nonzero history is nonzero
    zero = h[e0["season"].between(2021, 2023)] == 0.0
    assert set(zero[zero].index) == set(nohist.index)
    # hand case of the documented choice: one earlier value v at lag 1 race gives v * w / (1 + w), w = 0.5 ** (1 / 50); no value gives 0
    v = pd.DataFrame(dict(race_id=[1, 1, 3], driverId=["a", "b", "c"], cons_c=[2.0, np.nan, 1.0]))     # race 3 value: sets the last valued race
    ent = pd.DataFrame(dict(race_id=[1, 2, 2], driverId=["a", "a", "b"]))
    out = R2.history_column(ent, v, "cons_c", 50)
    w = 0.5 ** (1 / 50)
    assert out.tolist() == pytest.approx([0.0, 2.0 * w / (1 + w), 0.0], abs=1e-15)


@needs_real
def test_value_rows_without_entry_rows_stay_in_history_only(world):
    """norris 2019 R18 and alonso 2023 R19 have lap-table values but no entries2 row: no feature row is created for them,
    and the value does feed the same driver's later history."""
    e0, vals = world["entries0"], world["vals"]
    ids = e0[["season", "round", "race_id"]].drop_duplicates()
    cases = [("norris", 2019, 18), ("alonso", 2023, 19)]
    h = _hist(e0, vals)
    assert len(h) == len(e0)
    for drv, season, rnd in cases:
        rid = int(ids[(ids["season"] == season) & (ids["round"] == rnd)]["race_id"].iloc[0])
        row = vals[(vals["driverId"] == drv) & (vals["race_id"] == rid)]
        assert len(row) == 1 and row["cons_c"].notna().all(), (drv, season, rnd)
        assert e0[(e0["driverId"] == drv) & (e0["race_id"] == rid)].empty
        after = e0[(e0["driverId"] == drv) & (e0["race_id"] > rid)].iloc[0]
        j = after.name
        with_it = _brute(vals, drv, int(after["race_id"]), CFG["halflife"])
        without = _brute(vals[~((vals["driverId"] == drv) & (vals["race_id"] == rid))], drv, int(after["race_id"]), CFG["halflife"])
        assert h.loc[j] == pytest.approx(with_it, abs=1e-12) and abs(with_it - without) > 1e-6, (drv, j)


@needs_real
def test_history_column_requires_unique_entries_and_is_row_order_free(world):
    e0, vals = world["entries0"], world["vals"]
    base = _hist(e0, vals)
    perm = e0.sample(frac=1.0, random_state=4)
    got = _hist(perm, vals)
    assert np.array_equal(got.to_numpy(), base.loc[perm.index].to_numpy(), equal_nan=True)
    with pytest.raises(AssertionError):
        _hist(pd.concat([e0, e0.iloc[:3]], ignore_index=True), vals)
    assert not e0.duplicated(["race_id", "driverId"]).any()


@needs_real
def test_races_beyond_the_last_valued_race_are_nan_and_dev_is_complete(featured, world):
    ent, _ = featured
    e0, vals = world["entries0"], world["vals"]
    last = int(vals["race_id"].max())
    assert ent.loc[ent["race_id"] > last, "cons_x"].isna().all() and ent.loc[ent["race_id"] > last, "season"].min() >= 2024
    assert ent.loc[ent["race_id"] <= last, "cons_x"].notna().all()
    assert len(ent) == len(e0) and ent.drop(columns=["cons_x", "cons_tm_x"]).equals(e0)     # nothing else in the frame changed
    dev = ent[ent["season"].between(2021, 2023)]
    assert len(dev) == 1311 and dev["race_id"].nunique() == 66 and dev["cons_x"].between(-5, 15).all()
    # the held-back races never receive a value: seasons >= 2024 are NaN, so no later-season data can enter a model by accident
    assert ent.loc[ent["season"] >= 2024, ["cons_x", "cons_tm_x"]].isna().all().all()


# ====================================================================================== (3) scaling constant
@needs_real
def test_scale_constant_is_the_sd_of_the_history_over_seasons_up_to_2020(featured, world):
    ent, info = featured
    e0, vals = world["entries0"], world["vals"]
    assert CFG["scale_seasons_max"] == 2020
    sd = info["cons_x"]["scale_sd"]
    assert sd == pytest.approx(0.0718, abs=5e-5)
    raw = _hist(e0, vals)
    old = (e0["season"] <= 2020).to_numpy()
    assert sd == pytest.approx(float(raw[old].std()), abs=1e-15)
    # recompute from a frame that contains ONLY seasons <= 2020 (entries and values): identical, so no later data is involved
    e_old = e0[e0["season"] <= 2020].reset_index(drop=True)
    v_old = vals[vals["season"] <= 2020]
    assert float(_hist(e_old, v_old).std()) == pytest.approx(sd, abs=1e-15)
    # a dev-included SD would differ (the constant is not the all-season SD)
    allsd = float(raw[(e0["season"] <= 2023).to_numpy()].std())
    assert abs(allsd - sd) > 1e-4
    # feature = history / sd
    np.testing.assert_allclose(ent["cons_x"].to_numpy(float), (raw / sd).to_numpy(float), rtol=0, atol=1e-12, equal_nan=True)
    # value stored in the result file
    stored = json.loads((RES / "phase2_feature_info.json").read_text(encoding="utf-8"))
    assert stored["cons_x"]["scale_sd"] == pytest.approx(sd, abs=1e-15)
    assert stored["truncation_check_max_abs_diff"] == 0.0
    # the teammate-relative post hoc column is scaled the same way
    tm_sd = info["cons_tm_x"]["scale_sd"]
    assert stored["cons_tm_x"]["scale_sd"] == pytest.approx(tm_sd, abs=1e-15)


def test_scaling_code_uses_only_the_declared_season_cut():
    src = (METH / "run_phase2.py").read_text(encoding="utf-8")
    assert 'entries["season"] <= cfg["scale_seasons_max"]' in src
    assert "scale_seasons_max: 2020" in (METH / "config_phase2.yaml").read_text(encoding="utf-8")


# ====================================================================================== (4) reproduction
def _stored_ref(label):
    return U.stored_run(CFG["plan1_leaderboard"], label)


@needs_runs
@pytest.mark.parametrize("plan1,new", [("latent_111", "latent_111"), ("gbm_100", "gbm_100")])
def test_reference_reruns_match_the_stored_plan1_dev_runs(plan1, new):
    d = _dir(LABELS[new])
    old_pm, old_pr = _stored_ref(plan1)
    pm, pr = pd.read_csv(d / "per_race_metrics.csv"), pd.read_parquet(d / "predictions.parquet")
    diff, n = U.max_diff(pm, old_pm)
    assert n == 66 and diff < 1e-12, diff
    sdiff, ne = U.score_diff(pr, old_pr)
    assert ne == 1311 and sdiff <= 1e-12, sdiff
    assert list(pm["race_id"]) == list(old_pm["race_id"])
    assert not pr.duplicated(["race_id", "driverId"]).any()
    rep = json.loads((RES / "phase2_reproduction_check.json").read_text(encoding="utf-8"))[plan1]
    assert rep["races"] == 66 and rep["entries"] == 1311
    assert rep["max_abs_diff_scores"] == pytest.approx(sdiff, abs=1e-15) and rep["max_abs_diff_per_race_metrics"] == pytest.approx(diff, abs=1e-15)
    cfg = load_yaml(d / "config.yaml")
    frozen = load_yaml(ROOT / CFG["frozen_path"])
    want = frozen[CFG["latent_ref" if plan1 == "latent_111" else "gbm_ref"]["key"]][CFG["latent_ref" if plan1 == "latent_111" else "gbm_ref"]["cell"]]
    for k, v in want.items():
        assert cfg[k] == v, k
    assert "x_cols" not in cfg and "extra_cols" not in cfg


@needs_runs
@needs_real
@pytest.mark.parametrize("which", ["latent_111", "gbm_100"])
def test_reference_with_a_zero_feature_column_reproduces_the_stored_reference_exactly(featured, which):
    """Switch off: the same model with the new column present but identically zero equals the stored reference scores on all 1311
    dev entries, bit for bit (latent: a zero column has zero gradient and a ridge keeps its weight at 0; gbm: a constant column
    cannot split). The FULL entries frame is used on purpose: the latent scores move by ~5e-4 when the frame handed to
    build_model is cut after the last dev race (different parameter universe for the optimiser), so a truncated frame is not a
    reproduction of the stored run."""
    ent, _ = featured
    frozen = load_yaml(ROOT / CFG["frozen_path"])
    run = ent.assign(zero_x=0.0)
    ref = pd.read_parquet(_dir(LABELS[which]) / "predictions.parquet")
    if which == "latent_111":
        cfg = dict(frozen[CFG["latent_ref"]["key"]][CFG["latent_ref"]["cell"]], x_cols=["zero_x"], lam_x=CFG["lam_x"])
    else:
        cfg = dict(frozen[CFG["gbm_ref"]["key"]][CFG["gbm_ref"]["cell"]], extra_cols=["zero_x"])
    p, per = walk_forward(run, build_model(cfg, run), "dev", refit_every=1)
    m = p.merge(ref, on=["race_id", "driverId"], suffixes=("_n", "_o"))
    assert len(m) == len(ref) == 1311 and len(per) == 66
    assert float((m["score_n"] - m["score_o"]).abs().max()) == 0.0
    old_pm = pd.read_csv(_dir(LABELS[which]) / "per_race_metrics.csv")
    assert U.max_diff(per, old_pm)[0] < 1e-12            # csv round trip only


@needs_runs
@needs_real
@pytest.mark.parametrize("which,key", [("latent_111", "x_cols"), ("gbm_100", "extra_cols")])
def test_candidate_runs_equal_a_fresh_run_with_the_recomputed_feature(featured, which, key):
    """The stored candidate runs are reproduced from the recomputed cons_x / cons_tm_x columns (all 66 dev races, 1311 entries)."""
    ent, _ = featured
    frozen = load_yaml(ROOT / CFG["frozen_path"])
    ref = CFG["latent_ref"] if which == "latent_111" else CFG["gbm_ref"]
    base = dict(frozen[ref["key"]][ref["cell"]])
    for comp, col, label in (("cons", "cons_x", f"m20p2_cons_{which}"), ("tm", "cons_tm_x", f"m20p2_posthoc_tm_{which}")):
        cfg = dict(base, **{key: [col]}, **({"lam_x": CFG["lam_x"]} if which == "latent_111" else {}))
        p, _ = walk_forward(ent, build_model(cfg, ent), "dev", refit_every=1)
        stored = pd.read_parquet(_dir(label) / "predictions.parquet")
        m = p.merge(stored, on=["race_id", "driverId"], suffixes=("_n", "_o"))
        assert len(m) == 1311
        assert float((m["score_n"] - m["score_o"]).abs().max()) <= 1e-9, (label, float((m["score_n"] - m["score_o"]).abs().max()))
        sc = load_yaml(_dir(label) / "config.yaml")
        assert sc[key] == [col] and (sc["lam_x"] == 1.0 if which == "latent_111" else "lam_x" not in sc)
        if comp == "cons":
            assert float((m["score_n"] - m["score_o"]).abs().max()) <= 1e-9
    # control: the feature matters (candidate scores are not the reference scores)
    r = pd.read_parquet(_dir(LABELS[which]) / "predictions.parquet")
    c = pd.read_parquet(_dir(f"m20p2_cons_{which}") / "predictions.parquet").merge(r, on=["race_id", "driverId"], suffixes=("_c", "_r"))
    assert float((c["score_c"] - c["score_r"]).abs().max()) > 1e-3


# ====================================================================================== (5) paired gains and attribution rows
def _table(label: str) -> pd.DataFrame:
    pr = pd.read_parquet(_dir(label) / "predictions.parquet")
    return A.race_table(U.outcome_free(pr, CFG["tie_break"]))


PAIRS = [("M20_consistency", "latent_111"), ("M20_consistency", "gbm_100"),
         ("M20_consistency_tm", "latent_111"), ("M20_consistency_tm", "gbm_100")]


@pytest.fixture(scope="module")
def tabs():
    return {k: _table(v) for k, v in LABELS.items()}


@needs_runs
def test_stored_outcome_free_tables_and_per_race_metrics_agree_with_predictions(tabs):
    for k, label in LABELS.items():
        d = _dir(label)
        pr = pd.read_parquet(d / "predictions.parquet")
        assert pr["season"].between(2021, 2023).all() and pr["race_id"].nunique() == 66 and len(pr) == 1311
        stored = pd.read_csv(d / "outcome_free_table.csv")
        pd.testing.assert_frame_equal(stored.reset_index(drop=True), tabs[k].reset_index(drop=True), check_exact=False, rtol=0, atol=1e-12)
        pm = pd.read_csv(d / "per_race_metrics.csv").set_index("race_id")
        t = tabs[k].set_index("race_id")
        shared = [c for c in pm.columns if c in t.columns and c not in ("n", "season")]
        assert {"spearman", "mae", "exact", "logloss_points"} <= set(shared)
        for c in shared:                                # harness metrics (row-order ties) = outcome-free metrics (driverId ties)
            assert pm[c].to_numpy(float) == pytest.approx(t.loc[pm.index, c].to_numpy(float), abs=1e-9, nan_ok=True), (k, c)
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        for c in ("spearman", "mae", "exact"):
            assert m[c] == pytest.approx(float(pm[c].mean()), abs=1e-12)


@needs_runs
def test_paired_gains_csv_rederived_with_the_shared_harness(tabs):
    nb = CFG["bootstrap"]
    assert nb == {"n_boot": 10000, "seed": 0}
    full = pd.read_csv(RES / "phase2_paired_gains_dev.csv")
    assert set(map(tuple, full[["component", "reference"]].drop_duplicates().to_numpy())) == set(PAIRS)
    for comp, ref in PAIRS:
        g = A.paired_gain(tabs[ref], tabs[f"{comp}_{ref}"], n_boot=10_000, seed=0).set_index("metric")
        sub = full[(full["component"] == comp) & (full["reference"] == ref)].set_index("metric")
        assert set(sub.index) == set(g.index) and len(g) == 12
        for m in g.index:
            for c in ("gain", "lo", "hi", "ref_mean", "cand_mean"):
                assert sub.loc[m, c] == pytest.approx(g.loc[m, c], abs=1e-9, nan_ok=True), (comp, ref, m, c)
            assert sub.loc[m, "n"] == g.loc[m, "n"]
            assert sub.loc[m, "verdict"] == A.verdict(g.loc[m])
    # the verdict file is the same rows restricted to the verdict metrics
    v = pd.read_csv(RES / "phase2_verdicts.csv")
    assert set(v["metric"]) == set(R2.VERDICT_METRICS)
    for _, r in v.iterrows():
        s = full[(full["component"] == r["component"]) & (full["reference"] == r["reference"]) & (full["metric"] == r["metric"])].iloc[0]
        assert r["gain"] == pytest.approx(s["gain"], abs=1e-12) and r["verdict"] == s["verdict"]
        assert bool(r["primary"]) == (r["metric"] in ("spearman", "mae"))


@needs_runs
def test_attribution_rows_match_rederived_gains_and_are_the_only_two_m20_rows(tabs):
    att = pd.read_csv(ROOT / "results/plan3/attribution.csv")
    rows = att[att["component"].str.startswith("M20")]
    assert len(rows) == 2 and set(rows["component"]) == {"M20_consistency"}
    assert set(rows["reference"]) == {"latent_111", "gbm_100"}
    assert not att["component"].str.contains("M20_consistency_tm").any()          # post hoc variant gets no attribution row
    assert not att.duplicated(["component", "reference"]).any()
    for _, row in rows.iterrows():
        ref = row["reference"]
        g = A.paired_gain(tabs[ref], tabs[f"M20_consistency_{ref}"], n_boot=10_000, seed=0).set_index("metric")
        for m in g.index:
            for c in ("gain", "lo", "hi"):
                stored = row[f"{c}_{m}"]
                if pd.isna(stored) and pd.isna(g.loc[m, c]):
                    continue
                assert stored == pytest.approx(g.loc[m, c], abs=1e-9), (ref, m, c)
        assert row["verdict_metric"] == "spearman" == CFG["verdict_metric"] and row["n_races"] == 66 and row["declared"] == "yes"
        assert row["verdict"] == A.verdict(g.loc["spearman"])
        assert row["run_id"] == _dir(f"m20p2_cons_{ref}").name
        assert pd.isna(row["loo_gain"]) and pd.isna(row["fresh_holdout_gain"])           # nothing on the fresh holdout
    # per-season file: 4 pairs x 3 seasons x 6 metrics, seasons are dev seasons only
    bs = pd.read_csv(RES / "phase2_paired_gains_by_season.csv")
    assert set(bs["season"]) == {2021, 2022, 2023} and len(bs) == 4 * 3 * 6
    comp, ref = PAIRS[0]
    for s in (2021, 2022, 2023):
        a, b = tabs[ref][tabs[ref]["season"] == s], tabs[f"{comp}_{ref}"][tabs[f"{comp}_{ref}"]["season"] == s]
        g = A.paired_gain(a, b, ["spearman", "mae", "exact"], n_boot=10_000, seed=0).set_index("metric")
        sub = bs[(bs["component"] == comp) & (bs["reference"] == ref) & (bs["season"] == s)].set_index("metric")
        for m in g.index:
            assert sub.loc[m, "gain"] == pytest.approx(g.loc[m, "gain"], abs=1e-9)
            assert sub.loc[m, "lo"] == pytest.approx(g.loc[m, "lo"], abs=1e-9) and sub.loc[m, "hi"] == pytest.approx(g.loc[m, "hi"], abs=1e-9)


@needs_runs
def test_no_gain_is_called_an_improvement_inside_the_noise(tabs):
    """README states 'no measurable difference' on rho, MAE, exact: re-check that every primary CI of the declared component contains 0."""
    full = pd.read_csv(RES / "phase2_paired_gains_dev.csv")
    d = full[(full["component"] == "M20_consistency") & full["metric"].isin(["spearman", "mae", "exact"])]
    assert len(d) == 6
    assert ((d["lo"] <= 0) & (d["hi"] >= 0)).all(), d[["reference", "metric", "gain", "lo", "hi"]]
    assert (d["verdict"] == "no measurable difference").all()


# ====================================================================================== (6) shared files, append-only
def _pinned(rel: str) -> str:
    try:
        out = subprocess.check_output(["git", "show", f"{PHASE1_COMMIT}:{rel}"], cwd=ROOT)
    except Exception as e:                                             # pragma: no cover
        pytest.skip(f"git show failed: {e}")
    return out.decode("utf-8").replace(chr(13) + chr(10), chr(10))


def _current(rel: str) -> str:
    return (ROOT / rel).read_bytes().decode("utf-8").replace(chr(13) + chr(10), chr(10))


@pytest.mark.parametrize("rel,prefix,n_new", [("results/plan3/attribution.csv", "M20_consistency,", 2),
                                              ("results/plan3/leaderboard.csv", "M20,", 6)])
def test_shared_files_head_lines_byte_identical_and_m20_adds_exactly_its_rows(rel, prefix, n_new):
    """The file at the phase-1 commit is a byte prefix of the current file. Of the appended lines exactly n_new belong to M20
    phase 2; M16 (and other later) rows may be appended before or after them but must not use the M20 names."""
    old, cur = _pinned(rel), _current(rel)
    assert "M20" not in old
    assert cur.startswith(old), "an existing line was rewritten"
    extra = cur[len(old):]
    assert extra == "" or extra.endswith("\n")
    lines = extra.splitlines()
    mine = [ln for ln in lines if ln.startswith(prefix)]
    assert len(mine) == n_new, (len(mine), n_new)
    assert not [ln for ln in lines if "M20" in ln.split(",")[0] and ln not in mine], "another M20-named row was appended"
    # no pre-existing row was changed in meaning: parsed frames agree on the old rows, columns unchanged
    o, c = pd.read_csv(StringIO(old), dtype=str), pd.read_csv(StringIO(cur), dtype=str)
    assert list(o.columns) == list(c.columns)
    pd.testing.assert_frame_equal(c.iloc[:len(o)].reset_index(drop=True), o.reset_index(drop=True), check_exact=True)
    assert len(c) == len(o) + len(lines)


@needs_runs
def test_leaderboard_m20_rows_match_their_run_folders():
    lb = pd.read_csv(ROOT / "results/plan3/leaderboard.csv", dtype={"cell": str})
    rows = lb[lb["method"] == "M20"]
    assert len(rows) == 6 and set(rows["label"]) == set(LABELS.values()) and (rows["split"] == "dev").all()
    assert not lb.duplicated("run_id").any()
    for _, r in rows.iterrows():
        d = ROOT / r["run_dir"].replace("\\", "/")
        assert d.exists() and d.name == r["run_id"] and d.name.endswith(f"_{r['label']}_dev")
        pm = pd.read_csv(d / "per_race_metrics.csv")
        assert r["n_races"] == len(pm) == 66
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        for c in ("spearman", "mae", "exact", "logloss_points"):
            assert float(r[c]) == pytest.approx(m[c], abs=1e-12), (r["run_id"], c)
            assert m[c] == pytest.approx(float(pm[c].mean()), abs=1e-12)
        for f in ("config.yaml", "metrics.json", "per_race_metrics.csv", "predictions.parquet"):
            assert (d / f).exists(), (d.name, f)
        assert load_yaml(d / "config.yaml").get("git_commit")
    assert not (RES / "_lb_tmp.csv").exists()
    for rel in ("results/leaderboard.csv", "results/plan2/leaderboard.csv"):
        assert not (pd.read_csv(ROOT / rel, dtype=str)["method"] == "M20").any(), rel


# ====================================================================================== (7) split discipline
def test_phase2_code_and_config_touch_no_test_season_or_holdout_and_tune_nothing():
    for f in ("run_phase2.py", "config_phase2.yaml"):
        txt = (METH / f).read_text(encoding="utf-8")
        assert not re.search(r"\b202[4-9]\b", txt), f
        assert "holdout" not in txt.lower() and "fresh" not in txt.lower(), f
    src = (METH / "run_phase2.py").read_text(encoding="utf-8")
    assert 'assert cfg["split"] == "dev"' in src
    assert CFG["split"] == "dev" and CFG["refit_every"] == 1
    # lam_x is the plan-2 value, declared not tuned; no tuning loop in the script
    assert CFG["lam_x"] == 1.0
    assert not re.search(r"for\s+\w+\s+in\s+.*(lam|grid|halflife)", src.replace("for comp, ref", "")), "unexpected sweep in run_phase2"
    # the scaling and the half-life are config values, not data-dependent beyond the declared seasons <= 2020 cut
    assert CFG["halflife"] == 50
    # phase-1 half-life of the declared definition
    assert load_yaml(ROOT / CFG["race_values_config"])["halflife_primary"] == CFG["halflife"]


@needs_real
def test_race_values_used_are_dev_and_burn_in_seasons_only(world):
    assert world["vals"]["season"].between(2014, 2023).all() and world["ent_le"]["season"].max() == 2023
    assert not world["vals"].duplicated(["season", "round", "driverId"]).any()
    assert world["vals"]["cons_c"].notna().sum() > 3000
