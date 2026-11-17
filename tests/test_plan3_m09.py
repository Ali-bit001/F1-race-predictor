"""PLAN3 R6 / M09 race-pace model (exploratory, Amendment 3 item 6): independent checks.

Sections: (1) leakage of pace_hist_x / q1off_x (truncation and scrambling of race r and later), (2) decayed means by brute force,
(3) Q1 missing stays missing and the scaling SDs come from seasons <= 2020, (4) references reproduce the stored plan-1 dev runs
(and a zero column reproduces them bit for bit), (5) paired gains and attribution rows re-derived, (6) tuning file consistency,
(7) shared files are append-only (pinned commit bbe026e), (8) split discipline and data checks.
Real-data tests skip when the git-ignored inputs are missing. M16 / M20 rows in the shared files are allowed.
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
from pitlane.features.build import PSEUDO
from pitlane.models.registry import build_model
from pitlane.plan3 import attribution as A
from pitlane.plan3 import runutil as U

METH = ROOT / "methodologies/M09_race_pace"
RES = METH / "results"
CFG = load_yaml(METH / "config.yaml")
PINNED = "bbe026e"
STAMP = "20261117-1533"
HIST, OFF = "pace_hist_x", "q1off_x"
LABELS = {"latent_111": "m09_ref_latent_111", "gbm_100": "m09_ref_gbm_100",
          "M09_car_pace_hist_latent_111": "m09_hist_latent_111", "M09_car_pace_hist_gbm_100": "m09_hist_gbm_100",
          "M09_car_q1_offset_latent_111": "m09_hist_q1off_latent_111", "M09_car_q1_offset_gbm_100": "m09_hist_q1off_gbm_100",
          "M09_car_q1_offset_alone_latent_111": "m09_q1off_alone_latent_111", "M09_car_q1_offset_alone_gbm_100": "m09_q1off_alone_gbm_100"}
COMPS = {"M09_car_pace_hist": [HIST], "M09_car_q1_offset": [HIST, OFF], "M09_car_q1_offset_alone": [OFF]}
DECLARED = {"M09_car_pace_hist": "yes", "M09_car_q1_offset": "yes", "M09_car_q1_offset_alone": "no"}
needs_runs = pytest.mark.skipif(not (RES / "paired_gains_dev.csv").exists(), reason="M09 results missing")
needs_real = pytest.mark.skipif(not ((PROCESSED / "entries2.parquet").exists() and (ROOT / CFG["pace_table"]).exists()),
                                reason="processed data / pace table missing (regenerable)")


def _load_run():
    spec = importlib.util.spec_from_file_location("m09_run_under_test", METH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_run()
COLS = ["race_id", "season", "round", "driverId", "lineage", "q1_gap", "q1_missing"]


def _dir(label: str):
    c = sorted(RES.glob(f"{STAMP}_{label}_dev"))
    assert len(c) == 1, (label, c)
    return c[0]


# ====================================================================================== fixtures
@pytest.fixture(scope="module")
def world():
    e0 = P.load_entries()
    pace = pd.read_parquet(ROOT / CFG["pace_table"])
    ent_le = e0[e0["season"] <= 2023][COLS]
    full = R.feature_columns(ent_le, pace, CFG)
    return dict(e0=e0, pace=pace, ent_le=ent_le, full=full)


@pytest.fixture(scope="module")
def featured(world):
    return R.build_features(world["e0"], world["pace"], CFG)


def _rid(world, season, rnd) -> int:
    ids = world["e0"][["season", "round", "race_id"]].drop_duplicates()
    return int(ids[(ids["season"] == season) & (ids["round"] == rnd)]["race_id"].iloc[0])


RACES = [(2021, 10), (2022, 2), (2023, 15)]       # mid 2021, early 2022 (first season of the new rules), late 2023


def _cmp(a: pd.DataFrame, b: pd.DataFrame, m: np.ndarray) -> None:
    assert (a["race_id"].to_numpy() == b["race_id"].to_numpy()).all() and (a["driverId"].to_numpy() == b["driverId"].to_numpy()).all()
    for c in (HIST, OFF):
        assert np.array_equal(a.loc[m, c].to_numpy(float), b.loc[m, c].to_numpy(float), equal_nan=True), c


# ====================================================================================== (1) leakage
@needs_real
@pytest.mark.parametrize("season,rnd", RACES)
def test_features_of_race_r_unchanged_when_race_r_and_later_are_blanked(world, season, rnd):
    """Drop every pace row and blank every Q1 time of race r and later: features of every entry of race <= r (race r included) are
    bit for bit the same. (run.leak_check does the same at one race, using nanmax.)"""
    r = _rid(world, season, rnd)
    ent, pace, full = world["ent_le"], world["pace"], world["full"]
    keys = ent[["season", "round", "race_id"]].drop_duplicates()
    pc = pace.merge(keys, on=["season", "round"], how="left")
    pace_cut = pc[pc["race_id"] < r].drop(columns="race_id")
    late = ent["race_id"] >= r
    ent_cut = ent.assign(q1_gap=ent["q1_gap"].where(~late), q1_missing=ent["q1_missing"].where(~late, 1))
    cut = R.feature_columns(ent_cut, pace_cut, CFG)
    m = (full["race_id"] <= r).to_numpy()
    _cmp(full, cut, m)
    rm = (full["race_id"] == r).to_numpy()
    assert rm.sum() >= 18 and full.loc[rm, [HIST]].notna().all().all()     # race r itself is non-vacuous
    assert full.loc[rm, OFF].notna().mean() > 0.8


@needs_real
@pytest.mark.parametrize("season,rnd", RACES)
def test_features_of_race_r_unchanged_when_race_r_and_later_are_scrambled(world, season, rnd):
    """Replace pace gaps and Q1 times of race r and later by noise (values permuted across drivers plus a shift; random Q1 missing
    flags). Race <= r features unchanged; control: scrambling an earlier race moves later entries only."""
    r = _rid(world, season, rnd)
    ent, pace, full = world["ent_le"], world["pace"], world["full"]
    rng = np.random.default_rng(r)
    pace_s = pace.merge(ent[["season", "round", "race_id"]].drop_duplicates(), on=["season", "round"], how="left")
    lp = (pace_s["race_id"] >= r).to_numpy()
    vals = pace_s.loc[lp, "med_gap_pct"].to_numpy(float)
    pace_s.loc[lp, "med_gap_pct"] = rng.permutation(vals) + rng.normal(0, 3.0, lp.sum())
    pace_s = pace_s.drop(columns="race_id")
    late = (ent["race_id"] >= r).to_numpy()
    ent_s = ent.copy()
    ent_s.loc[late, "q1_gap"] = rng.uniform(0, 0.1, late.sum())
    ent_s.loc[late, "q1_missing"] = rng.integers(0, 2, late.sum())
    got = R.feature_columns(ent_s, pace_s, CFG)
    _cmp(full, got, (full["race_id"] <= r).to_numpy())
    # control (power): scramble the race 3 before r instead; entries of races <= r-3 unchanged, later ones move
    r0 = r - 3
    pace_c = pace.merge(ent[["season", "round", "race_id"]].drop_duplicates(), on=["season", "round"], how="left")
    sel = (pace_c["race_id"] == r0).to_numpy()
    assert sel.sum() > 15
    pace_c.loc[sel, "med_gap_pct"] += rng.normal(0, 3.0, sel.sum())
    c = R.feature_columns(ent, pace_c.drop(columns="race_id"), CFG)
    _cmp(full, c, (full["race_id"] <= r0).to_numpy())
    later = ((full["race_id"] > r0) & (full["race_id"] <= r0 + 10)).to_numpy()
    assert (full.loc[later, HIST].to_numpy() != c.loc[later, HIST].to_numpy()).mean() > 0.3
    assert (full.loc[later, OFF].to_numpy() != c.loc[later, OFF].to_numpy()).mean() > 0.3


@needs_real
def test_q1_of_race_r_alone_does_not_move_hist_and_pace_of_race_r_alone_does_not_move_its_own_features(world):
    r = _rid(world, 2022, 2)
    ent, pace, full = world["ent_le"], world["pace"], world["full"]
    ent_s = ent.copy()
    late = (ent["race_id"] >= r).to_numpy()
    ent_s.loc[late, "q1_gap"] = 0.0
    ent_s.loc[late, "q1_missing"] = 0
    _cmp(full, R.feature_columns(ent_s, pace, CFG), (full["race_id"] <= r).to_numpy())
    # features after r do move when Q1 of r changes (non-vacuity)
    got = R.feature_columns(ent_s, pace, CFG)
    nxt = (full["race_id"] == r + 1).to_numpy()
    assert (full.loc[nxt, OFF].to_numpy() != got.loc[nxt, OFF].to_numpy()).any()


# ====================================================================================== (2) brute force
def _brute(world, lineage: str, r: int, kind: str, hl: float = 50.0) -> float:
    """Independent re-derivation. Per earlier race s of the lineage: car value = mean over the lineage's entries of the race gap
    minus the field median of the race gap (kind 'pace'), or of (race gap - 100 * Q1 gap, Q1 real) minus the field median of that
    offset (kind 'off'). Result: sum(w v) / (PSEUDO + sum(w)), w = 0.5 ** ((r - s) / hl); NaN without an earlier valid race."""
    e, pace = world["ent_le"], world["pace"]
    d = e.merge(pace[["season", "round", "driverId", "med_gap_pct"]], on=["season", "round", "driverId"], how="left")
    d["g"] = d["med_gap_pct"]
    if kind == "off":
        d["g"] = d["med_gap_pct"] - 100.0 * d["q1_gap"].where(d["q1_missing"] == 0)
    d["c"] = d["g"] - d.groupby("race_id")["g"].transform("median")
    car = d[(d["lineage"] == lineage) & (d["race_id"] < r)].groupby("race_id")["c"].mean().dropna()
    if car.empty:
        return np.nan
    w = 0.5 ** ((r - car.index.to_numpy(float)) / hl)
    return float((w * car.to_numpy()).sum() / (PSEUDO + w.sum()))


@needs_real
def test_halflife_and_ladder_columns_declared():
    assert CFG["halflife"] == 50 and CFG["pace_col"] == "med_gap_pct" and CFG["min_prior_obs"] == 1
    assert R.COMPS == COMPS


@needs_real
def test_decayed_means_equal_brute_force_on_a_sample_of_entries(world):
    full, ent = world["full"], world["ent_le"]
    assert len(full) == len(ent) and not full.duplicated(["race_id", "driverId"]).any()
    f = full.merge(ent[["race_id", "driverId", "lineage"]], on=["race_id", "driverId"])
    rng = np.random.default_rng(0)
    dev = f[f["season"].between(2021, 2023)]
    pick = list(rng.choice(dev.index.to_numpy(), 40, replace=False)) + list(rng.choice(f[f["season"] <= 2017].index.to_numpy(), 12, replace=False))
    # include early-2022 entries (rule change) and entries with no history
    pick += list(dev[dev["race_id"] == _rid(world, 2022, 2)].index[:6])
    for i in pick:
        row = f.loc[i]
        for col, kind in ((HIST, "pace"), (OFF, "off")):
            want = _brute(world, row["lineage"], int(row["race_id"]), kind)
            got = float(row[col])
            if np.isnan(want):
                assert np.isnan(got), (i, col)
            else:
                assert got == pytest.approx(want, abs=1e-10), (i, col, row[["race_id", "driverId"]].tolist())


@needs_real
def test_first_race_and_new_lineages_are_missing_not_zero(world):
    full, ent = world["full"], world["ent_le"]
    first = (full["race_id"] == full["race_id"].min()).to_numpy()
    assert full.loc[first, [HIST, OFF]].isna().all().all()
    # a lineage's first appearance has no earlier observation
    f = full.merge(ent[["race_id", "driverId", "lineage"]], on=["race_id", "driverId"])
    firstseen = f.groupby("lineage")["race_id"].transform("min") == f["race_id"]
    assert firstseen.sum() > 10 and f.loc[firstseen, [HIST, OFF]].isna().all().all()
    assert f.loc[~firstseen & (f["race_id"] > 40), HIST].notna().all()


# ====================================================================================== (3) Q1 missing, scaling
@needs_real
def test_q1_missing_is_left_missing_not_imputed(world):
    ent, pace, full = world["ent_le"], world["pace"], world["full"]
    miss = (ent["q1_missing"] == 1).to_numpy()
    assert miss.sum() >= 60 and np.allclose(ent.loc[miss, "q1_gap"], 0.10)          # the plan-1 stand-in value
    # the stand-in value is never read: changing it leaves every feature bit for bit identical
    alt = ent.copy()
    alt.loc[miss, "q1_gap"] = 0.5
    _cmp(full, R.feature_columns(alt, pace, CFG), np.ones(len(full), bool))
    # ... while flagging those rows as real Q1 times (i.e. imputing 0.10) does change q1off_x, and never hist
    imp = ent.assign(q1_missing=0)
    got = R.feature_columns(imp, pace, CFG)
    assert np.array_equal(full[HIST].to_numpy(float), got[HIST].to_numpy(float), equal_nan=True)
    assert (full[OFF].to_numpy() != got[OFF].to_numpy()).any()
    # brute force on a lineage-race whose most recent teammate rows had a missing Q1 is covered by _brute (q1_missing excluded)


@needs_real
def test_dev_hist_and_offset_have_no_missing_and_values_are_in_range(featured):
    ent, info = featured
    dev = ent[ent["season"].between(2021, 2023)]
    assert len(dev) == 1311 and dev["race_id"].nunique() == 66
    for c in (HIST, OFF):
        assert dev[c].notna().all() and dev[c].between(-5, 5).all(), c
        assert info[c]["dev_missing_share"] == 0.0
    assert ent.loc[ent["season"] >= 2024, [HIST, OFF]].isna().all().all()       # no later-season values can enter a model
    assert ent.drop(columns=[HIST, OFF]).equals(P.load_entries())


@needs_real
def test_scaling_sds_come_from_seasons_up_to_2020_only(world, featured):
    ent, info = featured
    assert CFG["scale_seasons_max"] == 2020
    sd_h, sd_o = info[HIST]["scale_sd"], info[OFF]["scale_sd"]
    assert sd_h == pytest.approx(0.9007, abs=5e-5) and sd_o == pytest.approx(0.2928, abs=5e-5)
    full, e0, ent_le = world["full"], world["e0"], world["ent_le"]
    old = full["season"] <= 2020
    assert sd_h == pytest.approx(float(full.loc[old, HIST].std()), abs=1e-15)
    assert sd_o == pytest.approx(float(full.loc[old, OFF].std()), abs=1e-15)
    # recomputed from a frame holding only seasons <= 2020 (entries and pace rows): identical
    pace = world["pace"]
    e_old = e0[e0["season"] <= 2020].reset_index(drop=True)
    _, info_old = R.build_features(e_old, pace[pace["season"] <= 2020], CFG)
    assert info_old[HIST]["scale_sd"] == pytest.approx(sd_h, abs=1e-15) and info_old[OFF]["scale_sd"] == pytest.approx(sd_o, abs=1e-15)
    # scrambling every pace gap / Q1 time of seasons >= 2021 leaves the SDs and the <= 2020 features alone
    rng = np.random.default_rng(1)
    pace_s = pace.copy()
    lt = (pace_s["season"] >= 2021).to_numpy()
    pace_s.loc[lt, "med_gap_pct"] = rng.normal(0, 5, lt.sum())
    e_s = e0.copy()
    lt2 = (e_s["season"] >= 2021).to_numpy()
    e_s.loc[lt2, "q1_gap"] = rng.uniform(0, 0.1, lt2.sum())
    ent2, info2 = R.build_features(e_s, pace_s, CFG)
    assert info2[HIST]["scale_sd"] == info[HIST]["scale_sd"] and info2[OFF]["scale_sd"] == info[OFF]["scale_sd"]
    o = (e0["season"] <= 2020).to_numpy()
    for c in (HIST, OFF):
        assert np.array_equal(ent.loc[o, c].to_numpy(), ent2.loc[o, c].to_numpy(), equal_nan=True)
    # a dev-inclusive SD would differ, so the constant is not the all-season value
    assert abs(float(full[HIST].std()) - sd_h) > 1e-3 or abs(float(full[OFF].std()) - sd_o) > 1e-3
    # feature = raw / sd, and the stored file agrees
    np.testing.assert_allclose(ent[HIST].to_numpy(float)[: len(ent_le)], (full[HIST] / sd_h).to_numpy(float), atol=1e-12, equal_nan=True)
    stored = json.loads((RES / "feature_info.json").read_text(encoding="utf-8"))
    assert stored[HIST]["scale_sd"] == pytest.approx(sd_h, abs=1e-15) and stored[OFF]["scale_sd"] == pytest.approx(sd_o, abs=1e-15)
    assert stored["truncation_check_max_abs_diff"] == 0.0


# ====================================================================================== (4) reproduction
@needs_runs
@pytest.mark.parametrize("plan1", ["latent_111", "gbm_100"])
def test_reference_reruns_match_the_stored_plan1_dev_runs(plan1):
    d = _dir(LABELS[plan1])
    old_pm, old_pr = U.stored_run(CFG["plan1_leaderboard"], plan1)
    pm, pr = pd.read_csv(d / "per_race_metrics.csv"), pd.read_parquet(d / "predictions.parquet")
    diff, n = U.max_diff(pm, old_pm)
    assert n == 66 and diff < 1e-12, diff
    sdiff, ne = U.score_diff(pr, old_pr)
    assert ne == 1311 and sdiff <= 1e-12, sdiff
    assert list(pm["race_id"]) == list(old_pm["race_id"])
    for c in old_pm.columns:                                    # per score: every per-race column agrees
        if c in pm.columns and c != "race_id":
            assert pm[c].to_numpy(float) == pytest.approx(old_pm[c].to_numpy(float), abs=1e-12, nan_ok=True), c
    rep = json.loads((RES / "reproduction_check.json").read_text(encoding="utf-8"))[plan1]
    assert rep["races"] == 66 and rep["entries"] == 1311
    assert rep["max_abs_diff_scores"] == pytest.approx(sdiff, abs=1e-15)
    cfg = load_yaml(d / "config.yaml")
    ref = CFG["latent_ref"] if plan1 == "latent_111" else CFG["gbm_ref"]
    for k, v in load_yaml(ROOT / CFG["frozen_path"])[ref["key"]][ref["cell"]].items():
        assert cfg[k] == v, k
    assert "x_cols" not in cfg and "extra_cols" not in cfg


@needs_runs
@needs_real
@pytest.mark.parametrize("which", ["latent_111", "gbm_100"])
def test_reference_with_a_zero_feature_column_reproduces_the_stored_reference_exactly(featured, which):
    """Switch off: both new columns present but identically zero give the stored reference scores bit for bit (full frame, as in M20)."""
    ent, _ = featured
    frozen = load_yaml(ROOT / CFG["frozen_path"])
    run = ent.assign(zero_x=0.0)
    ref = pd.read_parquet(_dir(LABELS[which]) / "predictions.parquet")
    if which == "latent_111":
        cfg = dict(frozen[CFG["latent_ref"]["key"]][CFG["latent_ref"]["cell"]], x_cols=["zero_x"], lam_x=1.0)
    else:
        cfg = dict(frozen[CFG["gbm_ref"]["key"]][CFG["gbm_ref"]["cell"]], extra_cols=["zero_x"])
    p, per = walk_forward(run, build_model(cfg, run), "dev", refit_every=1)
    m = p.merge(ref, on=["race_id", "driverId"], suffixes=("_n", "_o"))
    assert len(m) == len(ref) == 1311 and len(per) == 66
    assert float((m["score_n"] - m["score_o"]).abs().max()) == 0.0


@needs_runs
@needs_real
@pytest.mark.parametrize("which,key", [("latent_111", "x_cols"), ("gbm_100", "extra_cols")])
def test_stored_candidate_runs_equal_a_fresh_run_with_the_recomputed_features(featured, which, key):
    ent, _ = featured
    frozen = load_yaml(ROOT / CFG["frozen_path"])
    ref = CFG["latent_ref"] if which == "latent_111" else CFG["gbm_ref"]
    base = dict(frozen[ref["key"]][ref["cell"]])
    comp = "M09_car_q1_offset"                                           # both columns: covers both new features
    lam = json.loads((RES / "feature_info.json").read_text(encoding="utf-8"))["lam_x_chosen"][comp]
    cfg = dict(base, **{key: COMPS[comp]}, **({"lam_x": lam} if which == "latent_111" else {}))
    p, _ = walk_forward(ent, build_model(cfg, ent), "dev", refit_every=1)
    stored = pd.read_parquet(_dir(LABELS[f"{comp}_{which}"]) / "predictions.parquet")
    m = p.merge(stored, on=["race_id", "driverId"], suffixes=("_n", "_o"))
    assert len(m) == 1311
    assert float((m["score_n"] - m["score_o"]).abs().max()) <= 1e-9
    r = pd.read_parquet(_dir(LABELS[which]) / "predictions.parquet").merge(stored, on=["race_id", "driverId"], suffixes=("_r", "_c"))
    assert float((r["score_r"] - r["score_c"]).abs().max()) > 1e-3         # control: the features matter


# ====================================================================================== (5) paired gains and attribution
@pytest.fixture(scope="module")
def tabs():
    out = {}
    for k, label in LABELS.items():
        pr = pd.read_parquet(_dir(label) / "predictions.parquet")
        out[k] = A.race_table(U.outcome_free(pr, CFG["tie_break"]))
    return out


PAIRS = [(c, f) for c in COMPS for f in ("latent_111", "gbm_100")]


@needs_runs
def test_run_folders_are_complete_dev_only_and_agree_with_predictions(tabs):
    for k, label in LABELS.items():
        d = _dir(label)
        pr = pd.read_parquet(d / "predictions.parquet")
        assert pr["season"].between(2021, 2023).all() and pr["race_id"].nunique() == 66 and len(pr) == 1311
        assert not pr.duplicated(["race_id", "driverId"]).any()
        for f in ("config.yaml", "metrics.json", "per_race_metrics.csv", "predictions.parquet", "outcome_free_table.csv"):
            assert (d / f).exists(), (label, f)
        assert load_yaml(d / "config.yaml").get("git_commit")
        stored = pd.read_csv(d / "outcome_free_table.csv")
        pd.testing.assert_frame_equal(stored.reset_index(drop=True), tabs[k].reset_index(drop=True), check_exact=False, rtol=0, atol=1e-12)
        pm = pd.read_csv(d / "per_race_metrics.csv").set_index("race_id")
        t = tabs[k].set_index("race_id")
        for c in ("spearman", "mae", "exact", "logloss_points"):
            assert pm[c].to_numpy(float) == pytest.approx(t.loc[pm.index, c].to_numpy(float), abs=1e-9, nan_ok=True), (k, c)


@needs_runs
@pytest.mark.parametrize("comp,ref", PAIRS)
def test_paired_gains_rederived_from_stored_per_race_metrics(tabs, comp, ref):
    nb = CFG["bootstrap"]
    assert nb == {"n_boot": 10000, "seed": 0} and CFG["tie_break"] == "driverId"
    got = A.paired_gain(tabs[ref], tabs[f"{comp}_{ref}"], n_boot=nb["n_boot"], seed=nb["seed"])
    stored = pd.read_csv(RES / "paired_gains_dev.csv")
    st = stored[(stored["component"] == comp) & (stored["reference"] == ref)].reset_index(drop=True)
    assert len(st) == len(got) and list(st["metric"]) == list(got["metric"])
    for c in ("ref_mean", "cand_mean", "gain", "lo", "hi"):
        assert st[c].to_numpy(float) == pytest.approx(got[c].to_numpy(float), abs=1e-12, nan_ok=True), (comp, ref, c)
    assert list(st["n"]) == list(got["n"])
    assert list(st["verdict"]) == [A.verdict(r) for _, r in got.iterrows()]
    # gains are the mean of the per-race difference of the stored per_race_metrics.csv (sign: positive = better)
    a = pd.read_csv(_dir(LABELS[ref]) / "per_race_metrics.csv").set_index("race_id")
    b = pd.read_csv(_dir(LABELS[f"{comp}_{ref}"]) / "per_race_metrics.csv").set_index("race_id")
    g = got.set_index("metric")
    assert g.loc["spearman", "gain"] == pytest.approx(float((b["spearman"] - a["spearman"]).mean()), abs=1e-9)
    assert g.loc["mae", "gain"] == pytest.approx(float((a["mae"] - b["mae"]).mean()), abs=1e-9)
    assert (g["lo"] <= g["gain"] + 1e-12).all() and (g["gain"] <= g["hi"] + 1e-12).all()


@needs_runs
@pytest.mark.parametrize("ref", ["latent_111", "gbm_100"])
def test_increment_rows_over_hist_rederived(tabs, ref):
    got = A.paired_gain(tabs[f"M09_car_pace_hist_{ref}"], tabs[f"M09_car_q1_offset_{ref}"], n_boot=10000, seed=0)
    stored = pd.read_csv(RES / "paired_gains_dev.csv")
    st = stored[(stored["component"] == "M09_car_q1_offset_increment_over_hist") & (stored["reference"] == f"M09_car_pace_hist_{ref}")].reset_index(drop=True)
    assert len(st) == len(got)
    for c in ("gain", "lo", "hi"):
        assert st[c].to_numpy(float) == pytest.approx(got[c].to_numpy(float), abs=1e-12, nan_ok=True)


@needs_runs
@pytest.mark.parametrize("comp,ref", PAIRS)
def test_attribution_rows_match_rederived_gains(tabs, comp, ref):
    att = pd.read_csv(ROOT / "results/plan3/attribution.csv")
    row = att[(att["component"] == comp) & (att["reference"] == ref)]
    assert len(row) == 1, (comp, ref)
    row = row.iloc[0]
    got = A.paired_gain(tabs[ref], tabs[f"{comp}_{ref}"], n_boot=10000, seed=0).set_index("metric")
    for m in got.index:
        if f"gain_{m}" in row.index and not pd.isna(row[f"gain_{m}"]):
            for pre, col in (("gain", "gain"), ("lo", "lo"), ("hi", "hi")):
                assert float(row[f"{pre}_{m}"]) == pytest.approx(float(got.loc[m, col]), abs=1e-12), (comp, ref, m, pre)
    assert row["verdict_metric"] == "spearman" and row["verdict"] == A.verdict(got.loc["spearman"])
    assert row["declared"] == DECLARED[comp] and int(row["n_races"]) == 66
    assert row["run_id"] == _dir(LABELS[f"{comp}_{ref}"]).name
    assert pd.isna(row["loo_gain"]) and pd.isna(row["fresh_holdout_gain"])      # no holdout number invented


@needs_runs
def test_only_the_alone_diagnostic_is_marked_undeclared_and_verdicts_are_not_overstated():
    att = pd.read_csv(ROOT / "results/plan3/attribution.csv")
    m = att[att["component"].str.startswith("M09_")]
    assert len(m) == 6
    assert set(m.loc[m["declared"] == "no", "component"]) == {"M09_car_q1_offset_alone"}
    # the verdict follows the CI rule on the stored numbers
    for _, r in m.iterrows():
        want = "passes alone" if r["lo_spearman"] > 0 else ("worse" if r["hi_spearman"] < 0 else "no measurable difference")
        assert r["verdict"] == want


# ====================================================================================== (6) tuning
@needs_runs
def test_tuning_file_has_fifteen_configurations_and_the_chosen_lam_x_is_the_best_row():
    t = pd.read_csv(RES / "tuning_lam_x.csv")
    info = json.loads((RES / "feature_info.json").read_text(encoding="utf-8"))
    # 3 rungs x 4 lam_x values (latent) are in the file; the 3 LightGBM rungs have no tunable parameter: 12 + 3 = 15 <= 30
    assert len(t) == 12 and set(t["family"]) == {"latent_111"} and set(t["component"]) == set(COMPS)
    assert len(t) + len(COMPS) == info["configurations_evaluated"] == 15
    assert (t["n_races"] == 66).all() and sorted(t["lam_x"].unique()) == CFG["lam_x_grid"]
    assert not t.duplicated(["component", "lam_x"]).any() and t[["spearman", "mae", "exact"]].notna().all().all()
    for comp in COMPS:
        s = t[t["component"] == comp]
        best = float(s.loc[s["spearman"].idxmax(), "lam_x"])
        assert (s["spearman"] == s["spearman"].max()).sum() == 1                        # no tie to break by hand
        assert info["lam_x_chosen"][comp] == best, comp
        for fam in ("latent_111", "gbm_100"):
            cfg = load_yaml(_dir(LABELS[f"{comp}_{fam}"]) / "config.yaml")
            if fam == "latent_111":
                assert cfg["lam_x"] == best and cfg["x_cols"] == COMPS[comp], comp
            else:
                assert "lam_x" not in cfg and cfg["extra_cols"] == COMPS[comp] and cfg.get("x_cols") is None
        # selection used dev mean rho only (refit_every 4), not exact rate or a verdict
        assert CFG["tune_refit_every"] == 4
    # the tuned lam_x are on the grid interior or edge, but only dev data were used
    src = (METH / "run.py").read_text(encoding="utf-8")
    assert 't.loc[t["spearman"].idxmax(), "lam_x"]' in src


@needs_runs
@needs_real
@pytest.mark.slow
def test_one_tuning_cell_is_reproduced(featured):
    ent, _ = featured
    frozen = load_yaml(ROOT / CFG["frozen_path"])
    lat = frozen[CFG["latent_ref"]["key"]][CFG["latent_ref"]["cell"]]
    t = pd.read_csv(RES / "tuning_lam_x.csv")
    row = t[(t["component"] == "M09_car_pace_hist") & (t["lam_x"] == 3.0)].iloc[0]
    _, per = walk_forward(ent, build_model(dict(lat, x_cols=[HIST], lam_x=3.0), ent), "dev", refit_every=4)
    assert float(per["spearman"].mean()) == pytest.approx(float(row["spearman"]), abs=1e-9)


# ====================================================================================== (7) shared files
def _pinned(rel: str) -> str:
    try:
        out = subprocess.check_output(["git", "show", f"{PINNED}:{rel}"], cwd=ROOT)
    except Exception as e:                                             # pragma: no cover
        pytest.skip(f"git show failed: {e}")
    return out.decode("utf-8").replace("\r\n", "\n")


def _current(rel: str) -> str:
    return (ROOT / rel).read_bytes().decode("utf-8").replace("\r\n", "\n")


@pytest.mark.parametrize("rel,prefix,n_new", [("results/plan3/attribution.csv", "M09_", 6), ("results/plan3/leaderboard.csv", "M09,", 8)])  # n_new = minimum: later M09 diagnostic rows may be appended
def test_shared_files_head_byte_identical_and_m09_adds_exactly_its_rows(rel, prefix, n_new):
    old, cur = _pinned(rel), _current(rel)
    assert "M09" not in old
    assert cur.startswith(old), "an existing line was rewritten"
    extra = cur[len(old):]
    assert extra == "" or extra.endswith("\n")
    lines = extra.splitlines()
    mine = [ln for ln in lines if ln.startswith(prefix)]
    assert len(mine) >= n_new, (len(mine), n_new)
    assert not [ln for ln in lines if "M09" in ln.split(",")[0] and ln not in mine]
    o, c = pd.read_csv(StringIO(old), dtype=str), pd.read_csv(StringIO(cur), dtype=str)
    assert list(o.columns) == list(c.columns)
    pd.testing.assert_frame_equal(c.iloc[:len(o)].reset_index(drop=True), o.reset_index(drop=True), check_exact=True)
    assert len(c) == len(o) + len(lines)
    key = "component" if "component" in c.columns else "label"
    names = c.iloc[len(o):]
    assert not names.duplicated(["component", "reference"] if key == "component" else ["run_id"]).any()   # no row reuses an earlier name
    # other packages may append (M16 rows, M20 rows); nothing else may carry an M09 component/label
    others = [ln for ln in lines if ln not in mine]
    for ln in others:
        assert not ln.split(",")[0].startswith("M09")


@needs_runs
def test_leaderboard_m09_rows_match_their_run_folders():
    lb = pd.read_csv(ROOT / "results/plan3/leaderboard.csv", dtype={"cell": str})
    rows = lb[lb["method"] == "M09"]
    assert len(rows) >= 8 and set(LABELS.values()) <= set(rows["label"]) and (rows["split"] == "dev").all()
    assert not rows.duplicated("label").any()
    assert not lb.duplicated("run_id").any()
    for _, r in rows[rows["label"].isin(set(LABELS.values()))].iterrows():
        d = ROOT / r["run_dir"].replace("\\", "/")
        assert d.exists() and d.name == r["run_id"] and d.name.endswith(f"_{r['label']}_dev")
        pm = pd.read_csv(d / "per_race_metrics.csv")
        assert r["n_races"] == len(pm) == 66
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        for c in ("spearman", "mae", "exact", "logloss_points"):
            assert float(r[c]) == pytest.approx(m[c], abs=1e-12), (r["run_id"], c)
            assert m[c] == pytest.approx(float(pm[c].mean()), abs=1e-12)
    assert not (RES / "_lb_tmp.csv").exists()
    for rel in ("results/leaderboard.csv", "results/plan2/leaderboard.csv"):
        assert not (pd.read_csv(ROOT / rel, dtype=str)["method"] == "M09").any(), rel


# ====================================================================================== (8) split discipline
def test_code_and_config_touch_no_test_season_or_holdout():
    for f in ("run.py", "config.yaml"):
        txt = (METH / f).read_text(encoding="utf-8")
        assert not re.search(r"\b202[4-9]\b", txt), f
        assert "fresh" not in txt.lower(), f
    src = (METH / "run.py").read_text(encoding="utf-8")
    assert 'assert cfg["split"] == "dev"' in src and CFG["split"] == "dev"
    assert 'entries["season"] <= cfg["scale_seasons_max"]' in src and "LAST_PACE_SEASON = 2023" in src
    assert CFG["scale_seasons_max"] == 2020


@needs_real
def test_pace_table_has_no_season_after_2023_and_data_checks(world):
    pace = world["pace"]
    assert pace["season"].max() <= 2023
    assert not pace.duplicated(["season", "round", "driverId"]).any()
    p = pace["med_gap_pct"].dropna()
    assert p.between(-1, 60).all() and p.median() < 8
    e = world["ent_le"]
    assert not e.duplicated(["race_id", "driverId"]).any()
    assert e["q1_gap"].between(0, 0.5).all()
    per_season = e.groupby("season")["race_id"].nunique()
    assert per_season.loc[2021] == 22 and per_season.loc[2022] == 22 and per_season.loc[2023] == 22


# ====================================================================================== Task A: car-form control (second pass)
CTRL_STAMP = "20261117-1544"
CTRL_HEAD = "f23c2a5"
FORM = "form_x"
CTRL_LABELS = {"M09_ctrl_form": "m09_ctrl_form_latent_111", "M09_ctrl_form_hist": "m09_ctrl_form_hist_latent_111"}
needs_ctrl = pytest.mark.skipif(not (RES / "control_paired_gains_dev.csv").exists(), reason="M09 control results missing")


def _load_ctrl():
    import sys
    sys.path.insert(0, str(METH))
    try:
        spec = importlib.util.spec_from_file_location("m09_run_control_under_test", METH / "run_control.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(METH))
    return mod


RC_ = _load_ctrl()


def _cdir(label: str):
    c = sorted(RES.glob(f"{CTRL_STAMP}_{label}_dev"))
    assert len(c) == 1, (label, c)
    return c[0]


def _form(entries: pd.DataFrame) -> pd.DataFrame:
    return RC_.form_raw(entries, CFG["halflife"], CFG["min_prior_obs"])


def _same(a: pd.DataFrame, b: pd.DataFrame, m: np.ndarray) -> bool:
    assert (a["race_id"].to_numpy() == b["race_id"].to_numpy()).all() and (a["driverId"].to_numpy() == b["driverId"].to_numpy()).all()
    return np.array_equal(a.loc[m, FORM].to_numpy(float), b.loc[m, FORM].to_numpy(float), equal_nan=True)


# ---- (1) leakage and brute force
@needs_real
@pytest.mark.parametrize("season,rnd", RACES)
def test_form_x_of_race_r_unchanged_when_outcomes_from_r_onward_are_blanked_or_scrambled(world, season, rnd):
    e0 = world["e0"].reset_index(drop=True)
    r = _rid(world, season, rnd)
    base = _form(e0)
    m = (base["race_id"] <= r).to_numpy()
    rm = (base["race_id"] == r).to_numpy()
    assert base.loc[rm, FORM].notna().all() and rm.sum() >= 18            # race r itself non-vacuous
    late = (e0["race_id"] >= r).to_numpy()
    y0 = e0["y"].to_numpy(float)
    assert _same(base, _form(e0.assign(y=np.where(late, np.nan, y0))), m)            # blank
    rng = np.random.default_rng(r)
    ys = y0.copy()
    for _, ix in e0[late].groupby("race_id").indices.items():                         # scramble: permute within each late race + noise
        pos = np.flatnonzero(late)[ix]
        ys[pos] = rng.permutation(y0[pos]) + rng.normal(0, 0.3, len(pos))
    assert _same(base, _form(e0.assign(y=ys)), m)
    # outcomes of race r alone reversed: race <= r unchanged, race r+1 moves (power)
    only = (e0["race_id"] == r).to_numpy()
    yr = y0.copy()
    yr[only] = yr[only][::-1]
    alt = _form(e0.assign(y=yr))
    assert _same(base, alt, m)
    nxt = (base["race_id"] == r + 1).to_numpy()
    assert (base.loc[nxt, FORM].to_numpy() != alt.loc[nxt, FORM].to_numpy()).mean() > 0.3
    # control: noise in race r-3 moves later entries, not earlier ones
    r0 = r - 3
    s0 = (e0["race_id"] == r0).to_numpy()
    c = _form(e0.assign(y=np.where(s0, y0 + rng.normal(0, 0.3, len(e0)), y0)))
    assert _same(base, c, (base["race_id"] <= r0).to_numpy())
    later = ((base["race_id"] > r0) & (base["race_id"] <= r0 + 10)).to_numpy()
    assert (base.loc[later, FORM].to_numpy() != c.loc[later, FORM].to_numpy()).mean() > 0.3


@needs_real
def test_form_x_equals_brute_force_decayed_car_mean_of_centred_finishing_position(world):
    e0 = world["e0"].reset_index(drop=True)
    f = _form(e0)
    assert len(f) == len(e0) and not f.duplicated(["race_id", "driverId"]).any()
    f = f.merge(e0[["race_id", "driverId", "lineage", "season"]], on=["race_id", "driverId"])
    d = e0[["race_id", "lineage", "y"]].copy()
    d["yc"] = d["y"] - d.groupby("race_id")["y"].transform("mean")
    car = d.groupby(["lineage", "race_id"])["yc"].mean()
    rng = np.random.default_rng(2)
    dev = f[f["season"].between(2021, 2023)]
    pick = list(rng.choice(dev.index.to_numpy(), 40, replace=False)) + list(rng.choice(f[f["season"] <= 2017].index.to_numpy(), 12, replace=False))
    pick += list(dev[dev["race_id"] == _rid(world, 2022, 2)].index[:6])
    for i in pick:
        row = f.loc[i]
        s = car.loc[row["lineage"]]
        s = s[s.index < row["race_id"]].dropna()
        if s.empty:
            assert np.isnan(row[FORM])
            continue
        w = 0.5 ** ((row["race_id"] - s.index.to_numpy(float)) / 50.0)
        assert row[FORM] == pytest.approx(float((w * s.to_numpy()).sum() / (PSEUDO + w.sum())), abs=1e-10), (i, row["race_id"])
    assert f.loc[f["race_id"] == f["race_id"].min(), FORM].isna().all()


@needs_real
def test_form_x_scale_sd_is_seasons_up_to_2020_only_and_dev_is_complete(featured):
    ent, _ = featured
    e0 = P.load_entries().reset_index(drop=True)
    out, info = RC_.add_form(ent, CFG)
    raw = _form(e0).merge(e0[["race_id", "driverId", "season"]], on=["race_id", "driverId"])
    sd = info["scale_sd"]
    assert sd == pytest.approx(0.1783, abs=5e-5)
    assert sd == pytest.approx(float(raw.loc[raw["season"] <= 2020, FORM].std()), abs=1e-15)
    assert abs(float(raw.loc[raw["season"] <= 2023, FORM].std()) - sd) > 1e-3        # not a dev-inclusive constant
    e_old = e0[e0["season"] <= 2020].reset_index(drop=True)
    assert RC_.add_form(e_old, CFG)[1]["scale_sd"] == pytest.approx(sd, abs=1e-15)
    rng = np.random.default_rng(5)
    e_s = e0.copy()
    late = (e_s["season"] >= 2021).to_numpy()
    e_s.loc[late, "y"] = rng.uniform(0, 1, late.sum())
    out2, info2 = RC_.add_form(e_s, CFG)
    assert info2["scale_sd"] == sd
    o = (e0["season"] <= 2020).to_numpy()
    assert np.array_equal(out.loc[o, FORM].to_numpy(), out2.loc[o, FORM].to_numpy(), equal_nan=True)
    dev = out[out["season"].between(2021, 2023)]
    assert dev[FORM].notna().all() and dev[FORM].between(-6, 6).all() and len(dev) == 1311
    stored = json.loads((RES / "control_feature_info.json").read_text(encoding="utf-8"))
    assert stored["form_x"]["scale_sd"] == pytest.approx(sd, abs=1e-15) and stored["truncation_check_max_abs_diff"] == 0.0


def test_control_code_touches_no_test_season_or_holdout():
    txt = (METH / "run_control.py").read_text(encoding="utf-8")
    assert not re.search(r"\b202[4-9]\b", txt) and "fresh" not in txt.lower()
    assert 'assert cfg["split"] == "dev"' in txt and 'entries["season"] <= cfg["scale_seasons_max"]' in txt


# ---- (4) rank correlations
@needs_ctrl
@needs_real
def test_reported_within_race_rank_correlations(featured):
    from scipy.stats import spearmanr
    ent, _ = featured
    out, _ = RC_.add_form(ent, CFG)
    dev = out[out["season"].between(2021, 2023)]

    def corr(a, b):
        v = [spearmanr(g[a], g[b])[0] for _, g in dev.dropna(subset=[a, b]).groupby("race_id") if len(g) > 3]
        return float(np.mean(v))
    c1, c2 = corr(FORM, HIST), corr(FORM, "con_ew32")
    assert c1 == pytest.approx(0.969, abs=5e-4) and c2 == pytest.approx(0.988, abs=5e-4)
    stored = json.loads((RES / "control_feature_info.json").read_text(encoding="utf-8"))["rank_corr_dev"]
    assert stored["form_x~pace_hist_x"] == pytest.approx(c1, abs=1e-9) and stored["form_x~con_ew32"] == pytest.approx(c2, abs=1e-9)
    assert corr(HIST, "con_ew32") == pytest.approx(0.963, abs=5e-4) and corr(FORM, OFF) == pytest.approx(0.898, abs=5e-4)


# ---- (2) stored runs and tuning
@needs_ctrl
def test_control_tuning_unique_best_rows_and_budget():
    t = pd.read_csv(RES / "control_tuning_lam_x.csv")
    info = json.loads((RES / "control_feature_info.json").read_text(encoding="utf-8"))
    assert len(t) == 8 and set(t["component"]) == set(CTRL_LABELS) and (t["n_races"] == 66).all()
    assert sorted(t["lam_x"].unique()) == CFG["lam_x_grid"] and not t.duplicated(["component", "lam_x"]).any()
    assert 15 + len(t) == info["configurations_evaluated"] == 23 <= 30
    for comp, lam in {"M09_ctrl_form": 1.0, "M09_ctrl_form_hist": 3.0}.items():
        s = t[t["component"] == comp]
        assert (s["spearman"] == s["spearman"].max()).sum() == 1
        assert float(s.loc[s["spearman"].idxmax(), "lam_x"]) == lam == info["lam_x_chosen"][comp]
        cfg = load_yaml(_cdir(CTRL_LABELS[comp]) / "config.yaml")
        assert cfg["lam_x"] == lam and cfg["x_cols"] == RC_.RUNGS[comp] and "extra_cols" not in cfg
        assert cfg.get("git_commit")
    assert RC_.RUNGS == {"M09_ctrl_form": [FORM], "M09_ctrl_form_hist": [FORM, HIST]}


@needs_ctrl
@needs_real
@pytest.mark.parametrize("comp", ["M09_ctrl_form", "M09_ctrl_form_hist"])
def test_stored_control_runs_equal_fresh_runs(featured, comp):
    ent, _ = featured
    ent2, _ = RC_.add_form(ent, CFG)
    lat = load_yaml(ROOT / CFG["frozen_path"])[CFG["latent_ref"]["key"]][CFG["latent_ref"]["cell"]]
    lam = {"M09_ctrl_form": 1.0, "M09_ctrl_form_hist": 3.0}[comp]
    p, _ = walk_forward(ent2, build_model(dict(lat, x_cols=RC_.RUNGS[comp], lam_x=lam), ent2), "dev", refit_every=1)
    stored = pd.read_parquet(_cdir(CTRL_LABELS[comp]) / "predictions.parquet")
    m = p.merge(stored, on=["race_id", "driverId"], suffixes=("_n", "_o"))
    assert len(m) == 1311 and stored["race_id"].nunique() == 66 and stored["season"].between(2021, 2023).all()
    assert float((m["score_n"] - m["score_o"]).abs().max()) <= 1e-9
    ref = pd.read_parquet(_dir(LABELS["latent_111"]) / "predictions.parquet").merge(stored, on=["race_id", "driverId"], suffixes=("_r", "_c"))
    assert float((ref["score_r"] - ref["score_c"]).abs().max()) > 1e-3


# ---- (3) paired gains
@pytest.fixture(scope="module")
def ctabs():
    def tab(d):
        return A.race_table(U.outcome_free(pd.read_parquet(d / "predictions.parquet"), CFG["tie_break"]))
    return {"ref": tab(_dir(LABELS["latent_111"])), "hist": tab(_dir(LABELS["M09_car_pace_hist_latent_111"])),
            "C1": tab(_cdir(CTRL_LABELS["M09_ctrl_form"])), "C2": tab(_cdir(CTRL_LABELS["M09_ctrl_form_hist"]))}


CMP = {"C1_form_vs_ref": ("ref", "C1"), "hist_vs_C1": ("C1", "hist"), "C2_vs_C1": ("C1", "C2")}
SEAS_METRICS = ["spearman", "mae", "exact", "spearman_fin", "mae_fin"]


@needs_ctrl
@pytest.mark.parametrize("name", list(CMP))
def test_control_paired_gains_rederived(ctabs, name):
    a, b = CMP[name]
    got = A.paired_gain(ctabs[a], ctabs[b], n_boot=10000, seed=0)
    st = pd.read_csv(RES / "control_paired_gains_dev.csv")
    st = st[st["comparison"] == name].reset_index(drop=True)
    assert list(st["metric"]) == list(got["metric"]) and {"spearman", "spearman_fin", "mae", "exact"} <= set(got["metric"])
    for c in ("ref_mean", "cand_mean", "gain", "lo", "hi"):
        assert st[c].to_numpy(float) == pytest.approx(got[c].to_numpy(float), abs=1e-12, nan_ok=True), (name, c)
    assert list(st["verdict"]) == [A.verdict(r) for _, r in got.iterrows()]
    sb = pd.read_csv(RES / "control_paired_gains_by_season.csv")
    for s in (2021, 2022, 2023):
        gs = A.paired_gain(ctabs[a][ctabs[a]["season"] == s], ctabs[b][ctabs[b]["season"] == s], SEAS_METRICS, n_boot=10000, seed=0)
        ss = sb[(sb["comparison"] == name) & (sb["season"] == s)].reset_index(drop=True)
        assert list(ss["metric"]) == list(gs["metric"]) and (ss["n"] == 22).all()
        for c in ("gain", "lo", "hi"):
            assert ss[c].to_numpy(float) == pytest.approx(gs[c].to_numpy(float), abs=1e-12, nan_ok=True), (name, s, c)


# ---- (5) shared files
@needs_ctrl
@pytest.mark.parametrize("rel,n_m09", [("results/plan3/attribution.csv", 6), ("results/plan3/leaderboard.csv", 10)])
def test_shared_files_after_control(rel, n_m09):
    try:
        head = subprocess.check_output(["git", "show", f"{CTRL_HEAD}:{rel}"], cwd=ROOT).decode("utf-8").replace("\r\n", "\n")
    except Exception as e:                                             # pragma: no cover
        pytest.skip(str(e))
    cur = _current(rel)
    assert cur.startswith(head), "a line present at HEAD was rewritten"
    assert "M09" not in head                                           # HEAD's shared CSVs end before the M09 rows
    c = pd.read_csv(StringIO(cur), dtype=str)
    m09 = c[c.iloc[:, 0].str.startswith("M09")]
    assert len(m09) == n_m09
    if rel.endswith("leaderboard.csv"):
        diag = m09[m09["label"].str.endswith("_diagnostic")]
        assert set(diag["label"]) == {"m09_ctrl_form_latent_111_diagnostic", "m09_ctrl_form_hist_latent_111_diagnostic"}
        assert not c.duplicated("run_id").any() and not c.duplicated("label").any()
        for _, r in diag.iterrows():
            d = ROOT / r["run_dir"].replace("\\", "/")
            assert d.name == r["run_id"] and d.exists() and r["split"] == "dev" and int(r["n_races"]) == 66
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            assert float(r["spearman"]) == pytest.approx(m["spearman"], abs=1e-12)
    else:
        assert not c["component"].str.contains("ctrl|form").any()      # no control row in the attribution table
    assert not (RES / "_lb_tmp_ctrl.csv").exists()


# ---- (6) pre-statement order (weak evidence: file mtimes, which a copy or checkout resets)
@needs_ctrl
def test_prestatement_mtime_predates_control_runs_weak_evidence():
    pre = METH / "control_prestatement.md"
    t_pre = pre.stat().st_mtime
    for lab in CTRL_LABELS.values():
        assert t_pre < (_cdir(lab) / "metrics.json").stat().st_mtime, lab
    txt = pre.read_text(encoding="utf-8")
    assert "written before the runs" in txt and "NO DECISION WEIGHT" in txt.upper() and "23 <= 30" in txt


# ---- README numbers
@needs_ctrl
def test_readme_control_numbers_match_csvs(ctabs):
    readme = (METH / "README.md").read_text(encoding="utf-8")
    g = pd.read_csv(RES / "control_paired_gains_dev.csv")
    names = {"C1_form_vs_ref": "| C1 vs reference |", "hist_vs_C1": "| hist vs C1 |", "C2_vs_C1": "| C2 vs C1 |"}
    for name, prefix in names.items():
        line = next(ln for ln in readme.splitlines() if ln.startswith(prefix))
        s = g[g["comparison"] == name].set_index("metric")
        for m, nd in (("spearman", 4), ("spearman_fin", 4), ("mae", 3), ("exact", 4)):
            r = s.loc[m]
            want = f"{r['gain']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]"
            assert want in line, (name, m, want, line)
    sb = pd.read_csv(RES / "control_paired_gains_by_season.csv")
    sb = sb[sb["metric"] == "spearman"]
    gs_ = {n: [float(sb[(sb["comparison"] == n) & (sb["season"] == s)]["gain"].iloc[0]) for s in (2021, 2022, 2023)] for n in CMP}
    num = r"([+-]\d\.\d{3})(?: \[[^\]]*\])?"
    for n, label in (("C1_form_vs_ref", "C1 vs ref"), ("hist_vs_C1", "hist vs C1"), ("C2_vs_C1", "C2 vs C1")):
        mt = re.search(rf"{label} {num} / {num} / {num}", readme[readme.index("By season (rho):"):])
        assert mt, label
        # README rounds to 3 decimals; allow 6e-4 (known: C1 2022 0.012497 is printed +0.013, hist-vs-C1 2021 -0.002495 as -0.003)
        assert [float(x) for x in mt.groups()] == pytest.approx(gs_[n], abs=6e-4), (n, mt.groups(), gs_[n])
    means = {k: float(v["spearman"].mean()) for k, v in ctabs.items()}
    assert f"reference {means['ref']:.4f}, C1 {means['C1']:.4f}, hist {means['hist']:.4f}, C2 {means['C2']:.4f}" in readme
    info = json.loads((RES / "control_feature_info.json").read_text(encoding="utf-8"))
    assert f"{info['form_x']['scale_sd']:.4f}" in readme
    rc = info["rank_corr_dev"]
    assert (f"form_x vs pace_hist_x {rc['form_x~pace_hist_x']:.3f}, vs q1off_x {rc['form_x~q1off_x']:.3f}, vs con_ew32 {rc['form_x~con_ew32']:.3f}; "
            f"pace_hist_x vs con_ew32 {rc['pace_hist_x~con_ew32']:.3f}") in readme
    assert "C1 = reference + `form_x` (lam_x 1.0)" in readme and "(lam_x 3.0)" in readme
    h = g[(g["comparison"] == "hist_vs_C1") & (g["metric"] == "spearman")].iloc[0]
    c = g[(g["comparison"] == "C2_vs_C1") & (g["metric"] == "spearman")].iloc[0]
    assert h["lo"] <= 0 <= h["hi"] and c["lo"] <= 0 <= c["hi"]         # the README's "credited to car form" follows the pre-stated rule
