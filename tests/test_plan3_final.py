"""PLAN3 R14 final pre-race model and freeze: leakage, reproduction and freeze-file checks.

Sections: (1) frozen config hash against FREEZE.md and what predict.py does when it differs, (2) every switch off is latent_111 bit for
bit, (3) truncation / scrambling: predictions for race r do not change when data after r are removed or the outcomes of r and later races
are scrambled, (4) predict.py needs nothing from the race or later, (5) predict.py agrees with the harness up to the documented
universe/warm-start difference, (6) the pace column agrees with the M09 column, (7) the frozen config matches the tuned settings,
(8) attribution rows re-derived from the saved run folders.
Real-data tests skip when the git-ignored inputs (data/processed, data/interim) are missing.
"""
from __future__ import annotations

import importlib.util
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from pitlane.config import PROCESSED, ROOT, load_yaml
from pitlane.evaluation.harness import walk_forward
from pitlane.models.registry import build_model
from pitlane.plan3 import attribution as A
from pitlane.plan3 import final as F
from pitlane.plan3 import runutil as U

HERE = ROOT / "methodologies" / "P3_final"
CONFIG, FREEZE = HERE / "frozen_config.yaml", HERE / "FREEZE.md"
HAVE_DATA = (PROCESSED / "entries2.parquet").exists() and (ROOT / "data/interim/plan3/laps.parquet").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="git-ignored data not available")
OUTCOMES = ["pos", "y", "classified", "dnf", "finish_order"]


# ------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_yaml(CONFIG)


@pytest.fixture(scope="module")
def entries() -> pd.DataFrame:
    from pitlane.plan2 import pipeline2 as Q
    return Q.load_all()


@pytest.fixture(scope="module")
def pace(cfg) -> pd.DataFrame:
    return F.build_pace_table(cfg["pace_feature"])


@pytest.fixture(scope="module")
def dev_race(entries) -> int:
    ids = entries[entries["season"] == 2022]["race_id"].unique()
    return int(ids[len(ids) // 2])


def scramble_outcomes(e: pd.DataFrame, from_race: int, seed: int = 0) -> pd.DataFrame:
    """Permute the outcome columns within each race of race_id >= from_race (row order and structure unchanged)."""
    rng = np.random.default_rng(seed)
    out = e.copy()
    for _, ix in out[out["race_id"] >= from_race].groupby("race_id").indices.items():
        rows = out[out["race_id"] >= from_race].index.to_numpy()[ix]
        perm = rng.permutation(len(rows))
        for c in OUTCOMES:
            out.loc[rows, c] = out.loc[rows, c].to_numpy()[perm]
    return out


def pace_before(pace: pd.DataFrame, entries: pd.DataFrame, race_id: int) -> pd.DataFrame:
    keys = entries[["season", "round", "race_id"]].drop_duplicates()
    m = pace.merge(keys, on=["season", "round"], how="left")
    return m[m["race_id"] < race_id].drop(columns="race_id").reset_index(drop=True)


def load_predict_module():
    spec = importlib.util.spec_from_file_location("p3_predict", HERE / "predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------ (1) freeze file
def test_frozen_config_hash_matches_freeze_md():
    assert F.sha256_file(CONFIG) == F.freeze_hash(FREEZE)
    assert F.verify_freeze(CONFIG, FREEZE) == F.freeze_hash(FREEZE)


def test_hash_ignores_crlf_checkout(tmp_path):
    crlf = tmp_path / "c.yaml"
    crlf.write_bytes(CONFIG.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert F.sha256_file(crlf) == F.freeze_hash(FREEZE)


def test_edited_config_is_detected(tmp_path):
    edited = tmp_path / "frozen_config.yaml"
    edited.write_bytes(CONFIG.read_bytes() + b"# edited\n")
    with pytest.raises(RuntimeError, match="edited after the freeze"):
        F.verify_freeze(edited, FREEZE)


def test_predict_py_refuses_an_edited_config(tmp_path, monkeypatch, capsys):
    mod = load_predict_module()
    edited = tmp_path / "frozen_config.yaml"
    edited.write_bytes(CONFIG.read_bytes() + b"# edited\n")
    monkeypatch.setattr(mod, "CONFIG", edited)
    assert mod.main(["--season", "2023", "--round", "1"]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_freeze_md_content():
    txt = FREEZE.read_text(encoding="utf-8")
    assert "2026-11-18" in txt and "2026 round 20" in txt


# ------------------------------------------------------------------------------------ (7) config content
def test_frozen_config_has_everything_needed_to_predict(cfg):
    assert cfg["frozen"]["last_race_included"] == {"season": 2026, "round": 20}
    assert set(cfg["horizons"]) == {"H2", "H1"}
    assert cfg["components"]["included"] == ["M11_reverse", "M09_car_pace_hist"]
    h2 = cfg["horizons"]["H2"]
    assert h2["model"]["reverse"] is True and h2["model"]["x_cols"] == ["pace_hist_x"] and h2["uses_pace_hist"] is True
    assert set(h2["simulator"]["theta"]) == {"tau", "dnf_scale", "kappa", "p0", "s_dis", "a_vol", "a_rain"}
    assert {"stick_ref", "vol_mu", "vol_sd", "draws", "seed"} <= set(h2["simulator"])
    assert cfg["retirement_model"]["C"] == 0.02
    assert cfg["walk_forward"]["refit_every"] == 1
    assert isinstance(cfg["pace_feature"]["scale_sd"], float)
    assert "D0" in cfg["components"]["decoder"]


def test_hyperparameters_are_the_dev_tuned_ones(cfg):
    base = load_yaml(ROOT / "configs/frozen.yaml")["M03"]["111"]
    m = cfg["horizons"]["H2"]["model"]
    assert m["lam"] == base["lam"] and m["halflife"] == base["halflife"] and "lam_a" not in m      # car penalty not re-tuned (failed alone)
    assert m["lam_x"] == 3.0                                                                         # M09's dev-tuned value, kept
    h1 = load_yaml(ROOT / "configs/frozen_plan2.yaml")["H1"]["111"]
    assert cfg["horizons"]["H1"]["model"] == h1
    assert cfg["plan2_frozen_hash_inherited"] == load_yaml(ROOT / "configs/frozen_plan2.yaml")["hash"]


@needs_data
def test_scale_sd_is_a_constant_from_seasons_up_to_2020(cfg, entries, pace):
    spec = cfg["pace_feature"]
    raw = F.pace_hist_raw(entries, pace, spec)
    assert F.fit_scale_sd(raw, entries["season"].to_numpy(), 2020) == spec["scale_sd"]
    info = json.loads((ROOT / "methodologies/M09_race_pace/results/feature_info.json").read_text(encoding="utf-8"))
    assert spec["scale_sd"] == info["pace_hist_x"]["scale_sd"]


# ------------------------------------------------------------------------------------ (2) switches off
def test_all_switches_off_config_is_the_base_config():
    base = load_yaml(ROOT / "configs/frozen.yaml")["M03"]["111"]
    assert F.latent_cfg(base) == base
    assert F.latent_cfg(base, car_lam=base["lam"]) == base                 # lam equal to the base lam: nothing changes
    c = F.latent_cfg(base, car_lam=16.0)
    assert c["lam"] == 16.0 and c["lam_a"] == base["lam"]                   # the driver penalty stays at the base value


@needs_data
def test_switches_off_reproduces_latent_111_bitwise(entries):
    base = load_yaml(ROOT / "configs/frozen.yaml")["M03"]["111"]
    off = dict(F.latent_cfg(base), x_cols=[])
    p_new, _ = walk_forward(entries, build_model(off, entries), "dev", refit_every=1)
    old_pm, old_pr = U.stored_run("results/leaderboard.csv", "latent_111")
    m = p_new.merge(old_pr, on=["race_id", "driverId"], suffixes=("_n", "_o"))
    assert len(m) == len(old_pr) == 1311
    assert (m["score_n"].to_numpy() == m["score_o"].to_numpy()).all()


@needs_data
def test_leave_one_out_runs_reduce_to_the_stored_component_runs(entries, pace, cfg):
    """full minus pace_hist is M11 reverse alone and full minus reverse is M09 pace_hist alone: stored runs reproduce bit for bit."""
    base = load_yaml(ROOT / "configs/frozen.yaml")["M03"]["111"]
    e = F.add_pace_hist(entries, pace, cfg["pace_feature"])
    lb = pd.read_csv(ROOT / "results/plan3/leaderboard.csv")
    for lab, c in (("m11_latent_111_reverse", F.latent_cfg(base, reverse=True)),
                   ("m09_hist_latent_111", F.latent_cfg(base, pace_hist=True, lam_x=3.0))):
        p, _ = walk_forward(e, build_model(c, e), "dev", refit_every=1)
        row = lb[(lb["label"] == lab) & (lb["split"] == "dev")].iloc[-1]
        old = pd.read_parquet(ROOT / row["run_dir"] / "predictions.parquet")
        m = p.merge(old, on=["race_id", "driverId"], suffixes=("_n", "_o"))
        assert len(m) == 1311 and (m["score_n"].to_numpy() == m["score_o"].to_numpy()).all(), lab


# ------------------------------------------------------------------------------------ (3) truncation and scrambling
@needs_data
def test_pace_column_of_race_r_unchanged_when_laps_of_r_and_later_are_removed(entries, pace, cfg, dev_race):
    spec = cfg["pace_feature"]
    full = F.add_pace_hist(entries, pace, spec)
    cut_pace = pace_before(pace, entries, dev_race)
    assert cut_pace["season"].max() <= 2022
    ent_cut = entries[entries["race_id"] <= dev_race].reset_index(drop=True)
    cut = F.add_pace_hist(ent_cut, cut_pace, spec)
    a = full.loc[full["race_id"] <= dev_race, F.PACE_COL].to_numpy()
    b = cut[F.PACE_COL].to_numpy()
    assert np.array_equal(a, b, equal_nan=True)
    assert np.isfinite(b[(ent_cut["race_id"] == dev_race).to_numpy()]).any()


@needs_data
def test_pace_column_unchanged_when_later_races_are_scrambled(entries, pace, cfg, dev_race):
    spec = cfg["pace_feature"]
    full = F.add_pace_hist(entries, pace, spec)
    scr = F.add_pace_hist(scramble_outcomes(entries, dev_race), pace, spec)
    m = (entries["race_id"] <= dev_race).to_numpy()
    assert np.array_equal(full.loc[m, F.PACE_COL].to_numpy(), scr.loc[m, F.PACE_COL].to_numpy(), equal_nan=True)


@needs_data
def test_final_model_scores_of_race_r_ignore_outcomes_of_r_and_later(entries, pace, cfg, dev_race):
    """Same structure (every entry kept), outcomes of race r and all later races permuted: race r's score is bit for bit the same."""
    e = F.add_pace_hist(entries, pace, cfg["pace_feature"])
    mc = cfg["horizons"]["H2"]["model"]
    races = sorted(e["race_id"].unique())
    targets = [r for r in races if r < dev_race][-3:] + [dev_race]
    s1 = F.forward_scores(e, build_model(mc, e), targets)
    e2 = scramble_outcomes(e, dev_race)
    assert not e2[["pos", "y"]].equals(e[["pos", "y"]])
    s2 = F.forward_scores(e2, build_model(mc, e2), targets)
    m = (e["race_id"] == dev_race).to_numpy()
    assert np.isfinite(s1[m]).all() and (s1[m] == s2[m]).all()


@needs_data
def test_predict_race_unchanged_when_data_after_r_are_removed(entries, pace, cfg, dev_race):
    """H2 with retirement model and simulator: full table against a table cut at r (laps after r removed too)."""
    full = F.predict_race(entries, dev_race, cfg, "H2", with_sim=True, pace=pace, n_warm=3)
    cut_e = entries[entries["race_id"] <= dev_race].reset_index(drop=True)
    cut = F.predict_race(cut_e, dev_race, cfg, "H2", with_sim=True, pace=pace_before(pace, entries, dev_race), n_warm=3)
    pd.testing.assert_frame_equal(full, cut, check_exact=True)
    assert len(full) == int((entries["race_id"] == dev_race).sum())


@needs_data
def test_predict_race_unchanged_when_outcomes_of_r_and_later_are_scrambled(entries, pace, cfg, dev_race):
    base = F.predict_race(entries, dev_race, cfg, "H2", with_sim=True, pace=pace, n_warm=3)
    scr = F.predict_race(scramble_outcomes(entries, dev_race), dev_race, cfg, "H2", with_sim=True, pace=pace, n_warm=3)
    pd.testing.assert_frame_equal(base, scr, check_exact=True)


@needs_data
def test_prepare_pre_race_blanks_the_outcomes_of_the_race_only(entries, dev_race):
    p = F.prepare_pre_race(entries, dev_race)
    m = (p["race_id"] == dev_race).to_numpy()
    assert p["race_id"].max() == dev_race
    assert p.loc[m, "pos"].isna().all() and p.loc[m, "y"].isna().all() and p.loc[m, "dnf"].isna().all() and not p.loc[m, "classified"].any()
    assert p.loc[~m, "pos"].notna().all()                                   # earlier races keep their outcomes
    keep = ["grid", "grid_norm", "quali_norm", "q1_gap", "pit_start", "lineage", "driverId", "circuit_type"]
    assert p.loc[m, keep].reset_index(drop=True).equals(entries.loc[entries["race_id"] == dev_race, keep].reset_index(drop=True))


# ------------------------------------------------------------------------------------ (4) nothing needed from after the race
@needs_data
@pytest.mark.parametrize("horizon", ["H2", "H1"])
def test_predict_needs_nothing_from_the_race_itself_or_later(entries, pace, cfg, dev_race, horizon):
    mod = load_predict_module()
    cut = entries[entries["race_id"] <= dev_race].reset_index(drop=True).copy()
    m = (cut["race_id"] == dev_race).to_numpy()
    for c in ["pos", "y", "dnf", "finish_order", "points", "inc", "mech"]:      # outcome columns of the race absent
        cut[c] = cut[c].astype(float)
        cut.loc[m, c] = np.nan
    cut["classified"] = cut["classified"].astype(object)
    cut.loc[m, "classified"] = None
    cut["status"] = cut["status"].astype(object)
    cut.loc[m, "status"] = None
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["walk_forward"]["n_warm"] = 3
    season, rnd = int(cut.loc[m, "season"].iloc[0]), int(cut.loc[m, "round"].iloc[0])
    pt = pace_before(pace, entries, dev_race)                                  # no laps of the race or later
    out = F.predict_race(cut, dev_race, cfg2, horizon, with_sim=True, pace=pt, n_warm=3)
    assert len(out) == m.sum() and out["score"].notna().all()
    assert sorted(out["pred_place"]) == list(range(1, len(out) + 1))
    if horizon == "H2":
        assert np.isclose(out["p_win"].sum(), 1.0, atol=1e-9) and (out["p_points"].between(0, 1)).all()
    # the same through predict.py's own entry point, with the laps of the race and of later races still on disk
    res = mod.predict(season, rnd, [horizon], cut, cfg2, with_sim=True)[horizon]
    pd.testing.assert_frame_equal(out, res, check_exact=True)


@needs_data
def test_predict_py_function_finds_the_race_and_reports_a_missing_one(entries, cfg):
    mod = load_predict_module()
    with pytest.raises(LookupError, match="no entries rows"):
        mod.predict(2099, 1, ["H2"], entries, cfg, with_sim=False)


# ------------------------------------------------------------------------------------ (5) agreement with the harness
@needs_data
def test_predict_race_agrees_with_the_harness_up_to_the_documented_difference(entries, pace, cfg, dev_race):
    """predict_race uses the universe of rows up to r and a 15-race warm start; the harness uses the full universe. The 150-iteration
    L-BFGS is not run to convergence, so scores differ by about 1e-3 but the order is the same."""
    e = F.add_pace_hist(entries, pace, cfg["pace_feature"])
    mc = cfg["horizons"]["H2"]["model"]
    races = sorted(e["race_id"].unique())
    targets = [r for r in races if r < dev_race][-15:] + [dev_race]
    harness = F.forward_scores(e, build_model(mc, e), targets)
    m = (e["race_id"] == dev_race).to_numpy()
    out = F.predict_race(entries, dev_race, cfg, "H2", with_sim=False, pace=pace)
    d = np.abs(out["score"].to_numpy() - harness[m])
    assert d.max() < 5e-3
    assert pd.Series(out["score"].to_numpy()).rank().corr(pd.Series(harness[m]).rank()) > 0.999


# ------------------------------------------------------------------------------------ (6) pace column against M09
@needs_data
def test_pace_column_equals_the_m09_column_on_dev(entries, pace, cfg):
    from pitlane import pipeline as P
    spec = importlib.util.spec_from_file_location("m09_run", ROOT / "methodologies/M09_race_pace/run.py")
    m09 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m09)
    m09cfg = load_yaml(ROOT / "methodologies/M09_race_pace/config.yaml")
    old_pace = pd.read_parquet(ROOT / m09cfg["pace_table"])
    e1 = P.load_entries()
    old, _ = m09.build_features(e1, old_pace, m09cfg)
    new = F.add_pace_hist(entries, pace, cfg["pace_feature"])
    m = (entries["season"] <= 2023).to_numpy()
    assert (e1["race_id"].to_numpy() == entries["race_id"].to_numpy()).all()
    assert np.array_equal(old.loc[m, "pace_hist_x"].to_numpy(), new.loc[m, F.PACE_COL].to_numpy(), equal_nan=True)
    assert new.loc[~m, F.PACE_COL].notna().mean() > 0.9                        # seasons 2024-26 now have the column


@needs_data
def test_extended_pace_table_equals_the_a3_table_up_to_2023(pace):
    old = pd.read_parquet(ROOT / "results/plan3/a3_pace_table_2014_2023.parquet")
    pd.testing.assert_frame_equal(pace[pace["season"] <= 2023].reset_index(drop=True), old, check_exact=True)


# ------------------------------------------------------------------------------------ (8) attribution rows and run folders
def _rows():
    att = pd.read_csv(ROOT / "results/plan3/attribution.csv", dtype={"loo_gain": object, "fresh_holdout_gain": object})
    return att


def test_attribution_has_the_r14_rows_and_loo_gain_is_filled():
    att = _rows()
    loo = att[att["reference"].str.startswith("leave-one-out")]
    assert set(loo["component"]) == {"P3_LOO_reverse", "P3_LOO_pace_hist"}
    assert loo["loo_gain"].notna().all() and (loo["verdict"].str.startswith("retained")).all()
    assert {"P3_car_penalty_retune", "P3_final", "P3_final_S3refit"} <= set(att["component"])
    old_alone = att[(att["component"].isin(["M11_reverse", "M09_car_pace_hist"])) & (att["reference"] == "latent_111")]
    assert len(old_alone) == 2 and old_alone["loo_gain"].isna().all()           # earlier rows were not edited


def _final_run_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lb = pd.read_csv(ROOT / "results/plan3/leaderboard.csv")
    get = lambda lab: pd.read_parquet(ROOT / lb[lb["label"] == lab].iloc[-1]["run_dir"] / "predictions.parquet")
    t = lambda p: A.race_table(U.outcome_free(p, "driverId"))
    return t(get("p3_ref_latent_111")), t(get("p3_final_model")), t(get("p3_loo_r1_minus_reverse")), t(get("p3_loo_r1_minus_pace_hist"))


def test_loo_and_final_gains_are_rederived_from_saved_predictions():
    ref, fin, minus_rev, minus_pace = _final_run_tables()
    att = _rows()
    for comp, minus in (("P3_LOO_reverse", minus_rev), ("P3_LOO_pace_hist", minus_pace)):
        g = A.paired_gain(minus, fin, n_boot=10_000, seed=0).set_index("metric")
        r = att[(att["component"] == comp) & att["reference"].str.startswith("leave-one-out")].iloc[0]
        for m in ("spearman", "mae", "exact"):
            assert abs(g.loc[m, "gain"] - r[f"gain_{m}"]) < 1e-12 and abs(g.loc[m, "lo"] - r[f"lo_{m}"]) < 1e-12
    g = A.paired_gain(ref, fin, n_boot=10_000, seed=0).set_index("metric")
    r = att[(att["component"] == "P3_final") & (att["reference"] == "latent_111")].iloc[0]
    for m in ("spearman", "mae", "exact", "exact_1_3"):
        assert abs(g.loc[m, "gain"] - r[f"gain_{m}"]) < 1e-12 and abs(g.loc[m, "hi"] - r[f"hi_{m}"]) < 1e-12


def test_run_folders_have_the_required_files_and_hash():
    lb = pd.read_csv(ROOT / "results/plan3/leaderboard.csv")
    rows = lb[lb["method"] == "P3"]
    assert len(rows) >= 9
    for _, r in rows.iterrows():
        d = ROOT / r["run_dir"]
        for f in ("config.yaml", "metrics.json", "per_race_metrics.csv", "predictions.parquet"):
            assert (d / f).exists(), (r["label"], f)
        assert load_yaml(d / "config.yaml")["git_commit"]
