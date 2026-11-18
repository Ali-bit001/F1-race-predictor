"""PLAN3 R15: scoring script and helpers (data-integrity check, n = 0, Holm and minimum-n gating, write-once verdicts,
reproduction of stored runs, leakage). Synthetic tests need no data; the others skip if the git-ignored data are missing."""
from __future__ import annotations

import importlib.util
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from pitlane.config import PROCESSED, ROOT, load_yaml
from pitlane.plan3 import attribution as A
from pitlane.plan3 import final as F
from pitlane.plan3 import holdout as H

HERE = ROOT / "methodologies" / "P3_final"
HAVE_DATA = (PROCESSED / "entries2.parquet").exists() and (ROOT / "data/interim/plan3/laps.parquet").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="git-ignored data not available")
LAST = dict(race_id=3, season=2026, round=20)


def load_score():
    spec = importlib.util.spec_from_file_location("p3_score", HERE / "score.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------ synthetic data
def make_root(tmp_path):
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/interim").mkdir(parents=True)
    (tmp_path / "data/external").mkdir(parents=True)
    ent = pd.DataFrame({"race_id": np.repeat([1, 2, 3], 3), "driverId": list("abcabcabc"), "pos": [1, 2, 3, 2, 1, 3, 1, 3, 2],
                        "name": ["x", "y", None] * 3, "f": np.linspace(0, 1, 9)})
    laps = pd.DataFrame({"season": [2025] * 4 + [2026] * 4, "round": [23, 23, 24, 24, 19, 20, 20, 21], "lap": [1, 2, 1, 2, 1, 1, 2, 1],
                         "t": np.linspace(80, 90, 8)})
    ent.to_parquet(tmp_path / "data/processed/entries.parquet", index=False)
    laps.to_parquet(tmp_path / "data/interim/laps.parquet", index=False)
    (tmp_path / "data/external/ext.csv").write_text("k,v\na,1\nb,2\nc,3\n", encoding="utf-8")
    return tmp_path


def spec_of(root):
    return H.build_spec(root, ["data/processed/entries.parquet", "data/interim/laps.parquet"], ["data/external/ext.csv"], LAST)


# ------------------------------------------------------------------------------------ 1. integrity
def test_unchanged_data_passes(tmp_path):
    root = make_root(tmp_path)
    spec = spec_of(root)
    assert H.check_spec(spec, root) == []
    H.assert_unchanged(spec, root)


def test_edited_old_row_is_caught(tmp_path):
    root = make_root(tmp_path)
    spec = spec_of(root)
    e = pd.read_parquet(root / "data/processed/entries.parquet")
    e.loc[4, "pos"] = 9                                              # one cell of race 2, which existed at the freeze
    e.to_parquet(root / "data/processed/entries.parquet", index=False)
    with pytest.raises(H.DataIntegrityError, match=r"groups \['2'\]"):
        H.assert_unchanged(spec, root)


def test_edited_old_lap_row_is_caught(tmp_path):
    root = make_root(tmp_path)
    spec = spec_of(root)
    laps = pd.read_parquet(root / "data/interim/laps.parquet")
    laps.loc[1, "t"] += 0.001
    laps.to_parquet(root / "data/interim/laps.parquet", index=False)
    problems = H.check_spec(spec, root)
    assert len(problems) == 1 and "2025-23" in problems[0]


def test_removed_old_row_and_row_order_changes(tmp_path):
    root = make_root(tmp_path)
    spec = spec_of(root)
    e = pd.read_parquet(root / "data/processed/entries.parquet")
    e.sample(frac=1.0, random_state=0).to_parquet(root / "data/processed/entries.parquet", index=False)   # order only: fine
    assert H.check_spec(spec, root) == []
    e.iloc[:-1].to_parquet(root / "data/processed/entries.parquet", index=False)                              # a row removed: caught
    assert H.check_spec(spec, root)


def test_appended_rows_are_allowed_in_every_table(tmp_path):
    root = make_root(tmp_path)
    spec = spec_of(root)
    e = pd.read_parquet(root / "data/processed/entries.parquet")
    new = pd.DataFrame({"race_id": [4, 4], "driverId": ["a", "b"], "pos": [1, 2], "name": ["x", "y"], "f": [0.1, 0.2]})
    pd.concat([e, new], ignore_index=True).to_parquet(root / "data/processed/entries.parquet", index=False)
    laps = pd.read_parquet(root / "data/interim/laps.parquet")
    laps = pd.concat([laps, pd.DataFrame({"season": [2026], "round": [21], "lap": [1], "t": [85.0]})], ignore_index=True)
    laps.to_parquet(root / "data/interim/laps.parquet", index=False)
    with open(root / "data/external/ext.csv", "a", encoding="utf-8") as f:
        f.write("d,4\n")
    assert H.check_spec(spec, root) == []


def test_edited_or_removed_external_row_is_caught(tmp_path):
    root = make_root(tmp_path)
    spec = spec_of(root)
    p = root / "data/external/ext.csv"
    p.write_text("k,v\na,1\nb,9\nc,3\n", encoding="utf-8")
    problems = H.check_spec(spec, root)
    assert len(problems) == 1 and "lines [3]" in problems[0]
    p.write_text("k,v\na,1\nb,2\n", encoding="utf-8")
    assert "removed" in H.check_spec(spec, root)[0]


def test_spec_is_written_once(tmp_path):
    root = make_root(tmp_path)
    spec = spec_of(root)
    path = tmp_path / "rowhash.json"
    H.write_spec_once(spec, path)
    with pytest.raises(FileExistsError):
        H.write_spec_once(spec, path)


@needs_data
def test_real_data_rows_at_the_freeze_are_unchanged():
    spec = json.loads((HERE / "freeze_rowhash.json").read_text(encoding="utf-8"))
    assert spec["last"]["race_id"] == 271 and "procedure" in spec and "pandas" in spec["versions"]
    H.assert_unchanged(spec, ROOT)


def test_rowhash_json_covers_the_freeze_files_and_matches_freeze_md():
    spec = json.loads((HERE / "freeze_rowhash.json").read_text(encoding="utf-8"))
    want = H.freeze_data_hashes(HERE / "FREEZE.md")
    assert set(spec["file_sha256_at_generation"]) == set(want) and spec["file_sha256_at_generation"] == want
    assert set(spec["parquet"]) | set(spec["csv"]) == set(want)


# ------------------------------------------------------------------------------------ 2. n = 0 and membership
def entries_frame(rows):
    return pd.DataFrame(rows, columns=["race_id", "pos", "classified", "date"]).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_no_rows_after_the_freeze_gives_n_zero():
    sc = load_score()
    e = entries_frame([(270, 1, True, "2026-11-01"), (271, 2, True, "2026-11-08")])
    assert sc.fresh_race_ids(e, 271) == []


def test_race_without_a_result_is_not_counted():
    sc = load_score()
    e = entries_frame([(271, 1, True, "2026-11-08"), (272, np.nan, False, "2026-11-22"), (272, np.nan, False, "2026-11-22")])
    assert sc.fresh_race_ids(e, 271) == []
    e = entries_frame([(271, 1, True, "2026-11-08"), (272, 1, True, "2026-11-22"), (272, 2, True, "2026-11-22"), (273, np.nan, False, "2026-11-29")])
    assert sc.fresh_race_ids(e, 271) == [272]


def test_result_after_a_missing_one_is_an_error():
    sc = load_score()
    e = entries_frame([(272, np.nan, False, "2026-11-22"), (273, 1, True, "2026-11-29")])
    with pytest.raises(RuntimeError):
        sc.fresh_race_ids(e, 271)


def test_race_with_unclassified_rows_only_in_part_is_counted():
    sc = load_score()
    e = entries_frame([(272, 1, True, "2026-11-22"), (272, 2, False, "2026-11-22")])
    assert sc.fresh_race_ids(e, 271) == [272]


def test_n_zero_state_and_no_readings(tmp_path):
    sc = load_score()
    assert H.verdict_state(0) == H.NO_VERDICT == "no verdict (n < 20)"
    out = sc.record_readings(synthetic_tabs(0), 0, (20, 40), 0.05, dict(n_boot=200, seed=0), tmp_path, {})
    assert out == {} and list(tmp_path.iterdir()) == []


@needs_data
def test_recorded_status_matches_the_data():
    st = json.loads((ROOT / "results/plan3/holdout_status.json").read_text(encoding="utf-8"))
    sc = load_score()
    e = pd.read_parquet(PROCESSED / "entries2.parquet", columns=["race_id", "pos", "classified", "date"])
    assert st["third_look_done"] is True
    if st["n"] == 0:
        assert st["verdict_state"] == "no verdict (n < 20)" and st["fresh_race_ids"] == []
    assert st["n"] == len(st["fresh_race_ids"]) <= len(sc.fresh_race_ids(e, 271))


# ------------------------------------------------------------------------------------ 3. Holm and minimum n
def synthetic_tabs(n: int, seed: int = 0, shift: float = 0.1) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(300, 300 + n)
    ref = pd.DataFrame({"race_id": ids, "spearman": 0.6 + 0.05 * rng.standard_normal(n), "mae": 3.4 + 0.2 * rng.standard_normal(n)})

    def cand(rho_gain: float, mae_gain: float) -> pd.DataFrame:
        return pd.DataFrame({"race_id": ids, "spearman": ref["spearman"] + rho_gain + 0.002 * rng.standard_normal(n),
                             "mae": ref["mae"] - mae_gain + 0.02 * rng.standard_normal(n)})
    rev = cand(shift, shift * 10)                                      # reverse_only already has the whole effect: T3 has none
    final = rev.assign(spearman=rev["spearman"] + 0.002 * rng.standard_normal(n), mae=rev["mae"] + 0.02 * rng.standard_normal(n))
    return {"latent_111": ref, "final_minus_reverse": cand(0, 0), "reverse_only": rev, "final": final}


def test_boot_means_reproduce_the_attribution_interval():
    tabs = synthetic_tabs(25)
    for metric in ("spearman", "mae"):
        d = H.paired_diffs(tabs["latent_111"], tabs["final"], metric, A.HIGHER_BETTER[metric])
        b = H.boot_means(d, metric, 10_000, 0)
        g = A.paired_gain(tabs["latent_111"], tabs["final"], [metric], n_boot=10_000, seed=0).iloc[0]
        assert np.isclose(d.mean(), g["gain"], atol=1e-12)
        assert np.allclose(np.percentile(b, [2.5, 97.5]), [g["lo"], g["hi"]], atol=1e-12)


def test_holm_confirms_clear_gains_and_rejects_nulls():
    rng = np.random.default_rng(1)
    strong = {k: 1.0 + 0.1 * rng.standard_normal(5000) for k in ("T1", "T2", "T3")}
    assert H.holm_stepdown(strong)["confirmed"].all()
    null = {k: 0.1 * rng.standard_normal(5000) for k in ("T1", "T2", "T3")}
    assert not H.holm_stepdown(null)["confirmed"].any()
    wrong_way = {k: -1.0 + 0.1 * rng.standard_normal(5000) for k in ("T1", "T2", "T3")}
    assert not H.holm_stepdown(wrong_way)["confirmed"].any()


def test_holm_does_not_confirm_an_uncorrected_pass():
    """A test whose 95% interval excludes 0 but whose alpha/3 interval does not (it is the smallest p of three) is not confirmed."""
    rng = np.random.default_rng(2)
    z = rng.standard_normal(20_000)
    borderline = 2.2 * 0.1 + 0.1 * z                                    # mean 2.2 se: 95% interval excludes 0, 98.3% (z = 2.39) does not
    assert np.percentile(borderline, 2.5) > 0 > np.percentile(borderline, 0.83)
    boots = {"T1": borderline, "T2": 0.1 * rng.standard_normal(20_000), "T3": 0.1 * rng.standard_normal(20_000)}
    r = H.holm_stepdown(boots).set_index("test")
    assert r.loc["T1", "rank"] == 1 and abs(r.loc["T1", "alpha_k"] - 0.05 / 3) < 1e-12
    assert not r["confirmed"].any()


def test_holm_stops_at_the_first_failure():
    rng = np.random.default_rng(3)
    boots = {"T1": 1.0 + 0.1 * rng.standard_normal(5000), "T2": 0.1 * rng.standard_normal(5000), "T3": 0.9 + 0.1 * rng.standard_normal(5000)}
    r = H.holm_stepdown(boots).set_index("test")
    assert r.loc["T1", "confirmed"] and r.loc["T3", "confirmed"] and not r.loc["T2", "confirmed"]
    boots["T3"] = 0.01 + 0.1 * rng.standard_normal(5000)               # weak, ranked after the null in p: stays unconfirmed
    boots["T2"] = 0.3 + 0.1 * rng.standard_normal(5000)
    r = H.holm_stepdown(boots).set_index("test")
    assert r["confirmed"].sum() == 2 and not r.loc["T3", "confirmed"]


def test_minimum_n_gating():
    for n, lv in ((0, []), (19, []), (20, [20]), (39, [20]), (40, [20, 40]), (55, [20, 40])):
        assert H.read_levels(n) == lv
    assert H.verdict_state(19) == "no verdict (n < 20)" and H.verdict_state(20).startswith("verdict allowed")


def test_no_verdict_file_below_twenty_races(tmp_path):
    sc = load_score()
    out = sc.record_readings(synthetic_tabs(19), 19, (20, 40), 0.05, dict(n_boot=500, seed=0), tmp_path, {})
    assert out == {} and not list(tmp_path.glob("holdout_verdict_*.json"))


def test_reading_at_twenty_uses_the_first_twenty_races_and_confirms_a_large_effect(tmp_path):
    sc = load_score()
    tabs = synthetic_tabs(23, shift=0.2)
    out = sc.record_readings(tabs, 23, (20, 40), 0.05, dict(n_boot=2000, seed=0), tmp_path, {})
    assert set(out) == {20} and out[20]["n_races_read"] == 20 and out[20]["n_races_available"] == 23
    assert out[20]["race_ids"] == list(range(300, 320))
    assert out[20]["confirmed"]["T1"] and out[20]["confirmed"]["T2"] and not out[20]["confirmed"]["T3"]   # T3 has no true effect
    assert (tmp_path / "holdout_verdict_n20.json").exists() and not (tmp_path / "holdout_verdict_n40.json").exists()


def test_first_verdict_is_never_overwritten(tmp_path):
    sc = load_score()
    nb = dict(n_boot=500, seed=0)
    first = sc.record_readings(synthetic_tabs(21, shift=0.3), 21, (20, 40), 0.05, nb, tmp_path, {})
    path = tmp_path / "holdout_verdict_n20.json"
    before = path.read_bytes()
    again = sc.record_readings(synthetic_tabs(30, seed=5, shift=-0.3), 30, (20, 40), 0.05, nb, tmp_path, {})   # opposite data, more races
    assert path.read_bytes() == before and again[20] == json.loads(before.decode("utf-8")) and again[20]["confirmed"] == first[20]["confirmed"]
    assert H.write_verdict_once(path, {"x": 1}) is False and path.read_bytes() == before
    forty = sc.record_readings(synthetic_tabs(41, shift=0.3), 41, (20, 40), 0.05, nb, tmp_path, {})
    assert set(forty) == {20, 40} and path.read_bytes() == before and forty[40]["n_races_read"] == 40
    n40 = (tmp_path / "holdout_verdict_n40.json").read_bytes()
    sc.record_readings(synthetic_tabs(45, seed=7, shift=-0.3), 45, (20, 40), 0.05, nb, tmp_path, {})
    assert (tmp_path / "holdout_verdict_n40.json").read_bytes() == n40 and path.read_bytes() == before


# ------------------------------------------------------------------------------------ 4. stored runs and leakage (data needed)
@needs_data
@pytest.mark.parametrize("split", ["test_stable", "test_shift"])
def test_third_look_latent_111_equals_stored_runs(split):
    tl = pd.read_csv(ROOT / "results/plan3/third_look.csv")
    assert (tl["label"] == "third look, no decision weight").all()
    m = tl[(tl["table"] == "mean") & (tl["block"] == split) & (tl["name"] == "latent_111")].set_index("metric")["cand_mean"]
    m2 = tl[(tl["table"] == "mean") & (tl["block"] == split) & (tl["name"] == "plan2_H2")].set_index("metric")["cand_mean"]
    for lb, label in (("results/leaderboard.csv", "latent_111"), ("results/plan2/leaderboard.csv", "m06_L_r0_111")):
        d = pd.read_csv(ROOT / lb)
        row = d[(d["label"] == label) & (d["split"] == split)].iloc[-1]
        per = pd.read_csv(ROOT / str(row["run_dir"]).replace("\\", "/") / "per_race_metrics.csv")
        for k in [k for k in ("spearman", "mae", "exact", "within1", "within2") if k in per.columns and k in m.index]:
            assert abs(m[k] - per[k].mean()) < 1e-12 and abs(m2[k] - per[k].mean()) < 1e-12, (lb, k)


@needs_data
def test_stored_reproduction_numbers_in_the_r15_summary():
    st = json.loads((ROOT / "results/plan3/holdout_status.json").read_text(encoding="utf-8"))
    summ = json.loads((HERE / "results" / st["last_run"] / "metrics.json").read_text(encoding="utf-8"))
    rep = pd.DataFrame(summ["third_look"]["reproduction"])
    assert len(rep) == 10 and rep["max_abs_diff_per_race_metrics"].max() < 1e-12
    assert rep["max_abs_diff_scores"].dropna().max() == 0.0


@needs_data
def test_rerun_of_latent_111_on_test_shift_reproduces_the_stored_scores_exactly():
    sc = load_score()
    from pitlane.plan2 import pipeline2 as Q
    e = Q.load_all()
    e = e[e["race_id"] <= 271].reset_index(drop=True)
    cfgs = sc.model_configs(load_yaml(HERE / "frozen_config.yaml"), load_yaml(ROOT / "configs/frozen.yaml"))
    p, _ = sc.run_walk(e, cfgs["latent_111"], "test_shift")
    _, old_p, _ = sc.stored("results/leaderboard.csv", "latent_111", "test_shift")
    assert sc.U.score_diff(p, old_p) == (0.0, len(p))


@pytest.fixture(scope="module")
def race_2025():
    if not HAVE_DATA:
        pytest.skip("data not available")
    from pitlane.plan2 import pipeline2 as Q
    e = Q.load_all()
    ids = e[e["season"] == 2025]["race_id"].unique()
    return e, int(ids[len(ids) // 2])


def pace_before(pace: pd.DataFrame, entries: pd.DataFrame, race_id: int) -> pd.DataFrame:
    keys = entries[["season", "round", "race_id"]].drop_duplicates()
    m = pace.merge(keys, on=["season", "round"], how="left")
    return m[m["race_id"] < race_id].drop(columns="race_id").reset_index(drop=True)


@needs_data
def test_truncation_leakage_2025_race_final_model_and_simulator(race_2025):
    """The frozen H2 prediction of a 2025 race is bit for bit the same when every later row, lap and pit stop is removed."""
    entries, r = race_2025
    cfg = load_yaml(HERE / "frozen_config.yaml")
    pace = F.build_pace_table(cfg["pace_feature"])
    full = F.predict_race(entries, r, cfg, "H2", with_sim=True, pace=pace, n_warm=3)
    cut_e = entries[entries["race_id"] <= r].reset_index(drop=True)
    cut = F.predict_race(cut_e, r, cfg, "H2", with_sim=True, pace=pace_before(pace, entries, r), n_warm=3)
    pd.testing.assert_frame_equal(full, cut, check_exact=True)


@needs_data
def test_harness_scores_of_a_2025_race_ignore_the_outcome_of_that_race_and_later(race_2025):
    """The scoring path (`forward_scores`, as the harness). (a) Same rows, outcomes of r and later blanked: the score of r is bit for bit
    the same. (b) Rows after r removed too: the driver/lineage universe shrinks, which moves scores by about 2e-4 (documented in README, not
    an outcome effect) and must not change the order."""
    entries, r = race_2025
    cfg = load_yaml(HERE / "frozen_config.yaml")
    from pitlane.models.registry import build_model
    pace = F.build_pace_table(cfg["pace_feature"])
    e = F.add_pace_hist(entries, pace, cfg["pace_feature"])
    mc = cfg["horizons"]["H2"]["model"]
    targets = [x for x in sorted(e["race_id"].unique()) if x < r][-3:] + [r]
    s_full = F.forward_scores(e, build_model(mc, e), targets)
    m = (e["race_id"] == r).to_numpy()
    blank = e.copy()
    for c in F.OUTCOME_COLS:
        if c in blank.columns and blank[c].dtype.kind == "f":
            blank.loc[blank["race_id"] >= r, c] = np.nan
    s_blank = F.forward_scores(blank, build_model(mc, blank), targets)
    assert np.isfinite(s_full[m]).all() and (s_full[m] == s_blank[m]).all()
    cut = F.prepare_pre_race(e, r)
    s_cut = F.forward_scores(cut, build_model(mc, cut), targets)
    m_cut = (cut["race_id"] == r).to_numpy()
    assert np.abs(s_full[m] - s_cut[m_cut]).max() < 1e-3
    assert (pd.Series(s_full[m]).rank().to_numpy() == pd.Series(s_cut[m_cut]).rank().to_numpy()).all()


def test_score_script_refuses_an_edited_frozen_config(tmp_path):
    for f in ("frozen_config.yaml", "FREEZE.md"):
        shutil.copy(HERE / f, tmp_path / f)
    (tmp_path / "frozen_config.yaml").write_bytes((tmp_path / "frozen_config.yaml").read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="edited after the freeze"):
        F.verify_freeze(tmp_path / "frozen_config.yaml", tmp_path / "FREEZE.md")


# ------------------------------------------------------------------------------------ R15 fixes (main-session decisions 2026-11-18)
def test_holm_skeptics_case_t1_confirmed_despite_a_wrong_way_t3():
    rng = np.random.default_rng(11)
    boots = {"T1": 0.27 + 0.1 * rng.standard_normal(20_000), "T2": 0.1 * rng.standard_normal(20_000), "T3": -1.0 + 0.1 * rng.standard_normal(20_000)}
    r = H.holm_stepdown(boots).set_index("test")
    assert r.loc["T1", "rank"] == 1 and r.loc["T3", "rank"] == 3
    assert r.loc["T1", "confirmed"] and not r.loc["T2", "confirmed"] and not r.loc["T3", "confirmed"]


@pytest.mark.parametrize("neg", ["T1", "T2", "T3"])
def test_holm_testers_case_two_right_way_tests_confirmed_in_every_arrangement(neg):
    rng = np.random.default_rng(12)
    boots = {t: (-1.5 if t == neg else 1.0) + 0.1 * rng.standard_normal(10_000) for t in ("T1", "T2", "T3")}
    got = H.holm_stepdown(boots).set_index("test")["confirmed"].to_dict()
    assert got == {t: t != neg for t in boots}


def test_holm_ties_are_broken_in_the_preregistered_order():
    b = np.full(1000, 1.0)
    r = H.holm_stepdown({"T3": b.copy(), "T1": b.copy(), "T2": b.copy()}).set_index("test")
    assert [r.loc[t, "rank"] for t in ("T1", "T2", "T3")] == [1, 2, 3]
    assert abs(r.loc["T1", "p_boot"] - 1 / 1001) < 1e-15


def test_constants_are_hard_coded_and_a_disagreeing_config_is_refused():
    sc = load_score()
    assert (H.ALPHA, H.LEVELS, H.N_BOOT, H.SEED) == (0.05, (20, 40), 10_000, 0) and len(H.PARQUET_FILES + H.CSV_FILES) == 8
    cfg = load_yaml(HERE / "config.yaml")
    sc.check_config_constants(cfg)
    for mod in (lambda c: c["bootstrap"].update(n_boot=500), lambda c: c["score"].update(alpha=0.1), lambda c: c["score"].update(min_n_levels=[10]),
                lambda c: c["score"].update(parquet_files=["x"])):
        bad = load_yaml(HERE / "config.yaml")
        mod(bad)
        with pytest.raises(SystemExit):
            sc.check_config_constants(bad)


def test_holdout_prereg_hash_is_verified(tmp_path):
    shutil.copy(HERE / "HOLDOUT_PREREG.md", tmp_path / "HOLDOUT_PREREG.md")
    assert H.verify_prereg(tmp_path / "HOLDOUT_PREREG.md", HERE / "FREEZE.md") == H.prereg_hash(HERE / "FREEZE.md")
    (tmp_path / "HOLDOUT_PREREG.md").write_bytes((tmp_path / "HOLDOUT_PREREG.md").read_bytes() + b"x")
    with pytest.raises(H.DataIntegrityError):
        H.verify_prereg(tmp_path / "HOLDOUT_PREREG.md", HERE / "FREEZE.md")


def test_unsorted_entries_are_refused():
    sc = load_score()
    ok = pd.DataFrame({"race_id": [1, 1, 2, 2], "pos": [1, 2, 1, 2]})
    sc.assert_entries_sorted(ok)
    with pytest.raises(SystemExit):
        sc.assert_entries_sorted(ok.iloc[[1, 0, 2, 3]])


def test_csv_column_change_message(tmp_path):
    root = make_root(tmp_path)
    spec = spec_of(root)
    (root / "data/external/ext.csv").write_text("k,v,w\na,1,0\nb,2,0\nc,3,0\n", encoding="utf-8")
    msg = H.check_spec(spec, root)[0]
    assert "columns changed" in msg and "frozen columns only" not in msg


def test_pinned_reference_lookups_use_run_ids():
    sc = load_score()
    assert all(rid.endswith(k[2]) for k, rid in sc.PINNED.items())
    assert sc.THETA_RUN[3] == "20261118-0705_p3_S3refit_latent_111_dev"
    if HAVE_DATA:
        assert set(sc.theta_latent({})) >= {"tau", "p0"}


def test_holdout_digests_detect_a_changed_holdout_row(tmp_path):
    root = make_root(tmp_path)
    H_PQ = ["data/processed/entries.parquet", "data/interim/laps.parquet"]
    orig = H.PARQUET_FILES
    try:
        H.PARQUET_FILES = H_PQ
        keys = {2: (2025, 23), 3: (2026, 19)}
        d = H.holdout_digests(root, keys)
        assert H.check_holdout_digests(d, root, keys) == []
        e = pd.read_parquet(root / H_PQ[0])
        e.loc[e["race_id"] == 3, "pos"] = 9
        e.to_parquet(root / H_PQ[0], index=False)
        assert H.check_holdout_digests(d, root, keys)
    finally:
        H.PARQUET_FILES = orig


def _dry_fixture(tmp, e_all, boundary_round, end_id):
    """Run score.main in tmp on the data cut at race end_id with the freeze boundary after 2025 round `boundary_round` (old races as 'holdout')."""
    from pitlane.config import save_yaml
    from pitlane.evaluation import harness
    from pitlane.plan2 import pipeline2 as Q
    sc = load_score()
    mdir, plan3 = tmp / "P3", tmp / "plan3"
    mdir.mkdir()
    plan3.mkdir()
    for f in ("frozen_config.yaml", "FREEZE.md", "freeze_rowhash.json", "HOLDOUT_PREREG.md"):
        shutil.copy(HERE / f, mdir / f)
    b = int(e_all[(e_all["season"] == 2025) & (e_all["round"] == boundary_round)]["race_id"].iloc[0])
    lb = tmp / "lb.csv"
    shutil.copy(ROOT / "results/plan3/leaderboard.csv", lb)
    cfg = load_yaml(HERE / "config.yaml")
    cfg["leaderboard"] = str(lb)
    cfg["score"]["last_race"] = dict(race_id=b, season=2025, round=boundary_round)
    save_yaml(cfg, tmp / "cfg.yaml")
    (plan3 / "holdout_status.json").write_text(json.dumps(dict(third_look_done=True, fresh_race_ids=[], n=0)), encoding="utf-8")
    mp = pytest.MonkeyPatch()
    mp.setattr(sc, "MDIR", mdir)
    mp.setattr(sc, "RES", mdir / "results")
    mp.setattr(sc, "PLAN3", plan3)
    mp.setattr(harness, "FRESH_AFTER", (2025, boundary_round))
    mp.setattr(harness, "RESULTS", tmp / "results")
    mp.setattr(Q, "load_all", lambda rebuild=False: e_all[e_all["race_id"] <= end_id].reset_index(drop=True))
    try:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            rc = sc.main(["--config", str(tmp / "cfg.yaml")])
    finally:
        mp.undo()
    return rc, plan3


@needs_data
def test_n20_reading_is_the_same_whether_the_script_runs_at_n20_or_n23(tmp_path):
    from pitlane.plan2 import pipeline2 as Q
    e_all = Q.load_all()
    ids = sorted(e_all[(e_all["season"] == 2025) & (e_all["round"] > 1)]["race_id"].unique())
    assert len(ids) >= 23
    rc20, p20 = _dry_fixture(tmp_path / "a", e_all, 1, int(ids[19])) if (tmp_path / "a").mkdir() is None else (None, None)
    rc23, p23 = _dry_fixture(tmp_path / "b", e_all, 1, int(ids[22])) if (tmp_path / "b").mkdir() is None else (None, None)
    assert rc20 == 0 and rc23 == 0
    v20 = json.loads((p20 / "holdout_verdict_n20.json").read_text(encoding="utf-8"))
    v23 = json.loads((p23 / "holdout_verdict_n20.json").read_text(encoding="utf-8"))
    assert v20["n_races_available"] == 20 and v23["n_races_available"] == 23
    for k in ("race_ids", "tests", "confirmed", "row_digests", "n_races_read"):
        assert v20[k] == v23[k], k
    assert v23["git_commit"] and v23["freeze_rowhash_sha256"] == F.sha256_file(HERE / "freeze_rowhash.json")


def test_no_write_is_refused_when_a_reading_is_due_and_the_tree_dirty_check_exists():
    src = (HERE / "score.py").read_text(encoding="utf-8")
    assert "--no-write refused" in src and "tree_dirty()" in src and "is_relative_to(ROOT.resolve())" in src
