"""Independent review tests for PLAN3 R14 (P3 final). Written by the tester, not the coder.

Where possible the checks avoid the coder's helpers: the model configs are built by hand from the frozen YAML, the harness
`walk_forward` is called directly, the pace column is cross-checked against the M09 run script, and the paired bootstrap is
re-implemented here. Real-data tests skip when the git-ignored data are missing. Dev block 2021-2023 only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import zlib

import numpy as np
import pandas as pd
import pytest

from pitlane.config import PROCESSED, ROOT, load_yaml
from pitlane.evaluation.harness import walk_forward
from pitlane.models.registry import build_model
from pitlane.plan3 import final as F

HERE = ROOT / "methodologies" / "P3_final"
CONFIG, FREEZE = HERE / "frozen_config.yaml", HERE / "FREEZE.md"
HAVE = (PROCESSED / "entries2.parquet").exists() and (ROOT / "data/interim/plan3/laps.parquet").exists() and CONFIG.exists()
needs_data = pytest.mark.skipif(not HAVE, reason="git-ignored data not available")
PRIOR = "20261118-0705_p3_"
OUTCOMES = ["finish_order", "positionText", "classified", "status", "points", "pos", "y", "dnf", "inc", "mech"]


# ------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_yaml(CONFIG)


@pytest.fixture(scope="module")
def entries0() -> pd.DataFrame:
    from pitlane.plan2 import pipeline2 as Q
    return Q.load_all()


@pytest.fixture(scope="module")
def pace(cfg) -> pd.DataFrame:
    """Pace table built by hand (own source split), not through F.build_pace_table."""
    from pitlane.features import pace as PACE
    sp = cfg["pace_feature"]
    laps = pd.read_parquet(ROOT / "data/interim/plan3/laps.parquet")
    pits = pd.read_parquet(ROOT / "data/interim/plan3/pits.parquet")
    keep = lambda d: d[((d["source"] == "jolpica") & (d["season"] <= 2023)) | ((d["source"] == "openf1") & (d["season"] >= 2024))]
    return PACE.build_pace_table(keep(laps), keep(pits), sp["thresh"], sp["min_clean"])


@pytest.fixture(scope="module")
def entries(entries0, cfg, pace) -> pd.DataFrame:
    return F.add_pace_hist(entries0, pace, cfg["pace_feature"])


@pytest.fixture(scope="module")
def base() -> dict:
    return dict(load_yaml(ROOT / "configs/frozen.yaml")["M03"]["111"])


@pytest.fixture(scope="module")
def full_cfg(cfg) -> dict:
    return dict(cfg["horizons"]["H2"]["model"])


def stored(leaderboard: str, label: str, split: str = "dev"):
    lb = pd.read_csv(ROOT / leaderboard)
    row = lb[(lb["label"] == label) & (lb["split"] == split)].iloc[-1]
    d = ROOT / row["run_dir"]
    return pd.read_csv(d / "per_race_metrics.csv"), pd.read_parquet(d / "predictions.parquet")


def walk(entries: pd.DataFrame, mcfg: dict) -> pd.DataFrame:
    p, _ = walk_forward(entries, build_model(mcfg, entries), "dev", refit_every=1)
    return p


def merged_diff(a: pd.DataFrame, b: pd.DataFrame, col: str = "score") -> float:
    m = a[["race_id", "driverId", col]].merge(b[["race_id", "driverId", col]], on=["race_id", "driverId"], suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b)
    return float(np.abs(m[col + "_a"] - m[col + "_b"]).max())


# ------------------------------------------------------------------------------------ 4. freeze
def test_freeze_hash_recomputed_independently():
    raw = CONFIG.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == "24ef86b6271b9df4d12a4d08e5fff587d2f2d5bb877e79213c82dad1e19abe00"
    assert "24ef86b6271b9df4d12a4d08e5fff587d2f2d5bb877e79213c82dad1e19abe00" in FREEZE.read_text(encoding="utf-8")
    # a pure CRLF re-save must not change the hash, a lone-CR or one extra byte must
    assert hashlib.sha256(raw.replace(b"\n", b"\r\n").replace(b"\r\n", b"\n")).hexdigest() == hashlib.sha256(raw).hexdigest()
    assert hashlib.sha256(raw + b" ").hexdigest() != hashlib.sha256(raw).hexdigest()


def _copy_freeze_dir(tmp_path):
    for f in ("predict.py", "frozen_config.yaml", "FREEZE.md"):
        shutil.copy(HERE / f, tmp_path / f)
    return tmp_path


def _run_predict(d, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), OMP_NUM_THREADS="1")
    return subprocess.run([sys.executable, str(d / "predict.py"), *args], capture_output=True, text=True, env=env, timeout=600)


@needs_data
def test_predict_exit_codes(tmp_path):
    d = _copy_freeze_dir(tmp_path)
    before = (HERE / "frozen_config.yaml").read_bytes()
    # untouched copy, race without entry rows -> 3
    r = _run_predict(d, "--season", "2031", "--round", "1", "--out", str(d / "o.parquet"))
    assert r.returncode == 3, (r.returncode, r.stderr[-400:])
    # one-byte edit of the temp copy -> 2, before any data are read
    (d / "frozen_config.yaml").write_bytes((d / "frozen_config.yaml").read_bytes() + b" ")
    r = _run_predict(d, "--season", "2021", "--round", "1", "--out", str(d / "o.parquet"))
    assert r.returncode == 2 and "REFUSED" in r.stderr, (r.returncode, r.stderr[-400:])
    assert not (d / "o.parquet").exists()
    # a changed value (not only whitespace) -> 2 as well
    (d / "frozen_config.yaml").write_bytes(before.replace(b"lam_x: 3.0", b"lam_x: 3.1", 1))
    assert (d / "frozen_config.yaml").read_bytes() != before
    assert _run_predict(d, "--season", "2021", "--round", "1").returncode == 2
    assert (HERE / "frozen_config.yaml").read_bytes() == before


@needs_data
def test_predict_runs_and_writes_for_a_dev_race(tmp_path):
    r = _run_predict(HERE, "--season", "2022", "--round", "10", "--no-sim", "--out", str(tmp_path / "o.parquet"))
    assert r.returncode == 0, r.stderr[-600:]
    o = pd.read_parquet(tmp_path / "o.parquet")
    assert sorted(o["pred_place"]) == list(range(1, len(o) + 1)) and o["config_sha256"].nunique() == 1


@needs_data
def test_frozen_yaml_statements(cfg, entries0):
    lr = cfg["frozen"]["last_race_included"]
    assert (lr["season"], lr["round"]) == (2026, 20)
    ent = entries0[(entries0["season"] == 2026) & (entries0["round"] == 20)]
    assert ent["race_id"].nunique() == 1 and int(ent["race_id"].iloc[0]) == 271
    assert int(entries0["race_id"].max()) == 271, "the data end at the last included race"
    f2 = load_yaml(ROOT / "configs/frozen_plan2.yaml")["H1"]["111"]
    h1 = dict(cfg["horizons"]["H1"]["model"])
    assert h1 == f2, (h1, f2)
    assert cfg["horizons"]["H1"]["uses_pace_hist"] is False and "simulator" not in cfg["horizons"]["H1"]
    m = cfg["horizons"]["H2"]["model"]
    assert (m["lam"], m["halflife"], m["reverse"], m["x_cols"], m["lam_x"]) == (32.0, 50, True, ["pace_hist_x"], 3.0)
    assert "car_penalty" not in str(cfg["components"]["included"]) and "lam_a" not in m
    th = cfg["horizons"]["H2"]["simulator"]["theta"]
    assert (th["tau"], th["p0"], th["s_dis"], th["a_vol"]) == (1.7, 0.05, 1.5, 1.0)


@needs_data
def test_frozen_constants_come_from_dev_and_before(cfg, entries0, pace):
    """scale_sd = SD over seasons <= 2020; simulator constants = dev 2021-2023 means (no test season in a fixed constant)."""
    sp = cfg["pace_feature"]
    assert sp["scale_seasons_max"] == 2020
    raw = F.pace_hist_raw(entries0, pace, sp)
    sd = pd.Series(raw)[entries0["season"].to_numpy() <= 2020].std()
    assert sd == pytest.approx(sp["scale_sd"], rel=0, abs=1e-12)
    dev = entries0[entries0["season"].between(2021, 2023)]
    sim = cfg["horizons"]["H2"]["simulator"]
    assert dev["stick"].mean() == pytest.approx(sim["stick_ref"], abs=1e-12)
    assert dev["circ_vol"].mean() == pytest.approx(sim["vol_mu"], abs=1e-12)
    assert dev["circ_vol"].std() == pytest.approx(sim["vol_sd"], abs=1e-12)


# ------------------------------------------------------------------------------------ 2. reproduction (bit for bit)
@pytest.fixture(scope="module")
def walks(entries, base, full_cfg):
    """Own configs, harness walk_forward called directly."""
    no_rev = {k: v for k, v in full_cfg.items() if k != "reverse"}
    no_pace = {k: v for k, v in full_cfg.items() if k not in ("x_cols", "lam_x")}
    return {"off": walk(entries, dict(base)), "full": walk(entries, full_cfg),
            "minus_reverse": walk(entries, no_rev), "minus_pace": walk(entries, no_pace)}


@needs_data
def test_all_switches_off_equals_latent_111_bit_for_bit(walks, entries, base):
    from pitlane.plan3 import final as FF
    assert FF.latent_cfg(base) == base                       # the coder's builder returns the base dict
    _, old = stored("results/leaderboard.csv", "latent_111")
    assert np.array_equal(walks["off"]["score"].to_numpy(), old["score"].to_numpy())
    assert walks["off"]["race_id"].equals(old["race_id"])
    # the coder's builder with every switch off, run through the harness
    p = walk(entries, dict(FF.latent_cfg(base, reverse=False, pace_hist=False)))
    assert np.array_equal(p["score"].to_numpy(), old["score"].to_numpy())


@needs_data
def test_full_minus_pace_equals_stored_m11_reverse_run(walks):
    _, old = stored("results/plan3/leaderboard.csv", "m11_latent_111_reverse")
    assert np.array_equal(walks["minus_pace"]["score"].to_numpy(), old["score"].to_numpy())


@needs_data
def test_full_minus_reverse_equals_stored_m09_history_run(walks):
    _, old = stored("results/plan3/leaderboard.csv", "m09_hist_latent_111")
    assert np.array_equal(walks["minus_reverse"]["score"].to_numpy(), old["score"].to_numpy())


@needs_data
def test_full_model_equals_stored_final_run_and_is_not_trivially_the_same(walks):
    _, old = stored("results/plan3/leaderboard.csv", "p3_final_model")
    assert np.array_equal(walks["full"]["score"].to_numpy(), old["score"].to_numpy())
    # sanity: the switches do something
    for k in ("off", "minus_reverse", "minus_pace"):
        assert merged_diff(walks["full"], walks[k]) > 1e-3


# ------------------------------------------------------------------------------------ 3. dev numbers and bootstrap
def boot(ref: np.ndarray, cand: np.ndarray, metric: str, higher: bool, n_boot: int = 10_000, seed: int = 0):
    """Paired bootstrap over races, the attribution.py scheme: rng [seed, crc32(metric)], percentile 2.5/97.5."""
    d = (1.0 if higher else -1.0) * (cand - ref)
    rng = np.random.default_rng([seed, zlib.crc32(metric.encode())])
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def per_race(p: pd.DataFrame) -> pd.DataFrame:
    """Own per-race spearman / mae / exact from predictions: rank of score with a driverId tie-break, no harness code."""
    rows = []
    for rid, g in p.groupby("race_id", sort=True):
        g = g.sort_values("driverId")
        order = np.lexsort((g["driverId"].to_numpy(), -g["score"].to_numpy()))
        pred = np.empty(len(g))
        pred[order] = np.arange(1, len(g) + 1)
        pos = g["pos"].to_numpy(float)
        rho = pd.Series(pred).corr(pd.Series(pos), method="spearman")
        rows.append((rid, rho, np.abs(pred - pos).mean(), float((pred == pos).mean())))
    return pd.DataFrame(rows, columns=["race_id", "spearman", "mae", "exact"]).set_index("race_id")


def _csv_row(path, **kw):
    t = pd.read_csv(path)
    for k, v in kw.items():
        t = t[t[k] == v]
    assert len(t) == 1, (path, kw, len(t))
    return t.iloc[0]


def _check_row(row, ref, cand, tol=1e-9):
    for m, hb in (("spearman", True), ("mae", False), ("exact", True)):
        g, lo, hi = boot(ref[m].to_numpy(), cand[m].to_numpy(), m, hb)
        for name, mine in (("gain", g), ("lo", lo), ("hi", hi)):
            assert row[f"{name}_{m}"] == pytest.approx(mine, abs=tol), (m, name, row[f"{name}_{m}"], mine)


@needs_data
def test_dev_headline_numbers():
    _, p = stored("results/plan3/leaderboard.csv", "p3_final_model")
    t = per_race(p)
    assert len(t) == 66 and len(p) == 1311
    assert t["spearman"].mean() == pytest.approx(0.6632, abs=5e-5)
    assert t["mae"].mean() == pytest.approx(3.236, abs=5e-4)
    assert t["exact"].mean() == pytest.approx(0.1512, abs=5e-5)


@needs_data
@pytest.mark.parametrize("cand,ref,att_ref,att_comp", [
    ("p3_final_model", "p3_ref_latent_111", "latent_111", "P3_final"),
    ("p3_final_model", "p3_loo_r1_minus_reverse", "leave-one-out: full (M11_reverse+M09_car_pace_hist) vs full minus M11_reverse", "P3_LOO_reverse"),
    ("p3_final_model", "p3_loo_r1_minus_pace_hist", "leave-one-out: full (M11_reverse+M09_car_pace_hist) vs full minus M09_car_pace_hist", "P3_LOO_pace_hist"),
])
def test_bootstrap_rows_rederived_from_saved_predictions(cand, ref, att_ref, att_comp):
    _, pc = stored("results/plan3/leaderboard.csv", cand)
    _, pr = stored("results/plan3/leaderboard.csv", ref)
    row = _csv_row(ROOT / "results/plan3/attribution.csv", component=att_comp, reference=att_ref)
    _check_row(row, per_race(pr), per_race(pc))


@needs_data
def test_final_vs_latent_111_literal_values():
    _, pc = stored("results/plan3/leaderboard.csv", "p3_final_model")
    _, pr = stored("results/plan3/leaderboard.csv", "p3_ref_latent_111")
    c, r = per_race(pc), per_race(pr)
    g, lo, hi = boot(r["spearman"].to_numpy(), c["spearman"].to_numpy(), "spearman", True)
    assert (round(g, 4), round(lo, 4), round(hi, 4)) == (0.0172, -0.0002, 0.0340)
    g, lo, hi = boot(r["mae"].to_numpy(), c["mae"].to_numpy(), "mae", False)
    assert (round(g, 3), round(lo, 3), round(hi, 3)) == (0.119, 0.013, 0.222)


@needs_data
def test_attribution_and_leaderboard_diffs_are_pure_additions():
    for f in ("results/plan3/attribution.csv", "results/plan3/leaderboard.csv"):
        out = subprocess.run(["git", "diff", "--numstat", "--", f], capture_output=True, text=True, cwd=ROOT).stdout.split()
        assert out and out[1] == "0", (f, out)          # zero deleted lines
    att = pd.read_csv(ROOT / "results/plan3/attribution.csv")
    assert not att.duplicated(["component", "reference"]).any()
    lb = pd.read_csv(ROOT / "results/plan3/leaderboard.csv")
    assert not lb.duplicated(["run_id"]).any() if "run_id" in lb else True


# ------------------------------------------------------------------------------------ 7. simulator refit
@needs_data
def test_s3_refit_theta_values_and_determinism(cfg, entries0):
    from pitlane.plan2 import dnf as D
    from pitlane.plan2 import simulate as S
    f2 = load_yaml(ROOT / "configs/frozen_plan2.yaml")
    consts = {k: f2["M08"]["S3"][k] for k in ("stick_ref", "vol_mu", "vol_sd")}
    pr_dnf, _, _ = D.walk_dnf(entries0, "dev", f2["M07"]["C"])
    cols = ["race_id", "driverId", "grid_norm", "stick", "circ_vol", "rain_fc"]

    def fit(p):
        fr = p.merge(pr_dnf[["race_id", "driverId", "p_dnf"]], on=["race_id", "driverId"]).merge(entries0[cols], on=["race_id", "driverId"])
        return S.fit_theta(S.prepare(fr, consts), "S3", k=300)

    _, ref = stored("results/plan3/leaderboard.csv", "p3_ref_latent_111")
    th1, rps1 = fit(ref)
    th2, rps2 = fit(ref)
    assert th1 == th2 and rps1 == rps2, "refit is not deterministic"
    assert th1 == {k: pytest.approx(float(f2["M08"]["S3"][k])) for k in S.DEFAULT}
    _, fin = stored("results/plan3/leaderboard.csv", "p3_final_model")
    thf, rpsf = fit(fin)
    frozen = cfg["horizons"]["H2"]["simulator"]
    assert thf == {k: pytest.approx(float(frozen["theta"][k])) for k in S.DEFAULT}
    assert rpsf == pytest.approx(frozen["fit_rps_dev"], abs=1e-12)
    # grid-edge warning is reported, not asserted: final theta sits on grid edges for p0, s_dis, a_vol


# ------------------------------------------------------------------------------------ 1/5 leakage, truncation
def _late_laps_scrambled(pace: pd.DataFrame, entries0: pd.DataFrame, rid: int, seed: int = 1) -> pd.DataFrame:
    """Pace table with the rows of every race >= rid replaced by absurd values (shuffled and shifted)."""
    keys = entries0[["season", "round", "race_id"]].drop_duplicates()
    m = pace.merge(keys, on=["season", "round"], how="left")
    rng = np.random.default_rng(seed)
    late = (m["race_id"] >= rid).to_numpy()
    for c in ("med_gap_pct", "corr_gap_pct", "med_lap_s", "corr_lap_s"):
        v = m[c].to_numpy(float).copy()
        v[late] = rng.permutation(v[late]) + 5.0
        m[c] = v
    return m.drop(columns="race_id")


def _drop_late(pace, entries0, rid):
    keys = entries0[["season", "round", "race_id"]].drop_duplicates()
    m = pace.merge(keys, on=["season", "round"], how="left")
    return m[m["race_id"] < rid].drop(columns="race_id").reset_index(drop=True)


def _scramble_outcomes(e: pd.DataFrame, rid: int, seed: int = 2) -> pd.DataFrame:
    out = e.copy()
    rng = np.random.default_rng(seed)
    for _, ix in out[out["race_id"] >= rid].groupby("race_id").indices.items():
        rows = out[out["race_id"] >= rid].index.to_numpy()[ix]
        perm = rng.permutation(len(rows))
        for c in OUTCOMES:
            if c in out.columns:
                out.loc[rows, c] = out.loc[rows, c].to_numpy()[perm]
    return out


@pytest.fixture(scope="module")
def race_ids(entries0):
    """Dev races: round 1 of 2022 and 2023 (history crosses a season boundary), round 2 of 2021, a mid-season 2022 race."""
    def rid(s, r):
        return int(entries0[(entries0["season"] == s) & (entries0["round"] == r)]["race_id"].iloc[0])
    return {"2021R2": rid(2021, 2), "2022R1": rid(2022, 1), "2022R12": rid(2022, 12), "2023R1": rid(2023, 1), "2023R3": rid(2023, 3)}


@needs_data
def test_pace_column_agrees_with_m09_run_script_for_dev(entries0, entries):
    """M09's own feature code and its stored a3 pace table (seasons <= 2023) against the final column, all rows of seasons <= 2023."""
    mcfg = load_yaml(ROOT / "methodologies/M09_race_pace/config.yaml")
    tab_path = ROOT / mcfg["pace_table"]
    if not tab_path.exists():
        pytest.skip("a3 pace table missing")
    spec = importlib.util.spec_from_file_location("m09_run_rev1", ROOT / "methodologies/M09_race_pace/run.py")
    m09 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m09)
    out, _ = m09.build_features(entries0, pd.read_parquet(tab_path), mcfg)
    le = (entries0["season"] <= 2023).to_numpy()
    a, b = out[m09.HIST].to_numpy(float)[le], entries[F.PACE_COL].to_numpy(float)[le]
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.nanmax(np.abs(a - b)) < 1e-12


@needs_data
def test_pace_column_independent_of_race_r_and_later(entries0, pace, cfg, race_ids):
    spec = cfg["pace_feature"]
    full = F.pace_hist_raw(entries0, pace, spec)
    for name, rid in race_ids.items():
        ref = full[(entries0["race_id"] == rid).to_numpy()]
        assert np.isfinite(ref).sum() > 0, name
        # (a) laps of race r and everything after removed, entries truncated
        cut = entries0[entries0["race_id"] <= rid].reset_index(drop=True)
        a = F.pace_hist_raw(cut, _drop_late(pace, entries0, rid), spec)[(cut["race_id"] == rid).to_numpy()]
        # (b) laps of r and later scrambled, entries not truncated, outcomes scrambled
        e2 = _scramble_outcomes(entries0, rid)
        b = F.pace_hist_raw(e2, _late_laps_scrambled(pace, entries0, rid), spec)[(e2["race_id"] == rid).to_numpy()]
        np.testing.assert_array_equal(ref, a, err_msg=name)
        np.testing.assert_array_equal(ref, b, err_msg=name)
    # not vacuous: changing an EARLIER race's laps does change the column
    rid = race_ids["2023R3"]
    keys = entries0[["season", "round", "race_id"]].drop_duplicates()
    m = pace.merge(keys, on=["season", "round"], how="left")
    prev = int(entries0[entries0["race_id"] < rid]["race_id"].max())
    sel = (m["race_id"] == prev).to_numpy()   # a constant shift cancels in the field-median centring, so use a driver-varying one
    m.loc[sel, "med_gap_pct"] = m.loc[sel, "med_gap_pct"].to_numpy()[::-1] + np.arange(sel.sum()) * 0.1
    alt = F.pace_hist_raw(entries0, m.drop(columns="race_id"), spec)[(entries0["race_id"] == rid).to_numpy()]
    assert not np.allclose(np.nan_to_num(alt), np.nan_to_num(full[(entries0["race_id"] == rid).to_numpy()]))


@needs_data
def test_window_crosses_season_boundary(entries0, pace, cfg, race_ids):
    """Round 1 of 2022 must use the 2021 races (history is not reset at a season boundary)."""
    rid = race_ids["2022R1"]
    v = F.pace_hist_raw(entries0, pace, cfg["pace_feature"])[(entries0["race_id"] == rid).to_numpy()]
    assert np.isfinite(v).mean() > 0.8


@needs_data
def test_final_scores_unchanged_by_truncation_and_scrambling(entries0, pace, cfg, full_cfg, race_ids):
    """Same warm-started fit sequence (3 target races ending at r) on: full data / data cut after r / outcomes of r and later scrambled
    + laps of r and later scrambled. The scores of race r must be identical."""
    sp = cfg["pace_feature"]
    for name in ("2022R1", "2023R1", "2022R12", "2021R2"):
        rid = race_ids[name]
        ids = sorted(entries0["race_id"].unique())
        targets = [i for i in ids if i < rid][-3:] + [rid]

        def scores(e: pd.DataFrame, pc: pd.DataFrame) -> np.ndarray:
            e = F.add_pace_hist(e.reset_index(drop=True), pc, sp)
            # blank the race's own outcome columns exactly as a live race would have them
            m = (e["race_id"] == rid).to_numpy()
            for c in ("pos", "y", "finish_order", "points"):
                e.loc[m, c] = np.nan
            model = build_model(full_cfg, e)
            race_rows = e.groupby("race_id").indices
            s = None
            for r in targets:
                tr = race_rows[r]
                s = model.fit_predict(e.iloc[:tr.min()], e.iloc[tr])
            return s

        base_s = scores(entries0[entries0["race_id"] <= rid], _drop_late(pace, entries0, rid))
        full_s = scores(entries0[entries0["race_id"] <= rid], pace)            # laps of r (and later) still in the pace table
        scr = _scramble_outcomes(entries0, rid)
        scr_s = scores(scr[scr["race_id"] <= rid], _late_laps_scrambled(pace, entries0, rid))
        np.testing.assert_array_equal(base_s, full_s, err_msg=name)
        np.testing.assert_array_equal(base_s, scr_s, err_msg=name)
        assert np.isfinite(base_s).all()


@needs_data
def test_predict_race_h2_and_h1_unchanged_by_future_and_own_outcome(entries0, pace, cfg, race_ids):
    for name in ("2022R1", "2023R3", "2022R12"):
        rid = race_ids[name]
        pace_cut = _drop_late(pace, entries0, rid)
        for hz in ("H2", "H1"):
            ref = F.predict_race(entries0, rid, cfg, hz, with_sim=True, pace=pace_cut, n_warm=4)
            # data after r removed up front, own outcomes scrambled
            cut = entries0[entries0["race_id"] <= rid].reset_index(drop=True)
            a = F.predict_race(_scramble_outcomes(cut, rid), rid, cfg, hz, with_sim=True, pace=pace_cut, n_warm=4)
            # data after r present and scrambled, laps of r and later scrambled and present
            scr = _scramble_outcomes(entries0, rid)
            b = F.predict_race(scr, rid, cfg, hz, with_sim=True, pace=_late_laps_scrambled(pace, entries0, rid), n_warm=4)
            for other in (a, b):
                assert ref["driverId"].tolist() == other["driverId"].tolist()
                for c in ref.columns:
                    if ref[c].dtype.kind in "fiu":
                        np.testing.assert_array_equal(ref[c].to_numpy(), other[c].to_numpy(), err_msg=f"{name} {hz} {c}")
            if hz == "H2":
                assert {"p_win", "e_pos", "p_dnf"} <= set(ref.columns)
                assert np.isfinite(ref["e_pos"]).all()


@needs_data
def test_predict_race_changes_when_an_earlier_race_changes(entries0, pace, cfg, race_ids):
    """Non-vacuous: flipping the outcomes of an earlier race changes the prediction."""
    rid = race_ids["2022R12"]
    pace_cut = _drop_late(pace, entries0, rid)
    ref = F.predict_race(entries0, rid, cfg, "H2", with_sim=False, pace=pace_cut, n_warm=4)
    prev = rid - 1
    ix = entries0.index[entries0["race_id"] == prev].to_numpy()
    order = np.arange(len(entries0))
    order[ix] = ix[::-1]                    # the model reads the row order of a past race as its finishing order
    e2 = entries0.iloc[order].reset_index(drop=True)
    out = F.predict_race(e2, rid, cfg, "H2", with_sim=False, pace=pace_cut, n_warm=4)
    assert np.abs(ref["score"].to_numpy() - out["score"].to_numpy()).max() > 1e-6


@needs_data
def test_outcome_columns_blanked_cover_every_outcome_like_column(entries0):
    """Columns that exist in the plan-2 entries and look like results of the race itself must be in OUTCOME_COLS or be pre-race."""
    allowed_prerace = {"grid", "grid_norm", "quali_pos", "q1_gap", "sprint_pos", "sprint_gain"}
    suspicious = [c for c in entries0.columns if c in ("finish_order", "positionText", "classified", "status", "points", "pos", "y", "dnf",
                                                      "inc", "mech") or c.startswith("result") or c.startswith("final")]
    assert set(suspicious) <= set(F.OUTCOME_COLS) | allowed_prerace
    e = F.prepare_pre_race(entries0, 150)
    assert e["race_id"].max() == 150
    m = e["race_id"] == 150
    for c in F.OUTCOME_COLS:
        if c in e.columns:
            assert e.loc[m, c].isna().all() or (e[c].dtype == bool and not e.loc[m, c].any()), c


# ------------------------------------------------------------------------------------ 6. predict vs harness, >= 10 races
@needs_data
def test_predict_race_vs_harness_on_twelve_dev_races(entries0, pace, cfg, walks, race_ids):
    _, d0 = stored("results/plan3/leaderboard.csv", "p3_final_model")
    _, d1 = stored("results/plan3/leaderboard.csv", "p3_S3refit_final")
    dev_ids = sorted(d0["race_id"].unique())
    pick = [dev_ids[i] for i in np.linspace(0, len(dev_ids) - 1, 12).round().astype(int)]
    assert len(set(pick)) == 12
    worst, worst_sim, flips, flips_sim, p_diff = 0.0, 0.0, [], [], 0.0
    for rid in pick:
        out = F.predict_race(entries0, rid, cfg, "H2", with_sim=True, pace=pace)
        h = d0[d0["race_id"] == rid][["driverId", "score"]].merge(out[["driverId", "score", "pred_place"]], on="driverId", suffixes=("_h", "_p"))
        assert len(h) == len(out)
        worst = max(worst, float(np.abs(h["score_h"] - h["score_p"]).max()))
        # harness order from the model score, tie-break driverId
        ho = h.sort_values(["score_h", "driverId"], ascending=[False, True])["driverId"].tolist()
        po = h.sort_values(["score_p", "driverId"], ascending=[False, True])["driverId"].tolist()
        if ho != po:
            flips.append(rid)
        # D0 order from the simulator against the stored S3 refit run (score = -E[pos], tie-broken)
        s_ = d1[d1["race_id"] == rid][["driverId", "score"]].merge(out[["driverId", "pred_place", "e_pos"]], on="driverId")
        sp = s_["score"].rank(ascending=False, method="first")
        shift = int(np.abs(sp - s_["pred_place"]).max())
        worst_sim = max(worst_sim, shift)
        if shift:
            flips_sim.append((int(rid), shift, round(float(np.abs(-s_["score"] - s_["e_pos"]).max()), 3)))
    print(f"\nmax |score diff| predict vs harness over 12 races: {worst:.3e}; score-order flips: {flips}; D0-order (race, max place shift, max|E[pos] diff|) vs stored S3 run: {flips_sim}")
    assert worst < 5e-3
    assert not flips, f"score order flips in races {flips}"
    # The harness simulates all dev races from ONE random stream, predict_race one race from a fresh seed: Monte-Carlo noise in E[pos].
    # Reported, not hidden: (race, max place shift, max |E[pos] diff|). The first dev race has the same stream start, so it must match.
    assert worst_sim <= 3, flips_sim
    first = int(dev_ids[0])
    assert first in pick and first not in [f[0] for f in flips_sim], "first race of the stream should reproduce exactly"


# ------------------------------------------------------------------------------------ 6. split discipline
def test_no_test_season_or_fresh_split_in_new_code():
    import re
    src = (ROOT / "src/pitlane/plan3/final.py").read_text(encoding="utf-8")
    run = (HERE / "run.py").read_text(encoding="utf-8")
    pred = (HERE / "predict.py").read_text(encoding="utf-8")
    for name, text in (("final.py", src), ("run.py", run)):
        assert not re.search(r"""["'](test_stable|test_shift|fresh)["']""", text), name
        assert not re.search(r"SPLITS\[[\"'](test|fresh)", text), name
    assert 'assert cfg["split"] == "dev"' in run
    assert "walk_forward" not in pred


@needs_data
def test_predict_race_deterministic_and_independent_of_row_order_of_the_race(entries0, pace, cfg, race_ids):
    """Same call twice -> identical. Permuting the rows of race r (they are stored in finishing order) must not change a driver's
    model score; the simulator output may move by Monte-Carlo noise only (reported)."""
    rid = race_ids["2022R12"]
    pc = _drop_late(pace, entries0, rid)
    a = F.predict_race(entries0, rid, cfg, "H2", with_sim=True, pace=pc, n_warm=4)
    b = F.predict_race(entries0, rid, cfg, "H2", with_sim=True, pace=pc, n_warm=4)
    for c in ("score", "e_pos", "p_win", "pred_place"):
        np.testing.assert_array_equal(a[c].to_numpy(), b[c].to_numpy())
    ix = entries0.index[entries0["race_id"] == rid].to_numpy()
    order = np.arange(len(entries0))
    order[ix] = np.random.default_rng(5).permutation(ix)
    e2 = entries0.iloc[order].reset_index(drop=True)
    c_ = F.predict_race(e2, rid, cfg, "H2", with_sim=True, pace=pc, n_warm=4)
    m = a.merge(c_, on="driverId", suffixes=("_a", "_c"))
    assert np.abs(m["score_a"] - m["score_c"]).max() < 1e-9
    print("row-order permutation of the target race: max |E[pos] diff| %.3f, max place shift %d"
          % (np.abs(m["e_pos_a"] - m["e_pos_c"]).max(), int(np.abs(m["pred_place_a"] - m["pred_place_c"]).max())))
