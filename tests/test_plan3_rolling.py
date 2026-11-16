"""R3 / A4 rolling-origin: harness split spec, leakage of the rolling re-tune, and the stored result tables.

Speed: the leakage tests use the real `tune_window` / `rolling_predict` / models, but a 4-configuration grid
(pipeline.grid_for monkeypatched), refit_every=10 while tuning, n_jobs=1, and (final-prediction test) a 20-tree
GBM next to one real latent_111 config. Tests marked `slow` take several seconds to a minute.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import re
import subprocess
import types

import numpy as np
import pandas as pd
import pytest

from pitlane import pipeline as P
from pitlane.config import METHODS, PROCESSED, RESULTS, ROOT, WARMUP_RACES, load_yaml
from pitlane.evaluation import harness as H
from pitlane.evaluation.harness import walk_forward
from pitlane.models.baselines import Heuristic
from pitlane.models.registry import build_model
from pitlane.plan3 import rolling as R

ENTRIES = PROCESSED / "entries_feat.parquet"
A4 = METHODS / "A4_rolling_origin"
OUT = RESULTS / "plan3"
needs_entries = pytest.mark.skipif(not ENTRIES.exists(), reason="entries_feat.parquet not built")
needs_a4 = pytest.mark.skipif(not (OUT / "season_table.csv").exists(), reason="A4 outputs missing")

KEYS = ["race_id", "season", "round", "date", "circuitId", "raceName", "driverId", "constructorId", "lineage", "n"]
MODELS = ["quali", "latent_111", "gbm_100"]
RACES_PER_SEASON = [19, 21, 20, 21, 21, 17, 22, 22, 22, 24, 24, 20]   # 2015..2026 (2026: rounds 1-20)
SEED_BOOT = 0


@pytest.fixture(scope="module")
def entries() -> pd.DataFrame:
    return pd.read_parquet(ENTRIES)


# ------------------------------------------------------------------------------------------ helpers
def head_harness() -> types.ModuleType:
    """The committed (HEAD) harness.py, loaded as a separate module."""
    src = subprocess.check_output(["git", "show", "HEAD:src/pitlane/evaluation/harness.py"], cwd=ROOT, text=True,
                                  encoding="utf-8")
    mod = types.ModuleType("harness_head")
    exec(compile(src, "harness_head.py", "exec"), mod.__dict__)
    return mod


def scramble(e: pd.DataFrame, from_race_id: int, seed: int = 0) -> pd.DataFrame:
    """Permute every non-key column across all rows of races >= from_race_id (results, grid, features, everything)."""
    out = e.copy()
    idx = np.flatnonzero((e["race_id"] >= from_race_id).to_numpy())
    rng = np.random.default_rng(seed)
    for c in e.columns:
        if c in KEYS:
            continue
        perm = rng.permutation(idx)
        out.iloc[idx, out.columns.get_loc(c)] = e.iloc[perm, e.columns.get_loc(c)].to_numpy()
    return out


def first_race(e: pd.DataFrame, season: int) -> int:
    return int(e.loc[e["season"] == season, "race_id"].min())


SMALL_LATENT = [{"kind": "latent", "cell": "111", "lam": lam, "halflife": hl} for lam in (8.0, 32.0) for hl in (20, 50)]
TINY_GBM = {"kind": "gbm", "cell": "100", "num_leaves": 4, "n_estimators": 20, "learning_rate": 0.1,
            "min_child_samples": 40, "halflife": 32}


# ------------------------------------------------------------------------------------------ 1. harness
@needs_entries
@pytest.mark.parametrize("split", ["burn_in", "dev", "test_stable", "test_shift", "dev+test_stable",
                                   "test_stable+test_shift", "dev+test_stable+test_shift", "burn_in+dev", "fresh"])
def test_named_splits_unchanged_vs_head(entries, split):
    old = head_harness()
    assert H.split_races(entries, split) == old.split_races(entries, split)


@needs_entries
@pytest.mark.parametrize("a,b", [(2021, 2021), (2018, 2020), (2021, 2023), (2015, 2026), (2026, 2026), (2024, 2025)])
def test_seasons_spec_returns_exactly_those_races(entries, a, b):
    warm, counted = H.split_races(entries, f"seasons:{a}-{b}")
    races = entries.groupby("race_id")["season"].first()
    want = races[(races >= a) & (races <= b)].index.tolist()
    assert counted == want and counted == sorted(counted) and len(counted) > 0
    assert warm == [r for r in races.index if r < counted[0]][-WARMUP_RACES:]
    assert all(r < counted[0] for r in warm)
    assert R.season_split(a, b) == f"seasons:{a}-{b}" and R.season_split(a) == f"seasons:{a}-{a}"


@needs_entries
@pytest.mark.parametrize("seasons,named", [("seasons:2021-2023", "dev"), ("seasons:2024-2025", "test_stable"),
                                           ("seasons:2026-2026", "test_shift"), ("seasons:2014-2020", "burn_in")])
def test_seasons_spec_equals_named_split(entries, seasons, named):
    assert H.split_races(entries, seasons) == H.split_races(entries, named)


@needs_entries
@pytest.mark.parametrize("bad", ["seasons:", "seasons:2021", "seasons:2021-", "seasons:-2021", "seasons:a-b",
                                 "seasons:2021-2022-2023", "seasons:2021-2023+dev", "seasons:2030-2031",
                                 "seasons:2023-2021", "season:2021-2022", "nonsense", ""])
def test_malformed_split_raises(entries, bad):
    with pytest.raises((ValueError, KeyError, IndexError)):
        H.split_races(entries, bad)


@needs_entries
@pytest.mark.parametrize("model", ["quali", "gbm"])
def test_walk_forward_seasons_equals_dev(entries, model):
    def mk():
        return Heuristic(which="quali") if model == "quali" else build_model(TINY_GBM, entries)
    p1, m1 = walk_forward(entries, mk(), "dev", refit_every=40, n_samples=100)
    p2, m2 = walk_forward(entries, mk(), "seasons:2021-2023", refit_every=40, n_samples=100)
    pd.testing.assert_frame_equal(p1, p2, check_exact=True)
    pd.testing.assert_frame_equal(m1, m2, check_exact=True)
    assert len(m1) == 66


# ------------------------------------------------------------------------------------------ 2. leakage
class Spy:
    """Wraps a model; records (max train race_id, min/max test race_id) of every fit_predict call."""
    calls: list = []

    def __init__(self, inner):
        self.inner = inner

    def fit_predict(self, train, test):
        Spy.calls.append((int(train["race_id"].max()), int(test["race_id"].min()), int(test["race_id"].max())))
        return self.inner.fit_predict(train, test)


@needs_entries
@pytest.mark.slow
def test_tuning_window_only_sees_races_before_each_evaluated_race(entries, monkeypatch):
    S, lo, hi = 2022, 2019, 2021
    monkeypatch.setattr(P, "grid_for", lambda k, c: SMALL_LATENT[:2])
    real_build = R.build_model
    Spy.calls = []
    monkeypatch.setattr(R, "build_model", lambda cfg, e: Spy(real_build(cfg, e)))
    R.tune_window("latent", "111", entries, lo, hi, 10, 1, 30)
    assert Spy.calls
    last_hi = int(entries.loc[entries["season"] == hi, "race_id"].max())
    first_S = first_race(entries, S)
    for tmax, tmin, tmaxx in Spy.calls:
        assert tmax < tmin, "a model was fitted on a race it predicts or later"
        assert tmaxx <= last_hi < first_S, "race of season S or later touched during tuning of S"
    # the evaluated (counted) races are exactly seasons lo..hi
    _, counted = H.split_races(entries, R.season_split(lo, hi))
    seasons = entries[entries["race_id"].isin(counted)]["season"].unique()
    assert set(seasons) == {2019, 2020, 2021}


@needs_entries
@pytest.mark.slow
@pytest.mark.parametrize("S", [2022, 2023])
def test_chosen_config_does_not_depend_on_season_S_or_later(entries, monkeypatch, S):
    lo, hi = S - 3, S - 1
    monkeypatch.setattr(P, "grid_for", lambda k, c: SMALL_LATENT)
    best0, tab0 = R.tune_window("latent", "111", entries, lo, hi, 10, 1, 30)
    scr = scramble(entries, first_race(entries, S))
    assert not scr["pos"].equals(entries["pos"])                      # the scramble did something
    best1, tab1 = R.tune_window("latent", "111", scr, lo, hi, 10, 1, 30)
    pd.testing.assert_frame_equal(tab0, tab1, check_exact=True)
    assert best0 == best1
    # Physically removing S and later gives the same table up to optimiser noise only. It is not bit-exact because
    # LatentModel(universe) lays out its parameter vector over every driver/car of the universe passed in, and L-BFGS-B
    # stops at a tolerance that depends on that layout (measured: max |score| change 5.5e-4 for the same training rows,
    # also with maxiter=2000). The bit-exact claim is the scramble comparison above.
    trunc = entries[entries["race_id"] < first_race(entries, S)].reset_index(drop=True)
    best2, tab2 = R.tune_window("latent", "111", trunc, lo, hi, 10, 1, 30)
    np.testing.assert_allclose(tab0["spearman"], tab2["spearman"], atol=1e-3)
    assert tab0["spearman"].nunique() > 1                              # configurations really differ


@needs_entries
@pytest.mark.slow
def test_tuning_leakage_test_is_not_vacuous(entries, monkeypatch):
    """Scrambling a season INSIDE the window must change the table."""
    monkeypatch.setattr(P, "grid_for", lambda k, c: SMALL_LATENT[:2])
    _, tab0 = R.tune_window("latent", "111", entries, 2019, 2021, 10, 1, 30)
    scr = scramble(entries, first_race(entries, 2021))
    _, tab1 = R.tune_window("latent", "111", scr, 2019, 2021, 10, 1, 30)
    assert not np.allclose(tab0["spearman"], tab1["spearman"])


@needs_entries
@pytest.mark.slow
def test_gbm_chosen_config_independent_of_later_seasons(entries, monkeypatch):
    grid = [dict(TINY_GBM), dict(TINY_GBM, num_leaves=7, n_estimators=30)]
    monkeypatch.setattr(P, "grid_for", lambda k, c: grid)
    S = 2022
    best0, tab0 = R.tune_window("gbm", "100", entries, S - 3, S - 1, 10, 1, 30)
    best1, tab1 = R.tune_window("gbm", "100", scramble(entries, first_race(entries, S)), S - 3, S - 1, 10, 1, 30)
    pd.testing.assert_frame_equal(tab0, tab1, check_exact=True)
    assert best0 == best1


@needs_entries
@pytest.mark.slow
@pytest.mark.parametrize("cfg", [TINY_GBM, {"kind": "latent", "cell": "111", "lam": 32.0, "halflife": 50, "maxiter": 40}],
                         ids=["gbm", "latent"])
def test_final_prediction_unchanged_when_later_races_scrambled(entries, cfg):
    S = 2022
    ids = sorted(entries.loc[entries["season"] == S, "race_id"].unique())
    target = ids[9]
    scr = scramble(entries, target + 1)
    p0, m0 = R.rolling_predict(entries, cfg, S)
    p1, m1 = R.rolling_predict(scr, cfg, S)
    a, b = p0[p0["race_id"] <= target].reset_index(drop=True), p1[p1["race_id"] <= target].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(m0[m0["race_id"] <= target].reset_index(drop=True),
                                  m1[m1["race_id"] <= target].reset_index(drop=True), check_exact=True)
    assert not np.allclose(p0.loc[p0["race_id"] > target, "score"], p1.loc[p1["race_id"] > target, "score"])


# ------------------------------------------------------------------------------------------ 3. season table
def run_folder(label: str):
    c = sorted((A4 / "results").glob(f"*_{label}*"))
    assert len(c) == 1, (label, c)
    return c[0]


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_csv(OUT / "season_table.csv")


@needs_a4
def test_season_table_shape_and_counts(table):
    assert set(table["model"]) == set(MODELS)
    assert table.duplicated(["model", "season", "half"]).sum() == 0
    assert len(table) == 3 * 12 * 3
    assert sorted(table["season"].unique()) == list(range(2015, 2027))
    assert table[["spearman", "mae", "exact", "n_races"]].notna().all().all()
    assert table["spearman"].between(-1, 1).all() and table["exact"].between(0, 1).all() and (table["mae"] > 0).all()
    for m in MODELS:
        w = table[(table["model"] == m) & (table["half"] == "all")].sort_values("season")
        assert w["n_races"].tolist() == RACES_PER_SEASON
        for s, n in zip(range(2015, 2027), RACES_PER_SEASON):
            h = table[(table["model"] == m) & (table["season"] == s)].set_index("half")
            assert h.loc["H1", "n_races"] == n // 2 and h.loc["H2", "n_races"] == n - n // 2
            whole = (h.loc["H1", "spearman"] * h.loc["H1", "n_races"] + h.loc["H2", "spearman"] * h.loc["H2", "n_races"]) / n
            assert whole == pytest.approx(h.loc["all", "spearman"], abs=1e-12)


@needs_a4
def test_season_table_labels(table):
    for s, st in table.groupby("season")["status"]:
        u = set(st)
        assert len(u) == 1
        lab = u.pop()
        if s < 2021:
            assert lab.startswith("not clean")
        elif s <= 2023:
            assert lab.startswith("dev")
        else:
            assert lab == "third look, no decision"


@needs_a4
@pytest.mark.parametrize("name", MODELS)
def test_season_table_reproduces_from_run_folder(table, name):
    d = run_folder(f"season_table_{name}")
    per = pd.read_csv(d / "per_race_metrics.csv")
    preds = pd.read_parquet(d / "predictions.parquet")
    assert per["race_id"].is_unique and len(per) == sum(RACES_PER_SEASON)
    assert per["season"].max() == 2026 and per.loc[per["season"] == 2026, "round"].max() == 20
    assert (per[per["season"] == 2026]["round"].sort_values().to_numpy() == np.arange(1, 21)).all()
    for s, g in per.groupby("season"):
        assert sorted(g["round"]) == list(range(1, len(g) + 1))        # rounds contiguous: halves partition the season
    assert not preds.duplicated(["race_id", "driverId"]).any() and preds["score"].notna().all()
    assert preds["race_id"].nunique() == len(per)
    # per-race metrics are what the unchanged race_metrics gives on the stored predictions
    re = R.rescore(preds).set_index("race_id").sort_index()
    pr = per.set_index("race_id").sort_index()
    for c in R.COLS:
        np.testing.assert_allclose(re[c].to_numpy(), pr[c].to_numpy(), atol=1e-12)
    whole = per.groupby("season")[R.COLS].mean()
    t = table[(table["model"] == name) & (table["half"] == "all")].set_index("season")
    for c in R.COLS:
        np.testing.assert_allclose(t.loc[whole.index, c].to_numpy(), whole[c].to_numpy(), atol=1e-12)
    h1 = R.add_half(per).query("half == 'H1'").groupby("season")[R.COLS].mean()
    t1 = table[(table["model"] == name) & (table["half"] == "H1")].set_index("season")
    for c in R.COLS:
        np.testing.assert_allclose(t1.loc[h1.index, c].to_numpy(), h1[c].to_numpy(), atol=1e-12)


@needs_a4
@needs_entries
@pytest.mark.parametrize("name", MODELS)
def test_season_table_predictions_cover_entries(entries, name):
    preds = pd.read_parquet(run_folder(f"season_table_{name}") / "predictions.parquet")
    e = entries[entries["season"] >= 2015]
    assert len(preds) == len(e)
    assert preds.groupby("race_id").size().equals(e.groupby("race_id").size())
    assert e["race_id"].nunique() == sum(RACES_PER_SEASON)
    assert e.groupby("season")["race_id"].nunique().tolist() == RACES_PER_SEASON
    m = preds.merge(e[["race_id", "driverId", "pos"]], on=["race_id", "driverId"], suffixes=("", "_e"))
    assert len(m) == len(preds) and (m["pos"] == m["pos_e"]).all()


@needs_a4
@pytest.mark.parametrize("name", MODELS)
def test_season_table_dev_and_test_rows_reproduce_plan1_leaderboard(table, name):
    lb = pd.read_csv(RESULTS / "leaderboard.csv")
    t = table[(table["model"] == name) & (table["half"] == "all")].set_index("season")
    for split, seasons in (("dev", [2021, 2022, 2023]), ("test_stable", [2024, 2025]), ("test_shift", [2026])):
        row = lb[(lb["label"] == name) & (lb["split"] == split)]
        assert len(row) == 1
        n = t.loc[seasons, "n_races"]
        assert n.sum() == row.iloc[0]["n_races"]
        got = float((t.loc[seasons, "spearman"] * n).sum() / n.sum())
        assert got == pytest.approx(row.iloc[0]["spearman"], abs=1e-12)


@needs_a4
@pytest.mark.parametrize("name", MODELS)
def test_season_table_2021_plus_reuses_plan1_scores(name):
    stored = R.stored_predictions(name)
    mine = pd.read_parquet(run_folder(f"season_table_{name}") / "predictions.parquet")
    mine = mine[mine["season"] >= 2021].sort_values(["race_id", "driverId"]).reset_index(drop=True)
    stored = stored.sort_values(["race_id", "driverId"]).reset_index(drop=True)
    assert len(mine) == len(stored)
    np.testing.assert_allclose(mine["score"], stored["score"], atol=1e-12)


# ------------------------------------------------------------------------------------------ 4. paired file
@pytest.fixture(scope="module")
def paired() -> pd.DataFrame:
    return pd.read_csv(OUT / "a4_paired_dev.csv")


@needs_a4
@pytest.mark.parametrize("name", ["latent_111", "gbm_100"])
def test_paired_file_matches_run_folders(paired, name):
    p = paired[paired["model"] == name].sort_values("race_id").reset_index(drop=True)
    assert len(p) == 66 and p["race_id"].is_unique and set(p["season"]) == {2021, 2022, 2023}
    assert p.groupby("season").size().tolist() == [22, 22, 22]
    roll = pd.read_csv(run_folder(f"rolling_{name}_dev") / "per_race_metrics.csv").sort_values("race_id").reset_index(drop=True)
    assert roll["race_id"].tolist() == p["race_id"].tolist()
    fixed = R.rescore(R.stored_predictions(name).query("season <= 2023")).sort_values("race_id").reset_index(drop=True)
    for c in R.COLS:
        np.testing.assert_allclose(p[f"{c}_rolling"], roll[c], atol=1e-12)
        np.testing.assert_allclose(p[f"{c}_fixed"], fixed[c], atol=1e-12)
        d = (roll[c] - fixed[c]).to_numpy()
        s = json.loads((OUT / "a4_summary.json").read_text())["retune"][name][c]
        assert d.mean() == pytest.approx(s["delta"], abs=1e-12)
        assert s["rolling"] == pytest.approx(roll[c].mean(), abs=1e-12) and s["fixed"] == pytest.approx(fixed[c].mean(), abs=1e-12)
        mean, lo, hi = R.paired_ci(d, 5000, SEED_BOOT)
        assert (mean, lo, hi) == pytest.approx((s["delta"], s["ci_lo"], s["ci_hi"]), abs=1e-12)
        assert s["ci_excludes_0"] == bool(lo > 0 or hi < 0)
    # the plan-1 per-race file of the dev run agrees with the re-scored Spearman
    lb = pd.read_csv(RESULTS / "leaderboard.csv")
    r = lb[(lb["label"] == name) & (lb["split"] == "dev")].iloc[0]
    old = pd.read_csv(METHODS.parent / r["run_dir"] / "per_race_metrics.csv").sort_values("race_id").reset_index(drop=True)
    np.testing.assert_allclose(old["spearman"], p["spearman_fixed"], atol=1e-12)


@needs_a4
@pytest.mark.parametrize("name,season", [("latent_111", 2021), ("latent_111", 2023), ("gbm_100", 2022)])
def test_same_config_season_has_exactly_zero_difference(paired, name, season):
    summ = json.loads((OUT / "a4_summary.json").read_text())["retune"][name]
    assert summ["chosen"][str(season)] == summ["fixed_cfg"]
    p = paired[(paired["model"] == name) & (paired["season"] == season)]
    assert len(p) == 22
    for c in R.COLS:
        assert (p[f"{c}_rolling"] - p[f"{c}_fixed"]).abs().max() == 0.0


@needs_a4
@pytest.mark.parametrize("name,seasons", [("latent_111", [2022]), ("gbm_100", [2021, 2023])])
def test_different_config_season_does_differ(paired, name, seasons):
    for s in seasons:
        p = paired[(paired["model"] == name) & (paired["season"] == s)]
        assert (p["spearman_rolling"] - p["spearman_fixed"]).abs().max() > 0


# ------------------------------------------------------------------------------------------ 5. tuning budget
def test_search_space_is_the_plan1_space():
    assert P.LAM_GRID == [0.5, 2.0, 8.0, 32.0] and P.HL_GRID == [8, 20, 50] and P.GBM_HL_GRID == [8, 16, 32]
    assert P.GBM_GRID == [dict(num_leaves=7, n_estimators=150, learning_rate=0.05, min_child_samples=40),
                          dict(num_leaves=15, n_estimators=150, learning_rate=0.05, min_child_samples=40),
                          dict(num_leaves=7, n_estimators=300, learning_rate=0.03, min_child_samples=80),
                          dict(num_leaves=4, n_estimators=200, learning_rate=0.05, min_child_samples=40)]
    assert len(P.grid_for("latent", "111")) == 12 and len(P.grid_for("gbm", "100")) == 12


@needs_a4
@pytest.mark.parametrize("name,kind,cell,cols", [("latent_111", "latent", "111", ["lam", "halflife"]),
                                                 ("gbm_100", "gbm", "100", ["num_leaves", "n_estimators", "learning_rate",
                                                                            "min_child_samples", "halflife"])])
def test_tuning_tables_budget_and_space(name, kind, cell, cols):
    cfg = load_yaml(A4 / "config.yaml")
    tab = pd.read_csv(OUT / f"a4_tuning_{name}.csv")
    grid = pd.DataFrame(P.grid_for(kind, cell))
    summ = json.loads((OUT / "a4_summary.json").read_text())["retune"][name]
    assert cfg["retune"]["max_configs"] <= 30 and cfg["retune"]["refit_every_tuning"] == 4 and cfg["seed"] == 0
    assert tab["predict_season"].unique().tolist() == [2021, 2022, 2023]
    for s, g in tab.groupby("predict_season"):
        assert len(g) <= 30 and len(g) == len(grid)
        assert g["tuned_on"].unique().tolist() == [f"{s - 3}-{s - 1}"]
        got = {tuple(r) for r in g[cols].itertuples(index=False)}
        assert got == {tuple(r) for r in grid[cols].itertuples(index=False)}
        _, counted = H.split_races(pd.read_parquet(ENTRIES), R.season_split(s - 3, s - 1)) if ENTRIES.exists() else (0, [])
        if ENTRIES.exists():
            assert (g["n_races"] == len(counted)).all()
        # chosen = best mean Spearman, first of the grid on ties; matches summary and run-folder config
        best = g.loc[g["spearman"].idxmax(), cols]
        ch = summ["chosen"][str(s)]
        for c in cols:
            assert ch[c] == best[c]
        rc = load_yaml(run_folder(f"rolling_{name}_dev") / "config.yaml")
        assert {k: v for k, v in rc["chosen_per_season"][s].items()} == ch
    # the plan-1 script tunes with refit_every=4 and the harness default seed
    src = (ROOT / "scripts" / "run_all.py").read_text(encoding="utf-8")
    assert len(re.findall(r"refit_every=4", src)) >= 2
    assert inspect.signature(walk_forward).parameters["seed"].default == 0
    assert "seed" not in inspect.getsource(R._eval) and "seed" not in inspect.getsource(R.rolling_predict)


@needs_a4
def test_a4_does_not_use_test_seasons_for_choice():
    """Tuning windows end before 2024; chosen configs come only from the tuning tables (which end <= 2022)."""
    for name in ("latent_111", "gbm_100"):
        tab = pd.read_csv(OUT / f"a4_tuning_{name}.csv")
        assert max(int(x.split("-")[1]) for x in tab["tuned_on"]) <= 2022
    src = (A4 / "run.py").read_text(encoding="utf-8")
    assert not re.search(r"test_stable|test_shift", src.split("def stage_retune")[1].split("def main")[0])


# ------------------------------------------------------------------------------------------ 6. decision rule
def _load_run():
    spec = importlib.util.spec_from_file_location("a4_run_for_test", A4 / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_a4
@needs_entries
@pytest.mark.slow
@pytest.mark.parametrize("ci,adopt", [((0.01, 0.002, 0.02), True), ((0.01, -0.001, 0.02), False),
                                      ((0.0, -0.01, 0.01), False), ((-0.01, -0.02, -0.002), False),
                                      ((0.01, 0.0, 0.02), False)])
def test_decision_rule_in_code(monkeypatch, entries, ci, adopt):
    run = _load_run()
    cfg = load_yaml(A4 / "config.yaml")
    cfg["retune"] = dict(cfg["retune"], seasons=[2021], models=["latent_111"])
    frozen = load_yaml(ROOT / cfg["frozen_plan1"])
    fixed = frozen["M03"]["111"]
    monkeypatch.setattr(R, "tune_window", lambda *a, **k: (fixed, pd.DataFrame({"spearman": [0.0]})))
    monkeypatch.setattr(R, "rolling_predict",
                        lambda e, c, s: walk_forward(e, Heuristic(which="quali"), R.season_split(s), refit_every=40, n_samples=50))
    monkeypatch.setattr(R, "paired_ci", lambda x, b, seed: ci)
    monkeypatch.setattr(run, "OUT", RESULTS / "_nonexistent_a4_test")
    # stage_retune writes a4_tuning_<model>.csv when not quick: neutralise every csv write so no result file is touched
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, *a, **k: None)
    out = run.stage_retune(entries, cfg, frozen, quick=False)
    assert out["latent_111"]["decision"] == ("tune on the three seasons before the freeze" if adopt
                                             else "keep hyper-parameters tuned on 2021-23")


@needs_a4
def test_decision_text_and_stored_decision_consistent():
    summ = json.loads((OUT / "a4_summary.json").read_text())["retune"]
    for name, v in summ.items():
        want = ("tune on the three seasons before the freeze" if v["spearman"]["ci_lo"] > 0
                else "keep hyper-parameters tuned on 2021-23")
        assert v["decision"] == want
        assert v["n_races"] == 66
    plan = (ROOT / "PLAN3.md").read_text(encoding="utf-8")
    assert "95% bootstrap CI excluding 0" in plan
    readme = (A4 / "README.md").read_text(encoding="utf-8")
    assert "CI above 0" in readme and "5000" in readme
    cfg = load_yaml(A4 / "config.yaml")
    assert cfg["decision"]["metric"] == "spearman" and cfg["bootstrap"] == 5000
