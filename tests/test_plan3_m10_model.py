"""PLAN3 R9 part 2 / M10 model rung: feature availability, neutrality of missing values, H0, lam_x, freeze guard.

Synthetic tables for the logic; real-data tests skip when the git-ignored inputs are missing.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from pitlane.config import FRESH_AFTER, PROCESSED, ROOT, load_yaml
from pitlane.plan3 import m10_feature as M

CFG = load_yaml(ROOT / "methodologies/M10_upgrades/config.yaml")["model"]
_spec = importlib.util.spec_from_file_location("m10_run", ROOT / "methodologies/M10_upgrades/run.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

T0 = pd.Timestamp("2025-03-14 12:00")


def _docs() -> pd.DataFrame:
    rows = []
    for rnd, pub in ((2, T0), (3, T0 + pd.Timedelta(days=14)), (4, T0 + pd.Timedelta(days=28))):
        for team, n in (("a", 0), ("b", 2), ("c", 4), ("d", pd.NA)):
            rows.append(dict(season=2025, round=rnd, constructorId=team, declared=n, how="x", published_utc=pub))
    d = pd.DataFrame(rows)
    d["declared"] = d["declared"].astype("Int64")
    return d


def _cuts(offset_h: float = 5.0) -> pd.DataFrame:
    """FP1 and qualifying start `offset_h` / `offset_h + 24` hours after the (round 2) publication, same offsets later."""
    out = []
    for rnd in (2, 3, 4):
        pub = T0 + pd.Timedelta(days=14 * (rnd - 2))
        out.append(dict(season=2025, round=rnd, H1_cut=pub + pd.Timedelta(hours=offset_h),
                        H2_cut=pub + pd.Timedelta(hours=offset_h + 24)))
    return pd.DataFrame(out)


def _entries() -> pd.DataFrame:
    rows = [dict(season=2025, round=r, constructorId=t, driverId=f"{t}{k}") for r in (1, 2, 3, 4) for t in "abcd" for k in (1, 2)]
    return pd.DataFrame(rows)


def _col(docs, cuts, h, ent=None) -> pd.Series:
    ent = _entries() if ent is None else ent
    return M.build_column(ent, docs, cuts, h, scale=2.0)


# ---------------------------------------------------------------- value, scaling, missing -> neutral
def test_value_is_centred_within_race_and_scaled():
    ent = _entries()
    col = _col(_docs(), _cuts(), "H2", ent)
    r2 = ent["round"] == 2
    got = col[r2].groupby(ent.loc[r2, "constructorId"]).first()
    mean = (0 + 2 + 4) / 3                                   # constructors with a usable count; 'd' (NA) is not in the mean
    assert got["a"] == pytest.approx((0 - mean) / 2.0)
    assert got["c"] == pytest.approx((4 - mean) / 2.0)
    assert np.isnan(got["d"])
    assert col[r2 & (ent['constructorId'] == 'c')].nunique() == 1          # both cars of a team share the value


def test_missing_is_nan_and_enters_the_model_as_zero():
    col = _col(_docs(), _cuts(), "H2")
    ent = _entries()
    assert col[ent["round"] == 1].isna().all()               # no document for round 1
    assert col[(ent["round"] == 2) & (ent["constructorId"] == "d")].isna().all()     # NA count
    # the weekend term fills NaN with 0.0 (LatentModel), the centred value of an average car
    u = ent.assign(**{M.COL: col})
    assert u[[M.COL]].fillna(0.0).to_numpy().min() >= -1.0


def test_missing_does_not_move_other_cars():
    docs = _docs()
    a = _col(docs, _cuts(), "H2")
    docs2 = docs.copy()
    docs2.loc[(docs2["round"] == 3) & (docs2["constructorId"] == "d"), "declared"] = pd.NA     # already NA; drop team b of round 4
    docs2 = docs2[~((docs2["round"] == 4) & (docs2["constructorId"] == "b"))]
    b = _col(docs2, _cuts(), "H2")
    ent = _entries()
    assert (a[ent["round"] <= 3].fillna(9).to_numpy() == b[ent["round"] <= 3].fillna(9).to_numpy()).all()


# ---------------------------------------------------------------- cutoffs and truncation
def test_document_after_cutoff_is_not_used():
    docs, ent = _docs(), _entries()
    ok = _col(docs, _cuts(5.0), "H1", ent)                               # cutoff 5 h after publication: used
    assert ok[ent["round"] == 2].notna().any()
    late = _col(docs, _cuts(-1.0), "H1", ent)                            # cutoff 1 h BEFORE publication: dropped
    assert late.isna().all()
    exact = _col(docs, _cuts(0.0), "H1", ent)                            # published exactly at the cutoff: strict, not used
    assert exact.isna().all()


def test_h1_is_stricter_than_h2():
    docs, ent = _docs(), _entries()
    cuts = _cuts(5.0)
    cuts["H2_cut"] = cuts["H1_cut"] + pd.Timedelta(hours=24)
    docs2 = docs.assign(published_utc=docs["published_utc"] + pd.Timedelta(hours=10))   # published after FP1, before qualifying
    assert _col(docs2, cuts, "H1", ent).isna().all()
    assert _col(docs2, cuts, "H2", ent).notna().any()


def test_no_session_time_means_not_used_and_is_counted():
    docs, ent = _docs(), _entries()
    cuts = _cuts().copy()
    cuts.loc[cuts["round"] == 3, "H1_cut"] = pd.NaT
    col = _col(docs, cuts, "H1", ent)
    assert col[ent["round"] == 3].isna().all() and col[ent["round"] == 2].notna().any()
    _, av = M.usable_documents(docs, cuts, "H1")
    assert (av["status"] == "no_session_time").sum() == 1
    cov = M.coverage(ent, docs, cuts, "H1")
    assert cov["documents_dropped_no_session_time"] == 1 and cov["documents_used"] == 2


def test_publication_time_is_the_later_reading():
    f = pd.Series([pd.Timestamp("2025-06-01 10:00")])
    p = pd.Series([pd.Timestamp("2025-06-01 11:00")])
    assert M.publication_time(f, p).iloc[0] == pd.Timestamp("2025-06-01 11:00")
    assert M.publication_time(f, pd.Series([pd.NaT])).iloc[0] == pd.Timestamp("2025-06-01 11:00")   # DST gap: +1 h
    assert M.publication_time(p, f).iloc[0] == pd.Timestamp("2025-06-01 11:00")


def test_features_of_race_r_unchanged_when_later_documents_and_results_are_removed():
    docs, cuts, ent = _docs(), _cuts(), _entries()
    ent = ent.assign(pos=np.arange(len(ent)), finish_order=np.arange(len(ent)), grid=1)
    full = M.build_column(ent, docs, cuts, "H2", 2.0)
    r = 3
    ent_cut = ent[ent["round"] <= r].copy()
    ent_cut.loc[ent_cut["round"] == r, ["pos", "finish_order", "grid"]] = np.nan          # race r's own result is not known
    cut = M.build_column(ent_cut, docs[docs["round"] <= r], cuts[cuts["round"] <= r], "H2", 2.0)
    a, b = full[ent["round"] <= r].to_numpy(), cut.to_numpy()
    assert (np.isnan(a) == np.isnan(b)).all() and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)])
    scrambled = ent.copy()
    scrambled["pos"] = scrambled["pos"].to_numpy()[::-1]
    assert M.build_column(scrambled, docs, cuts, "H2", 2.0).equals(full)               # results are never read


def test_race_r_value_ignores_earlier_and_later_documents():
    docs, cuts, ent = _docs(), _cuts(), _entries()
    only3 = M.build_column(ent, docs[docs["round"] == 3], cuts, "H2", 2.0)
    full = M.build_column(ent, docs, cuts, "H2", 2.0)
    m = (ent["round"] == 3).to_numpy()
    assert np.allclose(only3[m].to_numpy(float), full[m].to_numpy(float), equal_nan=True)


# ---------------------------------------------------------------- H0
def test_h0_has_no_m10_input():
    docs, cuts, ent = _docs(), _cuts(), _entries()
    assert _col(docs, cuts, "H0", ent).isna().all()
    use, _ = M.usable_documents(docs, cuts, "H0")
    assert len(use) == 0
    with pytest.raises(ValueError):
        R.rung_cfg(CFG, "H0")
    assert "H0" not in CFG["horizons"]
    with pytest.raises(ValueError):
        M.usable_documents(docs, cuts, "H9")


def test_rung_adds_exactly_one_column_to_the_reference():
    for h in ("H2", "H1"):
        ref, rung = R.reference_cfg(CFG, h), R.rung_cfg(CFG, h)
        assert rung["x_cols"] == list(ref.get("x_cols", [])) + [M.COL]
        assert {k: v for k, v in rung.items() if k not in ("x_cols", "lam_x")} == {k: v for k, v in ref.items() if k not in ("x_cols", "lam_x")}


# ---------------------------------------------------------------- lam_x
def test_lam_x_is_the_model_default_and_locked():
    from pitlane.models.latent import LatentModel
    import inspect
    default = inspect.signature(LatentModel.__init__).parameters["lam_x"].default
    assert default == 1.0 and CFG["lam_x"] == 1.0 and R.resolve_lam_x(CFG) == 1.0
    assert CFG["unlock_lam_x"] is False
    for h in ("H2", "H1"):
        assert R.rung_cfg(CFG, h)["lam_x"] == 1.0
    with pytest.raises(ValueError):
        R.resolve_lam_x(dict(CFG, lam_x=3.0))
    assert R.resolve_lam_x(dict(CFG, lam_x=3.0, unlock_lam_x=True)) == 3.0      # only with the explicit flag


def test_scale_is_a_fixed_constant_in_config():
    assert CFG["scale"] == 2.0 and CFG["centre"] == "race_mean" and CFG["column"] == M.COL


# ---------------------------------------------------------------- freeze guard
def test_guard_refuses_any_race_up_to_the_freeze():
    assert tuple(FRESH_AFTER) == (2026, 20)
    for p in [(2021, 1), (2023, 22), (2024, 2), (2025, 24), (2026, 1), (2026, 20)]:
        with pytest.raises(R.FreezeError):
            R.assert_after_freeze([p])
    with pytest.raises(R.FreezeError):
        R.assert_after_freeze([(2027, 1), (2026, 20)])
    R.assert_after_freeze([(2026, 21), (2027, 1)])


def test_score_refuses_dev_and_test_races_before_building_anything():
    for races in ([(2022, 5)], [(2024, 3)], [(2026, 20)]):
        with pytest.raises(R.FreezeError):
            R.score(pd.DataFrame(), CFG, races=races, write=False)
    assert R.parse_races("2027:1,2027:2") == [(2027, 1), (2027, 2)]


@pytest.mark.skipif(not (PROCESSED / "entries2.parquet").exists(), reason="processed data missing")
def test_score_with_no_fresh_race_outputs_nothing(capsys):
    ent = pd.read_parquet(PROCESSED / "entries2.parquet")
    assert R.score(ent, CFG, write=False) == {}               # no race after the freeze yet: no metric is produced
    assert "rho=" not in capsys.readouterr().out


def test_config_only_accepts_the_fresh_split():
    assert CFG["split"] == "fresh"


# ---------------------------------------------------------------- real data (skipped when git-ignored inputs are missing)
needs_docs = pytest.mark.skipif(not (ROOT / "data/interim/m10_car_presentations.parquet").exists()
                                or not (PROCESSED / "entries2.parquet").exists(), reason="parsed documents or entries missing")


@needs_docs
def test_real_feature_columns_and_model_builds_without_scoring():
    from pitlane.models.registry import build_model
    ent = pd.read_parquet(PROCESSED / "entries2.parquet")
    docs, cuts = M.document_table(), M.session_starts(ent)
    assert cuts[["H1_cut", "H2_cut"]].notna().all().all()
    for h in ("H2", "H1"):
        e = M.add_column(ent, h, CFG, docs, cuts).reset_index(drop=True)
        assert e.loc[e["season"] <= 2023, M.COL].isna().all()          # no documents before 2024
        assert e.loc[(e["season"] == 2024) & (e["round"] == 1), M.COL].isna().all()
        m = build_model(R.rung_cfg(CFG, h), e)                         # builds only; no fit, no metric
        assert M.COL in m.x_names and m.pen_x[-1] == 1.0
        assert m.row_x.shape[1] == m.nX
    h2 = M.add_column(ent, "H2", CFG, docs, cuts)[M.COL]
    h1 = M.add_column(ent, "H1", CFG, docs, cuts)[M.COL]
    assert h1.notna().sum() <= h2.notna().sum()                         # H1 can only lose documents


@needs_docs
def test_real_event_map_join_uses_event_id_not_names():
    docs = M.document_table()
    assert docs.groupby(["season", "round", "constructorId"]).size().max() == 1
    assert docs["published_utc"].notna().all()
    assert set(docs["season"]) == {2024, 2025, 2026} and (docs["round"] >= 1).all()
