"""Independent review tests for PLAN3 R15 (scoring). Written by the tester, not the coder.

Nothing here writes into the repository: integrity edits are made on copies in a tmp dir, the fresh-path dry run moves the
holdout boundary to 2025 round 3 on an in-memory copy of the data and sends every output to tmp, the leakage tests only read.
Third-look numbers are re-derived from the saved predictions with an independent metric and bootstrap implementation.
Real-data tests skip when the git-ignored data are missing. Binding spec: methodologies/P3_final/HOLDOUT_PREREG.md.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import zlib

import numpy as np
import pandas as pd
import pytest

from pitlane.config import PROCESSED, ROOT, load_yaml, save_yaml
from pitlane.plan3 import final as F
from pitlane.plan3 import holdout as H

HERE = ROOT / "methodologies" / "P3_final"
RUN = HERE / "results" / "20261118-1407_p3_R15_score"
HAVE = (PROCESSED / "entries2.parquet").exists() and (ROOT / "data/interim/plan3/laps.parquet").exists() and RUN.exists()
needs_data = pytest.mark.skipif(not HAVE, reason="git-ignored data not available")
PARQ = ["data/processed/entries2.parquet", "data/processed/entries.parquet", "data/interim/plan3/laps.parquet", "data/interim/plan3/pits.parquet"]
CSVS = ["data/external/lineage.csv", "data/external/circuits.csv", "data/external/power_units.csv", "data/external/scheduled_laps.csv"]
TL = "third look, no decision weight"


def load_score():
    spec = importlib.util.spec_from_file_location("p3_score_rev1", HERE / "score.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def git_status() -> str:
    return subprocess.run(["git", "status", "--porcelain", "-uall"], capture_output=True, text=True, cwd=ROOT).stdout


def sha_raw(p) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------ 1. integrity on the real data (copies)
@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads((HERE / "freeze_rowhash.json").read_text(encoding="utf-8"))


@pytest.fixture()
def root_copy(tmp_path):
    for rel in PARQ + CSVS:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    return tmp_path


@needs_data
def test_data_files_equal_the_raw_freeze_hashes_and_the_rowhash_check_passes(spec):
    want = H.freeze_data_hashes(HERE / "FREEZE.md")
    assert set(want) == set(PARQ + CSVS)
    for rel, h in want.items():
        raw = sha_raw(ROOT / rel)
        crlf_norm = hashlib.sha256((ROOT / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert h in (raw, crlf_norm), (rel, h, raw)
    assert spec["file_sha256_at_generation"] == want
    assert H.check_spec(spec, ROOT) == []
    assert spec["last"] == {"race_id": 271, "season": 2026, "round": 20}


@needs_data
def test_rowhash_spec_row_counts_match_the_data(spec):
    for rel in PARQ:
        df = H.restrict(pd.read_parquet(ROOT / rel), spec["last"])
        assert spec["parquet"][rel]["n_rows"] == len(df) > 0
    for rel in CSVS:
        assert spec["csv"][rel]["n_rows"] == len(pd.read_csv(ROOT / rel, dtype=str, keep_default_na=False))


def _edit_parquet(root, rel, mask_fn, col, how):
    p = root / rel
    df = pd.read_parquet(p)
    ok = df.index[mask_fn(df) & (df[col].notna() if col in df.columns else True)]
    i = ok[len(ok) // 2]                                              # a row from the middle of the frozen part
    if how == "ulp":
        df[col] = df[col].astype(float)
        df.loc[i, col] = np.nextafter(float(df.loc[i, col]), np.inf)
    elif how == "str":
        df.loc[i, col] = str(df.loc[i, col]) + "x"
    elif how == "nan":
        df.loc[i, col] = np.nan
    df.to_parquet(p, index=False)
    return i


@needs_data
@pytest.mark.parametrize("rel,col,how,mask", [
    ("data/processed/entries2.parquet", "grid", "ulp", lambda d: d["race_id"] <= 271),
    ("data/processed/entries2.parquet", "driverId", "str", lambda d: d["race_id"] == 5),
    ("data/processed/entries.parquet", "pos", "ulp", lambda d: d["race_id"] <= 271),
    ("data/processed/entries.parquet", "pos", "nan", lambda d: d["race_id"] == 100),
    ("data/interim/plan3/laps.parquet", None, "ulp", lambda d: (d["season"] == 2024) & (d["round"] == 5)),
    ("data/interim/plan3/laps.parquet", None, "ulp", lambda d: (d["season"] == 2026) & (d["round"] == 20)),
    ("data/interim/plan3/pits.parquet", None, "ulp", lambda d: (d["season"] == 2023)),
])
def test_edit_of_an_old_parquet_row_is_caught(root_copy, spec, rel, col, how, mask):
    if col is None:                                                  # first float column that is not a key
        df = pd.read_parquet(root_copy / rel)
        sub = df[mask(df)]
        col = [c for c in df.columns if df[c].dtype.kind == "f" and sub[c].notna().sum() > 5][0]
    _edit_parquet(root_copy, rel, mask, col, how)
    problems = H.check_spec(spec, root_copy)
    assert len(problems) == 1 and problems[0].startswith(rel), problems
    with pytest.raises(H.DataIntegrityError):
        H.assert_unchanged(spec, root_copy)


@needs_data
def test_moving_an_old_row_out_of_the_frozen_range_or_deleting_it_is_caught(root_copy, spec):
    p = root_copy / PARQ[0]
    df = pd.read_parquet(p)
    df.loc[df.index[10], "race_id"] = 999                            # relabelled as a new race
    df.to_parquet(p, index=False)
    assert H.check_spec(spec, root_copy)
    shutil.copy(ROOT / PARQ[0], p)
    pd.read_parquet(p).iloc[1:].to_parquet(p, index=False)           # first row removed
    assert H.check_spec(spec, root_copy)


@needs_data
@pytest.mark.parametrize("rel", CSVS)
def test_edit_of_an_old_external_row_is_caught_and_append_is_allowed(root_copy, spec, rel):
    p = root_copy / rel
    lines = p.read_bytes().split(b"\n")
    k = len(lines) // 2
    cells = lines[k].split(b",")
    cells[-1] = cells[-1] + b"x" if cells[-1].strip(b"\r") else b"x"
    edited = lines.copy()
    edited[k] = b",".join(cells)
    p.write_bytes(b"\n".join(edited))
    problems = H.check_spec(spec, root_copy)
    assert len(problems) == 1 and "edited" in problems[0], problems
    p.write_bytes((ROOT / rel).read_bytes())                         # restore, then append a row
    cols = pd.read_csv(p, dtype=str, keep_default_na=False).columns
    sep = b"" if p.read_bytes().endswith(b"\n") else b"\n"
    with open(p, "ab") as f:
        f.write(sep + (",".join(["new"] * len(cols)) + "\n").encode())
    assert H.check_spec(spec, root_copy) == []


@needs_data
def test_appended_new_race_rows_are_allowed_and_row_order_does_not_matter(root_copy, spec):
    # new race: copy of the last frozen race with race_id 272 / round 21; laps and pits of (2026, 21)
    for rel in PARQ[:2]:
        df = pd.read_parquet(root_copy / rel)
        new = df[df["race_id"] == 271].assign(race_id=272, round=21) if "round" in df.columns else df[df["race_id"] == 271].assign(race_id=272)
        pd.concat([df, new], ignore_index=True).to_parquet(root_copy / rel, index=False)
    for rel in PARQ[2:]:
        df = pd.read_parquet(root_copy / rel)
        new = df[(df["season"] == 2026) & (df["round"] == 20)].assign(round=21)
        assert len(new) > 0
        pd.concat([df, new], ignore_index=True).to_parquet(root_copy / rel, index=False)
    assert H.check_spec(spec, root_copy) == []
    for rel in PARQ:
        df = pd.read_parquet(root_copy / rel)
        df.sample(frac=1.0, random_state=3).reset_index(drop=True).to_parquet(root_copy / rel, index=False)
    assert H.check_spec(spec, root_copy) == []
    # an edit of an old row is still caught after appending and reordering
    df = pd.read_parquet(root_copy / PARQ[1])
    i = df.index[df["race_id"] == 50][0]
    df.loc[i, "pos"] = df.loc[i, "pos"] + 1
    df.to_parquet(root_copy / PARQ[1], index=False)
    assert any(PARQ[1] in q for q in H.check_spec(spec, root_copy))


@needs_data
def test_a_changed_value_in_an_old_race_of_a_column_added_later_is_not_a_false_alarm(root_copy, spec):
    """Extra columns are ignored by design (documented in PROCEDURE): adding a column must not trip the check."""
    df = pd.read_parquet(root_copy / PARQ[0])
    df["new_col"] = 1.0
    df.to_parquet(root_copy / PARQ[0], index=False)
    assert H.check_spec(spec, root_copy) == []


@needs_data
def test_external_csv_row_reorder_is_caught_as_documented(root_copy, spec):
    """Characterisation: for the hand tables a reorder IS flagged (rows must be at the same position). Not a bug for the pre-registered
    use (rows are appended), but it differs from the parquet tables, where order is free."""
    p = root_copy / CSVS[0]
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    d.iloc[::-1].to_csv(p, index=False)
    assert H.check_spec(spec, root_copy)


# ------------------------------------------------------------------------------------ 2a. n counting (synthetic)
def frame(rows):
    return pd.DataFrame(rows, columns=["race_id", "pos", "classified", "date"]).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_n_counts_only_races_after_271_with_a_classified_result():
    sc = load_score()
    rows = [(270, 1, True, "2026-10-01"), (271, 1, True, "2026-11-01"), (272, 1, True, "2026-11-08"), (272, 2, False, "2026-11-08"),
            (273, 3, True, "2026-11-15"), (274, np.nan, False, "2026-11-22"), (274, np.nan, False, "2026-11-22")]
    assert sc.fresh_race_ids(frame(rows), 271) == [272, 273]
    # a race whose rows have a finishing position but nobody classified (all retired) is not counted, and stops the count
    assert sc.fresh_race_ids(frame([(272, 1, False, "2026-11-08"), (272, 2, False, "2026-11-08")]), 271) == []
    # a classified flag with no position is not counted either
    assert sc.fresh_race_ids(frame([(272, np.nan, True, "2026-11-08")]), 271) == []


def test_fresh_ids_come_back_in_date_order():
    sc = load_score()
    rows = [(272, 1, True, "2026-11-22"), (273, 1, True, "2026-11-15")]     # id order disagrees with date order
    assert sc.fresh_race_ids(frame(rows), 271) == [273, 272]


@needs_data
def test_race_ids_are_in_date_order_in_the_real_data():
    """record_readings takes 'the first t races' by sorted race_id, fresh_race_ids by date: they must agree."""
    e = pd.read_parquet(PROCESSED / "entries2.parquet", columns=["race_id", "date"]).groupby("race_id")["date"].first()
    assert e.is_monotonic_increasing and e.index.is_monotonic_increasing


def test_reading_is_on_the_first_races_by_race_id_not_by_input_order(tmp_path):
    sc = load_score()
    ids = np.arange(300, 330)
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"race_id": ids, "spearman": 0.6 + 0.01 * rng.standard_normal(30), "mae": 3.4 + 0 * ids})
    tabs = {k: ref.copy() for k in ("latent_111", "final_minus_reverse", "reverse_only", "final")}
    tabs["final"] = ref.assign(spearman=ref["spearman"] + 0.1, mae=ref["mae"] - 1.0)
    shuf = {k: v.sample(frac=1.0, random_state=1) for k, v in tabs.items()}
    out = sc.record_readings(shuf, 30, (20, 40), 0.05, dict(n_boot=300, seed=0), tmp_path, {})
    assert out[20]["race_ids"] == list(range(300, 320))


# ------------------------------------------------------------------------------------ 2b. Holm
def my_holm(boots: dict, alpha: float = 0.05) -> dict:
    """Independent Holm (main-session decision 2026-11-18): order by the ONE-sided bootstrap p in the stated direction,
    p = (1 + #{b <= 0}) / (n + 1), ties by name (T1, T2, T3); step k (1..m) uses level alpha/(m-k+1), the two-sided percentile interval must
    lie above 0, first failure stops."""
    m = len(boots)
    p = {k: (1 + (b <= 0).sum()) / (len(b) + 1) for k, b in boots.items()}
    order = sorted(boots, key=lambda k: (p[k], k))
    res, alive = {}, True
    for j, k in enumerate(order, start=1):
        a = alpha / (m - j + 1)
        lo = np.quantile(boots[k], a / 2)
        alive = alive and lo > 0
        res[k] = alive
    return res


@pytest.mark.parametrize("seed", range(40))
def test_holm_matches_an_independent_implementation(seed):
    rng = np.random.default_rng(seed)
    mu = rng.choice([-0.3, 0.0, 0.1, 0.18, 0.25, 0.5], size=3)
    boots = {t: mu[i] + 0.1 * rng.standard_normal(4000) for i, t in enumerate(("T1", "T2", "T3"))}
    got = H.holm_stepdown(boots).set_index("test")["confirmed"].to_dict()
    assert got == my_holm(boots), (mu, got)


def test_t2_sign_convention_gain_is_reference_minus_candidate_mae():
    sc = load_score()
    ids = np.arange(300, 330)
    rng = np.random.default_rng(1)
    base = pd.DataFrame({"race_id": ids, "spearman": 0.6 + 0.01 * rng.standard_normal(30), "mae": 3.5 + 0.1 * rng.standard_normal(30)})
    tabs = {k: base.copy() for k in ("latent_111", "final_minus_reverse", "reverse_only", "final")}
    tabs["final"] = base.assign(mae=base["mae"] - 0.5, spearman=base["spearman"] + 0.05)       # final has LOWER mae: gain +0.5
    r = sc.holm_reading(tabs, 0.05, dict(n_boot=2000, seed=0)).set_index("test")
    assert abs(r.loc["T2", "gain"] - 0.5) < 1e-9 and r.loc["T2", "confirmed"]
    tabs["final"] = base.assign(mae=base["mae"] + 0.5, spearman=base["spearman"] - 0.05)       # worse on every metric
    r = sc.holm_reading(tabs, 0.05, dict(n_boot=2000, seed=0)).set_index("test")
    assert abs(r.loc["T2", "gain"] + 0.5) < 1e-9
    assert not r["confirmed"].any() and (r["gain"] < 0).all() and (r["hi95"] < 0).all()        # significant in the wrong direction: not confirmed


def test_tests_and_references_are_the_preregistered_ones():
    sc = load_score()
    assert sc.TESTS == [("T1", "final", "latent_111", "spearman"), ("T2", "final", "final_minus_reverse", "mae"),
                        ("T3", "final", "reverse_only", "spearman")]


def test_holm_alpha_levels_are_alpha_over_3_2_1():
    rng = np.random.default_rng(0)
    out = H.holm_stepdown({t: 0.5 + 0.01 * rng.standard_normal(100) + i for i, t in enumerate(("T1", "T2", "T3"))}, 0.05)
    assert np.allclose(sorted(out["alpha_k"]), [0.05 / 3, 0.05 / 2, 0.05])


def test_holm_one_sided_order_a_wrong_way_test_ranks_last_and_cannot_block_the_others():
    """Main-session decision 2026-11-18: tests are ranked by the one-sided p in the stated direction, so a significantly wrong-way test has
    p near 1, ranks last, and the two right-way tests are confirmed whichever of T1-T3 is the wrong-way one."""
    rng = np.random.default_rng(5)
    for neg in ("T1", "T2", "T3"):
        boots = {t: (-1.5 if t == neg else 1.0) + 0.1 * rng.standard_normal(10000) for t in ("T1", "T2", "T3")}
        got = H.holm_stepdown(boots).set_index("test")["confirmed"].to_dict()
        assert got == {t: t != neg for t in ("T1", "T2", "T3")}, (neg, got)


def test_verdict_written_once_and_later_reading_is_on_first_forty(tmp_path):
    sc = load_score()
    rng = np.random.default_rng(2)

    def tabs(n, mean_gain):
        ids = np.arange(300, 300 + n)
        ref = pd.DataFrame({"race_id": ids, "spearman": 0.6 + 0.03 * rng.standard_normal(n), "mae": 3.4 + 0.1 * rng.standard_normal(n)})
        c = ref.assign(spearman=ref["spearman"] + mean_gain, mae=ref["mae"] - 5 * mean_gain)
        return {"latent_111": ref, "final_minus_reverse": ref, "reverse_only": ref, "final": c}
    nb = dict(n_boot=500, seed=0)
    first = sc.record_readings(tabs(20, 0.2), 20, (20, 40), 0.05, nb, tmp_path, {})
    b20 = (tmp_path / "holdout_verdict_n20.json").read_bytes()
    big = tabs(47, -0.2)                                                   # later data say the opposite
    out = sc.record_readings(big, 47, (20, 40), 0.05, nb, tmp_path, {})
    assert (tmp_path / "holdout_verdict_n20.json").read_bytes() == b20 and out[20]["confirmed"] == first[20]["confirmed"]
    v40 = json.loads((tmp_path / "holdout_verdict_n40.json").read_text(encoding="utf-8"))
    assert v40["n_races_read"] == 40 and v40["race_ids"] == list(range(300, 340)) and v40["n_races_available"] == 47
    # the n40 reading equals a direct reading of the first 40 races only
    direct = sc.holm_reading({k: v[v["race_id"] < 340] for k, v in big.items()}, 0.05, nb)
    assert [bool(x) for x in direct.set_index("test").loc[["T1", "T2", "T3"], "confirmed"]] == [v40["confirmed"][t] for t in ("T1", "T2", "T3")]


# ------------------------------------------------------------------------------------ 3. third-look numbers (independent)
def my_race_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in df.groupby("race_id"):
        order = np.lexsort((g["driverId"].to_numpy(), -g["score"].to_numpy()))        # best score first, ties by driverId
        rk = np.empty(len(g))
        rk[order] = np.arange(1, len(g) + 1)
        pos = g["pos"].to_numpy(float)
        rows.append(dict(race_id=rid, season=int(g["season"].iloc[0]), spearman=float(np.corrcoef(rk, pos)[0, 1]),
                         mae=float(np.abs(rk - pos).mean()), exact=float((rk == pos).mean()), within1=float((np.abs(rk - pos) <= 1).mean())))
    return pd.DataFrame(rows)


def my_boot(d: np.ndarray, metric: str):
    rng = np.random.default_rng([0, zlib.crc32(metric.encode())])
    b = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(1)
    return float(d.mean()), *map(float, np.percentile(b, [2.5, 97.5]))


@pytest.fixture(scope="module")
def tl_data():
    if not HAVE:
        pytest.skip("data not available")
    p = pd.read_parquet(RUN / "predictions.parquet")
    p = p[p["scope"] == "third_look"]
    tabs = {}
    for (b, m), g in p.groupby(["block", "model"]):
        if not m.startswith("S3_"):
            tabs[(b, m)] = my_race_table(g)
    per = pd.read_csv(RUN / "per_race_metrics.csv")
    return dict(p=p, tabs=tabs, per=per[per["scope"] == "third_look"], tl=pd.read_csv(ROOT / "results/plan3/third_look.csv"))


def test_block_sizes_are_48_and_20(tl_data):
    for m in ("final", "latent_111", "grid", "H1"):
        assert tl_data["tabs"][("test_stable", m)]["race_id"].nunique() == 48
        assert tl_data["tabs"][("test_shift", m)]["race_id"].nunique() == 20
    t = tl_data["tabs"][("test_stable", "final")]
    assert (t["season"].value_counts().sort_index() == [24, 24]).all()
    p = tl_data["p"]
    assert not p.duplicated(["block", "model", "race_id", "driverId"]).any()
    assert p[["pos", "score", "grid"]].notna().all().all() or p["score"].notna().all()


CONTRASTS = [("T1:final_vs_latent_111", "final", "latent_111", "spearman", 1), ("T2:final_vs_final_minus_reverse", "final", "final_minus_reverse", "mae", 0),
             ("T3:final_vs_reverse_only", "final", "reverse_only", "spearman", 1), ("exact_vs_grid:final", "final", "grid", "exact", 1),
             ("exact_vs_grid:reverse_only", "reverse_only", "grid", "exact", 1), ("exact_vs_grid:latent_111", "latent_111", "grid", "exact", 1),
             ("final_vs_plan2_H2", "final", "latent_111", "spearman", 1)]
BLOCKS = {"test_stable": ["test_stable"], "test_shift": ["test_shift"], "pooled test_stable+test_shift": ["test_stable", "test_shift"]}


@pytest.mark.parametrize("block", list(BLOCKS))
@pytest.mark.parametrize("name,cand,ref,metric,hb", CONTRASTS)
def test_third_look_contrasts_rederived(tl_data, block, name, cand, ref, metric, hb):
    t = tl_data["tabs"]
    c = pd.concat([t[(b, cand)] for b in BLOCKS[block]])
    r = pd.concat([t[(b, ref)] for b in BLOCKS[block]])
    m = r.merge(c, on="race_id", suffixes=("_r", "_c"))
    d = (1 if hb else -1) * (m[f"{metric}_c"] - m[f"{metric}_r"]).to_numpy()
    g, lo, hi = my_boot(d, metric)
    row = tl_data["tl"][(tl_data["tl"]["block"] == block) & (tl_data["tl"]["name"] == name) & (tl_data["tl"]["metric"] == metric)]
    assert len(row) == 1
    row = row.iloc[0]
    assert row["n"] == len(d) and row["label"] == TL
    assert max(abs(g - row["gain"]), abs(lo - row["lo"]), abs(hi - row["hi"])) < 1e-9, (g, lo, hi, dict(row))
    assert abs(row["ref_mean"] - m[f"{metric}_r"].mean()) < 1e-9 and abs(row["cand_mean"] - m[f"{metric}_c"].mean()) < 1e-9


@pytest.mark.parametrize("season,block", [(2024, "test_stable"), (2025, "test_stable"), (2026, "test_shift")])
def test_third_look_per_season_rows(tl_data, season, block):
    t = tl_data["tabs"]
    c, r = t[(block, "final")], t[(block, "latent_111")]
    c, r = c[c["season"] == season], r[r["season"] == season]
    m = r.merge(c, on="race_id", suffixes=("_r", "_c"))
    g, lo, hi = my_boot((m["spearman_c"] - m["spearman_r"]).to_numpy(), "spearman")
    row = tl_data["tl"][(tl_data["tl"]["block"] == f"season {season}") & (tl_data["tl"]["name"] == "T1:final_vs_latent_111") & (tl_data["tl"]["metric"] == "spearman")].iloc[0]
    assert max(abs(g - row["gain"]), abs(lo - row["lo"]), abs(hi - row["hi"])) < 1e-9 and row["n"] == len(m)


def test_third_look_s3_rps_contrast_rederived(tl_data):
    per = tl_data["per"]
    for blk, sets in BLOCKS.items():
        a = per[(per["model"] == "S3_latent_111") & per["block"].isin(sets)].set_index("race_id")["rps"]
        b = per[(per["model"] == "S3_final") & per["block"].isin(sets)].set_index("race_id")["rps"]
        d = (a - b.reindex(a.index)).to_numpy()                                    # lower RPS better: gain = ref - cand
        g, lo, hi = my_boot(d, "rps")
        row = tl_data["tl"][(tl_data["tl"]["block"] == blk) & (tl_data["tl"]["name"] == "S3:final_vs_latent_111") & (tl_data["tl"]["metric"] == "rps")].iloc[0]
        assert max(abs(g - row["gain"]), abs(lo - row["lo"]), abs(hi - row["hi"])) < 1e-9, blk


def test_third_look_means_rederived(tl_data):
    tl = tl_data["tl"]
    for (b, m), t in tl_data["tabs"].items():
        if m == "plan2_H2":
            continue
        for k in ("spearman", "mae", "exact", "within1"):
            row = tl[(tl["table"] == "mean") & (tl["block"] == b) & (tl["name"] == m) & (tl["metric"] == k)]
            assert len(row) == 1 and abs(row.iloc[0]["cand_mean"] - t[k].mean()) < 1e-9, (b, m, k)


def stored_means(lb: str, label: str, split: str):
    d = pd.read_csv(ROOT / lb)
    row = d[(d["label"] == label) & (d["split"] == split)].iloc[-1]
    run = ROOT / str(row["run_dir"]).replace("\\", "/")
    return pd.read_csv(run / "per_race_metrics.csv"), pd.read_parquet(run / "predictions.parquet")


@pytest.mark.parametrize("split", ["test_stable", "test_shift"])
@pytest.mark.parametrize("lb,label,model", [("results/leaderboard.csv", "latent_111", "latent_111"), ("results/plan2/leaderboard.csv", "m06_L_r0_111", "latent_111"),
                                            ("results/plan2/leaderboard.csv", "m06_L_r0_111", "plan2_H2"), ("results/plan2/leaderboard.csv", "h1_latent_111", "H1"),
                                            ("results/leaderboard.csv", "grid", "grid")])
def test_stored_runs_equal_the_third_look_models(tl_data, split, lb, label, model):
    per, pred = stored_means(lb, label, split)
    mine = tl_data["tabs"][(split, model)].set_index("race_id")
    mine_stored = my_race_table(pred.assign(score=pred["score"])).set_index("race_id")
    assert len(per) == len(mine) and set(per["race_id"]) == set(mine.index)
    keys = [k for k in ("spearman", "mae", "exact", "within1") if k in per.columns]
    assert "spearman" in keys and "mae" in keys
    for k in keys:
        assert (per.set_index("race_id")[k].reindex(mine.index) - mine[k]).abs().max() < 1e-12, (model, k)
    # scores of the stored predictions equal those saved by R15 (rank equality is what matters; plan-1/2 may not have broken ties)
    new = tl_data["p"][(tl_data["p"]["block"] == split) & (tl_data["p"]["model"] == model)].merge(pred, on=["race_id", "driverId"], suffixes=("_n", "_o"))
    assert len(new) == len(pred)
    if model != "grid":
        assert (new["score_n"] - new["score_o"]).abs().max() < 1e-6          # tie-break perturbation is ~1e-9


def test_m12_readme_says_h2_is_latent_111():
    txt = (ROOT / "methodologies/M12_two_stage/README.md").read_text(encoding="utf-8")
    assert "the H2 model is `latent_111`" in txt


def test_plan2_h2_rows_are_a_copy_of_latent_111(tl_data):
    p = tl_data["p"]
    for b in ("test_stable", "test_shift"):
        a = p[(p["block"] == b) & (p["model"] == "latent_111")].reset_index(drop=True)
        c = p[(p["block"] == b) & (p["model"] == "plan2_H2")].reset_index(drop=True)
        pd.testing.assert_frame_equal(a.drop(columns="model"), c.drop(columns="model"))


def test_final_scores_differ_from_the_s3_d0_order_but_probabilities_come_from_s3(tl_data):
    """Points metrics are on the score order (harness), RPS/log-loss from the S3 matrix. The saved `final` predictions must carry the harness
    score, not -E[position]."""
    p = tl_data["p"]
    a = p[(p["block"] == "test_shift") & (p["model"] == "final")].set_index(["race_id", "driverId"])
    b = p[(p["block"] == "test_shift") & (p["model"] == "S3_final")].set_index(["race_id", "driverId"])
    assert (a["score"] - b["score"].reindex(a.index)).abs().max() > 0.1
    assert a["score"].between(-50, 50).all() and b["p_win"].between(0, 1).all()
    per = tl_data["per"]
    assert per[per["model"] == "S3_final"]["rps"].notna().all()
    assert "rps" not in per[per["model"] == "final"].dropna(axis=1, how="all").columns


def test_third_look_logloss_points_in_the_plain_model_rows_is_not_the_s3_one(tl_data):
    """Finding (minor, labelling): `third_look.csv` mean rows for final/latent_111/... carry logloss_points from the harness Plackett-Luce
    probabilities, while HOLDOUT_PREREG section 2 says the probabilities of the holdout are the S3 matrix. The S3 values are in the
    '[S3 matrix, D0 order]' blocks. This test pins the difference so nobody quotes the wrong one."""
    tl = tl_data["tl"]
    a = tl[(tl["table"] == "mean") & (tl["block"] == "test_stable") & (tl["name"] == "final") & (tl["metric"] == "logloss_points")]["cand_mean"].iloc[0]
    b = tl[(tl["table"] == "mean") & (tl["block"] == "test_stable [S3 matrix, D0 order]") & (tl["name"] == "final") & (tl["metric"] == "logloss_points")]["cand_mean"].iloc[0]
    assert abs(a - b) > 0.01


# ------------------------------------------------------------------------------------ 3b. run folders and leaderboard
@needs_data
def test_leaderboard_is_a_pure_addition_of_ten_rows_and_run_folders_are_complete():
    cur = (ROOT / "results/plan3/leaderboard.csv").read_text(encoding="utf-8").replace("\r\n", "\n")
    old = subprocess.run(["git", "show", "plan3-freeze:results/plan3/leaderboard.csv"], capture_output=True, cwd=ROOT).stdout.decode("utf-8").replace("\r\n", "\n")
    assert cur.startswith(old) and len(cur[len(old):].strip().splitlines()) == 10
    lb = pd.read_csv(ROOT / "results/plan3/leaderboard.csv")
    new = lb[lb["label"].str.startswith("p3_R15_")]
    assert len(new) == 10 and not lb.duplicated("run_id").any()
    assert set(new["split"]) == {"test_stable", "test_shift"}
    for _, r in new.iterrows():
        d = ROOT / str(r["run_dir"]).replace("\\", "/")
        for f in ("config.yaml", "metrics.json", "per_race_metrics.csv", "predictions.parquet"):
            assert (d / f).exists(), (d, f)
        cfg = load_yaml(d / "config.yaml")
        assert cfg["git_commit"] and "third look" in cfg["scored"]
        per = pd.read_csv(d / "per_race_metrics.csv")
        assert abs(per["spearman"].mean() - r["spearman"]) < 1e-12 and len(per) == {"test_stable": 48, "test_shift": 20}[r["split"]]
        assert not per.duplicated("race_id").any()
    # models that already have a third-look row in an earlier leaderboard are not appended again
    assert not any(re.search(r"p3_R15_(latent_111|grid|H1|plan2_H2)_", s) for s in new["label"])


# ------------------------------------------------------------------------------------ 4. leakage through the score path
@pytest.fixture(scope="module")
def sc_ctx():
    if not HAVE:
        pytest.skip("data not available")
    from pitlane.plan2 import pipeline2 as Q
    sc = load_score()
    fc = load_yaml(HERE / "frozen_config.yaml")
    f1 = load_yaml(ROOT / "configs/frozen.yaml")
    cfgs = sc.model_configs(fc, f1)
    pace = F.build_pace_table(fc["pace_feature"])
    e0 = Q.load_all()
    return dict(sc=sc, fc=fc, cfgs=cfgs, pace=pace, e0=e0)


def score_path(ctx, entries, pace, cfg_name, rid, monkeypatch):
    """The score path (`score.run_walk` -> harness.walk_forward, refit before the race) for exactly one counted race."""
    from pitlane.evaluation import harness
    sc = ctx["sc"]
    e = F.add_pace_hist(entries, pace, ctx["fc"]["pace_feature"])
    monkeypatch.setattr(harness, "split_races", lambda entries_, split: ([], [rid]))
    preds, _ = sc.run_walk(e, ctx["cfgs"][cfg_name], "dummy")
    return U_outcome_free(sc, preds)


def U_outcome_free(sc, preds):
    return sc.U.outcome_free(preds, "driverId").set_index("driverId")["score"]


def scramble_from(entries: pd.DataFrame, pace: pd.DataFrame, rid: int, seed: int):
    """Outcomes of race rid and later shuffled among the drivers of the same race; laps and pits of those races perturbed."""
    rng = np.random.default_rng(seed)
    e = entries.copy()
    for r in e.loc[e["race_id"] >= rid, "race_id"].unique():
        ix = e.index[e["race_id"] == r].to_numpy()
        perm = rng.permutation(ix)
        for c in F.OUTCOME_COLS:
            if c in e.columns:
                e.loc[ix, c] = e.loc[perm, c].to_numpy()
    keys = e[["season", "round", "race_id"]].drop_duplicates()
    pc = pace.merge(keys, on=["season", "round"], how="left")
    late = (pc["race_id"] >= rid).to_numpy()
    num = [c for c in pc.columns if pc[c].dtype.kind == "f"]
    pc = pc.copy()
    for c in num:
        pc.loc[late, c] = pc.loc[late, c].to_numpy() * rng.uniform(0.5, 1.5, late.sum()) + rng.normal(0, 1, late.sum())
    return e, pc.drop(columns="race_id")


def pick_races(e0: pd.DataFrame):
    a = e0[e0["season"] == 2024]["race_id"].unique()
    b = e0[e0["season"] == 2026]["race_id"].unique()
    return int(a[len(a) // 2]), int(b[-3])


@needs_data
@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("cfg_name", ["final", "reverse_only"])
def test_scrambling_outcomes_laps_from_r_on_does_not_change_the_score_path_prediction(sc_ctx, monkeypatch, which, cfg_name):
    ctx = sc_ctx
    rid = pick_races(ctx["e0"])[which]
    base = score_path(ctx, ctx["e0"], ctx["pace"], cfg_name, rid, monkeypatch)
    e_s, pace_s = scramble_from(ctx["e0"], ctx["pace"], rid, seed=7 + which)
    assert (e_s["pos"].to_numpy() != ctx["e0"]["pos"].to_numpy()).any()           # the scramble really changed something
    got = score_path(ctx, e_s, pace_s, cfg_name, rid, monkeypatch)
    assert base.index.equals(got.index) and np.isfinite(base).all()
    assert (base.to_numpy() == got.to_numpy()).all(), float(np.abs(base - got).max())


@needs_data
@pytest.mark.parametrize("cfg_name", ["final", "reverse_only"])
def test_removing_later_rows_moves_scores_by_about_2e_4_and_never_changes_the_order(sc_ctx, monkeypatch, cfg_name):
    ctx = sc_ctx
    e0 = ctx["e0"]
    s24 = e0[e0["season"] == 2024]["race_id"].unique()
    s25 = e0[e0["season"] == 2025]["race_id"].unique()
    s26 = e0[e0["season"] == 2026]["race_id"].unique()
    races = [int(s24[5]), int(s24[18]), int(s25[10]), int(s26[4]), int(s26[-2]), int(s26[-1])]
    diffs, order_ok = [], []
    for rid in races:
        full = score_path(ctx, e0, ctx["pace"], cfg_name, rid, monkeypatch)
        cut = e0[e0["race_id"] <= rid].reset_index(drop=True)
        keys = e0[["season", "round", "race_id"]].drop_duplicates()
        pc = ctx["pace"].merge(keys, on=["season", "round"], how="left")
        pc = pc[pc["race_id"] <= rid].drop(columns="race_id").reset_index(drop=True)
        got = score_path(ctx, cut, pc, cfg_name, rid, monkeypatch)
        assert full.index.equals(got.index)
        diffs.append(float(np.abs(full - got).max()))
        order_ok.append(bool((full.rank(ascending=False, method="first").to_numpy() == got.rank(ascending=False, method="first").to_numpy()).all()))
    print(f"\n{cfg_name}: max |score(full) - score(truncated)| per race {dict(zip(races, np.round(diffs, 6)))}")
    assert max(diffs) < 5e-3          # documented ~2e-4; observed values are printed (-s) and given in the report
    assert all(order_ok), dict(zip(races, order_ok))


# ------------------------------------------------------------------------------------ 5. idempotence
@needs_data
def test_rerun_with_unchanged_data_is_a_noop_and_leaves_no_trace(tmp_path, capsys):
    """score.main on the real repository, with --no-write (so that even a failed no-op cannot write): it must take the no-op branch, which
    returns before any model is run; and the working tree, status files and data files are byte-identical afterwards."""
    sc = load_score()
    watched = [ROOT / "results/plan3/holdout_status.json", ROOT / "results/plan3/third_look.csv", ROOT / "results/plan3/leaderboard.csv",
               ROOT / "methodologies/P3_final/freeze_rowhash.json"] + [ROOT / r for r in PARQ + CSVS]
    before = {str(p): sha_raw(p) for p in watched}
    mtimes = {str(p): p.stat().st_mtime_ns for p in watched}
    runs_before = sorted(x.name for x in (HERE / "results").iterdir())
    st_before = git_status()
    rc = sc.main(["--no-write"])
    out = capsys.readouterr().out
    assert rc == 0 and "FRESH HOLDOUT RACE COUNT n = 0" in out and "no-op" in out
    assert "THIRD LOOK" not in out and "FRESH HOLDOUT: n" not in out
    assert before == {str(p): sha_raw(p) for p in watched} and mtimes == {str(p): p.stat().st_mtime_ns for p in watched}
    assert sorted(x.name for x in (HERE / "results").iterdir()) == runs_before and git_status() == st_before


@needs_data
def test_noop_also_with_writes_enabled_when_run_against_a_tmp_copy_of_the_outputs(tmp_path, monkeypatch, capsys):
    sc = load_score()
    plan3 = tmp_path / "plan3"
    plan3.mkdir()
    shutil.copy(ROOT / "results/plan3/holdout_status.json", plan3)
    res = tmp_path / "res"
    res.mkdir()
    lbtmp = tmp_path / "lb.csv"
    shutil.copy(ROOT / "results/plan3/leaderboard.csv", lbtmp)
    cfg = load_yaml(HERE / "config.yaml")
    cfg["leaderboard"] = str(lbtmp)                                                 # absolute path wins over ROOT / ...
    save_yaml(cfg, tmp_path / "cfg.yaml")
    monkeypatch.setattr(sc, "PLAN3", plan3)
    monkeypatch.setattr(sc, "RES", res)
    st0 = (plan3 / "holdout_status.json").read_bytes()
    lb0 = lbtmp.read_bytes()
    assert sc.main(["--config", str(tmp_path / "cfg.yaml")]) == 0
    assert "no-op" in capsys.readouterr().out
    assert (plan3 / "holdout_status.json").read_bytes() == st0 and lbtmp.read_bytes() == lb0
    assert list(res.iterdir()) == [] and sorted(p.name for p in plan3.iterdir()) == ["holdout_status.json"]


def test_a_missing_or_wrong_status_does_not_overwrite_an_existing_third_look(tmp_path, monkeypatch):
    """third_look.csv present but status missing: refuse (code path read: `SystemExit`)."""
    src = (HERE / "score.py").read_text(encoding="utf-8")
    assert "refusing to overwrite" in src and 'status.get("third_look_done")' in src


# ------------------------------------------------------------------------------------ 6. dry run of the fresh path (boundary moved to 2025 R3)
@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    """score.main on an in-memory copy of the data cut at the end of 2025, boundary after 2025 round 3 (21 'holdout' races). Every output
    goes to tmp (MDIR, RES, PLAN3, harness.RESULTS patched); the third look is marked done in a tmp status file so it is skipped."""
    if not HAVE:
        pytest.skip("data not available")
    from pitlane.evaluation import harness
    from pitlane.plan2 import pipeline2 as Q
    sc = load_score()
    tmp = tmp_path_factory.mktemp("dry")
    mdir, plan3 = tmp / "P3", tmp / "plan3"
    mdir.mkdir()
    plan3.mkdir()
    for f in ("frozen_config.yaml", "FREEZE.md", "freeze_rowhash.json", "HOLDOUT_PREREG.md"):
        shutil.copy(HERE / f, mdir / f)
    e_all = Q.load_all()
    b = int(e_all[(e_all["season"] == 2025) & (e_all["round"] == 3)]["race_id"].iloc[0])
    end = int(e_all[e_all["season"] == 2025]["race_id"].max())
    lb = tmp / "lb.csv"
    shutil.copy(ROOT / "results/plan3/leaderboard.csv", lb)
    cfg = load_yaml(HERE / "config.yaml")
    cfg["leaderboard"] = str(lb)
    cfg["score"]["last_race"] = dict(race_id=b, season=2025, round=3)
    save_yaml(cfg, tmp / "cfg.yaml")
    (plan3 / "holdout_status.json").write_text(json.dumps(dict(third_look_done=True, fresh_race_ids=[], n=0)), encoding="utf-8")
    mp = pytest.MonkeyPatch()
    mp.setattr(sc, "MDIR", mdir)
    mp.setattr(sc, "RES", mdir / "results")
    mp.setattr(sc, "PLAN3", plan3)
    mp.setattr(harness, "FRESH_AFTER", (2025, 3))
    mp.setattr(harness, "RESULTS", tmp / "results")
    mp.setattr(Q, "load_all", lambda rebuild=False: e_all[e_all["race_id"] <= end].reset_index(drop=True))
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = sc.main(["--config", str(tmp / "cfg.yaml")])
            first_out = buf.getvalue()
            buf.truncate(0)
            buf.seek(0)
            rc2 = sc.main(["--config", str(tmp / "cfg.yaml")])
    finally:
        mp.undo()
    second_out = buf.getvalue()
    return dict(sc=sc, tmp=tmp, plan3=plan3, mdir=mdir, b=b, end=end, e=e_all[e_all["race_id"] <= end], rc=rc, rc2=rc2, first=first_out, second=second_out,
                lb=lb, cfg=cfg)


@needs_data
def test_dry_run_counts_21_races_writes_one_verdict_and_second_run_is_a_noop(dry):
    assert dry["rc"] == 0 and dry["rc2"] == 0
    st = json.loads((dry["plan3"] / "holdout_status.json").read_text(encoding="utf-8"))
    ids = sorted(dry["e"][(dry["e"]["season"] == 2025) & (dry["e"]["round"] > 3)]["race_id"].unique())
    assert st["n"] == 21 and st["fresh_race_ids"] == [int(i) for i in ids] and ids[0] == dry["b"] + 1
    assert "FRESH HOLDOUT RACE COUNT n = 21" in dry["first"]
    assert sorted(p.name for p in dry["plan3"].glob("holdout_verdict_*")) == ["holdout_verdict_n20.json"]
    assert "no-op" in dry["second"] and "READING" not in dry["second"]
    assert not (dry["plan3"] / "holdout_verdict_n40.json").exists()


@needs_data
def test_dry_run_verdict_is_the_first_20_races_and_matches_an_independent_holm(dry):
    v = json.loads((dry["plan3"] / "holdout_verdict_n20.json").read_text(encoding="utf-8"))
    st = json.loads((dry["plan3"] / "holdout_status.json").read_text(encoding="utf-8"))
    assert v["level"] == 20 and v["n_races_read"] == 20 and v["n_races_available"] == 21 and v["race_ids"] == st["fresh_race_ids"][:20]
    rd = next((dry["mdir"] / "results").glob("*_p3_R15_score*"))
    p = pd.read_parquet(rd / "predictions.parquet")
    p = p[(p["scope"] == "fresh") & p["race_id"].isin(v["race_ids"])]
    tabs = {m: my_race_table(g).set_index("race_id") for m, g in p.groupby("model") if not m.startswith("S3_")}
    boots, gains = {}, {}
    for t, cand, ref, metric, hb in (("T1", "final", "latent_111", "spearman", 1), ("T2", "final", "final_minus_reverse", "mae", 0), ("T3", "final", "reverse_only", "spearman", 1)):
        d = (1 if hb else -1) * (tabs[cand][metric] - tabs[ref][metric].reindex(tabs[cand].index)).to_numpy()
        rng = np.random.default_rng([0, zlib.crc32(metric.encode())])
        boots[t] = d[rng.integers(0, 20, size=(10_000, 20))].mean(1)
        gains[t] = float(d.mean())
    mine = my_holm(boots)
    rec = {r["test"]: r for r in v["tests"]}
    for t in boots:
        assert abs(rec[t]["gain"] - gains[t]) < 1e-9, t
        assert abs(rec[t]["lo95"] - np.percentile(boots[t], 2.5)) < 1e-9 and abs(rec[t]["hi95"] - np.percentile(boots[t], 97.5)) < 1e-9
    assert v["confirmed"] == mine
    print("\nDRY RUN (boundary 2025 R3, 20 races read) T1-T3:", {t: (round(gains[t], 6), mine[t]) for t in gains})


@needs_data
def test_dry_run_seed_index_is_the_date_order_index_within_the_holdout(dry):
    """S3 probabilities of the first holdout race are drawn with seed 10000 (i = 0), the second with 10001, from the stored harness scores and
    the frozen theta, and equal the saved p_win exactly."""
    from pitlane.plan2 import simulate as S
    fc = load_yaml(HERE / "frozen_config.yaml")
    sim = fc["horizons"]["H2"]["simulator"]
    theta = {k: float(sim["theta"][k]) for k in S.DEFAULT}
    consts = {k: float(sim[k]) for k in ("stick_ref", "vol_mu", "vol_sd")}
    rd = next((dry["mdir"] / "results").glob("*_p3_R15_score*"))
    p = pd.read_parquet(rd / "predictions.parquet")
    p = p[p["scope"] == "fresh"]
    s3 = p[p["model"] == "S3_final"]
    fin = p[p["model"] == "final"][["race_id", "driverId", "score"]]
    cols = ["race_id", "driverId", "grid_norm", "stick", "circ_vol", "rain_fc"]
    fr = s3.drop(columns="score").merge(fin, on=["race_id", "driverId"]).merge(dry["e"][cols], on=["race_id", "driverId"])
    races = S.prepare(fr, consts)
    fresh_ids = json.loads((dry["plan3"] / "holdout_status.json").read_text(encoding="utf-8"))["fresh_race_ids"]
    assert [r["race_id"] for r in races] == fresh_ids
    for i in (0, 1, 7):
        r = races[i]
        P = S.simulate_race(r["u"], r["p_dnf"], r["grid"], r["stick_c"], r["zvol"], r["rain"], theta, S.SWITCHES["S3"], int(sim["draws"]),
                            np.random.default_rng(0 + 10_000 + i))
        saved = s3[s3["race_id"] == r["race_id"]].set_index("driverId").reindex(r["g"]["driverId"])
        assert np.allclose(saved["p_win"].to_numpy(), P[:, 0], atol=1e-12), i
        # and a wrong index (i + 1) would not reproduce it
        P2 = S.simulate_race(r["u"], r["p_dnf"], r["grid"], r["stick_c"], r["zvol"], r["rain"], theta, S.SWITCHES["S3"], int(sim["draws"]),
                             np.random.default_rng(10_000 + i + 1))
        assert not np.allclose(saved["p_win"].to_numpy(), P2[:, 0], atol=1e-12)


@needs_data
def test_dry_run_point_metrics_use_the_score_order_not_the_d0_order(dry):
    rd = next((dry["mdir"] / "results").glob("*_p3_R15_score*"))
    p = pd.read_parquet(rd / "predictions.parquet")
    per = pd.read_csv(rd / "per_race_metrics.csv")
    p, per = p[p["scope"] == "fresh"], per[per["scope"] == "fresh"]
    mine = my_race_table(p[p["model"] == "final"]).set_index("race_id")
    saved = per[per["model"] == "final"].set_index("race_id").reindex(mine.index)
    for k in ("spearman", "mae", "exact", "within1"):
        assert (saved[k] - mine[k]).abs().max() < 1e-12, k
    s3 = per[per["model"] == "S3_final"].set_index("race_id").reindex(mine.index)
    assert s3["rps"].notna().all() and (s3["spearman"] - saved["spearman"]).abs().max() > 1e-6        # a different order exists and is not used
    # leaderboard rows of the dry run: 8 models (final, reverse_only, final_minus_reverse, latent_111, H1, grid, 2 x S3) with the fresh label
    lb = pd.read_csv(dry["lb"])
    fresh = lb[lb["label"].str.contains("_fresh_n21")]
    assert len(fresh) == 8 and (fresh["split"] == "fresh").all()


@needs_data
def test_dry_run_leaves_the_repository_untouched(dry):
    assert not list((HERE / "results").glob("*fresh*")) and not (ROOT / "results/plan3/holdout_verdict_n20.json").exists()
    assert not (ROOT / "results/plan3/fresh_holdout_latest.csv").exists()


@needs_data
def test_rerun_cannot_overwrite_the_n20_verdict_even_if_status_is_lost(dry, monkeypatch):
    """Delete the tmp status and run again with the same data: the n20 verdict file is bytes-identical afterwards."""
    from pitlane.evaluation import harness
    from pitlane.plan2 import pipeline2 as Q
    sc = dry["sc"]
    before = (dry["plan3"] / "holdout_verdict_n20.json").read_bytes()
    (dry["plan3"] / "holdout_status.json").write_text(json.dumps(dict(third_look_done=True, fresh_race_ids=[], n=0)), encoding="utf-8")
    e_all = dry["e"]
    monkeypatch.setattr(sc, "MDIR", dry["mdir"])
    monkeypatch.setattr(sc, "RES", dry["mdir"] / "results")
    monkeypatch.setattr(sc, "PLAN3", dry["plan3"])
    monkeypatch.setattr(harness, "FRESH_AFTER", (2025, 3))
    monkeypatch.setattr(harness, "RESULTS", dry["tmp"] / "results")
    monkeypatch.setattr(Q, "load_all", lambda rebuild=False: e_all.reset_index(drop=True))
    with contextlib.redirect_stdout(io.StringIO()):
        assert sc.main(["--config", str(dry["tmp"] / "cfg.yaml")]) == 0
    assert (dry["plan3"] / "holdout_verdict_n20.json").read_bytes() == before


# ------------------------------------------------------------------------------------ 7. split discipline
def test_score_code_has_no_tuning_and_no_test_data_constants():
    src = (HERE / "score.py").read_text(encoding="utf-8") + (ROOT / "src/pitlane/plan3/holdout.py").read_text(encoding="utf-8")
    for bad in ("fit_theta", "fit_scale_sd", "minimize", "GridSearch", "optuna", "fit_tau"):
        assert bad not in src, bad
    fc = load_yaml(HERE / "frozen_config.yaml")
    assert "scale_sd" in fc["pace_feature"] and fc["pace_feature"]["scale_seasons_max"] == 2020      # scaling constant from the frozen file
    assert "theta" in fc["horizons"]["H2"]["simulator"]


def test_frozen_files_are_untouched_since_the_freeze_commit():
    for f in ("frozen_config.yaml", "HOLDOUT_PREREG.md", "FREEZE.md", "prestatement.md"):
        out = subprocess.run(["git", "diff", "--stat", "plan3-freeze", "--", f"methodologies/P3_final/{f}"], capture_output=True, text=True, cwd=ROOT).stdout
        assert out.strip() == "", (f, out)
