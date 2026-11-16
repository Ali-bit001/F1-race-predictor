"""Small helpers shared by plan-3 run scripts (same behaviour as the ones in methodologies/M11_likelihood_variants/run.py)."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from pitlane.config import ROOT
from pitlane.evaluation.harness import _git_hash
from pitlane.plan3 import decode as DEC


def code_stamp(used_files: list[str]) -> str:
    """`<hash>` if the files used are clean, else `<hash>+dirty-<sha1 of the used files>`."""
    head = _git_hash()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--"] + used_files, text=True, cwd=ROOT).strip()
    if not dirty:
        return head
    h = hashlib.sha1()
    for f in used_files:
        h.update(f.encode())
        h.update((ROOT / f).read_bytes())
    return f"{head}+dirty-{h.hexdigest()[:12]}"


def outcome_free(preds: pd.DataFrame, key: str = "driverId") -> pd.DataFrame:
    """Scores with exact ties broken by `key` (driverId), never by input row order (= finishing order)."""
    out = preds.copy()
    sc = np.empty(len(out))
    for _, ix in out.groupby("race_id").indices.items():
        sc[ix] = DEC.break_ties(out["score"].to_numpy(float)[ix], out[key].to_numpy()[ix])
    out["score"] = sc
    return out


def stored_run(leaderboard: str, label: str, split: str = "dev") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-race metrics, predictions) of the last stored run with this label and split."""
    lb = pd.read_csv(ROOT / leaderboard)
    row = lb[(lb["label"] == label) & (lb["split"] == split)].iloc[-1]
    d = ROOT / row["run_dir"]
    return pd.read_csv(d / "per_race_metrics.csv"), pd.read_parquet(d / "predictions.parquet")


def max_diff(new: pd.DataFrame, old: pd.DataFrame) -> tuple[float, int]:
    """Largest absolute difference over all shared numeric columns, matched by race_id."""
    a, b = new.set_index("race_id"), old.set_index("race_id")
    cols = [c for c in a.columns if c in b.columns]
    common = a.index.intersection(b.index)
    d = (a.loc[common, cols].astype(float) - b.loc[common, cols].astype(float)).abs()
    return float(np.nanmax(d.to_numpy())), len(common)


def score_diff(new: pd.DataFrame, old: pd.DataFrame) -> tuple[float, int]:
    m = new.merge(old, on=["race_id", "driverId"], suffixes=("_n", "_o"))
    return float((m["score_n"] - m["score_o"]).abs().max()), len(m)


def append_leaderboard(rows: pd.DataFrame, lb_path: Path) -> None:
    """Re-read the header right before writing; append only, existing rows are not rewritten."""
    header = pd.read_csv(lb_path, nrows=0).columns.tolist()
    assert not set(rows["run_id"]) & set(pd.read_csv(lb_path, usecols=["run_id"])["run_id"]), "run_id already in leaderboard"
    add = rows.reindex(columns=header)
    text = lb_path.read_text(encoding="utf-8")
    with open(lb_path, "a", encoding="utf-8", newline="") as f:
        if not text.endswith("\n"):
            f.write("\n")
        add.to_csv(f, index=False, header=False, lineterminator="\n")
