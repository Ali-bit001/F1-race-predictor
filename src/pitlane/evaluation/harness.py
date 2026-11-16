"""Walk-forward (prequential) evaluation, run writing and leaderboard (PLAN.md sections 2.3, 2.6).

To predict race r a model sees only entries from races < r. With refit_every=k the model is refit
once per chunk of k races (used only while tuning, to save time; final runs use k=1).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from pitlane.config import FRESH_AFTER, RESULTS, SPLITS, WARMUP_RACES, save_yaml
from pitlane.evaluation.metrics import (fit_tau, pos_dist, prob_scores, race_metrics, rps_score,
                                        sample_probs)


class Model(Protocol):
    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray: ...


def split_races(entries: pd.DataFrame, split: str) -> tuple[list[int], list[int]]:
    """Return (warmup race ids, counted race ids) for a split name or 'a+b' union of names."""
    races = entries.groupby("race_id")["season"].first()
    if split == "fresh":      # plan 2: races after FRESH_AFTER (possibly none yet)
        rnd = entries.groupby("race_id")["round"].first()
        counted = races.index[(races > FRESH_AFTER[0]) | ((races == FRESH_AFTER[0]) & (rnd > FRESH_AFTER[1]))].tolist()
        if not counted:
            return [], []
        first = counted[0]
        return [r for r in races.index if r < first][-WARMUP_RACES:], counted
    if split.startswith("seasons:"):   # plan 3 (A4): ad-hoc season range "seasons:2018-2020"; named splits unaffected
        lo, hi = (int(x) for x in split[len("seasons:"):].split("-"))
    else:
        lo = min(SPLITS[s][0] for s in split.split("+"))
        hi = max(SPLITS[s][1] for s in split.split("+"))
    counted = races[(races >= lo) & (races <= hi)].index.tolist()
    first = counted[0]
    warm = [r for r in races.index if r < first][-WARMUP_RACES:]
    return warm, counted


def walk_forward(entries: pd.DataFrame, model: Model, split: str, refit_every: int = 1,
                 seed: int = 0, n_samples: int = 1000, posdist: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (predictions per entry, metrics per counted race)."""
    assert (entries.index == np.arange(len(entries))).all(), "entries must have a RangeIndex"
    warm, counted = split_races(entries, split)
    if not counted:
        return pd.DataFrame(), pd.DataFrame()
    targets = warm + counted
    counted_set = set(counted)
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1)      # separate stream: plan-1 numbers stay reproducible
    race_rows = entries.groupby("race_id").indices
    scores = np.full(len(entries), np.nan)

    for i in range(0, len(targets), refit_every):
        chunk = targets[i:i + refit_every]
        test_rows = np.concatenate([race_rows[r] for r in chunk])
        first_row = test_rows.min()
        train = entries.iloc[:first_row]
        scores[test_rows] = model.fit_predict(train, entries.iloc[test_rows])

    preds, metrics = [], []
    hist_u, hist_pos = [], []
    for r in targets:
        rows = race_rows[r]
        d = entries.iloc[rows]
        u, pos = scores[rows], d["pos"].to_numpy()
        tau = fit_tau(hist_u[-200:], hist_pos[-200:], default=1.0)   # online calibration on earlier OOS races
        p = sample_probs(u, tau, rng, n_samples)
        if r in counted_set:
            m = race_metrics(pos, u, d["classified"].to_numpy())
            m.update(prob_scores(p, pos))
            if posdist:
                m["rps"] = rps_score(pos_dist(u, tau, rng2, min(n_samples, 400)), pos)
            m.update(race_id=r, season=int(d["season"].iloc[0]), round=int(d["round"].iloc[0]),
                     n=len(d), tau=tau)
            metrics.append(m)
            pr = d[["race_id", "season", "round", "driverId", "lineage", "pos", "grid", "classified"]].copy()
            pr["score"], pr["p_win"], pr["p_podium"], pr["p_points"] = u, p[:, 0], p[:, 1], p[:, 2]
            preds.append(pr)
        hist_u.append(u)
        hist_pos.append(pos)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(metrics)


SUMMARY_KEYS = ["spearman", "mae", "rmse", "kendall", "ndcg10", "win_hit", "podium", "points",
                "top5", "exact", "within1", "within2", "rps",
                "spearman_fin", "mae_fin", "brier_win", "logloss_win", "brier_podium", "logloss_podium",
                "brier_points", "logloss_points"]


def summarize(per_race: pd.DataFrame) -> dict:
    s = {k: float(per_race[k].mean()) for k in SUMMARY_KEYS if k in per_race}
    s["n_races"] = int(len(per_race))
    return s


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def save_run(method_dir: Path, label: str, config: dict, preds: pd.DataFrame, per_race: pd.DataFrame,
             split: str, method: str, cell: str = "", leaderboard: Path | None = None) -> str:
    """Write a run folder and append one leaderboard row. Returns run_id."""
    run_id = f"{datetime.now():%Y%m%d-%H%M}_{label}_{split}"
    out = method_dir / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    summ = summarize(per_race)
    save_yaml({**config, "git_commit": _git_hash(), "split": split}, out / "config.yaml")
    (out / "metrics.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    per_race.to_csv(out / "per_race_metrics.csv", index=False)
    preds.to_parquet(out / "predictions.parquet", index=False)

    lb = leaderboard or (RESULTS / "leaderboard.csv")
    row = pd.DataFrame([{"method": method, "cell": cell, "label": label, "split": split, "run_id": run_id,
                         "run_dir": str(out.relative_to(RESULTS.parent)), **summ}])
    if lb.exists():
        row = pd.concat([pd.read_csv(lb), row], ignore_index=True)
    row.to_csv(lb, index=False)
    return run_id
