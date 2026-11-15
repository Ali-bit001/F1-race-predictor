"""M07 — DNF hazard model (PLAN2.md section 4): logistic regression for "this entry is not classified",
fitted walk-forward (before each race, on all earlier races). Coefficients are the explanation.

Baselines: (a) constant expanding retirement rate, (b) the plan-1 decayed driver/constructor rate average.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from pitlane.config import METHODS, PLAN2, ROOT, save_yaml
from pitlane.evaluation.harness import _git_hash, split_races

FEATURES = ["drv_inc_ew16", "drv_dnf_ew16", "con_mech_ew16", "con_dnf_ew16", "circ_dnf", "grid_norm", "pit_start",
            "wet_q", "rain_fc", "newregs", "rookie", "new_team", "fp_laps", "fp_missing"]
LB_DNF = PLAN2 / "leaderboard_dnf.csv"
EPS = 1e-4
C_GRID = [0.02, 0.1, 0.5, 2.0]


def design(e: pd.DataFrame) -> np.ndarray:
    return e[FEATURES].fillna(0.0).to_numpy(float)     # missing weather / practice -> 0 (+ fp_missing, wx indicators)


def _scores(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    q = np.clip(p, EPS, 1 - EPS)
    out = {"brier": float(((q - y) ** 2).mean()), "logloss": float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())}
    out["auc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
    return out


def walk_dnf(entries: pd.DataFrame, split: str, C: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    warm, counted = split_races(entries, split)
    if not counted:
        return pd.DataFrame(), pd.DataFrame(), {}
    X, y = design(entries), entries["dnf"].to_numpy(float)
    race = entries["race_id"].to_numpy()
    rate_base = (0.5 * (entries["drv_dnf_ew16"] + entries["con_dnf_ew16"])).to_numpy(float)
    preds, per, coef = [], [], {}
    for r in counted:
        tr, te = (race < r) & ~np.isnan(y), race == r
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=500).fit(sc.transform(X[tr]), y[tr])
        p = m.predict_proba(sc.transform(X[te]))[:, 1]
        const = np.full(te.sum(), y[tr].mean())
        d = entries.loc[te, ["race_id", "season", "round", "driverId", "lineage", "pos", "classified", "dnf"]].copy()
        d["p_dnf"], d["p_const"], d["p_rate"] = p, const, rate_base[te]
        preds.append(d)
        row = {"race_id": r, "season": int(d["season"].iloc[0]), "round": int(d["round"].iloc[0]), "n_dnf": int(y[te].sum())}
        for tag, q in (("", p), ("_const", const), ("_rate", rate_base[te])):
            row.update({f"{k}{tag}": v for k, v in _scores(q, y[te]).items()})
        per.append(row)
        coef = dict(zip(FEATURES, (m.coef_[0]).round(4).tolist()), intercept=float(m.intercept_[0]), upto_race=int(r))
    return pd.concat(preds, ignore_index=True), pd.DataFrame(per), coef


def tune_C(entries: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    for C in C_GRID:
        _, per, _ = walk_dnf(entries, "dev", C)
        rows.append({"C": C, "logloss": per["logloss"].mean(), "brier": per["brier"].mean()})
    t = pd.DataFrame(rows)
    return float(t.loc[t["logloss"].idxmin(), "C"]), t


def save_dnf_run(preds: pd.DataFrame, per: pd.DataFrame, coef: dict, split: str, C: float) -> None:
    run_id = f"{datetime.now():%Y%m%d-%H%M}_dnf_hazard_{split}"
    out = METHODS / "M07_dnf_hazard" / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    save_yaml({"kind": "dnf_logit", "C": C, "features": FEATURES, "git_commit": _git_hash(), "split": split}, out / "config.yaml")
    summ = {k: float(per[k].mean()) for k in per.columns if k.startswith(("brier", "logloss", "auc"))}
    summ["n_races"] = int(len(per))
    (out / "metrics.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    (out / "coefficients.json").write_text(json.dumps(coef, indent=2), encoding="utf-8")
    per.to_csv(out / "per_race_metrics.csv", index=False)
    preds.to_parquet(out / "predictions.parquet", index=False)
    row = pd.DataFrame([{"label": "dnf_hazard", "split": split, "run_id": run_id, "run_dir": str(out.relative_to(ROOT)),
                         "C": C, **summ}])
    if LB_DNF.exists():
        row = pd.concat([pd.read_csv(LB_DNF), row], ignore_index=True)
    PLAN2.mkdir(parents=True, exist_ok=True)
    row.to_csv(LB_DNF, index=False)
