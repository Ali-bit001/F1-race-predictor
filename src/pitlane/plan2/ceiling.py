"""A1 — ceiling analysis (PLAN2.md section 4). Development block only; uses plan-1 run outputs.

Three numbers for the plan-1 proposed model (`latent_111`):
  1. self-consistency ceiling: rho of the model against orders sampled from its own calibrated
     Plackett-Luce distribution (what a perfectly specified model of this sharpness would score);
  2. retirement oracle: the same model, but told who retires (retirees placed at their true positions);
  3. race-pace oracle: classified finishers ranked by their median clean race lap, on the dev season
     for which lap times were fetched (2023). Oracles use race data on purpose and never enter a
     leaderboard.
Gate (PLAN2.md section 5): if the race-pace oracle beats the model by < 0.03 rho on finishers, order
accuracy is treated as saturated.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pitlane.config import PLAN2, ROOT
from pitlane.data.ingest2 import RAW2
from pitlane.evaluation.metrics import pred_rank

GATE = 0.03


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _lap_s(t: str) -> float:
    m, s = t.split(":") if ":" in t else ("0", t)
    return float(m) * 60 + float(s)


def pace_oracle_rank(laps: list, drivers: list[str]) -> dict[str, float]:
    """Median clean race lap per driver (lap 1 and laps > 107% of the race median dropped)."""
    df = pd.DataFrame(laps, columns=["lap", "driverId", "t"])
    df["s"] = df["t"].map(_lap_s)
    df = df[df["lap"] >= 2]
    df = df[df["s"] <= 1.07 * df["s"].median()]
    med = df.groupby("driverId")["s"].agg(["median", "size"])
    return {d: float(med.loc[d, "median"]) for d in drivers if d in med.index and med.loc[d, "size"] >= 5}


def run_ceiling(lb1: pd.DataFrame, seed: int = 0, k: int = 300) -> dict:
    row = lb1[(lb1["label"] == "latent_111") & (lb1["split"] == "dev")].iloc[0]
    run = ROOT / row["run_dir"]
    preds = pd.read_parquet(run / "predictions.parquet")
    per = pd.read_csv(run / "per_race_metrics.csv").set_index("race_id")
    rng = np.random.default_rng(seed)
    rows = []
    for rid, g in preds.groupby("race_id"):
        u, pos = g["score"].to_numpy(float), g["pos"].to_numpy(float)
        cls = g["classified"].to_numpy(bool)
        tau = float(per.loc[rid, "tau"])
        rk = pred_rank(u)
        # 1. self-consistency
        gum = rng.gumbel(size=(k, len(u)))
        order = np.argsort(-(tau * u + gum), axis=1)
        sampled = np.empty_like(order)
        sampled[np.arange(k)[:, None], order] = np.arange(1, len(u) + 1)[None, :]
        self_rho = float(np.nanmean([_rho(rk, s) for s in sampled]))
        # 2. retirement oracle: finishers keep the model's relative order, retirees sit at their true places
        orc = pos.copy()
        if cls.sum() >= 3:
            orc[cls] = pred_rank(u[cls])
        rows.append(dict(race_id=rid, season=int(g["season"].iloc[0]), model_rho=_rho(rk, pos),
                         self_consistency_rho=self_rho, retirement_oracle_rho=_rho(orc, pos),
                         model_rho_fin=float(per.loc[rid, "spearman_fin"])))
    d = pd.DataFrame(rows)
    # 3. pace oracle where lap times exist
    pace = []
    for rid, g in preds.groupby("race_id"):
        f = RAW2 / "laps" / f"laps_{int(g['season'].iloc[0])}_{int(g['round'].iloc[0])}.json"
        if not f.exists():
            continue
        fin = g[g["classified"]]
        med = pace_oracle_rank(json.loads(f.read_text(encoding="utf-8")), fin["driverId"].tolist())
        sub = fin[fin["driverId"].isin(med)]
        if len(sub) < 8:
            continue
        t = sub["driverId"].map(med).to_numpy()
        pace.append(dict(race_id=rid, pace_oracle_rho_fin=_rho(pred_rank(-t), sub["pos"].to_numpy(float))))
    pace_df = pd.DataFrame(pace)
    d = d.merge(pace_df, on="race_id", how="left")
    PLAN2.mkdir(parents=True, exist_ok=True)
    d.to_csv(PLAN2 / "ceiling_per_race.csv", index=False)
    sub = d.dropna(subset=["pace_oracle_rho_fin"])
    out = dict(
        n_races=int(len(d)), model_rho=float(d["model_rho"].mean()),
        self_consistency_rho=float(d["self_consistency_rho"].mean()),
        retirement_oracle_rho=float(d["retirement_oracle_rho"].mean()),
        retirement_oracle_gain=float((d["retirement_oracle_rho"] - d["model_rho"]).mean()),
        pace_races=int(len(sub)),
        pace_model_rho_fin=float(sub["model_rho_fin"].mean()) if len(sub) else None,
        pace_oracle_rho_fin=float(sub["pace_oracle_rho_fin"].mean()) if len(sub) else None,
    )
    out["pace_gap_fin"] = (out["pace_oracle_rho_fin"] - out["pace_model_rho_fin"]) if len(sub) else None
    out["saturated"] = bool(out["pace_gap_fin"] is not None and out["pace_gap_fin"] < GATE)
    (PLAN2 / "ceiling.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
