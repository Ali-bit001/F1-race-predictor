"""M08 — race-level Monte Carlo simulator (PLAN2.md section 4).

One simulated race (K draws per race, fixed seeds):
  1. retirements: each entry retires with probability dnf_scale * p_dnf (M07); retirees finish behind all
     finishers in random order                                                        [switch "dnf"]
  2. pace: tau * (M06 score + kappa * stickiness_c * grid_z) + Gumbel noise            [kappa: switch "stick"]
     (Gumbel noise = Plackett-Luce sampling)
  3. race regime: with probability p_dis (rises with the circuit's historical grid volatility and with a
     rain forecast) the draw is a disrupted race and the noise is s_dis times larger  [switch "regime"]
The output is a full driver x position probability matrix. Parameters are fitted on the development
block only (minimising mean RPS) and then frozen. This is a race-level model: tyres, pit stops and
overtaking are deliberately not simulated (PLAN2.md section 2, "Decided against").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitlane.evaluation.metrics import prob_scores, race_metrics, rank_hist, rps_score

SWITCHES = {"S0": dict(dnf=False, stick=False, regime=False),
            "S1": dict(dnf=True, stick=False, regime=False),
            "S2": dict(dnf=True, stick=True, regime=False),
            "S3": dict(dnf=True, stick=True, regime=True)}
GRID = {"tau": [0.8, 1.2, 1.7, 2.5, 3.5, 5.0], "dnf_scale": [0.6, 0.8, 1.0, 1.25, 1.5],
        "kappa": [-8.0, -4.0, 0.0, 4.0, 8.0, 16.0], "p0": [0.05, 0.10, 0.20, 0.35],
        "s_dis": [1.5, 2.0, 3.0], "a_vol": [0.0, 0.5, 1.0], "a_rain": [0.0, 1.0, 2.0]}
ACTIVE = {"S0": ["tau"], "S1": ["tau", "dnf_scale"], "S2": ["tau", "dnf_scale", "kappa"],
          "S3": ["tau", "dnf_scale", "kappa", "p0", "s_dis", "a_vol", "a_rain"]}
DEFAULT = dict(tau=2.0, dnf_scale=1.0, kappa=0.0, p0=0.10, s_dis=2.0, a_vol=0.0, a_rain=0.0)


def simulate_race(u: np.ndarray, p_dnf: np.ndarray, grid_norm: np.ndarray, stick_c: float, zvol: float, rain: float,
                  theta: dict, sw: dict, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(u)
    z = u + (theta["kappa"] * stick_c * (0.5 - grid_norm) if sw["stick"] else 0.0)
    g = rng.gumbel(size=(k, n))
    if sw["regime"]:
        logit = np.log(theta["p0"] / (1 - theta["p0"])) + theta["a_vol"] * zvol + theta["a_rain"] * rain
        dis = rng.random(k) < 1 / (1 + np.exp(-logit))
        g = g * np.where(dis, theta["s_dis"], 1.0)[:, None]
    key = theta["tau"] * z[None, :] + g
    if sw["dnf"]:
        retire = rng.random((k, n)) < np.clip(p_dnf * theta["dnf_scale"], 0.0, 0.95)[None, :]
        key = np.where(retire, -1e6 + rng.random((k, n)), key)
    order = np.argsort(-key, axis=1)
    ranks0 = np.empty_like(order)
    ranks0[np.arange(k)[:, None], order] = np.arange(n)[None, :]
    return rank_hist(ranks0, n)


def prepare(inputs: pd.DataFrame, consts: dict) -> list[dict]:
    """Group the joined (M06 score, M07 p_dnf, circuit and weather columns) frame into per-race arrays."""
    out = []
    for rid, g in inputs.groupby("race_id", sort=True):
        out.append(dict(
            race_id=rid, g=g, u=g["score"].to_numpy(float), p_dnf=g["p_dnf"].to_numpy(float),
            grid=g["grid_norm"].to_numpy(float), pos=g["pos"].to_numpy(float), cls=g["classified"].to_numpy(bool),
            stick_c=float(g["stick"].iloc[0] - consts["stick_ref"]),
            zvol=float((g["circ_vol"].iloc[0] - consts["vol_mu"]) / consts["vol_sd"]),
            rain=float(np.nan_to_num(g["rain_fc"].iloc[0], nan=0.0))))
    return out


def mean_rps(races: list[dict], theta: dict, sw: dict, k: int = 300) -> float:
    vals = []
    for i, r in enumerate(races):
        rng = np.random.default_rng(1000 + i)               # common random numbers across candidate thetas
        P = simulate_race(r["u"], r["p_dnf"], r["grid"], r["stick_c"], r["zvol"], r["rain"], theta, sw, k, rng)
        vals.append(rps_score(P, r["pos"]))
    return float(np.mean(vals))


def fit_theta(races: list[dict], variant: str, passes: int = 2, k: int = 300) -> tuple[dict, float]:
    """Coordinate search over small grids, objective = mean RPS on the fitting races (development only)."""
    sw, theta = SWITCHES[variant], dict(DEFAULT)
    best = mean_rps(races, theta, sw, k)
    for _ in range(passes):
        for name in ACTIVE[variant]:
            for v in GRID[name]:
                cand = {**theta, name: v}
                s = mean_rps(races, cand, sw, k)
                if s < best - 1e-9:
                    best, theta = s, cand
    return theta, best


def evaluate(races: list[dict], theta: dict, variant: str, k: int = 1000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply frozen parameters; returns (predictions per entry, metrics per race) in the harness format."""
    sw = SWITCHES[variant]
    preds, per = [], []
    for i, r in enumerate(races):
        rng = np.random.default_rng(seed + 10_000 + i)
        P = simulate_race(r["u"], r["p_dnf"], r["grid"], r["stick_c"], r["zvol"], r["rain"], theta, sw, k, rng)
        n = len(r["u"])
        e_pos = P @ np.arange(1, n + 1)
        pw, p3, p10 = P[:, 0], P[:, :3].sum(1), P[:, :min(10, n)].sum(1)
        score = -e_pos
        m = race_metrics(r["pos"], score, r["cls"])
        m.update(prob_scores(np.stack([pw, p3, p10], axis=1), r["pos"]))
        m["rps"] = rps_score(P, r["pos"])
        g = r["g"]
        m.update(race_id=r["race_id"], season=int(g["season"].iloc[0]), round=int(g["round"].iloc[0]), n=n, tau=theta["tau"])
        per.append(m)
        d = g[["race_id", "season", "round", "driverId", "lineage", "pos", "grid", "classified"]].copy()
        d["score"], d["p_win"], d["p_podium"], d["p_points"], d["p_dnf"] = score, pw, p3, p10, r["p_dnf"]
        preds.append(d)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(per)
