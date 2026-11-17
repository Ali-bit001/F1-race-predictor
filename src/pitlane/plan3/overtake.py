"""M18 — overtaking dead-band on the strength difference (PLAN3.md section 4, M18; R7).

Reading chosen (A3 failed, so there is no pace offset; the dead-band acts on the model's strength difference):
sort the field by starting grid (pit-lane starts at the back, ties by driverId, so row order = finishing order is
never used). For each car k and the car ahead of it on the grid, diff_k = u_k - u_(k-1), where u is the M06 score
the simulator already uses. A car behind that is stronger by less than thr = d * stick[circuit] cannot pass: only
the part of the advantage above thr counts, max(0, diff_k - thr). Negative differences (car ahead is stronger)
count fully. The adjusted score is u_adj_k = u_k - sum_{j <= k} min(max(diff_j, 0), thr), and u_adj replaces u in the
unchanged plan-2 simulator. `stick` is the plan-2 circuit grid stickiness (shrunk mean Spearman of grid vs finish
at earlier races of the circuit, `features/weekend.py`), uncentred. d = 0 returns u itself (bitwise).
"""
from __future__ import annotations

import numpy as np

from pitlane.plan2 import simulate as S


def grid_order(grid: np.ndarray, key: np.ndarray | None = None) -> np.ndarray:
    """Row indices sorted by starting grid, grid <= 0 (pit lane) at the back; ties by `key` (never row order)."""
    g = np.asarray(grid, float)
    g = np.where(g <= 0, g.max() + 1, g)
    if key is None:
        return np.argsort(g, kind="stable")
    return np.lexsort((np.unique(np.asarray(key), return_inverse=True)[1], g))


def deadband_scores(u: np.ndarray, grid: np.ndarray, key: np.ndarray | None, thr: float) -> np.ndarray:
    """Adjusted strengths; see the module docstring. thr <= 0 returns `u` unchanged."""
    if thr <= 0:
        return u
    o = grid_order(grid, key)
    adv = np.maximum(np.diff(u[o], prepend=u[o][0]), 0.0)     # advantage of each car over the car ahead
    red = np.cumsum(np.minimum(adv, thr))
    out = np.empty_like(u)
    out[o] = u[o] - red
    return out


def simulate_matrices_db(races: list[dict], theta: dict, d: float, variant: str = "S3", k: int = 1000,
                         seed: int = 0, base: int = 10_000) -> list[np.ndarray]:
    """Position matrices with the dead-band at the same random streams as pitlane.plan3.decode.simulate_matrices
    (seed + 10000 + i)."""
    sw = S.SWITCHES[variant]
    out = []
    for i, r in enumerate(races):
        u = deadband_scores(r["u"], r["g"]["grid"].to_numpy(float), r["g"]["driverId"].to_numpy(), d * float(r["g"]["stick"].iloc[0]))
        rng = np.random.default_rng(seed + base + i)
        out.append(S.simulate_race(u, r["p_dnf"], r["grid"], r["stick_c"], r["zvol"], r["rain"], theta, sw, k, rng))
    return out


def mean_rps_db(races: list[dict], theta: dict, d: float, variant: str = "S3", k: int = 300) -> float:
    """Tuning objective: as plan-2 `mean_rps` (streams 1000 + i, 300 draws) with the dead-band."""
    mats = simulate_matrices_db(races, theta, d, variant, k, seed=0, base=1000)
    return float(np.mean([S.rps_score(P, r["pos"]) for P, r in zip(mats, races)]))
