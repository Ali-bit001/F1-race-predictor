"""Descriptive calibration checks for a driver x place probability matrix (R2 revision; PLAN3.md M17/M18).

These are diagnostics, not components: nothing here is tuned and nothing goes into attribution.csv.
`grid_slot_calibration` is the check any later simulator (M16/M18) must pass before a decoder is run on it:
the matrix must put roughly the realised share of probability on "finishes in the place it started".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pitlane.evaluation.metrics import pred_rank
from pitlane.plan3 import decode as DEC


def _as_lists(matrix, grid, pos, keys=None):
    if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
        return [matrix], [np.asarray(grid, float)], [np.asarray(pos, float)], [keys]
    ks = keys if keys is not None else [None] * len(matrix)
    return list(matrix), [np.asarray(g, float) for g in grid], [np.asarray(p, float) for p in pos], list(ks)


def grid_slots(grid: np.ndarray, key: np.ndarray | None = None) -> np.ndarray:
    """Place (1..n) each entry would hold if the race finished in starting order (pit-lane starts at the back)."""
    return pred_rank(DEC.grid_utility(grid, key)).astype(int)


def grid_slot_calibration(matrix, grid, pos, keys=None, n_bins: int = 5) -> dict:
    """P(driver finishes in the place it started) under the matrix vs the realised share.

    Accepts one race (P, grid, pos) or lists of them (pooled over entries). Returns
    {"mean_pred", "realised", "table"}; the table has `n_bins` equal-count bins of the predicted probability
    (columns bin, n, pred, obs)."""
    Ps, gs, ps, ks = _as_lists(matrix, grid, pos, keys)
    pred, obs = [], []
    for P, g, p, k in zip(Ps, gs, ps, ks):
        slot = grid_slots(g, k)
        pred.append(P[np.arange(len(g)), slot - 1])
        obs.append((p == slot).astype(float))
    pred, obs = np.concatenate(pred), np.concatenate(obs)
    bins = pd.qcut(pd.Series(pred).rank(method="first"), n_bins, labels=False)
    tab = pd.DataFrame({"bin": bins, "pred": pred, "obs": obs}).groupby("bin").agg(
        n=("pred", "size"), pred=("pred", "mean"), obs=("obs", "mean")).reset_index()
    return {"mean_pred": float(pred.mean()), "realised": float(obs.mean()), "table": tab}


def cell_calibration(mats: list[np.ndarray], pos: list[np.ndarray],
                     edges=(0, .01, .02, .05, .1, .2, .4, 1.0001)) -> pd.DataFrame:
    """All driver x place cells: predicted probability vs realised frequency, in fixed probability bins."""
    p = np.concatenate([P.ravel() for P in mats])
    y = np.concatenate([(np.arange(1, len(q) + 1)[None, :] == q[:, None]).ravel() for q in pos]).astype(float)
    b = np.digitize(p, edges) - 1
    rows = [dict(lo=edges[i], hi=edges[i + 1], n=int((b == i).sum()), pred=float(p[b == i].mean()),
                 obs=float(y[b == i].mean())) for i in range(len(edges) - 1) if (b == i).any()]
    return pd.DataFrame(rows)


def expected_exact(P: np.ndarray, util: np.ndarray) -> float:
    """Exact-hit rate the matrix itself expects for the order given by `util`."""
    rk = pred_rank(util).astype(int)
    return float(P[np.arange(len(rk)), rk - 1].mean())


def exact_reranked_finishers(util: np.ndarray, pos: np.ndarray, classified: np.ndarray) -> float:
    """Exact rate after re-ranking both the prediction and the outcome among classified entries only."""
    c = np.asarray(classified, bool)
    if c.sum() < 5:
        return float("nan")
    return float((pred_rank(util[c]) == stats.rankdata(pos[c])).mean())


def blend_with_grid(P: np.ndarray, grid: np.ndarray, w: float, key: np.ndarray | None = None) -> np.ndarray:
    """(1-w)*P + w*(point mass at each entry's grid slot). Stays doubly stochastic."""
    slot = grid_slots(grid, key)
    M = np.zeros_like(P)
    M[np.arange(len(slot)), slot - 1] = 1.0
    return (1 - w) * P + w * M
