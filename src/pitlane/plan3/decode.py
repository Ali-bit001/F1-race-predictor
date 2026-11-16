"""M17 — metric-matched decoding (PLAN3.md section 4). Nothing is fitted.

Input: the driver x position probability matrix P (P[i, j] = P(driver i finishes in place j+1)) of the plan-2
simulator. The matrix has no separate retirement state: a retired car is placed behind all finishers in random
order inside each draw, so its mass is spread over the last places, and every decoder places all n entries.

D0  sort by the stored score (plan 2 stores score = -E[position], so D0 and D1 are the same ordering)
D1  sort by expected position computed from the matrix
D2  assignment of drivers to places maximising the summed probability of the exact place
    (scipy linear_sum_assignment). Remaining ties are broken without the input row order (which is the finishing
    order, i.e. the outcome): first by closeness of the place to E[position] (1e-9 weight), then by a tiny term
    keyed on a row key (driverId if given, else the rank of the row's own values), 1e-11 weight. Two rows with
    identical values AND identical keys are interchangeable, so the result is the same set of hits.

Tie-breaking helper `break_ties(u, key)` adds a perturbation below 1e-9 in total, keyed on `key` (driverId), so that
exact ties of a utility are resolved by the key and never by row order. Plan 2 broke ties by row order; that
convention is still available (key=None) so the plan-2 numbers can be reproduced.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from pitlane.plan2 import simulate as S

TIE = 1e-9
KEY_TIE = 1e-11


def _key_rank(key) -> np.ndarray:
    """Dense rank 0..m-1 of a key array (strings or numbers); equal keys get equal rank."""
    return np.unique(np.asarray(key), return_inverse=True)[1].astype(float)


def break_ties(u: np.ndarray, key=None, eps: float = TIE) -> np.ndarray:
    """Add a perturbation keyed on `key` (lower key = better) to utility u (higher = better). key=None: unchanged
    (ties then fall back to row order in pred_rank)."""
    u = np.asarray(u, float)
    if key is None:
        return u
    r = _key_rank(key)
    return u + eps * (1.0 - r / (len(u) + 1)) / len(u)


def expected_position(P: np.ndarray) -> np.ndarray:
    return P @ np.arange(1, P.shape[1] + 1)


def decode_d1(P: np.ndarray, key=None) -> np.ndarray:
    """Utility (higher = better) = -E[position]; exact ties broken by `key` when given."""
    return break_ties(-expected_position(P), key)


def decode_d2(P: np.ndarray, key=None) -> np.ndarray:
    """Utility = -assigned place, so that pred_rank(utility) returns the assignment exactly. Independent of the
    row order of P: ties are resolved by `key` (driverId) or, if None, by the rank of each row's own values."""
    n = P.shape[0]
    places = np.arange(1, n + 1)
    if key is None:
        key = np.unique(np.round(P, 12), axis=0, return_inverse=True)[1].ravel()
    r = _key_rank(key) / (n + 1)
    cost = (-P + TIE * np.abs(places[None, :] - expected_position(P)[:, None])
            - KEY_TIE * r[:, None] * (places[None, :] / n))
    rows, cols = linear_sum_assignment(cost)
    place = np.empty(n)
    place[rows] = cols + 1
    return -place


def grid_utility(grid: np.ndarray, key=None) -> np.ndarray:
    """Starting-grid order; a grid value of 0 (pit-lane start) goes to the back. Ties (several pit-lane starts,
    duplicated grid values) are broken by `key` (driverId) when given; key=None keeps input order (plan-2
    convention, which is finishing order)."""
    g = np.asarray(grid, float)
    return break_ties(-np.where(g <= 0, g.max() + 1, g), key)


def simulate_matrices(races: list[dict], theta: dict, variant: str = "S3", k: int = 1000, seed: int = 0) -> list[np.ndarray]:
    """Same random streams as pitlane.plan2.simulate.evaluate (seed + 10000 + i), so P equals the frozen run."""
    sw = S.SWITCHES[variant]
    out = []
    for i, r in enumerate(races):
        rng = np.random.default_rng(seed + 10_000 + i)
        out.append(S.simulate_race(r["u"], r["p_dnf"], r["grid"], r["stick_c"], r["zvol"], r["rain"], theta, sw, k, rng))
    return out
