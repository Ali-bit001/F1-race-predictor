"""Per-race metrics (PLAN.md section 2.5) and Plackett-Luce probability helpers.

Convention: `u` is a utility, higher = predicted to finish better. `pos` is the actual finishing
position (1 = win). All metrics are computed per race; the unit of analysis is the race.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar


def pred_rank(u: np.ndarray) -> np.ndarray:
    """1 = predicted best. Ties broken by input order (stable)."""
    order = np.argsort(-u, kind="stable")
    r = np.empty(len(u))
    r[order] = np.arange(1, len(u) + 1)
    return r


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def ndcg_at(rank_pred: np.ndarray, pos: np.ndarray, k: int = 10) -> float:
    n = len(pos)
    rel = (n - pos).astype(float)
    order = np.argsort(rank_pred)
    disc = 1 / np.log2(np.arange(2, k + 2))
    kk = min(k, n)
    dcg = (rel[order][:kk] * disc[:kk]).sum()
    ideal = (np.sort(rel)[::-1][:kk] * disc[:kk]).sum()
    return float(dcg / ideal) if ideal > 0 else np.nan


def race_metrics(pos: np.ndarray, u: np.ndarray, classified: np.ndarray) -> dict[str, float]:
    pos = np.asarray(pos, float)
    rk = pred_rank(u)
    out = {
        "spearman": _spearman(rk, pos),
        "kendall": float(stats.kendalltau(rk, pos)[0]) if len(pos) >= 3 else np.nan,
        "mae": float(np.abs(rk - pos).mean()),
        "rmse": float(np.sqrt(((rk - pos) ** 2).mean())),
        "ndcg10": ndcg_at(rk, pos, 10),
        "win_hit": float(rk[pos == 1][0] == 1) if (pos == 1).any() else np.nan,
        "podium": len(set(np.where(rk <= 3)[0]) & set(np.where(pos <= 3)[0])) / 3,
        "points": len(set(np.where(rk <= 10)[0]) & set(np.where(pos <= 10)[0])) / 10,
        # plan-2 additions (PLAN2.md section 5): `points` above is the top-10 overlap
        "top5": len(set(np.where(rk <= 5)[0]) & set(np.where(pos <= 5)[0])) / 5,
        "exact": float((rk == pos).mean()),
        "within1": float((np.abs(rk - pos) <= 1).mean()),
        "within2": float((np.abs(rk - pos) <= 2).mean()),
    }
    # Finishers-only view (DNFs are mostly noise): re-rank both sides among classified cars.
    c = np.asarray(classified, bool)
    if c.sum() >= 5:
        rk_f = pred_rank(u[c])
        pos_f = stats.rankdata(pos[c])
        out["spearman_fin"] = _spearman(rk_f, pos_f)
        out["mae_fin"] = float(np.abs(rk_f - pos_f).mean())
    else:
        out["spearman_fin"] = out["mae_fin"] = np.nan
    return out


# ---------------------------------------------------------------- Plackett-Luce helpers
def pl_loglik(u_list: list[np.ndarray], pos_list: list[np.ndarray], tau: float) -> float:
    """Sum of PL log-likelihoods of observed orders under utilities tau*u."""
    total = 0.0
    for u, pos in zip(u_list, pos_list):
        v = tau * u[np.argsort(pos, kind="stable")]
        m = v.max()
        s = np.cumsum(np.exp(v - m)[::-1])[::-1]
        total += float((v - m - np.log(s)).sum())
    return total


def fit_tau(u_list: list[np.ndarray], pos_list: list[np.ndarray], default: float = 1.0) -> float:
    if len(u_list) < 5:
        return default
    res = minimize_scalar(lambda t: -pl_loglik(u_list, pos_list, t), bounds=(0.05, 30.0), method="bounded")
    return float(res.x)


def sample_probs(u: np.ndarray, tau: float, rng: np.random.Generator, k: int = 1000) -> np.ndarray:
    """Monte-Carlo P(win), P(top3), P(top10) per entry from PL(tau*u). Gumbel-max trick."""
    g = rng.gumbel(size=(k, len(u)))
    order = np.argsort(-(tau * u + g), axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(k)[:, None]
    ranks[rows, order] = np.arange(1, len(u) + 1)[None, :]
    return np.stack([(ranks == 1).mean(0), (ranks <= 3).mean(0), (ranks <= 10).mean(0)], axis=1)


def prob_scores(p: np.ndarray, pos: np.ndarray) -> dict[str, float]:
    """Brier and log-loss for the three events, averaged over entries of one race."""
    eps = 1e-4
    out = {}
    for j, (name, thr) in enumerate([("win", 1), ("podium", 3), ("points", 10)]):
        y = (pos <= thr).astype(float)
        q = np.clip(p[:, j], eps, 1 - eps)
        out[f"brier_{name}"] = float(((q - y) ** 2).mean())
        out[f"logloss_{name}"] = float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
    return out


def pos_dist(u: np.ndarray, tau: float, rng: np.random.Generator, k: int = 1000) -> np.ndarray:
    """Position distribution P[i, j] = P(entry i finishes in position j+1) under PL(tau*u)."""
    n = len(u)
    g = rng.gumbel(size=(k, n))
    order = np.argsort(-(tau * u + g), axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(k)[:, None], order] = np.arange(n)[None, :]
    return rank_hist(ranks, n)


def rank_hist(ranks0: np.ndarray, n: int) -> np.ndarray:
    """ranks0: [draws, n] 0-based finishing positions -> P[i, j] histogram over draws."""
    k = ranks0.shape[0]
    P = np.zeros((n, n))
    for i in range(n):
        P[i] = np.bincount(ranks0[:, i], minlength=n)[:n] / k
    return P


def rps_score(P: np.ndarray, pos: np.ndarray) -> float:
    """Ranked probability score over the position distribution (0 = perfect), averaged over entries."""
    n = P.shape[0]
    F = np.cumsum(P, axis=1)
    O = (np.arange(1, n + 1)[None, :] >= np.asarray(pos)[:, None]).astype(float)
    return float((((F - O) ** 2)[:, :-1].sum(1) / max(n - 1, 1)).mean())


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)
