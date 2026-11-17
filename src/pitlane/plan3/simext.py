"""M16 simulator extensions S4-S6 (PLAN3.md section 4), as off-by-default switches on the plan-2 S3 simulator.

All three extensions share the random streams of plan-2 S3: the main generator `rng` is consumed in exactly the
order `pitlane.plan2.simulate.simulate_race` consumes it (Gumbel draws, regime uniform, retirement uniforms, retirement
tie-break uniforms). Anything new is drawn from a separate generator `aux`, so an extension never shifts the S3 stream.
With `ext = {}` (or S6 with c = 0) the matrix is bitwise the plan-2 S3 matrix (checked by run.py --check).

ext = {"kind": "S4", "m2": .., "m3": .., "m4": ..}
    Position-dependent discount (Henery / Stern / Benter type). The finishers are placed stage by stage: the winner
    is the arg-max of tau*z + Gumbel (the S3 key, so stage 1 is plain S3), the 2nd and 3rd place come from fresh
    Gumbel draws with temperature tau*m2 and tau*m3, the rest of the field is ordered with tau*m4. Retirees stay
    behind all finishers with the S3 tie-break. m = 1 everywhere is distributionally S3 but NOT the same draws
    (fresh Gumbel draws for stages 2+), so a neutral S4 differs from S3 by simulation noise only.
ext = {"kind": "S5", "sigma": .., "tau_team": .., "e_exp": .., "e_wet": ..}
    Normal performance draws with a team shock: key = tau*z + sigma_i*eps_i + tau_team*eta_team, with
    sigma_i = sigma * (1 + e_exp * [driver has < ROOKIE_STARTS earlier starts]) * (1 + e_wet * rain_fc).
    eps is the S3 Gumbel draw pushed through the quantile map Phi^-1(exp(-exp(-g))) (same stream, normal marginal);
    eta is drawn from `aux`. The disrupted-race regime scales the whole noise (as in S3). Retirements as in S3.
    sigma = pi/sqrt(6) (= the Gumbel sd), tau_team = e_exp = e_wet = 0 is the plain Thurstone-normal version.
ext = {"kind": "S6", "c": .., "c_early": ..}
    Parameter uncertainty: before each draw the strengths are perturbed by c_eff * N(0, Sigma_race), Sigma_race = the
    Laplace covariance of the entries' utilities u_i = a + b + g + beta.x (race["L"] = its Cholesky-type factor, from
    the walk-forward fit that produced the score), c_eff = c * (1 + c_early * [round <= EARLY_ROUNDS]).
    c = 0 is S3, bitwise.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import ndtri

from pitlane.evaluation.metrics import rank_hist, rps_score

GUMBEL_SD = float(np.pi / np.sqrt(6.0))
ROOKIE_STARTS = 20       # S5: "inexperienced" = fewer earlier starts than this (known before the race)
EARLY_ROUNDS = 3         # S6: the extra widening applies to the first rounds of a season
AUX_TAG = 7              # aux stream = default_rng([stream seed, AUX_TAG])


# ------------------------------------------------------------------------------------ race inputs
def add_extras(races: list[dict], entries: pd.DataFrame) -> list[dict]:
    """Add team code, driver experience and the early-season flag to the plan-2 per-race dicts (in place, returned).
    Experience = number of earlier rows of the driver in `entries` (rows are sorted by race_id), i.e. pre-race."""
    starts = entries.groupby("driverId").cumcount()
    key = pd.Series(starts.to_numpy(), index=pd.MultiIndex.from_arrays([entries["race_id"], entries["driverId"]]))
    for r in races:
        g = r["g"]
        r["team"] = pd.factorize(g["lineage"])[0]
        mi = pd.MultiIndex.from_arrays([g["race_id"], g["driverId"]])
        r["starts"] = key.reindex(mi).to_numpy(float)
        r["rookie"] = (r["starts"] < ROOKIE_STARTS).astype(float)
        r["round"] = int(g["round"].iloc[0])
        r["season"] = int(g["season"].iloc[0])
        r["early"] = float(r["round"] <= EARLY_ROUNDS)
    return races


def attach_cov_factors(races: list[dict], covs: dict[int, tuple[list[str], np.ndarray]]) -> list[dict]:
    """Attach the factor L (Sigma = L L^T, symmetric square root via eigh) in the row order of each race's frame."""
    for r in races:
        ids, sig = covs[int(r["race_id"])]
        order = [ids.index(d) for d in r["g"]["driverId"]]
        s = sig[np.ix_(order, order)]
        w, v = np.linalg.eigh(0.5 * (s + s.T))
        r["cov"] = s
        r["L"] = v * np.sqrt(np.clip(w, 0.0, None))[None, :]
    return races


# ------------------------------------------------------------------------------------ simulation
def _rank_matrix(key: np.ndarray) -> np.ndarray:
    k, n = key.shape
    order = np.argsort(-key, axis=1)
    ranks0 = np.empty_like(order)
    ranks0[np.arange(k)[:, None], order] = np.arange(n)[None, :]
    return ranks0


def _staged_key(z: np.ndarray, tau: float, g1: np.ndarray, scale, m: tuple[float, float, float], alive: np.ndarray,
                aux: np.random.Generator) -> np.ndarray:
    """S4: key whose arg-sort places the 1st, 2nd and 3rd finisher by their own stage temperature, the rest by m4."""
    k, n = g1.shape
    gs = []
    for _ in range(3):                       # stage 2, stage 3, rest: fresh Gumbel draws (aux stream)
        g = aux.gumbel(size=(k, n))
        gs.append(g * scale[:, None] if scale is not None else g)
    stage_keys = [tau * z[None, :] + g1, tau * m[0] * z[None, :] + gs[0], tau * m[1] * z[None, :] + gs[1]]
    final = tau * m[2] * z[None, :] + gs[2]
    rows = np.arange(k)
    picked = np.zeros((k, n), bool)
    for j, kj in enumerate(stage_keys):
        kk = np.where(alive & ~picked, kj, -np.inf)
        idx = kk.argmax(1)
        ok = np.isfinite(kk[rows, idx])
        picked[rows[ok], idx[ok]] = True
        final[rows[ok], idx[ok]] = (3 - j) * 1e6
    return final


def simulate_ranks(r: dict, theta: dict, sw: dict, k: int, rng: np.random.Generator,
                   aux: np.random.Generator, ext: dict | None) -> np.ndarray:
    """k simulated finishing orders of one race, 0-based rank of every entry, shape (k, n). `rng` is consumed
    exactly as plan-2 S3 consumes it; extensions draw from `aux`."""
    u, p_dnf, grid_norm = r["u"], r["p_dnf"], r["grid"]
    n = len(u)
    kind = (ext or {}).get("kind")
    z = u + (theta["kappa"] * r["stick_c"] * (0.5 - grid_norm) if sw["stick"] else 0.0)
    g_raw = rng.gumbel(size=(k, n))
    g, scale = g_raw, None
    if sw["regime"]:
        logit = np.log(theta["p0"] / (1 - theta["p0"])) + theta["a_vol"] * r["zvol"] + theta["a_rain"] * r["rain"]
        dis = rng.random(k) < 1 / (1 + np.exp(-logit))
        scale = np.where(dis, theta["s_dis"], 1.0)
        g = g_raw * scale[:, None]
    if kind == "S6":
        c_eff = ext["c"] * (1.0 + ext["c_early"] * r["early"])
        zk = z[None, :] + c_eff * (aux.standard_normal((k, n)) @ r["L"].T)
        key = theta["tau"] * zk + g
    elif kind == "S5":
        eps = ndtri(np.clip(np.exp(-np.exp(-g_raw)), 1e-12, 1 - 1e-12))
        sig = ext["sigma"] * (1.0 + ext["e_exp"] * r["rookie"]) * (1.0 + ext["e_wet"] * r["rain"])
        noise = sig[None, :] * eps
        if ext["tau_team"] > 0:
            team = r["team"]
            eta = aux.standard_normal((k, int(team.max()) + 1))
            noise = noise + ext["tau_team"] * eta[:, team]
        if scale is not None:
            noise = noise * scale[:, None]
        key = theta["tau"] * z[None, :] + noise
    else:
        key = theta["tau"] * z[None, :] + g
    if sw["dnf"]:
        retire = rng.random((k, n)) < np.clip(p_dnf * theta["dnf_scale"], 0.0, 0.95)[None, :]
        tie = rng.random((k, n))
        if kind == "S4":
            key = _staged_key(z, theta["tau"], g, scale, (ext["m2"], ext["m3"], ext["m4"]), ~retire, aux)
        key = np.where(retire, -1e6 + tie, key)
    elif kind == "S4":
        key = _staged_key(z, theta["tau"], g, scale, (ext["m2"], ext["m3"], ext["m4"]), np.ones((k, n), bool), aux)
    return _rank_matrix(key)


def streams(seed: int, i: int) -> tuple[np.random.Generator, np.random.Generator]:
    """(main, aux) generators of race i. The main stream is the plan-2 one (seed + 10000 + i for evaluation,
    1000 + i for tuning when seed = -9000)."""
    s = seed + 10_000 + i
    return np.random.default_rng(s), np.random.default_rng([s, AUX_TAG])


def simulate_matrices(races: list[dict], theta: dict, ext: dict | None, variant_sw: dict, k: int = 1000,
                      seed: int = 0, with_pairs: bool = False):
    """Position-probability matrices (and per-race team-pair probabilities) for all races, evaluation streams."""
    mats, pairs = [], []
    for i, r in enumerate(races):
        rng, aux = streams(seed, i)
        rk = simulate_ranks(r, theta, variant_sw, k, rng, aux, ext)
        mats.append(rank_hist(rk, len(r["u"])))
        if with_pairs:
            pairs.append(team_pair_probs(rk, r["team"]))
    return (mats, pairs) if with_pairs else mats


def team_pair_probs(ranks0: np.ndarray, team: np.ndarray, n_points: int = 10) -> np.ndarray:
    """P(both cars of a two-car team finish in the points places), one value per two-car team (code order)."""
    pts = ranks0 < n_points
    out = []
    for t in np.unique(team):
        ix = np.flatnonzero(team == t)
        if len(ix) == 2:
            out.append(float((pts[:, ix[0]] & pts[:, ix[1]]).mean()))
    return np.array(out)


def team_pair_outcomes(pos: np.ndarray, team: np.ndarray, n_points: int = 10) -> np.ndarray:
    out = []
    for t in np.unique(team):
        ix = np.flatnonzero(team == t)
        if len(ix) == 2:
            out.append(float((pos[ix] <= n_points).all()))
    return np.array(out)


def team_logloss(pairs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-4) -> float:
    """Mean binary log-loss of 'both cars of the team score points' over the race's two-car teams."""
    if len(pairs) == 0:
        return float("nan")
    q = np.clip(pairs, eps, 1 - eps)
    return float(-(outcomes * np.log(q) + (1 - outcomes) * np.log(1 - q)).mean())


# ------------------------------------------------------------------------------------ tuning
def mean_rps(races: list[dict], theta: dict, ext: dict | None, sw: dict, k: int = 300) -> float:
    """Plan-2 tuning objective: mean RPS, k draws, streams 1000 + i (common random numbers across candidates)."""
    vals = []
    for i, r in enumerate(races):
        rng, aux = streams(-9000, i)
        P = rank_hist(simulate_ranks(r, theta, sw, k, rng, aux, ext), len(r["u"]))
        vals.append(rps_score(P, r["pos"]))
    return float(np.mean(vals))


def coordinate_search(score: Callable[[dict], float], start: dict, grids: dict[str, list], order: list[str],
                      max_configs: int = 30, passes: int = 2) -> tuple[dict, pd.DataFrame]:
    """Plan-2 style coordinate search with a hard budget of distinct configurations (the start counts).
    `score` maps a parameter dict to the objective (lower is better). Returns the best parameters and the table of
    every distinct configuration evaluated, in order."""
    cache: dict[tuple, float] = {}
    log: list[dict] = []

    def ev(p: dict) -> float | None:
        key = tuple(sorted(p.items()))
        if key in cache:
            return cache[key]
        if len(cache) >= max_configs:
            return None
        cache[key] = score(p)
        log.append({**p, "rps": cache[key], "config": len(cache)})
        return cache[key]

    best_p, best = dict(start), ev(start)
    for _ in range(passes):
        for name in order:
            for v in grids[name]:
                cand = {**best_p, name: v}
                s = ev(cand)
                if s is not None and s < best - 1e-9:
                    best, best_p = s, cand
    return best_p, pd.DataFrame(log)
