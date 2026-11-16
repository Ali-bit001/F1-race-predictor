"""A5 strength audit (PLAN3.md section 4): converged snapshot fits, Laplace intervals, named checks,
identification. Reusable pieces only; the experiment (config, tables, figures) lives in
methodologies/A5_strength_audit/run.py.

The Laplace approximation uses the Hessian of the *unnormalised* penalised objective
(-log-likelihood + penalty, the thing `LatentModel._objective` divides by the total row weight W) at the
optimum: cov = inv(W * Hessian of the normalised objective). Two caveats, both stated in the method card:
the noise sd is the fixed SIGMA of the model (not estimated), and the time-decay weights make the
likelihood a power likelihood, so the intervals are conditional on the model, not frequentist coverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

from pitlane.models.latent import SIGMA, LatentModel
from pitlane.models.registry import build_model, cell_flags


# ------------------------------------------------------------------------------------ fitting
def snapshot_points(entries: pd.DataFrame) -> list[tuple[int, int, float]]:
    """(season, n_train, t_ref) for a fit through the last race of each season."""
    out = []
    for s, g in entries.groupby("season"):
        last = int(g["race_id"].max())
        n_train = int(np.searchsorted(entries["race_id"].to_numpy(), last, side="right"))
        out.append((int(s), n_train, float(last + 1)))
    return out


def fit_tight(model: LatentModel, n_train: int, t_ref: float, maxiter: int = 5000) -> dict:
    """Converge much further than the walk-forward fits (maxiter 150); returns convergence diagnostics."""
    w, pad = model.fit_inputs(n_train, t_ref)
    res = minimize(model._objective, model.theta, args=(n_train, w, pad), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-10, "maxcor": 30})
    model.theta = res.x
    gmax = float(np.abs(res.jac).max())
    return {"nit": int(res.nit), "grad_max": gmax, "success": bool(res.success)}


# ------------------------------------------------------------------------------------ Hessians
def hessian_fd(model: LatentModel, n_train: int, t_ref: float, eps: float = 1e-5) -> np.ndarray:
    """Hessian of the unnormalised objective by central differences of the analytic gradient."""
    w, pad = model.fit_inputs(n_train, t_ref)
    th0 = model.theta.copy()
    W = float(w.sum())
    P = len(th0)
    H = np.zeros((P, P))
    for j in range(P):
        d = np.zeros(P)
        d[j] = eps
        gp = model._objective(th0 + d, n_train, w, pad)[1]
        gm = model._objective(th0 - d, n_train, w, pad)[1]
        H[:, j] = (gp - gm) / (2 * eps)
    return 0.5 * (H + H.T) * W


def hessian_fd_value_only(model: LatentModel, n_train: int, t_ref: float, eps: float = 1e-4) -> np.ndarray:
    """Second differences of the objective *value* (independent of the analytic gradient); small problems only."""
    w, pad = model.fit_inputs(n_train, t_ref)
    th0 = model.theta.copy()
    W = float(w.sum())
    P = len(th0)
    f = lambda th: model._objective(th, n_train, w, pad)[0]
    f0 = f(th0)
    H = np.zeros((P, P))
    for i in range(P):
        ei = np.zeros(P); ei[i] = eps
        H[i, i] = (f(th0 + ei) - 2 * f0 + f(th0 - ei)) / eps ** 2
        for j in range(i + 1, P):
            ej = np.zeros(P); ej[j] = eps
            H[i, j] = H[j, i] = (f(th0 + ei + ej) - f(th0 + ei - ej) - f(th0 - ei + ej) + f(th0 - ei - ej)) / (4 * eps ** 2)
    return H * W


def hessian_analytic_gaussian(model: LatentModel, n_train: int, t_ref: float) -> np.ndarray:
    """Closed form for the Gaussian likelihood (ranking off): J' diag(w / sigma^2) J + penalty Hessian."""
    assert not model.ranking
    w, _ = model.fit_inputs(n_train, t_ref)
    nD, nB, nG, nX = model.nD, model.nB, model.nG, model.nX
    P = nD + nB + nG + nX
    rows = np.arange(n_train)
    cols = [model.row_d[rows]]
    blocks = [(model.row_d[rows], 0)]
    if model.decomp:
        blocks += [(model.row_b[rows], nD), (model.row_g[rows], nD + nB)]
    J = np.zeros((n_train, P))
    for idx, off in blocks:
        J[rows, off + idx] = 1.0
    if nX:
        J[:, nD + nB + nG:] = model.row_x[:n_train]
    H = (J * (w / SIGMA ** 2)[:, None]).T @ J
    pen = np.zeros(P)
    pen[:nD] = 2 * model.lam_a
    if model.decomp:
        prev = model.car_prev
        D = np.eye(nB)
        for i, p in enumerate(prev):
            if p >= 0:
                D[i, p] -= model.rho
        H[nD:nD + nB, nD:nD + nB] += 2 * model.lam * (D.T @ D)
        pen[nD + nB:nD + nB + nG] = 2 * model.lam * model.fit_mult
    pen[nD + nB + nG:] = 2 * model.pen_x
    return H + np.diag(pen)


def laplace_cov(H: np.ndarray) -> np.ndarray:
    """Inverse Hessian restricted to parameters that enter the model (unused blocks have a zero row)."""
    P = len(H)
    use = np.abs(np.diag(H)) > 0
    cov = np.zeros((P, P))
    sub = H[np.ix_(use, use)]
    cov[np.ix_(use, use)] = np.linalg.inv(sub)
    return 0.5 * (cov + cov.T)


# ------------------------------------------------------------------------------------ strength table
def final_tables(model: LatentModel, se: np.ndarray, entries: pd.DataFrame, tag: str, z: float = 1.96) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Driver and car-season strengths from one fit through the last race (every driver / lineage-season that ever raced)."""
    a, b, _, _ = model._split(model.theta)
    n_tot = entries.groupby("driverId").size()
    last = entries.groupby("driverId")["season"].max()
    n24 = entries[entries["season"] >= 2024].groupby("driverId").size()
    dr = pd.DataFrame([dict(model=tag, id=d, est=a[i], se=se[i], n_total=int(n_tot[d]), last_season=int(last[d]),
                            n_2024_26=int(n24.get(d, 0))) for d, i in model.drivers.items()])
    dr["lo"], dr["hi"] = dr["est"] - z * dr["se"], dr["est"] + z * dr["se"]
    ncar = entries.groupby(["lineage", "season"]).size()
    cr = pd.DataFrame([dict(model=tag, id=l, season=s, est=b[i], se=se[model.nD + i], n_season=int(ncar[(l, s)]))
                       for (l, s), i in model.car.items() if (l, s) in ncar.index])
    cr["lo"], cr["hi"] = cr["est"] - z * cr["se"], cr["est"] + z * cr["se"]
    return dr, cr


def _starts(entries: pd.DataFrame, n_train: int, season: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    tr = entries.iloc[:n_train]
    n_tot = tr.groupby("driverId").size()
    n_seas = tr[tr["season"] == season].groupby("driverId").size()
    n_car = tr[tr["season"] == season].groupby("lineage").size()
    return n_tot, n_seas, n_car


def snapshot_table(entries: pd.DataFrame, cfg: dict, tag: str, season: int, n_train: int, t_ref: float,
                   z: float = 1.96) -> tuple[pd.DataFrame, dict, LatentModel, np.ndarray]:
    """Strengths and Laplace intervals for every driver and car-season active in `season`, fitted through its end."""
    m = build_model(cfg, entries)
    diag = fit_tight(m, n_train, t_ref)
    cov = laplace_cov(hessian_fd(m, n_train, t_ref))
    a, b, g, beta = m._split(m.theta)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    se_a, se_b = se[:m.nD], se[m.nD:m.nD + m.nB]
    n_tot, n_seas, n_car = _starts(entries, n_train, season)
    rows = []
    for d, i in m.drivers.items():
        if d in n_seas.index:
            rows.append(dict(model=tag, season=season, kind="driver", id=d, est=a[i], se=se_a[i],
                             n_season=int(n_seas[d]), n_total=int(n_tot[d])))
    if m.decomp:
        for (l, s), i in m.car.items():
            if s == season and l in n_car.index:
                rows.append(dict(model=tag, season=season, kind="car", id=l, est=b[i], se=se_b[i],
                                 n_season=int(n_car[l]), n_total=int(n_car[l])))
    t = pd.DataFrame(rows)
    t["lo"], t["hi"] = t["est"] - z * t["se"], t["est"] + z * t["se"]
    t["low_sample"] = (t["kind"] == "driver") & (t["n_total"] < 10)
    # rank within the season among drivers (1 = highest a) / cars (1 = fastest b)
    t["rank"] = t.groupby("kind")["est"].rank(ascending=False, method="first").astype(int)
    main = t[~t["low_sample"]].copy()
    main["rank_main"] = main.groupby("kind")["est"].rank(ascending=False, method="first").astype(int)
    t = t.merge(main[["kind", "id", "rank_main"]], on=["kind", "id"], how="left")
    return t, {**diag, "season": season, "n_train": n_train}, m, se


def champions(entries: pd.DataFrame) -> pd.DataFrame:
    """Points leader per season (the champion for completed seasons) and the teammate with most starts."""
    out = []
    for s, g in entries.groupby("season"):
        pts = g.groupby("driverId")["points"].sum().sort_values(ascending=False)
        champ = pts.index[0]
        team = g[g["driverId"] == champ].groupby("lineage").size().idxmax()
        mates = g[(g["lineage"] == team) & (g["driverId"] != champ)].groupby("driverId").size().sort_values(ascending=False)
        out.append(dict(season=int(s), champion=champ, points=float(pts.iloc[0]), lineage=team,
                        teammate=mates.index[0] if len(mates) else None, rounds=int(g["round"].nunique())))
    return pd.DataFrame(out)


def champion_ranks(tab: pd.DataFrame, champs: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Season-by-season rank (of drivers racing that season; 1 = best `a`) of the champion and their teammate."""
    d = tab[(tab["model"] == tag) & (tab["kind"] == "driver")]
    rows = []
    for _, c in champs.iterrows():
        g = d[d["season"] == c["season"]].set_index("id")
        n = len(g)
        rec = dict(model=tag, season=c["season"], champion=c["champion"], n_drivers=n)
        for role, who in (("champ", c["champion"]), ("mate", c["teammate"])):
            if who in g.index:
                rec.update({f"{role}_id": who, f"{role}_rank": int(g.loc[who, "rank"]), f"{role}_a": g.loc[who, "est"],
                            f"{role}_se": g.loc[who, "se"], f"{role}_n": int(g.loc[who, "n_season"])})
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ face validity
def teammate_pairs(entries: pd.DataFrame, a: pd.Series, min_starts: int = 8) -> pd.DataFrame:
    """Same-team same-season driver pairs (plan-1 definition, report.py): d(a) vs d(mean finishing position)."""
    ds = entries.groupby(["season", "lineage", "driverId"]).agg(y=("y", "mean"), n=("y", "size")).reset_index()
    ds = ds[ds["n"] >= min_starts]
    rows = []
    for (s, l), grp in ds.groupby(["season", "lineage"]):
        if len(grp) == 2:
            x, zz = grp.iloc[0], grp.iloc[1]
            if x["driverId"] not in a.index or zz["driverId"] not in a.index:
                continue          # (out-of-sample use: a driver without a strength value at the earlier snapshot)
            rows.append(dict(season=int(s), lineage=l, d_a=a[x["driverId"]] - a[zz["driverId"]], d_y=zz["y"] - x["y"]))
    return pd.DataFrame(rows)


def car_pairs(entries: pd.DataFrame, car_b: dict[tuple[str, int], float]) -> pd.DataFrame:
    cs = entries.groupby(["lineage", "season"])["y"].mean().rename("y").reset_index()
    cs["b"] = [car_b[(l, s)] for l, s in zip(cs["lineage"], cs["season"])]
    cs["neg_y"] = -cs["y"]
    return cs


def corr_ci(x: np.ndarray, y: np.ndarray, b: int = 2000, seed: int = 0) -> tuple[float, float, float, int]:
    """Pearson r with a percentile bootstrap interval over pairs."""
    n = len(x)
    if n < 4:
        return (np.nan, np.nan, np.nan, n)
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(b):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            rs.append(np.corrcoef(x[i], y[i])[0, 1])
    lo, hi = np.percentile(rs, [2.5, 97.5]) if rs else (np.nan, np.nan)
    return r, float(lo), float(hi), n


def per_season_correlations(pairs_t: pd.DataFrame, pairs_c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in sorted(set(pairs_t["season"]) | set(pairs_c["season"])):
        t, c = pairs_t[pairs_t["season"] == s], pairs_c[pairs_c["season"] == s]
        rt = corr_ci(t["d_a"].to_numpy(), t["d_y"].to_numpy())
        rc = corr_ci(c["b"].to_numpy(), c["neg_y"].to_numpy())
        rows.append(dict(season=s, r_teammate=rt[0], r_teammate_lo=rt[1], r_teammate_hi=rt[2], n_pairs=rt[3],
                         r_car=rc[0], r_car_lo=rc[1], r_car_hi=rc[2], n_cars=rc[3]))
    return pd.DataFrame(rows)


def oos_teammate_correlations(entries: pd.DataFrame, strengths: pd.DataFrame, tag: str, min_starts: int = 8,
                              b: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Out-of-sample face validity: for each season S with a previous season, the difference in snapshot `a`
    (fit through the end of S-1) between two teammates of season S against their mean-finishing-position gap in S.
    Pairs are the plan-1 pairs of season S (>= min_starts each, exactly two such drivers in the team) where both drivers
    already have a snapshot value at S-1. Descriptive only; no decision uses it."""
    st = strengths[(strengths["model"] == tag) & (strengths["kind"] == "driver")]
    rows, allp = [], []
    for s in sorted(entries["season"].unique()):
        prev = st[st["season"] == s - 1].set_index("id")["est"]
        if prev.empty:
            continue
        p = teammate_pairs(entries[entries["season"] == s], prev, min_starts)
        allp.append(p)
        r = corr_ci(p["d_a"].to_numpy(), p["d_y"].to_numpy(), b, seed)
        rows.append(dict(model=tag, season=str(int(s)), r=r[0], lo=r[1], hi=r[2], n_pairs=r[3]))
    pool = pd.concat(allp, ignore_index=True)
    r = corr_ci(pool["d_a"].to_numpy(), pool["d_y"].to_numpy(), b, seed)
    rows.append(dict(model=tag, season="pooled", r=r[0], lo=r[1], hi=r[2], n_pairs=r[3]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ between-driver variance
def driver_variance(model: LatentModel, n_train: int, t_ref: float, sigmas: dict[str, float], tag: str,
                    min_starts: int = 30) -> pd.DataFrame:
    """Method-of-moments between-driver variance of the true `a` for drivers with >= min_starts starts.
    Observed variance of the unpenalised `a` (= a / data weight) minus the mean sampling noise of a decay-weighted
    mean of residuals, noise_i = sigma^2 * sum(w^2) / (sum(w))^2 (NOT sigma^2 / sum(w), which is the noise of an
    unweighted mean over the effective sample and is too small for a weighted mean). The noise ignores the car
    separation (it is a lower bound on the sampling noise), so tau is an upper bound on the true spread."""
    w, _ = model.fit_inputs(n_train, t_ref)
    sh = shrinkage_table(model, pd.DataFrame(), n_train, t_ref)
    d = pd.DataFrame({"d": model.row_d[:n_train], "w": w, "w2": w ** 2}).groupby("d").agg(sw=("w", "sum"), sw2=("w2", "sum"))
    names = {i: n for n, i in model.drivers.items()}
    d.index = [names[i] for i in d.index]
    sh = sh.set_index("driverId").join(d)
    sh = sh[sh["n_starts"] >= min_starts]
    var_obs = float(sh["a_unpenalised"].var())
    rows = []
    for lab, sg in sigmas.items():
        noise = float((sg ** 2 * sh["sw2"] / sh["sw"] ** 2).mean())
        naive = float((sg ** 2 / sh["sw"]).mean())
        tau2 = var_obs - noise
        rows.append(dict(model=tag, sigma_label=lab, sigma=sg, n_drivers=len(sh), var_unpenalised_a=var_obs,
                         mean_noise_var=noise, mean_naive_noise_var_sigma2_over_sw=naive, tau2=tau2,
                         tau=float(np.sqrt(max(tau2, 0.0))),
                         lam_a_eb_corrected=float(1 / (2 * tau2)) if tau2 > 0 else float("inf")))
    return pd.DataFrame(rows)


def resolved_pairs(a: np.ndarray, cov: np.ndarray, ids: list[str], z: float = 1.96) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Among the given drivers (indices into a / cov, full Laplace covariance): share of driver pairs with
    |a_i - a_j| > z * se(a_i - a_j), and per driver the contrast against the mean of the group."""
    idx = np.asarray(ids)
    n = len(idx)
    A_, C = a[idx], cov[np.ix_(idx, idx)]
    dv = np.diag(C)
    i, j = np.triu_indices(n, 1)
    se_d = np.sqrt(dv[i] + dv[j] - 2 * C[i, j])
    res = np.abs(A_[i] - A_[j]) > z * se_d
    summ = pd.DataFrame([dict(n_drivers=n, n_pairs=len(i), share_resolved=float(res.mean()), mean_marginal_se=float(np.sqrt(dv).mean()),
                              mean_pairwise_se=float(se_d.mean()), ratio_pairwise_to_marginal=float(se_d.mean() / np.sqrt(dv).mean()))])
    K = np.eye(n) - 1.0 / n
    se_f = np.sqrt(np.einsum("ij,jk,ik->i", K, C, K))
    fld = pd.DataFrame(dict(idx=idx, est=A_, contrast_vs_group_mean=K @ A_, se_contrast=se_f))
    fld["z"] = fld["contrast_vs_group_mean"] / fld["se_contrast"]
    fld["resolved"] = fld["z"].abs() > z
    return summ, fld


# ------------------------------------------------------------------------------------ identification
def identification(entries: pd.DataFrame, se_a: pd.Series | None = None, min_starts: int = 10,
                   min_mates: int = 2) -> pd.DataFrame:
    """Per driver: starts, distinct teammates (same lineage, same season, any race), lineages, team-seasons,
    and whether the driver sits in the largest connected component of the driver - team-season graph.
    A driver who never changes team is separated from the car only through teammates."""
    e = entries[["driverId", "lineage", "season", "race_id"]]
    ts = e.assign(ts=e["lineage"] + "_" + e["season"].astype(str))
    mates = {}
    for _, grp in ts.groupby("ts"):
        drs = set(grp["driverId"])
        for d in drs:
            mates.setdefault(d, set()).update(drs - {d})
    g = ts.groupby("driverId").agg(n_starts=("race_id", "size"), n_lineages=("lineage", "nunique"),
                                   n_team_seasons=("ts", "nunique"), first_season=("season", "min"),
                                   last_season=("season", "max")).reset_index()
    g["n_teammates"] = g["driverId"].map(lambda d: len(mates.get(d, ())))
    drivers = {d: i for i, d in enumerate(sorted(ts["driverId"].unique()))}
    tss = {t: i + len(drivers) for i, t in enumerate(sorted(ts["ts"].unique()))}
    pairs = ts[["driverId", "ts"]].drop_duplicates()
    r = pairs["driverId"].map(drivers).to_numpy()
    c = pairs["ts"].map(tss).to_numpy()
    n = len(drivers) + len(tss)
    _, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)), directed=False)
    big = np.bincount(lab).argmax()
    g["in_main_component"] = g["driverId"].map(lambda d: bool(lab[drivers[d]] == big))
    if se_a is not None:
        g["se_a"] = g["driverId"].map(se_a)
    g["weak"] = (g["n_starts"] < min_starts) | (g["n_teammates"] < min_mates) | ~g["in_main_component"]
    def why(row) -> str:
        w = []
        if row["n_starts"] < min_starts: w.append(f"<{min_starts} starts")
        if row["n_teammates"] < min_mates: w.append(f"<{min_mates} teammates")
        if not row["in_main_component"]: w.append("disconnected")
        return "; ".join(w)
    g["weak_reason"] = g.apply(why, axis=1)
    return g.sort_values(["n_teammates", "n_starts"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------ shrinkage diagnosis
def shrinkage_table(model: LatentModel, entries: pd.DataFrame, n_train: int, t_ref: float) -> pd.DataFrame:
    """Per driver: effective weight, fitted `a`, the residual that `a` has to explain, and the exact
    stationarity prediction a = - sum(w r) / (sigma^2 * 2 lam_a) from the first-order condition of the Gaussian
    objective (r is the residual *including* a). Drivers with a large weight have a close to the raw mean
    residual; drivers with a weight near 1 are pulled to 0 only by the penalty."""
    assert not model.ranking
    w, _ = model.fit_inputs(n_train, t_ref)
    a, _, _, _ = model._split(model.theta)
    rows = np.arange(n_train)
    u = model._utility(model.theta, rows)
    r = model.row_y[:n_train] - 0.5 + u
    df = pd.DataFrame({"d": model.row_d[:n_train], "w": w, "r": r, "wr": w * r})
    g = df.groupby("d").agg(eff_starts=("w", "sum"), n=("w", "size"), wr=("wr", "sum"))
    names = {i: d for d, i in model.drivers.items()}
    out = pd.DataFrame({"driverId": [names[i] for i in g.index], "n_starts": g["n"].to_numpy(),
                        "eff_starts": g["eff_starts"].to_numpy(), "a": a[g.index.to_numpy()],
                        "a_stationary": (-g["wr"] / (SIGMA ** 2 * 2 * model.lam_a)).to_numpy()})
    # weight of the data in the posterior mean: 1 / (1 + 2 lam_a sigma^2 / eff_starts)
    out["data_weight"] = out["eff_starts"] / (out["eff_starts"] + 2 * model.lam_a * SIGMA ** 2)
    out["a_unpenalised"] = out["a"] / out["data_weight"]
    return out
