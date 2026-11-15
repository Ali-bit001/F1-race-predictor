"""Family L: additive latent-strength model with the three research switches (PLAN.md M03).

    u(i, r) = a[driver] + b[car] + g[car, circuit type] + beta . x(i, r)        (higher = faster)

F1 dynamic    : OFF -> equal weight on all history, one static car effect per constructor lineage.
                ON  -> exponential time decay on the likelihood (half-life `halflife` races) and a
                       car effect per (lineage, season) tied to last season's by carry-over `rho`.
F2 decomposed : OFF -> only a driver effect (+ weekend signal): the car is absorbed into the driver.
                ON  -> separate driver, car and car x circuit-type effects.
F3 ranking    : OFF -> Gaussian regression of normalised finishing position on u.
                ON  -> Plackett-Luce likelihood over the whole finishing order.

Fitted by penalised maximum likelihood (L-BFGS, analytic gradients), warm-started between refits.
Every prediction is the exact sum of the named terms, so explanations are free (see components()).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SIGMA = 0.25            # fixed noise sd for the Gaussian likelihood (puts both losses on a similar scale)
WEEKEND_X = ["quali_pos", "grid_penalty", "q1_gap", "pit_start"]


def _x(df: pd.DataFrame) -> np.ndarray:
    # quali position, grid-minus-quali shift (penalties / pit-lane starts), Q1 pace gap, pit-lane flag
    return np.column_stack([df["quali_norm"] - 0.5, df["grid_norm"] - df["quali_norm"], df["q1_gap"] * 10,
                            df["pit_start"]]).astype(float)


class LatentModel:
    def __init__(self, universe: pd.DataFrame, dynamic: bool, decomp: bool, ranking: bool,
                 halflife: float = 20.0, lam: float = 2.0, rho: float = 0.8, fit_mult: float = 5.0,
                 maxiter: int = 150, base_x: bool = True, x_cols: tuple[str, ...] | list[str] = (),
                 lam_x: float = 1.0):
        self.dynamic, self.decomp, self.ranking = dynamic, decomp, ranking
        self.halflife, self.lam, self.rho, self.fit_mult, self.maxiter = halflife, lam, rho, fit_mult, maxiter
        u = universe
        self.drivers = {d: i for i, d in enumerate(sorted(u["driverId"].unique()))}
        self.types = {t: i for i, t in enumerate(sorted(u["circuit_type"].unique()))}
        self.lineages = {l: i for i, l in enumerate(sorted(u["lineage"].unique()))}
        if dynamic:
            pairs = sorted(set(zip(u["lineage"], u["season"])), key=lambda p: (p[1], p[0]))
            self.car = {p: i for i, p in enumerate(pairs)}
            self.car_prev = np.array([self.car.get((l, s - 1), -1) for (l, s) in pairs])
        else:
            self.car = {(l, None): i for l, i in self.lineages.items()}
            self.car_prev = np.full(len(self.car), -1)
        self.nD, self.nB = len(self.drivers), len(self.car)
        self.nG = len(self.lineages) * len(self.types)
        # Weekend signals (PLAN2.md section 3): the four plan-1 signals (base_x) plus extra pre-scaled
        # columns of `universe` named in x_cols. base_x=False and no x_cols is the H0 (pre-weekend) model.
        self.base_x, self.x_cols = base_x, list(x_cols)
        self.x_names = (list(WEEKEND_X) if base_x else []) + self.x_cols
        self.nX = len(self.x_names)
        self.pen_x = np.array([1e-4] * (len(WEEKEND_X) if base_x else 0) + [lam_x] * len(self.x_cols))
        self.theta = np.zeros(self.nD + self.nB + self.nG + self.nX)
        # Row-aligned index arrays for the whole universe (must be sorted by race_id, pos).
        self.row_d = u["driverId"].map(self.drivers).to_numpy()
        self.row_b = np.array([self.car[(l, s if dynamic else None)] for l, s in zip(u["lineage"], u["season"])])
        self.row_g = (u["lineage"].map(self.lineages) * len(self.types) + u["circuit_type"].map(self.types)).to_numpy()
        parts = ([_x(u)] if base_x else []) + ([u[self.x_cols].fillna(0.0).to_numpy(float)] if self.x_cols else [])
        self.row_x = np.hstack(parts) if parts else np.zeros((len(u), 0))
        self.row_y = u["y"].to_numpy(float)
        self.row_race = u["race_id"].to_numpy()
        self.row_season = u["season"].to_numpy()
        self.race_rows = u.groupby("race_id").indices
        self.last_components: pd.DataFrame | None = None

    # ------------------------------------------------------------------ parameter views
    def _split(self, th):
        a = th[:self.nD]
        b = th[self.nD:self.nD + self.nB]
        g = th[self.nD + self.nB:self.nD + self.nB + self.nG]
        beta = th[self.nD + self.nB + self.nG:]
        return a, b, g, beta

    def _utility(self, th, rows):
        a, b, g, beta = self._split(th)
        u = a[self.row_d[rows]] + self.row_x[rows] @ beta
        if self.decomp:
            u = u + b[self.row_b[rows]] + g[self.row_g[rows]]
        return u

    # ------------------------------------------------------------------ objective
    def _objective(self, th, n_train: int, w_row: np.ndarray, pad: np.ndarray | None):
        rows = np.arange(n_train)
        u = self._utility(th, rows)
        W = w_row.sum()
        if not self.ranking:
            r = self.row_y[:n_train] - 0.5 + u
            loss = (w_row * r * r).sum() / (2 * SIGMA ** 2)
            gu = w_row * r / SIGMA ** 2
        else:
            # Plackett-Luce, vectorised over races via a padded [race, finishing slot] matrix.
            idx, mask = pad
            U = np.where(mask, u[idx], -1e9)
            m = U.max(1, keepdims=True)
            E = np.exp(U - m)
            S = np.cumsum(E[:, ::-1], axis=1)[:, ::-1]
            S = np.where(mask, S, 1.0)
            ll = np.where(mask, (U - m) - np.log(S), 0.0)
            wr = w_row[idx[:, 0]][:, None]            # race weight (same for all rows of a race)
            loss = -(wr * ll).sum()
            C = np.cumsum(np.where(mask, 1.0 / S, 0.0), axis=1)
            G = np.where(mask, E * C - 1.0, 0.0) * wr
            gu = np.zeros(n_train)
            np.add.at(gu, idx[mask], G[mask])
        # scatter gradient to parameters
        ga = np.bincount(self.row_d[:n_train], gu, self.nD)
        gx = self.row_x[:n_train].T @ gu
        gb = np.zeros(self.nB)
        gg = np.zeros(self.nG)
        a, b, g, beta = self._split(th)
        pen = self.lam * (a @ a)
        pga = 2 * self.lam * a
        if self.decomp:
            gb = np.bincount(self.row_b[:n_train], gu, self.nB)
            gg = np.bincount(self.row_g[:n_train], gu, self.nG)
            prev = self.car_prev
            has = prev >= 0
            tgt = np.where(has, self.rho * b[np.where(has, prev, 0)], 0.0)
            d = b - tgt
            pen += self.lam * (d @ d) + self.lam * self.fit_mult * (g @ g)
            pgb = 2 * self.lam * d
            # derivative through the carry-over term of the following season
            np.add.at(pgb, prev[has], -2 * self.lam * self.rho * d[has])
            gb = gb + pgb
            gg = gg + 2 * self.lam * self.fit_mult * g
        pen += float((self.pen_x * beta * beta).sum())
        gbeta = gx + 2 * self.pen_x * beta
        total = (loss + pen) / W
        grad = np.concatenate([ga + pga, gb, gg, gbeta]) / W
        return total, grad

    # ------------------------------------------------------------------ fit / predict
    def _pad(self, n_races: int):
        nmax = max(len(self.race_rows[r]) for r in range(n_races))
        idx = np.zeros((n_races, nmax), dtype=int)
        mask = np.zeros((n_races, nmax), dtype=bool)
        for r in range(n_races):
            rows = self.race_rows[r]
            idx[r, :len(rows)] = rows
            mask[r, :len(rows)] = True
        return idx, mask

    def fit(self, n_train: int, t_ref: float):
        races = self.row_race[:n_train]
        if self.dynamic:
            w_row = 0.5 ** ((t_ref - races) / self.halflife)
        else:
            w_row = np.ones(n_train)
        pad = self._pad(int(races.max()) + 1) if self.ranking else None
        res = minimize(self._objective, self.theta, args=(n_train, w_row, pad), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.maxiter})
        self.theta = res.x

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        n_train = len(train)
        self.fit(n_train, t_ref=float(test["race_id"].min()))
        rows = test.index.to_numpy()
        self.last_components = self.components(rows)
        return self.last_components["u"].to_numpy()

    def components(self, rows: np.ndarray) -> pd.DataFrame:
        a, b, g, beta = self._split(self.theta)
        comp = pd.DataFrame({
            "a_driver": a[self.row_d[rows]],
            "b_car": b[self.row_b[rows]] if self.decomp else 0.0,
            "g_circuit": g[self.row_g[rows]] if self.decomp else 0.0,
            "weekend": self.row_x[rows] @ beta,
        }, index=rows)
        comp["u"] = comp.sum(axis=1)
        if self.nX:   # per-signal weekend contributions (explanation only; not part of u's sum above)
            wk = self.row_x[rows] * beta[None, :]
            for j, name in enumerate(self.x_names):
                comp[f"wk_{name}"] = wk[:, j]
        return comp

    @property
    def beta(self) -> dict[str, float]:
        return dict(zip(self.x_names, self._split(self.theta)[3]))


def walk_strengths(universe: pd.DataFrame, kwargs: dict, first_race: int) -> pd.DataFrame:
    """Refit before every race from `first_race` on and record each entry's (a, b, g) as known
    pre-race. Used to build the F2-on features of Family G without leakage."""
    m = LatentModel(universe, **kwargs)
    out = np.full((len(universe), 3), np.nan)
    for r in range(first_race, int(universe["race_id"].max()) + 1):
        rows = m.race_rows[r]
        first = rows.min()
        m.fit(first, t_ref=float(r))
        c = m.components(rows)
        out[rows] = c[["a_driver", "b_car", "g_circuit"]].to_numpy()
    return pd.DataFrame(out, columns=["a", "b", "g"], index=universe.index)
