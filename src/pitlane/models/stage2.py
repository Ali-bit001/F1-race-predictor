"""Second-stage in-race models (PLAN3 M19): a frozen pre-race score plus in-race state columns.

Stage 1 is a pre-race model with frozen hyper-parameters (`latent_111`, `gbm_100`) run walk-forward over all earlier races
(`walk_scores`): the score of race r comes from a fit on races < r only. Stage 2 never refits stage 1; it learns only how
the in-race columns shift that score:

    Stage2Latent:  u = c * score + beta . x     Plackett-Luce, c free (tiny ridge) or fixed to 1, ridge `lam_x` on beta
    Stage2GBM:     LightGBM regression on [score] + x, same hyper-parameters as the frozen gbm_100

Both are fitted on earlier races only, with the time-decay weights of the frozen pre-race model (0.5 ** (age / halflife)).
With no x columns and c fixed to 1, Stage2Latent returns the pre-race score unchanged (checked in the run).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pitlane.models.registry import build_model


def walk_scores(entries: pd.DataFrame, cfg: dict, last_race: int, first_race: int = 1) -> np.ndarray:
    """Pre-race score of every entry of races first_race..last_race (NaN elsewhere): refit before every race on all earlier
    races, one model object warm-started along the way (as in the harness with refit_every=1)."""
    assert (entries.index == np.arange(len(entries))).all()
    m = build_model(cfg, entries)
    rr = entries.groupby("race_id").indices
    out = np.full(len(entries), np.nan)
    for r in range(first_race, last_race + 1):
        rows = rr[r]
        out[rows] = m.fit_predict(entries.iloc[:rows.min()], entries.iloc[rows])
    return out


def _clean(train: pd.DataFrame, cols: list[str], min_race: int) -> pd.DataFrame:
    tr = train[train["race_id"] >= min_race]
    return tr.dropna(subset=cols)


class Stage2Latent:
    def __init__(self, score_col: str, x_cols: list[str], lam_x: float = 1.0, halflife: float = 50.0, fix_c: bool = False,
                 min_race: int = 0, lam_c: float = 1e-4, maxiter: int = 200):
        self.score_col, self.x_cols, self.lam_x, self.halflife = score_col, list(x_cols), lam_x, halflife
        self.fix_c, self.min_race, self.lam_c, self.maxiter = fix_c, min_race, lam_c, maxiter
        self.names = ([] if fix_c else ["c"]) + self.x_cols
        self.pen = np.array(([] if fix_c else [lam_c]) + [lam_x] * len(self.x_cols), dtype=float)
        self.theta = np.array(([] if fix_c else [1.0]) + [0.0] * len(self.x_cols))
        self.last_n_train_races = 0

    def _design(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        s = df[self.score_col].to_numpy(float)
        x = df[self.x_cols].fillna(0.0).to_numpy(float) if self.x_cols else np.zeros((len(df), 0))
        if self.fix_c:
            return x, s
        return np.column_stack([s, x]), np.zeros(len(df))

    def _obj(self, th, X, off, idx, mask, wr, W):
        u = X @ th + off
        U = np.where(mask, u[idx], -1e9)
        m = U.max(1, keepdims=True)
        E = np.exp(U - m)
        S = np.cumsum(E[:, ::-1], axis=1)[:, ::-1]
        S = np.where(mask, S, 1.0)
        ll = np.where(mask, (U - m) - np.log(S), 0.0)
        loss = -(wr * ll).sum()
        C = np.cumsum(np.where(mask, 1.0 / S, 0.0), axis=1)
        G = np.where(mask, E * C - 1.0, 0.0) * wr
        gu = np.zeros(len(u))
        np.add.at(gu, idx[mask], G[mask])
        total = (loss + (self.pen * th * th).sum()) / W
        grad = (X.T @ gu + 2 * self.pen * th) / W
        return total, grad

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        tr = _clean(train, [self.score_col] + self.x_cols, self.min_race)
        t_ref = float(test["race_id"].min())
        X, off = self._design(tr)
        groups = tr.groupby("race_id", sort=True).indices            # rows are in finishing order inside a race
        races = sorted(groups)
        nmax = max(len(groups[r]) for r in races)
        idx = np.zeros((len(races), nmax), dtype=int)
        mask = np.zeros((len(races), nmax), dtype=bool)
        for i, r in enumerate(races):
            idx[i, :len(groups[r])] = groups[r]
            mask[i, :len(groups[r])] = True
        w_race = 0.5 ** ((t_ref - np.array(races, dtype=float)) / self.halflife)
        wr = w_race[:, None]
        W = float((w_race[:, None] * mask).sum())
        self.last_n_train_races = len(races)
        if len(self.theta):
            res = minimize(self._obj, self.theta, args=(X, off, idx, mask, wr, W), jac=True, method="L-BFGS-B",
                           options={"maxiter": self.maxiter})
            self.theta = res.x
        Xt, offt = self._design(test)
        return Xt @ self.theta + offt

    @property
    def beta(self) -> dict[str, float]:
        return dict(zip(self.names, self.theta))


class Stage2GBM:
    """LightGBM regression on [score] + in-race columns; hyper-parameters of the frozen pre-race gbm_100 (no tuning)."""

    def __init__(self, score_col: str, x_cols: list[str], halflife: float = 32, num_leaves: int = 4, n_estimators: int = 200,
                 learning_rate: float = 0.05, min_child_samples: int = 40, seed: int = 0, min_race: int = 0):
        self.score_col, self.x_cols, self.halflife, self.min_race = score_col, list(x_cols), halflife, min_race
        self.params = dict(num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=learning_rate,
                           min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1)
        self.model: lgb.LGBMRegressor | None = None

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        cols = [self.score_col] + self.x_cols
        tr = _clean(train, cols, self.min_race)
        w = 0.5 ** ((float(test["race_id"].min()) - tr["race_id"].to_numpy()) / self.halflife)
        self.model = lgb.LGBMRegressor(objective="regression", **self.params)
        self.model.fit(tr[cols], tr["y"], sample_weight=w)
        return -self.model.predict(test[cols])
