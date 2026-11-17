"""S6 input: the Laplace covariance of every entry's utility, from the walk-forward fit that produced its score.

The walk is the harness walk (`pitlane.evaluation.harness.walk_forward`, refit_every = 1): for every target race
(15 warm-up races, then the counted ones) the model is refitted on all earlier races, warm-started from the previous
fit, exactly as in plan 2, so the utilities reproduce the stored plan-2 scores. For the counted races the Hessian of
the unnormalised penalised objective at the fitted parameters is computed (`pitlane.plan3.audit.hessian_fd`), inverted
over the parameters that enter the model, and propagated to the race's utilities: Sigma = J cov J', where row i of J
selects a[driver], b[car-season], g[car, circuit type] and carries the weekend signals x (beta uncertainty).
Caveats (also in the method card): the fit stops at 150 L-BFGS iterations (not converged), the time-decay weights make
the likelihood a power likelihood, and penalised estimates are biased; the interval is conditional on the model, so a
fitted multiplier `c` in the simulator absorbs part of the mismatch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitlane.evaluation.harness import split_races
from pitlane.models.latent import LatentModel
from pitlane.plan3 import audit as AU


def utility_jacobian(model: LatentModel, rows: np.ndarray) -> np.ndarray:
    """d u_i / d theta for the entries `rows` (u = a + b + g + x beta; shape (len(rows), n_params))."""
    P = len(model.theta)
    J = np.zeros((len(rows), P))
    ar = np.arange(len(rows))
    J[ar, model.row_d[rows]] = 1.0
    if model.decomp:
        J[ar, model.nD + model.row_b[rows]] = 1.0
        J[ar, model.nD + model.nB + model.row_g[rows]] = 1.0
    if model.nX:
        J[:, model.nD + model.nB + model.nG:] = model.row_x[rows]
    return J


def race_utility_cov(model: LatentModel, n_train: int, t_ref: float, rows: np.ndarray) -> np.ndarray:
    cov = AU.laplace_cov(AU.hessian_fd(model, n_train, t_ref))
    J = utility_jacobian(model, rows)
    S = J @ cov @ J.T
    return 0.5 * (S + S.T)


def walk_with_cov(entries: pd.DataFrame, model: LatentModel, split: str = "dev", log=print):
    """Replay the walk-forward fits of `model` (refit before every race, as in plan 2). Returns
    ({race_id: (driverIds, u)}, {race_id: (driverIds, Sigma)}) for the counted races of `split`."""
    warm, counted = split_races(entries, split)
    race_rows = entries.groupby("race_id").indices
    cset = set(counted)
    us, covs = {}, {}
    for j, r in enumerate(warm + counted):
        rows = race_rows[r]
        first = int(rows.min())
        u = model.fit_predict(entries.iloc[:first], entries.iloc[rows])
        if r in cset:
            ids = entries["driverId"].to_numpy()[rows].tolist()
            us[r] = (ids, u)
            covs[r] = (ids, race_utility_cov(model, first, float(r), rows))
            if (j - len(warm)) % 10 == 0:
                log(f"  race {r}: sd(u) mean {np.sqrt(np.diag(covs[r][1])).mean():.3f}")
    return us, covs
