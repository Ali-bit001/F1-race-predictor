"""PLAN3 M05: ensemble pieces. Nothing here is fitted beyond what the component models already fit.

- MonotoneGBM: `GBMModel` (same features, same hyper-parameters) with LightGBM monotone constraints. The target `y` is the
  normalised finishing position (0 = win), so a better grid slot or qualifying position (smaller `grid_norm`,
  `quali_norm`) must never predict a larger `y`: constraint +1 on exactly those two columns, 0 on all others.
- rank_average_score: equal-weight mean of the within-race ranks of several score vectors (higher score = better),
  returned as a utility (higher = better). Exact ties are broken by a key (driverId), never by input row order.
- RankAverageModel: a `Model` that refits its members on the same training rows before every race and averages ranks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitlane.evaluation.metrics import pred_rank
from pitlane.models.gbm import GBMModel
from pitlane.plan3.decode import break_ties

MONOTONE_UP = ("grid_norm", "quali_norm")      # predicted y (worse finish) must not decrease in these


class MonotoneGBM(GBMModel):
    """GBMModel plus `monotone_constraints`: +1 on MONOTONE_UP columns, 0 elsewhere (non-ranking cells only)."""

    def __init__(self, *args, constrained: tuple[str, ...] = MONOTONE_UP, **kw):
        super().__init__(*args, **kw)
        if self.ranking:
            raise ValueError("monotone constraints are defined here for the regression target y only")
        missing = [c for c in constrained if c not in self.cols]
        if missing:
            raise ValueError(f"constrained columns not in the feature set: {missing}")
        self.constrained = tuple(constrained)
        self.constraints = [1 if c in self.constrained else 0 for c in self.cols]
        self.params["monotone_constraints"] = self.constraints


def rank_average_score(scores: list[np.ndarray], key=None) -> np.ndarray:
    """Utility (higher = better) = -(mean within-race rank). `scores` are the models' utilities for ONE race.
    Exact ties inside a member and of the average are broken by `key`, never by row order."""
    r = np.mean([pred_rank(break_ties(np.asarray(s, float), key)) for s in scores], axis=0)
    return break_ties(-r, key)


class RankAverageModel:
    """Fits every member on the same training rows, then averages their within-race ranks for each test race.
    Used with refit_every=1 (one test race per call); with more races per call each race is ranked on its own."""

    def __init__(self, members: list, key: str = "driverId"):
        self.members, self.key = members, key

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        sc = [np.asarray(m.fit_predict(train, test), float) for m in self.members]
        out = np.empty(len(test))
        for _, ix in test.reset_index(drop=True).groupby("race_id").indices.items():
            out[ix] = rank_average_score([s[ix] for s in sc], test[self.key].to_numpy()[ix])
        return out
