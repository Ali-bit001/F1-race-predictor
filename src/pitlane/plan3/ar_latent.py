"""M16 S7 - mean-reverting season carry-over with a larger innovation at a rule change (PLAN3.md section 4).

The plan-1 latent model already ties each car-season strength to the previous season's with a penalty
`lam * (b_s - rho * b_{s-1})^2`, i.e. an AR(1) process with carry-over `rho` and innovation variance 1/(2 lam), and the
frozen plan-2 model uses the default rho = 0.8 and one innovation size for every season. S7 changes two things:
`rho` (phi) is fitted instead of fixed, and in a rule-change season the innovation variance is larger:
the penalty weight of that season's car effects becomes `lam / rule_mult`.

Implemented as a subclass that adds the difference to the plan-1 objective, so that `rule_mult = 1` returns the
parent's objective and gradient unchanged (bitwise; the extra term is skipped). latent.py is not edited.
Rule-change seasons (new technical regulations; a fact known in advance, not estimated): 2014, 2017, 2022, 2026.
Season 2014 has no previous season in the data, so only 2017, 2022 and 2026 matter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitlane.models.latent import LatentModel

RULE_SEASONS = (2014, 2017, 2022, 2026)


class LatentModelAR(LatentModel):
    def __init__(self, universe: pd.DataFrame, *args, rule_mult: float = 1.0,
                 rule_seasons: tuple[int, ...] = RULE_SEASONS, **kwargs):
        super().__init__(universe, *args, **kwargs)
        self.rule_mult = float(rule_mult)
        self.rule_seasons = tuple(rule_seasons)
        seasons = np.zeros(self.nB, dtype=int)
        for (lin, s), i in self.car.items():
            seasons[i] = -1 if s is None else int(s)
        self.car_season = seasons
        self.lam_b = np.where(np.isin(seasons, list(self.rule_seasons)), self.lam / self.rule_mult, self.lam)

    def _objective(self, th, n_train: int, w_row: np.ndarray, pad):
        total, grad = super()._objective(th, n_train, w_row, pad)
        if self.rule_mult == 1.0 or not self.decomp:
            return total, grad
        W = w_row.sum()
        _, b, _, _ = self._split(th)
        prev = self.car_prev
        has = prev >= 0
        tgt = np.where(has, self.rho * b[np.where(has, prev, 0)], 0.0)
        d = b - tgt
        dl = self.lam_b - self.lam                       # change of the penalty weight of each car-season
        total = total + float((dl * d * d).sum()) / W
        gb = 2 * dl * d
        np.add.at(gb, prev[has], -2 * dl[has] * self.rho * d[has])
        grad = grad.copy()
        grad[self.nD:self.nD + self.nB] += gb / W
        return total, grad
