"""Latent model with a shared engine-supplier term (PLAN3.md section 4 M14, part 1; Amendment 3 item 4).

    b_total[car] = b[car] + e[supplier, season]          car = (lineage, season), as in LatentModel (dynamic = True)

`b` keeps the plan-1 car penalty and carry-over decay unchanged. `e` has its own ridge penalty `lam_e * e'e`
(no carry-over across seasons: every (supplier, season) level starts from 0 and is pooled only over the cars that
share the supplier in that season). The supplier of a car in season s is read from the `supplier` column of the
universe (merged from data/external/power_units.csv on (season, constructorId), a season-start fact known before
the season), so a race never uses information from after it. Pooling is by supplier value, not constructorId.

`lam_e = inf` (or None) switches the term off: no e parameters exist and every method is the LatentModel code path,
so results are bit for bit those of LatentModel.

theta layout: [LatentModel theta (a, b, g, beta), e].

A (supplier, season) level with no row among the training rows of a fit has no data term; its parameter is pinned to exactly 0
(gradient masked, start value reset), so a level cannot move before its first race.

POST-HOC variant `carry_on_total=True` (M14 prestatement.md, section dated 2026-11-17): the carry-over penalty
lam * (T_s - rho * T_{s-1})^2 acts on the car's total strength T = b + e instead of on b alone; b gets a plain ridge
lam_b * b'b (default lam_b = lam) and e keeps lam_e * e'e. Default False: the original model, unchanged.
Revision 2 (look-ahead fix): in this variant a car with no row among the training rows is pinned: no carry-over term, no lam_b
ridge, so which lineages exist in later seasons cannot move the fit. Its predicted b is rho * b of last season's car (0 if none),
the value LatentModel's carry-over assigns a car without data; its e is the supplier level's (0 before the level's first race).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitlane.models.latent import LatentModel


class LatentSupplierModel(LatentModel):
    def __init__(self, universe: pd.DataFrame, dynamic: bool, decomp: bool, ranking: bool,
                 lam_e: float | None = 10.0, carry_on_total: bool = False,
                 lam_b: float | None = None, **kw):
        super().__init__(universe, dynamic=dynamic, decomp=decomp, ranking=ranking, **kw)
        self.carry_total = bool(carry_on_total)
        self.lam_b = self.lam if lam_b is None else float(lam_b)
        self._act_car: np.ndarray | None = None
        self.nbase = len(self.theta)
        self.on = lam_e is not None and np.isfinite(lam_e)
        self.lam_e = float(lam_e) if self.on else float("inf")
        if not self.on:
            self.nE = 0
            return
        if not (dynamic and decomp):
            raise ValueError("supplier pooling needs dynamic=True and decomp=True (e adds to the per-season car effect)")
        if "supplier" not in universe or universe["supplier"].isna().any():
            raise ValueError("universe needs a complete `supplier` column")
        pairs = sorted(set(zip(universe["supplier"], universe["season"])), key=lambda p: (p[1], p[0]))
        self.sup_levels = {p: i for i, p in enumerate(pairs)}
        self.nE = len(pairs)
        # supplier level of every car-season (a lineage has one supplier per season: checked)
        per_car: dict[int, set[int]] = {}
        row_e = np.array([self.sup_levels[(s, y)] for s, y in zip(universe["supplier"], universe["season"])])
        for b, e in zip(self.row_b, row_e):
            per_car.setdefault(int(b), set()).add(int(e))
        if any(len(v) != 1 for v in per_car.values()):
            raise ValueError("a car-season has more than one supplier")
        self.row_e = row_e
        self.car_e = np.zeros(self.nB, dtype=int)
        for b, v in per_car.items():
            self.car_e[b] = next(iter(v))
        self.theta = np.concatenate([self.theta, np.zeros(self.nE)])

    def _active(self, n_train: int) -> np.ndarray:
        """Levels with at least one training row."""
        act = np.zeros(self.nE, dtype=bool)
        act[self.row_e[:n_train]] = True
        return act

    def _active_cars(self, n_train: int) -> np.ndarray:
        """Cars (b levels) with at least one training row."""
        act = np.zeros(self.nB, dtype=bool)
        act[self.row_b[:n_train]] = True
        return act

    def pin_inactive(self, n_train: int) -> None:
        """Reset levels without a training row to exactly 0 (a warm start cannot leave them elsewhere)."""
        if self.on:
            self.theta[self.nbase:][~self._active(n_train)] = 0.0
            self._act_car = self._active_cars(n_train)

    def _b_pinned(self) -> np.ndarray:
        """b with every car lacking a training row set to rho * b of last season's car (0 without one), in season order."""
        b = self._split(self.theta)[1].copy()
        for i in np.flatnonzero(~self._act_car):
            b[i] = self.rho * b[self.car_prev[i]] if self.car_prev[i] >= 0 else 0.0
        return b

    def fit(self, n_train: int, t_ref: float):
        self.pin_inactive(n_train)
        super().fit(n_train, t_ref)

    # ------------------------------------------------------------------ views
    def _split(self, th):
        return super()._split(th[:self.nbase])

    def _e(self, th) -> np.ndarray:
        return th[self.nbase:]

    def _utility(self, th, rows):
        u = super()._utility(th, rows)
        if self.on:
            u = u + self._e(th)[self.row_e[rows]]
        return u

    # ------------------------------------------------------------------ objective
    def _objective(self, th, n_train: int, w_row: np.ndarray, pad):
        total, grad = super()._objective(th[:self.nbase] if not self.on else th, n_train, w_row, pad)
        if not self.on:
            return total, grad
        W = w_row.sum()
        act = self._active(n_train)
        e = np.where(act, self._e(th), 0.0)         # a level without a training row is a constant 0 (value and gradient agree)
        # d loss / d b: base gradient (already divided by W) minus the analytic car-penalty gradient
        _, b, _, _ = self._split(th)
        prev, has = self.car_prev, self.car_prev >= 0
        d = b - np.where(has, self.rho * b[np.where(has, prev, 0)], 0.0)
        pgb = 2 * self.lam * d
        np.add.at(pgb, prev[has], -2 * self.lam * self.rho * d[has])
        gb_loss = grad[self.nD:self.nD + self.nB] * W - pgb
        if not self.carry_total:
            ge = (np.bincount(self.car_e, gb_loss, self.nE) + 2 * self.lam_e * e) / W
            ge[~act] = 0.0
            return total + self.lam_e * float(e @ e) / W, np.concatenate([grad, ge])
        # post-hoc variant: carry-over on T = b + e, plain ridge on b, ridge on e.
        # A car with no training row is pinned (revision 2): it has no carry-over and no ridge term, so the future entry list
        # cannot reach the fit; its prediction value is set in components() (rho * b of last season's car, as LatentModel).
        ac = self._active_cars(n_train)
        T = b + e[self.car_e]
        dT = np.where(ac, T - np.where(has, self.rho * T[np.where(has, prev, 0)], 0.0), 0.0)
        pgT = 2 * self.lam * dT
        np.add.at(pgT, prev[has], -2 * self.lam * self.rho * dT[has])
        bm = np.where(ac, b, 0.0)
        value = total * W - self.lam * float(d @ d) + self.lam * float(dT @ dT) + self.lam_b * float(bm @ bm) + self.lam_e * float(e @ e)
        out = grad.copy()
        out[self.nD:self.nD + self.nB] = (gb_loss + pgT + 2 * self.lam_b * bm) / W
        ge = (np.bincount(self.car_e, gb_loss + pgT, self.nE) + 2 * self.lam_e * e) / W
        ge[~act] = 0.0
        return value / W, np.concatenate([out, ge])

    # ------------------------------------------------------------------ explanation
    def components(self, rows: np.ndarray) -> pd.DataFrame:
        comp = super().components(rows)
        if self.on and self.carry_total and self._act_car is not None:
            delta = self._b_pinned()[self.row_b[rows]] - self._split(self.theta)[1][self.row_b[rows]]
            comp["b_car"] = comp["b_car"] + delta
            comp["u"] = comp["u"] + delta
        if self.on:
            e = self._e(self.theta)[self.row_e[rows]]
            comp.insert(comp.columns.get_loc("g_circuit") + 1, "e_supplier", e)
            comp["u"] = comp["u"] + e
        return comp

    def supplier_effects(self) -> pd.DataFrame:
        """Fitted e per (supplier, season)."""
        if not self.on:
            return pd.DataFrame(columns=["supplier", "season", "e"])
        e = self._e(self.theta)
        return pd.DataFrame([dict(supplier=s, season=y, e=float(e[i])) for (s, y), i in self.sup_levels.items()])
