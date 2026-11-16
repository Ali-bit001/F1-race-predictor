"""PLAN3 M12: two-stage Friday forecast.

Stage 1 predicts the qualifying order from information available before qualifying (the plan-2 H1 information set:
history of earlier races + practice). Stage 2 is the frozen plan-1 H2 model (latent_111, `LatentModel` fitted on REAL
qualifying of earlier races) applied to race r with its four qualifying-derived inputs filled from stage 1:

    quali_norm - 0.5   <- (predicted qualifying rank - 1) / (n - 1) - 0.5        (stage 1)
    grid_norm - quali_norm <- 0   (penalties are not known on Friday)
    q1_gap * 10        <- mean q1_gap of earlier races at the predicted qualifying rank (expanding mean, past races only)
    pit_start          <- 0

No real qualifying value of race r enters stage 2 (they are overwritten in the test rows before prediction and restored
afterwards, so later refits still train on real qualifying of earlier races).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from pitlane.features.build import _series_feature
from pitlane.models.latent import LatentModel
from pitlane.plan3.decode import break_ties

STAGE1_HIST = ["drv_q_ew8", "drv_q_ew32", "con_q_ew8", "con_q_ew32", "drv_ew16", "con_ew16"]
STAGE1_PRACTICE = ["x_fp_gap", "x_fp_long", "x_fp_laps"]
HALFLIVES_Q = (8, 32)


def add_quali_history(entries: pd.DataFrame, halflives: tuple[int, ...] = HALFLIVES_Q) -> pd.DataFrame:
    """Shrunk, time-decayed means of EARLIER races' normalised qualifying position, per driver and per constructor
    lineage (mean of its cars per race). Same construction as the plan-1 finishing-form features (build._ewm_prior:
    race i sees only races before i). Returns the entries with the new columns, same row order."""
    df = entries.copy()
    df["race_t"] = df["race_id"].astype(float)
    cr = (df.groupby(["lineage", "race_id"], as_index=False).agg(race_t=("race_t", "first"), cq=("quali_norm", "mean")))
    cr = cr.sort_values(["lineage", "race_id"]).reset_index(drop=True)
    for h in halflives:
        cr[f"con_q_ew{h}"] = _series_feature(cr, ["lineage"], "cq", h)
        df[f"drv_q_ew{h}"] = _series_feature(df, ["driverId"], "quali_norm", h)   # rows are in race order
    df = df.merge(cr[["lineage", "race_id"] + [f"con_q_ew{h}" for h in halflives]], on=["lineage", "race_id"], how="left")
    assert (df["race_id"].to_numpy() == entries["race_id"].to_numpy()).all() and (df["driverId"].to_numpy() == entries["driverId"].to_numpy()).all()
    return df


def rank_within(values: np.ndarray, key) -> np.ndarray:
    """1 = predicted best (smallest value); exact ties broken by key, never by input order."""
    u = break_ties(-np.asarray(values, float), key)
    order = np.argsort(-u, kind="stable")
    r = np.empty(len(u))
    r[order] = np.arange(1, len(u) + 1)
    return r


class QualiStage1:
    """Ridge regression of the normalised qualifying position on pre-qualifying features; fitted on earlier races."""

    def __init__(self, cols: list[str], alpha: float = 1.0):
        self.cols, self.alpha = list(cols), alpha
        self.m: Ridge | None = None

    def fit(self, train: pd.DataFrame) -> "QualiStage1":
        self.m = Ridge(alpha=self.alpha).fit(train[self.cols].fillna(0.0).to_numpy(float), train["quali_norm"].to_numpy(float))
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self.m.predict(test[self.cols].fillna(0.0).to_numpy(float))


def q1_by_rank(train: pd.DataFrame) -> pd.Series:
    """Expanding mean of q1_gap by actual qualifying position, from earlier races only."""
    return train.groupby(train["quali_pos"].astype(int))["q1_gap"].mean()


class TwoStageH2:
    """Stage 2 = LatentModel(H2 config) fitted on real qualifying; test rows get stage-1 inputs. `log` collects, per race,
    the stage-1 prediction for scoring qualifying prediction quality."""

    def __init__(self, universe: pd.DataFrame, h2_cfg: dict, stage1_cols: list[str], alpha: float = 1.0):
        cfg = {k: v for k, v in h2_cfg.items() if k not in ("kind", "cell")}
        cell = h2_cfg["cell"]
        self.latent = LatentModel(universe, dynamic=cell[0] == "1", decomp=cell[1] == "1", ranking=cell[2] == "1", **cfg)
        assert self.latent.base_x and not self.latent.x_cols and self.latent.nX == 4, "H2 model = the four plan-1 weekend signals"
        self.stage1_cols, self.alpha = stage1_cols, alpha
        self.log: list[pd.DataFrame] = []

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        L = self.latent
        L.fit(len(train), t_ref=float(test["race_id"].min()))
        s1 = QualiStage1(self.stage1_cols, self.alpha).fit(train)
        q1 = q1_by_rank(train)
        out = np.empty(len(test))
        rows_all = test.index.to_numpy()
        for _, ix in test.reset_index(drop=True).groupby("race_id").indices.items():
            d = test.iloc[ix]
            raw = s1.predict(d)
            rk = rank_within(raw, d["driverId"].to_numpy())
            n = len(d)
            qn = (rk - 1) / max(n - 1, 1)
            gap = q1.reindex(rk.astype(int)).ffill().bfill().to_numpy(float)
            x = np.column_stack([qn - 0.5, np.zeros(n), gap * 10, np.zeros(n)])
            rows = rows_all[ix]
            saved = L.row_x[rows].copy()
            L.row_x[rows] = x
            try:
                out[ix] = L.components(rows)["u"].to_numpy()
            finally:
                L.row_x[rows] = saved
            self.log.append(pd.DataFrame({"race_id": d["race_id"].to_numpy(), "driverId": d["driverId"].to_numpy(),
                                          "q1_raw": raw, "q1_rank": rk, "quali_pos": d["quali_pos"].to_numpy(float)}))
        return out
