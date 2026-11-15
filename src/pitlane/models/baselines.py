"""M00: heuristic baselines. Each maps a pre-race column to a utility (higher = better)."""
from __future__ import annotations

import numpy as np
import pandas as pd

HEURISTICS = {
    "grid": "grid_norm",          # start from the grid order
    "quali": "quali_norm",        # qualifying order
    "standings": "pts_norm",      # championship order before the race
    "last_race": "drv_last",      # previous race finishing position
    "season_mean": "drv_season",  # season-to-date mean finishing position
}


class Heuristic:
    def __init__(self, which: str, **_):
        self.col = HEURISTICS[which]

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return -test[self.col].to_numpy(float)
