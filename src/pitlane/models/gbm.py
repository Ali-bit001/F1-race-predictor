"""Family G: tree models with the same three switches (PLAN.md M01, M02, M04, RF baseline)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from pitlane.features.build import gbm_feature_cols


class GBMModel:
    """F1 dynamic -> EWMA features + time-decayed sample weights; F2 decomposed -> latent strengths
    replace raw driver/constructor form; F3 ranking -> LambdaRank instead of regression."""

    def __init__(self, dynamic: bool, decomp: bool, ranking: bool, halflife: int = 16, num_leaves: int = 7,
                 n_estimators: int = 150, learning_rate: float = 0.05, min_child_samples: int = 40,
                 seed: int = 0, n_jobs: int = 1, **_):
        self.dynamic, self.decomp, self.ranking, self.halflife = dynamic, decomp, ranking, halflife
        self.cols = gbm_feature_cols(dynamic, decomp, halflife if dynamic else None)
        self.params = dict(num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=learning_rate,
                           min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=seed, n_jobs=n_jobs, verbose=-1)
        self.model = None

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        # Rows whose latent strengths are unavailable (earliest seasons) are simply dropped from training.
        tr = train.dropna(subset=self.cols) if self.decomp else train
        w = None
        if self.dynamic:
            w = 0.5 ** ((float(test["race_id"].min()) - tr["race_id"].to_numpy()) / self.halflife)
        if self.ranking:
            rel = np.clip(tr["n"].to_numpy() - tr["pos"].to_numpy(), 0, 30).astype(int)
            groups = tr.groupby("race_id", sort=True).size().to_numpy()
            self.model = lgb.LGBMRanker(objective="lambdarank", label_gain=list(range(31)), **self.params)
            self.model.fit(tr[self.cols], rel, group=groups, sample_weight=w)
            return self.model.predict(test[self.cols])
        self.model = lgb.LGBMRegressor(objective="regression", **self.params)
        self.model.fit(tr[self.cols], tr["y"], sample_weight=w)
        return -self.model.predict(test[self.cols])


class RFModel:
    """Random-forest direct regression on the conventional (F1/F2/F3 all off) feature set."""

    def __init__(self, n_estimators: int = 200, min_samples_leaf: int = 20, max_features: float = 0.5,
                 seed: int = 0, n_jobs: int = 1, **_):
        self.cols = gbm_feature_cols(False, False)
        self.kw = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,
                       random_state=seed, n_jobs=n_jobs)

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        m = RandomForestRegressor(**self.kw).fit(train[self.cols], train["y"])
        return -m.predict(test[self.cols])
