"""Plan-2 latent model options: extra weekend columns, H0 (no weekend signal), analytic gradients."""
import numpy as np
import pytest
from scipy.optimize import check_grad

from pitlane.models.latent import LatentModel
from test_latent_recovery import _synthetic


def _frame(seed=0):
    df, a, b = _synthetic(seed, n_races=30)
    rng = np.random.default_rng(seed + 5)
    df["x_sig"] = rng.normal(size=len(df))
    df["x_miss"] = (rng.random(len(df)) < 0.5).astype(float)
    return df.sort_values(["race_id", "pos"]).reset_index(drop=True)


@pytest.mark.parametrize("ranking", [False, True])
@pytest.mark.parametrize("dynamic,decomp", [(False, True), (True, True), (True, False)])
def test_gradient_matches_finite_differences(ranking, dynamic, decomp):
    df = _frame()
    m = LatentModel(df, dynamic=dynamic, decomp=decomp, ranking=ranking, x_cols=["x_sig", "x_miss"], lam_x=0.7)
    n = len(df)
    w = np.ones(n) if not dynamic else 0.5 ** ((df["race_id"].max() - df["race_id"].to_numpy()) / 10)
    pad = m._pad(int(df["race_id"].max()) + 1) if ranking else None
    th0 = np.random.default_rng(1).normal(0, 0.3, len(m.theta))
    f = lambda th: m._objective(th, n, w, pad)[0]
    g = lambda th: m._objective(th, n, w, pad)[1]
    err = check_grad(f, g, th0, epsilon=1e-6)
    assert err < 1e-4 * max(1.0, np.linalg.norm(g(th0))), err


def test_h0_has_no_weekend_parameters_and_runs():
    df = _frame()
    m = LatentModel(df, dynamic=True, decomp=True, ranking=True, base_x=False)
    assert m.nX == 0 and m.row_x.shape == (len(df), 0)
    train, test = df[df.race_id < 25], df[df.race_id == 25]
    s = m.fit_predict(train.reset_index(drop=True), test)
    assert len(s) == len(test) and np.isfinite(s).all()


def test_extra_signal_coefficient_is_recovered():
    rng = np.random.default_rng(3)
    df = _frame(2)
    # re-generate utilities with a planted effect of x_sig and re-rank within race
    df["u"] = rng.normal(size=len(df)) * 0.5 + 1.5 * df["x_sig"]
    df["pos"] = df.groupby("race_id")["u"].rank(ascending=False).astype(int)
    df["y"] = (df["pos"] - 1) / 11
    df = df.sort_values(["race_id", "pos"]).reset_index(drop=True)
    m = LatentModel(df, dynamic=False, decomp=False, ranking=True, base_x=False, x_cols=["x_sig"], lam_x=0.01)
    m.fit(len(df), t_ref=float(df["race_id"].max() + 1))
    assert m.beta["x_sig"] > 0.8
