"""Parameter recovery on synthetic data: driver and car effects must be separable (PLAN.md WP5)."""
import numpy as np
import pandas as pd

from pitlane.models.latent import LatentModel


def _synthetic(seed=0, n_races=60):
    rng = np.random.default_rng(seed)
    drivers = [f"d{i}" for i in range(12)]
    cars = [f"c{i}" for i in range(6)]
    a = dict(zip(drivers, rng.normal(0, 0.6, 12)))
    b = dict(zip(cars, rng.normal(0, 1.0, 6)))
    rows = []
    for r in range(n_races):
        # drivers swap teams every 20 races so the effects are identifiable
        shift = r // 20
        order = []
        u = {}
        for i, d in enumerate(drivers):
            c = cars[(i // 2 + shift) % 6]
            u[(d, c)] = a[d] + b[c] + rng.gumbel()
        keys = sorted(u, key=lambda k: -u[k])
        for pos, (d, c) in enumerate(keys, 1):
            rows.append(dict(race_id=r, season=2020, driverId=d, lineage=c, circuit_type="x", pos=pos, n=12,
                             y=(pos - 1) / 11, grid_norm=0.5, quali_norm=0.5, q1_gap=0.0, pit_start=0))
    return pd.DataFrame(rows), a, b


def test_recovers_driver_and_car_strengths():
    df, a, b = _synthetic()
    m = LatentModel(df, dynamic=False, decomp=True, ranking=True, lam=0.5, maxiter=300)
    m.fit(len(df), t_ref=60)
    a_hat = np.array([m.theta[m.drivers[d]] for d in sorted(a)])
    b_hat = np.array([m.theta[m.nD + m.car[(c, None)]] for c in sorted(b)])
    assert np.corrcoef(a_hat, [a[d] for d in sorted(a)])[0, 1] > 0.8
    assert np.corrcoef(b_hat, [b[c] for c in sorted(b)])[0, 1] > 0.9


def test_gradient_matches_finite_differences():
    df, _, _ = _synthetic(n_races=10)
    for ranking in (False, True):
        m = LatentModel(df, dynamic=True, decomp=True, ranking=ranking, halflife=5, lam=1.0)
        rng = np.random.default_rng(1)
        th = rng.normal(0, 0.3, m.theta.size)
        w = 0.5 ** ((10 - df["race_id"].to_numpy()) / 5)
        pad = m._pad(10) if ranking else None
        f, g = m._objective(th, len(df), w, pad)
        for j in rng.choice(th.size, 6, replace=False):
            e = np.zeros_like(th); e[j] = 1e-6
            fd = (m._objective(th + e, len(df), w, pad)[0] - m._objective(th - e, len(df), w, pad)[0]) / 2e-6
            assert abs(fd - g[j]) < 1e-5, (ranking, j, fd, g[j])
