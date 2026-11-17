"""Weekend features on hand-made inputs, and the plan-2 metrics."""
import numpy as np
import pandas as pd

from pitlane.evaluation.metrics import pos_dist, race_metrics, rank_hist, rps_score
from pitlane.features.weekend import _shrunk_by_circuit, session_practice_metrics
from pitlane.plan2.simulate import SWITCHES, DEFAULT, simulate_race


def _laps(driver, times, pit_out_laps=()):
    return [[driver, i + 1, t, int((i + 1) in pit_out_laps)] for i, t in enumerate(times)]


def test_practice_metrics_gap_longrun_and_laps():
    rows = []
    # 6 drivers; each has an out lap, then a 7-lap stint at a constant pace, then an in lap
    for k, d in enumerate([1, 2, 3, 4, 5, 6]):
        pace = 90.0 + 0.2 * k
        rows += _laps(d, [None, 120.0] + [pace] * 7 + [130.0], pit_out_laps=(1,))
    m = session_practice_metrics(rows).set_index("dn")
    assert abs(m.loc[1, "gap_best"]) < 1e-9
    assert abs(m.loc[6, "gap_best"] - (91.0 / 90.0 - 1)) < 1e-9
    assert m["long_gap"].notna().all() and m.loc[1, "long_gap"] < 0 < m.loc[6, "long_gap"]
    assert (m["laps_rel"] == 1.0).all()                      # the out lap has no time and is not counted


def test_short_runs_have_no_long_run_pace():
    rows = []
    for d in range(1, 7):
        rows += _laps(d, [None, 90.0, 90.0, 90.0, 95.0], pit_out_laps=(1,))     # only 3 clean laps
    assert session_practice_metrics(rows)["long_gap"].isna().all()


def test_circuit_features_use_only_earlier_races():
    rl = pd.DataFrame({"circuitId": ["a", "b", "a", "a"], "v": [1.0, 0.0, 0.5, 9.0]})
    out = _shrunk_by_circuit(rl, "v", default=0.6, pseudo=2.0)
    assert np.isclose(out[0], 0.6)                           # nothing known yet -> the default
    assert np.isclose(out[2], (1.0 + 2 * 0.5) / 3)          # circuit a had one race (1.0); global mean 0.5
    rl.loc[3, "v"] = 123.0                                   # changing the *current* value must not change its feature
    assert np.isclose(_shrunk_by_circuit(rl, "v", 0.6)[3], out[3])


def test_new_metrics_hand_computed():
    pos = np.arange(1, 11)
    u = -np.array([2, 1, 3, 4, 5, 6, 7, 8, 9, 10], float)    # swaps the first two places
    m = race_metrics(pos, u, np.ones(10, bool))
    assert m["exact"] == 0.8 and m["within1"] == 1.0 and m["top5"] == 1.0 and m["points"] == 1.0


def test_within_k_hand_computed():
    pos = np.arange(1, 11)
    u = -np.array([4, 2, 3, 1, 5, 6, 7, 8, 10, 9], float)    # 1<->4 (off by 3), 9<->10 (off by 1)
    m = race_metrics(pos, u, np.ones(10, bool))
    assert m["exact"] == 0.6 and m["within1"] == 0.8 and m["within2"] == 0.8 and m["within3"] == 1.0


def test_within3_not_in_attribution_schema_and_leaderboard_header_kept(tmp_path, monkeypatch):
    import pandas as pd
    from pitlane.evaluation import harness
    from pitlane.plan3.attribution import METRICS, HIGHER_BETTER
    assert "within3" not in METRICS and HIGHER_BETTER["within3"] is True
    monkeypatch.setattr(harness, "RESULTS", tmp_path / "results")
    lb = tmp_path / "lb.csv"
    pd.DataFrame([{"method": "x", "run_id": "old", "spearman": 0.1}]).to_csv(lb, index=False)
    per = pd.DataFrame([{"spearman": 0.2, "within3": 0.5}])
    harness.save_run(tmp_path, "t", {}, pd.DataFrame({"a": [1]}), per, "dev", "y", leaderboard=lb)
    out = pd.read_csv(lb)
    assert list(out.columns) == ["method", "run_id", "spearman"] and len(out) == 2


def test_rps_is_zero_for_a_perfect_forecast_and_larger_when_wrong():
    n = 5
    perfect = np.eye(n)
    assert rps_score(perfect, np.arange(1, n + 1)) == 0.0
    shifted = np.roll(np.eye(n), 1, axis=1)
    assert rps_score(shifted, np.arange(1, n + 1)) > 0.05


def test_pos_dist_rows_and_columns_sum_to_one():
    P = pos_dist(np.array([2.0, 1.0, 0.0, -1.0]), 1.0, np.random.default_rng(0), 4000)
    assert np.allclose(P.sum(0), 1.0) and np.allclose(P.sum(1), 1.0)
    assert P[0, 0] > P[3, 0]


def test_simulator_matrix_is_doubly_stochastic_and_retirements_push_cars_back():
    u = np.linspace(1, -1, 8)
    p_dnf = np.zeros(8); p_dnf[0] = 0.9
    theta = dict(DEFAULT, tau=3.0)
    P = simulate_race(u, p_dnf, np.linspace(0, 1, 8), 0.1, 0.0, 0.0, theta, SWITCHES["S3"], 6000, np.random.default_rng(1))
    assert np.allclose(P.sum(0), 1.0) and np.allclose(P.sum(1), 1.0)
    P0 = simulate_race(u, np.zeros(8), np.linspace(0, 1, 8), 0.1, 0.0, 0.0, theta, SWITCHES["S3"], 6000, np.random.default_rng(1))
    assert P[0] @ np.arange(8) > P0[0] @ np.arange(8) + 2           # the likely retiree finishes much later on average
