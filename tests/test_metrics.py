import numpy as np

from pitlane.evaluation.metrics import ece, pred_rank, race_metrics, sample_probs


def test_perfect_and_reversed_order():
    pos = np.arange(1, 11)
    cls = np.ones(10, bool)
    m = race_metrics(pos, -pos.astype(float), cls)       # utility higher = better -> perfect
    assert abs(m["spearman"] - 1) < 1e-9 and m["mae"] == 0 and m["win_hit"] == 1 and abs(m["ndcg10"] - 1) < 1e-9
    r = race_metrics(pos, pos.astype(float), cls)        # fully reversed
    assert abs(r["spearman"] + 1) < 1e-9 and r["mae"] == 5.0 and r["win_hit"] == 0


def test_hand_computed_mae():
    # predicted ranks (2,1,3) vs actual (1,2,3): errors 1,1,0 -> MAE 2/3
    m = race_metrics(np.array([1, 2, 3]), np.array([0.5, 0.9, 0.1]), np.ones(3, bool))
    assert abs(m["mae"] - 2 / 3) < 1e-12


def test_rank_and_probabilities():
    assert list(pred_rank(np.array([0.1, 0.9, 0.5]))) == [3, 1, 2]
    p = sample_probs(np.array([5.0, 0.0, 0.0, 0.0]), 1.0, np.random.default_rng(0), 2000)
    assert p[0, 0] > 0.95 and abs(p[:, 0].sum() - 1) < 1e-9
    assert ece(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0
