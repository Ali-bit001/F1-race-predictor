"""Plan-2 history features for race R must not change when R's own outcome (and everything after) is removed.

Weekend features (practice, qualifying detail, sprint, weather) come from sessions that finish before the
race; they are checked separately for the horizon: no column may be derived from the race result.
"""
import numpy as np
import pandas as pd
import pytest

from pitlane.config import PROCESSED
from pitlane.features.build import add_features
from pitlane.features.weekend import add_history2

RAW = PROCESSED / "entries.parquet"
OUTCOME = ["y", "dnf", "points", "pos", "status", "classified", "positionText", "finish_order"]
HISTORY = ["drv_inc_ew16", "con_mech_ew16", "stick", "circ_vol", "circ_dnf", "newregs", "new_team", "rookie"]


@pytest.mark.skipif(not RAW.exists(), reason="run python -m pitlane.data.ingest first")
@pytest.mark.parametrize("race_id", [60, 140, 230])
def test_history_features_do_not_see_the_target_race(race_id):
    raw = pd.read_parquet(RAW)
    full = add_history2(add_features(raw))
    cut = raw[raw["race_id"] <= race_id].copy()
    m = cut["race_id"] == race_id
    for c in OUTCOME:
        cut.loc[m, c] = np.nan if cut[c].dtype != object and cut[c].dtype != bool else None
    trunc = add_history2(add_features(cut))
    a = full[full["race_id"] == race_id].set_index("driverId").sort_index()
    b = trunc[trunc["race_id"] == race_id].set_index("driverId").sort_index()
    pd.testing.assert_frame_equal(a[HISTORY], b[HISTORY], check_exact=False, rtol=1e-9, atol=1e-9)


def test_history_features_change_when_earlier_races_change():
    """Sanity: the features really do depend on earlier outcomes (so the test above is not vacuous)."""
    if not RAW.exists():
        pytest.skip("no data")
    raw = pd.read_parquet(RAW)
    full = add_history2(add_features(raw))
    alt = raw.copy()
    alt.loc[alt["race_id"] == 50, "dnf"] = 1
    alt.loc[alt["race_id"] == 50, "status"] = "Engine"
    other = add_history2(add_features(alt))
    r = 140
    a = full[full["race_id"] == r][["drv_dnf_ew16", "con_mech_ew16", "circ_dnf"]].to_numpy()
    b = other[other["race_id"] == r][["drv_dnf_ew16", "con_mech_ew16", "circ_dnf"]].to_numpy()
    assert not np.allclose(a, b)
