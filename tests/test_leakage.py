"""Features for race R must not change when R's own result and everything after it are removed."""
import numpy as np
import pandas as pd
import pytest

from pitlane.config import PROCESSED
from pitlane.features.build import add_features

RAW = PROCESSED / "entries.parquet"


@pytest.mark.skipif(not RAW.exists(), reason="run python -m pitlane.data.ingest first")
@pytest.mark.parametrize("race_id", [40, 120, 200])
def test_no_future_information(race_id):
    raw = pd.read_parquet(RAW)
    full = add_features(raw)
    cut = raw[raw["race_id"] <= race_id].copy()
    # blank the target race's outcome: only pre-race columns may remain
    m = cut["race_id"] == race_id
    for c in ["y", "dnf", "points"]:
        cut.loc[m, c] = np.nan
    trunc = add_features(cut)
    a = full[full["race_id"] == race_id].set_index("driverId").sort_index()
    b = trunc[trunc["race_id"] == race_id].set_index("driverId").sort_index()
    cols = [c for c in full.columns if c.startswith(("drv_", "con_", "pts_"))]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, rtol=1e-9, atol=1e-9)
