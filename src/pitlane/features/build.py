"""Walk-forward-safe feature construction.

Every history feature for a row in race t uses only rows from races < t. This is verified by
tests/test_leakage.py (features built on a truncated frame with the race's own results blanked
must equal features built on the full frame).

Targets are `y` (normalised finishing position, 0 = win, 1 = last) and `dnf`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIOR = 0.5       # prior mean of normalised finishing position
PSEUDO = 1.0      # pseudo-observations of the prior (shrinkage for cold starts)
HALFLIVES = (8, 16, 32)  # in races


def _ewm_prior(t: np.ndarray, v: np.ndarray, h: float | None, prior: float = PRIOR) -> np.ndarray:
    """For a chronological series (race ordinals t, values v) return, for each element, the
    shrunk time-decayed mean of all *earlier* elements. h=None -> plain expanding mean."""
    out = np.empty(len(t))
    s = w = 0.0
    last_t = None
    for i in range(len(t)):
        if h is not None and last_t is not None:
            f = 0.5 ** ((t[i] - last_t) / h)
            s, w = s * f, w * f
        out[i] = (s + prior * PSEUDO) / (w + PSEUDO)
        if not np.isnan(v[i]):
            s += v[i]
            w += 1.0
        last_t = t[i]
    return out


def _series_feature(df: pd.DataFrame, keys: list[str], col: str, h: float | None, prior: float = PRIOR,
                    t_col: str = "race_t") -> pd.Series:
    res = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(keys, sort=False):
        res.loc[g.index] = _ewm_prior(g[t_col].to_numpy(float), g[col].to_numpy(float), h, prior)
    return res


def add_features(entries: pd.DataFrame, halflives: tuple[int, ...] = HALFLIVES) -> pd.DataFrame:
    df = entries.sort_values(["race_id", "pos"]).reset_index(drop=True).copy()
    df["race_t"] = df["race_id"].astype(float)

    # Constructor-race series: mean y / dnf of the constructor's cars in each race.
    cr = (df.groupby(["lineage", "race_id"], as_index=False)
            .agg(race_t=("race_t", "first"), cy=("y", "mean"), cdnf=("dnf", "mean"),
                 circuitId=("circuitId", "first"), season=("season", "first")))
    cr = cr.sort_values(["lineage", "race_id"]).reset_index(drop=True)

    def con_feat(col: str, h: float | None, prior: float) -> pd.DataFrame:
        s = _series_feature(cr, ["lineage"], col, h, prior)
        return pd.DataFrame({"lineage": cr["lineage"], "race_id": cr["race_id"], "v": s})

    def merge_con(name: str, col: str, h: float | None, prior: float) -> None:
        nonlocal df
        m = con_feat(col, h, prior).rename(columns={"v": name})
        df = df.merge(m, on=["lineage", "race_id"], how="left")

    # --- Expanding (F1-off) and exponentially weighted (F1-on) driver / constructor form ---
    variants = [("all", None)] + [(f"ew{h}", h) for h in halflives]
    df = df.sort_values(["driverId", "race_id"]).reset_index(drop=True)
    for tag, h in variants:
        df[f"drv_{tag}"] = _series_feature(df, ["driverId"], "y", h)
        df[f"drv_dnf_{tag}"] = _series_feature(df, ["driverId"], "dnf", h, prior=0.15)
        merge_con(f"con_{tag}", "cy", h, PRIOR)
        merge_con(f"con_dnf_{tag}", "cdnf", h, 0.15)
        df = df.sort_values(["driverId", "race_id"]).reset_index(drop=True)
    df["drv_cnt"] = df.groupby("driverId").cumcount()

    # --- Circuit history (expanding, shrunk) ---
    df["drv_circ"] = _series_feature(df, ["driverId", "circuitId"], "y", None)
    cr2 = cr.copy()
    cr2["con_circ"] = _series_feature(cr2, ["lineage", "circuitId"], "cy", None)
    df = df.merge(cr2[["lineage", "race_id", "con_circ"]], on=["lineage", "race_id"], how="left")

    # --- Simple heuristics' inputs (M00) ---
    df = df.sort_values(["driverId", "race_id"]).reset_index(drop=True)
    df["drv_last"] = df.groupby("driverId")["y"].shift(1).fillna(PRIOR)
    df["drv_season"] = _series_feature(df, ["driverId", "season"], "y", None)
    df["pts_cum"] = (df.groupby(["driverId", "season"])["points"].cumsum()
                     .groupby([df["driverId"], df["season"]]).shift(1).fillna(0.0))
    df["pts_rank"] = df.groupby("race_id")["pts_cum"].rank(ascending=False, method="average")
    df["pts_norm"] = (df["pts_rank"] - 1) / (df["n"] - 1).clip(lower=1)

    return df.sort_values(["race_id", "pos"]).reset_index(drop=True)


WEEKEND = ["grid_norm", "quali_norm", "q1_gap", "pit_start", "n"]
CIRCUIT = ["drv_circ", "con_circ"]


def gbm_feature_cols(dynamic: bool, decomp: bool, halflife: int | None = None) -> list[str]:
    """Feature set for Family G cell (F1 dynamic, F2 decomposed). See PLAN.md section 4 / M04."""
    tag = f"ew{halflife}" if dynamic else "all"
    cols = WEEKEND + CIRCUIT + [f"drv_dnf_{tag}", f"con_dnf_{tag}", "drv_cnt"]
    if decomp:
        v = "lat1" if dynamic else "lat0"
        cols += [f"{v}_a", f"{v}_b", f"{v}_g"]
    else:
        cols += [f"drv_{tag}", f"con_{tag}"]
    return cols
