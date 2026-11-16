"""A3 post-hoc sensitivity of the car-level persistence (descriptive, NOT a gate re-run).

Chosen AFTER seeing the gate result, following a referee review. Dev 2021-2023 only; reads the saved pace table.
Writes results/plan3/a3_reference_sensitivity.csv.

    python methodologies/A3_pace_persistence/sensitivity.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pitlane.config import ROOT, load_yaml
from pitlane.features import pace as P
from pitlane.plan3 import pace_gate as G

CFG = load_yaml(ROOT / "methodologies/A3_pace_persistence/config.yaml")
H, NMIN, DEV = CFG["halflife_primary"], CFG["min_prior_obs"], CFG["dev_seasons"]
B = CFG["bootstrap"]
DEFS = {"D1": "med_gap_pct", "D2": "corr_gap_pct"}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    ent = pd.read_parquet(ROOT / "data/processed/entries2.parquet")
    ent = ent[ent["season"] <= max(DEV)].sort_values(["race_id", "pos"]).reset_index(drop=True)
    pace = pd.read_parquet(ROOT / "results/plan3/a3_pace_table_2014_2023.parquet")
    return ent, pace


def refresh_car(rq: pd.DataFrame) -> pd.DataFrame:
    """Recompute car_rq / rq_tm from rq_c after rq_c was modified."""
    g = rq.groupby(["race_id", "lineage"])["rq_c"]
    rq = rq.copy()
    rq["car_rq"], rq["car_n"] = g.transform("mean"), g.transform("count")
    rq["rq_tm"] = np.where(rq["car_n"] >= 2, rq["rq_c"] - rq["car_rq"], np.nan)
    return rq


def car_frame(rq: pd.DataFrame) -> pd.DataFrame:
    """Car-race rows with the history predictor `x` (half-life 50, >= 5 earlier observations) and outcome `y`."""
    h = P.rq_history_features(rq, tuple(CFG["halflives"]), None)
    m = rq.merge(h, on=["race_id", "driverId"], how="left")
    m = m[m["season"].isin(DEV)].drop_duplicates(["race_id", "lineage"])
    m = m[m["con_n"] >= NMIN].copy()
    m["x"], m["y"] = m[f"con_rq_ew{H}"], m["car_rq"]
    # lag-1: the car's most recent earlier race offset
    full = rq.drop_duplicates(["race_id", "lineage"]).sort_values(["lineage", "race_id"])
    full["lag1"] = full.groupby("lineage")["car_rq"].shift(1)
    return m.merge(full[["race_id", "lineage", "lag1"]], on=["race_id", "lineage"], how="left")


def row(name: str, d: str, df: pd.DataFrame, x: str = "x", y: str = "y") -> dict:
    r = G.pooled_corr(df, x, y, n_boot=B["n_boot"], seed=B["seed"])
    return dict(variant=name, definition=d, statistic="pearson", **r)


def residual_rq(rq: pd.DataFrame, centre: bool = True) -> pd.DataFrame:
    """Per-race OLS of race gap on the qualifying gap (q_best); residual replaces rq_c. With `centre` the residual is
    also centred on its race median, as rq_table centres rq_c; otherwise it is the raw (mean-zero) residual."""
    rq = rq.copy()
    out = pd.Series(np.nan, index=rq.index)
    for _, g in rq.groupby("race_id"):
        ok = g[["race_gap_pct", "q_gap_pct"]].notna().all(axis=1)
        if ok.sum() < 5 or g.loc[ok, "q_gap_pct"].std() == 0:
            continue
        s = stats.linregress(g.loc[ok, "q_gap_pct"], g.loc[ok, "race_gap_pct"])
        out.loc[g.index[ok]] = g.loc[ok, "race_gap_pct"] - (s.intercept + s.slope * g.loc[ok, "q_gap_pct"])
    rq["rq_c"] = out - out.groupby(rq["race_id"]).transform("median") if centre else out
    return refresh_car(rq)


def winsorise(rq: pd.DataFrame, p: float = 0.05) -> pd.DataFrame:
    """rq_c clipped at its p / 1-p quantiles over all rows 2014-2023 (descriptive; uses the whole table)."""
    rq = rq.copy()
    lo, hi = rq["rq_c"].quantile([p, 1 - p])
    rq["rq_c"] = rq["rq_c"].clip(lo, hi)
    return refresh_car(rq)


def residualise(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    z1 = np.column_stack([np.ones(len(z)), z])
    return y - z1 @ np.linalg.lstsq(z1, y, rcond=None)[0]


def partial(df: pd.DataFrame, controls: list[str], name: str, d: str) -> dict:
    """Pooled partial correlation of x with y given `controls` (all car-level, dev rows); bootstrap over races."""
    c = df.dropna(subset=["x", "y"] + controls).copy()
    c["x"], c["y"] = residualise(c["x"].to_numpy(), c[controls].to_numpy()), residualise(c["y"].to_numpy(), c[controls].to_numpy())
    r = G.pooled_corr(c, "x", "y", n_boot=B["n_boot"], seed=B["seed"])
    return dict(variant=name, definition=d, statistic="partial_pearson", **r)


def fieldrel(e: pd.DataFrame, col: str) -> pd.Series:
    return e[col] - e.groupby("race_id")[col].transform("median")


def main() -> None:
    ent, pace = load()
    ent = ent.copy()
    rows = []
    for d, col in DEFS.items():
        base = P.rq_table(pace, ent, col)                                   # gate definition (q_best)
        # entries2.q1_gap fills a missing Q1 time with 0.10 (plan-1 imputation, ingest.py); the faithful analogue of the
        # gate's NaN q_best_gap sets those (q1_missing) to NaN.
        e1 = ent.assign(q_best_gap=ent["q1_gap"].where(ent["q1_missing"] == 0))
        q1 = P.rq_table(pace, e1, col)                                      # Q1-based offset (missing Q1 = NaN)
        q1f = P.rq_table(pace, ent.assign(q_best_gap=ent["q1_gap"]), col)   # Q1 filled with 0.10
        cf_base, cf_q1 = car_frame(base), car_frame(q1)
        rows.append(row("gate: rq vs q_best (reproduces a3_persistence)", d, cf_base))
        rows.append(row("rq vs q1_gap (missing Q1 = NaN)", d, cf_q1))
        rows.append(row("rq vs q1_gap, Q1 filled with plan-1 imputation 0.10", d, car_frame(q1f)))
        rows.append(row("residual of race gap on q_best_gap (per race, median-centred)", d, car_frame(residual_rq(base))))
        rows.append(row("residual of race gap on q_best_gap (per race, raw mean-zero)", d, car_frame(residual_rq(base, False))))
        rows.append(row("winsorised 5% rq_c (rq vs q_best)", d, car_frame(winsorise(base))))
        sp = cf_base.dropna(subset=["x", "y"]).copy()
        sp["x"], sp["y"] = stats.rankdata(sp["x"]), stats.rankdata(sp["y"])
        rows.append({**row("spearman (ranks over pooled dev rows; rq vs q_best)", d, sp), "statistic": "spearman"})
        rows.append(row("lag-1 (previous race's car offset; rq vs q_best)", d, cf_base, x="lag1"))

        # partial correlations for the Q1-based offset, car level, field-relative controls
        qq = e1[["race_id", "lineage", "q_best_gap", "q1_gap"]].assign(qb=lambda t: t["q_best_gap"] * 100, q1=lambda t: t["q1_gap"] * 100)
        qq["qb"], qq["q1"] = fieldrel(qq, "qb"), fieldrel(qq, "q1")
        qc = qq.groupby(["race_id", "lineage"], as_index=False)[["qb", "q1"]].mean()
        gp = q1[["race_id", "driverId", "lineage", "season", "race_gap_pct"]].copy()
        gp["rq_c"] = fieldrel(gp, "race_gap_pct")                           # race pace level, field-relative
        gp["rq_tm"] = np.nan
        gh = gp.groupby(["race_id", "lineage"], as_index=False)["rq_c"].mean().rename(columns={"rq_c": "pace_now"})
        hist = P.rq_history_features(refresh_car(gp), (H,), None)
        gx = gp.merge(hist, on=["race_id", "driverId"], how="left").drop_duplicates(["race_id", "lineage"])
        gx = gx[["race_id", "lineage", f"con_rq_ew{H}"]].rename(columns={f"con_rq_ew{H}": "pace_hist"})
        m = cf_q1.merge(qc, on=["race_id", "lineage"], how="left").merge(gx, on=["race_id", "lineage"], how="left")
        rows.append(partial(m, ["qb", "q1"], "Q1-based offset, partial given this race's q_best and q1 gaps", d))
        rows.append(partial(m, ["qb", "q1", "pace_hist"],
                            "Q1-based offset, partial given q_best, q1 and the decayed mean of the car's earlier race-pace gap", d))
        # field-relative rq_c against the same race's field-relative qualifying gap, mean of per-race Pearson (dev)
        t = base[base["season"].isin(DEV)].dropna(subset=["rq_c", "q_gap_pct"]).copy()
        t["qc"] = fieldrel(t, "q_gap_pct")
        pr = t.groupby("race_id").apply(lambda g: g["rq_c"].corr(g["qc"]), include_groups=False)
        rows.append(dict(variant="rq_c vs same-race field-relative q_best gap (mean of per-race Pearson)", definition=d,
                         statistic="within_race_mean_pearson", r=float(pr.mean()), lo=np.nan, hi=np.nan,
                         n_obs=len(t), n_races=int(pr.notna().sum())))
        t["qraw"] = t["q_gap_pct"]
        rows.append(dict(variant="uncentred rq vs q_best gap, pooled Pearson", definition=d, statistic="pooled_pearson",
                         r=float(t["rq"].corr(t["qraw"])), lo=np.nan, hi=np.nan, n_obs=len(t), n_races=t["race_id"].nunique()))
    out = pd.DataFrame(rows)
    out.insert(0, "note", "post-hoc descriptive; half-life 50; car level; >=5 earlier obs; 10,000 race resamples seed 0; dev 2021-2023")
    path = ROOT / "results/plan3/a3_reference_sensitivity.csv"
    out.to_csv(path, index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 100)
    print(out.drop(columns="note").round(3).to_string(index=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
