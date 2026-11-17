"""M20 phase 1, POST-HOC robustness of the persistence numbers (referee review). Nothing here changes the declared
decision; every row is labelled post hoc. Dev decision block 2021-2023, half-life 50, as run.py.

    python methodologies/M20_racecraft/robustness.py   ->  results/plan3/m20_robustness.csv
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pitlane.config import ROOT, load_yaml
from pitlane.features import racecraft as RC
from pitlane.plan3 import pace_gate as G

spec = importlib.util.spec_from_file_location("m20run", Path(__file__).parent / "run.py")
M = importlib.util.module_from_spec(spec)
sys.modules["m20run"] = M
spec.loader.exec_module(M)
CFG = load_yaml(Path(__file__).parent / "config.yaml")
H = CFG["halflife_primary"]
ROWS: list[dict] = []


def persist(tab: pd.DataFrame, col: str, unit: str, group: str, name: str, nmin: int = 5, drop=None,
            scope: str = "dev", note: str = "") -> float:
    """Pooled Pearson of `col` with its decayed history (earlier races of `unit`); history built from ALL rows of the
    table passed in (so `drop` removes rows only from the scored set, not from the history)."""
    t = tab.dropna(subset=[col]).rename(columns={col: "y"})
    h = RC.history_values(t.rename(columns={"y": "yv"}), unit, "yv", (H,))
    d = t.merge(h, on=["race_id", unit], how="left")
    d = d[(d["yv_n"] >= nmin) & d["season"].isin(CFG["dev_seasons"])]
    if drop is not None:
        d = d[~drop(d)]
    r = G.pooled_corr(d, f"yv_h{H}", "y", n_boot=10000, seed=0)
    sp = float(stats.spearmanr(d[f"yv_h{H}"], d["y"])[0])
    ROWS.append(dict(group=group, check=name, scope=scope, r=r["r"], lo=r["lo"], hi=r["hi"], spearman=sp,
                     n_obs=r["n_obs"], n_races=r["n_races"], note=note))
    return r["r"]


def resid(d: pd.DataFrame, y: str, ctrl: str) -> np.ndarray:
    ok = d[[y, ctrl]].notna().all(axis=1)
    out = np.full(len(d), np.nan)
    x = d.loc[ok, ctrl].to_numpy(float)
    v = d.loc[ok, y].to_numpy(float)
    b = np.polyfit(x, v, 1)
    out[ok.to_numpy()] = v - np.polyval(b, x)
    return out


def main() -> None:
    laps, pits, ent = M.load(CFG)
    vals = M.race_values(laps, pits, ent, CFG)
    drv, team_tab = vals["driver"].dropna(subset=["race_id"]), vals["team"]
    teamdef = ent[RC.KEY + ["lineage"]]
    grid = ent[RC.KEY + ["grid", "pos"]]

    # ---------------- consistency
    c = drv.dropna(subset=["cons_c"]).merge(grid, on=RC.KEY, how="left")
    persist(c, "cons_c", "driverId", "consistency", "declared")
    # within-race rank correlation with grid and finishing position
    sp_g = c.groupby("race_id").apply(lambda g: stats.spearmanr(g["cons_c"], g["grid"])[0] if len(g) > 5 else np.nan)
    sp_p = c.groupby("race_id").apply(lambda g: stats.spearmanr(g["cons_c"], g["pos"])[0] if len(g) > 5 else np.nan)
    dv = c[c["season"].isin(CFG["dev_seasons"])]["race_id"].unique()
    for nm, s in (("within-race Spearman cons_c vs grid (mean over dev races)", sp_g), ("within-race Spearman cons_c vs finishing position (mean over dev races)", sp_p)):
        v = s.loc[s.index.isin(dv)].to_numpy(float)
        m, lo, hi = G.boot_mean(v)
        ROWS.append(dict(group="consistency", check=nm, scope="dev", r=m, lo=lo, hi=hi, n_races=int((~np.isnan(v)).sum())))
    # partial out current grid; partial out the car's own consistency history
    t = c.rename(columns={"cons_c": "y"})
    h = RC.history_values(t.rename(columns={"y": "yv"}), "driverId", "yv", (H,))
    d = t.merge(h, on=["race_id", "driverId"], how="left")
    car = c.groupby(["race_id", "lineage"], as_index=False).agg(season=("season", "first"), car_y=("cons_c", "mean"))
    ch = RC.history_values(car.rename(columns={"car_y": "cy"}), "lineage", "cy", (H,)).rename(columns={"cy_n": "car_n"})
    d = d.merge(ch, on=["race_id", "lineage"], how="left")
    d = d[(d["yv_n"] >= 5) & d["season"].isin(CFG["dev_seasons"])].copy()
    for nm, ctrl in (("partial: current grid", "grid"), ("partial: car's own consistency history", f"cy_h{H}")):
        e = d.copy()
        e["ry"], e["rx"] = resid(e, "y", ctrl), resid(e, f"yv_h{H}", ctrl)
        r = G.pooled_corr(e.dropna(subset=["ry", "rx"]), "rx", "ry", n_boot=10000, seed=0)
        ROWS.append(dict(group="consistency", check=nm, scope="dev", n_obs=r["n_obs"], n_races=r["n_races"], r=r["r"], lo=r["lo"], hi=r["hi"]))
    # car-level persistence
    persist(car, "car_y", "lineage", "consistency", "car-level (mean of the car's drivers)")
    for nm, names in (("without mazepin", ["mazepin"]), ("without mazepin, mick_schumacher, zhou", ["mazepin", "mick_schumacher", "zhou"])):
        persist(c, "cons_c", "driverId", "consistency", nm, drop=lambda x, n=names: x["driverId"].isin(n))
    persist(c, "cons_c", "driverId", "consistency", ">= 20 earlier observations", nmin=20)
    c["tm"] = RC.teammate_relative(c, "cons_c", teamdef)
    persist(c, "tm", "driverId", "consistency", "teammate-relative")

    # ---------------- tyre degradation, lap 1
    td = drv.dropna(subset=["deg_c"]).copy()
    td["tm"] = RC.teammate_relative(td, "deg_c", teamdef)
    persist(td, "deg_c", "driverId", "tyre_degradation", "declared")
    persist(td, "tm", "driverId", "tyre_degradation", "teammate-relative (the PLAN3 2.1 wording)")
    l1 = drv.dropna(subset=["lap1_c"]).copy()
    l1["tm"] = RC.teammate_relative(l1, "lap1_c", teamdef)
    l1["clip5"], l1["tm_clip5"] = l1["lap1_c"].clip(-5, 5), l1["tm"].clip(-5, 5)
    persist(l1, "lap1_c", "driverId", "lap1_gain", "declared")
    persist(l1, "clip5", "driverId", "lap1_gain", "clipped +-5")
    persist(l1, "lap1_c", "driverId", "lap1_gain", "dropping |value| > 6 (scored rows only)", drop=lambda x: x["y"].abs() > 6)
    persist(l1, "tm", "driverId", "lap1_gain", "teammate-relative")
    persist(l1, "tm_clip5", "driverId", "lap1_gain", "teammate-relative, clipped +-5")
    persist(l1, "tm", "driverId", "lap1_gain", "teammate-relative, dropping |value| > 6", drop=lambda x: x["y"].abs() > 6)

    # ---------------- crew time loss
    base = lambda **k: RC.crew_values(pits, ent, k.get("margin", 20.0), 3.0, k.get("min_stops", 2), k.get("cap", 60.0))
    persist(base().merge(team_tab[["season", "round", "race_id"]].drop_duplicates(), on=RC.RACE), "crew_loss_c", "lineage", "crew_time_loss", "declared (cap 60, margin 20)")
    rid = ent[RC.RACE + ["race_id"]].drop_duplicates()
    allp = pits[pits["season"] <= 2023]
    rf = allp[allp["duration_s"] > 300].groupby(RC.RACE).size().reset_index()[RC.RACE]
    rf_ids = set(rf.merge(rid, on=RC.RACE)["race_id"])
    n_rf_dev = len(rf_ids & set(ent[ent["season"].isin(CFG["dev_seasons"])]["race_id"]))
    ROWS.append(dict(group="crew_time_loss", check="races with any stop > 300 s (all 2014-2023 / dev)", scope="all", n_races=len(rf_ids), note=f"dev {n_rf_dev}"))
    cv = base().merge(rid, on=RC.RACE)
    persist(cv, "crew_loss_c", "lineage", "crew_time_loss", "drop races with a stop > 300 s (scored rows only)", drop=lambda x: x["race_id"].isin(rf_ids))
    for nm, kw in (("margin 10 s", dict(margin=10.0)), ("margin 30 s", dict(margin=30.0)), ("minimum 1 stop", dict(min_stops=1)),
                   ("cap 40 s", dict(cap=40.0)), ("cap 45 s", dict(cap=45.0)), ("cap 50 s", dict(cap=50.0)),
                   ("cap 100 s", dict(cap=100.0)), ("cap 300 s", dict(cap=300.0))):
        persist(base(**kw).merge(rid, on=RC.RACE), "crew_loss_c", "lineage", "crew_time_loss", nm)
    # mean loss instead of median
    p = RC.valid_stops(pits[pits["season"] <= 2023]).merge(ent[RC.KEY + ["lineage"]], on=RC.KEY)
    g = p.groupby(RC.RACE + ["lineage"]).agg(n=("loss_s", "size"), m=("loss_s", "mean"), med=("loss_s", "median")).reset_index()
    g = g[g["n"] >= 2]
    g["mean_c"] = g["m"] - g.groupby(RC.RACE)["m"].transform("median")
    persist(g.merge(rid, on=RC.RACE), "mean_c", "lineage", "crew_time_loss", "mean loss instead of median")
    # driver level
    dd = p.groupby(RC.KEY).agg(n=("loss_s", "size"), m=("loss_s", "median")).reset_index()
    dd["m_c"] = dd["m"] - dd.groupby(RC.RACE)["m"].transform("median")
    persist(dd.merge(rid, on=RC.RACE), "m_c", "driverId", "crew_time_loss", "driver level (median loss of the driver's stops, >= 1)")
    haas = [x for x in cv["lineage"].unique() if "haas" in x]
    persist(cv, "crew_loss_c", "lineage", "crew_time_loss", f"leaving out {haas} (scored rows only)", drop=lambda x: x["lineage"].isin(haas))
    for s in CFG["dev_seasons"]:
        persist(cv, "crew_loss_c", "lineage", "crew_time_loss", f"season {s}", drop=lambda x, s=s: x["season"] != s)
    # stops over 60 s / 300 s counts
    pp = pits[pits["season"] <= 2023]["duration_s"]
    ROWS.append(dict(group="crew_time_loss", check="stop counts: > 300 s / 60-300 s", scope="2014-2023", n_obs=int((pp > 300).sum()), note=f"60-300 s: {int(((pp > 60) & (pp <= 300)).sum())}"))

    out = pd.DataFrame(ROWS)
    out.insert(0, "label", "post hoc")
    out.to_csv(ROOT / "results/plan3/m20_robustness.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
