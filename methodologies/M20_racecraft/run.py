"""M20 phase 1 - persistence of the four race-craft features (PLAN3.md section 4 M20, R8). Dev decision (2021-2023).

    python methodologies/M20_racecraft/run.py [--config methodologies/M20_racecraft/config.yaml]

Definitions: methodologies/M20_racecraft/README.md. No model is run; no leaderboard row. Reads Jolpica laps and
pits 2014-2023 only (nothing later is read).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from pitlane.config import ROOT, load_yaml, save_yaml
from pitlane.features import pace as P
from pitlane.features import racecraft as RC
from pitlane.plan3 import pace_gate as G
from pitlane.plan3.runutil import code_stamp

USED = ["src/pitlane/features/racecraft.py", "src/pitlane/features/pace.py", "src/pitlane/plan3/pace_gate.py",
        "methodologies/M20_racecraft/run.py", "methodologies/M20_racecraft/config.yaml"]
# feature -> (value column, unit column, level, teammate-relative applicable)
FEATURES = {
    "consistency": ("cons_c", "driverId", "driver", True),
    "tyre_degradation": ("deg_c", "driverId", "driver", True),
    "lap1_gain": ("lap1_c", "driverId", "driver", True),
    "crew_time_loss": ("crew_loss_c", "lineage", "team", False),
    "crew_slow_stop_rate": ("crew_slow_c", "lineage", "team", False),
}


def load(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    last = max(cfg["dev_seasons"])
    laps = pd.read_parquet(ROOT / "data/interim/plan3/laps.parquet")
    pits = pd.read_parquet(ROOT / "data/interim/plan3/pits.parquet")
    laps = laps[(laps["source"] == cfg["lap_source"]) & (laps["season"] <= last)]
    pits = pits[(pits["source"] == cfg["lap_source"]) & (pits["season"] <= last)]
    ent = pd.read_parquet(ROOT / "data/processed/entries2.parquet")
    ent = ent[ent["season"] <= last].reset_index(drop=True)
    return laps, pits, ent


def race_values(laps, pits, ent, cfg) -> dict[str, pd.DataFrame]:
    m = P.mark_laps(laps, pits, cfg["thresh"])
    dv = RC.driver_lap_values(m, cfg["min_stint_laps"], cfg["min_driver_laps"])
    pos1 = RC.lap1_positions(laps)
    l1 = RC.lap1_values(pos1, ent, cfg["min_lap1_drivers"])
    cv = RC.crew_values(pits, ent, cfg["red_flag_margin_s"], cfg["slow_stop_s"], cfg["min_crew_stops"], cfg["max_stop_s"])
    rid = ent[["season", "round", "race_id"]].drop_duplicates()
    team = ent[RC.KEY + ["lineage"]]
    drv = dv.merge(l1[RC.KEY + ["gain", "lap1_c", "p1", "rank_time", "p1_source"]], on=RC.KEY, how="outer")
    drv = drv.merge(rid, on=RC.RACE, how="left").merge(team, on=RC.KEY, how="left")
    crew = cv.merge(rid, on=RC.RACE, how="left")
    return dict(driver=drv, team=crew, pos1=pos1)


def winsorise(s: pd.Series, q: list[float]) -> pd.Series:
    return s.clip(s.quantile(q[0]), s.quantile(q[1]))


def variants(tab: pd.DataFrame, col: str, tm: bool, team: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Declared value plus the sensitivity versions: name -> frame with a `y` column."""
    out = {"declared": tab.assign(y=tab[col])}
    out["winsorised_1_99 (in-sample bounds)"] = tab.assign(y=winsorise(tab[col], cfg["winsor"]))
    if tm:
        out["teammate_relative"] = tab.assign(y=RC.teammate_relative(tab, col, team))
    return out


def persistence_rows(feature: str, name: str, tab: pd.DataFrame, unit: str, level: str, cfg: dict) -> list[dict]:
    b = cfg["bootstrap"]
    tab = tab.dropna(subset=["y"]).copy()
    hist = RC.history_values(tab.rename(columns={"y": "yv"}), unit, "yv", tuple(cfg["halflives"]))
    d = tab.merge(hist, on=["race_id", unit], how="left")
    d = d[d["yv_n"] >= cfg["min_prior_obs"]]
    scopes = [("dev", d["season"].isin(cfg["dev_seasons"])), ("burn_in_2014_2020", d["season"].isin(cfg["burn_in_seasons"]))]
    scopes += [(str(s), d["season"] == s) for s in cfg["dev_seasons"] + cfg["burn_in_seasons"]]
    rows = []
    for h in cfg["halflives"]:
        for scope, mask in scopes:
            if scope.isdigit() and int(scope) < 2021 and h != cfg["halflife_primary"]:
                continue
            sub = d[mask]
            r = G.pooled_corr(sub, f"yv_h{h}", "y", n_boot=b["n_boot"], seed=b["seed"])
            sp = float(stats.spearmanr(sub[f"yv_h{h}"], sub["y"])[0]) if len(sub) > 5 else np.nan
            burn = scope.startswith("burn") or (scope.isdigit() and int(scope) < 2021)
            rows.append(dict(feature=feature, level=level, variant=name, halflife=h, scope=scope, spearman=sp,
                             block="burn-in (not a decision block)" if burn else "dev", **r))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "methodologies/M20_racecraft/config.yaml"))
    cfg = load_yaml(ap.parse_args().config)
    assert cfg["split"] == "dev" and max(cfg["dev_seasons"]) == 2023, "R8 phase 1: development block only"
    laps, pits, ent = load(cfg)
    print(f"laps {len(laps)} (seasons {laps['season'].min()}-{laps['season'].max()}), pits {len(pits)}")
    vals = race_values(laps, pits, ent, cfg)
    team = ent[RC.KEY + ["lineage"]]

    rows = []
    for feat, (col, unit, level, tm) in FEATURES.items():
        tab = vals["driver"] if level == "driver" else vals["team"]
        tab = tab.dropna(subset=[col])
        for name, v in variants(tab, col, tm, team, cfg).items():
            rows += persistence_rows(feat, name, v, unit, level, cfg)
    pers = pd.DataFrame(rows)
    out_csv = ROOT / cfg["csv_out"]
    pers.to_csv(out_csv, index=False)

    hp = cfg["halflife_primary"]
    dec = pers[(pers["variant"] == "declared") & (pers["halflife"] == hp) & (pers["scope"] == "dev")].set_index("feature")
    decision = {f: dict(level=r["level"], persistence=r["r"], lo=r["lo"], hi=r["hi"], n_obs=int(r["n_obs"]), n_races=int(r["n_races"]),
                        persistent=bool(r["r"] >= cfg["persistence_min"]),
                        ci_straddles_threshold=bool(r["lo"] < cfg["persistence_min"] <= r["hi"]))
                for f, r in dec.iterrows()}
    show = pers[(pers["halflife"] == hp)].pivot_table(index=["feature", "variant"], columns="scope", values="r")
    print("\nPERSISTENCE r, half-life 50, by scope\n", show.round(3).to_string())
    print("\nDECISION (declared value, dev, half-life 50, threshold 0.2)\n", json.dumps(decision, indent=2))
    hl = pers[(pers["variant"] == "declared") & (pers["scope"] == "dev")].pivot_table(index="feature", columns="halflife", values="r")
    print("\nacross half-lives (declared, dev)\n", hl.round(3).to_string())

    drv, pos1 = vals["driver"], vals["pos1"]
    chk = pos1[(pos1["p1_source"] == "stored") & pos1["season"].between(2021, 2022)]
    agree = {int(s): float((g["p1"] == g["rank_time"]).mean()) for s, g in chk.groupby("season")}
    tabs = {"driver": vals["driver"], "team": vals["team"]}
    dev = lambda t: t[t["season"].isin(cfg["dev_seasons"])]
    counts = {f: int(tabs[lv][c].notna().sum()) for f, (c, u, lv, t) in FEATURES.items()}
    cov = {f: int(dev(tabs[lv].dropna(subset=[c]))[["season", "round"]].drop_duplicates().shape[0]) for f, (c, u, lv, t) in FEATURES.items()}
    kept = RC.valid_stops(pits, cfg["red_flag_margin_s"], cfg["max_stop_s"])
    n_all = int(pits["duration_s"].notna().sum())
    checks = dict(lap1_stored_vs_time_rank_agreement_2021_22=agree, values_per_feature_2014_2023=counts,
                  dev_races_total=int(dev(ent)["race_id"].nunique()), dev_races_with_values=cov,
                  stops_with_duration=n_all, stops_dropped_red_flag_or_repair=n_all - len(kept))
    print("\nDATA CHECKS\n", json.dumps(checks, indent=2, default=float))

    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M')}_m20_persistence_dev"
    out = ROOT / cfg["out_dir"] / run_id
    out.mkdir(parents=True, exist_ok=True)
    gid = code_stamp(USED)
    save_yaml({**cfg, "git_commit": gid, "run_id": run_id}, out / "config.yaml")
    (out / "metrics.json").write_text(json.dumps(dict(run_id=run_id, decision=decision, checks=checks), indent=2, default=float), encoding="utf-8")
    pers.to_csv(out / "per_race_metrics.csv", index=False)        # persistence table (no per-race metric exists in phase 1)
    drv.to_parquet(out / "predictions.parquet", index=False)      # driver race values (not predictions)
    vals["team"].to_parquet(out / "crew_values.parquet", index=False)
    (ROOT / cfg["summary_json"]).write_text(json.dumps(dict(run_id=run_id, code=gid, decision=decision, checks=checks), indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}\n{out_csv}\ncode stamp {gid}")


if __name__ == "__main__":
    main()
