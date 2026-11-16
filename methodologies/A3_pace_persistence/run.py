"""A3 — race-pace persistence and the M09 gate (PLAN3.md section 4, R4). Development block only (2021-2023).

    python methodologies/A3_pace_persistence/run.py [--config methodologies/A3_pace_persistence/config.yaml]

Reads laps/pits from data/interim/plan3 (Jolpica, seasons <= 2023 only), the plan-1 `latent_111` dev predictions and
the plan-2 entries table. Writes one run folder (config.yaml, metrics.json, per_race_metrics.csv, predictions.parquet),
result tables under results/plan3/a3_*.csv, a summary json and one leaderboard row (an ORACLE row, not a model).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pitlane.config import ROOT, load_yaml, save_yaml
from pitlane.features import pace as P
from pitlane.plan3 import pace_gate as G
from pitlane.plan3.runutil import append_leaderboard, code_stamp

USED = ["src/pitlane/features/pace.py", "src/pitlane/plan3/pace_gate.py", "src/pitlane/plan3/runutil.py",
        "methodologies/A3_pace_persistence/run.py", "methodologies/A3_pace_persistence/config.yaml"]
DEFS = {"med": "med_gap_pct", "corr": "corr_gap_pct"}


# ------------------------------------------------------------------------------------ inputs
def load_inputs(cfg: dict) -> dict:
    last = max(cfg["dev_seasons"])
    laps = pd.read_parquet(ROOT / "data/interim/plan3/laps.parquet")
    pits = pd.read_parquet(ROOT / "data/interim/plan3/pits.parquet")
    laps = laps[(laps["source"] == cfg["lap_source"]) & (laps["season"] <= last)]
    pits = pits[(pits["source"] == cfg["lap_source"]) & (pits["season"] <= last)]
    ent = pd.read_parquet(ROOT / "data/processed/entries2.parquet")
    ent = ent[ent["season"] <= last].sort_values(["race_id", "pos"]).reset_index(drop=True)
    lb = pd.read_csv(ROOT / cfg["model_leaderboard"])
    row = lb[(lb["label"] == cfg["model_label"]) & (lb["split"] == "dev")].iloc[-1]
    run = ROOT / row["run_dir"]
    preds = pd.read_parquet(run / "predictions.parquet")
    per = pd.read_csv(run / "per_race_metrics.csv").set_index("race_id")
    assert preds["season"].between(min(cfg["dev_seasons"]), last).all()
    return dict(laps=laps, pits=pits, ent=ent, preds=preds, per=per, model_run=row["run_id"])


# ------------------------------------------------------------------------------------ oracle gap
def oracle_table(inp: dict, pt: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """One row per dev race: plan-2-convention oracle, matched oracles (median, corrected, threshold sensitivity)."""
    preds, per, laps = inp["preds"], inp["per"], inp["laps"]
    by_race = {k: g for k, g in laps.groupby(["season", "round"])}
    rows = []
    for rid, g in preds.groupby("race_id"):
        s, r = int(g["season"].iloc[0]), int(g["round"].iloc[0])
        fin = g[g["classified"]]
        row = dict(race_id=rid, season=s, round=r, n_fin=len(fin), model_rho_fin_all=float(per.loc[rid, "spearman_fin"]))
        row["oracle_p2"] = G.plan2_rho(G.plan2_pace(by_race[(s, r)], fin["driverId"].tolist()), fin) if (s, r) in by_race else np.nan
        for name, tab in pt.items():
            col = DEFS["corr"] if name == "corr" else DEFS["med"]
            v = g[["driverId"]].merge(tab[(tab["season"] == s) & (tab["round"] == r)][["driverId", col]], on="driverId", how="left")
            val = v[col].to_numpy(float)
            res = G.matched_rhos(g, val, ~np.isnan(val))
            row[f"n_{name}"], row[f"oracle_{name}"], row[f"model_{name}"] = res["n"], res["oracle"], res["model"]
        rows.append(row)
    d = pd.DataFrame(rows)
    for name in pt:
        d[f"gap_{name}"] = d[f"oracle_{name}"] - d[f"model_{name}"]
    d["gap_p2"] = d["oracle_p2"] - d["model_rho_fin_all"]
    return d


def gap_summary(d: pd.DataFrame, cols: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for c in cols:
        m, lo, hi = G.boot_mean(d[c].to_numpy(float), n_boot, seed)
        rows.append(dict(quantity=c, mean=m, lo=lo, hi=hi, n_races=int(d[c].notna().sum())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ persistence
def persistence_tables(rq: pd.DataFrame, hist: pd.DataFrame, cfg: dict, name: str) -> pd.DataFrame:
    """Pooled and by-season persistence correlations for one pace definition."""
    m = rq.merge(hist, on=["race_id", "driverId"], how="left")
    m = m[m["season"].isin(cfg["dev_seasons"])]
    car = m.drop_duplicates(["race_id", "lineage"])
    nmin, b = cfg["min_prior_obs"], cfg["bootstrap"]
    rows = []
    specs = [("car", car, "car_rq", "con_rq_ew{h}", "con_n", cfg["halflives"]),
             ("driver", m, "rq_c", "drv_rq_ew{h}", "drv_n", cfg["halflives"]),
             ("driver_vs_teammate", m, "rq_tm", "drvtm_rq_ew{h}", "drvtm_n", cfg["halflives"]),
             ("car_same_season_only", car, "car_rq", "con_rq_ssn", "con_n_ssn", [cfg["halflife_primary"]])]
    specs = [s for s in specs if s[3].format(h=0) in m.columns or s[3].format(h=cfg["halflives"][0]) in m.columns]
    for level, df, y, xf, nc, hls in specs:
        for h in hls:
            x = xf.format(h=h)
            d = df[df[nc] >= (3 if level == "car_same_season_only" else nmin)]
            for scope, sub in [("dev", d)] + [(str(s), d[d["season"] == s]) for s in cfg["dev_seasons"]]:
                r = G.pooled_corr(sub, x, y, n_boot=b["n_boot"], seed=b["seed"])
                rows.append(dict(definition=name, level=level, halflife=h, scope=scope, **r))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ split of the gap
def ols_slopes(rq: pd.DataFrame, hist: pd.DataFrame, cfg: dict) -> tuple[float, float]:
    """Through-origin slopes of the car offset on `con_rq` and of the teammate-relative driver offset on `drvtm_rq`
    (dev rows with >= min_prior_obs earlier observations). IN-SAMPLE: used only for the optimistic diagnostic."""
    h, nmin = cfg["halflife_primary"], cfg["min_prior_obs"]
    m = rq.merge(hist, on=["race_id", "driverId"], how="left")
    m = m[m["season"].isin(cfg["dev_seasons"])]
    car = m.drop_duplicates(["race_id", "lineage"])
    car = car[(car["con_n"] >= nmin)].dropna(subset=["car_rq"])
    drv = m[(m["drvtm_n"] >= nmin)].dropna(subset=["rq_tm"])
    bc = float((car["car_rq"] * car[f"con_rq_ew{h}"]).sum() / (car[f"con_rq_ew{h}"] ** 2).sum())
    bd = float((drv["rq_tm"] * drv[f"drvtm_rq_ew{h}"]).sum() / (drv[f"drvtm_rq_ew{h}"] ** 2).sum())
    return bc, bd


def split_table(inp: dict, rq: pd.DataFrame, hist: pd.DataFrame, h: int, beta_c: float = 1.0, beta_d: float = 1.0) -> pd.DataFrame:
    """Per race: rho_fin of the model and of the pace-unit predictors q, q+bc*Pc, q+bc*Pc+bd*Pd, q+C+bd*Pd, q+C+D (oracle).
    beta = 1 is the pre-declared version; beta = in-sample OLS slope is an optimistic diagnostic."""
    m = inp["preds"].merge(rq[["race_id", "driverId", "q_gap_pct", "race_gap_pct", "rq_c", "car_rq"]], on=["race_id", "driverId"], how="left")
    m = m.merge(hist[["race_id", "driverId", f"con_rq_ew{h}", f"drvtm_rq_ew{h}"]], on=["race_id", "driverId"], how="left")
    rows = []
    for rid, g in m.groupby("race_id"):
        q = g["q_gap_pct"].to_numpy(float)
        pc, pd_ = beta_c * g[f"con_rq_ew{h}"].to_numpy(float), beta_d * g[f"drvtm_rq_ew{h}"].to_numpy(float)
        c = g["car_rq"].to_numpy(float)
        d = g["rq_c"].to_numpy(float) - c
        ok = ~np.isnan(q + pc + pd_ + c + d)
        steps = {"q": q, "q_pc": q + pc, "q_pc_pd": q + pc + pd_, "q_c_pd": q + c + pd_, "q_c_d": q + c + d}
        row = dict(race_id=rid, season=int(g["season"].iloc[0]))
        for k, v in steps.items():
            row[f"rho_{k}"] = G.matched_rhos(g, v, ok)["oracle"]
        row["rho_model"] = G.matched_rhos(g, q, ok)["model"]
        row["n"] = G.matched_rhos(g, q, ok)["n"]
        rows.append(row)
    d = pd.DataFrame(rows)
    d["t_model_vs_q"] = d["rho_q"] - d["rho_model"]
    d["t_car_persistent"] = d["rho_q_pc"] - d["rho_q"]
    d["t_driver_persistent"] = d["rho_q_pc_pd"] - d["rho_q_pc"]
    d["t_car_race_specific"] = d["rho_q_c_pd"] - d["rho_q_pc_pd"]
    d["t_driver_race_specific"] = d["rho_q_c_d"] - d["rho_q_c_pd"]
    d["t_total_gap"] = d["rho_q_c_d"] - d["rho_model"]
    return d


def sanity_table(pace: pd.DataFrame, inp: dict, cfg: dict) -> pd.DataFrame:
    """Same history machinery applied to the field-relative race gap and the qualifying gap themselves (not offsets):
    if the pipeline is sound these are strongly persistent, so a weak rq persistence is a property of rq."""
    rows = []
    for what in ("race_gap_pct", "q_gap_pct"):
        t = P.rq_table(pace, inp["ent"], DEFS["med"])
        t["rq_c"] = t[what] - t.groupby("race_id")[what].transform("median")
        g = t.groupby(["race_id", "lineage"])["rq_c"]
        t["car_rq"], t["rq_tm"] = g.transform("mean"), np.nan
        h = P.rq_history_features(t, tuple(cfg["halflives"]), None)
        p = persistence_tables(t, h, cfg, what)
        rows.append(p[(p["level"].isin(["car", "driver"])) & (p["halflife"] == cfg["halflife_primary"]) & (p["scope"] == "dev")])
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "methodologies/A3_pace_persistence/config.yaml"))
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["split"] == "dev" and max(cfg["dev_seasons"]) == 2023, "plan 3 / R4: development block only"
    b = cfg["bootstrap"]
    inp = load_inputs(cfg)
    print(f"model run {inp['model_run']}; laps {len(inp['laps'])}; pits {len(inp['pits'])}; seasons <= {inp['laps']['season'].max()}")

    pace = P.build_pace_table(inp["laps"], inp["pits"], cfg["thresh"], cfg["min_clean"])
    sens = {f"med_{t}": P.build_pace_table(inp["laps"], inp["pits"], t, cfg["min_clean"]) for t in cfg["thresh_sensitivity"]}
    pt = {"med": pace, "corr": pace, **sens}
    od = oracle_table(inp, {k: v for k, v in pt.items()}, cfg)

    # reproduction of the plan-2 oracle on the 22 races of 2023
    old = pd.read_csv(ROOT / cfg["plan2_ceiling_per_race"]).dropna(subset=["pace_oracle_rho_fin"]).set_index("race_id")
    new = od.set_index("race_id").loc[old.index, "oracle_p2"]
    repro = float((new - old["pace_oracle_rho_fin"]).abs().max())
    print(f"plan-2 oracle reproduction on {len(old)} races of 2023: max |diff| = {repro:.2e}")

    gaps = gap_summary(od, ["model_rho_fin_all", "oracle_p2", "gap_p2", "oracle_med", "model_med", "gap_med",
                            "oracle_corr", "model_corr", "gap_corr"] + [f"gap_med_{t}" for t in cfg["thresh_sensitivity"]],
                       b["n_boot"], b["seed"])
    by_season = od.groupby("season")[["gap_p2", "gap_med", "gap_corr"]].mean().reset_index()
    print("\nORACLE GAP on dev (finishers; mean over races, 95% CI over races)\n", gaps.round(4).to_string(index=False))
    print("\nby season\n", by_season.round(4).to_string(index=False))
    sub23 = od[od["season"] == 2023]
    print(f"\n2023 only ({len(sub23)} races): plan-2 conv gap {sub23['gap_p2'].mean():.4f} (plan 2: 0.0733 on 22 races)")

    # persistence, both definitions
    rqs, hists, pers = {}, {}, []
    for name, col in DEFS.items():
        rqs[name] = P.rq_table(pace, inp["ent"], col)
        hists[name] = P.rq_history_features(rqs[name], tuple(cfg["halflives"]), cfg["halflife_primary"])
        pers.append(persistence_tables(rqs[name], hists[name], cfg, name))
    pers = pd.concat(pers, ignore_index=True)
    show = pers[(pers["halflife"] == cfg["halflife_primary"]) & (pers["level"] != "car_same_season_only")]
    print("\nPERSISTENCE (half-life 50)\n", show.round(3).to_string(index=False))
    print("\nPERSISTENCE across half-lives, scope dev\n",
          pers[(pers["scope"] == "dev")].pivot_table(index=["definition", "level"], columns="halflife", values="r").round(3).to_string())

    # split of the gap
    tcols = ["t_model_vs_q", "t_car_persistent", "t_driver_persistent", "t_car_race_specific", "t_driver_race_specific", "t_total_gap"]
    scols = ["rho_model", "rho_q", "rho_q_pc", "rho_q_pc_pd", "rho_q_c_pd", "rho_q_c_d"] + tcols
    slopes = {n: ols_slopes(rqs[n], hists[n], cfg) for n in DEFS}
    print("\nin-sample OLS slopes (car, driver-vs-teammate):", {n: tuple(round(x, 3) for x in v) for n, v in slopes.items()})
    spl, split_sum = {}, []
    for scaling in ("beta=1 (pre-declared)", "beta=OLS in-sample (optimistic diagnostic)"):
        for n in DEFS:
            bc, bd = (1.0, 1.0) if scaling.startswith("beta=1") else slopes[n]
            t = split_table(inp, rqs[n], hists[n], cfg["halflife_primary"], bc, bd)
            if scaling.startswith("beta=1"):
                spl[n] = t
            split_sum.append(gap_summary(t, scols, b["n_boot"], b["seed"]).assign(definition=n, scaling=scaling, beta_car=bc, beta_driver=bd))
    split_sum = pd.concat(split_sum, ignore_index=True)
    print("\nSPLIT OF THE GAP (same finishers; terms add to total)\n", split_sum.round(4).to_string(index=False))

    sanity = sanity_table(pace, inp, cfg)
    print("\nSANITY: persistence of the race pace gap and of the qualifying gap themselves (same machinery)\n", sanity.round(3).to_string(index=False))

    # gate, mechanical
    g_cfg = cfg["gate"]
    car = pers[(pers["level"] == "car") & (pers["halflife"] == cfg["halflife_primary"]) & (pers["scope"] == "dev")].set_index("definition")["r"]
    gate = {}
    for name in DEFS:
        gap = float(od[f"gap_{name}"].mean())
        gate[name] = dict(oracle_gap=gap, oracle_gap_pass=bool(gap >= g_cfg["oracle_gap_min"]),
                          car_persistence=float(car[name]), car_persistence_pass=bool(car[name] >= g_cfg["car_persistence_min"]))
        gate[name]["gate_pass"] = bool(gate[name]["oracle_gap_pass"] and gate[name]["car_persistence_pass"])
    gate["plan2_convention_oracle_gap"] = float(od["gap_p2"].mean())
    gate["primary_definition"] = cfg["primary_definition"]
    gate["gate_pass_primary"] = gate[cfg["primary_definition"]]["gate_pass"]
    gate["definition_choice_swings_gate"] = bool(gate["med"]["gate_pass"] != gate["corr"]["gate_pass"])
    print("\nGATE\n", json.dumps(gate, indent=2))

    save(cfg, inp, pace, od, gaps, by_season, pers, spl, split_sum, gate, repro, rqs, hists, sanity)


def save(cfg, inp, pace, od, gaps, by_season, pers, spl, split_sum, gate, repro, rqs, hists, sanity) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_id = f"{stamp}_a3_pace_oracle_dev"
    out = ROOT / cfg["out_dir"] / run_id
    out.mkdir(parents=True, exist_ok=True)
    gid = code_stamp(USED)
    save_yaml({**cfg, "git_commit": gid, "run_id": run_id, "split": "dev"}, out / "config.yaml")
    race = od.merge(spl["med"].add_prefix("med_split_").rename(columns={"med_split_race_id": "race_id"}).drop(columns="med_split_season"),
                    on="race_id", how="left")
    race = race.merge(spl["corr"].add_prefix("corr_split_").rename(columns={"corr_split_race_id": "race_id"}).drop(columns="corr_split_season"),
                      on="race_id", how="left")
    race.to_csv(out / "per_race_metrics.csv", index=False)
    cols = ["race_id", "driverId", "lineage", "q_gap_pct", "race_gap_pct", "rq", "rq_c", "car_rq", "rq_tm"]
    ent = rqs["med"][cols].merge(hists["med"], on=["race_id", "driverId"], how="left")
    corr = rqs["corr"][["race_id", "driverId", "race_gap_pct", "rq", "rq_c", "car_rq", "rq_tm"]].add_prefix("corr_").rename(
        columns={"corr_race_id": "race_id", "corr_driverId": "driverId"})
    ent = ent.merge(corr, on=["race_id", "driverId"]).merge(hists["corr"].add_prefix("corr_").rename(
        columns={"corr_race_id": "race_id", "corr_driverId": "driverId"}), on=["race_id", "driverId"])
    ent = ent.merge(inp["preds"][["race_id", "driverId", "season", "round", "pos", "classified", "score"]], on=["race_id", "driverId"], how="left")
    ent.to_parquet(out / "predictions.parquet", index=False)
    pace_path = ROOT / "results/plan3/a3_pace_table_2014_2023.parquet"
    pace.to_parquet(pace_path, index=False)
    metrics = dict(n_races=int(len(od)), gate=gate, plan2_oracle_reproduction_max_abs_diff=repro,
                   oracle_gap=gaps.set_index("quantity").to_dict("index"),
                   corrected_fit_fallback_share=float(pace.drop_duplicates(["season", "round"])["corr_fallback"].mean()),
                   corrected_fit_fallback_by_reason=pace.drop_duplicates(["season", "round"])["corr_reason"].replace("", "fitted").value_counts().to_dict())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    res = ROOT / "results" / "plan3"
    gaps.to_csv(res / "a3_oracle_gap.csv", index=False)
    by_season.to_csv(res / "a3_oracle_gap_by_season.csv", index=False)
    pers.to_csv(res / "a3_persistence.csv", index=False)
    split_sum.to_csv(res / "a3_gap_split.csv", index=False)
    sanity.to_csv(res / "a3_sanity_pace_level.csv", index=False)
    (res / "a3_summary.json").write_text(json.dumps(dict(run_id=run_id, **metrics), indent=2, default=float), encoding="utf-8")
    row = {"method": "A3", "cell": "oracle", "label": "a3_pace_oracle", "split": "dev", "run_id": run_id,
           "run_dir": str(out.relative_to(ROOT)), "spearman_fin": float(od["oracle_med"].mean()), "n_races": int(len(od))}
    append_leaderboard(pd.DataFrame([row]), ROOT / cfg["leaderboard"])
    print(f"\nwrote {out}\nleaderboard row appended (oracle, not a model); code stamp {gid}")


if __name__ == "__main__":
    main()
