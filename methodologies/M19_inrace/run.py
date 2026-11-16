"""M19 in-race forecast at H4 (after lap 1), H5 (25% of the distance) and H6 (50%), development block only (PLAN3.md R13).

    python methodologies/M19_inrace/run.py [--quick]

For each horizon: the running order (baseline) and, as the MAIN model, two stages: a frozen pre-race score (latent_111 / gbm_100
walk-forward over all earlier races, generated for the training seasons too) plus five in-race columns (src/pitlane/features/inrace.py,
src/pitlane/models/stage2.py). The joint refit of v1 (pre-race strengths re-estimated together with the in-race coefficients) is kept
as a labelled variant. Paired bootstrap against the running order. Reported separately from pre-race results; no attribution rows.
--quick: 1 lam_x, no bootstrap resamples beyond 200, nothing written.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

from pitlane.config import ROOT, load_yaml, save_yaml
from pitlane.evaluation.harness import save_run, split_races, walk_forward
from pitlane.features import inrace as I
from pitlane.evaluation.metrics import pred_rank
from pitlane.models.registry import build_model
from pitlane.models.stage2 import Stage2GBM, Stage2Latent, walk_scores
from pitlane.plan2 import pipeline2 as Q
from pitlane.plan3 import attribution as A
from pitlane.plan3 import runutil as U

MDIR = Path(__file__).resolve().parent
USED = ["src/pitlane/features/inrace.py", "src/pitlane/models/stage2.py", "src/pitlane/models/latent.py", "src/pitlane/models/gbm.py", "src/pitlane/plan3/attribution.py",
        "src/pitlane/plan3/runutil.py", "src/pitlane/evaluation/harness.py", "methodologies/M19_inrace/run.py",
        "methodologies/M19_inrace/config.yaml"]
VERDICT_METRICS = ["spearman", "mae", "exact", "exact_fin", "exact_1_3", "exact_4_10", "exact_11p", "spearman_fin", "mae_fin"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entries = Q.load_all()
    assert (entries.index == np.arange(len(entries))).all()
    laps = pd.read_parquet(ROOT / cfg["laps_path"])
    pits = pd.read_parquet(ROOT / cfg["pits_path"])
    laps = laps[(laps["source"] == cfg["source"]) & (laps["season"] <= 2023)].reset_index(drop=True)
    pits = pits[(pits["source"] == cfg["source"]) & (pits["season"] <= 2023)].reset_index(drop=True)
    return entries, laps, pits


def k_by_race(entries: pd.DataFrame, dist: dict, avail: dict, frac: float | None) -> dict[int, int]:
    """Horizon lap per race from the scheduled distance `dist`; capped at the last lap that exists (`avail`)."""
    out = {}
    for rid, g in entries.groupby("race_id"):
        key = (int(g["season"].iloc[0]), int(g["round"].iloc[0]))
        if key in dist:
            out[int(rid)] = I.horizon_lap(dist[key], frac, avail[key])
    return out


def race_ids(entries: pd.DataFrame, keys: list) -> list[int]:
    m = entries.groupby("race_id")[["season", "round"]].first()
    return [int(i) for i, r in m.iterrows() if [int(r["season"]), int(r["round"])] in [list(k) for k in keys]]


def running_order_preds(table: pd.DataFrame, entries: pd.DataFrame, race_ids: list[int]) -> pd.DataFrame:
    """Baseline: score = minus running position (unique per race by construction, so no tie-break is needed)."""
    t = table[table["race_id"].isin(race_ids)][["race_id", "driverId", "run_pos"]]
    cols = ["race_id", "season", "round", "driverId", "lineage", "pos", "grid", "classified"]
    m = entries[entries["race_id"].isin(race_ids)][cols].merge(t, on=["race_id", "driverId"], how="left")
    m["score"] = -m.pop("run_pos").astype(float)
    return m


def data_checks(entries: pd.DataFrame, laps: pd.DataFrame, dist: dict, avail: dict, horizons: dict, margin: float) -> pd.DataFrame:
    """Agreement of the derived running order with the stored Jolpica position (2021-22), coverage and gaps."""
    rows = []
    for (s, r), g in entries[entries["season"] <= 2023].groupby(["season", "round"]):
        ll = laps[(laps["season"] == s) & (laps["round"] == r)]
        if ll.empty:
            rows.append(dict(season=s, round=r, check="no_laps", value=1.0))
            continue
        drivers = g["driverId"].tolist()
        rows.append(dict(season=s, round=r, check="entries_without_any_lap", value=float(len(set(drivers) - set(ll["driverId"])))))
        rows.append(dict(season=s, round=r, check="scheduled_distance_laps", value=float(dist[(s, r)])))
        rows.append(dict(season=s, round=r, check="last_lap_in_table", value=float(avail[(s, r)])))
        gaps = sum(int(((d["lap"].to_numpy() != np.arange(1, len(d) + 1)).any()) or d["lap_time_s"].isna().any()) for _, d in ll.groupby("driverId"))
        rows.append(dict(season=s, round=r, check="drivers_with_lap_gaps_or_nan", value=float(gaps)))
        if s in (2021, 2022):
            for h, hc in horizons.items():
                k = I.horizon_lap(dist[(s, r)], hc["frac"], avail[(s, r)])
                st = I.race_state(ll, None, drivers, k, margin)
                rec = ll[ll["lap"] == k].set_index("driverId")["position"]
                mine = st[st["out"] == 0].set_index("driverId")["run_pos"].rank().reindex(rec.index)
                rows.append(dict(season=s, round=r, check=f"agree_stored_position_{h}", value=float((mine == rec.rank()).mean())))
    return pd.DataFrame(rows)


def prep_frame(entries: pd.DataFrame, laps: pd.DataFrame, pits: pd.DataFrame, ks: dict[int, int],
               margin: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = I.build_state_table(entries[entries["season"] <= 2023], laps, pits, ks, margin)
    return I.attach(entries, table), table


def tune_latent(eh: pd.DataFrame, base: dict, cols: list[str], grid: list[float], every: int) -> pd.DataFrame:
    rows = []
    for lx in grid:
        cfg = dict(base, x_cols=cols, lam_x=lx)
        _, per = walk_forward(eh, build_model(cfg, eh), "dev", refit_every=every)
        rows.append(dict(lam_x=lx, spearman=per["spearman"].mean(), mae=per["mae"].mean(), exact=per["exact"].mean()))
    return pd.DataFrame(rows)


def tune_stage2(eh: pd.DataFrame, lat_cfg: dict, cols: list[str], grid: list[float], every: int, cfg: dict) -> pd.DataFrame:
    rows = []
    for lx in grid:
        _, per = walk_forward(eh, make_s2_latent(cfg, lat_cfg, cols, lx), "dev", refit_every=every)
        rows.append(dict(lam_x=lx, spearman=per["spearman"].mean(), mae=per["mae"].mean(), exact=per["exact"].mean()))
    return pd.DataFrame(rows)


def make_s2_latent(cfg: dict, lat_cfg: dict, cols: list[str], lam_x: float) -> Stage2Latent:
    return Stage2Latent("pre_lat", cols, lam_x=lam_x, halflife=float(lat_cfg["halflife"]), fix_c=bool(cfg["stage2"]["fix_c"]),
                        min_race=int(cfg["min_train_race"]))


def make_s2_gbm(cfg: dict, gbm_cfg: dict, cols: list[str]) -> Stage2GBM:
    keep = {k: gbm_cfg[k] for k in ("num_leaves", "n_estimators", "learning_rate", "min_child_samples", "halflife") if k in gbm_cfg}
    return Stage2GBM("pre_gbm", cols, min_race=int(cfg["min_train_race"]), **keep)


def last_refit_coefficients(model, eh: pd.DataFrame, last_race: int) -> dict[str, float]:
    rows = np.flatnonzero((eh["race_id"] == last_race).to_numpy())
    model.fit_predict(eh.iloc[:rows.min()], eh.iloc[rows])
    return {k: float(v) for k, v in model.beta.items()}


def boot_mean(d: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    bm = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def exact_groups(preds: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """Per race: exact hits among cars still running at lap k and among cars already out, each divided by the field size
    (so the two columns add up to the race's exact rate), plus the within-group hit rates."""
    p = preds.merge(state[["race_id", "driverId", "out"]], on=["race_id", "driverId"])
    rows = []
    for rid, g in p.groupby("race_id", sort=True):
        hit = pred_rank(g["score"].to_numpy(float)) == g["pos"].to_numpy(float)
        out = g["out"].to_numpy() == 1
        rows.append(dict(race_id=rid, n=len(g), n_out=int(out.sum()), hits_runners=int(hit[~out].sum()), hits_out=int(hit[out].sum()),
                         share_runners=float(hit[~out].sum() / len(g)), share_out=float(hit[out].sum() / len(g))))
    return pd.DataFrame(rows)


def gain_table(base_tab: pd.DataFrame, tab: pd.DataFrame, ids: list[int] | None, nb: dict) -> pd.DataFrame:
    if ids is not None:
        base_tab, tab = base_tab[base_tab["race_id"].isin(ids)], tab[tab["race_id"].isin(ids)]
    g = A.paired_gain(base_tab, tab, VERDICT_METRICS, n_boot=nb["n_boot"], seed=nb["seed"])
    g["verdict"] = [A.verdict(r) for _, r in g.iterrows()]
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MDIR / "config.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["split"] == "dev", "plan 3 / R13: development block only"
    t0 = time.time()
    stamp = U.code_stamp(USED)
    nb = dict(cfg["bootstrap"])
    if args.quick:
        nb["n_boot"] = 200
    margin = float(load_yaml(ROOT / cfg["ingest_config"])["pits"]["red_flag_margin_s"])
    entries, laps, pits = load_inputs(cfg)
    frozen = load_yaml(ROOT / cfg["frozen_path"])
    lat_cfg, gbm_cfg = dict(frozen[cfg["latent"]["key"]][cfg["latent"]["cell"]]), dict(frozen[cfg["gbm"]["key"]][cfg["gbm"]["cell"]])
    cols = list(cfg["feature_cols"])
    _, counted = split_races(entries, "dev")
    sched = pd.read_csv(ROOT / cfg["scheduled_laps_path"])
    avail = I.race_distance(laps)
    dist = I.scheduled_distance(laps, sched)
    short4 = race_ids(entries, cfg["shortened_dev_races"])
    cut1 = race_ids(entries, cfg["cut_before_start_dev_races"])
    assert len(short4) == 4 and len(cut1) == 1
    dev_keys = set(zip(entries.loc[entries["race_id"].isin(counted), "season"], entries.loc[entries["race_id"].isin(counted), "round"]))
    assert dev_keys <= set(zip(sched["season"], sched["round"])), "every dev race needs a scheduled distance"
    log(f"dev races {len(counted)}; entries {len(entries)}; lap tables {len(avail)} races (source {cfg['source']}); "
        f"scheduled distances from {cfg['scheduled_laps_path']} for {len(sched)} races; red-flag margin {margin} s (computed from stops <= k)")

    # ---------------------------------------------------------------- reproduction with the in-race switch off
    t = time.time()
    rep = {}
    for tag, c in (("latent_111", lat_cfg), ("gbm_100", gbm_cfg)):
        p_new, m_new = walk_forward(entries, build_model(c, entries), "dev", refit_every=1)
        old_pm, old_pr = U.stored_run(cfg["plan1_leaderboard"], tag)
        d, n = U.max_diff(m_new, old_pm)
        ds, ne = U.score_diff(p_new, old_pr)
        rep[tag] = dict(max_abs_diff_per_race_metrics=d, max_abs_diff_scores=ds, races=n, entries=ne)
        log(f"REPRODUCTION {tag} (harness, no in-race columns): max |per-race metric diff| = {d:.3e}, max |score diff| = {ds:.3e} ({n} races, {ne} entries) ({time.time() - t:.0f}s)")

    # ---------------------------------------------------------------- frozen pre-race scores for every earlier race (walk-forward)
    last_race = int(counted[-1])
    t = time.time()
    pre = {"pre_lat": walk_scores(entries, lat_cfg, last_race), "pre_gbm": walk_scores(entries, gbm_cfg, last_race)}
    log(f"frozen pre-race scores generated walk-forward for races 1..{last_race} ({time.time() - t:.0f}s)")
    entries = entries.assign(**pre)
    dev_rows = entries["race_id"].isin(counted).to_numpy()
    for col, tag in (("pre_lat", "latent_111"), ("pre_gbm", "gbm_100")):
        _, old_pr = U.stored_run(cfg["plan1_leaderboard"], tag)
        m = entries.loc[dev_rows, ["race_id", "driverId", col]].merge(old_pr[["race_id", "driverId", "score"]], on=["race_id", "driverId"])
        key = f"walk_scores_{tag}_vs_stored_dev"
        rep[key] = dict(max_abs_diff_scores=float((m[col] - m["score"]).abs().max()), entries=len(m))
        log(f"walk-forward pre-race score {tag} vs stored plan-1 dev scores: max |diff| = {rep[key]['max_abs_diff_scores']:.3e} ({len(m)} entries)")
    off = Stage2Latent("pre_lat", [], fix_c=True, min_race=int(cfg["min_train_race"]))
    p_off, _ = walk_forward(entries, off, "dev", refit_every=1)
    x = p_off.merge(entries[["race_id", "driverId", "pre_lat"]], on=["race_id", "driverId"])
    rep["stage2_latent_no_inrace_columns_fixed_c"] = dict(max_abs_diff_scores_vs_pre_race=float((x["score"] - x["pre_lat"]).abs().max()))
    log(f"stage-2 latent with no in-race columns and c fixed to 1 returns the pre-race score: max |diff| = "
        f"{rep['stage2_latent_no_inrace_columns_fixed_c']['max_abs_diff_scores_vs_pre_race']:.3e}")

    checks = data_checks(entries, laps, dist, avail, cfg["horizons"], margin)
    ag = checks[checks["check"].str.startswith("agree")].groupby("check")["value"].agg(["mean", "min", "count"])
    log("agreement of derived running order with stored Jolpica position (2021-22, share of cars):\n" + ag.round(4).to_string())
    log(f"races without lap table in 2014-2023: {int((checks['check'] == 'no_laps').sum())}; "
        f"entries without any lap in dev races: {int(checks[(checks['check'] == 'entries_without_any_lap') & (checks['season'] >= 2021)]['value'].sum())}; "
        f"drivers with gaps/NaN in dev races: {int(checks[(checks['check'] == 'drivers_with_lap_gaps_or_nan') & (checks['season'] >= 2021)]['value'].sum())}")

    # ---------------------------------------------------------------- per horizon
    res = {}
    tune_all = []
    affected = {}
    rkey = entries.groupby("race_id")[["season", "round"]].first()
    for h, hc in cfg["horizons"].items():
        ks = k_by_race(entries, dist, avail, hc["frac"])
        nominal = {r: I.horizon_lap(dist[(int(rkey.loc[r, "season"]), int(rkey.loc[r, "round"]))], hc["frac"]) for r in ks}
        affected[h] = [(int(rkey.loc[r, "season"]), int(rkey.loc[r, "round"]), nominal[r], ks[r]) for r in counted if ks[r] != nominal[r]]
        eh, table = prep_frame(entries, laps, pits, ks, margin)
        kd = pd.Series({r: ks[r] for r in counted})
        log(f"{h}: k in dev races: min {kd.min()}, median {int(kd.median())}, max {kd.max()}; dev races whose horizon lap was capped "
            f"at the last available lap (season, round, nominal, used): {affected[h]}")
        base_preds = running_order_preds(table, entries, counted)
        base_tab = A.race_table(base_preds)
        grid = cfg["tune_lam_x"][:1] if args.quick else cfg["tune_lam_x"]
        tu2 = tune_stage2(eh, lat_cfg, cols, grid, cfg["tune_refit_every"], cfg).assign(variant="two_stage", horizon=h)
        tu1 = tune_latent(eh, lat_cfg, cols, grid, cfg["tune_refit_every"]).assign(variant="joint", horizon=h)
        tune_all += [tu2, tu1]
        best2 = float(tu2.loc[tu2["spearman"].idxmax(), "lam_x"])
        best1 = float(tu1.loc[tu1["spearman"].idxmax(), "lam_x"])
        log(f"{h}: tuning (dev, refit every {cfg['tune_refit_every']}) picks lam_x={best2} (two-stage), {best1} (joint)\n"
            + pd.concat([tu2, tu1]).round(4).to_string(index=False))
        ljoint = dict(lat_cfg, x_cols=cols, lam_x=best1)
        gjoint = dict(gbm_cfg, extra_cols=cols)
        models = {"latent_s2": make_s2_latent(cfg, lat_cfg, cols, best2), "gbm_s2": make_s2_gbm(cfg, gbm_cfg, cols),
                  "latent_joint": build_model(ljoint, eh), "gbm_joint": build_model(gjoint, eh)}
        preds, tabs = {"running_order": base_preds}, {"running_order": base_tab}
        for nm, mdl in models.items():
            p, _ = walk_forward(eh, mdl, "dev", refit_every=cfg["refit_every"])
            preds[nm] = U.outcome_free(p, cfg["tie_break"])
            tabs[nm] = A.race_table(preds[nm])
        for t_ in tabs.values():
            t_["round"] = t_["race_id"].map(rkey["round"])
        beta = {"latent_s2": last_refit_coefficients(make_s2_latent(cfg, lat_cfg, cols, best2), eh, last_race)}
        bj = build_model(ljoint, eh)
        bj.fit(int((eh["race_id"] < last_race).sum()), t_ref=float(last_race))
        beta["latent_joint"] = {k_: float(v) for k_, v in bj.beta.items() if k_ in cols}
        log(f"{h}: latent beta (last dev refit) two-stage { {k_: round(v, 3) for k_, v in beta['latent_s2'].items()} } ; "
            f"joint { {k_: round(v, 3) for k_, v in beta['latent_joint'].items()} }")
        res[h] = dict(ks=ks, eh=eh, table=table, preds=preds, tabs=tabs, beta=beta, lam_x={"two_stage": best2, "joint": best1},
                      cfg={"latent_s2": dict(kind="stage2_latent", score="pre_lat (latent_111 walk-forward)", lam_x=best2, halflife=lat_cfg["halflife"],
                                             **cfg["stage2"], min_train_race=cfg["min_train_race"]),
                           "gbm_s2": dict(kind="stage2_gbm", score="pre_gbm (gbm_100 walk-forward)", params=gbm_cfg, min_train_race=cfg["min_train_race"]),
                           "latent_joint": ljoint, "gbm_joint": gjoint})
        line = " | ".join(f"{nm} rho {tb['spearman'].mean():.4f} MAE {tb['mae'].mean():.3f} exact {tb['exact'].mean():.4f}" for nm, tb in tabs.items())
        log(f"{h}: {line}  ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- paired comparisons (positive = model better)
    gains, ctx, grp, grp_gain = [], [], [], []
    _, pre_lat = U.stored_run(cfg["plan1_leaderboard"], cfg["latent"]["stored_label"])
    _, pre_gbm = U.stored_run(cfg["plan1_leaderboard"], cfg["gbm"]["stored_label"])
    pre_tabs = {"pre_race_latent_111": A.race_table(U.outcome_free(pre_lat, cfg["tie_break"])),
                "pre_race_gbm_100": A.race_table(U.outcome_free(pre_gbm, cfg["tie_break"]))}
    subsets = {"all_66": None, "excl_4_shortened": [r for r in counted if r not in short4],
               "excl_4_shortened_and_monza": [r for r in counted if r not in short4 + cut1]}
    for season in (2021, 2022, 2023):
        subsets[f"season_{season}"] = [x_ for x_ in counted if int(rkey.loc[x_, "season"]) == season]
    for h, r in res.items():
        for model in ("latent_s2", "gbm_s2", "latent_joint", "gbm_joint"):
            for tag, ids in subsets.items():
                g = gain_table(r["tabs"]["running_order"], r["tabs"][model], ids, nb)
                gains.append(g.assign(horizon=h, model=model, subset=tag, reference="running_order"))
        for fam in ("latent", "gbm"):
            g = gain_table(r["tabs"][f"{fam}_s2"], r["tabs"][f"{fam}_joint"], None, nb)
            gains.append(g.assign(horizon=h, model=f"{fam}_joint_vs_two_stage", subset="all_66", reference=f"{fam}_s2"))
        g = gain_table(r["tabs"]["latent_s2"], r["tabs"]["gbm_s2"], None, nb)
        gains.append(g.assign(horizon=h, model="gbm_vs_latent_two_stage", subset="all_66", reference="latent_s2"))
        for nm, pt in pre_tabs.items():     # context only: a pre-race model against the running order at this horizon
            g = gain_table(r["tabs"]["running_order"], pt, None, nb)
            ctx.append(g.assign(horizon=h, model=nm, subset="all_66", reference="running_order"))
        # exact hits split by cars running / already out at lap k (contribution to the race's exact rate)
        gb = {nm: exact_groups(r["preds"][nm], r["table"][r["table"]["race_id"].isin(counted)]) for nm in r["preds"]}
        for nm, d in gb.items():
            grp.append(dict(horizon=h, predictor=nm, mean_share_cars_out=float((d["n_out"] / d["n"]).mean()),
                            pooled_exact_runners=float(d["hits_runners"].sum() / (d["n"] - d["n_out"]).sum()),
                            pooled_exact_out=float(d["hits_out"].sum() / max(d["n_out"].sum(), 1)),
                            contrib_runners=float(d["share_runners"].mean()), contrib_out=float(d["share_out"].mean())))
            if nm != "running_order":
                for col in ("share_runners", "share_out"):
                    dd = d.set_index("race_id")[col] - gb["running_order"].set_index("race_id")[col]
                    mean, lo, hi = boot_mean(dd.to_numpy(), nb["n_boot"], nb["seed"])
                    grp_gain.append(dict(horizon=h, predictor=nm, component=col, gain_vs_running_order=mean, lo=lo, hi=hi, n=len(dd)))
    gains, ctx = pd.concat(gains, ignore_index=True), pd.concat(ctx, ignore_index=True)
    grp, grp_gain = pd.DataFrame(grp), pd.DataFrame(grp_gain)
    means = pd.concat([pd.DataFrame({m: t[VERDICT_METRICS].mean() for m, t in r["tabs"].items()}).T.assign(horizon=h)
                       for h, r in res.items()] +
                      [pd.DataFrame({m: t[VERDICT_METRICS].mean() for m, t in pre_tabs.items()}).T.assign(horizon="pre-race (context)")])
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print("\nMEANS (outcome-free ties; pre-race rows are a different horizon, context only)\n" + means.round(4).to_string())
    for h in res:
        for model in ("latent_s2", "gbm_s2", "latent_joint", "gbm_joint"):
            g = gains[(gains["horizon"] == h) & (gains["model"] == model) & (gains["subset"] == "all_66")]
            print(f"\nPAIRED GAIN {model} minus running order, {h}, 66 dev races (positive = model better)\n"
                  + g[["metric", "ref_mean", "cand_mean", "gain", "lo", "hi", "n", "verdict"]].round(4).to_string(index=False))
    sel = gains[(gains["metric"] == "spearman") & (gains["reference"] == "running_order") & (gains["subset"] != "all_66")]
    print("\nSENSITIVITY rho gain\n" + sel[["horizon", "model", "subset", "gain", "lo", "hi", "n", "verdict"]].round(4).to_string(index=False))
    sel = gains[gains["metric"].isin(["mae", "exact"]) & (gains["reference"] == "running_order") & (gains["subset"] == "excl_4_shortened")]
    print("\nSENSITIVITY MAE / exact gain, without the 4 shortened races\n" + sel[["horizon", "model", "metric", "gain", "lo", "hi", "n", "verdict"]].round(4).to_string(index=False))
    print("\nTWO-STAGE vs JOINT, GBM vs LATENT (rho)\n" + gains[(gains["reference"] != "running_order") & (gains["metric"] == "spearman")][["horizon", "model", "gain", "lo", "hi", "verdict"]].round(4).to_string(index=False))
    print("\nCONTEXT pre-race model minus running order, rho\n" + ctx[ctx["metric"] == "spearman"][["horizon", "model", "gain", "lo", "hi"]].round(4).to_string(index=False))
    print("\nEXACT HITS BY GROUP (runners = still running at lap k)\n" + grp.round(4).to_string(index=False))
    print("\nEXACT-RATE CONTRIBUTION GAIN vs running order (contribution = hits / field size)\n" + grp_gain.round(4).to_string(index=False))
    if args.quick:
        log("quick run: nothing written")
        return

    # ---------------------------------------------------------------- write
    tmp_lb = MDIR / "results" / "_lb_tmp.csv"
    tmp_lb.parent.mkdir(parents=True, exist_ok=True)
    tmp_lb.unlink(missing_ok=True)
    pfx = cfg["label_prefix"]
    cells = {"running_order": "000", "latent_s2": "111", "gbm_s2": "100", "latent_joint": "111", "gbm_joint": "100"}
    for h, r in res.items():
        for model, cell in cells.items():
            label = f"{pfx}_{h.lower()}_{model}"
            role = "baseline" if model == "running_order" else ("MAIN (two-stage, frozen pre-race score)" if model.endswith("_s2") else "variant (joint refit, as in v1)")
            conf = dict(horizon=h, frac=cfg["horizons"][h]["frac"], model=model, source=cfg["source"], feature_cols=cols, role=role,
                        k_dev_races={int(k_): int(v) for k_, v in r["ks"].items() if k_ in counted},
                        scheduled_laps_path=cfg["scheduled_laps_path"], red_flag_margin_s=margin, min_train_race=cfg["min_train_race"],
                        **({} if model == "running_order" else dict(model_config=r["cfg"][model])),
                        note="in-race forecast, reported separately from pre-race results; v2 supersedes the m19_* rows of the first run")
            tab = r["tabs"][model]
            rid = save_run(MDIR, label, conf, r["preds"][model], tab, "dev", "M19", cell, leaderboard=tmp_lb)
            cc = load_yaml(MDIR / "results" / rid / "config.yaml")
            cc["git_commit"] = stamp
            save_yaml(cc, MDIR / "results" / rid / "config.yaml")
            tab.to_csv(MDIR / "results" / rid / "outcome_free_table.csv", index=False)
            r["table"][r["table"]["race_id"].isin(counted)].to_csv(MDIR / "results" / rid / "state_table_dev.csv", index=False)
    new_rows = pd.read_csv(tmp_lb)
    tmp_lb.unlink()
    res_dir = MDIR / "results"
    gains.to_csv(res_dir / "paired_gains_dev.csv", index=False)
    ctx.to_csv(res_dir / "context_pre_race_vs_running_order.csv", index=False)
    means.to_csv(res_dir / "means_dev.csv")
    pd.concat(tune_all).to_csv(res_dir / "tuning_lam_x.csv", index=False)
    checks.to_csv(res_dir / "data_checks.csv", index=False)
    grp.to_csv(res_dir / "exact_hits_by_group.csv", index=False)
    grp_gain.to_csv(res_dir / "exact_contribution_gain_by_group.csv", index=False)
    pd.DataFrame([dict(horizon=h, season=s_, round=r_, nominal_k=a, used_k=b) for h, v in affected.items() for (s_, r_, a, b) in v],
                 columns=["horizon", "season", "round", "nominal_k", "used_k"]).to_csv(res_dir / "capped_horizons.csv", index=False)
    cols_v = ["horizon", "model", "metric", "ref_mean", "cand_mean", "gain", "lo", "hi", "n", "verdict"]
    allv = gains[(gains["subset"] == "all_66") & (gains["reference"] == "running_order") & gains["metric"].isin(VERDICT_METRICS)]
    allv[allv["model"].isin(["latent_s2", "gbm_s2"])][cols_v].to_csv(res_dir / "verdicts.csv", index=False)
    allv[allv["model"].isin(["latent_joint", "gbm_joint"])][cols_v].to_csv(res_dir / "verdicts_joint_variant.csv", index=False)
    (res_dir / "reproduction_check.json").write_text(json.dumps(dict(
        reproduction=rep, lam_x_chosen={h: r["lam_x"] for h, r in res.items()},
        latent_beta_last_dev_refit={h: r["beta"] for h, r in res.items()}, capped_horizons=affected), indent=2), encoding="utf-8")
    U.append_leaderboard(new_rows, ROOT / cfg["leaderboard"])
    log(f"done in {time.time() - t0:.0f}s; code stamp {stamp}; {len(new_rows)} leaderboard rows appended")


if __name__ == "__main__":
    main()
