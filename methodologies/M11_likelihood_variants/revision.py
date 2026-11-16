"""M11 revision after the referee's review (R5 revision). Development block (2021-23) only; nothing later is scored.

    python methodologies/M11_likelihood_variants/revision.py --stage post     # tasks A, B, C (from the stored run folders)
    python methodologies/M11_likelihood_variants/revision.py --stage mech     # task D (refits at the start of 2022 and 2023)
    python methodologies/M11_likelihood_variants/revision.py --stage tune     # task E (grid per variant, walk-forward)
    python methodologies/M11_likelihood_variants/revision.py --stage inter    # task F (finishers-only x M07 through the S3 simulator)
    python methodologies/M11_likelihood_variants/revision.py --stage final    # attribution.csv, leaderboard, method README row

The four existing run folders (`20261116-2036_m11_*_dev`), `paired_gains_dev.csv`, `descriptive_stage_shares.csv` and
`reproduction_check.json` are only read. Every output of this script is a NEW file or a NEW run folder.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
import zlib
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from pitlane import pipeline as P
from pitlane.config import ROOT, load_yaml, save_yaml
from pitlane.evaluation.harness import _git_hash, save_run, walk_forward
from pitlane.evaluation.metrics import rps_score
from pitlane.models.latent import LatentModel
from pitlane.models.registry import build_model
from pitlane.plan3 import attribution as A
from pitlane.plan3 import calibration as CAL
from pitlane.plan3 import decode as DEC

MDIR = Path(__file__).resolve().parent
RES = MDIR / "results"
CFG = load_yaml(MDIR / "config.yaml")
NB, SEED = CFG["bootstrap"]["n_boot"], CFG["bootstrap"]["seed"]
VARIANTS = list(CFG["variants"])                        # M11_finishers, M11_topk10, M11_reverse
RUN_SUFFIX = {"latent_111": "m11_ref_latent_111_dev", "M11_finishers": "m11_latent_111_finishers_dev",
              "M11_topk10": "m11_latent_111_top10_dev", "M11_reverse": "m11_latent_111_reverse_dev"}
PRIMARY = ["spearman", "mae", "exact"]
NICE = {"spearman": "rho", "mae": "MAE", "exact": "exact"}
LAM_GRID, HL_GRID = P.LAM_GRID, P.HL_GRID               # plan-1 grid for cell 111: 4 x 3 = 12 configurations (<= 30)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_run_module():
    spec = importlib.util.spec_from_file_location("m11_run", MDIR / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUNMOD = _load_run_module()


def run_dir(key: str) -> Path:
    hits = sorted(RES.glob(f"*_{RUN_SUFFIX[key]}"))
    assert len(hits) == 1, (key, hits)
    return hits[0]


def stored_preds() -> dict[str, pd.DataFrame]:
    return {k: pd.read_parquet(run_dir(k) / "predictions.parquet") for k in RUN_SUFFIX}


def free(preds: pd.DataFrame) -> pd.DataFrame:
    return RUNMOD.outcome_free(preds, CFG["tie_break"])


def tables(preds: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {k: A.race_table(free(v)) for k, v in preds.items()}


def grid_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Starting-grid order, outcome-free tie-break by driverId (same as M17 `grid_free`)."""
    out = preds.copy()
    sc = np.empty(len(out))
    for _, ix in out.groupby("race_id").indices.items():
        sc[ix] = DEC.grid_utility(out["grid"].to_numpy(float)[ix], out["driverId"].to_numpy()[ix])
    out["score"] = sc
    return A.race_table(out)


def verdict_word(lo: float, hi: float) -> str:
    return "gain" if lo > 0 else ("worse" if hi < 0 else "none")


# ------------------------------------------------------------------------------------------------ A: verdicts, Holm
def stage_post() -> None:
    preds = stored_preds()
    tabs = tables(preds)
    full = pd.read_csv(RES / "paired_gains_dev.csv")                     # read only
    # per-metric verdicts for rho, MAE, exact (rule: 95% paired bootstrap CI over races excludes 0)
    rows = []
    for comp in VARIANTS:
        g = A.paired_gain(tabs["latent_111"], tabs[comp], n_boot=NB, seed=SEED).set_index("metric")
        ch = full[full["component"] == comp].set_index("metric")
        for m in g.index:                                                  # harness numbers must equal the stored csv
            assert abs(g.loc[m, "gain"] - ch.loc[m, "gain"]) < 1e-12 and abs(g.loc[m, "lo"] - ch.loc[m, "lo"]) < 1e-12, (comp, m)
        for m in PRIMARY:
            r = g.loc[m]
            rows.append(dict(component=comp, metric=m, label=NICE[m], gain=r["gain"], lo=r["lo"], hi=r["hi"], n=int(r["n"]),
                             verdict=verdict_word(r["lo"], r["hi"])))
    ver = pd.DataFrame(rows)
    ver.to_csv(RES / "verdicts_per_metric_dev.csv", index=False)
    comp_str = {c: "; ".join(f"{NICE[m]}: {ver[(ver.component == c) & (ver.metric == m)].verdict.iloc[0]}" for m in PRIMARY)
                for c in VARIANTS}
    (RES / "verdict_strings_dev.json").write_text(json.dumps(comp_str, indent=2), encoding="utf-8")
    print("PER-METRIC VERDICTS\n" + ver.round(4).to_string(index=False))

    # multiplicity over all 36 stored intervals: normal-approximation p from the bootstrap interval, Holm at 0.05
    mult = full.copy()
    se = (mult["hi"] - mult["lo"]) / (2 * 1.959964)
    mult["se_from_ci"] = se
    mult["z"] = mult["gain"] / se
    mult["p_normal"] = 2 * stats.norm.sf(np.abs(mult["z"]))
    mult["ci_excludes_0"] = (mult["lo"] > 0) | (mult["hi"] < 0)
    order = mult["p_normal"].rank(method="first").astype(int)
    m_tot = len(mult)
    mult["holm_rank"] = order
    mult["holm_threshold"] = 0.05 / (m_tot - order + 1)
    srt = mult.sort_values("p_normal")
    rej, stop = {}, False
    for i, ix in enumerate(srt.index):
        ok = (not stop) and srt.loc[ix, "p_normal"] <= 0.05 / (m_tot - i)
        stop = stop or not ok
        rej[ix] = bool(ok)
    mult["holm_reject"] = pd.Series(rej)
    mult = mult.sort_values(["component", "metric"]).reset_index(drop=True)
    mult.to_csv(RES / "multiplicity_dev.csv", index=False)
    print(f"\nMULTIPLICITY: {m_tot} intervals, {int(mult.ci_excludes_0.sum())} exclude 0, {int(mult.holm_reject.sum())} survive Holm")
    print(mult[mult["ci_excludes_0"]].sort_values("p_normal")[["component", "metric", "gain", "lo", "hi", "p_normal", "holm_threshold",
                                                              "holm_reject"]].round(5).to_string(index=False))

    # ---- B: comparison with grid order (outcome-free) overall and per season; per-season gains vs the reference
    gt = grid_table(preds["latent_111"])
    metr = ["exact", "exact_fin", "exact_1_3", "exact_4_10", "exact_11p", "spearman", "mae"]
    unit = {m: ("points" if m.startswith("exact") else ("rho" if m == "spearman" else "positions")) for m in metr}
    rows = []
    for run in RUN_SUFFIX:
        t = tabs[run]
        for scope in ["all"] + sorted(t["season"].unique().tolist()):
            a = gt if scope == "all" else gt[gt["season"] == scope]
            b = t if scope == "all" else t[t["season"] == scope]
            g = A.paired_gain(a, b, metr, n_boot=NB, seed=SEED)
            for _, r in g.iterrows():
                k = 100.0 if r["metric"].startswith("exact") else 1.0
                rows.append(dict(run=run, scope=str(scope), metric=r["metric"], unit=unit[r["metric"]], run_mean=k * r["cand_mean"],
                                 grid_mean=k * r["ref_mean"], gain_vs_grid=k * r["gain"], lo=k * r["lo"], hi=k * r["hi"], n_races=int(r["n"]),
                                 verdict=verdict_word(r["lo"], r["hi"])))
    vg = pd.DataFrame(rows)
    vg.to_csv(RES / "vs_grid_dev.csv", index=False)
    print("\nVS GRID ORDER (positive = run better than grid; exact in points)\n" +
          vg[vg.scope == "all"].round(3).to_string(index=False))
    print("\nEXACT vs GRID by season\n" + vg[(vg.metric == "exact") & (vg.scope != "all")].round(2).to_string(index=False))

    rows = []
    for comp in VARIANTS:
        for s in sorted(tabs["latent_111"]["season"].unique()):
            a, b = tabs["latent_111"][tabs["latent_111"]["season"] == s], tabs[comp][tabs[comp]["season"] == s]
            g = A.paired_gain(a, b, ["spearman", "mae", "exact", "exact_1_3"], n_boot=NB, seed=SEED)
            for _, r in g.iterrows():
                k = 100.0 if r["metric"].startswith("exact") else 1.0
                rows.append(dict(component=comp, season=int(s), metric=r["metric"], gain=k * r["gain"], lo=k * r["lo"], hi=k * r["hi"],
                                 n_races=int(r["n"]), unit="points" if k == 100 else ("rho" if r["metric"] == "spearman" else "positions"),
                                 verdict=verdict_word(r["lo"], r["hi"])))
    ps = pd.DataFrame(rows)
    ps.to_csv(RES / "per_season_gains_dev.csv", index=False)
    print("\nPER-SEASON GAINS vs reference (positive = variant better)\n" + ps.round(3).to_string(index=False))

    # ---- C: face validity, predicted winners
    rows = []
    for run, p in preds.items():
        q = free(p)
        for rid, g in q.groupby("race_id"):
            rows.append(dict(source=run, season=int(g["season"].iloc[0]), driver=g.loc[g["score"].idxmax(), "driverId"]))
    acts = []
    for rid, g in preds["latent_111"].groupby("race_id"):
        s = int(g["season"].iloc[0])
        acts.append(dict(source="actual_winner", season=s, driver=g.loc[g["pos"].idxmin(), "driverId"]))
        for d in g.loc[g["grid"] == 1, "driverId"]:
            acts.append(dict(source="pole_sitter", season=s, driver=d))
    win = pd.DataFrame(rows + acts).groupby(["season", "source", "driver"]).size().rename("count").reset_index()
    wide = win.pivot_table(index=["season", "driver"], columns="source", values="count", fill_value=0).reset_index()
    cols = ["season", "driver", "actual_winner", "pole_sitter"] + list(RUN_SUFFIX)
    wide = wide[cols].sort_values(["season", "actual_winner", "latent_111"], ascending=[True, False, False])
    wide = wide[(wide[cols[2:]].sum(axis=1) > 0)]
    wide.to_csv(RES / "predicted_winner_by_season.csv", index=False)
    print("\nPREDICTED WINNER COUNTS (races per season: 22)\n" + wide.to_string(index=False))
    hit = {k: [] for k in RUN_SUFFIX}
    for run, p in preds.items():
        q = free(p)
        for rid, g in q.groupby("race_id"):
            hit[run].append((int(g["season"].iloc[0]), g.loc[g["score"].idxmax(), "pos"] == 1))
    tab = pd.DataFrame({k: pd.DataFrame(v, columns=["s", "h"]).groupby("s")["h"].sum() for k, v in hit.items()})
    tab.to_csv(RES / "winner_hits_by_season_dev.csv")
    print("\nCORRECT WINNER CALLS per season (of 22)\n" + tab.to_string())


# ------------------------------------------------------------------------------------------------ D: fitted components
GRAD_TOL = 1e-5


def _lbfgs(m: LatentModel, n_train: int, w_row, pad, theta0, maxiter: int, **opts):
    """Same call as LatentModel.fit, but keeps the scipy result so success, iterations and gradient can be recorded."""
    res = minimize(m._objective, theta0, args=(n_train, w_row, pad), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, **opts})
    g = m._objective(res.x, n_train, w_row, pad)[1]
    return res, float(np.max(np.abs(g)))


def fit_once(entries: pd.DataFrame, cfg: dict, first_race: int) -> LatentModel:
    """Cold fit at the frozen setting (identical to LatentModel.fit, maxiter 3000). Convergence is recorded on the model:
    m._conv (first fit) and, if max|grad| >= 1e-5, m._conv2 (refit from that result, 20000 iterations, ftol 1e-15, gtol 1e-8)."""
    c = {k: v for k, v in cfg.items() if k not in ("kind", "cell", "label")}
    m = LatentModel(entries, dynamic=True, decomp=True, ranking=True, maxiter=3000, **c)
    n_train = int(m.race_rows[first_race].min())
    w_row, pad = m.fit_inputs(n_train, float(first_race))
    res, gmax = _lbfgs(m, n_train, w_row, pad, m.theta, m.maxiter)
    m.theta = res.x
    m._n_train = n_train
    m._conv = dict(success=bool(res.success), n_iter=int(res.nit), max_abs_grad=gmax,
                   converged=bool(res.success and gmax < GRAD_TOL), message=str(res.message))
    m._conv2 = None
    m._theta_first = res.x.copy()
    if not m._conv["converged"]:
        res2, g2 = _lbfgs(m, n_train, w_row, pad, res.x, 20000, ftol=1e-15, gtol=1e-8)
        m.theta = res2.x
        m._conv2 = dict(success=bool(res2.success), n_iter=int(res2.nit), max_abs_grad=g2,
                        converged=bool(res2.success and g2 < GRAD_TOL), message=str(res2.message))
    return m


def stage_mech() -> None:
    entries = P.load_entries()
    base = load_yaml(ROOT / CFG["frozen_path"])[CFG["frozen_key"]]["111"]
    cfgs = {"latent_111": dict(base), **{c: dict(base, **v["switches"]) for c, v in CFG["variants"].items()}}
    firsts = {2022: int(entries.loc[entries["season"] == 2022, "race_id"].min()), 2023: int(entries.loc[entries["season"] == 2023, "race_id"].min())}
    rows = []
    for run, cfg in cfgs.items():
        for season, rid in firsts.items():
            t = time.time()
            m = fit_once(entries, cfg, rid)
            n = m._n_train
            first_theta, m.theta = m._theta_first, m.theta           # first fit (maxiter 3000) vs final (refit if needed)
            a, b, g, beta = m._split(m.theta)
            tr = entries.iloc[:n]
            last = int(tr["season"].max())
            lr = np.where(m.row_season[:n] == last)[0]
            drv_used = np.unique(m.row_d[:n])
            car_used = np.unique(m.row_b[:n])
            gcells = np.unique(m.row_g[:n])
            car_last = np.unique(m.row_b[lr])
            inv_d = {i: d for d, i in m.drivers.items()}
            inv_c = {i: p for p, i in m.car.items()}
            top_d = sorted(np.unique(m.row_d[lr]), key=lambda i: -a[i])[:5]
            top_c = sorted(car_last, key=lambda i: -b[i])[:5]
            wk = m.row_x[lr] @ beta
            drv_part, car_part = a[m.row_d[lr]], b[m.row_b[lr]] + g[m.row_g[lr]]
            row = dict(run=run, fit_before_season=season, fit_before_race_id=rid, n_train_rows=n, last_training_season=last,
                       sd_driver_a_all_used=float(a[drv_used].std()), sd_driver_a_last_season=float(a[np.unique(m.row_d[lr])].std()),
                       sd_car_b_all_used=float(b[car_used].std()), sd_car_b_last_season=float(b[car_last].std()),
                       sd_g_used=float(g[gcells].std()),
                       sd_u_driver_part_last=float(drv_part.std()), sd_u_car_part_last=float(car_part.std()),
                       sd_u_weekend_part_last=float(wk.std()), sd_u_last=float((drv_part + car_part + wk).std()),
                       top5_drivers="; ".join(f"{inv_d[i]} {a[i]:+.3f}" for i in top_d),
                       top5_cars="; ".join(f"{inv_c[i][0]}-{inv_c[i][1]} {b[i]:+.3f}" for i in top_c))
            for nm, v in zip(m.x_names, beta):
                row[f"beta_{nm}"] = float(v)
            cv = m._conv2 or m._conv
            row.update(converged=cv["converged"], max_abs_grad=cv["max_abs_grad"], n_iter=m._conv["n_iter"] + (m._conv2["n_iter"] if m._conv2 else 0),
                       first_fit_converged=m._conv["converged"], first_fit_max_abs_grad=m._conv["max_abs_grad"],
                       first_fit_n_iter=m._conv["n_iter"], refit_done=m._conv2 is not None)
            if m._conv2 is not None:                                 # old-vs-new for the three card quantities
                a1, b1, g1, beta1 = m._split(first_theta)
                wk1, u1 = m.row_x[lr] @ beta1, None
                u1 = a1[m.row_d[lr]] + b1[m.row_b[lr]] + g1[m.row_g[lr]] + wk1
                row.update(first_fit_beta_quali_pos=float(beta1[m.x_names.index("quali_pos")]), first_fit_sd_u_last=float(u1.std()),
                           first_fit_weekend_share=float(wk1.var() / u1.var()))
            row["weekend_share"] = float(wk.var() / (drv_part + car_part + wk).var())
            rows.append(row)
            log(f"  {run} before {season}: fit {time.time() - t:.1f}s, beta_quali_pos={row['beta_quali_pos']:+.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(RES / "fitted_components.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 80)
    print(out.drop(columns=["top5_drivers", "top5_cars"]).round(3).to_string(index=False))
    print(out[["run", "fit_before_season", "top5_drivers", "top5_cars"]].to_string(index=False))


# ------------------------------------------------------------------------------------------------ E: tuning
def stage_tune() -> None:
    entries = P.load_entries()
    base = load_yaml(ROOT / CFG["frozen_path"])[CFG["frozen_key"]]["111"]
    t0 = time.time()
    rows, store = [], {}
    comps = {"latent_111": {}, **{c: v["switches"] for c, v in CFG["variants"].items()}}
    for comp, sw in comps.items():
        for lam in LAM_GRID:
            for hl in HL_GRID:
                cfg = dict(base, lam=float(lam), halflife=hl, **sw)
                t = time.time()
                p, m = walk_forward(entries, build_model(cfg, entries), "dev", refit_every=1)
                store[(comp, lam, hl)] = (cfg, p, m)
                rows.append(dict(component=comp, lam=lam, halflife=hl, spearman=m["spearman"].mean(), mae=m["mae"].mean(),
                                 exact=m["exact"].mean(), spearman_fin=m["spearman_fin"].mean(), n_races=len(m),
                                 is_frozen_setting=bool(lam == base["lam"] and hl == base["halflife"])))
                log(f"  {comp} lam={lam} hl={hl}: rho={rows[-1]['spearman']:.4f} mae={rows[-1]['mae']:.3f} exact={rows[-1]['exact']:.4f} ({time.time() - t:.0f}s)")
    grid = pd.DataFrame(rows)
    grid["chosen_by_rho"] = False
    for comp in comps:
        ix = grid.index[grid["component"] == comp]
        grid.loc[grid.loc[ix, "spearman"].idxmax(), "chosen_by_rho"] = True
    grid.to_csv(RES / "tuning_grid_dev.csv", index=False)

    chosen = {c: grid[(grid.component == c) & grid.chosen_by_rho].iloc[0] for c in comps}
    key = lambda c: (c, float(chosen[c]["lam"]), int(chosen[c]["halflife"]))
    ref_frozen = A.race_table(free(store[("latent_111", base["lam"], base["halflife"])][1]))
    ref_tuned = A.race_table(free(store[key("latent_111")][1]))
    rows = []
    for comp in comps:
        cfg, p, m = store[key(comp)]
        t = A.race_table(free(p))
        frozen_cfg_tab = A.race_table(free(store[(comp, base["lam"], base["halflife"])][1]))
        for refname, ref in (("reference at frozen setting (lam 32, hl 50)", ref_frozen), ("reference at its own tuned setting", ref_tuned)):
            g = A.paired_gain(ref, t, ["spearman", "mae", "exact", "exact_1_3"], n_boot=NB, seed=SEED).set_index("metric")
            row = dict(component=comp, chosen_lam=chosen[comp]["lam"], chosen_halflife=int(chosen[comp]["halflife"]), versus=refname,
                       rho=t["spearman"].mean(), mae=t["mae"].mean(), exact_pct=100 * t["exact"].mean())
            for mm in ("spearman", "mae", "exact", "exact_1_3"):
                k = 100.0 if mm.startswith("exact") else 1.0
                row.update({f"gain_{mm}": k * g.loc[mm, "gain"], f"lo_{mm}": k * g.loc[mm, "lo"], f"hi_{mm}": k * g.loc[mm, "hi"]})
            rows.append(row)
        if comp != "latent_111":     # untuned (frozen-setting) variant vs reference at frozen setting: the earlier comparison
            g = A.paired_gain(ref_frozen, frozen_cfg_tab, ["spearman", "mae", "exact", "exact_1_3"], n_boot=NB, seed=SEED).set_index("metric")
            row = dict(component=comp, chosen_lam=base["lam"], chosen_halflife=base["halflife"], versus="UNTUNED variant vs reference at frozen setting",
                       rho=frozen_cfg_tab["spearman"].mean(), mae=frozen_cfg_tab["mae"].mean(), exact_pct=100 * frozen_cfg_tab["exact"].mean())
            for mm in ("spearman", "mae", "exact", "exact_1_3"):
                k = 100.0 if mm.startswith("exact") else 1.0
                row.update({f"gain_{mm}": k * g.loc[mm, "gain"], f"lo_{mm}": k * g.loc[mm, "lo"], f"hi_{mm}": k * g.loc[mm, "hi"]})
            rows.append(row)
    tv = pd.DataFrame(rows)
    tv.to_csv(RES / "tuned_vs_untuned_dev.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print("\nTUNING GRID (dev, walk-forward, refit every race)\n" + grid.round(4).to_string(index=False))
    print("\nTUNED vs UNTUNED (exact in points)\n" + tv.round(3).to_string(index=False))

    # run folders for the three tuned variants and the tuned reference (leaderboard rows are appended in the final stage)
    tmp = RES / "_lb_tune_tmp.csv"
    tmp.unlink(missing_ok=True)
    ids = {}
    for comp in comps:
        cfg, p, m = store[key(comp)]
        lab = "m11_tuned_ref_latent_111" if comp == "latent_111" else "m11_tuned_" + CFG["variants"][comp]["label"]
        rid = save_run(MDIR, lab, dict(cfg, component=comp, tuned_on="dev, 12-point plan-1 grid (lam x halflife), selected by mean rho"),
                       p, m, "dev", "M11", "111", leaderboard=tmp)
        pth = RES / rid / "config.yaml"
        c2 = load_yaml(pth)
        c2["git_commit"] = RUNMOD.code_stamp()
        save_yaml(c2, pth)
        ids[comp] = rid
    (RES / "tuned_run_ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    log(f"tuning done in {time.time() - t0:.0f}s; run ids {ids}")


# ------------------------------------------------------------------------------------------------ F: M11 x M07
def sim_frame(preds: pd.DataFrame, entries2: pd.DataFrame, dn: pd.DataFrame) -> pd.DataFrame:
    cols = ["race_id", "driverId", "grid_norm", "stick", "circ_vol", "rain_fc"]
    return preds.merge(dn, on=["race_id", "driverId"]).merge(entries2[cols], on=["race_id", "driverId"])


def stage_inter() -> None:
    from pitlane.plan2 import dnf as D
    from pitlane.plan2 import pipeline2 as Q
    from pitlane.plan2 import simulate as S
    t0 = time.time()
    frozen = load_yaml(ROOT / "configs" / "frozen_plan2.yaml")
    theta = {k: frozen["M08"]["S3"][k] for k in S.DEFAULT}
    consts = {k: frozen["M08"]["S3"][k] for k in ("stick_ref", "vol_mu", "vol_sd")}
    entries2 = Q.load_all()
    lb = pd.read_csv(D.LB_DNF)
    dn = pd.read_parquet(ROOT / lb[lb["split"] == "dev"].iloc[-1]["run_dir"] / "predictions.parquet")[["race_id", "driverId", "p_dnf"]]
    preds = stored_preds()
    pairs = {"latent_111": preds["latent_111"], "M11_finishers": preds["M11_finishers"]}
    # descriptive diagnostic only (not a component, no verdict): the frozen S3 tau was fitted on a score with within-race sd
    # about 0.7, and the finishers-only fit has a larger scale. The ratio of mean within-race sd (pre-race scores, no outcome)
    # puts the finishers-only score on the reference's scale.
    sd = {k: float(v.groupby("race_id")["score"].std().mean()) for k, v in pairs.items()}
    ratio = sd["latent_111"] / sd["M11_finishers"]
    scaled = preds["M11_finishers"].copy()
    scaled["score"] = scaled["score"] * ratio
    pairs["M11_finishers_scale_matched"] = scaled
    draws, seed = 1000, 0            # as plan-2 evaluate() and M17: stream seed + 10000 + race index
    tabs, mats, scale, calrows, built = {}, {}, {}, [], {}
    for name, p in pairs.items():
        fr = sim_frame(p, entries2, dn)
        assert len(fr) == len(p), (len(fr), len(p))
        races = S.prepare(fr, consts)
        mt = DEC.simulate_matrices(races, theta, "S3", draws, seed)
        keys = [r["g"]["driverId"].to_numpy() for r in races]
        util = [DEC.break_ties(-DEC.expected_position(P_), k) for P_, k in zip(mt, keys)]
        out = []
        for r, P_, u in zip(races, mt, util):
            d = r["g"][["race_id", "season", "round", "driverId", "lineage", "pos", "grid", "classified"]].copy()
            n = len(d)
            d["score"], d["p_win"], d["p_podium"], d["p_points"], d["p_dnf"] = u, P_[:, 0], P_[:, :3].sum(1), P_[:, :min(10, n)].sum(1), r["p_dnf"]
            out.append(d)
        built[name] = pd.concat(out, ignore_index=True)
        rps = {r["race_id"]: rps_score(P_, r["pos"]) for r, P_ in zip(races, mt)}
        tabs[name] = A.race_table(built[name], rps=rps)
        mats[name] = (races, mt, keys)
        scale[name] = float(np.mean([r["u"].std() for r in races]))
        gs = CAL.grid_slot_calibration(mt, [r["g"]["grid"].to_numpy(float) for r in races], [r["pos"] for r in races], keys)
        calrows.append(dict(run=name, mean_pred_P_grid_slot=gs["mean_pred"], realised_grid_slot=gs["realised"],
                            mean_within_race_sd_of_u=scale[name]))
        log(f"  {name}: races {len(races)}, rps {tabs[name]['rps'].mean():.4f}, ll_points {tabs[name]['logloss_points'].mean():.4f}, "
            f"exact {100 * tabs[name]['exact'].mean():.2f}, mean sd(u) {scale[name]:.3f}")
    cal = pd.DataFrame(calrows)
    cal.to_csv(RES / "interaction_calibration_dev.csv", index=False)
    print("\nGRID-SLOT CALIBRATION (and within-race sd of u)\n" + cal.round(4).to_string(index=False))
    ref, cand = tabs["latent_111"], tabs["M11_finishers"]
    gsm = A.paired_gain(ref, tabs["M11_finishers_scale_matched"], n_boot=NB, seed=SEED)
    gsm.to_csv(RES / "interaction_scale_matched_dev.csv", index=False)
    print(f"\nDESCRIPTIVE ONLY: finishers-only score multiplied by {ratio:.4f} (ratio of mean within-race sd of u), same simulator\n"
          + gsm.round(4).to_string(index=False))
    g = A.paired_gain(ref, cand, n_boot=NB, seed=SEED)
    g.to_csv(RES / "interaction_paired_gains_dev.csv", index=False)
    print("\nINTERACTION: finishers-only x M07 (S3) minus forward latent_111 x M07 (S3); positive = finishers better\n" + g.round(4).to_string(index=False))
    # interaction contrast: gain inside the simulator minus gain of the same likelihood change without it (plain order)
    plain = tables(preds)
    rows = []
    for m_ in ("spearman", "mae", "exact", "exact_1_3", "logloss_points"):
        sgn = 1.0 if A.HIGHER_BETTER[m_] else -1.0
        if m_ == "logloss_points":
            continue
        a = ref.merge(cand, on="race_id", suffixes=("_r", "_c"))
        b = plain["latent_111"].merge(plain["M11_finishers"], on="race_id", suffixes=("_r", "_c"))
        dsim = sgn * (a[f"{m_}_c"] - a[f"{m_}_r"]).to_numpy(float)
        dplain = sgn * (b[f"{m_}_c"] - b[f"{m_}_r"]).to_numpy(float)
        assert (a["race_id"].to_numpy() == b["race_id"].to_numpy()).all()
        d = dsim - dplain
        rng = np.random.default_rng([SEED, 77, zlib.crc32(m_.encode())])
        bm = d[rng.integers(0, len(d), size=(NB, len(d)))].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        rows.append(dict(metric=m_, gain_in_simulator=float(dsim.mean()), gain_plain_order=float(dplain.mean()), contrast=float(d.mean()),
                         lo=float(lo), hi=float(hi)))
    con = pd.DataFrame(rows)
    con.to_csv(RES / "interaction_contrast_dev.csv", index=False)
    print("\nINTERACTION CONTRAST (gain inside simulator minus gain with plain order)\n" + con.round(4).to_string(index=False))
    # run folders
    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    gid = RUNMOD.code_stamp()
    lbrows = []
    for name, lab in (("latent_111", "m11_inter_S3_ref_latent_111"), ("M11_finishers", "m11_inter_S3_latent_111_finishers")):  # scale-matched: no run folder
        rid = f"{stamp}_{lab}_dev"
        out = RES / rid
        out.mkdir(parents=True, exist_ok=True)
        save_yaml({"component": name, "simulator": "S3 (plan-2 frozen parameters, unchanged)", "order_model": name,
                   "retirement_model": "M07 dev predictions (plan-2 stored run)", "draws": draws, "seed": seed, "split": "dev",
                   "git_commit": gid}, out / "config.yaml")
        per = tabs[name]
        summ = {k: float(per[k].mean()) for k in per.columns if k not in ("race_id", "season", "round", "n", "tau")}
        summ["n_races"] = int(len(per))
        (out / "metrics.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
        per.to_csv(out / "per_race_metrics.csv", index=False)
        built[name].to_parquet(out / "predictions.parquet", index=False)
        lbrows.append({"method": "M11", "cell": "111xS3", "label": lab, "split": "dev", "run_id": rid, "run_dir": str(out.relative_to(ROOT)), **summ})
    pd.DataFrame(lbrows).to_csv(RES / "_lb_inter_rows.csv", index=False)
    (RES / "interaction_run_ids.json").write_text(json.dumps({"latent_111": lbrows[0]["run_id"], "M11_finishers": lbrows[1]["run_id"]}, indent=2),
                                                  encoding="utf-8")
    log(f"interaction done in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------ final writes
def stage_final() -> None:
    strings = json.loads((RES / "verdict_strings_dev.json").read_text(encoding="utf-8"))
    gains = pd.read_csv(RES / "interaction_paired_gains_dev.csv")
    inter_ids = json.loads((RES / "interaction_run_ids.json").read_text(encoding="utf-8"))
    # attribution.csv: re-read right before writing; change only M11 rows
    path = ROOT / CFG["attribution"]
    att = pd.read_csv(path, dtype={"loo_gain": object, "fresh_holdout_gain": object})
    for comp, s in strings.items():
        m = (att["component"] == comp) & (att["reference"] == "latent_111")
        assert m.sum() == 1, comp
        att.loc[m, "verdict_metric"] = "spearman; mae; exact (per-metric)"
        att.loc[m, "verdict"] = s + "; dev only"
    row = A.attribution_row("M11_finishers_x_M07", "latent_111+M07 (S3)", gains, "rps", inter_ids["M11_finishers"], int(gains["n"].max()), declared="yes")
    verdict_by = {mm: verdict_word(*gains[gains.metric == mm][["lo", "hi"]].iloc[0]) for mm in ("rps", "spearman", "mae", "exact")}
    row["verdict_metric"] = "rps; rho; mae; exact (per-metric)"
    row["verdict"] = (f"RPS: {verdict_by['rps']}; rho: {verdict_by['spearman']}; MAE: {verdict_by['mae']}; exact: {verdict_by['exact']}; dev only")
    att = att[~((att["component"] == row["component"]) & (att["reference"] == row["reference"]))]
    att = pd.concat([att, pd.DataFrame([row])], ignore_index=True)
    att.to_csv(path, index=False)
    log(f"attribution.csv updated: M11 verdict strings {strings}; interaction verdict {row['verdict']}")
    # leaderboard: append only, re-read header right before writing
    lb_path = ROOT / CFG["leaderboard"]
    header = pd.read_csv(lb_path, nrows=0).columns.tolist()
    have = set(pd.read_csv(lb_path, usecols=["run_id"])["run_id"])
    tune_rows = pd.read_csv(RES / "_lb_tune_tmp.csv")
    new = pd.concat([tune_rows, pd.read_csv(RES / "_lb_inter_rows.csv")], ignore_index=True)
    new = new[~new["run_id"].isin(have)]
    text = lb_path.read_text(encoding="utf-8")
    with open(lb_path, "a", encoding="utf-8", newline="") as f:
        if not text.endswith("\n"):
            f.write("\n")
        new.reindex(columns=header).to_csv(f, index=False, header=False, lineterminator="\n")
    (RES / "_lb_tune_tmp.csv").unlink(missing_ok=True)
    (RES / "_lb_inter_rows.csv").unlink(missing_ok=True)
    log(f"leaderboard: appended {len(new)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["post", "mech", "tune", "inter", "final"])
    a = ap.parse_args()
    t = time.time()
    {"post": stage_post, "mech": stage_mech, "tune": stage_tune, "inter": stage_inter, "final": stage_final}[a.stage]()
    log(f"stage {a.stage} took {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
