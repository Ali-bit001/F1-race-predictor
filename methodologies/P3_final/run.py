"""PLAN3 R14: assemble the final pre-race model on the development block, leave-one-out, re-fit the simulator, prepare the freeze.

    python methodologies/P3_final/run.py [--quick]

Dev 2021-2023 only. Rules are fixed in methodologies/P3_final/prestatement.md (written before any run). Steps:
  0. verify the facts taken from results/plan3/attribution.csv; build the pace column; reference reproduction
  1. component car_penalty_retune alone (8 configurations), judged against latent_111
  2. final candidate = latent_111 + every component that passed alone; lam_x check for the combination (4 configurations)
  3. leave-one-out; a component that is not retained is removed (backward elimination, as pre-stated)
  4. simulator: plan-2 S3 re-fitted on dev on top of the final score and on top of latent_111
  5. run folders, leaderboard rows, attribution rows (append only), frozen_config.yaml and FREEZE.md
--quick: first 6 dev races, nothing written.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import yaml

from pitlane.config import ROOT, SPLITS, load_yaml, save_yaml
from pitlane.evaluation.harness import _git_hash, save_run, walk_forward
from pitlane.evaluation.metrics import rps_score
from pitlane.models.registry import build_model
from pitlane.plan2 import dnf as D
from pitlane.plan2 import pipeline2 as Q
from pitlane.plan2 import simulate as S
from pitlane.plan3 import attribution as A
from pitlane.plan3 import calibration as CAL
from pitlane.plan3 import decode as DEC
from pitlane.plan3 import final as F
from pitlane.plan3 import runutil as U

MDIR = Path(__file__).resolve().parent
RES = MDIR / "results"
USED = ["src/pitlane/plan3/final.py", "src/pitlane/plan3/attribution.py", "src/pitlane/plan3/runutil.py", "src/pitlane/models/latent.py",
        "src/pitlane/features/pace.py", "src/pitlane/evaluation/harness.py", "methodologies/P3_final/run.py",
        "methodologies/P3_final/config.yaml", "methodologies/P3_final/prestatement.md"]
COMP_NAMES = {"reverse": "M11_reverse", "pace_hist": "M09_car_pace_hist", "car_penalty": "P3_car_penalty_retune"}
# Row names of the leave-one-out contrasts in attribution.csv. They must not start with M09/M11: the earlier packages' guard tests
# pin the set of rows whose component starts with their own prefix (the reference column carries the full names).
LOO_NAMES = {"reverse": "P3_LOO_reverse", "pace_hist": "P3_LOO_pace_hist", "car_penalty": "P3_LOO_car_penalty"}
PASS_METRICS = ["spearman", "mae", "exact"]
SHOW = ["spearman", "mae", "exact", "exact_fin", "spearman_fin", "mae_fin", "exact_1_3", "exact_4_10", "exact_11p", "logloss_points"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------ small helpers
def pass_flags(gains: pd.DataFrame) -> dict[str, bool]:
    """Section 5 rule per primary metric: the CI excludes 0 on the positive side (positive = candidate better)."""
    g = gains.set_index("metric")
    return {m: bool(g.loc[m, "lo"] > 0) for m in PASS_METRICS if m in g.index}


def fmt(g: pd.DataFrame, metrics=("spearman", "mae", "exact")) -> str:
    gi = g.set_index("metric")
    return "; ".join(f"{m} {gi.loc[m, 'gain']:+.4f} [{gi.loc[m, 'lo']:+.4f}, {gi.loc[m, 'hi']:+.4f}]" for m in metrics if m in gi.index)


def append_text_rows(rows: pd.DataFrame, path: Path, key=("component", "reference")) -> None:
    """Append rows as text lines in one write; existing lines are never rewritten (the files are append-only)."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    existing = pd.read_csv(path, usecols=list(key))
    for _, r in rows.iterrows():
        assert not ((existing[key[0]] == r[key[0]]) & (existing[key[1]] == r[key[1]])).any(), f"row exists: {r[key[0]]} / {r[key[1]]}"
    text = path.read_text(encoding="utf-8")
    body = rows.reindex(columns=header).to_csv(index=False, header=False, lineterminator="\n")
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(("\n" if not text.endswith("\n") else "") + body)


def grid_table(preds: pd.DataFrame, key: str) -> pd.DataFrame:
    gp = preds[["race_id", "season", "driverId", "pos", "grid", "classified"]].copy()
    gp["score"] = np.concatenate([DEC.grid_utility(g["grid"].to_numpy(float), g[key].to_numpy()) for _, g in gp.groupby("race_id", sort=False)])
    return A.race_table(gp)


def sim_tables(races: list[dict], mats: list[np.ndarray], keys: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(predictions, per-race table) of a simulator run; order = D0 (sort by -E[position]) with an outcome-free tie-break."""
    out = []
    for r, P_, k in zip(races, mats, keys):
        d = r["g"][["race_id", "season", "round", "driverId", "lineage", "pos", "grid", "classified"]].copy()
        n = len(d)
        d["score"] = DEC.break_ties(-DEC.expected_position(P_), k)
        d["p_win"], d["p_podium"], d["p_points"], d["p_dnf"] = P_[:, 0], P_[:, :3].sum(1), P_[:, :min(10, n)].sum(1), r["p_dnf"]
        out.append(d)
    preds = pd.concat(out, ignore_index=True)
    rps = {r["race_id"]: rps_score(P_, r["pos"]) for r, P_ in zip(races, mats)}
    return preds, A.race_table(preds, rps=rps)


# ------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MDIR / "config.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["split"] == "dev", "plan 3 / R14: development block only"
    t0 = time.time()
    nb = cfg["bootstrap"]
    key = cfg["tie_break"]
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)

    # ---------------------------------------------------------------- 0. facts, data, pace column
    att = pd.read_csv(ROOT / cfg["attribution"])
    for comp, info in cfg["included_alone"].items():
        r = att[(att["component"] == comp) & (att["reference"] == "latent_111")].iloc[0]
        ok = {m: bool(r[f"lo_{m}"] > 0) for m in PASS_METRICS}
        log(f"FACT {comp} vs latent_111: " + "; ".join(f"{m} {r[f'gain_{m}']:+.4f} [{r[f'lo_{m}']:+.4f}, {r[f'hi_{m}']:+.4f}]" for m in PASS_METRICS))
        assert all(ok[m] for m in info["criterion"]), f"{comp}: expected to pass alone on {info['criterion']}"
    for comp in cfg["not_included"]:
        rr = att[att["component"] == comp]
        assert len(rr), comp
    for comp in ("M09_car_q1_offset", "M09_car_q1_offset_alone"):
        log(f"FACT {comp}: declared={att[att.component == comp].declared.iloc[0]} (not included: increment null / not declared)")

    entries0 = Q.load_all()
    frozen = load_yaml(ROOT / cfg["frozen_path"])
    frozen2 = load_yaml(ROOT / cfg["frozen_plan2_path"])
    base = dict(frozen[cfg["base"]["key"]][cfg["base"]["cell"]])
    spec = dict(cfg["pace_feature"])
    pace = F.build_pace_table(spec)
    raw = F.pace_hist_raw(entries0, pace, spec)
    spec["scale_sd"] = F.fit_scale_sd(raw, entries0["season"].to_numpy(), int(spec["scale_seasons_max"]))
    m09_info = json.loads((ROOT / cfg["m09_feature_info"]).read_text(encoding="utf-8"))
    assert spec["scale_sd"] == m09_info["pace_hist_x"]["scale_sd"], (spec["scale_sd"], m09_info["pace_hist_x"]["scale_sd"])
    entries = F.add_pace_hist(entries0, pace, spec)
    lam_x_m09 = float(pd.read_csv(ROOT / cfg["m09_tuning_csv"]).query("component == 'M09_car_pace_hist'").sort_values("spearman").iloc[-1]["lam_x"])
    log(f"pace column built from {len(pace)} pace rows; scale SD {spec['scale_sd']!r} equals the M09 constant; M09 lam_x {lam_x_m09}")

    first = entries.groupby("race_id")["season"].first()
    dev_ids = first[(first >= SPLITS["dev"][0]) & (first <= SPLITS["dev"][1])].index
    entries_run = entries
    if args.quick:
        dev_ids = dev_ids[:6]
        entries_run = entries[entries["race_id"] <= dev_ids.max()].reset_index(drop=True)
    n_evals = {"car_penalty": 0, "lam_x_combo": 0}

    cache: dict[str, tuple] = {}

    def run(label: str, mcfg: dict, refit: int, split: str = "dev", ent: pd.DataFrame | None = None):
        k = json.dumps([mcfg, refit, split], sort_keys=True, default=str)
        if k in cache:
            log(f"  {label}: (cached)")
            return cache[k]
        t = time.time()
        e = entries_run if ent is None else ent
        p, m = walk_forward(e, build_model(mcfg, e), split, refit_every=refit)
        if args.quick and split == "dev":
            p, m = p[p["race_id"].isin(dev_ids)], m[m["race_id"].isin(dev_ids)]
        log(f"  {label}: rho={m['spearman'].mean():.4f} mae={m['mae'].mean():.3f} exact={m['exact'].mean():.4f} ({time.time() - t:.0f}s)")
        cache[k] = (p, m)
        return p, m

    def tab(p: pd.DataFrame) -> pd.DataFrame:
        return A.race_table(U.outcome_free(p, key))

    def gain(ref: pd.DataFrame, cand: pd.DataFrame, metrics=None) -> pd.DataFrame:
        return A.paired_gain(ref, cand, metrics, n_boot=nb["n_boot"], seed=nb["seed"])

    # reference and reproduction
    ref_cfg = F.latent_cfg(base)
    ref_p, ref_m = run("latent_111 (reference)", ref_cfg, cfg["refit_every"])
    rep = {}
    if not args.quick:
        old_pm, old_pr = U.stored_run(cfg["plan1_leaderboard"], "latent_111")
        d, n = U.max_diff(ref_m, old_pm)
        ds, ne = U.score_diff(ref_p, old_pr)
        rep["latent_111 vs stored plan-1 dev run"] = dict(max_abs_diff_per_race_metrics=d, max_abs_diff_scores=ds, races=n, entries=ne)
        log(f"REPRODUCTION latent_111: max |per-race metric diff| {d:.3e}, max |score diff| {ds:.3e} ({n} races, {ne} entries)")
        # all new switches off through the final-model builder is the reference, bit for bit
        off = F.latent_cfg(base, reverse=False, pace_hist=False, car_lam=None)
        assert off == ref_cfg
        p_off, _ = run("final builder, every switch off", dict(off, x_cols=[]), cfg["refit_every"])
        rep["every switch off vs latent_111"] = dict(max_abs_diff_scores=float(np.abs(p_off["score"].to_numpy() - ref_p["score"].to_numpy()).max()))
        log(f"REPRODUCTION switches off: max |score diff| {rep['every switch off vs latent_111']['max_abs_diff_scores']:.3e}")
    tabs = {"latent_111": tab(ref_p), "grid": grid_table(ref_p, key)}

    # ---------------------------------------------------------------- 1. car_penalty_retune alone
    log("STEP 1: car_penalty_retune (grid fixed in prestatement.md)")
    rows = []
    for lam in cfg["car_penalty"]["grid"]:
        _, m = run(f"tune car lam={lam}", F.latent_cfg(base, car_lam=lam), cfg["tune_refit_every"])
        rows.append(dict(lam_car=lam, lam_a=base["lam"], spearman=m["spearman"].mean(), mae=m["mae"].mean(), exact=m["exact"].mean(), n_races=len(m)))
        n_evals["car_penalty"] += 1
    tune_car = pd.DataFrame(rows)
    best_lam = float(tune_car.loc[tune_car["spearman"].idxmax(), "lam_car"])
    log(f"car lam picked {best_lam} (base {base['lam']}); configurations {n_evals['car_penalty']} of 30\n" + tune_car.round(4).to_string(index=False))
    carpen_cfg = F.latent_cfg(base, car_lam=best_lam)
    carpen_active = best_lam != float(base["lam"])
    carpen_p, carpen_m = run("car_penalty_retune alone", carpen_cfg, cfg["refit_every"])
    tabs["car_penalty"] = tab(carpen_p)
    g_car = gain(tabs["latent_111"], tabs["car_penalty"])
    g_car_season = pd.concat([gain(tabs["latent_111"][tabs["latent_111"]["season"] == s], tabs["car_penalty"][tabs["car_penalty"]["season"] == s],
                                   ["spearman", "mae", "exact"]).assign(season=s) for s in (2021, 2022, 2023)], ignore_index=True)
    car_flags = pass_flags(g_car)
    car_passes = carpen_active and any(car_flags.values())
    log(f"car_penalty_retune vs latent_111: {fmt(g_car)}; pass flags {car_flags}; active {carpen_active}; PASSES ALONE: {car_passes}")
    print(g_car_season.round(4).to_string(index=False))

    # ---------------------------------------------------------------- 2. final candidate
    crit = {"reverse": list(cfg["included_alone"]["M11_reverse"]["criterion"]),
            "pace_hist": list(cfg["included_alone"]["M09_car_pace_hist"]["criterion"])}
    if car_passes:
        crit["car_penalty"] = [m for m, v in car_flags.items() if v]
    comps = ["reverse", "pace_hist"] + (["car_penalty"] if car_passes else [])
    log(f"STEP 2: candidate components {[COMP_NAMES[c] for c in comps]}")

    def build(cs: list[str], lam_x: float) -> dict:
        return F.latent_cfg(base, reverse="reverse" in cs, pace_hist="pace_hist" in cs, lam_x=lam_x,
                            car_lam=best_lam if "car_penalty" in cs else None)

    rows = []
    for lx in cfg["lam_x_check"]["grid"]:
        _, m = run(f"tune combo lam_x={lx}", build(comps, lx), cfg["tune_refit_every"])
        rows.append(dict(lam_x=lx, spearman=m["spearman"].mean(), mae=m["mae"].mean(), exact=m["exact"].mean(), n_races=len(m)))
        n_evals["lam_x_combo"] += 1
    tune_lx = pd.DataFrame(rows)
    at_m09 = float(tune_lx.loc[tune_lx["lam_x"] == lam_x_m09, "spearman"].iloc[0])
    best_row = tune_lx.loc[tune_lx["spearman"].idxmax()]
    lam_x = float(best_row["lam_x"]) if best_row["spearman"] - at_m09 > cfg["lam_x_check"]["switch_margin"] else lam_x_m09
    log(f"combination lam_x check: M09 value {lam_x_m09} rho {at_m09:.4f}; best {best_row['lam_x']} rho {best_row['spearman']:.4f}; "
        f"margin {cfg['lam_x_check']['switch_margin']} -> lam_x {lam_x}\n" + tune_lx.round(4).to_string(index=False))

    # ---------------------------------------------------------------- 3. leave-one-out with backward elimination
    loo_rounds = []
    cur = list(comps)
    while True:
        full_cfg = build(cur, lam_x)
        full_p, _ = run(f"full model {[COMP_NAMES[c] for c in cur]}", full_cfg, cfg["refit_every"])
        tabs["full:" + ",".join(cur)] = tab(full_p)
        rnd = []
        for c in cur:
            minus = [x for x in cur if x != c]
            mp, _ = run(f"full minus {COMP_NAMES[c]}", build(minus, lam_x), cfg["refit_every"])
            tabs["minus:" + ",".join(minus)] = tab(mp)
            g = gain(tabs["minus:" + ",".join(minus)], tabs["full:" + ",".join(cur)])      # positive = full better
            fl = pass_flags(g)
            retained = any(fl.get(m, False) for m in crit[c])
            rnd.append(dict(component=c, minus_key=",".join(minus), full_key=",".join(cur), gains=g, retained=retained,
                            first_gain=float(g.set_index("metric").loc[crit[c][0], "gain"])))
            log(f"  LOO {COMP_NAMES[c]} (criterion {crit[c]}): {fmt(g)} -> {'retained' if retained else 'REDUNDANT'}")
        loo_rounds.append(dict(components=list(cur), rows=rnd))
        failing = [r for r in rnd if not r["retained"]]
        if not failing or len(cur) == 1:
            break
        drop = min(failing, key=lambda r: r["first_gain"])["component"]
        log(f"  backward elimination: removing {COMP_NAMES[drop]} (smallest point gain among the failing)")
        cur = [c for c in cur if c != drop]
    final_comps = cur
    final_key = "full:" + ",".join(final_comps)
    final_cfg = build(final_comps, lam_x)
    final_p = run("final", final_cfg, cfg["refit_every"])[0]       # cached: same config as the full model of the last round
    tabs["final"] = tabs[final_key]
    log(f"FINAL components: {[COMP_NAMES[c] for c in final_comps]}; lam_x {lam_x}; config {final_cfg}")
    if not args.quick:
        # reproduction of stored runs from the leave-one-out runs that reduce to them
        stored = {"reverse alone (M11, frozen setting)": ("m11_latent_111_reverse", ["reverse"]),
                  "pace_hist alone (M09, tuned lam_x)": ("m09_hist_latent_111", ["pace_hist"])}
        lb3 = pd.read_csv(ROOT / cfg["leaderboard"])
        for name, (lab, cs) in stored.items():
            if any(c == "car_penalty" for c in cs) or (set(cs) - set(comps)):
                continue
            row = lb3[(lb3["label"] == lab) & (lb3["split"] == "dev")].iloc[-1]
            sp = pd.read_parquet(ROOT / row["run_dir"] / "predictions.parquet")
            cp, _ = run(f"re-run for reproduction: {name}", build(cs, lam_x_m09 if "pace_hist" in cs else lam_x), cfg["refit_every"])
            ds, ne = U.score_diff(cp, sp)
            rep[name] = dict(max_abs_diff_scores=ds, entries=ne, stored_run=row["run_id"])
            log(f"REPRODUCTION {name}: max |score diff| vs stored run {ds:.3e} ({ne} entries)")

    # final vs references, all section 5 metrics
    g_final = gain(tabs["latent_111"], tabs["final"])
    g_grid = gain(tabs["grid"], tabs["final"])
    g_ref_grid = gain(tabs["grid"], tabs["latent_111"])
    by_season = pd.concat([gain(tabs["latent_111"][tabs["latent_111"]["season"] == s], tabs["final"][tabs["final"]["season"] == s]).assign(season=s)
                           for s in (2021, 2022, 2023)], ignore_index=True)
    means = pd.DataFrame({k: v.reindex(columns=SHOW + ["within1", "within2", "within3"]).mean() for k, v in
                          {"grid order": tabs["grid"], "latent_111": tabs["latent_111"], "final": tabs["final"]}.items()}).T
    print("\nMEANS (outcome-free tie-break)\n" + means.round(4).to_string())
    print("\nFINAL vs latent_111 (positive = final better)\n" + g_final.round(4).to_string(index=False))
    print("\nFINAL vs grid order (positive = final better)\n" + g_grid.round(4).to_string(index=False))
    print("\nBY SEASON, final vs latent_111\n" + by_season.round(4).to_string(index=False))

    # ---------------------------------------------------------------- 4. simulator re-fit on dev
    log("STEP 4: simulator S3 re-fitted on dev")
    dev_e = entries0[entries0["season"].between(*SPLITS["dev"])]
    consts = dict(stick_ref=float(dev_e["stick"].mean()), vol_mu=float(dev_e["circ_vol"].mean()), vol_sd=float(dev_e["circ_vol"].std()))
    for k in consts:
        assert abs(consts[k] - frozen2["M08"]["S3"][k]) < 1e-12, k
    pr_dnf, _, _ = D.walk_dnf(entries0, "dev", frozen2["M07"]["C"])
    lbd = pd.read_csv(D.LB_DNF)
    stored_dn = pd.read_parquet(ROOT / lbd[lbd["split"] == "dev"].iloc[-1]["run_dir"] / "predictions.parquet")[["race_id", "driverId", "p_dnf"]]
    chk = pr_dnf.merge(stored_dn, on=["race_id", "driverId"], suffixes=("_n", "_o"))
    rep["M07 p_dnf recomputed vs stored plan-2 dev run"] = dict(max_abs_diff=float((chk["p_dnf_n"] - chk["p_dnf_o"]).abs().max()), entries=len(chk))
    log(f"REPRODUCTION M07 p_dnf: max |diff| {rep['M07 p_dnf recomputed vs stored plan-2 dev run']['max_abs_diff']:.3e}")
    dn = pr_dnf[["race_id", "driverId", "p_dnf"]]
    cols = ["race_id", "driverId", "grid_norm", "stick", "circ_vol", "rain_fc"]
    sim_cfg = cfg["simulator"]
    sims, thetas = {}, {}
    theta_frozen = {k: frozen2["M08"]["S3"][k] for k in S.DEFAULT}
    for name, pp in (("latent_111", ref_p), ("final", final_p)):
        fr = pp.merge(dn, on=["race_id", "driverId"]).merge(entries0[cols], on=["race_id", "driverId"])
        assert len(fr) == len(pp)
        races = S.prepare(fr, consts)
        t = time.time()
        theta, fit_rps = S.fit_theta(races, "S3", k=int(sim_cfg["fit_k"]))
        thetas[name] = dict(theta=theta, fit_rps=float(fit_rps))
        keys = [r["g"]["driverId"].to_numpy() for r in races]
        mats = DEC.simulate_matrices(races, theta, "S3", int(sim_cfg["draws"]), int(sim_cfg["seed"]))
        sims[name] = (races, mats, keys) + sim_tables(races, mats, keys)
        gs = CAL.grid_slot_calibration(mats, [r["g"]["grid"].to_numpy(float) for r in races], [r["pos"] for r in races], keys)
        thetas[name]["grid_slot"] = dict(mean_pred=gs["mean_pred"], realised=gs["realised"])
        log(f"  S3 re-fit on {name} scores: theta {theta} (fit RPS {fit_rps:.5f}, {time.time() - t:.0f}s); "
            f"evaluated RPS {sims[name][4]['rps'].mean():.5f}, logloss_points {sims[name][4]['logloss_points'].mean():.4f}, "
            f"exact {sims[name][4]['exact'].mean():.4f}, grid-slot predicted {gs['mean_pred']:.3f} vs realised {gs['realised']:.3f}")
    n_evals["S3_refit_each"] = 1 + 2 * sum(len(S.GRID[n]) for n in S.ACTIVE["S3"])
    # descriptive: final score through the unchanged plan-2 S3 parameters; plan-2 S3 on the M06 H3 score (stored run)
    races_f = sims["final"][0]
    mats_fz = DEC.simulate_matrices(races_f, theta_frozen, "S3", int(sim_cfg["draws"]), int(sim_cfg["seed"]))
    p_fz, t_fz = sim_tables(races_f, mats_fz, sims["final"][2])
    l2 = pd.read_csv(Q.LB2)
    row = l2[(l2["label"] == "m08_S3") & (l2["split"] == "dev")].iloc[-1]
    p2_pred = pd.read_parquet(ROOT / row["run_dir"] / "predictions.parquet")
    p2_per = pd.read_csv(ROOT / row["run_dir"] / "per_race_metrics.csv")
    t_p2 = A.race_table(U.outcome_free(p2_pred, key), rps=dict(zip(p2_per["race_id"], p2_per["rps"])))
    tabs_s = {"latent_111 -> S3 refit": sims["latent_111"][4], "final -> S3 refit": sims["final"][4],
              "final -> plan-2 S3 theta (descriptive)": t_fz, "plan-2 S3 on M06 H3 score (stored, descriptive)": t_p2}
    g_sim = gain(tabs_s["latent_111 -> S3 refit"], tabs_s["final -> S3 refit"])
    g_sim_fz = gain(tabs_s["final -> plan-2 S3 theta (descriptive)"], tabs_s["final -> S3 refit"])
    g_sim_p2 = gain(tabs_s["plan-2 S3 on M06 H3 score (stored, descriptive)"], tabs_s["final -> S3 refit"])
    g_sim_grid = gain(tabs["grid"], tabs_s["final -> S3 refit"], [m for m in A.METRICS if m not in ("rps", "logloss_points")])
    smeans = pd.DataFrame({k: v.reindex(columns=SHOW + ["rps"]).mean() for k, v in tabs_s.items()}).T
    print("\nSIMULATOR MEANS\n" + smeans.round(4).to_string())
    print("\nFINAL -> S3 refit vs latent_111 -> S3 refit (positive = final better)\n" + g_sim.round(5).to_string(index=False))
    print("\nFINAL -> S3 refit vs final -> plan-2 theta\n" + g_sim_fz[g_sim_fz.metric.isin(["rps", "logloss_points", "spearman", "exact"])].round(5).to_string(index=False))
    print("\nFINAL -> S3 refit vs plan-2 S3 on M06 H3 score (stored)\n" + g_sim_p2[g_sim_p2.metric.isin(["rps", "logloss_points", "spearman", "exact"])].round(5).to_string(index=False))
    print("\nFINAL -> S3 refit order vs grid order, exact\n" + g_sim_grid[g_sim_grid.metric.str.startswith("exact")].round(4).to_string(index=False))

    # ---------------------------------------------------------------- 4b. predict.py consistency on a few dev races, H1 descriptive
    freeze_cfg = build_frozen_config(cfg, spec, final_cfg, final_comps, best_lam, lam_x, thetas["final"], consts, frozen2, base)
    pchk = {}
    if not args.quick:
        ids = [int(dev_ids[0]), int(dev_ids[len(dev_ids) // 2]), int(dev_ids[-1])][:int(cfg["predict_check_races"])]
        for r in ids:
            out = F.predict_race(entries0, r, freeze_cfg, "H2", with_sim=False, pace=pace)
            ref_sc = final_p[final_p["race_id"] == r][["driverId", "score"]].merge(out[["driverId", "score"]], on="driverId", suffixes=("_h", "_p"))
            rho = float(pd.Series(ref_sc["score_h"]).rank().corr(pd.Series(ref_sc["score_p"]).rank()))
            pchk[r] = dict(max_abs_score_diff=float((ref_sc["score_h"] - ref_sc["score_p"]).abs().max()), rank_corr=rho)
            log(f"predict.py vs harness, race {r}: max |score diff| {pchk[r]['max_abs_score_diff']:.2e}, rank corr {rho:.6f}")
    h1_runs = {}
    if not args.quick:
        h1 = dict(frozen2["H1"]["111"])
        h1_with = dict(h1, reverse=True, x_cols=list(h1["x_cols"]) + [F.PACE_COL])
        for lab, c in (("H1 plan-2 frozen 111 (2023)", h1), ("H1 + final components (2023, descriptive)", h1_with)):
            h1_runs[lab] = (c,) + run(lab, c, cfg["refit_every"], cfg["h1_descriptive_split"])
        g_h1 = gain(tab(h1_runs["H1 plan-2 frozen 111 (2023)"][1]), tab(h1_runs["H1 + final components (2023, descriptive)"][1]))
        print("\nH1 on 2023, components on vs plan-2 H1 (descriptive, 22 races)\n" + g_h1.round(4).to_string(index=False))

    if args.quick:
        log("quick run: nothing written")
        return

    # ---------------------------------------------------------------- 5. write everything
    RES.mkdir(parents=True, exist_ok=True)
    stamp = U.code_stamp(USED)
    tmp_lb = RES / "_lb_tmp.csv"
    tmp_lb.unlink(missing_ok=True)
    run_ids: dict[str, str] = {}

    def save(label: str, key_: str, c: dict, p: pd.DataFrame, per: pd.DataFrame, extra: dict, split: str = "dev") -> None:
        rid = save_run(MDIR, label, dict(c, **extra), p, per, split, "P3", "111", leaderboard=tmp_lb)
        pth = MDIR / "results" / rid / "config.yaml"
        cc = load_yaml(pth)
        cc["git_commit"] = stamp
        save_yaml(cc, pth)
        run_ids[key_] = rid

    status = dict(status="plan-3 R14 dev run; not scored on any test block")
    save("p3_ref_latent_111", "ref", ref_cfg, ref_p, ref_m, dict(component="latent_111", **status))
    if carpen_active:
        save(f"p3_car_penalty_lam{int(best_lam)}_alone", "car_penalty", carpen_cfg, carpen_p, carpen_m, dict(component="car_penalty_retune", **status))
    else:
        run_ids["car_penalty"] = run_ids["ref"]
    for rnd_i, rnd in enumerate(loo_rounds):
        cur_c = rnd["components"]
        fp, fm = run("", build(cur_c, lam_x), cfg["refit_every"])
        label = "p3_final_model" if rnd_i == len(loo_rounds) - 1 else f"p3_candidate_round{rnd_i + 1}"
        save(label, "full:" + ",".join(cur_c), build(cur_c, lam_x), fp, fm,
             dict(component="+".join(COMP_NAMES[c] for c in cur_c), final=(rnd_i == len(loo_rounds) - 1),
                  lam_x_pace_hist=lam_x, frozen_hyperparameters="plan-1 + dev-tuned component settings (see run.py)", **status))
        for r in rnd["rows"]:
            mc = build([x for x in cur_c if x != r["component"]], lam_x)
            mp, mm = run("", mc, cfg["refit_every"])
            lab = f"p3_loo_r{rnd_i + 1}_minus_{r['component']}"
            save(lab, f"{lab}", mc, mp, mm, dict(component=f"full minus {COMP_NAMES[r['component']]}", **status))
    for name, lab in (("latent_111", "p3_S3refit_latent_111"), ("final", "p3_S3refit_final")):
        races_, mats_, keys_, preds_, tab_ = sims[name]
        save(lab, lab, dict(kind="m08", cell="S3", theta=thetas[name]["theta"], consts=consts, order_model=name, draws=int(sim_cfg["draws"]),
                            seed=int(sim_cfg["seed"]), retirement_model="M07 dev walk-forward (plan-2 C)"), preds_, tab_,
             dict(component=f"S3 re-fitted on {name} scores", **status))
    for lab_, (c_, p_, m_) in h1_runs.items():
        save("p3_" + ("H1_plan2_ref_2023" if "plan-2" in lab_ else "H1_with_components_2023_diagnostic"), lab_, c_, p_, m_,
             dict(component=lab_, **status, note="descriptive, 22 races of 2023, no decision weight", scored_split=cfg["h1_descriptive_split"]), "dev2023")
    new_lb = pd.read_csv(tmp_lb)
    tmp_lb.unlink()

    # tables
    tune_car.to_csv(RES / "tuning_car_penalty.csv", index=False)
    tune_lx.to_csv(RES / "tuning_lam_x_combination.csv", index=False)
    g_car_season.to_csv(RES / "car_penalty_by_season.csv", index=False)
    g_final.to_csv(RES / "final_vs_latent_111.csv", index=False)
    g_grid.to_csv(RES / "final_vs_grid.csv", index=False)
    g_ref_grid.to_csv(RES / "latent_111_vs_grid.csv", index=False)
    by_season.to_csv(RES / "final_vs_latent_111_by_season.csv", index=False)
    means.to_csv(RES / "means.csv")
    smeans.to_csv(RES / "simulator_means.csv")
    pd.concat([g_sim.assign(contrast="final->S3refit minus latent_111->S3refit"), g_sim_fz.assign(contrast="final S3refit minus plan-2 theta"),
               g_sim_p2.assign(contrast="final S3refit minus plan-2 S3 on M06 H3 (stored)"),
               g_sim_grid.assign(contrast="final S3refit order vs grid order")]).to_csv(RES / "simulator_paired.csv", index=False)
    loo_tab = pd.concat([r["gains"].assign(round=i + 1, component=COMP_NAMES[r["component"]], retained=r["retained"], full=",".join(rd["components"]))
                         for i, rd in enumerate(loo_rounds) for r in rd["rows"]], ignore_index=True)
    loo_tab.to_csv(RES / "leave_one_out.csv", index=False)
    (RES / "reproduction_check.json").write_text(json.dumps(dict(reproduction=rep, predict_vs_harness=pchk, budget_configurations=n_evals), indent=2), encoding="utf-8")

    # attribution rows (append only)
    rows = []
    r_ = A.attribution_row("P3_car_penalty_retune", "latent_111", g_car, "spearman", run_ids["car_penalty"], len(tabs["car_penalty"]), declared="yes")
    r_["verdict"] = (f"lam_car {best_lam:g} (lam_a {base['lam']:g}): " + ("component inactive (pick = reference); " if not carpen_active else "")
                     + "; ".join(f"{k}: {'gain' if v else 'none'}" for k, v in car_flags.items()) + "; dev only")
    rows.append(r_)
    for i, rd in enumerate(loo_rounds):
        for r in rd["rows"]:
            c = r["component"]
            rr = A.attribution_row(LOO_NAMES[c], f"leave-one-out: full ({'+'.join(COMP_NAMES[x] for x in rd['components'])}) vs full minus {COMP_NAMES[c]}",
                                   r["gains"], crit[c][0], run_ids[f"full:{','.join(rd['components'])}"], len(tabs["latent_111"]), declared="yes")
            rr["loo_gain"] = fmt(r["gains"])
            rr["verdict_metric"] = "; ".join(crit[c]) + " (criterion on which it passed alone)"
            rr["verdict"] = ("retained" if r["retained"] else "redundant, removed") + f" (round {i + 1}); dev only"
            rows.append(rr)
    rf = A.attribution_row("P3_final", "latent_111", g_final, "spearman", run_ids[final_key], len(tabs["final"]), declared="yes")
    rf["verdict_metric"] = "spearman; mae; exact (per-metric)"
    fl = pass_flags(g_final)
    rf["verdict"] = "; ".join(f"{k}: {'gain' if v else 'none'}" for k, v in fl.items()) + "; dev only"
    rows.append(rf)
    rg = A.attribution_row("P3_final", "grid", g_grid, "exact", run_ids[final_key], len(tabs["final"]), declared="yes")
    gi = g_grid.set_index("metric")
    rg["verdict"] = A.verdict(gi.loc["exact"]) + "; dev only (grid order is the exact-rate baseline)"
    rows.append(rg)
    rs = A.attribution_row("P3_final_S3refit", "latent_111_S3refit", g_sim, "rps", run_ids["p3_S3refit_final"], len(sims["final"][4]), declared="yes")
    rs["verdict"] = f"RPS: {A.verdict(g_sim.set_index('metric').loc['rps'])}; logloss_points: {A.verdict(g_sim.set_index('metric').loc['logloss_points'])}; dev only"
    rows.append(rs)
    rs2 = A.attribution_row("P3_final_S3refit", "plan-2 S3 on M06 H3 score (stored run)", g_sim_p2, "rps", run_ids["p3_S3refit_final"], len(sims["final"][4]), declared="no")
    rs2["verdict"] = f"descriptive only (H2 final vs H3 information); RPS: {A.verdict(g_sim_p2.set_index('metric').loc['rps'])}; dev only"
    rows.append(rs2)
    append_text_rows(pd.DataFrame(rows), ROOT / cfg["attribution"])
    U.append_leaderboard(new_lb, ROOT / cfg["leaderboard"])

    # frozen config and FREEZE.md (deterministic content: no timestamps other than the date fixed in config.yaml)
    (MDIR / "frozen_config.yaml").write_bytes(yaml.safe_dump(freeze_cfg, sort_keys=False).encode("utf-8"))      # LF line endings
    sha = F.sha256_file(MDIR / "frozen_config.yaml")
    head = _git_hash()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=ROOT).strip()
    write_freeze_md(MDIR / "FREEZE.md", cfg, sha, head, bool(dirty), run_ids[final_key], final_comps)
    log(f"frozen_config.yaml sha256 {sha}")
    log(f"verdicts: {[(r['component'], r['reference'][:40], r['verdict']) for r in rows]}")
    log(f"configurations: {n_evals}; done in {time.time() - t0:.0f}s; code stamp {stamp}")


def build_frozen_config(cfg: dict, spec: dict, final_cfg: dict, comps: list[str], best_lam: float, lam_x: float, sim_fit: dict,
                        consts: dict, frozen2: dict, base: dict) -> dict:
    fz = cfg["freeze"]
    theta = {k: float(v) for k, v in sim_fit["theta"].items()}
    return {
        "frozen": {"plan": "PLAN3 R14", "date": fz["date"], "last_race_included": dict(fz["last_race_included"]),
                   "note": "Hyper-parameters tuned on 2021-2023 only. The model is re-fitted before every race on all data up to that race; "
                           "nothing here is re-tuned on 2024-2026."},
        "components": {"included": [COMP_NAMES[c] for c in comps],
                       "not_included": list(cfg["not_included"]),
                       "decoder": "D0 (sort by -E[position] of the simulator; M17 found none better)",
                       "ensemble": "none (M05 not included)", "rolling_retune": "no (A4: null, hyper-parameters stay as tuned on 2021-2023)"},
        "walk_forward": {"refit_every": 1, "n_warm": 15, "tie_break": cfg["tie_break"]},
        "pace_feature": {k: (float(v) if k == "scale_sd" else v) for k, v in spec.items()},
        "retirement_model": {"kind": "M07 logistic regression, refit on all earlier races", "C": float(frozen2["M07"]["C"]), "features": list(D.FEATURES)},
        "horizons": {
            "H2": {"description": "after qualifying: final model (components above), simulator S3 re-fitted on dev, decoder D0",
                   "entries": "pipeline2.load_all() (plan-2 entries)", "uses_pace_hist": "pace_hist" in comps,
                   "model": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in {"kind": "latent", "cell": base["cell"], **{k: final_cfg[k] for k in final_cfg if k not in ("kind", "cell")}}.items()},
                   "simulator": {"variant": "S3", "draws": int(cfg["simulator"]["draws"]), "seed": int(cfg["simulator"]["seed"]), "theta": theta,
                                 "fit_rps_dev": float(sim_fit["fit_rps"]), **{k: float(v) for k, v in consts.items()},
                                 "fitted_on": "dev 2021-2023, on the final model's walk-forward scores"}},
            "H1": {"description": "before qualifying (practice only): the plan-2 frozen H1 latent_111 model, unchanged; the final components "
                                  "were not tested at H1. No simulator is fitted at H1.",
                   "entries": "pipeline2.load_all() (plan-2 entries)", "uses_pace_hist": False,
                   "model": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in frozen2["H1"]["111"].items()}},
        },
        "plan2_frozen_hash_inherited": str(frozen2.get("hash")),
    }


def write_freeze_md(path: Path, cfg: dict, sha: str, head: str, dirty: bool, final_run_id: str, comps: list[str]) -> None:
    fz = cfg["freeze"]
    lr = fz["last_race_included"]
    txt = f"""# Plan-3 freeze (R14)

Date: {fz['date']}

SHA-256 of frozen_config.yaml: `{sha}`
(hash of the file with CRLF turned into LF, `pitlane.plan3.final.sha256_file`)

- Last race included in the data: {lr['season']} round {lr['round']} (race_id 271). The fresh holdout is every race run after it.
- Components in the final pre-race model (H2): {', '.join(COMP_NAMES[c] for c in comps)}; simulator S3 re-fitted on dev; decoder D0.
  H1 is the plan-2 frozen H1 model unchanged.
- Dev run that produced the final numbers: `{final_run_id}` (methodologies/P3_final/results/).
- Git state when the config was written: base commit `{head}`{' with uncommitted changes (nothing is committed yet; the main session commits and tags after review)' if dirty else ''}.
- No git tag has been created by the run script. After review the main session creates the tag; no file in this folder named
  `frozen_config.yaml` may be edited after it (`tests/test_plan3_final.py` checks the hash above against the file).
- Hyper-parameters were tuned on 2021-2023 only. The re-fit on data through {lr['season']} round {lr['round']} happens at prediction time
  (every race is predicted from a model fitted on all earlier races); nothing is re-tuned on 2024-2026.
- Predict a race: `python methodologies/P3_final/predict.py --season S --round R` (refuses to run if the hash above does not match).
"""
    path.write_bytes(txt.encode("utf-8"))


if __name__ == "__main__":
    main()
