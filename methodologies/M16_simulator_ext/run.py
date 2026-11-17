"""M16 simulator extensions S4-S7 on the plan-2 simulator S3, development block only (PLAN3 R12).

    python methodologies/M16_simulator_ext/run.py [--config methodologies/M16_simulator_ext/config.yaml] [--check]

--check: acceptance (all new steps off reproduces the frozen S3 matrices bitwise; S7 neutral reproduces the stored M06
scores); writes nothing. Otherwise: tunes each step alone (<= 30 configurations), evaluates with the plan-2 streams, writes
run folders, and appends leaderboard and attribution rows as new text lines (existing lines are never rewritten).
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pitlane.config import METHODS, ROOT, load_yaml, save_yaml
from pitlane.evaluation.harness import walk_forward
from pitlane.evaluation.metrics import pred_rank, rps_score
from pitlane.plan2 import dnf as D
from pitlane.plan2 import pipeline2 as Q
from pitlane.plan2 import simulate as S
from pitlane.plan3 import attribution as A
from pitlane.plan3 import calibration as CAL
from pitlane.plan3 import decode as DEC
from pitlane.plan3 import simext as SX
from pitlane.plan3 import simext_laplace as SL
from pitlane.plan3.ar_latent import LatentModelAR

OUT = METHODS / "M16_simulator_ext"
JOINT = "logloss_team_points"
A.HIGHER_BETTER[JOINT] = False            # the joint metric is registered here; attribution.py is not edited
PAIRED_METRICS = A.METRICS + [JOINT]
TAB_EXCLUDE = ("race_id", "season", "round", "n")


def _m17():
    spec = importlib.util.spec_from_file_location("m17_run", METHODS / "M17_decoding" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_id() -> str:
    h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=ROOT).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=ROOT).strip()
    return h + ("+dirty" if dirty else "")


def ext_of(step: str, p: dict) -> dict | None:
    if step == "S4":
        return dict(kind="S4", m2=p["m2"], m3=p["m3"], m4=p["m4"])
    if step == "S5":
        return dict(kind="S5", sigma=p["sigma"], tau_team=p["tau_team"], e_exp=p["e_exp"], e_wet=p["e_wet"])
    if step == "S6":
        return dict(kind="S6", c=p["c"], c_early=p["c_early"])
    return None


# ------------------------------------------------------------------------------------ inputs
def sim_inputs_from(preds: pd.DataFrame, entries: pd.DataFrame, split: str) -> pd.DataFrame:
    """Same join as M17's sim_inputs, for a score table that is not a stored run (S7)."""
    lb = pd.read_csv(D.LB_DNF)
    row = lb[lb["split"] == split].iloc[-1]
    dn = pd.read_parquet(ROOT / row["run_dir"] / "predictions.parquet")[["race_id", "driverId", "p_dnf"]]
    cols = ["race_id", "driverId", "grid_norm", "stick", "circ_vol", "rain_fc"]
    return preds.merge(dn, on=["race_id", "driverId"]).merge(entries[cols], on=["race_id", "driverId"])


def laplace_covs(entries: pd.DataFrame, frozen: dict, cfg: dict):
    """Walk-forward replay with Hessians (about 40 s); cached in data/interim (git-ignored), checked against the
    stored plan-2 scores either way."""
    import pickle
    from pitlane.models.registry import build_model
    cache = ROOT / "data" / "interim" / "m16_laplace_dev.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    model = build_model(frozen["M06L"]["111"], entries)
    us, covs = SL.walk_with_cov(entries, model, cfg["split"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps((us, covs)))
    return us, covs


# ------------------------------------------------------------------------------------ tables
def table_for(m17, races, mats, kind: str, pairs: list | None = None) -> tuple[list, pd.DataFrame, pd.DataFrame]:
    """(utilities, predictions, per-race table) of one decoder on one set of matrices; the joint team metric is added
    when the per-race team-pair probabilities are given."""
    keys = m17.driver_keys(races)
    if kind == "D0":
        util = [DEC.break_ties(-DEC.expected_position(P), k) for P, k in zip(mats, keys)]
    elif kind == "D2":
        util = [DEC.decode_d2(P) for P in mats]
    else:
        util = [DEC.grid_utility(r["g"]["grid"].to_numpy(float), k) for r, k in zip(races, keys)]
    preds = m17.build_preds(races, mats, util)
    rps = {r["race_id"]: rps_score(P, r["pos"]) for r, P in zip(races, mats)}
    tab = A.race_table(preds, rps=rps)
    if pairs is not None:
        ll = {r["race_id"]: SX.team_logloss(p, SX.team_pair_outcomes(r["pos"], r["team"])) for r, p in zip(races, pairs)}
        tab[JOINT] = tab["race_id"].map(ll)
    return util, preds, tab


def save(label, cfg, extra, preds, per, stamp, gid, mats, races) -> tuple[str, dict]:
    run_id = f"{stamp}_m16_{label}_dev"
    out = OUT / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    save_yaml({**cfg, "label": label, **extra, "git_commit": gid, "split": "dev"}, out / "config.yaml")
    summ = {k: float(per[k].mean()) for k in per.columns if k not in TAB_EXCLUDE}
    summ["n_races"] = int(len(per))
    (out / "metrics.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    per.to_csv(out / "per_race_metrics.csv", index=False)
    preds.to_parquet(out / "predictions.parquet", index=False)
    np.savez_compressed(out / "position_matrices.npz", **{str(r["race_id"]): P for r, P in zip(races, mats)})
    return run_id, {"method": "M16", "cell": "S3", "label": f"m16_{label}", "split": "dev", "run_id": run_id,
                    "run_dir": str(out.relative_to(ROOT)), **summ}


def append_rows(path: Path, df: pd.DataFrame) -> None:
    """Append rows as new text lines in the existing column order; earlier lines stay byte-identical."""
    cols = list(pd.read_csv(path, nrows=0).columns)
    txt = df.reindex(columns=cols).to_csv(header=False, index=False, lineterminator="\n")
    with open(path, "r+b") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(size - 1, 0))
        needs_nl = size > 0 and f.read(1) != b"\n"
    with open(path, "a", newline="", encoding="utf-8") as f:
        f.write(("\n" if needs_nl else "") + txt)


# ------------------------------------------------------------------------------------ steps
def tune_step(step: str, cfg: dict, races, theta: dict, sw: dict) -> tuple[dict, pd.DataFrame]:
    sc = cfg[step]

    def score(p: dict) -> float:
        return SX.mean_rps(races, {**theta, "tau": p["tau"]}, ext_of(step, p), sw, cfg["tune_draws"])

    best, tab = SX.coordinate_search(score, sc["start"], sc["grids"], sc["order"], cfg["max_configs"], cfg["passes"])
    assert len(tab) <= cfg["max_configs"]
    return best, tab


def acceptance(races, theta, cfg, sw, ref_mats) -> dict:
    out = {}
    off = SX.simulate_matrices(races, theta, None, sw, cfg["draws"], cfg["seed"])
    out["ext_off_vs_S3"] = max(float(np.abs(a - b).max()) for a, b in zip(ref_mats, off))
    c0 = SX.simulate_matrices(races, theta, dict(kind="S6", c=0.0, c_early=0.0), sw, cfg["draws"], cfg["seed"])
    out["S6_c0_vs_S3"] = max(float(np.abs(a - b).max()) for a, b in zip(ref_mats, c0))
    out["S6_c0_bitwise"] = all(np.array_equal(a, b) for a, b in zip(ref_mats, c0))
    out["ext_off_bitwise"] = all(np.array_equal(a, b) for a, b in zip(ref_mats, off))
    return out


def noise_table(label: str, races_c, theta_c, ext_c, races_r, theta_r, m17, cfg, sw, seeds) -> pd.DataFrame:
    """Main-metric means of the reference and the candidate for several simulation seeds (D0, driverId tie-break)."""
    rows = []
    for sd in seeds:
        res = {}
        for nm, (rs, th, ex) in {"ref": (races_r, theta_r, None), "cand": (races_c, theta_c, ext_c)}.items():
            mats, prs = SX.simulate_matrices(rs, th, ex, sw, cfg["draws"], sd, with_pairs=True)
            tab = table_for(m17, rs, mats, "D0", prs)[2]
            res[nm] = tab[["rps", "logloss_points", "spearman", "mae", "exact", JOINT]].mean()
        for m in res["ref"].index:
            rows.append(dict(candidate=label, seed=sd, metric=m, ref=res["ref"][m], cand=res["cand"][m],
                             diff=res["cand"][m] - res["ref"][m]))
    return pd.DataFrame(rows)


def s7_search(cfg, entries, frozen):
    p = {k: v for k, v in frozen["M06L"]["111"].items() if k not in ("kind", "cell")}
    make = lambda phi, m: LatentModelAR(entries, dynamic=True, decomp=True, ranking=True, rho=phi, rule_mult=m, **p)
    rows = []
    for phi, m in itertools.product(cfg["S7"]["phi_grid"], cfg["S7"]["rule_mult_grid"]):
        _, per = walk_forward(entries, make(phi, m), "dev", refit_every=cfg["S7"]["latent_refit_every"], n_samples=200)
        rows.append(dict(phi=phi, rule_mult=m, spearman=float(per["spearman"].mean())))
    return make, pd.DataFrame(rows)


def descriptive(cfg: dict, races, theta, sw, m17, tabs_ref) -> None:
    """Post-hoc, NOT fitted: S5 at fixed team-shock sizes and S6 at fixed multipliers (e.g. the literal Laplace interval,
    c = 1), against S3 D0. Paired CI at seed 0 plus the mean difference over the noise seeds. No leaderboard or
    attribution rows."""
    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    nb, seed = cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"]
    base5 = {k: cfg["S5"]["start"][k] for k in ("sigma", "tau_team", "e_exp", "e_wet")}
    cases = [(f"S5 tau_team={v}", {**theta}, dict(kind="S5", **{**base5, "tau_team": v})) for v in cfg["descriptive"]["S5_tau_team"]]
    cases += [(f"S6 c={v}", {**theta}, dict(kind="S6", c=v, c_early=0.0)) for v in cfg["descriptive"]["S6_c"]]
    tab_ref = tabs_ref[0]
    rows = []
    for name, th, ext in cases:
        mats, prs = SX.simulate_matrices(races, th, ext, sw, cfg["draws"], cfg["seed"], with_pairs=True)
        tb = table_for(m17, races, mats, "D0", prs)[2]
        g = A.paired_gain(tab_ref, tb, ["rps", "logloss_points", JOINT, "spearman", "mae", "exact"], n_boot=nb, seed=seed)
        nz = noise_table(name, races, th, ext, races, theta, m17, cfg, sw, [cfg["seed"]] + list(cfg["noise_seeds"]))
        nzs = nz.groupby("metric")["diff"].agg(["mean", "std"])
        for _, x in g.iterrows():
            rows.append(dict(case=name, metric=x["metric"], gain_seed0=x["gain"], lo=x["lo"], hi=x["hi"], n=int(x["n"]),
                             cand_minus_ref_mean_5seeds=float(nzs.loc[x["metric"], "mean"]) if x["metric"] in nzs.index else np.nan,
                             sd_5seeds=float(nzs.loc[x["metric"], "std"]) if x["metric"] in nzs.index else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results" / f"{stamp}_descriptive_dev.csv", index=False)
    print("DESCRIPTIVE (post hoc, not fitted; gain positive = better; cand_minus_ref over 5 seeds: negative = better "
          "for losses)\n", df.round(5).to_string(index=False))


def s7_by_rps(cfg, races, theta, consts, sw, m17, entries, frozen, tab_ref0) -> None:
    """R12 revision: re-score the 25 S7 configurations by the simulator objective (mean RPS, 300 draws, streams 1000+i,
    frozen S3 theta, scores from the refit-every-4 walk used for the Spearman tuning); run the RPS pick as the declared S7."""
    stamp, gid = f"{datetime.now():%Y%m%d-%H%M}", git_id()
    nb, seed = cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"]
    p = {k: v for k, v in frozen["M06L"]["111"].items() if k not in ("kind", "cell")}
    make = lambda phi, m: LatentModelAR(entries, dynamic=True, decomp=True, ranking=True, rho=phi, rule_mult=m, **p)
    rows = []
    for phi, m in itertools.product(cfg["S7"]["phi_grid"], cfg["S7"]["rule_mult_grid"]):
        preds, per = walk_forward(entries, make(phi, m), "dev", refit_every=cfg["S7"]["latent_refit_every"], n_samples=200)
        rc = SX.add_extras(S.prepare(sim_inputs_from(preds, entries, "dev"), consts), entries)
        rows.append(dict(phi=phi, rule_mult=m, spearman=float(per["spearman"].mean()),
                         rps=SX.mean_rps(rc, theta, None, sw, cfg["tune_draws"])))
    t = pd.DataFrame(rows)
    old = pd.read_csv(sorted((OUT / "results").glob("*_tuning_S7_dev.csv"))[0])
    assert np.abs(old["spearman"].to_numpy() - t["spearman"].to_numpy()).max() == 0.0
    t.to_csv(OUT / "results" / f"{stamp}_tuning_S7_rps_dev.csv", index=False)
    pick = t.loc[t["rps"].idxmin()]
    sp = t.loc[t["spearman"].idxmax()]
    print(f"S7 re-scored by RPS ({len(t)} configurations, the same as the Spearman tuning):\n{t.round(5).to_string(index=False)}\n"
          f"RPS pick: phi {pick['phi']}, rule_mult {pick['rule_mult']} (RPS {pick['rps']:.5f}); "
          f"Spearman pick: phi {sp['phi']}, rule_mult {sp['rule_mult']}")
    cand = {"s7": (float(sp["phi"]), float(sp["rule_mult"])), "s7rps": (float(pick["phi"]), float(pick["rule_mult"]))}
    same = cand["s7rps"] == cand["s7"]
    if same:
        print("RPS picks the same configuration as Spearman: nothing appended")
    stored_per = A.race_table(Q.load_run_preds(cfg["m06_label"], "dev").assign(p_points=0.5), "score")
    row1 = pd.read_csv(ROOT / "results" / "leaderboard.csv").query("label == 'latent_111' and split == 'dev'").iloc[-1]
    p1 = A.race_table(pd.read_parquet(ROOT / row1["run_dir"] / "predictions.parquet").assign(p_points=0.5), "score")
    out, tau_rows, lat_rows = {}, [], []
    for name, (phi, m) in cand.items():
        if name == "s7rps" and same:
            continue
        preds7, _ = walk_forward(entries, make(phi, m), "dev", refit_every=1, posdist=True)
        races7 = SX.add_extras(S.prepare(sim_inputs_from(preds7, entries, "dev"), consts), entries)
        t7 = A.race_table(preds7.assign(p_points=0.5), "score")
        for rn, ref in (("m06_L_r4_111 (S3 input)", stored_per), ("latent_111 (plan 1)", p1)):
            g = A.paired_gain(ref, t7, ["spearman", "spearman_fin", "mae"], n_boot=nb, seed=seed)
            for _, x in g.iterrows():
                lat_rows.append(dict(candidate=name, phi=phi, rule_mult=m, reference=rn, metric=x["metric"], ref_mean=x["ref_mean"],
                                     cand_mean=x["cand_mean"], gain=x["gain"], lo=x["lo"], hi=x["hi"], n=int(x["n"])))
        for tau in cfg["S7"]["tau_grid"]:
            tau_rows.append(dict(candidate=name, phi=phi, rule_mult=m, tau=tau,
                                 rps=SX.mean_rps(races7, {**theta, "tau": tau}, None, sw, cfg["tune_draws"])))
        out[name] = (phi, m, races7)
    pd.DataFrame(tau_rows).to_csv(OUT / "results" / f"{stamp}_s7_tau_refit_dev.csv", index=False)
    lat = pd.DataFrame(lat_rows)
    lat.to_csv(OUT / "results" / f"{stamp}_s7_latent_score_rho_dev.csv", index=False)
    print("\nS7 TAU RE-FIT (RPS)\n", pd.DataFrame(tau_rows).round(5).to_string(index=False))
    print("\nDETERMINISTIC LATENT SCORE (no simulator; gain positive = better)\n", lat.round(4).to_string(index=False))
    if "s7rps" not in out:
        return
    phi, m, races7 = out["s7rps"]
    mats, prs = SX.simulate_matrices(races7, theta, None, sw, cfg["draws"], cfg["seed"], with_pairs=True)
    grids_, poss_ = [r["g"]["grid"].to_numpy(float) for r in races], [r["pos"] for r in races]
    keys = m17.driver_keys(races)
    ref_mats = SX.simulate_matrices(races, theta, None, sw, cfg["draws"], cfg["seed"])
    realised = CAL.grid_slot_calibration(ref_mats, grids_, poss_, keys)["realised"]
    gs = CAL.grid_slot_calibration(mats, grids_, poss_, keys)
    ok = bool(gs["mean_pred"] >= realised - cfg["calibration_pass_gap"])
    print(f"S7rps grid-slot calibration: {100 * gs['mean_pred']:.2f}% vs realised {100 * realised:.2f}% -> "
          f"gate {'passed' if ok else 'FAILED; D2 not run'}")
    tabs = {"S3_D0": tab_ref0, "grid": table_for(m17, races, ref_mats, "grid")[2]}
    lb_rows, att, allg = [], [], []
    for kind in ["D0"] + (["D2"] if ok else []):
        u, pr, tb = table_for(m17, races7, mats, kind, prs)
        tabs[f"s7rps_{kind}"] = tb
        rid, row = save(f"s7rps_{kind}", cfg, dict(step="S7_rps", params=dict(phi=phi, rule_mult=m, tau=theta["tau"]),
                                                    theta={k: float(v) for k, v in theta.items()}, calibration_gate_passed=ok,
                                                    selected_by="mean RPS over the 25 S7 configurations"),
                        pr, tb, stamp, gid, mats, races7)
        lb_rows.append(row)
        cname = "M16_S7_rps" + ("_D2" if kind == "D2" else "")
        for rk in ("S3_D0", "grid"):
            g = A.paired_gain(tabs[rk], tb, PAIRED_METRICS, n_boot=nb, seed=seed)
            g.insert(0, "reference", rk)
            g.insert(0, "candidate", cname)
            allg.append(g)
            att.append(A.attribution_row(cname, rk, g[g["metric"].isin(A.METRICS)], "exact" if rk == "grid" else cfg["verdict_metric"],
                                         rid, len(tb)))
    full = pd.concat(allg, ignore_index=True)
    full.to_csv(OUT / "results" / f"{stamp}_s7rps_paired_gains_dev.csv", index=False)
    show = ["spearman", "mae", "exact", "exact_1_3", "exact_4_10", "exact_11p", "rps", "logloss_points", JOINT]
    print("\nS7rps PAIRED GAINS (vs S3 D0 / grid)\n", full[full["metric"].isin(show)].round(5).to_string(index=False))
    seeds = [cfg["seed"]] + list(cfg["noise_seeds"])
    nz = noise_table("S7rps", races7, theta, None, races, theta, m17, cfg, sw, seeds)
    nz.to_csv(OUT / "results" / f"{stamp}_s7rps_mc_noise_dev.csv", index=False)
    print("\nMC NOISE (seeds 0-4)\n", nz.groupby("metric").agg(ref_sd=("ref", lambda x: x.std(ddof=1)), diff_mean=("diff", "mean"),
                                                             diff_sd=("diff", lambda x: x.std(ddof=1))).round(5).to_string())
    meta = pd.DataFrame([dict(race_id=r["race_id"], season=r["season"]) for r in races])
    sub = []
    for sn in sorted(meta["season"].unique()):
        ids = meta[meta["season"] == sn]["race_id"]
        g = A.paired_gain(tab_ref0[tab_ref0["race_id"].isin(ids)], tabs["s7rps_D0"][tabs["s7rps_D0"]["race_id"].isin(ids)],
                          ["rps", "spearman", "logloss_points"], n_boot=nb, seed=seed)
        sub += [dict(season=sn, metric=x["metric"], gain=x["gain"], lo=x["lo"], hi=x["hi"], n=int(x["n"])) for _, x in g.iterrows()]
    sub = pd.DataFrame(sub)
    sub.to_csv(OUT / "results" / f"{stamp}_s7rps_by_season_dev.csv", index=False)
    print("\nBY SEASON\n", sub.round(5).to_string(index=False))
    append_rows(ROOT / cfg["leaderboard"], pd.DataFrame(lb_rows))
    append_rows(ROOT / cfg["attribution"], pd.DataFrame(att))
    print("appended:", [(r["component"], r["reference"], r["verdict"]) for r in att])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(OUT / "config.yaml"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--s7rps", action="store_true",
                    help="R12 revision: re-select S7 by RPS over the same 25 configurations; runs the pick and appends its rows")
    ap.add_argument("--descriptive", action="store_true",
                    help="post-hoc descriptive comparisons at fixed (not fitted) parameters; writes csv only, no leaderboard/attribution rows")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["split"] == "dev"
    pd.set_option("display.width", 220, "display.max_rows", 500, "display.max_columns", 50)
    m17 = _m17()
    frozen = load_yaml(ROOT / cfg["frozen_path"])
    assert frozen["hash"] == cfg["frozen_hash"], "frozen plan-2 config changed"
    theta = {k: frozen["M08"][cfg["simulator"]][k] for k in S.DEFAULT}
    consts = {k: frozen["M08"][cfg["simulator"]][k] for k in ("stick_ref", "vol_mu", "vol_sd")}
    sw = S.SWITCHES[cfg["simulator"]]
    entries = Q.load_all()
    races = SX.add_extras(S.prepare(m17.sim_inputs(entries, "dev", cfg["m06_label"]), consts), entries)

    # S6 inputs: replay of the walk-forward fits with Laplace covariances; scores must equal the stored plan-2 scores
    us, covs = laplace_covs(entries, frozen, cfg)
    SX.attach_cov_factors(races, covs)
    dscore = max(float(np.abs(np.array([us[int(r["race_id"])][1][us[int(r["race_id"])][0].index(d)]
                                         for d in r["g"]["driverId"]]) - r["u"]).max()) for r in races)
    print(f"races {len(races)}, entries {sum(len(r['u']) for r in races)}; replay vs stored plan-2 score: max diff {dscore:.1e}")
    assert dscore == 0.0

    ref_mats, ref_pairs = SX.simulate_matrices(races, theta, None, sw, cfg["draws"], cfg["seed"], with_pairs=True)
    old = DEC.simulate_matrices(races, theta, cfg["simulator"], cfg["draws"], cfg["seed"])
    acc = acceptance(races, theta, cfg, sw, old)
    print("ACCEPTANCE (all new steps off vs plan-2 S3 matrices, max |P difference|):", acc)
    assert max(float(np.abs(a - b).max()) for a, b in zip(ref_mats, old)) == 0.0

    # S7 neutral replay (rule_mult = 1, rho = 0.8) must equal the stored M06 scores
    p_lat = {k: v for k, v in frozen["M06L"]["111"].items() if k not in ("kind", "cell")}
    neutral = LatentModelAR(entries, dynamic=True, decomp=True, ranking=True, **p_lat)
    npreds, _ = walk_forward(entries, neutral, "dev", refit_every=1, posdist=True)
    stored = Q.load_run_preds(cfg["m06_label"], "dev")
    s7_acc = float(np.abs(npreds["score"].to_numpy() - stored["score"].to_numpy()).max())
    print(f"S7 neutral (rho 0.8, rule_mult 1) vs stored M06 scores: max |difference| = {s7_acc:.1e}")

    tab_ref0 = table_for(m17, races, ref_mats, "D0", ref_pairs)[2]
    print("S3 D0 means:", tab_ref0[["rps", "logloss_points", "spearman", "mae", "exact", JOINT]].mean().round(5).to_dict())
    for nm, ext in (("S4 neutral", dict(kind="S4", m2=1.0, m3=1.0, m4=1.0)),
                    ("S5 plain Thurstone", dict(kind="S5", sigma=cfg["S5"]["start"]["sigma"], tau_team=0.0, e_exp=0.0, e_wet=0.0))):
        mm = SX.simulate_matrices(races, theta, ext, sw, cfg["draws"], cfg["seed"])
        print(f"{nm}: max |P - S3 P| {max(float(np.abs(a - b).max()) for a, b in zip(ref_mats, mm)):.3f}, "
              f"mean RPS {np.mean([rps_score(P, r['pos']) for P, r in zip(mm, races)]):.5f} vs S3 "
              f"{np.mean([rps_score(P, r['pos']) for P, r in zip(ref_mats, races)]):.5f}")
    if args.check:
        return
    if args.s7rps:
        s7_by_rps(cfg, races, theta, consts, sw, m17, entries, frozen, tab_ref0)
        return
    if args.descriptive:
        descriptive(cfg, races, theta, sw, m17, tabs_ref=(tab_ref0, ref_mats))
        return

    stamp, gid = f"{datetime.now():%Y%m%d-%H%M}", git_id()
    nb, seed = cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"]
    grids_, poss_ = [r["g"]["grid"].to_numpy(float) for r in races], [r["pos"] for r in races]
    keys = m17.driver_keys(races)
    cal_ref = CAL.grid_slot_calibration(ref_mats, grids_, poss_, keys)
    realised = cal_ref["realised"]
    print(f"\nS3 grid-slot calibration: predicted {100 * cal_ref['mean_pred']:.2f}% vs realised {100 * realised:.2f}%")

    tabs = {"S3_D0": tab_ref0, "S3_D2": table_for(m17, races, ref_mats, "D2", ref_pairs)[2],
            "grid": table_for(m17, races, ref_mats, "grid")[2]}
    base_rps = SX.mean_rps(races, theta, None, sw, cfg["tune_draws"])
    cands: dict[str, dict] = {}

    # ---------------- S4, S5, S6: tune alone, then evaluate with the plan-2 streams
    for step in [s for s in cfg["steps"] if s in ("S4", "S5", "S6")]:
        best, tab = tune_step(step, cfg, races, theta, sw)
        tab.to_csv(OUT / "results" / f"{stamp}_tuning_{step}_dev.csv", index=False)
        print(f"\n{step} TUNING (mean RPS, 300 draws, streams 1000+i; S3 frozen = {base_rps:.5f}); "
              f"{len(tab)} configurations\n{tab.round(5).to_string(index=False)}\nbest: {best}")
        th = {**theta, "tau": best["tau"]}
        ext = ext_of(step, best)
        mats, prs = SX.simulate_matrices(races, th, ext, sw, cfg["draws"], cfg["seed"], with_pairs=True)
        cands[step] = dict(races=races, mats=mats, pairs=prs, theta=th, ext=ext, best=best, tune=tab)

    # ---------------- S7: latent model with fitted phi and rule-change innovation
    if "S7" in cfg["steps"]:
        make, t7 = s7_search(cfg, entries, frozen)
        t7.to_csv(OUT / "results" / f"{stamp}_tuning_S7_dev.csv", index=False)
        print(f"\nS7 TUNING (mean Spearman, refit every {cfg['S7']['latent_refit_every']} races; plan-2 model = phi 0.8, "
              f"rule_mult 1); {len(t7)} configurations\n{t7.round(5).to_string(index=False)}")
        pick = [t7.loc[t7["spearman"].idxmax()]]
        if (pick[0]["phi"], pick[0]["rule_mult"]) == (0.8, 1.0):
            alt = t7[(t7["phi"] != 0.8) | (t7["rule_mult"] != 1.0)]
            pick.append(alt.loc[alt["spearman"].idxmax()])
        for j, row in enumerate(pick):
            phi, mult = float(row["phi"]), float(row["rule_mult"])
            name = "S7" if j == 0 else "S7_alt"
            preds7, _ = walk_forward(entries, make(phi, mult), "dev", refit_every=1, posdist=True)
            inp7 = sim_inputs_from(preds7, entries, "dev")
            races7 = SX.add_extras(S.prepare(inp7, consts), entries)
            assert [r["race_id"] for r in races7] == [r["race_id"] for r in races]
            mats, prs = SX.simulate_matrices(races7, theta, None, sw, cfg["draws"], cfg["seed"], with_pairs=True)
            cands[name] = dict(races=races7, mats=mats, pairs=prs, theta=theta, ext=None, phi=phi, rule_mult=mult,
                               best=dict(tau=theta["tau"], phi=phi, rule_mult=mult), tune=t7)
            print(f"{name}: phi {phi}, rule_mult {mult} (tuning Spearman {row['spearman']:.4f}); frozen S3 theta")
            if j == 0:       # secondary: tau re-fitted by RPS on the S7 scores (<= 3 more configurations)
                taus = {t: SX.mean_rps(races7, {**theta, "tau": t}, None, sw, cfg["tune_draws"]) for t in cfg["S7"]["tau_grid"]}
                tbest = min(taus, key=taus.get)
                print(f"S7 tau re-fit (RPS by tau): { {k: round(v, 5) for k, v in taus.items()} } -> tau {tbest}")
                if tbest != theta["tau"]:
                    th = {**theta, "tau": tbest}
                    mats, prs = SX.simulate_matrices(races7, th, None, sw, cfg["draws"], cfg["seed"], with_pairs=True)
                    cands["S7_tau"] = dict(races=races7, mats=mats, pairs=prs, theta=th, ext=None, phi=phi, rule_mult=mult,
                                           best=dict(tau=tbest, phi=phi, rule_mult=mult), tune=t7)

    # ---------------- calibration gate, decoders, tables
    cal_rows, dec_tabs, dec_preds = [], {}, {}
    for name, c in cands.items():
        gs = CAL.grid_slot_calibration(c["mats"], grids_, poss_, keys)
        passes = gs["mean_pred"] >= realised - cfg["calibration_pass_gap"]
        c["cal_pass"] = bool(passes)
        kinds = ["D0"] + (["D2"] if (passes or name == "S6") else [])      # S6 x D2 is declared in PLAN3 section 5
        for kind in kinds:
            u, pr, tb = table_for(m17, c["races"], c["mats"], kind, c["pairs"])
            dec_tabs[f"{name}_{kind}"], dec_preds[f"{name}_{kind}"] = tb, pr
            exp = np.mean([CAL.expected_exact(P, x) for P, x in zip(c["mats"], u)])
            cal_rows.append(dict(matrix=name, decoder=kind, grid_slot_pred_pct=100 * gs["mean_pred"],
                                 grid_slot_realised_pct=100 * gs["realised"], passes_gate=bool(passes),
                                 expected_exact_pct=100 * exp, realised_exact_pct=100 * tb["exact"].mean()))
        print(f"\n{name} grid-slot calibration: predicted {100 * gs['mean_pred']:.2f}% vs realised "
              f"{100 * gs['realised']:.2f}% -> gate {'passed' if passes else 'FAILED'}"
              f"{'' if passes else ('; D2 run only because the interaction is declared' if name == 'S6' else '; D2 not run')}")
    tabs.update(dec_tabs)
    cal = pd.DataFrame(cal_rows)
    print(cal.round(2).to_string(index=False))

    # ---------------- run folders, leaderboard rows
    lb_rows, run_ids = [], {}
    comp = {"S4": "M16_S4", "S5": "M16_S5", "S6": "M16_S6", "S7": "M16_S7", "S7_alt": "M16_S7_alt", "S7_tau": "M16_S7_tau"}
    for name, c in cands.items():
        for kind in ("D0", "D2"):
            key = f"{name}_{kind}"
            if key not in dec_tabs:
                continue
            lab = f"{name.lower()}_{kind}"
            run_ids[key], row = save(lab, cfg, dict(step=name, params={k: float(v) for k, v in c["best"].items()},
                                                    theta={k: float(v) for k, v in c["theta"].items()},
                                                    calibration_gate_passed=c["cal_pass"]),
                                     dec_preds[key], dec_tabs[key], stamp, gid, c["mats"], c["races"])
            lb_rows.append(row)

    # ---------------- paired gains and attribution rows
    att, allg = [], []
    refs = {"S3_D0": "S3_D0", "grid": "grid", "S3_D2": "S3_D2"}
    for name in cands:
        for kind in ("D0", "D2"):
            key = f"{name}_{kind}"
            if key not in tabs:
                continue
            cname = comp[name] + ("_D2" if kind == "D2" else "")
            for rk, rname in refs.items():
                if rk == "S3_D2" and kind != "D2":
                    continue
                g = A.paired_gain(tabs[rk], tabs[key], PAIRED_METRICS, n_boot=nb, seed=seed)
                g.insert(0, "reference", rname)
                g.insert(0, "candidate", cname)
                allg.append(g)
                vm = "exact" if rk == "grid" else cfg["verdict_metric"]     # grid order has no matrix: judged on exact
                att.append(A.attribution_row(cname, rname, g[g["metric"].isin(A.METRICS)], vm,
                                             run_ids[key], len(tabs[key])))
    # declared interaction S6 x D2: (S6 - S3 under D2) - (S6 - S3 under D0), paired by race
    if "S6_D2" in tabs:
        num = [c for c in tabs["S3_D0"].columns if c not in TAB_EXCLUDE]
        sub = ["spearman", "spearman_fin", "mae", "mae_fin", "exact", "exact_fin", "exact_1_3", "exact_4_10", "exact_11p",
               "within1", "within2"]
        r0 = tabs["S6_D0"].set_index("race_id")[sub] + tabs["S3_D2"].set_index("race_id")[sub]
        c0 = tabs["S6_D2"].set_index("race_id")[sub] + tabs["S3_D0"].set_index("race_id")[sub]
        ref_t, cand_t = r0.reset_index(), c0.reset_index()
        gI = A.paired_gain(ref_t, cand_t, sub, n_boot=nb, seed=seed)
        gI.insert(0, "reference", "S6_gain_D0")
        gI.insert(0, "candidate", "M16_S6xD2")
        allg.append(gI)
        att.append(A.attribution_row("M16_S6xD2", "S6_gain_D0", gI, "exact", run_ids["S6_D2"], len(ref_t)))
    full = pd.concat(allg, ignore_index=True)
    full.to_csv(OUT / "results" / f"{stamp}_paired_gains_dev.csv", index=False)
    cal.to_csv(OUT / "results" / f"{stamp}_grid_slot_calibration_dev.csv", index=False)
    show = ["spearman", "mae", "exact", "exact_1_3", "exact_4_10", "exact_11p", "rps", "logloss_points", JOINT]
    print("\nPAIRED GAINS (positive = candidate better; 95% CI over 66 races; 10,000 resamples, seed 0)")
    print(full[full["metric"].isin(show)].round(4).to_string(index=False))
    for r in att:
        sub = full[(full["candidate"] == r["component"]) & (full["reference"] == r["reference"])].set_index("metric")
        print(r["component"], "vs", r["reference"], {m: A.verdict(sub.loc[m]) for m in show if m in sub.index})

    # ---------------- Monte Carlo noise of one 1000-draw run (extra seeds), per candidate vs S3 with the same seed
    seeds = [cfg["seed"]] + list(cfg["noise_seeds"])
    nz = pd.concat([noise_table(nm, c["races"], c["theta"], c["ext"], races, theta, m17, cfg, sw, seeds)
                    for nm, c in cands.items() if nm in ("S4", "S5", "S6", "S7")], ignore_index=True)
    nz.to_csv(OUT / "results" / f"{stamp}_mc_noise_dev.csv", index=False)
    summ = nz.groupby(["candidate", "metric"]).agg(ref_mean=("ref", "mean"), ref_sd=("ref", lambda x: x.std(ddof=1)),
                                                   diff_mean=("diff", "mean"), diff_sd=("diff", lambda x: x.std(ddof=1)))
    print(f"\nMONTE CARLO NOISE over seeds {seeds} (D0; sd across seeds of the mean over 66 races)\n{summ.round(5).to_string()}")
    s4n = noise_table("S4_neutral", races, theta, dict(kind="S4", m2=1.0, m3=1.0, m4=1.0), races, theta, m17, cfg, sw, seeds)
    s4n.to_csv(OUT / "results" / f"{stamp}_neutral_s4_vs_s3_dev.csv", index=False)
    print("\nNEUTRAL S4 (m = 1) minus S3, same seeds (pure simulation noise of the staged sampler)\n",
          s4n.groupby("metric")["diff"].agg(["mean", "std"]).round(5).to_string())

    # ---------------- descriptive: S6 width of the Laplace interval, and gains by season and early rounds
    if "S6" in cands:
        sd = pd.DataFrame([dict(race_id=r["race_id"], season=r["season"], round=r["round"], early=bool(r["early"]),
                                mean_sd=float(np.sqrt(np.diag(r["cov"])).mean())) for r in races])
        print("\nS6 mean Laplace sd of u by season:\n", sd.groupby("season")["mean_sd"].mean().round(3).to_string(),
              "\nrounds <= 3 vs later:", sd.groupby("early")["mean_sd"].mean().round(3).to_dict())
        sd.to_csv(OUT / "results" / f"{stamp}_s6_laplace_sd_dev.csv", index=False)
    rows = []
    meta = pd.DataFrame([dict(race_id=r["race_id"], season=r["season"], early=bool(r["early"])) for r in races])
    for name in cands:
        t = tabs[f"{name}_D0"]
        for lab, ids in [(f"season {s}", meta[meta["season"] == s]["race_id"]) for s in sorted(meta["season"].unique())] + \
                        [("rounds 1-3", meta[meta["early"]]["race_id"]), ("rounds 4+", meta[~meta["early"]]["race_id"])]:
            a, b = tabs["S3_D0"][tabs["S3_D0"]["race_id"].isin(ids)], t[t["race_id"].isin(ids)]
            g = A.paired_gain(a, b, ["rps", "logloss_points", "spearman", JOINT], n_boot=nb, seed=seed)
            for _, x in g.iterrows():
                rows.append(dict(candidate=name, subset=lab, metric=x["metric"], n=int(x["n"]), gain=x["gain"], lo=x["lo"], hi=x["hi"]))
    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(OUT / "results" / f"{stamp}_gains_by_subset_dev.csv", index=False)
    print("\nGAINS BY SUBSET vs S3 D0 (subsets under 10 races are not interpreted)\n",
          sub_df[sub_df["metric"].isin(["rps", JOINT])].round(5).to_string(index=False))

    append_rows(ROOT / cfg["leaderboard"], pd.DataFrame(lb_rows))
    append_rows(ROOT / cfg["attribution"], pd.DataFrame(att))
    means = pd.DataFrame({n: t[[c for c in PAIRED_METRICS if c in t.columns]].mean() for n, t in tabs.items()}).T
    print("\nMEANS\n", means.round(4).to_string())
    print("\nattribution rows appended:", [(r["component"], r["reference"], r["verdict"]) for r in att])
    print("run_ids:", run_ids)


if __name__ == "__main__":
    main()
