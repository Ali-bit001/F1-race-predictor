"""M18 overtaking dead-band on the plan-2 simulator S3, development block only (PLAN3 R7).

    python methodologies/M18_overtaking/run.py [--config methodologies/M18_overtaking/config.yaml] [--check]

--check: acceptance (d = 0 reproduces the frozen S3 matrices) and the RPS tuning table; writes nothing.
Otherwise: tunes d (and tau once), evaluates, writes run folders, appends leaderboard and attribution rows as new
text lines (existing lines are never rewritten).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pitlane.config import METHODS, ROOT, load_yaml, save_yaml
from pitlane.evaluation.metrics import rps_score
from pitlane.plan2 import pipeline2 as Q
from pitlane.plan2 import simulate as S
from pitlane.plan3 import attribution as A
from pitlane.plan3 import calibration as CAL
from pitlane.plan3 import decode as DEC
from pitlane.plan3 import overtake as OT

OUT = METHODS / "M18_overtaking"


def _m17():
    spec = importlib.util.spec_from_file_location("m17_run", METHODS / "M17_decoding" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_id() -> str:
    h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=ROOT).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=ROOT).strip()
    return h + ("+dirty" if dirty else "")


def tune(races, theta, cfg) -> tuple[pd.DataFrame, float, float, float]:
    """RPS over the d grid at frozen tau, then tau in tau_grid at the d values in tau_at_d. Returns the table, the
    d with the lowest RPS (may be 0), the best POSITIVE d at frozen tau (the one evaluated if 0 wins), and the tau
    of the best positive-d configuration overall."""
    rows = [dict(d=d, tau=theta["tau"], rps=OT.mean_rps_db(races, theta, d, cfg["simulator"], cfg["tune_draws"]))
            for d in cfg["d_grid"]]
    for d in cfg["tau_at_d"]:
        for tau in cfg["tau_grid"]:
            if tau != theta["tau"]:
                rows.append(dict(d=d, tau=tau, rps=OT.mean_rps_db(races, {**theta, "tau": tau}, d, cfg["simulator"],
                                                                 cfg["tune_draws"])))
    tab = pd.DataFrame(rows)
    assert len(tab) <= cfg["max_configs"]
    fz = tab[tab["tau"] == theta["tau"]]
    d_min = float(fz.loc[fz["rps"].idxmin(), "d"])
    pos = fz[fz["d"] > 0]
    d_pos = float(pos.loc[pos["rps"].idxmin(), "d"])
    pa = tab[tab["d"] > 0]
    return tab, d_min, d_pos, float(pa.loc[pa["rps"].idxmin(), "tau"])


def table_for(m17, races, mats, kind: str):
    """(utilities, predictions, per-race table) of one decoder on one set of matrices."""
    keys = m17.driver_keys(races)
    if kind == "D0":
        util = [DEC.break_ties(-DEC.expected_position(P), k) for P, k in zip(mats, keys)]
    elif kind == "D2":
        util = [DEC.decode_d2(P) for P in mats]
    else:
        util = [DEC.grid_utility(r["g"]["grid"].to_numpy(float), k) for r, k in zip(races, keys)]
    preds = m17.build_preds(races, mats, util)
    rps = {r["race_id"]: rps_score(P, r["pos"]) for r, P in zip(races, mats)}
    return util, preds, A.race_table(preds, rps=rps)


def save(label, cfg, theta, d, preds, per, stamp, gid, mats, races):
    run_id = f"{stamp}_m18_{label}_dev"
    out = OUT / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    save_yaml({**cfg, "label": label, "d": float(d), "theta": {k: float(v) for k, v in theta.items()},
               "git_commit": gid, "split": "dev"}, out / "config.yaml")
    summ = {k: float(per[k].mean()) for k in per.columns if k not in ("race_id", "season", "round", "n")}
    summ["n_races"] = int(len(per))
    (out / "metrics.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    per.to_csv(out / "per_race_metrics.csv", index=False)
    preds.to_parquet(out / "predictions.parquet", index=False)
    np.savez_compressed(out / "position_matrices.npz", **{str(r["race_id"]): P for r, P in zip(races, mats)})
    return run_id, {"method": "M18", "cell": "S3", "label": f"m18_{label}", "split": "dev", "run_id": run_id,
                    "run_dir": str(out.relative_to(ROOT)), **summ}


def append_rows(path: Path, df: pd.DataFrame) -> None:
    """Append rows as new text lines in the existing column order; earlier lines stay byte-identical."""
    cols = list(pd.read_csv(path, nrows=0).columns)
    txt = df.reindex(columns=cols).to_csv(header=False, index=False, lineterminator="\n")
    with open(path, "a", newline="", encoding="utf-8") as f:
        f.write(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(OUT / "config.yaml"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["split"] == "dev"
    m17 = _m17()
    frozen = load_yaml(ROOT / cfg["frozen_path"])
    assert frozen["hash"] == cfg["frozen_hash"]
    theta = {k: frozen["M08"][cfg["simulator"]][k] for k in S.DEFAULT}
    consts = {k: frozen["M08"][cfg["simulator"]][k] for k in ("stick_ref", "vol_mu", "vol_sd")}
    races = S.prepare(m17.sim_inputs(Q.load_all(), "dev", cfg["m06_label"]), consts)

    ref = DEC.simulate_matrices(races, theta, cfg["simulator"], cfg["draws"], cfg["seed"])
    zero = OT.simulate_matrices_db(races, theta, 0.0, cfg["simulator"], cfg["draws"], cfg["seed"])
    diff = max(float(np.abs(a - b).max()) for a, b in zip(ref, zero))
    print(f"ACCEPTANCE d=0 vs frozen S3: max |P difference| = {diff:.3e}; bitwise equal = "
          f"{all(np.array_equal(a, b) for a, b in zip(ref, zero))}")
    tab, d_min, d_star, tau_star = tune(races, theta, cfg)
    pd.set_option("display.width", 200)
    print("\nTUNING (mean RPS, 300 draws, streams 1000+i; lower is better)\n", tab.round(5).to_string(index=False))
    print(f"\nlowest RPS at frozen tau: d = {d_min}; best positive d = {d_star} (evaluated); tau of best positive-d "
          f"configuration = {tau_star} (frozen {theta['tau']}); configurations {len(tab)}")
    if args.check:
        return

    stamp, gid = f"{datetime.now():%Y%m%d-%H%M}", git_id()
    cands = {"ref": (ref, theta, 0.0), "dstar": (OT.simulate_matrices_db(races, theta, d_star, cfg["simulator"], cfg["draws"],
                                                                         cfg["seed"]), theta, d_star)}
    if tau_star != theta["tau"]:
        th2 = {**theta, "tau": tau_star}
        cands["dstar_tau"] = (OT.simulate_matrices_db(races, th2, d_star, cfg["simulator"], cfg["draws"], cfg["seed"]),
                              th2, d_star)
    for sd in cfg.get("show_d", []):      # descriptive only: not fitted, no leaderboard or attribution row
        cands[f"show{sd}"] = (OT.simulate_matrices_db(races, theta, sd, cfg["simulator"], cfg["draws"], cfg["seed"]), theta, sd)
    tabs, utils, preds, mats_of = {}, {}, {}, {}
    for name, (mats, _, _) in cands.items():
        mats_of[name] = mats
        for kind in ("D0", "D2"):
            utils[f"{name}_{kind}"], preds[f"{name}_{kind}"], tabs[f"{name}_{kind}"] = table_for(m17, races, mats, kind)
    _, preds["grid"], tabs["grid"] = table_for(m17, races, ref, "grid")

    keys = m17.driver_keys(races)
    grids, poss = [r["g"]["grid"].to_numpy(float) for r in races], [r["pos"] for r in races]
    print("\nGRID-SLOT CALIBRATION (mean predicted P(finish = grid slot) vs realised) and expected exact")
    cal_rows = []
    for name, (mats, _, _) in cands.items():
        gs = CAL.grid_slot_calibration(mats, grids, poss, keys)
        for kind in ("D0", "D2"):
            exp = np.mean([CAL.expected_exact(P, u) for P, u in zip(mats, utils[f"{name}_{kind}"])])
            cal_rows.append(dict(matrix=name, decoder=kind, grid_slot_pred_pct=100 * gs["mean_pred"],
                                 grid_slot_realised_pct=100 * gs["realised"], expected_exact_pct=100 * exp,
                                 realised_exact_pct=100 * tabs[f"{name}_{kind}"]["exact"].mean()))
        if name == "dstar":
            print(gs["table"].round(4).to_string(index=False))
    cal = pd.DataFrame(cal_rows)
    print(cal.round(2).to_string(index=False))

    lb_rows, run_ids = [], {}
    for name, (mats, th, d) in cands.items():
        if name == "ref" or name.startswith("show"):
            continue
        for kind in ("D0", "D2"):
            label = f"d{d}{'_tau' + str(th['tau']) if name == 'dstar_tau' else ''}_{kind}"
            run_ids[f"{name}_{kind}"], row = save(label, cfg, th, d, preds[f"{name}_{kind}"], tabs[f"{name}_{kind}"],
                                                  stamp, gid, mats, races)
            lb_rows.append(row)

    comps = {"dstar_D0": "M18_deadband", "dstar_D2": "M18_deadband_D2", "dstar_tau_D0": "M18_deadband_tau"}
    refs = {"ref_D0": "S3_D0", "grid": "grid", "ref_D2": "S3_D2"}
    att, allg = [], []
    for cand, comp in comps.items():
        if cand not in tabs:
            continue
        for ref_key, ref_name in refs.items():
            if ref_key == "ref_D2" and cand != "dstar_D2":
                continue
            g = A.paired_gain(tabs[ref_key], tabs[cand], n_boot=cfg["bootstrap"]["n_boot"], seed=cfg["bootstrap"]["seed"])
            g.insert(0, "reference", ref_name)
            g.insert(0, "candidate", comp)
            allg.append(g)
            att.append(A.attribution_row(comp, ref_name, g, cfg["verdict_metric"], run_ids[cand], len(tabs[cand])))
    desc = []
    for sd in cfg.get("show_d", []):
        for kind in ("D0", "D2"):
            for ref_key, ref_name in (("ref_D0", "S3_D0"), ("grid", "grid")):
                g = A.paired_gain(tabs[ref_key], tabs[f"show{sd}_{kind}"], n_boot=cfg["bootstrap"]["n_boot"],
                                  seed=cfg["bootstrap"]["seed"])
                g.insert(0, "reference", ref_name)
                g.insert(0, "candidate", f"descriptive_d{sd}_{kind}")
                desc.append(g)
    if desc:
        dd = pd.concat(desc, ignore_index=True)
        dd.to_csv(OUT / "results" / f"{stamp}_descriptive_large_d_paired_dev.csv", index=False)
        print("\nDESCRIPTIVE large-d candidates (not fitted)")
        print(dd[dd["metric"].isin(["spearman", "mae", "exact", "rps"])].round(4).to_string(index=False))
    full = pd.concat(allg, ignore_index=True)
    full.to_csv(OUT / "results" / f"{stamp}_paired_gains_dev.csv", index=False)
    tab.to_csv(OUT / "results" / f"{stamp}_tuning_dev.csv", index=False)
    cal.to_csv(OUT / "results" / f"{stamp}_grid_slot_calibration_dev.csv", index=False)
    append_rows(ROOT / cfg["leaderboard"], pd.DataFrame(lb_rows))
    append_rows(ROOT / cfg["attribution"], pd.DataFrame(att))

    show = full[full["metric"].isin(["spearman", "mae", "exact", "exact_1_3", "exact_4_10", "exact_11p", "rps", "logloss_points"])]
    print("\nPAIRED GAINS (positive = candidate better; 95% CI over 66 races; 10,000 resamples seed 0)")
    print(show.round(4).to_string(index=False))
    print("\nverdicts (exact):", [(r["component"], r["reference"], r["verdict"]) for r in att])
    for r in att:
        sub = full[(full["candidate"] == r["component"]) & (full["reference"] == r["reference"])].set_index("metric")
        print(r["component"], "vs", r["reference"], {m: A.verdict(sub.loc[m]) for m in ("spearman", "mae", "exact")})
    print("\nMEANS\n", pd.DataFrame({n: t[A.METRICS].mean() for n, t in tabs.items()}).T.round(4).to_string())


if __name__ == "__main__":
    main()
