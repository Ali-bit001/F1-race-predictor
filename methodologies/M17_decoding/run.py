"""M17 metric-matched decoding on the plan-2 simulator S3, development block only.

    python methodologies/M17_decoding/run.py [--config methodologies/M17_decoding/config.yaml]

Re-runs S3 with the frozen plan-2 parameters (same streams), stores the position-probability matrices, decodes with
D0/D1/D2 (+ grid order, + extra diagnostic D0u), writes one run folder and one leaderboard row per decoder, and
upserts attribution rows (each decoder vs D0 and vs grid order).

Two tie conventions. D0, D1, grid: plan-2 convention (exact ties broken by input row order = finishing order), kept
only to reproduce the plan-2 acceptance numbers. D0_free, grid_free, D2, D0u: ties broken without row order
(driverId; D2 by row content). Attribution rows use the outcome-free versions.
Also writes descriptive diagnostics (calibration, per-season, blend, retirement-off); none of them is a component.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pitlane.config import METHODS, ROOT, load_yaml, save_yaml
from pitlane.evaluation.metrics import pred_rank, rps_score
from pitlane.plan2 import dnf as D
from pitlane.plan2 import pipeline2 as Q
from pitlane.plan2 import simulate as S
from pitlane.plan3 import attribution as A
from pitlane.plan3 import calibration as CAL
from pitlane.plan3 import decode as DEC

OUT = METHODS / "M17_decoding"


def git_id() -> str:
    h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=ROOT).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=ROOT).strip()
    return h + ("+dirty" if dirty else "")


def sim_inputs(entries: pd.DataFrame, split: str, m06_label: str) -> pd.DataFrame:
    """Same join as scripts/run_plan2.py: M06 score + M07 p_dnf + circuit/weather columns."""
    m06 = Q.load_run_preds(m06_label, split)
    lb = pd.read_csv(D.LB_DNF)
    row = lb[lb["split"] == split].iloc[-1]
    dn = pd.read_parquet(ROOT / row["run_dir"] / "predictions.parquet")[["race_id", "driverId", "p_dnf"]]
    cols = ["race_id", "driverId", "grid_norm", "stick", "circ_vol", "rain_fc"]
    return m06.merge(dn, on=["race_id", "driverId"]).merge(entries[cols], on=["race_id", "driverId"])


def driver_keys(races: list[dict]) -> list[np.ndarray]:
    return [r["g"]["driverId"].to_numpy() for r in races]


def utilities(races: list[dict], mats: list[np.ndarray], name: str) -> list[np.ndarray]:
    keys = driver_keys(races)
    e = [DEC.expected_position(P) for P in mats]
    if name == "D0":      # plan-2 convention: stored score is -E[pos]; ties by row order
        return [-x for x in e]
    if name == "D1":
        return [DEC.decode_d1(P) for P in mats]
    if name == "D0_free":
        return [DEC.break_ties(-x, k) for x, k in zip(e, keys)]
    if name == "D2":
        return [DEC.decode_d2(P) for P in mats]
    if name == "D2_old":  # pre-revision D2 (comparison only; never written)
        return [legacy_d2(P) for P in mats]
    if name == "grid":
        return [DEC.grid_utility(r["g"]["grid"].to_numpy(float)) for r in races]
    if name == "grid_free":
        return [DEC.grid_utility(r["g"]["grid"].to_numpy(float), k) for r, k in zip(races, keys)]
    if name == "D0u":
        return [DEC.break_ties(r["u"], k) for r, k in zip(races, keys)]
    raise ValueError(name)


def legacy_d2(P: np.ndarray) -> np.ndarray:
    """D2 as first written (closeness-to-E[pos] tie-break only), for the old-vs-new comparison."""
    from scipy.optimize import linear_sum_assignment
    n = P.shape[0]
    pl = np.arange(1, n + 1)
    rows, cols = linear_sum_assignment(-P + DEC.TIE * np.abs(pl[None, :] - DEC.expected_position(P)[:, None]))
    place = np.empty(n)
    place[rows] = cols + 1
    return -place


def build_preds(races: list[dict], mats: list[np.ndarray], util: list[np.ndarray]) -> pd.DataFrame:
    out = []
    for r, P, u in zip(races, mats, util):
        d = r["g"][["race_id", "season", "round", "driverId", "lineage", "pos", "grid", "classified"]].copy()
        n = len(d)
        d["score"], d["p_win"], d["p_podium"], d["p_points"] = u, P[:, 0], P[:, :3].sum(1), P[:, :min(10, n)].sum(1)
        d["p_dnf"] = r["p_dnf"]
        out.append(d)
    return pd.concat(out, ignore_index=True)


def band_table(preds: pd.DataFrame) -> pd.Series:
    """Exact rate by actual finishing band (pooled over entries), for the acceptance check."""
    hit, pos = [], []
    for _, g in preds.groupby("race_id"):
        hit.append(pred_rank(g["score"].to_numpy(float)) == g["pos"].to_numpy(float))
        pos.append(g["pos"].to_numpy(float))
    hit, pos = np.concatenate(hit), np.concatenate(pos)
    bands = {"1-3": (1, 3), "4-6": (4, 6), "7-10": (7, 10), "11-15": (11, 15), "16+": (16, 99)}
    return pd.Series({k: 100 * hit[(pos >= a) & (pos <= b)].mean() for k, (a, b) in bands.items()}
                     | {"overall_pooled": 100 * hit.mean()})


def save(label: str, cfg: dict, preds: pd.DataFrame, per: pd.DataFrame, stamp: str, gid: str) -> tuple[str, Path, dict]:
    """Write one run folder; returns (run_id, folder, leaderboard row). The leaderboard is written once, later."""
    run_id = f"{stamp}_m17_{label}_dev"
    out = OUT / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    save_yaml({**cfg, "decoder": label, "git_commit": gid, "split": "dev"}, out / "config.yaml")
    summ = {k: float(per[k].mean()) for k in per.columns if k not in ("race_id", "season", "round", "n", "tau")}
    summ["n_races"] = int(len(per))
    (out / "metrics.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    per.to_csv(out / "per_race_metrics.csv", index=False)
    preds.to_parquet(out / "predictions.parquet", index=False)
    row = {"method": "M17", "cell": "S3", "label": f"m17_{label}", "split": "dev", "run_id": run_id,
           "run_dir": str(out.relative_to(ROOT)), **summ}
    return run_id, out, row


def write_leaderboard(rows: list[dict], lb_path: Path) -> None:
    """Replace this method's own rows (method == M17) and append the new ones; other rows are untouched.
    Read-modify-write right before writing, because other work packages append to the same file."""
    lb = pd.read_csv(lb_path)
    lb = lb[lb["method"] != "M17"]
    new = pd.DataFrame([{k: v for k, v in r.items() if k in lb.columns} for r in rows])
    pd.concat([lb, new], ignore_index=True).to_csv(lb_path, index=False)


def clear_old_runs() -> None:
    """The revision replaces the earlier M17 run folders instead of duplicating them."""
    for f in (OUT / "results").glob("*"):
        shutil.rmtree(f) if f.is_dir() else f.unlink()


def exact_summary(races, mats, tabs, utils, names) -> pd.DataFrame:
    """Expected (under the model's own matrix) vs realised exact rate, plus re-ranked-among-finishers rate."""
    rows = []
    for nm in names:
        exp = np.mean([CAL.expected_exact(P, u) for P, u in zip(mats, utils[nm])])
        rer = np.mean([CAL.exact_reranked_finishers(u, r["pos"], r["cls"]) for r, u in zip(races, utils[nm])])
        rows.append(dict(decoder=nm, expected_exact_pct=100 * exp, realised_exact_pct=100 * tabs[nm]["exact"].mean(),
                         exact_full_field_classified_pct=100 * tabs[nm]["exact_fin"].mean(),
                         exact_reranked_among_finishers_pct=100 * rer))
    return pd.DataFrame(rows)


def season_gains(tabs: dict, pairs: list[tuple[str, str]], n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for cand, ref in pairs:
        for s in sorted(tabs[ref]["season"].unique()):
            a, b = tabs[ref][tabs[ref]["season"] == s], tabs[cand][tabs[cand]["season"] == s]
            g = A.paired_gain(a, b, ["exact"], n_boot=n_boot, seed=seed).iloc[0]
            rows.append(dict(season=int(s), candidate=cand, reference=ref, n_races=int(g["n"]),
                             gain_points=100 * g["gain"], lo=100 * g["lo"], hi=100 * g["hi"]))
    return pd.DataFrame(rows)


def blend_table(races, mats, ws) -> pd.DataFrame:
    keys = driver_keys(races)
    rows = []
    for w in ws:
        hit = [(pred_rank(DEC.decode_d2(CAL.blend_with_grid(P, r["g"]["grid"].to_numpy(float), w, k))) == r["pos"]).mean()
               for r, P, k in zip(races, mats, keys)]
        rows.append(dict(w=w, d2_exact_pct=100 * float(np.mean(hit))))
    return pd.DataFrame(rows)


def retirement_off(races, theta, cfg, names) -> pd.DataFrame:
    """Diagnostic: S3 re-simulated with p_dnf = 0 (same random streams), decoded and scored on the real outcomes."""
    r0 = [dict(r, p_dnf=np.zeros_like(r["p_dnf"])) for r in races]
    mats0 = DEC.simulate_matrices(r0, theta, cfg["simulator"], cfg["draws"], cfg["seed"])
    keys = driver_keys(races)
    rows = []
    for nm in names:
        u = ([DEC.break_ties(-DEC.expected_position(P), k) for P, k in zip(mats0, keys)] if nm == "D0_free"
             else [DEC.decode_d2(P) for P in mats0])
        hit = np.mean([(pred_rank(x) == r["pos"]).mean() for x, r in zip(u, races)])
        exp = np.mean([CAL.expected_exact(P, x) for P, x in zip(mats0, u)])
        rows.append(dict(decoder=nm, realised_exact_pct=100 * hit, expected_exact_pct=100 * exp))
    cal = CAL.grid_slot_calibration(mats0, [r["g"]["grid"].to_numpy(float) for r in races], [r["pos"] for r in races], keys)
    rows.append(dict(decoder="grid_slot_P_mean_vs_realised", realised_exact_pct=100 * cal["realised"],
                     expected_exact_pct=100 * cal["mean_pred"]))
    return pd.DataFrame(rows)


def seed_noise(races, theta, cfg, seeds) -> pd.DataFrame:
    """Simulation noise: D2 - D0_free exact (points) over independent simulation seeds."""
    keys = driver_keys(races)
    rows = []
    for sd in seeds:
        mats = DEC.simulate_matrices(races, theta, cfg["simulator"], cfg["draws"], sd)
        d0 = np.mean([(pred_rank(DEC.break_ties(-DEC.expected_position(P), k)) == r["pos"]).mean()
                      for P, r, k in zip(mats, races, keys)])
        d2 = np.mean([(pred_rank(DEC.decode_d2(P)) == r["pos"]).mean() for P, r in zip(mats, races)])
        rows.append(dict(seed=sd, d0_free_pct=100 * d0, d2_pct=100 * d2, d2_minus_d0_points=100 * (d2 - d0)))
    return pd.DataFrame(rows)


def tie_sensitivity(races, mats, reps: int = 200, seed: int = 0) -> pd.DataFrame:
    """Mean-of-races exact rate when tied entries are ordered by a random key (outcome-free), `reps` draws."""
    rng = np.random.default_rng(seed)
    res = {"D0": [], "grid": [], "D2": []}
    for _ in range(reps):
        h = {k: [] for k in res}
        for r, P in zip(races, mats):
            key = rng.permutation(len(P))
            g = r["g"]["grid"].to_numpy(float)
            h["D0"].append((pred_rank(DEC.decode_d1(P, key)) == r["pos"]).mean())
            h["grid"].append((pred_rank(DEC.grid_utility(g, key)) == r["pos"]).mean())
            h["D2"].append((pred_rank(DEC.decode_d2(P, key)) == r["pos"]).mean())
        for k in res:
            res[k].append(100 * float(np.mean(h[k])))
    return pd.DataFrame([dict(decoder=k, mean=np.mean(v), sd=np.std(v, ddof=1), min=np.min(v), max=np.max(v))
                         for k, v in res.items()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(OUT / "config.yaml"))
    ap.add_argument("--check", action="store_true", help="print the acceptance numbers and exit; writes nothing")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["split"] == "dev", "plan 3 / R2: development block only"
    frozen = load_yaml(ROOT / cfg["frozen_path"])
    assert frozen["hash"] == cfg["frozen_hash"], "frozen plan-2 config changed"
    theta = {k: frozen["M08"][cfg["simulator"]][k] for k in S.DEFAULT}
    consts = {k: frozen["M08"][cfg["simulator"]][k] for k in ("stick_ref", "vol_mu", "vol_sd")}

    entries = Q.load_all()
    races = S.prepare(sim_inputs(entries, "dev", cfg["m06_label"]), consts)
    mats = DEC.simulate_matrices(races, theta, cfg["simulator"], cfg["draws"], cfg["seed"])
    print(f"races {len(races)}, entries {sum(len(r['u']) for r in races)}")

    # sanity: re-simulated matrices reproduce the stored plan-2 M08 S3 dev scores
    lb2 = pd.read_csv(Q.LB2)
    old = lb2[(lb2["label"] == f"m08_{cfg['simulator']}") & (lb2["split"] == "dev")].iloc[-1]
    stored = pd.read_parquet(ROOT / old["run_dir"] / "predictions.parquet")
    mine = build_preds(races, mats, utilities(races, mats, "D0"))
    diff = float(np.abs(stored["score"].to_numpy() - mine["score"].to_numpy()).max())
    print(f"max |stored M08 S3 score - re-simulated -E[pos]| = {diff:.3e} (stored run {old['run_id']})")

    stamp, gid, lb_path = f"{datetime.now():%Y%m%d-%H%M}", git_id(), ROOT / cfg["leaderboard"]
    rps = {r["race_id"]: rps_score(P, r["pos"]) for r, P in zip(races, mats)}
    tabs, predd, utils = {}, {}, {}
    for name in cfg["decoders"] + ["D2_old"]:
        utils[name] = utilities(races, mats, name)
        predd[name] = build_preds(races, mats, utils[name])
        tabs[name] = A.race_table(predd[name], rps=rps)
    ties = sum(len(np.unique(-u)) < len(u) for u in utilities(races, mats, "D0"))
    print(f"races with tied D0 utilities (plan-2 convention): {ties}")

    print("\nACCEPTANCE (plan-2 convention: D0, grid; pooled over entries, exact rate %):")
    for name in ("D0", "grid"):
        print(name, " mean-of-races exact =", f"{100 * tabs[name]['exact'].mean():.2f}", " | pooled bands:",
              band_table(predd[name]).round(1).to_dict())
    print("\nTIE-BREAK OLD vs NEW (mean-of-races exact %):")
    for a, b in (("D0", "D0_free"), ("grid", "grid_free"), ("D2_old", "D2")):
        print(f"  {a:7s} {100 * tabs[a]['exact'].mean():.3f}  ->  {b:9s} {100 * tabs[b]['exact'].mean():.3f}")
    moved = sum(int((pred_rank(a) != pred_rank(b)).sum()) for a, b in zip(utils["D2_old"], utils["D2"]))
    print(f"  D2 cells whose assigned place changed old->new: {moved}")

    if args.check:
        return
    clear_old_runs()
    run_ids, lb_rows = {}, []
    for name in cfg["decoders"]:
        run_ids[name], folder, row = save(name, cfg, predd[name], tabs[name], stamp, gid)
        lb_rows.append(row)
        if name == "D0":
            np.savez_compressed(folder / "position_matrices.npz", **{str(r["race_id"]): P for r, P in zip(races, mats)})
    write_leaderboard(lb_rows, lb_path)

    # attribution rows: outcome-free decoders only (names as in PLAN3; D0u is off-plan -> declared "no")
    comp = {"D2": "M17_D2", "grid_free": "M17_grid", "D0_free": "M17_D0", "D0u": "M17_D0u"}
    refname = {"D0_free": "D0", "grid_free": "grid"}
    rows, allg = [], []
    for ref in cfg["reference_runs"]:
        for name in comp:
            if name == ref:
                continue
            g = A.paired_gain(tabs[ref], tabs[name], n_boot=cfg["bootstrap"]["n_boot"], seed=cfg["bootstrap"]["seed"])
            g.insert(0, "reference", refname[ref])
            g.insert(0, "candidate", comp[name])
            allg.append(g)
            rows.append(A.attribution_row(comp[name], refname[ref], g, cfg["verdict_metric"], run_ids[name],
                                          len(tabs[name]), declared="no" if name == "D0u" else "yes"))
    full = pd.concat(allg, ignore_index=True)
    full.to_csv(OUT / "results" / f"{stamp}_paired_gains_dev.csv", index=False)
    att_path = ROOT / cfg["attribution"]
    if att_path.exists():                                  # replace this method's own rows only
        old_att = pd.read_csv(att_path, dtype={"loo_gain": object, "fresh_holdout_gain": object})
        old_att[~old_att["component"].str.startswith("M17_")].to_csv(att_path, index=False)
    A.upsert_attribution(rows, att_path)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print("\nPAIRED GAINS (positive = candidate better; 95% bootstrap CI over races; outcome-free tie-breaks)")
    print(full.round(4).to_string(index=False))
    print("\nverdicts:", [(r["component"], r["reference"], r["declared"], r["verdict"]) for r in rows])
    means = pd.DataFrame({n: t[A.METRICS].mean() for n, t in tabs.items()}).T
    print("\nMEANS\n", means.round(4).to_string())

    # descriptive diagnostics (no component, no verdict)
    diag = OUT / "results" / f"{stamp}_m17_diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    names = ["D0_free", "D2", "grid_free"]
    keys, grids, poss = driver_keys(races), [r["g"]["grid"].to_numpy(float) for r in races], [r["pos"] for r in races]
    ev = exact_summary(races, mats, tabs, utils, names)
    gs = CAL.grid_slot_calibration(mats, grids, poss, keys)
    cell = CAL.cell_calibration(mats, poss)
    sea = season_gains(tabs, [("grid_free", "D0_free"), ("D2", "D0_free"), ("D2", "grid_free")],
                       cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
    bl = blend_table(races, mats, cfg["blend_w"])
    ro = retirement_off(races, theta, cfg, ["D0_free", "D2"]) if cfg.get("retirement_off") else pd.DataFrame()
    sn = seed_noise(races, theta, cfg, cfg["noise_seeds"]) if cfg.get("noise_seeds") else pd.DataFrame()
    ts = tie_sensitivity(races, mats)
    outs = {"tie_break_sensitivity": ts, "expected_vs_realised": ev, "grid_slot_calibration": gs["table"], "cell_calibration": cell,
            "per_season_paired": sea, "blend_diagnostic": bl, "retirement_off": ro, "simulation_seed_noise": sn}
    for nm, df in outs.items():
        if len(df):
            df.to_csv(diag / f"{nm}.csv", index=False)
            df.to_csv(ROOT / "results" / "plan3" / f"m17_{nm}.csv", index=False)
    print(f"\nEXPECTED vs REALISED exact (%), and re-ranked among classified finishers\n{ev.round(2).to_string(index=False)}")
    print(f"\nGRID-SLOT CALIBRATION: mean predicted P(finish = grid slot) {100 * gs['mean_pred']:.2f}% vs realised "
          f"{100 * gs['realised']:.2f}%\n{gs['table'].round(4).to_string(index=False)}")
    print(f"\nCELL CALIBRATION\n{cell.round(4).to_string(index=False)}")
    print(f"\nPER-SEASON PAIRED (points)\n{sea.round(2).to_string(index=False)}")
    print(f"\nBLEND (in-sample diagnostic; w not tuned out of sample) D2 exact %\n{bl.round(2).to_string(index=False)}")
    if len(ro):
        print(f"\nRETIREMENT OFF (p_dnf = 0, diagnostic)\n{ro.round(2).to_string(index=False)}")
    if len(sn):
        print(f"\nSIMULATION SEED NOISE\n{sn.round(2).to_string(index=False)}\n"
              f"sd of D2-D0 over seeds: {sn['d2_minus_d0_points'].std(ddof=1):.2f} points")


if __name__ == "__main__":
    main()
