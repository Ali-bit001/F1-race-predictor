"""M05 ensemble (rank average of latent_111 and gbm_100) and monotone LightGBM, development block only (PLAN3.md R10).

    python methodologies/M05_ensemble/run.py [--quick]

Re-runs the references latent_111 and gbm_100 (frozen plan-1 hyper-parameters, refit before every dev race), the equal-weight
rank average and the monotone-constrained gbm_100 in the same harness; checks the references against the stored plan-1 runs;
writes one run folder per run, per-metric verdicts, a grid-order comparison, upserts attribution rows, and appends the
leaderboard rows at the very end. --quick: first 6 dev races, nothing written.
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

from pitlane import pipeline as P
from pitlane.config import ROOT, SPLITS, load_yaml, save_yaml
from pitlane.evaluation.harness import save_run, walk_forward
from pitlane.models.ensemble import MonotoneGBM, RankAverageModel
from pitlane.models.registry import build_model, cell_flags
from pitlane.plan3 import attribution as A
from pitlane.plan3 import decode as DEC
from pitlane.plan3 import runutil as U

MDIR = Path(__file__).resolve().parent
USED = ["src/pitlane/models/ensemble.py", "src/pitlane/models/gbm.py", "src/pitlane/models/latent.py", "src/pitlane/plan3/attribution.py",
        "src/pitlane/plan3/runutil.py", "src/pitlane/evaluation/harness.py",
        "methodologies/M05_ensemble/run.py", "methodologies/M05_ensemble/config.yaml"]
VERDICT_METRICS = ["spearman", "mae", "exact", "exact_fin", "exact_1_3", "exact_4_10", "exact_11p", "spearman_fin", "mae_fin"]
PRIMARY = {"spearman", "mae"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_monotone(cfg: dict, constrained: list[str]) -> MonotoneGBM:
    dyn, dec, rank = cell_flags(cfg["cell"])
    return MonotoneGBM(dynamic=dyn, decomp=dec, ranking=rank, constrained=tuple(constrained),
                       **{k: v for k, v in cfg.items() if k not in ("kind", "cell")})


def monotone_violations(model, entries: pd.DataFrame, train_n: int, race_id: int, col: str) -> dict:
    """Refit `model` on rows before `race_id`, then sweep `col` over its range for each entry of the race with all other
    inputs fixed and count entries whose prediction of y goes DOWN somewhere as `col` grows (violations)."""
    train = entries.iloc[:train_n]
    test = entries[entries["race_id"] == race_id].copy()
    model.fit_predict(train, test)
    sweep = np.linspace(0.0, 1.0, 21)
    bad = 0
    for _, r in test.iterrows():
        d = pd.DataFrame([r] * len(sweep)).reset_index(drop=True)
        d[col] = sweep
        y = model.model.predict(d[model.cols].astype(float))
        bad += int((np.diff(y) < -1e-12).any())
    return {"race_id": int(race_id), "entries": len(test), "entries_with_decreasing_y": bad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MDIR / "config.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["split"] == "dev", "plan 3 / R10: development block only"
    t0 = time.time()
    entries = P.load_entries()
    frozen = load_yaml(ROOT / cfg["frozen_path"])
    lat = frozen[cfg["latent_ref"]["key"]][cfg["latent_ref"]["cell"]]
    gbm = frozen[cfg["gbm_ref"]["key"]][cfg["gbm_ref"]["cell"]]
    stamp = U.code_stamp(USED)
    split = cfg["split"]
    first = entries.groupby("race_id")["season"].first()
    dev_ids = first[(first >= SPLITS["dev"][0]) & (first <= SPLITS["dev"][1])].index
    if args.quick:
        dev_ids = dev_ids[:6]
        entries_run = entries[entries["race_id"] <= dev_ids.max()].reset_index(drop=True)
    else:
        entries_run = entries

    def run(label: str, model):
        t = time.time()
        p, m = walk_forward(entries_run, model, split, refit_every=cfg["refit_every"])
        if args.quick:
            p, m = p[p["race_id"].isin(dev_ids)], m[m["race_id"].isin(dev_ids)]
        log(f"  {label}: rho={m['spearman'].mean():.4f} mae={m['mae'].mean():.3f} ({time.time() - t:.0f}s)")
        return p, m

    key = cfg["tie_break"]
    runs = {
        "latent_111": (dict(lat), *run("latent_111", build_model(lat, entries_run))),
        "gbm_100": (dict(gbm), *run("gbm_100", build_model(gbm, entries_run))),
        "M05_rank_avg": ({"kind": "rank_average", "members": ["latent_111", "gbm_100"], "weights": [0.5, 0.5]},
                         *run("M05_rank_avg", RankAverageModel([build_model(lat, entries_run), build_model(gbm, entries_run)], key))),
        "M05_gbm_monotone": ({**gbm, "monotone_increasing_y": cfg["monotone"]["increasing_y"]},
                             *run("M05_gbm_monotone", build_monotone(gbm, cfg["monotone"]["increasing_y"]))),
    }

    # ---- reproduction of the stored plan-1 runs
    rep = {}
    if not args.quick:
        for k in ("latent_111", "gbm_100"):
            old_pm, old_pr = U.stored_run(cfg["plan1_leaderboard"], k)
            d, n = U.max_diff(runs[k][2], old_pm)
            ds, ne = U.score_diff(runs[k][1], old_pr)
            rep[k] = dict(max_abs_diff_per_race_metrics=d, max_abs_diff_scores=ds, races=n, entries=ne)
            log(f"REPRODUCTION {k}: max |per-race metric diff| = {d:.3e}, max |score diff| = {ds:.3e} ({n} races, {ne} entries)")

    # ---- monotone constraint check (descriptive): last dev race, sweeps of grid_norm and quali_norm
    rid = int(dev_ids.max())
    n_train = int(entries_run.index[entries_run["race_id"] == rid].min())
    chk = {}
    for name, mdl in (("gbm_100", build_model(gbm, entries_run)), ("M05_gbm_monotone", build_monotone(gbm, cfg["monotone"]["increasing_y"]))):
        chk[name] = {c: monotone_violations(mdl, entries_run, n_train, rid, c) for c in ("grid_norm", "quali_norm")}
    log(f"MONOTONE CHECK (entries whose predicted y decreases somewhere along a sweep): {json.dumps(chk)}")

    # ---- paired attribution on the outcome-free ordering
    tabs = {k: A.race_table(U.outcome_free(v[1], key)) for k, v in runs.items()}
    grid_preds = runs["latent_111"][1][["race_id", "season", "driverId", "pos", "grid", "classified"]].copy()
    grid_preds["score"] = np.concatenate([DEC.grid_utility(g["grid"].to_numpy(float), g[key].to_numpy())
                                          for _, g in grid_preds.groupby("race_id", sort=False)])
    tabs["grid"] = A.race_table(grid_preds)
    nb = cfg["bootstrap"]
    pairs = [("M05_rank_avg", "latent_111"), ("M05_rank_avg", "gbm_100"), ("M05_gbm_monotone", "gbm_100")]
    allg = []
    for comp, ref in pairs:
        g = A.paired_gain(tabs[ref], tabs[comp], n_boot=nb["n_boot"], seed=nb["seed"])
        g.insert(0, "reference", ref)
        g.insert(0, "component", comp)
        allg.append(g)
    full = pd.concat(allg, ignore_index=True)
    full["verdict"] = [A.verdict(r) for _, r in full.iterrows()]
    # descriptive: exact-position rate against grid order, for every run
    gg = []
    for k in runs:
        g = A.paired_gain(tabs["grid"], tabs[k], ["exact", "exact_fin", "exact_1_3", "exact_4_10", "exact_11p"], n_boot=nb["n_boot"], seed=nb["seed"])
        g.insert(0, "run", k)
        gg.append(g)
    grid_cmp = pd.concat(gg, ignore_index=True)
    # descriptive: agreement of the two members
    rho = []
    lt, gt = runs["latent_111"][1], runs["gbm_100"][1]
    for rid_, a in lt.groupby("race_id"):
        b = gt[gt["race_id"] == rid_]
        rho.append(float(np.corrcoef(a["score"].rank(), b.set_index("driverId").loc[a["driverId"], "score"].rank())[0, 1]))
    agree = dict(mean_rank_corr_latent_vs_gbm=float(np.mean(rho)), min=float(np.min(rho)), races=len(rho))
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print("\nMEANS\n", pd.DataFrame({k: t[["spearman", "mae", "exact", "exact_fin", "spearman_fin", "mae_fin", "exact_1_3", "exact_4_10", "exact_11p"]].mean()
                                     for k, t in tabs.items()}).T.round(4).to_string())
    print("\nPAIRED GAINS (positive = candidate better; 95% bootstrap CI over races)\n"
          + full[full["metric"].isin(VERDICT_METRICS)].round(4).to_string(index=False))
    print("\nEXACT RATE vs GRID ORDER (descriptive; candidate minus grid)\n" + grid_cmp.round(4).to_string(index=False))
    print("\nMEMBER AGREEMENT", agree)
    if args.quick:
        log("quick run: nothing written")
        return

    # ---- write run folders (leaderboard rows are collected, then appended at the very end)
    tmp_lb = MDIR / "results" / "_lb_tmp.csv"
    tmp_lb.parent.mkdir(parents=True, exist_ok=True)
    tmp_lb.unlink(missing_ok=True)
    labels = {"latent_111": "m05_ref_latent_111", "gbm_100": "m05_ref_gbm_100", "M05_rank_avg": "m05_rank_avg",
              "M05_gbm_monotone": "m05_gbm_monotone"}
    run_ids = {}
    for k, (c, p, m) in runs.items():
        rid_ = save_run(MDIR, labels[k], dict(c, component=k, frozen_hyperparameters="plan-1, not re-tuned"), p, m, split, "M05",
                        c.get("cell", "111" if k == "M05_rank_avg" else ""), leaderboard=tmp_lb)
        pth = MDIR / "results" / rid_ / "config.yaml"
        cc = load_yaml(pth)
        cc["git_commit"] = stamp
        save_yaml(cc, pth)
        tabs[k].to_csv(MDIR / "results" / rid_ / "outcome_free_table.csv", index=False)
        run_ids[k] = rid_
    new_rows = pd.read_csv(tmp_lb)
    tmp_lb.unlink()
    res = MDIR / "results"
    full.to_csv(res / "paired_gains_dev.csv", index=False)
    full[full["metric"].isin(VERDICT_METRICS)][["component", "reference", "metric", "gain", "lo", "hi", "n", "verdict"]] \
        .assign(primary=lambda d: d["metric"].isin(PRIMARY)).to_csv(res / "verdicts.csv", index=False)
    grid_cmp.to_csv(res / "exact_vs_grid_dev.csv", index=False)
    (res / "reproduction_check.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (res / "monotone_check.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
    (res / "member_agreement.json").write_text(json.dumps(agree, indent=2), encoding="utf-8")
    rows = []
    for comp, ref in pairs:
        g = full[(full["component"] == comp) & (full["reference"] == ref)].drop(columns=["component", "reference", "verdict"])
        rows.append(A.attribution_row(comp, ref, g, cfg["verdict_metric"], run_ids[comp], len(tabs[comp]), declared="yes"))
    A.upsert_attribution(rows, ROOT / cfg["attribution"])
    U.append_leaderboard(new_rows, ROOT / cfg["leaderboard"])
    log(f"verdicts (rho): {[(r['component'], r['reference'], r['verdict']) for r in rows]}")
    log(f"done in {time.time() - t0:.0f}s; code stamp {stamp}")


if __name__ == "__main__":
    main()
