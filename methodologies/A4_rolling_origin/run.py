"""A4 rolling-origin evaluation (PLAN3.md section 4, R3): season table 2015-2026 and rolling re-tune on dev.

    python methodologies/A4_rolling_origin/run.py [--quick]

--quick: 2-config grids, one tuning season, no run folders / leaderboard rows / result files.
Reusable code is in src/pitlane/plan3/rolling.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitlane import pipeline as P
from pitlane.config import RESULTS, ROOT, load_yaml, save_yaml
from pitlane.evaluation.harness import _git_hash, save_run, walk_forward
from pitlane.models.baselines import Heuristic
from pitlane.models.registry import build_model
from pitlane.plan3 import rolling as R

MDIR = Path(__file__).resolve().parent
OUT = RESULTS / "plan3"
USED_FILES = ["src/pitlane/plan3/rolling.py", "src/pitlane/evaluation/harness.py", "src/pitlane/pipeline.py",
              "methodologies/A4_rolling_origin/run.py", "methodologies/A4_rolling_origin/config.yaml"]
STAMP = ""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def code_stamp() -> str:
    head = _git_hash()
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--"] + USED_FILES, text=True, cwd=ROOT).strip()
    except Exception:
        return head + "+unknown"
    if not dirty:
        return head
    h = hashlib.sha1()
    for f in USED_FILES:
        h.update(f.encode()); h.update((ROOT / f).read_bytes())
    return f"{head}+dirty-{h.hexdigest()[:12]}"


def save(label: str, cfg: dict, preds: pd.DataFrame, per: pd.DataFrame, split: str, cell: str) -> str:
    rid = save_run(MDIR, label, cfg, preds, per, split, "A4", cell, leaderboard=OUT / "leaderboard.csv")
    p = MDIR / "results" / rid / "config.yaml"
    c = load_yaml(p); c["git_commit"] = STAMP; save_yaml(c, p)
    return rid


def model_for(name: str, cfg: dict, frozen: dict, entries: pd.DataFrame):
    spec = cfg["models"][name]
    if "heuristic" in spec:
        return {"kind": "heuristic", "which": spec["heuristic"]}, Heuristic(which=spec["heuristic"])
    c = frozen[spec["key"]][spec["cell"]]
    return c, build_model(c, entries)


# ------------------------------------------------------------------------------------ season table
def stage_season_table(entries: pd.DataFrame, cfg: dict, frozen: dict) -> tuple[dict, dict, dict]:
    st = cfg["season_table"]
    per_all, preds_all, verify = {}, {}, {}
    first, reuse = st["first_season"], st["reuse_from"]
    for name in cfg["models"]:
        c, model = model_for(name, cfg, frozen, entries)
        log(f"{name}: re-running seasons {first}-{reuse - 1} (refit_every=1)")
        p_new, m_new = walk_forward(entries, model, R.season_split(first, reuse - 1), refit_every=1)
        lab = {"quali": "quali", "latent_111": "latent_111", "gbm_100": "gbm_100"}[name]
        p_old = R.stored_predictions(lab)
        m_old = R.rescore(p_old)
        cols = [k for k in p_new.columns if k in p_old.columns]
        preds_all[name] = pd.concat([p_new[cols], p_old[cols]], ignore_index=True)
        per_all[name] = pd.concat([m_new[m_old.columns.intersection(m_new.columns)], m_old], ignore_index=True)
        if st["verify_reuse"]:
            log(f"{name}: verifying stored 2021-2026 by a full re-run")
            _, c_model = model_for(name, cfg, frozen, entries)
            p_chk, m_chk = walk_forward(entries, c_model, R.season_split(reuse, st["last_season"]), refit_every=1)
            a = m_chk.set_index("race_id")[R.COLS + ["rmse", "kendall", "ndcg10", "mae_fin", "spearman_fin"]]
            b = m_old.set_index("race_id").loc[a.index, a.columns]
            sc = p_chk.merge(p_old[["race_id", "driverId", "score"]], on=["race_id", "driverId"], suffixes=("_new", "_old"))
            verify[name] = {"n_races_checked": int(len(a)),
                            "max_abs_diff_metrics": {k: float((a[k] - b[k]).abs().max()) for k in a.columns},
                            "max_abs_diff_score": float((sc["score_new"] - sc["score_old"]).abs().max()),
                            "n_entries_matched": int(len(sc)), "n_entries_stored": int(len(p_old))}
            log(f"  {name}: {verify[name]}")
    return per_all, preds_all, verify


def figure(tab: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    names = {"quali": "qualifying order", "latent_111": "latent_111", "gbm_100": "gbm_100"}
    col = {"quali": "tab:gray", "latent_111": "tab:blue", "gbm_100": "tab:orange"}
    for ax, (met, lab) in zip(axes, [("spearman", "mean per-race Spearman"), ("mae", "mean per-race MAE (positions)"),
                                     ("exact", "exact-position rate")]):
        for m, g in tab[tab["half"] != "all"].groupby("model"):
            g = g.assign(x=g["season"] + np.where(g["half"] == "H1", 0.0, 0.5)).sort_values("x")
            ax.plot(g["x"], g[met], "-o", ms=3, color=col[m], label=names[m])
        ax.axvspan(2014.9, 2020.99, color="0.9", zorder=0)
        ax.axvspan(2023.99, 2027, color="#ffe9c9", zorder=0)
        ax.set_ylabel(lab)
        ax.grid(alpha=.3)
    axes[0].legend(ncol=3, loc="lower right")
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi + 0.1 * (hi - lo))
    top = axes[0].get_ylim()[1] - 0.02 * (axes[0].get_ylim()[1] - lo)
    axes[0].text(2015.05, top, "not clean", va="top", fontsize=8)
    axes[0].text(2021.05, top, "dev (tuned on)", va="top", fontsize=8)
    axes[0].text(2024.05, top, "third look, no decision (2026: R1-20)", va="top", fontsize=8)
    axes[2].set_xlabel("season (points at season start = first half, +0.5 = second half)")
    fig.suptitle("A4: walk-forward accuracy by season half, refit before every race")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------ rolling re-tune
def stage_retune(entries: pd.DataFrame, cfg: dict, frozen: dict, quick: bool) -> dict:
    rt = cfg["retune"]
    seasons = rt["seasons"][:1] if quick else rt["seasons"]
    out = {}
    for name in rt["models"]:
        spec = cfg["models"][name]
        fixed_cfg = frozen[spec["key"]][spec["cell"]]
        fixed_preds = R.stored_predictions(name)
        fixed_preds = fixed_preds[fixed_preds["season"].isin(seasons)]
        fixed_per = R.rescore(fixed_preds)
        chosen, tabs, preds, pers = {}, [], [], []
        for s in seasons:
            lo, hi = s - rt["lookback"], s - 1
            grid = P.grid_for(spec["kind"], spec["cell"])
            log(f"{name}: tuning for {s} on {lo}-{hi} ({len(grid)} configs, refit_every={rt['refit_every_tuning']})")
            if quick:
                P_grid = P.grid_for
                P.grid_for = lambda k, c, _g=P_grid: _g(k, c)[:2]
            best, tab = R.tune_window(spec["kind"], spec["cell"], entries, lo, hi, rt["refit_every_tuning"],
                                      rt["n_jobs"], rt["max_configs"])
            if quick:
                P.grid_for = P_grid
            tab.insert(0, "predict_season", s); tab.insert(1, "tuned_on", f"{lo}-{hi}")
            tabs.append(tab)
            chosen[s] = {k: v for k, v in best.items()}
            log(f"  chosen for {s}: {chosen[s]}")
            p, m = R.rolling_predict(entries, best, s)
            preds.append(p); pers.append(m)
        preds, per = pd.concat(preds, ignore_index=True), pd.concat(pers, ignore_index=True)
        per = per.sort_values("race_id").reset_index(drop=True)
        fx = fixed_per.set_index("race_id")
        rl = per.set_index("race_id").loc[fx.index]
        res = {"chosen": chosen, "fixed_cfg": fixed_cfg, "n_races": int(len(fx))}
        for met in R.COLS:
            d = (rl[met] - fx[met]).to_numpy()
            mean, lo_, hi_ = R.paired_ci(d, cfg["bootstrap"], cfg["seed"])
            res[met] = {"fixed": float(fx[met].mean()), "rolling": float(rl[met].mean()),
                        "delta": mean, "ci_lo": lo_, "ci_hi": hi_, "ci_excludes_0": bool(lo_ > 0 or hi_ < 0)}
        bys = {}
        for s in seasons:
            sel = fx["season"] == s
            d = (rl["spearman"] - fx["spearman"]).to_numpy()[sel.to_numpy()]
            bys[int(s)] = {"n": int(sel.sum()), "fixed": float(fx.loc[sel, "spearman"].mean()),
                           "rolling": float(rl.loc[sel, "spearman"].mean()), "delta": float(d.mean())}
        res["by_season_spearman"] = bys
        res["decision"] = ("tune on the three seasons before the freeze" if res["spearman"]["ci_lo"] > 0
                           else "keep hyper-parameters tuned on 2021-23")
        out[name] = res
        pd.concat(tabs).to_csv(OUT / f"a4_tuning_{name}.csv", index=False) if not quick else None
        per_paired = pd.DataFrame({"race_id": fx.index, "season": fx["season"].to_numpy(), "round": fx["round"].to_numpy(),
                                   **{f"{m}_fixed": fx[m].to_numpy() for m in R.COLS},
                                   **{f"{m}_rolling": rl[m].to_numpy() for m in R.COLS}})
        res["_paired"], res["_preds"], res["_per"] = per_paired, preds, per
    return out


def postprocess(cfg: dict) -> None:
    """Gain over qualifying order from the stored season-table run folders (no refit)."""
    per = R.load_season_table_per_race(MDIR / "results")
    tab = R.gain_over_qualifying(per, cfg["bootstrap"], cfg["seed"])
    tab.to_csv(OUT / "a4_gain_over_qualifying.csv", index=False)
    R.gain_figure(tab, OUT / "figures" / "a4_gain_over_qualifying.png")
    figure(pd.read_csv(OUT / "season_table.csv"), OUT / "figures" / "a4_season_table.png")
    log(f"wrote {OUT / 'a4_gain_over_qualifying.csv'} ({len(tab)} rows)")


def main() -> None:
    global STAMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MDIR / "config.yaml"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--postprocess-only", action="store_true",
                    help="only a4_gain_over_qualifying.* from existing run folders, no refit")
    args = ap.parse_args()
    if args.postprocess_only:
        return postprocess(load_yaml(args.config))
    t0 = time.time()
    cfg = load_yaml(args.config)
    STAMP = code_stamp()
    frozen = load_yaml(ROOT / cfg["frozen_plan1"])
    entries = pd.read_parquet(ROOT / cfg["entries"])
    assert entries["race_id"].is_monotonic_increasing
    log(f"entries {entries.shape}, last race {entries['season'].iloc[-1]} R{entries['round'].iloc[-1]}, stamp {STAMP}")

    metrics: dict = {"git_commit": STAMP}
    per_all, preds_all, verify = stage_season_table(entries, cfg, frozen)
    tab = R.season_table(per_all)
    metrics["reuse_verification"] = verify
    retune = stage_retune(entries, cfg, frozen, args.quick)
    metrics["retune"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in retune.items()}
    metrics["runtime_min"] = round((time.time() - t0) / 60, 2)

    if args.quick:
        print(tab.query("half=='all'").pivot(index="season", columns="model", values="spearman").round(3))
        print(json.dumps(metrics, indent=1, default=str)[:3000])
        return

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    tab.to_csv(OUT / "season_table.csv", index=False)
    figure(tab, OUT / "figures" / "a4_season_table.png")
    paired = pd.concat([v["_paired"].assign(model=k) for k, v in retune.items()])
    paired.to_csv(OUT / "a4_paired_dev.csv", index=False)
    (OUT / "a4_summary.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    runs = {}
    for name in cfg["models"]:
        c = {"a4": "season_table", "model": name, "reuse_from": cfg["season_table"]["reuse_from"],
             "note": "2015-2020 re-run, 2021-2026 stored plan-1 predictions re-scored; 2026 = rounds 1-20",
             "reuse_verification": verify.get(name)}
        runs[f"season_table_{name}"] = save(f"season_table_{name}", c, preds_all[name],
                                            per_all[name].sort_values("race_id"), "all2015_2026", name)
    for name, v in retune.items():
        spec = cfg["models"][name]
        c = {"a4": "rolling_retune", "model": name, "chosen_per_season": v["chosen"], "fixed": v["fixed_cfg"],
             "tuning": {"lookback": cfg["retune"]["lookback"], "refit_every": cfg["retune"]["refit_every_tuning"],
                        "grid": "pipeline.grid_for (plan-1 search space)"}, "paired": {m: v[m] for m in R.COLS}}
        runs[f"rolling_{name}"] = save(f"rolling_{name}", c, v["_preds"], v["_per"], "dev", spec["cell"])
    metrics["run_ids"] = runs
    (OUT / "a4_summary.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (MDIR / "results").mkdir(exist_ok=True)
    (MDIR / "results" / "a4_summary.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    postprocess(cfg)
    log(f"done in {metrics['runtime_min']} min")


if __name__ == "__main__":
    sys.exit(main())
