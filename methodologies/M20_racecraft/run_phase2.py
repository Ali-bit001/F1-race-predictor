"""M20 phase 2 - driver consistency alone in the weekend term of latent_111 and as a column of gbm_100 (PLAN3.md R8). Dev only.

    python methodologies/M20_racecraft/run_phase2.py [--quick]

Feature (declared in the card, phase 1): history of `cons_c`, shrunk decayed mean (half-life 50, race_id as time, prior 0) over
EARLIER races of the same driver. The history is computed on a frame that has one row per entry (value NaN where the driver's
own race value is invalid), so for race r it is the decayed mean of all valid values of races < r (this is the forward-fill
done correctly: a stored history row of race s holds only races < s, so a literal "latest stored history row" would lag by
one observation). Divided by a fixed constant (SD of the history over seasons <= 2020). Post hoc, no decision weight: the same
with the teammate-relative `cons_c`.

References latent_111 / gbm_100 are re-run in the same harness and checked against the stored plan-1 dev runs.
--quick: first 6 dev races, nothing written.
"""
from __future__ import annotations

import argparse
import importlib.util
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
from pitlane.features import racecraft as RC
from pitlane.models.registry import build_model
from pitlane.plan3 import attribution as A
from pitlane.plan3 import decode as DEC
from pitlane.plan3 import runutil as U

MDIR = Path(__file__).resolve().parent
USED = ["src/pitlane/features/racecraft.py", "src/pitlane/features/pace.py", "src/pitlane/models/gbm.py",
        "src/pitlane/models/latent.py", "src/pitlane/plan3/attribution.py", "src/pitlane/plan3/runutil.py",
        "src/pitlane/evaluation/harness.py", "methodologies/M20_racecraft/run.py",
        "methodologies/M20_racecraft/run_phase2.py", "methodologies/M20_racecraft/config_phase2.yaml"]
VERDICT_METRICS = ["spearman", "mae", "exact", "exact_fin", "exact_1_3", "exact_4_10", "exact_11p", "spearman_fin", "mae_fin",
                   "logloss_points"]
PRIMARY = {"spearman", "mae"}
COL, COL_TM = "cons_x", "cons_tm_x"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_phase1():
    spec = importlib.util.spec_from_file_location("m20_phase1_run", MDIR / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def race_values_dev(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase-1 driver race values (cons_c; seasons 2014-2023 only: nothing later is read) and entries <= 2023."""
    p1 = _load_phase1()
    c1 = load_yaml(ROOT / cfg["race_values_config"])
    laps, pits, ent = p1.load(c1)
    drv = p1.race_values(laps, pits, ent, c1)["driver"]
    return drv, ent


def history_column(entries: pd.DataFrame, vals: pd.DataFrame, value_col: str, halflife: int) -> pd.Series:
    """Per entry of `entries` (all seasons): shrunk decayed mean of `value_col` over earlier races of the driver. Values exist
    only for races of seasons <= 2023; later rows get NaN (never used: the walk-forward stays inside dev)."""
    v = vals.dropna(subset=[value_col])[["race_id", "driverId", value_col]].rename(columns={value_col: "yv"})
    base = entries[["race_id", "driverId"]].drop_duplicates()
    assert len(base) == len(entries), "entries must be unique on (race_id, driverId)"
    frame = base.merge(v, on=["race_id", "driverId"], how="outer")      # value rows without an entry row stay (history only)
    h = RC.history_values(frame, "driverId", "yv", (halflife,))
    out = entries[["race_id", "driverId"]].merge(h, on=["race_id", "driverId"], how="left")
    last_valued = int(v["race_id"].max())
    col = out[f"yv_h{halflife}"].where(out["race_id"] <= last_valued)
    return pd.Series(col.to_numpy(float), index=entries.index)


def with_features(entries: pd.DataFrame, vals: pd.DataFrame, ent_le_2023: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    vals = vals.copy()
    vals["cons_tm"] = RC.teammate_relative(vals, "cons_c", ent_le_2023)
    out = entries.copy()
    info = {}
    for name, src in ((COL, "cons_c"), (COL_TM, "cons_tm")):
        raw = history_column(entries, vals, src, cfg["halflife"])
        ok = (entries["season"] <= cfg["scale_seasons_max"]) & raw.notna()
        sd = float(raw[ok].std())
        out[name] = raw / sd
        info[name] = dict(scale_sd=sd, mean_dev=float(out.loc[entries["season"].between(2021, 2023), name].mean()),
                          sd_dev=float(out.loc[entries["season"].between(2021, 2023), name].std()))
    return out, info


def leak_check(entries: pd.DataFrame, vals: pd.DataFrame, rid: int, halflife: int) -> float:
    """Feature of race `rid` unchanged when the values of race `rid` and later are blanked (max abs difference)."""
    full = history_column(entries, vals, "cons_c", halflife)
    cut = history_column(entries, vals.assign(cons_c=vals["cons_c"].where(vals["race_id"] < rid)), "cons_c", halflife)
    m = (entries["race_id"] <= rid).to_numpy()
    return float(np.nanmax(np.abs(full[m].to_numpy() - cut[m].to_numpy())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MDIR / "config_phase2.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["split"] == "dev", "plan 3 / R8 phase 2: development block only"
    t0 = time.time()
    entries0 = P.load_entries()
    frozen = load_yaml(ROOT / cfg["frozen_path"])
    lat = frozen[cfg["latent_ref"]["key"]][cfg["latent_ref"]["cell"]]
    gbm = frozen[cfg["gbm_ref"]["key"]][cfg["gbm_ref"]["cell"]]
    vals, ent_le = race_values_dev(cfg)
    entries, info = with_features(entries0, vals, ent_le, cfg)
    first = entries.groupby("race_id")["season"].first()
    dev_ids = first[(first >= SPLITS["dev"][0]) & (first <= SPLITS["dev"][1])].index
    lk = leak_check(entries0, vals, int(dev_ids[len(dev_ids) // 2]), cfg["halflife"])
    log(f"feature info {json.dumps(info)}; truncation check at race {int(dev_ids[len(dev_ids) // 2])}: max abs diff {lk:.2e}")
    assert lk == 0.0, "history feature changed when later values were blanked"
    stamp = U.code_stamp(USED)
    split = cfg["split"]
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

    def lat_cfg(col):
        return dict(lat, x_cols=[col], lam_x=cfg["lam_x"])

    def gbm_cfg(col):
        return dict(gbm, extra_cols=[col])

    runs = {"latent_111": (dict(lat), *run("latent_111", build_model(lat, entries_run))),
            "gbm_100": (dict(gbm), *run("gbm_100", build_model(gbm, entries_run)))}
    for comp, col in (("M20_consistency", COL), ("M20_consistency_tm", COL_TM)):
        lc, gc = lat_cfg(col), gbm_cfg(col)
        runs[f"{comp}_latent_111"] = (lc, *run(f"{comp}_latent_111", build_model(lc, entries_run)))
        runs[f"{comp}_gbm_100"] = (gc, *run(f"{comp}_gbm_100", build_model(gc, entries_run)))

    rep = {}
    if not args.quick:
        for k in ("latent_111", "gbm_100"):
            old_pm, old_pr = U.stored_run(cfg["plan1_leaderboard"], k)
            d, n = U.max_diff(runs[k][2], old_pm)
            ds, ne = U.score_diff(runs[k][1], old_pr)
            rep[k] = dict(max_abs_diff_per_race_metrics=d, max_abs_diff_scores=ds, races=n, entries=ne)
            log(f"REPRODUCTION {k}: max |per-race metric diff| = {d:.3e}, max |score diff| = {ds:.3e} ({n} races, {ne} entries)")

    key = cfg["tie_break"]
    tabs = {k: A.race_table(U.outcome_free(v[1], key)) for k, v in runs.items()}
    grid_preds = runs["latent_111"][1][["race_id", "season", "driverId", "pos", "grid", "classified"]].copy()
    grid_preds["score"] = np.concatenate([DEC.grid_utility(g["grid"].to_numpy(float), g[key].to_numpy())
                                          for _, g in grid_preds.groupby("race_id", sort=False)])
    tabs["grid"] = A.race_table(grid_preds)
    nb = cfg["bootstrap"]
    pairs = [("M20_consistency", "latent_111"), ("M20_consistency", "gbm_100"),
             ("M20_consistency_tm", "latent_111"), ("M20_consistency_tm", "gbm_100")]
    allg, seasong = [], []
    for comp, ref in pairs:
        rk, ck = ref, f"{comp}_{ref}"
        g = A.paired_gain(tabs[rk], tabs[ck], n_boot=nb["n_boot"], seed=nb["seed"])
        g.insert(0, "reference", ref)
        g.insert(0, "component", comp)
        allg.append(g)
        for s in (2021, 2022, 2023):
            a, b = tabs[rk][tabs[rk]["season"] == s], tabs[ck][tabs[ck]["season"] == s]
            gs = A.paired_gain(a, b, ["spearman", "mae", "exact", "spearman_fin", "mae_fin", "logloss_points"], n_boot=nb["n_boot"], seed=nb["seed"])
            gs.insert(0, "season", s)
            gs.insert(0, "reference", ref)
            gs.insert(0, "component", comp)
            seasong.append(gs)
    full = pd.concat(allg, ignore_index=True)
    full["verdict"] = [A.verdict(r) for _, r in full.iterrows()]
    by_season = pd.concat(seasong, ignore_index=True)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    mcols = ["spearman", "mae", "exact", "exact_fin", "spearman_fin", "mae_fin", "exact_1_3", "exact_4_10", "exact_11p", "logloss_points"]
    means = pd.DataFrame({k: t.reindex(columns=mcols).mean() for k, t in tabs.items()}).T
    print("\nMEANS\n", means.round(4).to_string())
    print("\nPAIRED GAINS (positive = candidate better; 95% bootstrap CI over races)\n"
          + full[full["metric"].isin(VERDICT_METRICS)].round(4).to_string(index=False))
    print("\nPER SEASON\n" + by_season.round(4).to_string(index=False))
    if args.quick:
        log("quick run: nothing written")
        return

    tmp_lb = MDIR / "results" / "_lb_tmp.csv"
    tmp_lb.parent.mkdir(parents=True, exist_ok=True)
    tmp_lb.unlink(missing_ok=True)
    labels = {"latent_111": "m20p2_ref_latent_111", "gbm_100": "m20p2_ref_gbm_100",
              "M20_consistency_latent_111": "m20p2_cons_latent_111", "M20_consistency_gbm_100": "m20p2_cons_gbm_100",
              "M20_consistency_tm_latent_111": "m20p2_posthoc_tm_latent_111", "M20_consistency_tm_gbm_100": "m20p2_posthoc_tm_gbm_100"}
    run_ids = {}
    for k, (c, p, m) in runs.items():
        extra = dict(component=k, frozen_hyperparameters="plan-1, not re-tuned; lam_x not tuned")
        if "tm_" in k:
            extra["status"] = "post hoc, no decision weight"
        rid_ = save_run(MDIR, labels[k], dict(c, **extra), p, m, split, "M20", c.get("cell", ""), leaderboard=tmp_lb)
        pth = MDIR / "results" / rid_ / "config.yaml"
        cc = load_yaml(pth)
        cc["git_commit"] = stamp
        save_yaml(cc, pth)
        tabs[k].to_csv(MDIR / "results" / rid_ / "outcome_free_table.csv", index=False)
        run_ids[k] = rid_
    new_rows = pd.read_csv(tmp_lb)
    tmp_lb.unlink()
    res = MDIR / "results"
    full.to_csv(res / "phase2_paired_gains_dev.csv", index=False)
    by_season.to_csv(res / "phase2_paired_gains_by_season.csv", index=False)
    full[full["metric"].isin(VERDICT_METRICS)][["component", "reference", "metric", "gain", "lo", "hi", "n", "verdict"]] \
        .assign(primary=lambda d: d["metric"].isin(PRIMARY)).to_csv(res / "phase2_verdicts.csv", index=False)
    means.to_csv(res / "phase2_means.csv")
    (res / "phase2_reproduction_check.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (res / "phase2_feature_info.json").write_text(json.dumps(dict(info, truncation_check_max_abs_diff=lk), indent=2), encoding="utf-8")

    # attribution rows (declared consistency only; the teammate-relative version gets none) appended as text lines
    rows = []
    for comp, ref in pairs[:2]:
        g = full[(full["component"] == comp) & (full["reference"] == ref)].drop(columns=["component", "reference", "verdict"])
        rows.append(A.attribution_row(comp, ref, g, cfg["verdict_metric"], run_ids[f"{comp}_{ref}"], len(tabs[f"{comp}_{ref}"]), declared="yes"))
    append_attribution_text(pd.DataFrame(rows), ROOT / cfg["attribution"])
    U.append_leaderboard(new_rows, ROOT / cfg["leaderboard"])
    log(f"verdicts (rho): {[(r['component'], r['reference'], r['verdict']) for r in rows]}")
    log(f"done in {time.time() - t0:.0f}s; code stamp {stamp}")


def append_attribution_text(rows: pd.DataFrame, path: Path) -> None:
    """Append text lines only (existing lines untouched; another run may append to the same file)."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    existing = pd.read_csv(path, usecols=["component", "reference"])
    for _, r in rows.iterrows():
        assert not ((existing["component"] == r["component"]) & (existing["reference"] == r["reference"])).any(), "row exists"
    text = path.read_text(encoding="utf-8")
    with open(path, "a", encoding="utf-8", newline="") as f:
        if not text.endswith("\n"):
            f.write("\n")
        rows.reindex(columns=header).to_csv(f, index=False, header=False, lineterminator="\n")


if __name__ == "__main__":
    main()
