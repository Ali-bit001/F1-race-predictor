"""M09 race-pace model, exploratory post-gate component (PLAN3.md Amendment 3 item 6, R6). Dev only.

    python methodologies/M09_race_pace/run.py [--quick]

Columns added to the frozen latent_111 weekend term (x_cols) and to the frozen gbm_100 (extra_cols), D1 pace (median clean lap):
  pace_hist_x : decayed mean (half-life 50) of the car's field-relative race-pace gap in EARLIER races
  q1off_x     : decayed mean (half-life 50) of the car's field-relative offset (race gap - Q1 gap) in EARLIER races,
                Q1 gap missing stays missing (not imputed as 0.10)
Ladder: reference -> + pace_hist -> + pace_hist + q1off. Off-ladder diagnostic: q1off alone. No driver or stickiness rung.
References are re-run in the same harness and checked against the stored plan-1 dev runs.
--quick: first 6 dev races, 2 lam_x values, nothing written.
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
from pitlane.features import pace as PACE
from pitlane.models.registry import build_model
from pitlane.plan3 import attribution as A
from pitlane.plan3 import decode as DEC
from pitlane.plan3 import runutil as U

MDIR = Path(__file__).resolve().parent
USED = ["src/pitlane/features/pace.py", "src/pitlane/models/gbm.py", "src/pitlane/models/latent.py",
        "src/pitlane/plan3/attribution.py", "src/pitlane/plan3/runutil.py", "src/pitlane/evaluation/harness.py",
        "methodologies/M09_race_pace/run.py", "methodologies/M09_race_pace/config.yaml"]
VERDICT_METRICS = ["spearman", "mae", "exact", "exact_fin", "spearman_fin", "mae_fin", "exact_1_3", "exact_4_10", "exact_11p",
                   "logloss_points"]
HIST, OFF = "pace_hist_x", "q1off_x"
# component -> (columns, is_ladder_declared)
COMPS = {"M09_car_pace_hist": [HIST], "M09_car_q1_offset": [HIST, OFF], "M09_car_q1_offset_alone": [OFF]}
LAST_PACE_SEASON = 2023


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------ features
def _car_history(tab: pd.DataFrame, halflife: int) -> pd.DataFrame:
    """Per (race_id, driverId): decayed mean (earlier races only) of the car's `rq_c` and the number of earlier valid obs."""
    t = tab.assign(rq_tm=np.nan)
    h = PACE.rq_history_features(t, (halflife,), None)
    return h[["race_id", "driverId", f"con_rq_ew{halflife}", "con_n"]].rename(columns={f"con_rq_ew{halflife}": "v", "con_n": "n"})


def feature_columns(ent: pd.DataFrame, pace: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Unscaled features for the entries of seasons <= 2023 (`ent`: race_id, season, round, driverId, lineage, q1_gap, q1_missing).
    Race gap and Q1 gap enter only through the history of EARLIER races of the same car lineage."""
    q1 = ent["q1_gap"].where(ent["q1_missing"] == 0)                  # missing Q1 stays NaN (A3 sensitivity, not 0.10)
    base = ent.assign(q_best_gap=q1)
    rq = PACE.rq_table(pace, base, cfg["pace_col"])
    gp = rq[["race_id", "season", "driverId", "lineage", "race_gap_pct"]].copy()
    gp["rq_c"] = gp["race_gap_pct"] - gp.groupby("race_id")["race_gap_pct"].transform("median")   # field-relative race pace
    h_pace = _car_history(gp, cfg["halflife"]).rename(columns={"v": HIST, "n": "n_hist"})
    h_off = _car_history(rq[["race_id", "season", "driverId", "lineage", "rq_c"]], cfg["halflife"]).rename(columns={"v": OFF, "n": "n_off"})
    out = ent[["race_id", "season", "driverId"]].merge(h_pace, on=["race_id", "driverId"], how="left") \
                                                 .merge(h_off, on=["race_id", "driverId"], how="left")
    out[HIST] = out[HIST].where(out["n_hist"] >= cfg["min_prior_obs"])
    out[OFF] = out[OFF].where(out["n_off"] >= cfg["min_prior_obs"])
    return out


def build_features(entries: pd.DataFrame, pace: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """`entries` with HIST and OFF columns added (scaled by the SD over seasons <= scale_seasons_max; NaN after 2023)."""
    ent_le = entries[entries["season"] <= LAST_PACE_SEASON][["race_id", "season", "round", "driverId", "lineage", "q1_gap", "q1_missing"]]
    f = feature_columns(ent_le, pace, cfg)
    out, info = entries.copy(), {}
    for c in (HIST, OFF):
        raw = entries[["race_id", "driverId"]].merge(f[["race_id", "driverId", c]], on=["race_id", "driverId"], how="left")[c].to_numpy(float)
        raw = pd.Series(raw, index=entries.index)
        sd = float(raw[entries["season"] <= cfg["scale_seasons_max"]].std())
        out[c] = raw / sd
        dv = out.loc[entries["season"].between(*SPLITS["dev"]), c]
        info[c] = dict(scale_sd=sd, dev_mean=float(dv.mean()), dev_sd=float(dv.std()), dev_missing_share=float(dv.isna().mean()))
    return out, info


def leak_check(entries: pd.DataFrame, pace: pd.DataFrame, cfg: dict, rid: int) -> float:
    """Features of races <= rid unchanged when the pace rows and the Q1 times of race rid and later are blanked."""
    ent_le = entries[entries["season"] <= LAST_PACE_SEASON][["race_id", "season", "round", "driverId", "lineage", "q1_gap", "q1_missing"]]
    full = feature_columns(ent_le, pace, cfg)
    keys = ent_le[["season", "round", "race_id"]].drop_duplicates()
    pc = pace.merge(keys, on=["season", "round"], how="left")
    pace_cut = pc[pc["race_id"] < rid].drop(columns="race_id")
    late = ent_le["race_id"] >= rid
    ent_cut = ent_le.assign(q1_gap=ent_le["q1_gap"].where(~late), q1_missing=ent_le["q1_missing"].where(~late, 1))
    cut = feature_columns(ent_cut, pace_cut, cfg)
    m = (full["race_id"] <= rid).to_numpy()
    d = 0.0
    for c in (HIST, OFF):
        a, b = full.loc[m, c].to_numpy(float), cut.loc[m, c].to_numpy(float)
        assert (np.isnan(a) == np.isnan(b)).all(), f"{c}: missingness changed when later races were blanked"
        d = max(d, float(np.nanmax(np.abs(a - b))))
    return d


# ------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MDIR / "config.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["split"] == "dev", "plan 3 / R6: development block only"
    t0 = time.time()
    entries0 = P.load_entries()
    frozen = load_yaml(ROOT / cfg["frozen_path"])
    lat = frozen[cfg["latent_ref"]["key"]][cfg["latent_ref"]["cell"]]
    gbm = frozen[cfg["gbm_ref"]["key"]][cfg["gbm_ref"]["cell"]]
    pace = pd.read_parquet(ROOT / cfg["pace_table"])
    assert pace["season"].max() <= LAST_PACE_SEASON
    entries, info = build_features(entries0, pace, cfg)
    first = entries.groupby("race_id")["season"].first()
    dev_ids = first[(first >= SPLITS["dev"][0]) & (first <= SPLITS["dev"][1])].index
    rid_chk = int(dev_ids[len(dev_ids) // 2])
    lk = leak_check(entries0, pace, cfg, rid_chk)
    log(f"feature info {json.dumps(info)}; truncation check at race {rid_chk}: max abs diff {lk:.2e}")
    assert lk == 0.0, "feature changed when later races were blanked"
    stamp = U.code_stamp(USED)
    split, grid_lam = cfg["split"], list(cfg["lam_x_grid"])
    entries_run = entries
    if args.quick:
        dev_ids = dev_ids[:6]
        grid_lam = grid_lam[:2]
        entries_run = entries[entries["race_id"] <= dev_ids.max()].reset_index(drop=True)

    def run(label: str, model, refit: int):
        t = time.time()
        p, m = walk_forward(entries_run, model, split, refit_every=refit)
        if args.quick:
            p, m = p[p["race_id"].isin(dev_ids)], m[m["race_id"].isin(dev_ids)]
        log(f"  {label}: rho={m['spearman'].mean():.4f} mae={m['mae'].mean():.3f} ({time.time() - t:.0f}s)")
        return p, m

    # ---- tuning of lam_x (latent only), dev, refit_every=tune_refit_every: 3 rungs x len(grid) configurations
    tune_rows = []
    best_lam = {}
    for comp, cols in COMPS.items():
        for lam in grid_lam:
            _, m = run(f"tune {comp} lam_x={lam}", build_model(dict(lat, x_cols=cols, lam_x=lam), entries_run), cfg["tune_refit_every"])
            tune_rows.append(dict(component=comp, family="latent_111", lam_x=lam, spearman=m["spearman"].mean(), mae=m["mae"].mean(),
                                  exact=m["exact"].mean(), n_races=len(m)))
        t = pd.DataFrame([r for r in tune_rows if r["component"] == comp])
        best_lam[comp] = float(t.loc[t["spearman"].idxmax(), "lam_x"])
    tune = pd.DataFrame(tune_rows)
    n_cfg = len(tune) + len(COMPS)          # + one LightGBM configuration per rung (no tunable ridge)
    log(f"lam_x chosen {best_lam}; configurations evaluated for M09: {n_cfg} (budget 30)")

    # ---- final runs, refit before every race
    runs = {"latent_111": (dict(lat), *run("latent_111", build_model(lat, entries_run), cfg["refit_every"])),
            "gbm_100": (dict(gbm), *run("gbm_100", build_model(gbm, entries_run), cfg["refit_every"]))}
    for comp, cols in COMPS.items():
        lc, gc = dict(lat, x_cols=cols, lam_x=best_lam[comp]), dict(gbm, extra_cols=cols)
        runs[f"{comp}_latent_111"] = (lc, *run(f"{comp}_latent_111", build_model(lc, entries_run), cfg["refit_every"]))
        runs[f"{comp}_gbm_100"] = (gc, *run(f"{comp}_gbm_100", build_model(gc, entries_run), cfg["refit_every"]))

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
    gp = runs["latent_111"][1][["race_id", "season", "driverId", "pos", "grid", "classified"]].copy()
    gp["score"] = np.concatenate([DEC.grid_utility(g["grid"].to_numpy(float), g[key].to_numpy()) for _, g in gp.groupby("race_id", sort=False)])
    tabs["grid"] = A.race_table(gp)
    nb = cfg["bootstrap"]
    allg, seasong, gridg = [], [], []
    pairs = [(c, f) for c in COMPS for f in ("latent_111", "gbm_100")]
    for comp, ref in pairs:
        ck = f"{comp}_{ref}"
        g = A.paired_gain(tabs[ref], tabs[ck], n_boot=nb["n_boot"], seed=nb["seed"])
        g.insert(0, "reference", ref)
        g.insert(0, "component", comp)
        allg.append(g)
        for s in (2021, 2022, 2023):
            a, b = tabs[ref][tabs[ref]["season"] == s], tabs[ck][tabs[ck]["season"] == s]
            gs = A.paired_gain(a, b, ["spearman", "mae", "exact", "spearman_fin", "mae_fin"], n_boot=nb["n_boot"], seed=nb["seed"])
            gs.insert(0, "season", s)
            gs.insert(0, "reference", ref)
            gs.insert(0, "component", comp)
            seasong.append(gs)
        gg = A.paired_gain(tabs["grid"], tabs[ck], ["exact", "exact_fin"], n_boot=nb["n_boot"], seed=nb["seed"])
        gridg.append(gg.assign(component=comp, reference="grid order", family=ref))
    for ref in ("latent_111", "gbm_100"):                      # context: reference vs grid; ladder increment
        gridg.append(A.paired_gain(tabs["grid"], tabs[ref], ["exact", "exact_fin"], n_boot=nb["n_boot"], seed=nb["seed"])
                     .assign(component=f"{ref} (reference)", reference="grid order", family=ref))
        g = A.paired_gain(tabs[f"M09_car_pace_hist_{ref}"], tabs[f"M09_car_q1_offset_{ref}"], n_boot=nb["n_boot"], seed=nb["seed"])
        allg.append(g.assign(component="M09_car_q1_offset_increment_over_hist", reference=f"M09_car_pace_hist_{ref}"))
    full = pd.concat(allg, ignore_index=True)
    full["verdict"] = [A.verdict(r) for _, r in full.iterrows()]
    by_season = pd.concat(seasong, ignore_index=True)
    vs_grid = pd.concat(gridg, ignore_index=True)
    vs_grid["verdict"] = [A.verdict(r) for _, r in vs_grid.iterrows()]
    pd.set_option("display.width", 250, "display.max_rows", 500)
    mcols = ["spearman", "mae", "exact", "exact_fin", "spearman_fin", "mae_fin", "exact_1_3", "exact_4_10", "exact_11p", "logloss_points"]
    means = pd.DataFrame({k: t.reindex(columns=mcols).mean() for k, t in tabs.items()}).T
    print("\nTUNING\n", tune.round(4).to_string(index=False))
    print("\nMEANS\n", means.round(4).to_string())
    print("\nPAIRED GAINS (positive = candidate better; 95% bootstrap CI over races)\n"
          + full[full["metric"].isin(VERDICT_METRICS)].round(4).to_string(index=False))
    print("\nEXACT VS GRID ORDER\n" + vs_grid.round(4).to_string(index=False))
    print("\nPER SEASON\n" + by_season.round(4).to_string(index=False))
    if args.quick:
        log("quick run: nothing written")
        return

    tmp_lb = MDIR / "results" / "_lb_tmp.csv"
    tmp_lb.parent.mkdir(parents=True, exist_ok=True)
    tmp_lb.unlink(missing_ok=True)
    short = {"M09_car_pace_hist": "hist", "M09_car_q1_offset": "hist_q1off", "M09_car_q1_offset_alone": "q1off_alone"}
    labels = {"latent_111": "m09_ref_latent_111", "gbm_100": "m09_ref_gbm_100"}
    labels.update({f"{c}_{f}": f"m09_{short[c]}_{f}" for c in COMPS for f in ("latent_111", "gbm_100")})
    run_ids = {}
    for k, (c, p, m) in runs.items():
        extra = dict(component=k, frozen_hyperparameters="plan-1; lam_x tuned on dev (latent only)" if "latent" in k and k != "latent_111"
                     else "plan-1, not re-tuned", status="exploratory post-gate (Amendment 3 item 6); dev only")
        rid_ = save_run(MDIR, labels[k], dict(c, **extra), p, m, split, "M09", c.get("cell", ""), leaderboard=tmp_lb)
        pth = MDIR / "results" / rid_ / "config.yaml"
        cc = load_yaml(pth)
        cc["git_commit"] = stamp
        save_yaml(cc, pth)
        tabs[k].to_csv(MDIR / "results" / rid_ / "outcome_free_table.csv", index=False)
        run_ids[k] = rid_
    new_rows = pd.read_csv(tmp_lb)
    tmp_lb.unlink()
    res = MDIR / "results"
    tune.to_csv(res / "tuning_lam_x.csv", index=False)
    full.to_csv(res / "paired_gains_dev.csv", index=False)
    by_season.to_csv(res / "paired_gains_by_season.csv", index=False)
    vs_grid.to_csv(res / "exact_vs_grid.csv", index=False)
    means.to_csv(res / "means.csv")
    (res / "reproduction_check.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (res / "feature_info.json").write_text(json.dumps(dict(info, truncation_check_max_abs_diff=lk, truncation_race_id=rid_chk,
                                                           lam_x_chosen=best_lam, configurations_evaluated=n_cfg), indent=2), encoding="utf-8")

    declared = {"M09_car_pace_hist": "yes", "M09_car_q1_offset": "yes", "M09_car_q1_offset_alone": "no"}
    rows = []
    for comp, ref in pairs:
        g = full[(full["component"] == comp) & (full["reference"] == ref)].drop(columns=["component", "reference", "verdict"])
        rows.append(A.attribution_row(comp, ref, g, cfg["verdict_metric"], run_ids[f"{comp}_{ref}"], len(tabs[f"{comp}_{ref}"]), declared=declared[comp]))
    append_attribution_text(pd.DataFrame(rows), ROOT / cfg["attribution"])
    U.append_leaderboard(new_rows, ROOT / cfg["leaderboard"])
    log(f"verdicts (rho): {[(r['component'], r['reference'], r['verdict']) for r in rows]}")
    log(f"done in {time.time() - t0:.0f}s; code stamp {stamp}")


def append_attribution_text(rows: pd.DataFrame, path: Path) -> None:
    """Append text lines only, in one write (existing lines untouched; another run may append to the same file)."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    existing = pd.read_csv(path, usecols=["component", "reference"])
    for _, r in rows.iterrows():
        assert not ((existing["component"] == r["component"]) & (existing["reference"] == r["reference"])).any(), "row exists"
    text = path.read_text(encoding="utf-8")
    body = rows.reindex(columns=header).to_csv(index=False, header=False, lineterminator="\n")
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(("\n" if not text.endswith("\n") else "") + body)


if __name__ == "__main__":
    main()
