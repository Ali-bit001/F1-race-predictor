"""M09 car-form control (diagnostic, post hoc, no decision weight). See control_prestatement.md. Dev only.

    python methodologies/M09_race_pace/run_control.py [--quick]

form_x = decayed mean (half-life 50, earlier races only) of the car's field-relative finishing position (car mean of the
normalised finishing position y, centred by the race mean), same lineage map / min_prior_obs / scale rule as the pace columns.
C1 = ref + form_x ; C2 = ref + form_x + pace_hist_x (latent_111, lam_x tuned). The existing 8 M09 runs are NOT rerun: their
outcome-free tables are read for the paired comparisons.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MDIR))
import run as M09                                                     # noqa: E402
from pitlane import pipeline as P                                     # noqa: E402
from pitlane.config import ROOT, SPLITS, load_yaml, save_yaml         # noqa: E402
from pitlane.evaluation.harness import save_run, walk_forward         # noqa: E402
from pitlane.features.pace import _decayed                            # noqa: E402
from pitlane.models.registry import build_model                       # noqa: E402
from pitlane.plan3 import attribution as A                            # noqa: E402
from pitlane.plan3 import runutil as U                                # noqa: E402

FORM, HIST = "form_x", M09.HIST
RUNGS = {"M09_ctrl_form": [FORM], "M09_ctrl_form_hist": [FORM, HIST]}
USED = M09.USED + ["methodologies/M09_race_pace/run_control.py", "methodologies/M09_race_pace/control_prestatement.md"]
log = M09.log


def form_raw(entries: pd.DataFrame, halflife: int, min_obs: int) -> pd.DataFrame:
    """Unscaled form per (race_id, driverId): decayed car mean of centred y over EARLIER races of the lineage."""
    e = entries[["race_id", "driverId", "lineage", "y"]].copy()
    e["yc"] = e["y"] - e.groupby("race_id")["y"].transform("mean")
    cr = e.groupby(["lineage", "race_id"], as_index=False)["yc"].mean().sort_values(["lineage", "race_id"]).reset_index(drop=True)
    cr[FORM], cr["n"] = _decayed(cr, ["lineage"], "yc", halflife)
    cr[FORM] = cr[FORM].where(cr["n"] >= min_obs)
    m = e[["race_id", "driverId", "lineage"]].merge(cr[["lineage", "race_id", FORM]], on=["lineage", "race_id"], how="left")
    return m[["race_id", "driverId", FORM]]


def add_form(entries: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    f = form_raw(entries, cfg["halflife"], cfg["min_prior_obs"])
    raw = entries[["race_id", "driverId"]].merge(f, on=["race_id", "driverId"], how="left")[FORM].to_numpy(float)
    raw = pd.Series(raw, index=entries.index)
    sd = float(raw[entries["season"] <= cfg["scale_seasons_max"]].std())
    out = entries.copy()
    out[FORM] = raw / sd
    dv = out.loc[entries["season"].between(*SPLITS["dev"]), FORM]
    return out, dict(scale_sd=sd, dev_mean=float(dv.mean()), dev_sd=float(dv.std()), dev_missing_share=float(dv.isna().mean()))


def leak_check_form(entries: pd.DataFrame, cfg: dict, rid: int) -> float:
    """form_x of races <= rid unchanged when the outcomes (y) of race rid and later are blanked."""
    full = form_raw(entries, cfg["halflife"], cfg["min_prior_obs"])
    cut = form_raw(entries.assign(y=entries["y"].where(entries["race_id"] < rid)), cfg["halflife"], cfg["min_prior_obs"])
    m = (full["race_id"] <= rid).to_numpy()
    a, b = full.loc[m, FORM].to_numpy(float), cut.loc[m, FORM].to_numpy(float)
    assert (np.isnan(a) == np.isnan(b)).all(), "form_x missingness changed"
    return float(np.nanmax(np.abs(a - b)))


def within_race_rank_corr(df: pd.DataFrame, a: str, b: str) -> float:
    r = [g[a].rank().corr(g[b].rank()) for _, g in df.dropna(subset=[a, b]).groupby("race_id") if len(g) > 3]
    return float(np.mean(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(MDIR / "config.yaml")
    assert cfg["split"] == "dev"
    entries0 = P.load_entries()
    frozen = load_yaml(ROOT / cfg["frozen_path"])
    lat = frozen[cfg["latent_ref"]["key"]][cfg["latent_ref"]["cell"]]
    pace = pd.read_parquet(ROOT / cfg["pace_table"])
    entries, _ = M09.build_features(entries0, pace, cfg)
    entries, finfo = add_form(entries, cfg)
    first = entries.groupby("race_id")["season"].first()
    dev_ids = first[(first >= SPLITS["dev"][0]) & (first <= SPLITS["dev"][1])].index
    rid_chk = int(dev_ids[len(dev_ids) // 2])
    lk = leak_check_form(entries0, cfg, rid_chk)
    log(f"form_x info {finfo}; truncation check at race {rid_chk}: max abs diff {lk:.2e}")
    assert lk == 0.0
    dv = entries[entries["season"].between(*SPLITS["dev"])]
    corr = {"form_x~pace_hist_x": within_race_rank_corr(dv, FORM, HIST), "form_x~q1off_x": within_race_rank_corr(dv, FORM, M09.OFF),
            "form_x~con_ew32": within_race_rank_corr(dv, FORM, "con_ew32"), "pace_hist_x~con_ew32": within_race_rank_corr(dv, HIST, "con_ew32"),
            "pace_hist_x~q1off_x": within_race_rank_corr(dv, HIST, M09.OFF)}
    log(f"dev within-race rank correlations (mean over races): {corr}")
    stamp = U.code_stamp(USED)
    grid_lam, ents = list(cfg["lam_x_grid"]), entries
    if args.quick:
        dev_ids, grid_lam = dev_ids[:6], grid_lam[:2]
        ents = entries[entries["race_id"] <= dev_ids.max()].reset_index(drop=True)

    def run(label, model, refit):
        t = time.time()
        p, m = walk_forward(ents, model, cfg["split"], refit_every=refit)
        if args.quick:
            p, m = p[p["race_id"].isin(dev_ids)], m[m["race_id"].isin(dev_ids)]
        log(f"  {label}: rho={m['spearman'].mean():.4f} ({time.time() - t:.0f}s)")
        return p, m

    rows, best = [], {}
    for comp, cols in RUNGS.items():
        for lam in grid_lam:
            _, m = run(f"tune {comp} lam_x={lam}", build_model(dict(lat, x_cols=cols, lam_x=lam), ents), cfg["tune_refit_every"])
            rows.append(dict(component=comp, family="latent_111", lam_x=lam, spearman=m["spearman"].mean(), mae=m["mae"].mean(),
                             exact=m["exact"].mean(), n_races=len(m)))
        t = pd.DataFrame([r for r in rows if r["component"] == comp])
        best[comp] = float(t.loc[t["spearman"].idxmax(), "lam_x"])
    tune = pd.DataFrame(rows)
    log(f"lam_x chosen {best}")
    runs = {}
    for comp, cols in RUNGS.items():
        c = dict(lat, x_cols=cols, lam_x=best[comp])
        runs[comp] = (c, *run(comp, build_model(c, ents), cfg["refit_every"]))
    if args.quick:
        log("quick: nothing written")
        return

    key, nb = cfg["tie_break"], cfg["bootstrap"]
    tabs = {k: A.race_table(U.outcome_free(v[1], key)) for k, v in runs.items()}
    res = MDIR / "results"
    tabs["ref"] = pd.read_csv(next(res.glob("*_m09_ref_latent_111_dev")) / "outcome_free_table.csv")
    tabs["hist"] = pd.read_csv(next(res.glob("*_m09_hist_latent_111_dev")) / "outcome_free_table.csv")
    C1, C2 = tabs["M09_ctrl_form"], tabs["M09_ctrl_form_hist"]
    comps = {"C1_form_vs_ref": (tabs["ref"], C1), "hist_vs_C1": (C1, tabs["hist"]), "C2_vs_C1": (C1, C2)}
    allg, seas = [], []
    for name, (a, b) in comps.items():
        allg.append(A.paired_gain(a, b, n_boot=nb["n_boot"], seed=nb["seed"]).assign(comparison=name))
        for s in (2021, 2022, 2023):
            gs = A.paired_gain(a[a["season"] == s], b[b["season"] == s], ["spearman", "mae", "exact", "spearman_fin", "mae_fin"],
                               n_boot=nb["n_boot"], seed=nb["seed"])
            seas.append(gs.assign(comparison=name, season=s))
    full = pd.concat(allg, ignore_index=True)
    full["verdict"] = [A.verdict(r) for _, r in full.iterrows()]
    by_season = pd.concat(seas, ignore_index=True)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print("\nTUNING\n", tune.round(4).to_string(index=False))
    print("\nCONTROL PAIRED GAINS\n", full[full["metric"].isin(["spearman", "spearman_fin", "mae", "exact"])].round(4).to_string(index=False))
    print("\nPER SEASON (rho)\n", by_season[by_season["metric"] == "spearman"].round(4).to_string(index=False))

    tmp = res / "_lb_tmp_ctrl.csv"
    tmp.unlink(missing_ok=True)
    labels = {"M09_ctrl_form": "m09_ctrl_form_latent_111", "M09_ctrl_form_hist": "m09_ctrl_form_hist_latent_111"}
    for k, (c, p, m) in runs.items():
        extra = dict(component=k, frozen_hyperparameters="plan-1; lam_x tuned on dev",
                     status="DIAGNOSTIC post hoc car-form control, no decision weight; dev only")
        rid_ = save_run(MDIR, labels[k], dict(c, **extra), p, m, cfg["split"], "M09", c.get("cell", ""), leaderboard=tmp)
        pth = res / rid_ / "config.yaml"
        cc = load_yaml(pth)
        cc["git_commit"] = stamp
        save_yaml(cc, pth)
        tabs[k].to_csv(res / rid_ / "outcome_free_table.csv", index=False)
    new_rows = pd.read_csv(tmp)
    tmp.unlink()
    new_rows["label"] = new_rows["label"] + "_diagnostic"
    tune.to_csv(res / "control_tuning_lam_x.csv", index=False)
    full.to_csv(res / "control_paired_gains_dev.csv", index=False)
    by_season.to_csv(res / "control_paired_gains_by_season.csv", index=False)
    (res / "control_feature_info.json").write_text(json.dumps(dict(form_x=finfo, truncation_check_max_abs_diff=lk, truncation_race_id=rid_chk,
                                                                    lam_x_chosen=best, rank_corr_dev=corr, configurations_evaluated=15 + len(tune)), indent=2), encoding="utf-8")
    U.append_leaderboard(new_rows, ROOT / cfg["leaderboard"])
    log(f"done; stamp {stamp}")


if __name__ == "__main__":
    main()
