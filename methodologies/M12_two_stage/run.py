"""M12 two-stage Friday forecast, development block only (PLAN3.md R10).

    python methodologies/M12_two_stage/run.py [--quick]

Stage 1 (ridge on history + practice) predicts the qualifying order; stage 2 is the frozen plan-1 H2 model (latent_111) applied
with the predicted qualifying position in place of the real one. Compared with the plan-2 H1 model (re-run here and checked against
the stored plan-2 run) on the dev races that have practice data (2023). Also scores how well qualifying itself is predicted.
--quick: nothing is written.
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
from scipy.stats import rankdata

from pitlane.config import ROOT, SPLITS, load_yaml, save_yaml
from pitlane.evaluation.harness import save_run, split_races, walk_forward
from pitlane.models.registry import build_model
from pitlane.models.two_stage import QualiStage1, TwoStageH2, add_quali_history, rank_within
from pitlane.plan2 import pipeline2 as Q
from pitlane.plan3 import attribution as A
from pitlane.plan3 import runutil as U

MDIR = Path(__file__).resolve().parent
USED = ["src/pitlane/models/two_stage.py", "src/pitlane/models/latent.py", "src/pitlane/plan3/attribution.py", "src/pitlane/plan3/runutil.py",
        "src/pitlane/evaluation/harness.py", "methodologies/M12_two_stage/run.py", "methodologies/M12_two_stage/config.yaml"]
VERDICT_METRICS = ["spearman", "mae", "exact", "exact_fin", "exact_1_3", "exact_4_10", "exact_11p", "spearman_fin", "mae_fin"]
PRIMARY = {"spearman", "mae"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def leakage_check(entries: pd.DataFrame, race_ids: list[int], hl: tuple[int, ...]) -> float:
    """Recompute the qualifying-history columns on the data cut at race r with race r's qualifying blanked; the values for
    race r must equal the ones from the full frame. Returns the largest absolute difference."""
    cols = [f"drv_q_ew{h}" for h in hl] + [f"con_q_ew{h}" for h in hl]
    worst = 0.0
    for r in race_ids:
        cut = entries[entries["race_id"] <= r].drop(columns=cols).reset_index(drop=True).copy()
        cut.loc[cut["race_id"] == r, "quali_norm"] = np.nan
        new = add_quali_history(cut, hl)
        a = new[new["race_id"] == r][cols].to_numpy(float)
        b = entries[entries["race_id"] == r][cols].to_numpy(float)
        worst = max(worst, float(np.abs(a - b).max()))
    return worst


def quality_table(d: pd.DataFrame) -> pd.DataFrame:
    """Per-race Spearman rho and MAE of a predicted qualifying rank (`p`) against the actual one (`a`)."""
    rows = []
    for rid, g in d.groupby("race_id"):
        p, a = g["p"].to_numpy(float), g["a"].to_numpy(float)
        rows.append(dict(race_id=rid, spearman=float(np.corrcoef(p, a)[0, 1]), mae=float(np.abs(p - a).mean())))
    return pd.DataFrame(rows)


def actual_quali_rank(entries: pd.DataFrame) -> pd.Series:
    return entries.groupby("race_id")["quali_pos"].transform(lambda s: pd.Series(rankdata(s.to_numpy(float), method="average"), index=s.index))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MDIR / "config.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["split"] == "dev", "plan 3 / R10: development block only"
    t0 = time.time()
    s1c = cfg["stage1"]
    stamp = U.code_stamp(USED)
    entries = add_quali_history(Q.load_all(), tuple(s1c["halflives_quali"]))
    assert (entries.index == np.arange(len(entries))).all()
    h2 = load_yaml(ROOT / cfg["frozen_path"])[cfg["h2"]["key"]][cfg["h2"]["cell"]]
    h1 = load_yaml(ROOT / cfg["frozen_plan2_path"])[cfg["h1"]["key"]][cfg["h1"]["cell"]]
    split = cfg["split"]
    _, counted = split_races(entries, split)
    prac = entries.groupby("race_id")["fp_gap"].apply(lambda s: bool(s.notna().any()))
    prac_ids = [r for r in counted if prac[r]]
    non_prac = [r for r in counted if not prac[r]]
    log(f"dev races {len(counted)}; with practice data {len(prac_ids)} (seasons {sorted(entries[entries.race_id.isin(prac_ids)].season.unique())}); "
        f"dev races without: {len(non_prac)}")
    leak = leakage_check(entries, [prac_ids[0], prac_ids[len(prac_ids) // 2], prac_ids[-1], counted[0]], tuple(s1c["halflives_quali"]))
    log(f"LEAKAGE CHECK (quali-history features, race r quali blanked and data cut at r): max |diff| = {leak:.3e}")
    assert leak == 0.0

    stage_cols = s1c["history_cols"] + s1c["practice_cols"]
    # ---- runs
    t = time.time()
    p_h1, m_h1 = walk_forward(entries, build_model(h1, entries), split, refit_every=cfg["refit_every"])
    log(f"  H1 re-run: rho={m_h1['spearman'].mean():.4f} mae={m_h1['mae'].mean():.3f} ({time.time() - t:.0f}s)")
    t = time.time()
    ts = TwoStageH2(entries, h2, stage_cols, s1c["alpha"])
    p_ts, m_ts = walk_forward(entries, ts, split, refit_every=cfg["refit_every"])
    log(f"  two-stage: rho={m_ts['spearman'].mean():.4f} mae={m_ts['mae'].mean():.3f} ({time.time() - t:.0f}s)")

    # ---- reproduction of the stored plan-2 H1 run
    old_pm, old_pr = U.stored_run(cfg["plan2_leaderboard"], cfg["h1"]["label"])
    d, n = U.max_diff(m_h1, old_pm)
    ds, ne = U.score_diff(p_h1, old_pr)
    rep = {cfg["h1"]["label"]: dict(max_abs_diff_per_race_metrics=d, max_abs_diff_scores=ds, races=n, entries=ne)}
    log(f"REPRODUCTION {cfg['h1']['label']}: max |per-race metric diff| = {d:.3e}, max |score diff| = {ds:.3e} ({n} races, {ne} entries)")

    # ---- order metrics on the same races, outcome-free ties
    key = cfg["tie_break"]
    tabs = {"two_stage": A.race_table(U.outcome_free(p_ts, key)), "h1": A.race_table(U.outcome_free(p_h1, key))}
    _, pr0 = U.stored_run(cfg["plan2_leaderboard"], cfg["baselines"]["h0_label"])
    _, pr2 = U.stored_run(cfg["plan1_leaderboard"], "latent_111")
    tabs["h0"], tabs["h2_real_quali"] = A.race_table(U.outcome_free(pr0, key)), A.race_table(U.outcome_free(pr2, key))
    nb = cfg["bootstrap"]
    sub = lambda t, ids: t[t["race_id"].isin(ids)].reset_index(drop=True)
    gains = {}
    for tag, ids in (("practice_2023", prac_ids), ("no_practice_2021_22", non_prac), ("all_66", counted)):
        g = A.paired_gain(sub(tabs["h1"], ids), sub(tabs["two_stage"], ids), n_boot=nb["n_boot"], seed=nb["seed"])
        g["verdict"] = [A.verdict(r) for _, r in g.iterrows()]
        gains[tag] = g
    ctx = {}
    for name, other in (("two_stage_vs_h0", "h0"), ("two_stage_vs_h2_real_quali", "h2_real_quali"), ("h1_vs_h0", None)):
        ref, cand = (tabs[other], tabs["two_stage"]) if other else (tabs["h0"], tabs["h1"])
        ctx[name] = A.paired_gain(sub(ref, prac_ids), sub(cand, prac_ids), VERDICT_METRICS, n_boot=nb["n_boot"], seed=nb["seed"])
    means = pd.DataFrame({k: sub(t, prac_ids)[["spearman", "mae", "exact", "exact_fin", "spearman_fin", "mae_fin", "exact_1_3", "exact_4_10", "exact_11p"]].mean()
                          for k, t in tabs.items()}).T

    # ---- how well is qualifying itself predicted?
    s1 = pd.concat(ts.log, ignore_index=True)
    s1["a"] = entries.set_index(["race_id", "driverId"]).loc[list(zip(s1["race_id"], s1["driverId"])), "quali_pos"].to_numpy()
    s1["a"] = s1.groupby("race_id")["a"].transform(lambda s: rankdata(s.to_numpy(float), method="average"))
    s1 = s1.rename(columns={"q1_rank": "p"})
    qd = {"stage1": s1[s1["race_id"].isin(counted)][["race_id", "driverId", "p", "a"]]}
    ent = entries[entries["race_id"].isin(counted)][["race_id", "driverId", "quali_pos", "quali_norm", "lineage"]].copy()
    ent["a"] = actual_quali_rank(entries).loc[ent.index]
    last = entries.groupby("driverId")["quali_norm"].shift(1).fillna(0.5).loc[ent.index]
    ent["p"] = np.concatenate([rank_within(last.loc[g.index].to_numpy(), g["driverId"].to_numpy()) for _, g in ent.groupby("race_id", sort=False)])
    qd["prev_race_quali_order"] = ent[["race_id", "driverId", "p", "a"]]
    for nm, pr in (("h0_strength_order", pr0), ("h1_score_order", p_h1)):
        z = pr[["race_id", "driverId", "score"]].merge(ent[["race_id", "driverId", "a"]], on=["race_id", "driverId"])
        z["p"] = np.concatenate([rank_within(-g["score"].to_numpy(), g["driverId"].to_numpy()) for _, g in z.groupby("race_id", sort=False)])
        qd[nm] = z[["race_id", "driverId", "p", "a"]]
    rows = []                                    # stage 1 without the practice columns
    for r in counted:
        tr, te = entries[entries["race_id"] < r], entries[entries["race_id"] == r]
        raw = QualiStage1(s1c["history_cols"], s1c["alpha"]).fit(tr).predict(te)
        rows.append(pd.DataFrame({"race_id": r, "driverId": te["driverId"].to_numpy(), "p": rank_within(raw, te["driverId"].to_numpy()),
                                  "a": ent.set_index(["race_id", "driverId"]).loc[list(zip(te["race_id"], te["driverId"])), "a"].to_numpy()}))
    qd["stage1_no_practice"] = pd.concat(rows, ignore_index=True)
    qt = {k: quality_table(v) for k, v in qd.items()}
    qmeans, qgain = [], []
    for tag, ids in (("practice_2023", prac_ids), ("no_practice_2021_22", non_prac), ("all_66", counted)):
        for k, t_ in qt.items():
            s = sub(t_, ids)
            qmeans.append(dict(subset=tag, predictor=k, races=len(s), spearman=s["spearman"].mean(), mae=s["mae"].mean()))
            if k != "stage1":
                g = A.paired_gain(s, sub(qt["stage1"], ids), ["spearman", "mae"], n_boot=nb["n_boot"], seed=nb["seed"])
                g.insert(0, "subset", tag)
                g.insert(1, "stage1_versus", k)
                qgain.append(g)
    qmeans, qgain = pd.DataFrame(qmeans), pd.concat(qgain, ignore_index=True)
    # stage-1 log must equal the no-practice-free full model on the same races (consistency: same ridge, columns differ only)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print("\nMEANS on the practice races (outcome-free ties)\n", means.round(4).to_string())
    for tag, g in gains.items():
        print(f"\nPAIRED GAIN two_stage minus H1, subset {tag} (n={g['n'].iloc[0]}; positive = two-stage better)\n"
              + g[g["metric"].isin(VERDICT_METRICS)].round(4).to_string(index=False))
    for k, g in ctx.items():
        print(f"\nCONTEXT (practice races, descriptive) {k}\n" + g.round(4).to_string(index=False))
    print("\nQUALIFYING PREDICTION QUALITY (mean per race)\n" + qmeans.round(4).to_string(index=False))
    print("\nSTAGE 1 vs BASELINES (paired; positive = stage 1 better)\n" + qgain.round(4).to_string(index=False))
    if args.quick:
        log("quick run: nothing written")
        return

    # ---- write
    tmp_lb = MDIR / "results" / "_lb_tmp.csv"
    tmp_lb.parent.mkdir(parents=True, exist_ok=True)
    tmp_lb.unlink(missing_ok=True)
    out = {}
    for k, label, c, p, m in (("h1", "m12_ref_h1_latent_111", dict(h1, component="plan2_H1_latent_111"), p_h1, m_h1),
                              ("two_stage", "m12_two_stage", dict(h2, component="M12_two_stage", stage1=s1c, h2_inputs_from_stage1=
                                                                  {"quali_norm": "stage-1 predicted rank", "grid_minus_quali": 0, "q1_gap": "mean by predicted rank, earlier races",
                                                                   "pit_start": 0}), p_ts, m_ts)):
        rid = save_run(MDIR, label, dict(c, frozen_hyperparameters="plan-1 H2 / plan-2 H1, not re-tuned"), p, m, split, "M12", "111", leaderboard=tmp_lb)
        cc = load_yaml(MDIR / "results" / rid / "config.yaml")
        cc["git_commit"] = stamp
        save_yaml(cc, MDIR / "results" / rid / "config.yaml")
        tabs[k].to_csv(MDIR / "results" / rid / "outcome_free_table.csv", index=False)
        out[k] = rid
    s1.drop(columns=["quali_pos"]).to_parquet(MDIR / "results" / out["two_stage"] / "stage1_predictions.parquet", index=False)
    new_rows = pd.read_csv(tmp_lb)
    tmp_lb.unlink()
    res = MDIR / "results"
    allg = pd.concat([g.assign(subset=tag) for tag, g in gains.items()], ignore_index=True)
    allg.to_csv(res / "paired_gains_dev.csv", index=False)
    allg[(allg["subset"] == "practice_2023") & allg["metric"].isin(VERDICT_METRICS)][["metric", "gain", "lo", "hi", "n", "verdict"]] \
        .assign(component="M12_two_stage", reference="plan2_H1_latent_111", primary=lambda d: d["metric"].isin(PRIMARY)) \
        .to_csv(res / "verdicts.csv", index=False)
    pd.concat([g.assign(comparison=k) for k, g in ctx.items()]).to_csv(res / "context_practice_races.csv", index=False)
    means.to_csv(res / "means_practice_races.csv")
    qmeans.to_csv(res / "quali_prediction_quality.csv", index=False)
    qgain.to_csv(res / "quali_prediction_paired.csv", index=False)
    (res / "reproduction_check.json").write_text(json.dumps(dict(rep, leakage_check_max_abs_diff=leak, practice_races=len(prac_ids),
                                                                  practice_race_ids=prac_ids), indent=2), encoding="utf-8")
    g = gains["practice_2023"].drop(columns="verdict")
    row = A.attribution_row("M12_two_stage", "plan2_H1_latent_111", g, cfg["verdict_metric"], out["two_stage"], len(prac_ids), declared="yes")
    A.upsert_attribution([row], ROOT / cfg["attribution"])
    U.append_leaderboard(new_rows, ROOT / cfg["leaderboard"])
    log(f"verdict (rho): {row['verdict']}; done in {time.time() - t0:.0f}s; code stamp {stamp}")


if __name__ == "__main__":
    main()
