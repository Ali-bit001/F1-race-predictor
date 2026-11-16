"""M11 likelihood variants of the latent model (cell 111), development block only (PLAN3.md section 4, R5).

    python methodologies/M11_likelihood_variants/run.py [--quick]

Re-runs the reference latent_111 and the three variants (finishers-only, truncated top-k, reverse), each alone, with the
frozen plan-1 hyper-parameters (no re-tuning), walk-forward with a refit before every dev race. Checks that the
reference reproduces the stored plan-1 run, writes one run folder per run, upserts three attribution rows
(component vs latent_111) and appends the leaderboard rows at the very end.
--quick: first 6 dev races only, nothing written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

from pitlane import pipeline as P
from pitlane.config import METHODS, ROOT, SPLITS, load_yaml, save_yaml
from pitlane.evaluation.harness import _git_hash, save_run, walk_forward
from pitlane.models.registry import build_model
from pitlane.plan3 import attribution as A
from pitlane.plan3 import decode as DEC

MDIR = Path(__file__).resolve().parent
USED_FILES = ["src/pitlane/models/latent.py", "src/pitlane/plan3/attribution.py", "src/pitlane/evaluation/harness.py",
              "methodologies/M11_likelihood_variants/run.py", "methodologies/M11_likelihood_variants/config.yaml"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def code_stamp() -> str:
    """`<hash>` if the files used are clean, else `<hash>+dirty-<sha1 of the used files>` (as the M17/A5/A4 runs)."""
    head = _git_hash()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--"] + USED_FILES, text=True, cwd=ROOT).strip()
    if not dirty:
        return head
    h = hashlib.sha1()
    for f in USED_FILES:
        h.update(f.encode())
        h.update((ROOT / f).read_bytes())
    return f"{head}+dirty-{h.hexdigest()[:12]}"


def outcome_free(preds: pd.DataFrame, key: str) -> pd.DataFrame:
    """Predicted order with ties broken by `key` (driverId), never by input row order (which is finishing order)."""
    out = preds.copy()
    sc = np.empty(len(out))
    for _, ix in out.groupby("race_id").indices.items():
        sc[ix] = DEC.break_ties(out["score"].to_numpy(float)[ix], out[key].to_numpy()[ix])
    out["score"] = sc
    return out


def stored_run(frozen_label: str, cell_label: str, split: str = "dev") -> pd.DataFrame:
    lb = pd.read_csv(ROOT / "results" / "leaderboard.csv")
    row = lb[(lb["label"] == cell_label) & (lb["split"] == split)].iloc[-1]
    return pd.read_csv(ROOT / row["run_dir"] / "per_race_metrics.csv")


def max_diff(new: pd.DataFrame, old: pd.DataFrame) -> tuple[float, int]:
    """Largest absolute difference over all shared numeric columns, matched by race_id."""
    a, b = new.set_index("race_id"), old.set_index("race_id")
    cols = [c for c in a.columns if c in b.columns and c != "race_id"]
    common = a.index.intersection(b.index)
    d = (a.loc[common, cols].astype(float) - b.loc[common, cols].astype(float)).abs()
    return float(np.nanmax(d.to_numpy())), len(common)


def stage_shares(entries: pd.DataFrame, k: int, upto_race: int, dev_lo: int) -> pd.DataFrame:
    """Descriptive: retirees and the share of PL stages each switch removes. Two scopes: the 66 dev races, and every
    race up to the last dev race (what the last refit trains on). Stages = entries per race (one pick each)."""
    rows = []
    for scope, d in (("dev races", entries[(entries["race_id"] >= dev_lo) & (entries["race_id"] <= upto_race)]),
                     ("all races up to last dev race", entries[entries["race_id"] <= upto_race])):
        n_ent, n_ret = len(d), int((~d["classified"]).sum())
        n_top = int(d.groupby("race_id").size().clip(upper=k).sum())
        rows.append(dict(scope=scope, races=d["race_id"].nunique(), entries=n_ent, retirees=n_ret,
                         retiree_share=n_ret / n_ent, finishers_removed_share=n_ret / n_ent,
                         topk_removed_share=1 - n_top / n_ent, reverse_removed_share=0.0))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MDIR / "config.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["split"] == "dev", "plan 3 / R5: development block only"
    t0 = time.time()
    entries = P.load_entries()
    base = load_yaml(ROOT / cfg["frozen_path"])[cfg["frozen_key"]]["111"]
    stamp = code_stamp()
    split = cfg["split"]
    if args.quick:
        first = entries.groupby("race_id")["season"].first()
        dev_ids = first[(first >= SPLITS["dev"][0]) & (first <= SPLITS["dev"][1])].index[:6]
        entries_run, split_run = entries[entries["race_id"] <= dev_ids.max()].reset_index(drop=True), "dev"
    else:
        entries_run, split_run = entries, "dev"

    def run(c: dict):
        t = time.time()
        model = build_model({k: v for k, v in c.items() if k != "label"}, entries_run)
        p, m = walk_forward(entries_run, model, split_run, refit_every=cfg["refit_every"])
        if args.quick:
            m = m[m["race_id"].isin(dev_ids)]
            p = p[p["race_id"].isin(dev_ids)]
        log(f"  {c.get('label')}: rho={m['spearman'].mean():.4f} mae={m['mae'].mean():.3f} ({time.time() - t:.0f}s)")
        return p, m

    # ---- reference + variants (same harness, same races)
    runs = {"latent_111": (dict(base), *run(dict(base, label="reference")))}
    for comp, v in cfg["variants"].items():
        c = dict(base, **v["switches"])
        log(f"{comp}: {v['switches']}")
        runs[comp] = (c, *run(dict(c, label=v["label"])))

    # ---- reproduction check: switches off must equal the stored plan-1 run
    rep = {}
    lab = {"latent_111": "latent_111"}
    for cell in cfg["reproduction_extra_cells"]:
        c = dict(load_yaml(ROOT / cfg["frozen_path"])[cfg["frozen_key"]][cell])
        log(f"re-running latent_{cell} for the reproduction check")
        p, m = run(dict(c, label=f"latent_{cell}"))
        runs[f"latent_{cell}"] = (c, p, m)
        lab[f"latent_{cell}"] = f"latent_{cell}"
    for k, l in lab.items():
        if args.quick:
            continue
        d, n = max_diff(runs[k][2], stored_run(None, l))
        sc = runs[k][1].merge(pd.read_parquet(ROOT / pd.read_csv(ROOT / "results" / "leaderboard.csv")
                                              .query("label == @l and split == 'dev'").iloc[-1]["run_dir"] / "predictions.parquet"),
                              on=["race_id", "driverId"], suffixes=("_n", "_o"))
        ds = float((sc["score_n"] - sc["score_o"]).abs().max())
        rep[k] = dict(max_abs_diff_per_race_metrics=d, max_abs_diff_scores=ds, races=n, entries=len(sc))
        log(f"REPRODUCTION {k}: max |per-race metric diff| = {d:.3e}, max |score diff| = {ds:.3e} ({n} races, {len(sc)} entries)")

    # ---- descriptive extra
    last_dev = int(entries.loc[entries["season"] <= SPLITS["dev"][1], "race_id"].max())
    dev_lo = int(entries.loc[entries["season"] >= SPLITS["dev"][0], "race_id"].min())
    shares = stage_shares(entries, int(cfg["variants"]["M11_topk10"]["switches"]["topk"]), last_dev, dev_lo)
    log("DESCRIPTIVE (not a component)\n" + shares.round(4).to_string(index=False))

    # ---- paired attribution on the outcome-free ordering
    tabs = {k: A.race_table(outcome_free(v[1], cfg["tie_break"])) for k, v in runs.items()}
    ties = {k: int(sum(len(np.unique(g["score"])) < len(g) for _, g in v[1].groupby("race_id"))) for k, v in runs.items()}
    log(f"races with tied scores (before the tie-break): {ties}")
    nb = cfg["bootstrap"]
    ref = tabs["latent_111"]
    allg, rows = [], []
    for comp in cfg["variants"]:
        g = A.paired_gain(ref, tabs[comp], n_boot=nb["n_boot"], seed=nb["seed"])
        g.insert(0, "component", comp)
        allg.append(g)
    full = pd.concat(allg, ignore_index=True)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    keep = full[full["metric"].isin(["spearman", "mae", "exact", "exact_fin", "spearman_fin", "mae_fin", "exact_1_3", "exact_4_10", "exact_11p"])]
    print("\nPAIRED GAINS vs latent_111 (positive = variant better; 95% bootstrap CI over races)\n" + keep.round(4).to_string(index=False))
    print("\nMEANS\n", pd.DataFrame({k: t[["spearman", "mae", "exact", "exact_fin", "spearman_fin", "mae_fin"]].mean()
                                     for k, t in tabs.items()}).T.round(4).to_string())
    if args.quick:
        log("quick run: nothing written")
        return

    # ---- write run folders (leaderboard rows are collected, not written, until the very end)
    tmp_lb = MDIR / "results" / "_lb_tmp.csv"
    tmp_lb.parent.mkdir(parents=True, exist_ok=True)
    tmp_lb.unlink(missing_ok=True)
    run_ids = {}
    for k in ["latent_111"] + list(cfg["variants"]):
        c, p, m = runs[k]
        label = "latent_111" if k == "latent_111" else cfg["variants"][k]["label"]
        label = "m11_ref_latent_111" if k == "latent_111" else "m11_" + label
        rid = save_run(MDIR, label, dict(c, component=k, frozen_hyperparameters="plan-1, not re-tuned"), p, m, split,
                       "M11", "111", leaderboard=tmp_lb)
        pth = MDIR / "results" / rid / "config.yaml"
        cc = load_yaml(pth)
        cc["git_commit"] = stamp
        save_yaml(cc, pth)
        run_ids[k] = rid
    new_rows = pd.read_csv(tmp_lb)
    tmp_lb.unlink()
    tabs["latent_111"].to_csv(MDIR / "results" / f"{run_ids['latent_111']}_outcome_free_table.csv", index=False)
    for comp in cfg["variants"]:
        g = full[full["component"] == comp].drop(columns="component")
        rows.append(A.attribution_row(comp, "latent_111", g, cfg["verdict_metric"], run_ids[comp], len(tabs[comp]), declared="yes"))
    full.to_csv(MDIR / "results" / "paired_gains_dev.csv", index=False)
    shares.to_csv(MDIR / "results" / "descriptive_stage_shares.csv", index=False)
    (MDIR / "results" / "reproduction_check.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    A.upsert_attribution(rows, ROOT / cfg["attribution"])
    # re-read right before writing: other work packages append to the same file
    lb_path = ROOT / cfg["leaderboard"]
    header = pd.read_csv(lb_path, nrows=0).columns.tolist()
    assert not set(new_rows["run_id"]) & set(pd.read_csv(lb_path, usecols=["run_id"])["run_id"]), "run_id already in leaderboard"
    add = new_rows.reindex(columns=header)
    text = lb_path.read_text(encoding="utf-8")
    with open(lb_path, "a", encoding="utf-8", newline="") as f:      # append only: existing rows are not rewritten
        if not text.endswith("\n"):
            f.write("\n")
        add.to_csv(f, index=False, header=False, lineterminator="\n")
    log(f"verdicts: {[(r['component'], r['verdict']) for r in rows]}")
    log(f"done in {time.time() - t0:.0f}s; code stamp {stamp}")


if __name__ == "__main__":
    main()
