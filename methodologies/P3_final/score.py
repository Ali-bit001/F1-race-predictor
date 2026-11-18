"""PLAN3 R15: score the frozen plan-3 models. Third look on the old test blocks (no decision weight) and the fresh holdout.

    python methodologies/P3_final/score.py                    # score what is new; no-op if nothing is new
    python methodologies/P3_final/score.py --write-rowhash    # once, at R15: write freeze_rowhash.json (refuses to overwrite)
    python methodologies/P3_final/score.py --no-write         # compute and print, write nothing

Nothing is tuned or selected here. Rules: methodologies/P3_final/HOLDOUT_PREREG.md (binding) and FREEZE.md.
  1. verify_freeze (frozen_config.yaml hash) and the row hashes of every row that existed at the freeze (stop on a mismatch)
  2. third look on test_stable (2024-25) and test_shift (2026 R1-20), once: final, reverse_only, final_minus_reverse, latent_111,
     plan2_H2 (the same model as latent_111, see README), grid and H1; paired bootstrap, 10,000 resamples, seed 0; no Holm
  3. fresh holdout = races with race_id > 271 and a classified result: n is printed and saved; no verdict below 20 races;
     T1-T3 with Holm at the first n >= 20 and again at the first n >= 40; each reading is written once and never overwritten
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import date, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

from pitlane.config import ROOT, load_yaml, save_yaml
from pitlane.evaluation.harness import _git_hash, save_run, walk_forward
from pitlane.models.registry import build_model
from pitlane.plan2 import dnf as D
from pitlane.plan2 import pipeline2 as Q
from pitlane.plan2 import simulate as S
from pitlane.plan3 import attribution as A
from pitlane.plan3 import decode as DEC
from pitlane.plan3 import final as F
from pitlane.plan3 import holdout as H
from pitlane.plan3 import runutil as U
from pitlane.evaluation.metrics import rps_score

MDIR = Path(__file__).resolve().parent
RES = MDIR / "results"
PLAN3 = ROOT / "results" / "plan3"
USED = ["src/pitlane/plan3/holdout.py", "src/pitlane/plan3/final.py", "src/pitlane/plan3/attribution.py", "src/pitlane/plan3/runutil.py",
        "src/pitlane/evaluation/harness.py", "methodologies/P3_final/score.py", "methodologies/P3_final/config.yaml",
        "methodologies/P3_final/frozen_config.yaml", "methodologies/P3_final/HOLDOUT_PREREG.md"]
THIRD_LOOK = "third look, no decision weight"
POINT = ["spearman", "mae", "exact", "spearman_fin", "mae_fin", "exact_fin", "exact_1_3", "exact_4_10", "exact_11p", "within1", "within2", "within3"]
PROB = ["rps", "logloss_points"]
# Primary tests (HOLDOUT_PREREG section 4): name, candidate, reference, metric. Pre-registered, not configurable.
TESTS = [("T1", "final", "latent_111", "spearman"), ("T2", "final", "final_minus_reverse", "mae"), ("T3", "final", "reverse_only", "spearman")]
# Other contrasts, descriptive (section 5): name, candidate, reference, primary metric.
OTHERS = [("exact_vs_grid:final", "final", "grid", "exact"), ("exact_vs_grid:reverse_only", "reverse_only", "grid", "exact"),
          ("exact_vs_grid:latent_111", "latent_111", "grid", "exact"), ("final_vs_plan2_H2", "final", "plan2_H2", "spearman")]
S3_CONTRAST = ("S3:final_vs_latent_111", "final", "latent_111", "rps")
# Models whose third-look run already exists in an earlier plan's leaderboard (label, leaderboard): not appended again.
EXISTING = {"latent_111": [("results/leaderboard.csv", "latent_111"), ("results/plan2/leaderboard.csv", "m06_L_r0_111")],
            "plan2_H2": [("results/plan2/leaderboard.csv", "m06_L_r0_111")], "grid": [("results/leaderboard.csv", "grid")],
            "H1": [("results/plan2/leaderboard.csv", "h1_latent_111")]}


# Stored reference runs, pinned to explicit run_ids (a later run with the same label must not change the references).
PINNED = {("results/leaderboard.csv", "latent_111", "test_stable"): "20261115-1848_latent_111_test_stable",
          ("results/leaderboard.csv", "latent_111", "test_shift"): "20261115-1850_latent_111_test_shift",
          ("results/leaderboard.csv", "grid", "test_stable"): "20261115-1848_grid_test_stable",
          ("results/leaderboard.csv", "grid", "test_shift"): "20261115-1850_grid_test_shift",
          ("results/plan2/leaderboard.csv", "m06_L_r0_111", "test_stable"): "20261115-2046_m06_L_r0_111_test_stable",
          ("results/plan2/leaderboard.csv", "m06_L_r0_111", "test_shift"): "20261115-2046_m06_L_r0_111_test_shift",
          ("results/plan2/leaderboard.csv", "h1_latent_111", "test_stable"): "20261115-2046_h1_latent_111_test_stable",
          ("results/plan2/leaderboard.csv", "h1_latent_111", "test_shift"): "20261115-2046_h1_latent_111_test_shift"}
THETA_RUN = ("results/plan3/leaderboard.csv", "p3_S3refit_latent_111", "dev", "20261118-0705_p3_S3refit_latent_111_dev")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------ models and tables
def model_configs(frozen_cfg: dict, frozen1: dict) -> dict[str, dict]:
    """The five order models of the pre-registration, built from the frozen files. `plan2_H2` is not run separately: it is the
    plan-1 frozen model at the plan-2 information set H2 (methodologies/M12_two_stage/README.md), i.e. `latent_111`."""
    base = dict(frozen1["M03"]["111"])
    final = dict(frozen_cfg["horizons"]["H2"]["model"])
    lam_x = float(final["lam_x"])
    cfgs = {"latent_111": F.latent_cfg(base), "final": F.latent_cfg(base, reverse=True, pace_hist=True, lam_x=lam_x),
            "reverse_only": F.latent_cfg(base, reverse=True), "final_minus_reverse": F.latent_cfg(base, pace_hist=True, lam_x=lam_x)}
    if cfgs["final"] != final:
        raise RuntimeError(f"frozen H2 model {final} differs from the model built from configs/frozen.yaml {cfgs['final']}")
    return cfgs


def tab(p: pd.DataFrame, key: str = "driverId") -> pd.DataFrame:
    return A.race_table(U.outcome_free(p, key))


def grid_table(preds: pd.DataFrame, key: str = "driverId") -> tuple[pd.DataFrame, pd.DataFrame]:
    gp = preds[["race_id", "season", "round", "driverId", "lineage", "pos", "grid", "classified"]].copy()
    gp["score"] = np.concatenate([DEC.grid_utility(g["grid"].to_numpy(float), g[key].to_numpy()) for _, g in gp.groupby("race_id", sort=False)])
    return gp, A.race_table(gp)


def sim_tables(races: list[dict], mats: list[np.ndarray], keys: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(predictions, per-race table) of the S3 matrix; D0 order (sort by -E[position]) with an outcome-free tie-break."""
    out = []
    for r, P_, k in zip(races, mats, keys):
        d = r["g"][["race_id", "season", "round", "driverId", "lineage", "pos", "grid", "classified"]].copy()
        n = len(d)
        d["score"] = DEC.break_ties(-DEC.expected_position(P_), k)
        d["p_win"], d["p_podium"], d["p_points"], d["p_dnf"] = P_[:, 0], P_[:, :3].sum(1), P_[:, :min(10, n)].sum(1), r["p_dnf"]
        out.append(d)
    preds = pd.concat(out, ignore_index=True)
    rps = {r["race_id"]: rps_score(P_, r["pos"]) for r, P_ in zip(races, mats)}
    return preds, A.race_table(preds, rps=rps)


def run_walk(entries: pd.DataFrame, mcfg: dict, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return walk_forward(entries, build_model(mcfg, entries), split, refit_every=1)


def s3_block(pp: pd.DataFrame, entries: pd.DataFrame, dn: pd.DataFrame, consts: dict, theta: dict, draws: int, seed: int):
    """S3 matrices on the harness scores. Seed of race i = seed + 10000 + i, i = the race's index in date order within the block."""
    cols = ["race_id", "driverId", "grid_norm", "stick", "circ_vol", "rain_fc"]
    fr = pp.merge(dn, on=["race_id", "driverId"]).merge(entries[cols], on=["race_id", "driverId"])
    assert len(fr) == len(pp)
    races = S.prepare(fr, consts)
    keys = [r["g"]["driverId"].to_numpy() for r in races]
    mats = DEC.simulate_matrices(races, {k: float(theta[k]) for k in S.DEFAULT}, "S3", draws, seed)
    preds, t = sim_tables(races, mats, keys)
    return preds, t


# ------------------------------------------------------------------------------------ contrasts and readings
def gain_rows(block: str, name: str, ref_name: str, cand_name: str, ref: pd.DataFrame, cand: pd.DataFrame, metrics: list[str],
              primary: str, status: str, nb: dict) -> list[dict]:
    g = A.paired_gain(ref, cand, metrics, n_boot=nb["n_boot"], seed=nb["seed"])
    return [dict(label=status, block=block, table="contrast", name=name, reference=ref_name, candidate=cand_name, metric=r["metric"],
                 primary=r["metric"] == primary, ref_mean=r["ref_mean"], cand_mean=r["cand_mean"], gain=r["gain"], lo=r["lo"], hi=r["hi"],
                 n=r["n"]) for _, r in g.iterrows()]


def mean_rows(block: str, tabs: dict[str, pd.DataFrame], status: str, prob_source: str = "") -> list[dict]:
    """`prob_source` labels where logloss_points / rps come from: the plain-model tables carry the harness Plackett-Luce probabilities,
    the "[S3 matrix, D0 order]" tables the S3 simulator matrix."""
    rows = []
    for m, t in tabs.items():
        for k in POINT + PROB:
            if k in t.columns:
                rows.append(dict(label=status, block=block, table="mean", name=m, metric=k, cand_mean=float(t[k].mean()), n=int(t[k].notna().sum()),
                                 prob_source=(prob_source if k in PROB else "")))
    return rows


def block_rows(block: str, tabs: dict[str, pd.DataFrame], s3: dict[str, pd.DataFrame], status: str, nb: dict) -> list[dict]:
    """Means and every contrast of the pre-registration for one set of races (a block, a pool, or a season)."""
    rows = mean_rows(block, tabs, status, "Plackett-Luce (harness), not S3") + mean_rows(block + " [S3 matrix, D0 order]", s3, status, "S3 matrix")
    for name, cand, ref, primary in [(f"{t}:{c}_vs_{r}", c, r, m) for t, c, r, m in TESTS] + OTHERS:
        if cand in tabs and ref in tabs:
            rows += gain_rows(block, name, ref, cand, tabs[ref], tabs[cand], POINT, primary, status, nb)
    name, cand, ref, primary = S3_CONTRAST
    if cand in s3 and ref in s3:
        rows += gain_rows(block, name, ref + " (own S3 re-fit)", cand + " (S3 re-fit)", s3[ref], s3[cand], PROB, primary, status, nb)
    return rows


def holm_reading(tabs: dict[str, pd.DataFrame], alpha: float, nb: dict) -> pd.DataFrame:
    """T1-T3 with Holm over the three tests on the given per-race tables (positive gain = candidate better)."""
    boots, info = {}, {}
    for t, cand, ref, metric in TESTS:
        d = H.paired_diffs(tabs[ref], tabs[cand], metric, A.HIGHER_BETTER[metric])
        boots[t] = H.boot_means(d, metric, nb["n_boot"], nb["seed"])
        lo, hi = np.percentile(boots[t], [2.5, 97.5])
        info[t] = dict(metric=metric, candidate=cand, reference=ref, n_races=int(len(d)), gain=float(d.mean()), lo95=float(lo), hi95=float(hi))
    out = H.holm_stepdown(boots, alpha)
    return out.merge(pd.DataFrame(info).T.rename_axis("test").reset_index(), on="test").sort_values("test").reset_index(drop=True)


def record_readings(fresh_tabs: dict[str, pd.DataFrame], n: int, levels: tuple[int, ...], alpha: float, nb: dict, out_dir: Path,
                    meta: dict, write: bool = True, tabs_for_level=None, meta_for_level=None) -> dict:
    """The readings due at n races. Level t (20, 40) is read once, on the first t races in date order, and written with an exclusive
    create: a later call never overwrites it. Returns {level: reading dict (loaded from disk if it already existed)}."""
    out: dict[int, dict] = {}
    for t in H.read_levels(n, levels):
        path = out_dir / f"holdout_verdict_n{t}.json"
        if path.exists():
            out[t] = json.loads(path.read_text(encoding="utf-8"))
            continue
        # main-session decision (N2): the level-t reading uses only data up to the t-th holdout race, so it does not depend on when the
        # script is run. `tabs_for_level(t)` returns the tables of a walk-forward run on the rows through that race.
        src = fresh_tabs if tabs_for_level is None else tabs_for_level(t)
        ids = sorted(src["final"]["race_id"])[:t]
        sub = {k: v[v["race_id"].isin(ids)] for k, v in src.items()}
        res = holm_reading(sub, alpha, nb)
        payload = dict(level=t, n_races_read=len(ids), n_races_available=int(n), race_ids=[int(i) for i in ids], alpha=alpha,
                       bootstrap=nb, tests=res.to_dict("records"), confirmed={r["test"]: bool(r["confirmed"]) for _, r in res.iterrows()},
                       written=str(datetime.now().isoformat(timespec="seconds")), **meta,
                       **(meta_for_level(t, [int(i) for i in ids]) if meta_for_level else {}))
        if write:
            assert H.write_verdict_once(path, payload)
        out[t] = payload
    return out


# ------------------------------------------------------------------------------------ holdout membership
def fresh_race_ids(entries: pd.DataFrame, last_race_id: int) -> list[int]:
    """Races after the freeze with a classified result, in date order. A race without a result is not counted, and nothing after
    it is either (it would be an unfinished or future race); a race with a result after one without is an error."""
    e = entries[entries["race_id"] > last_race_id]
    ids, ok = [], []
    for rid, g in e.groupby("race_id", sort=True):
        ids.append(int(rid))
        ok.append(bool(g["pos"].notna().any() and g["classified"].astype(bool).any()))
    out = []
    for rid, o in zip(ids, ok):
        if not o:
            break
        out.append(rid)
    if any(ok[len(out):]):
        raise RuntimeError(f"races without a classified result before races with one after race_id {last_race_id}: {ids}, {ok}")
    d = e.groupby("race_id")["date"].first()
    return sorted(out, key=lambda r: (d[r], r))


# ------------------------------------------------------------------------------------ stored-run reproduction
def stored(lb: str, label: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rid = PINNED[(lb, label, split)]
    lbd = pd.read_csv(ROOT / lb)
    m = lbd[lbd["run_id"] == rid]
    assert len(m) == 1 and m.iloc[0]["label"] == label and m.iloc[0]["split"] == split, (lb, label, split, rid)
    row = m.iloc[0]
    d = ROOT / str(row["run_dir"]).replace("\\", "/")
    return pd.read_csv(d / "per_race_metrics.csv"), pd.read_parquet(d / "predictions.parquet"), str(row["run_id"])


def repro(name: str, new_m: pd.DataFrame, new_p: pd.DataFrame, lb: str, label: str, split: str) -> dict:
    om, op, rid = stored(lb, label, split)
    d, nr = U.max_diff(new_m, om)
    ds, ne = U.score_diff(new_p, op)
    return dict(check=name, stored_run=rid, races=nr, entries=ne, max_abs_diff_per_race_metrics=d, max_abs_diff_scores=ds)


# ------------------------------------------------------------------------------------ main
def load_inputs(cfg: dict):
    frozen_cfg = load_yaml(MDIR / "frozen_config.yaml")
    frozen1 = load_yaml(ROOT / cfg["frozen_path"])
    entries0 = Q.load_all()
    spec = dict(frozen_cfg["pace_feature"])
    pace = F.build_pace_table(spec)
    entries = F.add_pace_hist(entries0, pace, spec)
    return frozen_cfg, frozen1, entries


def write_rowhash(cfg: dict) -> None:
    sc = cfg["score"]
    path = MDIR / sc["rowhash_file"]
    if path.exists():
        raise SystemExit(f"{path} exists; it is written once and never rewritten")
    want = H.freeze_data_hashes(MDIR / "FREEZE.md")
    files = list(H.PARQUET_FILES) + list(H.CSV_FILES)
    bad = {f: (H.sha256_bytes(ROOT / f), want.get(f)) for f in files if H.sha256_bytes(ROOT / f) != want.get(f)}
    if bad:
        raise SystemExit(f"current files differ from the FREEZE.md hashes, not writing row hashes: {bad}")
    spec = H.build_spec(ROOT, H.PARQUET_FILES, H.CSV_FILES, sc["last_race"],
                        versions=dict(pandas=pd.__version__, numpy=np.__version__, python=platform.python_version()))
    spec["file_sha256_at_generation"] = {f: want[f] for f in files}
    spec["generated"] = datetime.now().isoformat(timespec="seconds")
    spec["freeze_commit"] = "df9b8e8 (tag plan3-freeze)"
    H.write_spec_once(spec, path)
    log(f"wrote {path.name}: " + ", ".join(f"{k} {v['n_rows']} rows" for k, v in {**spec['parquet'], **spec['csv']}.items()))


def save_row(label: str, split: str, cfgd: dict, preds: pd.DataFrame, per: pd.DataFrame, tmp_lb: Path, stamp: str) -> str:
    rid = save_run(MDIR, label, dict(cfgd, scored=f"{split}: {THIRD_LOOK if split != 'fresh' else 'fresh holdout'}"), preds, per, split, "P3", "111", leaderboard=tmp_lb)
    pth = MDIR / "results" / rid / "config.yaml"
    c = load_yaml(pth)
    c["git_commit"] = stamp
    save_yaml(c, pth)
    mp = MDIR / "results" / rid / "metrics.json"
    mj = json.loads(mp.read_text(encoding="utf-8"))
    mj["scored"] = c["scored"]
    mp.write_text(json.dumps(mj, indent=2), encoding="utf-8")
    return rid


def assert_entries_sorted(entries: pd.DataFrame) -> None:
    """The harness needs the rows of every race in finishing order (race_id, pos): `pos` is the derived order of all entries."""
    e = entries[entries["pos"].notna()]
    key = e["race_id"].to_numpy(float) * 1000 + e["pos"].to_numpy(float)
    if not (np.diff(key) > 0).all():
        raise SystemExit("entries are not sorted by (race_id, pos) with unique positions: the harness needs that order")


def check_config_constants(cfg: dict) -> None:
    """The pre-registered constants live in holdout.py. A config that disagrees is refused (the config cannot change them)."""
    nb = dict(cfg.get("bootstrap", {}))
    if nb != dict(n_boot=H.N_BOOT, seed=H.SEED):
        raise SystemExit(f"config bootstrap {nb} differs from the pre-registered n_boot {H.N_BOOT}, seed {H.SEED}")
    sc = cfg["score"]
    for k, want in (("alpha", H.ALPHA), ("min_n_levels", list(H.LEVELS)), ("parquet_files", H.PARQUET_FILES), ("csv_files", H.CSV_FILES)):
        if k in sc and sc[k] != want:
            raise SystemExit(f"config score.{k} = {sc[k]} differs from the pre-registered {want}")


def tree_dirty() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "methodologies/P3_final", "configs",
                                   ":(exclude)methodologies/P3_final/results"], text=True, cwd=ROOT)
    return bool(out.strip())


def existing_run(model: str, split: str) -> str | None:
    for lb, label in EXISTING.get(model, []):
        d = pd.read_csv(ROOT / lb, usecols=["label", "split", "run_id"])
        m = d[(d["label"] == label) & (d["split"] == split)]
        if len(m):
            return f"{lb}:{m.iloc[-1]['run_id']}"
    return None


def score_blocks(entries_run: pd.DataFrame, blocks: list[str], frozen_cfg: dict, cfgs: dict, frozen1: dict, thetas: dict, nb: dict,
                 check_stored: bool) -> dict:
    """Walk-forward of every model on the named blocks of `entries_run`, S3 matrices, per-race tables. Nothing is written."""
    sim = frozen_cfg["horizons"]["H2"]["simulator"]
    consts = {k: float(sim[k]) for k in ("stick_ref", "vol_mu", "vol_sd")}
    C = float(frozen_cfg["retirement_model"]["C"])
    h1_cfg = dict(frozen_cfg["horizons"]["H1"]["model"])
    out: dict = {"blocks": {}, "repro": []}
    for split in blocks:
        t0 = time.time()
        res = {name: run_walk(entries_run, c, split) for name, c in {**cfgs, "H1": h1_cfg}.items()}
        log(f"  {split}: walk-forward done ({len(res['final'][1])} races, {time.time() - t0:.0f}s)")
        tabs = {n: tab(p) for n, (p, _) in res.items()}
        tabs["plan2_H2"] = tabs["latent_111"]
        gp, tabs["grid"] = grid_table(res["latent_111"][0])
        dn = D.walk_dnf(entries_run, split, C)[0][["race_id", "driverId", "p_dnf"]]
        s3p, s3t = {}, {}
        for name in ("final", "latent_111"):
            s3p[name], s3t[name] = s3_block(res[name][0], entries_run, dn, consts, thetas[name], int(sim["draws"]), int(sim["seed"]))
        out["blocks"][split] = dict(res=res, tabs=tabs, grid_preds=gp, s3p=s3p, s3t=s3t)
        if check_stored:
            lat_p, lat_m = res["latent_111"]
            out["repro"].append(dict(split=split, **repro("latent_111 vs stored plan-1 run", lat_m, lat_p, "results/leaderboard.csv", "latent_111", split)))
            out["repro"].append(dict(split=split, **repro("plan2_H2 (= latent_111) vs stored plan-2 m06_L_r0_111", lat_m, lat_p, "results/plan2/leaderboard.csv", "m06_L_r0_111", split)))
            h1p, h1m = res["H1"]
            out["repro"].append(dict(split=split, **repro("H1 (frozen) vs stored plan2_H1 (h1_latent_111)", h1m, h1p, "results/plan2/leaderboard.csv", "h1_latent_111", split)))
            om, _, rid = stored("results/leaderboard.csv", "grid", split)
            d, nr = U.max_diff(tabs["grid"], om)
            out["repro"].append(dict(split=split, check="grid order (outcome-free tie-break) vs stored plan-1 grid run", stored_run=rid, races=nr,
                                     max_abs_diff_per_race_metrics=d, max_abs_diff_scores=float("nan"), entries=0))
            p1m, _, r1 = stored("results/leaderboard.csv", "latent_111", split)
            p2m, _, r2 = stored("results/plan2/leaderboard.csv", "m06_L_r0_111", split)
            out["repro"].append(dict(split=split, check="P0: stored plan-1 latent_111 vs stored plan-2 m06_L_r0_111, per-race rho",
                                     stored_run=f"{r1} / {r2}", races=len(p1m), max_abs_diff_per_race_metrics=float((p1m.set_index("race_id")["spearman"] - p2m.set_index("race_id")["spearman"]).abs().max()),
                                     max_abs_diff_scores=float("nan"), entries=0))
    return out


def theta_latent(cfg: dict) -> dict:
    """The simulator parameters of `latent_111`'s own S3 re-fit (R14 dev run, stored)."""
    lbp, label, split, rid = THETA_RUN
    lb = pd.read_csv(ROOT / lbp)
    m = lb[lb["run_id"] == rid]
    assert len(m) == 1 and m.iloc[0]["label"] == label and m.iloc[0]["split"] == split
    row = m.iloc[0]
    return load_yaml(ROOT / str(row["run_dir"]).replace("\\", "/") / "config.yaml")["theta"]


def collect(sc_out: dict, blocks: list[str], model_names: list[str]):
    """Long tables of per-race metrics and predictions of the scored blocks."""
    per, pr = [], []
    for b in blocks:
        bo = sc_out["blocks"][b]
        for m in model_names:
            t = bo["tabs"][m].copy()
            per.append(t.assign(block=b, model=m))
            p = (bo["res"][m][0] if m in bo["res"] else (bo["grid_preds"] if m == "grid" else bo["res"]["latent_111"][0]))
            p = U.outcome_free(p, "driverId") if m != "grid" else p
            pr.append(p.assign(block=b, model=m))
        for m, p in bo["s3p"].items():
            pr.append(p.assign(block=b, model=f"S3_{m}"))
            per.append(bo["s3t"][m].assign(block=b, model=f"S3_{m}"))
    return pd.concat(per, ignore_index=True), pd.concat(pr, ignore_index=True)


def write_leaderboard_rows(sc_out: dict, blocks: list[str], split_of: dict, label_fmt: str, cfgs: dict, theta_map: dict, frozen_cfg: dict,
                           plan3_lb: Path, stamp: str, skip_existing_models: bool) -> list[dict]:
    """One run folder and one leaderboard row per (model, block). A model that already has a third-look row in an earlier plan's
    leaderboard, or a row with the same label and split in the plan-3 leaderboard, is not written again."""
    tmp_lb = RES / "_lb_tmp.csv"
    tmp_lb.unlink(missing_ok=True)
    have = pd.read_csv(plan3_lb, usecols=["label", "split"])
    written = []
    for b in blocks:
        split = split_of[b]
        bo = sc_out["blocks"][b]
        items = [(m, cfgs.get(m, dict(frozen_cfg["horizons"]["H1"]["model"])), bo["res"][m][0], bo["res"][m][1]) for m in ("final", "reverse_only", "final_minus_reverse", "latent_111", "H1")]
        items.append(("grid", dict(kind="grid_order"), bo["grid_preds"], bo["tabs"]["grid"]))
        items.append(("S3refit_final", dict(kind="m08", cell="S3", theta=theta_map["final"], order_model="final"), bo["s3p"]["final"], bo["s3t"]["final"]))
        items.append(("S3refit_latent_111", dict(kind="m08", cell="S3", theta=theta_map["latent_111"], order_model="latent_111"), bo["s3p"]["latent_111"], bo["s3t"]["latent_111"]))
        for m, c, p, per in items:
            label = label_fmt.format(m=m, split=split)
            ex = existing_run(m, split) if skip_existing_models else None
            if ex or ((have["label"] == label) & (have["split"] == split)).any():
                written.append(dict(model=m, split=split, row="not appended", reason=ex or "same label and split already in the plan-3 leaderboard"))
                continue
            rid = save_row(label, split, dict(c, component=m), p, per, tmp_lb, stamp)
            written.append(dict(model=m, split=split, row="appended", run_id=rid))
    if tmp_lb.exists():
        U.append_leaderboard(pd.read_csv(tmp_lb), plan3_lb)
        tmp_lb.unlink()
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MDIR / "config.yaml"))
    ap.add_argument("--write-rowhash", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="compute and print only; nothing is written")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    sc = cfg["score"]
    check_config_constants(cfg)
    nb = dict(n_boot=H.N_BOOT, seed=H.SEED)
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)
    write = not args.no_write

    # ---- 1. the freeze and the data it was made on
    sha = F.verify_freeze(MDIR / "frozen_config.yaml", MDIR / "FREEZE.md")
    log(f"freeze verified: frozen_config.yaml sha256 {sha}")
    prereg_sha = H.verify_prereg(MDIR / "HOLDOUT_PREREG.md", MDIR / "FREEZE.md")
    log(f"pre-registration verified: HOLDOUT_PREREG.md sha256 {prereg_sha}")
    if args.write_rowhash:
        write_rowhash(cfg)
        return 0
    rh = MDIR / sc["rowhash_file"]
    if not rh.exists():
        raise SystemExit(f"{rh.name} is missing: run with --write-rowhash once (only while the data files equal the FREEZE.md hashes)")
    rh_sha = F.sha256_file(rh)  # LF-normalised: a CRLF checkout must not read as a change
    status_path = PLAN3 / "holdout_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    if status.get("freeze_rowhash_sha256") and status["freeze_rowhash_sha256"] != rh_sha:
        raise SystemExit(f"{rh.name} changed since the first scoring: sha256 {rh_sha} != recorded {status['freeze_rowhash_sha256']}")
    spec = json.loads(rh.read_text(encoding="utf-8"))
    H.assert_unchanged(spec, ROOT)
    log("data integrity: every row that existed at the freeze is unchanged (entries2, entries, laps, pits, external tables)")

    # ---- 2. inputs and what is new
    last_id = int(sc["last_race"]["race_id"])
    frozen_cfg, frozen1, entries = load_inputs(cfg)
    assert_entries_sorted(entries)
    fids = fresh_race_ids(entries, last_id)
    n = len(fids)
    keys = {int(r): (int(a), int(b)) for r, a, b in entries[["race_id", "season", "round"]].drop_duplicates().itertuples(index=False)}
    for t in H.LEVELS:                                       # a verdict file stores the digests of the holdout rows it used: they must not change
        vp = PLAN3 / f"holdout_verdict_n{t}.json"
        if vp.exists():
            v = json.loads(vp.read_text(encoding="utf-8"))
            probs = H.check_holdout_digests(v["row_digests"], ROOT, {int(i): keys[int(i)] for i in v["race_ids"]}) if "row_digests" in v else ["no row digests stored"]
            if probs:
                raise H.DataIntegrityError(f"{vp.name}: the holdout rows it used have changed: {probs}")
    print(f"FRESH HOLDOUT RACE COUNT n = {n} (races with race_id > {last_id} and a classified result; data end race_id {int(entries['race_id'].max())})")
    if status.get("third_look_done") and status.get("fresh_race_ids") == fids:
        log("nothing new since the last scoring (same third look, same fresh races): no-op")
        return 0
    due = [t for t in H.read_levels(n) if not (PLAN3 / f"holdout_verdict_n{t}.json").exists()]
    if due and not write:
        raise SystemExit(f"--no-write refused: reading level(s) {due} are due and have no verdict file; a reading is only made with writing enabled")
    if due and write and PLAN3.resolve().is_relative_to(ROOT.resolve()) and tree_dirty():
        raise SystemExit(f"refusing to write the verdict for level(s) {due}: the git tree is dirty in src/, methodologies/P3_final/ or configs/; commit first")
    do_third = not status.get("third_look_done")
    if do_third and (PLAN3 / "third_look.csv").exists():
        raise SystemExit("results/plan3/third_look.csv exists but holdout_status.json does not record it; refusing to overwrite")

    stamp = U.code_stamp(USED)
    cfgs = model_configs(frozen_cfg, frozen1)
    theta_map = {"final": {k: float(v) for k, v in frozen_cfg["horizons"]["H2"]["simulator"]["theta"].items()}, "latent_111": theta_latent(cfg)}
    models = ["final", "reverse_only", "final_minus_reverse", "latent_111", "plan2_H2", "grid", "H1"]
    run_id = f"{datetime.now():%Y%m%d-%H%M}_p3_R15_score"
    rdir = RES / run_id
    k = 2
    while rdir.exists():
        rdir = RES / f"{run_id}-{k}"
        k += 1
    run_id = rdir.name
    summary: dict = dict(run_id=run_id, date=str(date.today()), fresh_n=n, code=stamp, frozen_config_sha256=sha,
                         holdout_prereg_sha256=prereg_sha, freeze_rowhash_sha256=rh_sha)
    all_per, all_pred, rows, written, repro_rows = [], [], [], [], []

    # ---- 3. third look (once)
    if do_third:
        log(f"THIRD LOOK ({THIRD_LOOK}): test_stable and test_shift on the data through race_id {last_id}")
        e_old = entries[entries["race_id"] <= last_id].reset_index(drop=True)
        o = score_blocks(e_old, ["test_stable", "test_shift"], frozen_cfg, cfgs, frozen1, theta_map, nb, check_stored=True)
        repro_rows = o["repro"]
        print("\nREPRODUCTION CHECKS\n" + pd.DataFrame(repro_rows).to_string(index=False))
        tabs_b = o["blocks"]
        pooled = {m: pd.concat([tabs_b[b]["tabs"][m] for b in tabs_b], ignore_index=True) for m in tabs_b["test_stable"]["tabs"]}
        pooled_s3 = {m: pd.concat([tabs_b[b]["s3t"][m] for b in tabs_b], ignore_index=True) for m in ("final", "latent_111")}
        sets = {b: (tabs_b[b]["tabs"], tabs_b[b]["s3t"]) for b in tabs_b}
        sets["pooled test_stable+test_shift"] = (pooled, pooled_s3)
        for s in (2024, 2025, 2026):
            sets[f"season {s}"] = ({m: t[t["season"] == s] for m, t in pooled.items()}, {m: t[t["season"] == s] for m, t in pooled_s3.items()})
        for b, (tt, ss) in sets.items():
            rows += block_rows(b, {m: tt[m] for m in models if m in tt}, ss, THIRD_LOOK, nb)
        tl = pd.DataFrame(rows)
        print("\nTHIRD LOOK, primary contrasts (positive = candidate better; " + THIRD_LOOK + ")")
        print(tl[(tl["table"] == "contrast") & tl["primary"]][["block", "name", "metric", "ref_mean", "cand_mean", "gain", "lo", "hi", "n"]].round(4).to_string(index=False))
        print("\nTHIRD LOOK, S3 contrast and means")
        print(tl[(tl["name"] == S3_CONTRAST[0])][["block", "metric", "ref_mean", "cand_mean", "gain", "lo", "hi", "n"]].round(4).to_string(index=False))
        per, pr = collect(o, list(tabs_b), models)
        all_per.append(per.assign(scope="third_look"))
        all_pred.append(pr.assign(scope="third_look"))
        if write:
            PLAN3.mkdir(parents=True, exist_ok=True)
            tl.to_csv(PLAN3 / "third_look.csv", index=False)
            written += write_leaderboard_rows(o, list(tabs_b), {b: b for b in tabs_b}, "p3_R15_{m}_{split}", cfgs, theta_map, frozen_cfg, ROOT / cfg["leaderboard"], stamp, True)
        summary["third_look"] = dict(label=THIRD_LOOK, blocks={b: {m: {k_: float(v[k_].mean()) for k_ in POINT if k_ in v} for m, v in tabs_b[b]["tabs"].items()} for b in tabs_b},
                                     reproduction=repro_rows)

    # ---- 4. fresh holdout
    state = H.verdict_state(n, min(H.LEVELS))
    readings: dict = {}
    if n == 0:
        log(f"FRESH HOLDOUT: n = 0, {H.NO_VERDICT}. Nothing to score.")
        summary["fresh"] = dict(n=0, state=H.NO_VERDICT)
    else:
        log(f"FRESH HOLDOUT: n = {n} races ({fids[0]} .. {fids[-1]}), {state}")
        e_f = entries[entries["race_id"] <= fids[-1]].reset_index(drop=True)
        o = score_blocks(e_f, ["fresh"], frozen_cfg, cfgs, frozen1, theta_map, nb, check_stored=False)
        bo = o["blocks"]["fresh"]
        assert sorted(bo["tabs"]["final"]["race_id"]) == sorted(fids), "harness fresh split differs from race_id > last with a result"
        status_label = H.NO_VERDICT if n < min(H.LEVELS) else "n >= 20: readings are in the verdict files"
        sets = {"fresh": (bo["tabs"], bo["s3t"])}
        seasons = sorted(bo["tabs"]["final"]["season"].unique())
        for s in seasons:
            sets[f"fresh season {s}"] = ({m: t[t["season"] == s] for m, t in bo["tabs"].items()}, {m: t[t["season"] == s] for m, t in bo["s3t"].items()})
        for b, (tt, ss) in sets.items():
            rows += block_rows(b, {m: tt[m] for m in models if m in tt}, ss, status_label, nb)
        fr = pd.DataFrame(rows)
        print(f"\nFRESH HOLDOUT n = {n}, {state}; primary contrasts (positive = candidate better)")
        print(fr[(fr["block"] == "fresh") & (fr["table"] == "contrast") & fr["primary"]][["name", "metric", "ref_mean", "cand_mean", "gain", "lo", "hi", "n"]].round(4).to_string(index=False))
        meta = dict(frozen_config_sha256=sha, holdout_prereg_sha256=prereg_sha, freeze_rowhash_sha256=rh_sha, code=stamp, git_commit=_git_hash())

        def tabs_for_level(t: int) -> dict:
            """Walk-forward on the rows through the t-th holdout race only (N2): the reading does not depend on when the script is run."""
            e_t = entries[entries["race_id"] <= fids[t - 1]].reset_index(drop=True)
            return {m: tab(run_walk(e_t, c, "fresh")[0]) for m, c in cfgs.items()}

        def meta_for_level(t: int, ids: list[int]) -> dict:
            return dict(row_digests=H.holdout_digests(ROOT, {i: keys[i] for i in ids}))
        readings = record_readings(bo["tabs"], n, H.LEVELS, H.ALPHA, nb, PLAN3, meta, write=write, tabs_for_level=tabs_for_level, meta_for_level=meta_for_level)
        for t, r in readings.items():
            print(f"\nREADING at n >= {t} (first {r['n_races_read']} races; written {r['written']}); Holm over T1-T3 (alpha {r['alpha']})")
            print(pd.DataFrame(r["tests"]).round(4).to_string(index=False))
        per, pr = collect(o, ["fresh"], models)
        all_per.append(per.assign(scope="fresh"))
        all_pred.append(pr.assign(scope="fresh"))
        if write:
            PLAN3.mkdir(parents=True, exist_ok=True)
            fr.to_csv(PLAN3 / "fresh_holdout_latest.csv", index=False)
            written += write_leaderboard_rows(o, ["fresh"], {"fresh": "fresh"}, f"p3_R15_{{m}}_fresh_n{n}", cfgs, theta_map, frozen_cfg, ROOT / cfg["leaderboard"], stamp, False)
        summary["fresh"] = dict(n=n, state=state, race_ids=fids, readings={str(t): dict(confirmed=r["confirmed"], file=f"results/plan3/holdout_verdict_n{t}.json") for t, r in readings.items()})

    # ---- 5. records
    new_status = dict(n=n, date=str(date.today()), last_race_id=(fids[-1] if fids else None), fresh_race_ids=fids,
                      verdict_state=("no verdict (n < 20)" if n < min(H.LEVELS) else f"readings recorded at levels {sorted(readings)}; later numbers are descriptive until the next level"),
                      readings={str(t): f"results/plan3/holdout_verdict_n{t}.json" for t in readings},
                      third_look_done=True, third_look_label=THIRD_LOOK, last_run=run_id, code=stamp, frozen_config_sha256=sha,
                      holdout_prereg_sha256=prereg_sha, freeze_rowhash_sha256=rh_sha)
    if write:
        rdir.mkdir(parents=True, exist_ok=True)
        save_yaml(dict(cfg, scored_run=run_id, git_commit=stamp, frozen_config_sha256=sha, fresh_n=n,
                       leaderboard_rows=written, note="R15 scoring; nothing tuned or selected"), rdir / "config.yaml")
        (rdir / "git_commit.txt").write_text(stamp + "\n", encoding="utf-8")
        (rdir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        if all_per:
            pd.concat(all_per, ignore_index=True).to_csv(rdir / "per_race_metrics.csv", index=False)
            pd.concat(all_pred, ignore_index=True).to_parquet(rdir / "predictions.parquet", index=False)
        else:
            pd.DataFrame(columns=["race_id"]).to_csv(rdir / "per_race_metrics.csv", index=False)
            pd.DataFrame(columns=["race_id"]).to_parquet(rdir / "predictions.parquet", index=False)
        PLAN3.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(new_status, indent=1), encoding="utf-8")
        log(f"wrote {rdir}, {status_path}")
        if written:
            print(pd.DataFrame(written).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
