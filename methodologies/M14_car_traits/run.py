"""M14 part 1 - engine-supplier pooling in the latent model (PLAN3.md R11, Amendment 3 item 4). Dev only.

    python methodologies/M14_car_traits/run.py [--quick] [--checks-only]

b_total[car] = b[car] + e[supplier, season] (src/pitlane/models/latent_supplier.py). Reference latent_111 (plan-1 frozen
hyper-parameters) is re-run in the same harness and checked against the stored plan-1 dev runs. The universe holds all seasons (as in earlier
plan-3 runs, for exact reproduction); seasons after the dev block are neither fitted nor scored. lam_e is chosen by dev mean rho at refit_every=4.
--quick: first 6 dev races, 2 lam_e values, nothing written. --checks-only: unit checks only, written to results/checks.json.
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
from pitlane.models.latent import LatentModel
from pitlane.models.latent_supplier import LatentSupplierModel
from pitlane.models.registry import build_model
from pitlane.plan3 import attribution as A
from pitlane.plan3 import decode as DEC
from pitlane.plan3 import runutil as U

MDIR = Path(__file__).resolve().parent
USED = ["src/pitlane/models/latent.py", "src/pitlane/models/latent_supplier.py", "src/pitlane/models/registry.py",
        "src/pitlane/plan3/attribution.py", "src/pitlane/plan3/runutil.py", "src/pitlane/evaluation/harness.py",
        "data/external/power_units.csv", "methodologies/M14_car_traits/run.py", "methodologies/M14_car_traits/config.yaml"]
VERDICT_METRICS = ["spearman", "mae", "exact", "exact_fin", "spearman_fin", "mae_fin", "exact_1_3", "exact_4_10", "exact_11p",
                   "logloss_points"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------ data
def load_universe(cfg: dict) -> pd.DataFrame:
    """Entries of the full universe with the season-start supplier of the constructor (joined on season, constructorId)."""
    ent = P.load_entries()
    ent = ent[ent["season"] <= cfg["universe_last_season"]].reset_index(drop=True)
    pu = pd.read_csv(ROOT / cfg["power_units"])[["season", "constructorId", "supplier"]]
    out = ent.merge(pu, on=["season", "constructorId"], how="left", validate="many_to_one")
    assert out["supplier"].notna().all(), "entry without a supplier row"
    assert len(out) == len(ent) and (out.index == np.arange(len(out))).all()
    return out


def lat_kwargs(frozen: dict, cfg: dict) -> dict:
    return dict(frozen[cfg["latent_ref"]["key"]][cfg["latent_ref"]["cell"]])


def dev_race_ids(entries: pd.DataFrame) -> pd.Index:
    first = entries.groupby("race_id")["season"].first()
    return first[(first >= SPLITS["dev"][0]) & (first <= SPLITS["dev"][1])].index


# ------------------------------------------------------------------------------------ unit checks
def gradient_check(entries: pd.DataFrame, lat: dict, lam_e: float, seed: int = 0, **extra) -> dict:
    """Central finite differences of the penalised objective over every parameter (a, b, g, beta, e) at a random point."""
    u = entries[entries["season"] <= 2016].reset_index(drop=True)
    kw = {k: v for k, v in lat.items() if k not in ("kind", "cell")}
    m = LatentSupplierModel(u, dynamic=True, decomp=True, ranking=True, lam_e=lam_e, **extra, **kw)
    rid = int(u["race_id"].max()) - 2
    n_train = int(m.race_rows[rid].min())
    w_row, pad = m.fit_inputs(n_train, float(rid))
    rng = np.random.default_rng(seed)
    th = rng.normal(0, 0.4, len(m.theta))
    f = lambda t: m._objective(t, n_train, w_row, pad)[0]
    g = m._objective(th, n_train, w_row, pad)[1]
    num = np.empty_like(g)
    h = 1e-6
    for i in range(len(th)):
        d = np.zeros_like(th)
        d[i] = h
        num[i] = (f(th + d) - f(th - d)) / (2 * h)
    err = np.abs(g - num)
    scale = np.maximum(1.0, np.abs(num))
    e_idx = np.arange(m.nbase, len(th))
    return dict(n_params=len(th), n_e=int(m.nE), lam_e=lam_e, max_abs_err=float(err.max()), max_rel_err=float((err / scale).max()),
                max_abs_err_e_coords=float(err[e_idx].max()), max_abs_grad_e=float(np.abs(g[e_idx]).max()))


def off_switch_check(entries: pd.DataFrame, lat: dict, dev_ids: pd.Index, n_races: int) -> dict:
    """lam_e = inf must reproduce LatentModel bit for bit (warm-started refits before each of the first dev races);
    a very large finite lam_e is reported for information."""
    kw = {k: v for k, v in lat.items() if k not in ("kind", "cell")}
    ref = LatentModel(entries, dynamic=True, decomp=True, ranking=True, **kw)
    off = LatentSupplierModel(entries, dynamic=True, decomp=True, ranking=True, lam_e=float("inf"), **kw)
    big = LatentSupplierModel(entries, dynamic=True, decomp=True, ranking=True, lam_e=1e8, **kw)
    d_off, d_big = 0.0, 0.0
    for rid in dev_ids[:n_races]:
        rows = ref.race_rows[int(rid)]
        tr, te = entries.iloc[:rows.min()], entries.iloc[rows]
        s0 = ref.fit_predict(tr, te)
        d_off = max(d_off, float(np.abs(off.fit_predict(tr, te) - s0).max()))
        d_big = max(d_big, float(np.abs(big.fit_predict(tr, te) - s0).max()))
    return dict(races=int(min(n_races, len(dev_ids))), max_abs_score_diff_off=d_off, bitwise_equal_off=bool(d_off == 0.0),
                max_abs_score_diff_lam_e_1e8=d_big)


def leakage_check(entries: pd.DataFrame, lat: dict, lam_e: float, dev_ids: pd.Index, n_races: int) -> dict:
    """Score of race r from a cold fit must not change when (a) outcomes of race r and later are scrambled and the supplier
    of later seasons is changed, or (b) all rows after race r are deleted (supplier of race r comes from its own season)."""
    kw = {k: v for k, v in lat.items() if k not in ("kind", "cell")}
    ids = [int(dev_ids[i]) for i in np.linspace(0, len(dev_ids) - 1, n_races).astype(int)]
    d_perm, d_trunc, d_trunc_base = 0.0, 0.0, 0.0
    for rid in ids:
        m = LatentSupplierModel(entries, dynamic=True, decomp=True, ranking=True, lam_e=lam_e, **kw)
        rows = m.race_rows[rid]
        first = int(rows.min())
        s0 = m.fit_predict(entries.iloc[:first], entries.iloc[rows])
        season = int(entries.loc[first, "season"])
        e2 = entries.copy()
        rng = np.random.default_rng(rid)
        late = e2.index >= first
        for col in ("y", "pos", "finish_order", "dnf", "classified", "points"):
            if col in e2:
                e2.loc[late, col] = rng.permutation(e2.loc[late, col].to_numpy())
        e2.loc[e2["season"] > season, "supplier"] = "scrambled"
        m2 = LatentSupplierModel(e2, dynamic=True, decomp=True, ranking=True, lam_e=lam_e, **kw)
        d_perm = max(d_perm, float(np.abs(m2.fit_predict(e2.iloc[:first], e2.iloc[rows]) - s0).max()))
        e3 = entries[entries["race_id"] <= rid].reset_index(drop=True)
        m3 = LatentSupplierModel(e3, dynamic=True, decomp=True, ranking=True, lam_e=lam_e, **kw)
        r3 = m3.race_rows[rid]
        d_trunc = max(d_trunc, float(np.abs(m3.fit_predict(e3.iloc[:r3.min()], e3.iloc[r3]) - s0).max()))
        # same truncation for plain LatentModel: a cold fit with a different parameter set (fewer unused drivers) gives score
        # differences of this size even without an e term. Not a maxiter effect (fits converge, see convergence_*.csv);
        # the likely cause, stopping inside the L-BFGS-B tolerance at a different point, is not isolated
        b0 = LatentModel(entries, dynamic=True, decomp=True, ranking=True, **kw)
        sb0 = b0.fit_predict(entries.iloc[:first], entries.iloc[rows])
        b3 = LatentModel(e3, dynamic=True, decomp=True, ranking=True, **kw)
        d_trunc_base = max(d_trunc_base, float(np.abs(b3.fit_predict(e3.iloc[:r3.min()], e3.iloc[r3]) - sb0).max()))
    return dict(races=ids, max_abs_diff_scrambled_outcomes_and_later_suppliers=d_perm, max_abs_diff_truncated_universe=d_trunc,
                max_abs_diff_truncated_universe_plain_latent_model=d_trunc_base)


# ------------------------------------------------------------------------------------ descriptive
def variance_share(m: LatentSupplierModel, seasons: list[int]) -> list[dict]:
    """Share of the variance of the car strength b + e across car-seasons taken by the supplier term (unweighted)."""
    a, b, g, beta = m._split(m.theta)
    e = m._e(m.theta)
    out = []
    inv = {i: k for k, i in m.car.items()}
    for s in seasons:
        ids = [i for i, (l, y) in inv.items() if y == s]
        bb, ee = b[ids], e[m.car_e[ids]]
        tot = bb + ee
        v = float(tot.var())
        out.append(dict(season=s, n_cars=len(ids), var_total=v, share_e=float(ee.var() / v), share_own=float(bb.var() / v),
                        share_cov=float(2 * np.cov(bb, ee, bias=True)[0, 1] / v)))
    return out


def fit_e_at(entries: pd.DataFrame, lat: dict, lam_e: float, n_train: int, t_ref: float) -> LatentSupplierModel:
    """Cold fit on the first n_train rows (descriptive)."""
    kw = {k: v for k, v in lat.items() if k not in ("kind", "cell")}
    m = LatentSupplierModel(entries, dynamic=True, decomp=True, ranking=True, lam_e=lam_e, **kw)
    m.fit(n_train, t_ref)
    return m


# ------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MDIR / "config.yaml"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--checks-only", action="store_true")
    ap.add_argument("--posthoc", action="store_true", help="R11 revision: per-lam_e tuning metrics, convergence, post-hoc carry_on_total variant")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["split"] == "dev", "plan 3 / R11: development block only"
    t0 = time.time()
    entries = load_universe(cfg)
    lat = lat_kwargs(load_yaml(ROOT / cfg["frozen_path"]), cfg)
    dev_ids = dev_race_ids(entries)
    res = MDIR / "results"
    res.mkdir(parents=True, exist_ok=True)

    if args.posthoc:
        posthoc(cfg, entries, lat, dev_ids, res)
        return

    # ---- unit checks
    chk = dict(gradient={f"lam_e={l}": gradient_check(entries, lat, l) for l in (3.0, 32.0)},
               off_switch=off_switch_check(entries, lat, dev_ids, cfg["check_races"]))
    log(f"gradient check: {json.dumps(chk['gradient'])}")
    log(f"off-switch check: {json.dumps(chk['off_switch'])}")
    assert chk["off_switch"]["bitwise_equal_off"], "lam_e = inf is not bit for bit LatentModel"
    assert all(v["max_rel_err"] < 1e-5 for v in chk["gradient"].values()), "finite-difference gradient check failed"
    chk["leakage"] = leakage_check(entries, lat, 10.0, dev_ids, cfg["check_races"])
    log(f"leakage check: {json.dumps(chk['leakage'])}")
    lk = chk["leakage"]
    assert lk["max_abs_diff_scrambled_outcomes_and_later_suppliers"] < 1e-12, "score changed when later outcomes/suppliers changed"
    assert lk["max_abs_diff_truncated_universe"] < 1e-2, "score changed when later races were deleted"
    if args.checks_only:
        (res / "checks.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
        log("checks written")
        return

    stamp = U.code_stamp(USED)
    split, grid = cfg["split"], list(cfg["lam_e_grid"])
    entries_run = entries
    if args.quick:
        dev_ids = dev_ids[:6]
        grid = grid[:2]
        entries_run = entries[entries["race_id"] <= dev_ids.max()].reset_index(drop=True)

    def run(label: str, mcfg: dict, refit: int):
        t = time.time()
        p, m = walk_forward(entries_run, build_model(mcfg, entries_run), split, refit_every=refit)
        if args.quick:
            p, m = p[p["race_id"].isin(dev_ids)], m[m["race_id"].isin(dev_ids)]
        log(f"  {label}: rho={m['spearman'].mean():.4f} mae={m['mae'].mean():.3f} ({time.time() - t:.0f}s)")
        return p, m

    def sup_cfg(lam_e: float) -> dict:
        return dict(lat, kind="latent_supplier", lam_e=lam_e)

    # ---- tuning of lam_e: dev, refit_every = tune_refit_every
    tune_rows = []
    _, mref4 = run("tune reference latent_111 (not a configuration)", dict(lat), cfg["tune_refit_every"])
    for lam_e in grid:
        _, m = run(f"tune lam_e={lam_e}", sup_cfg(lam_e), cfg["tune_refit_every"])
        tune_rows.append(dict(lam_e=lam_e, spearman=m["spearman"].mean(), mae=m["mae"].mean(), exact=m["exact"].mean(), n_races=len(m)))
    tune = pd.DataFrame(tune_rows)
    tune["ref_spearman"] = float(mref4["spearman"].mean())
    best = float(tune.loc[tune["spearman"].idxmax(), "lam_e"])
    n_cfg = len(tune)
    log(f"lam_e chosen {best}; configurations evaluated for M14: {n_cfg} (budget 30)")

    # ---- final runs, refit before every race
    runs = {"latent_111": (dict(lat), *run("latent_111", dict(lat), cfg["refit_every"])),
            "M14_supplier_pool_latent_111": (sup_cfg(best), *run("supplier pool", sup_cfg(best), cfg["refit_every"]))}
    off_cfg = sup_cfg(float("inf"))
    p_off, m_off = run("supplier pool OFF (lam_e = inf)", off_cfg, cfg["refit_every"])

    rep = {}
    if not args.quick:
        for k in ("latent_111",):
            old_pm, old_pr = U.stored_run(cfg["plan1_leaderboard"], k)
            d, n = U.max_diff(runs[k][2], old_pm)
            ds, ne = U.score_diff(runs[k][1], old_pr)
            rep[k] = dict(max_abs_diff_per_race_metrics=d, max_abs_diff_scores=ds, races=n, entries=ne)
            log(f"REPRODUCTION {k} vs stored plan-1 dev run: max |per-race metric diff| = {d:.3e}, max |score diff| = {ds:.3e} "
                f"({n} races, {ne} entries)")
        d_off = U.max_diff(m_off, runs["latent_111"][2])
        s_off = U.score_diff(p_off, runs["latent_111"][1])
        rep["off_arm_vs_latent_111_same_harness"] = dict(max_abs_diff_per_race_metrics=d_off[0], max_abs_diff_scores=s_off[0])
        log(f"OFF ARM vs latent_111 (same harness): max |metric diff| = {d_off[0]:.3e}, max |score diff| = {s_off[0]:.3e}")

    assert max(int(p["season"].max()) for _, p, _ in runs.values()) <= 2023, "a score outside dev"
    key = cfg["tie_break"]
    tabs = {k: A.race_table(U.outcome_free(v[1], key)) for k, v in runs.items()}
    gp = runs["latent_111"][1][["race_id", "season", "driverId", "pos", "grid", "classified"]].copy()
    gp["score"] = np.concatenate([DEC.grid_utility(g["grid"].to_numpy(float), g[key].to_numpy()) for _, g in gp.groupby("race_id", sort=False)])
    tabs["grid"] = A.race_table(gp)
    nb = cfg["bootstrap"]
    ref, ck = "latent_111", "M14_supplier_pool_latent_111"
    full = A.paired_gain(tabs[ref], tabs[ck], n_boot=nb["n_boot"], seed=nb["seed"])
    full.insert(0, "reference", ref)
    full.insert(0, "component", cfg["component"])
    full["verdict"] = [A.verdict(r) for _, r in full.iterrows()]
    seasong = []
    for s in (2021, 2022, 2023):
        a, b = tabs[ref][tabs[ref]["season"] == s], tabs[ck][tabs[ck]["season"] == s]
        if args.quick and a.empty:
            continue
        gs = A.paired_gain(a, b, ["spearman", "mae", "exact", "spearman_fin", "mae_fin"], n_boot=nb["n_boot"], seed=nb["seed"])
        seasong.append(gs.assign(season=s))
    by_season = pd.concat(seasong, ignore_index=True)
    gridg = [A.paired_gain(tabs["grid"], tabs[ck], ["exact", "exact_fin"], n_boot=nb["n_boot"], seed=nb["seed"]).assign(candidate="M14_supplier_pool"),
             A.paired_gain(tabs["grid"], tabs[ref], ["exact", "exact_fin"], n_boot=nb["n_boot"], seed=nb["seed"]).assign(candidate="latent_111 (reference)")]
    vs_grid = pd.concat(gridg, ignore_index=True)
    vs_grid["verdict"] = [A.verdict(r) for _, r in vs_grid.iterrows()]

    # ---- descriptive: 2022 rounds 1-5 and fitted e
    r_lo, r_hi = cfg["descriptive_2022_rounds"]
    pr = tabs[ref].merge(tabs[ck], on="race_id", suffixes=("_ref", "_m14"))
    rnd = entries.groupby("race_id")[["season", "round"]].first()
    pr = pr.join(rnd, on="race_id")
    early = pr[(pr["season"] == 2022) & pr["round"].between(r_lo, r_hi)][["race_id", "round", "spearman_ref", "spearman_m14", "mae_ref", "mae_m14"]]
    early = early.assign(d_spearman=early["spearman_m14"] - early["spearman_ref"], d_mae=early["mae_ref"] - early["mae_m14"])
    desc = dict(early_2022=early.round(4).to_dict("records"),
                early_2022_mean_d_spearman=float(early["d_spearman"].mean()) if len(early) else None)
    last_id = int(dev_ids[-1])
    n_end = int(entries_run.index[entries_run["race_id"] == last_id].max()) + 1
    m_end = fit_e_at(entries_run, lat, best, n_end, float(last_id + 1))
    eff_end = m_end.supplier_effects()
    desc["e_end_of_dev"] = eff_end[eff_end["season"] >= 2021].round(4).to_dict("records")
    desc["variance_share_end_of_dev"] = variance_share(m_end, [2021, 2022, 2023])
    r6 = entries_run[(entries_run["season"] == 2022) & (entries_run["round"] == 6)]
    if len(r6):
        id6 = int(r6["race_id"].iloc[0])
        m6 = fit_e_at(entries_run, lat, best, int(entries_run.index[entries_run["race_id"] == id6].min()), float(id6))
        e6 = m6.supplier_effects()
        desc["e_after_2022_round5"] = e6[e6["season"] == 2022].round(4).to_dict("records")
        desc["variance_share_after_2022_round5"] = variance_share(m6, [2022])

    pd.set_option("display.width", 250, "display.max_rows", 500)
    mcols = ["spearman", "mae", "exact", "exact_fin", "spearman_fin", "mae_fin", "exact_1_3", "exact_4_10", "exact_11p", "logloss_points"]
    means = pd.DataFrame({k: t.reindex(columns=mcols).mean() for k, t in tabs.items()}).T
    print("\nTUNING (refit_every=4)\n", tune.round(4).to_string(index=False))
    print("\nMEANS\n", means.round(4).to_string())
    print("\nPAIRED GAINS (positive = candidate better; 95% bootstrap CI over races)\n"
          + full[full["metric"].isin(VERDICT_METRICS)].round(4).to_string(index=False))
    print("\nPER SEASON\n" + by_season.round(4).to_string(index=False))
    print("\nEXACT VS GRID ORDER\n" + vs_grid.round(4).to_string(index=False))
    print("\nDESCRIPTIVE\n" + json.dumps(desc, indent=1))
    if args.quick:
        log("quick run: nothing written")
        return

    tmp_lb = res / "_lb_tmp.csv"
    tmp_lb.unlink(missing_ok=True)
    labels = {"latent_111": "m14_ref_latent_111", "M14_supplier_pool_latent_111": "m14_supplier_pool_latent_111"}
    run_ids = {}
    for k, (c, p, m) in runs.items():
        extra = dict(component=k, frozen_hyperparameters="plan-1, not re-tuned" if k == "latent_111" else "plan-1; lam_e tuned on dev",
                     status="dev only; no gbm variant")
        rid_ = save_run(MDIR, labels[k], dict(c, **extra), p, m, split, "M14", c.get("cell", ""), leaderboard=tmp_lb)
        pth = res / rid_ / "config.yaml"
        cc = load_yaml(pth)
        cc["git_commit"] = stamp
        save_yaml(cc, pth)
        tabs[k].to_csv(res / rid_ / "outcome_free_table.csv", index=False)
        run_ids[k] = rid_
    new_rows = pd.read_csv(tmp_lb)
    tmp_lb.unlink()
    tune.to_csv(res / "tuning_lam_e.csv", index=False)
    full.to_csv(res / "paired_gains_dev.csv", index=False)
    by_season.to_csv(res / "paired_gains_by_season.csv", index=False)
    vs_grid.to_csv(res / "exact_vs_grid.csv", index=False)
    means.to_csv(res / "means.csv")
    (res / "reproduction_check.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (res / "descriptive.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")
    (res / "checks.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
    (res / "tuning_summary.json").write_text(json.dumps(dict(lam_e_chosen=best, configurations_evaluated=n_cfg, budget=30,
                                                            grid=grid), indent=2), encoding="utf-8")

    g = full.drop(columns=["component", "reference", "verdict"])
    row = A.attribution_row(cfg["component"], ref, g, cfg["verdict_metric"], run_ids[ck], len(tabs[ck]), declared="yes")
    append_attribution_text(pd.DataFrame([row]), ROOT / cfg["attribution"])
    U.append_leaderboard(new_rows, ROOT / cfg["leaderboard"])
    log(f"verdict (rho): {row['verdict']}")
    log(f"done in {time.time() - t0:.0f}s; code stamp {stamp}")


# ------------------------------------------------------------------------------------ revision: convergence, per-lam_e metrics, post-hoc variant
def logged(cls):
    """Subclass whose fit is the library fit (same minimize call) and also records the L-BFGS-B outcome of every fit."""
    from scipy.optimize import minimize

    class Logged(cls):
        fit_log: list = []

        def fit(self, n_train: int, t_ref: float):
            if hasattr(self, "pin_inactive"):
                self.pin_inactive(n_train)
            w_row, pad = self.fit_inputs(n_train, t_ref)
            r = minimize(self._objective, self.theta, args=(n_train, w_row, pad), jac=True, method="L-BFGS-B",
                         options={"maxiter": self.maxiter})
            self.theta = r.x
            self.fit_log.append(dict(n_iter=int(r.nit), n_fev=int(r.nfev), status=int(r.status), message=str(r.message),
                                     grad_inf=float(np.abs(r.jac).max()), grad_2=float(np.linalg.norm(r.jac)), f=float(r.fun)))
    Logged.fit_log = []
    return Logged


def conv_summary(log_rows: list[dict], maxiter: int) -> dict:
    d = pd.DataFrame(log_rows)
    return dict(n_fits=len(d), median_n_iter=float(d["n_iter"].median()), max_n_iter=int(d["n_iter"].max()),
                share_hit_maxiter=float((d["n_iter"] >= maxiter).mean()), share_converged_status0=float((d["status"] == 0).mean()),
                median_grad_inf=float(d["grad_inf"].median()), max_grad_inf=float(d["grad_inf"].max()),
                median_grad_2=float(d["grad_2"].median()), share_grad_inf_below_1e5=float((d["grad_inf"] < 1e-5).mean()))


def tune_logged(entries: pd.DataFrame, lat: dict, grid: list, refit: int, maxiter: int, extra: dict):
    """Tuning runs (reference first) with per-race metrics and convergence logs. Returns (tune, per_race, convergence)."""
    kw = {k: v for k, v in lat.items() if k not in ("kind", "cell")}
    rows, per, conv = [], [], []
    ref_rho = float("nan")
    for lam_e in [None] + list(grid):
        if lam_e is None:
            cls = logged(LatentModel)
            m = cls(entries, dynamic=True, decomp=True, ranking=True, **kw)
        else:
            cls = logged(LatentSupplierModel)
            m = cls(entries, dynamic=True, decomp=True, ranking=True, lam_e=lam_e, **extra, **kw)
        t = time.time()
        _, pm = walk_forward(entries, m, "dev", refit_every=refit)
        log(f"  tune {'reference' if lam_e is None else lam_e}: rho={pm['spearman'].mean():.4f} mae={pm['mae'].mean():.3f} ({time.time() - t:.0f}s)")
        per.append(pm.assign(lam_e=np.inf if lam_e is None else lam_e))
        conv.append(dict(arm="reference" if lam_e is None else "lam_e", lam_e=np.inf if lam_e is None else lam_e, **conv_summary(cls.fit_log, maxiter)))
        if lam_e is None:
            ref_rho = float(pm["spearman"].mean())
        else:
            rows.append(dict(lam_e=lam_e, spearman=pm["spearman"].mean(), mae=pm["mae"].mean(), exact=pm["exact"].mean(), n_races=len(pm)))
    tune = pd.DataFrame(rows)
    tune["ref_spearman"] = ref_rho
    return tune, pd.concat(per, ignore_index=True), pd.DataFrame(conv)


def solver_probe(entries: pd.DataFrame, lat: dict, dev_ids: pd.Index, n_races: int) -> dict:
    """Is the score difference at lam_e = 1e8 an e effect or an optimiser effect? Same first dev races, maxiter 150 and 2000."""
    kw = {k: v for k, v in lat.items() if k not in ("kind", "cell")}
    out: dict = {}
    for it in (150, 2000):
        k2 = dict(kw, maxiter=it)
        ref = logged(LatentModel)(entries, dynamic=True, decomp=True, ranking=True, **k2)
        big = logged(LatentSupplierModel)(entries, dynamic=True, decomp=True, ranking=True, lam_e=1e8, **k2)
        d = 0.0
        for rid in dev_ids[:n_races]:
            rows = ref.race_rows[int(rid)]
            tr, te = entries.iloc[:rows.min()], entries.iloc[rows]
            d = max(d, float(np.abs(big.fit_predict(tr, te) - ref.fit_predict(tr, te)).max()))
        out[f"maxiter_{it}"] = dict(max_abs_score_diff_lam_e_1e8_vs_reference=d, reference=conv_summary(type(ref).fit_log, it),
                                   lam_e_1e8=conv_summary(type(big).fit_log, it))
    a = LatentModel(entries, dynamic=True, decomp=True, ranking=True, **kw)
    b = LatentModel(entries, dynamic=True, decomp=True, ranking=True, **dict(kw, maxiter=2000))
    d = 0.0
    for rid in dev_ids[:n_races]:
        rows = a.race_rows[int(rid)]
        tr, te = entries.iloc[:rows.min()], entries.iloc[rows]
        d = max(d, float(np.abs(a.fit_predict(tr, te) - b.fit_predict(tr, te)).max()))
    out["reference_maxiter_150_vs_2000_max_abs_score_diff"] = d
    return out


def fit_e_at_cfg(entries: pd.DataFrame, lat: dict, lam_e: float, n_train: int, t_ref: float, extra: dict) -> LatentSupplierModel:
    kw = {k: v for k, v in lat.items() if k not in ("kind", "cell")}
    m = LatentSupplierModel(entries, dynamic=True, decomp=True, ranking=True, lam_e=lam_e, **extra, **kw)
    m.fit(n_train, t_ref)
    return m


def posthoc(cfg: dict, entries: pd.DataFrame, lat: dict, dev_ids: pd.Index, res: Path) -> None:
    """R11 revision. (1) re-derive the declared tuning surface with per-race metrics and convergence logs; (2) post-hoc variant."""
    t0 = time.time()
    ph, maxiter, refit = cfg["posthoc_variant"], cfg["convergence_maxiter"], cfg["tune_refit_every"]
    assert ph["name"] == "carry_on_total" and cfg["split"] == "dev"
    extra = dict(carry_on_total=True, lam_b=ph["lam_b"])
    gc = {f"lam_e={l}": gradient_check(entries, lat, l, **extra) for l in (3.0, 32.0)}
    log(f"post-hoc variant gradient check: {json.dumps(gc)}")
    assert all(v["max_rel_err"] < 1e-5 for v in gc.values())
    # (1) declared model: per-lam_e per-race metrics + convergence
    tune0, per0, conv0 = tune_logged(entries, lat, list(cfg["lam_e_grid"]), refit, maxiter, {})
    old = pd.read_csv(res / "tuning_lam_e.csv")
    log(f"re-derived tuning vs stored tuning_lam_e.csv: max |rho diff| = {np.abs(tune0['spearman'].to_numpy() - old['spearman'].to_numpy()).max():.2e}")
    per0.to_csv(res / "tuning_per_race_declared.csv", index=False)
    conv0.to_csv(res / "convergence_declared.csv", index=False)
    probe = solver_probe(entries, lat, dev_ids, cfg["check_races"])
    (res / "solver_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    log(f"solver probe: {json.dumps(probe)}")
    # (2) post-hoc variant
    tune1, per1, conv1 = tune_logged(entries, lat, list(ph["lam_e_grid"]), refit, maxiter, extra)
    per1.to_csv(res / "tuning_per_race_posthoc.csv", index=False)
    conv1.to_csv(res / "convergence_posthoc.csv", index=False)
    tune1.to_csv(res / "tuning_lam_e_posthoc.csv", index=False)
    best = float(tune1.loc[tune1["spearman"].idxmax(), "lam_e"])
    n_cfg = len(tune1)
    log(f"POST-HOC lam_e chosen {best}; configurations: {n_cfg} (M14 total {len(old) + n_cfg} of 30)")
    split, stamp = cfg["split"], U.code_stamp(USED)
    ck_cfg = dict(lat, kind="latent_supplier", lam_e=best, **extra)
    t = time.time()
    p_ref, m_ref = walk_forward(entries, build_model(dict(lat), entries), split, refit_every=cfg["refit_every"])
    p_ck, m_ck = walk_forward(entries, build_model(ck_cfg, entries), split, refit_every=cfg["refit_every"])
    log(f"final runs done ({time.time() - t:.0f}s): ref rho={m_ref['spearman'].mean():.4f} post-hoc rho={m_ck['spearman'].mean():.4f}")
    stored = pd.read_csv(res / "20261117-1617_m14_ref_latent_111_dev" / "per_race_metrics.csv")
    d_ref = U.max_diff(m_ref, stored)
    log(f"reference re-run vs stored M14 reference run: max |per-race metric diff| = {d_ref[0]:.3e}")
    assert int(p_ck["season"].max()) <= 2023 and int(p_ref["season"].max()) <= 2023, "a score outside dev"
    key, nb = cfg["tie_break"], cfg["bootstrap"]
    tabs = {"ref": A.race_table(U.outcome_free(p_ref, key)), "ck": A.race_table(U.outcome_free(p_ck, key))}
    gp = p_ref[["race_id", "season", "driverId", "pos", "grid", "classified"]].copy()
    gp["score"] = np.concatenate([DEC.grid_utility(g["grid"].to_numpy(float), g[key].to_numpy()) for _, g in gp.groupby("race_id", sort=False)])
    tabs["grid"] = A.race_table(gp)
    full = A.paired_gain(tabs["ref"], tabs["ck"], n_boot=nb["n_boot"], seed=nb["seed"])
    full.insert(0, "reference", "latent_111")
    full.insert(0, "component", ph["component"])
    full["verdict"] = [A.verdict(r) for _, r in full.iterrows()]
    by_season = pd.concat([A.paired_gain(tabs["ref"][tabs["ref"]["season"] == s], tabs["ck"][tabs["ck"]["season"] == s],
                                         ["spearman", "mae", "exact", "spearman_fin", "mae_fin"], n_boot=nb["n_boot"], seed=nb["seed"]).assign(season=s)
                           for s in (2021, 2022, 2023)], ignore_index=True)
    vs_grid = pd.concat([A.paired_gain(tabs["grid"], tabs["ck"], ["exact", "exact_fin"], n_boot=nb["n_boot"], seed=nb["seed"]).assign(candidate="post-hoc carry_on_total"),
                         A.paired_gain(tabs["grid"], tabs["ref"], ["exact", "exact_fin"], n_boot=nb["n_boot"], seed=nb["seed"]).assign(candidate="latent_111 (reference)")],
                        ignore_index=True)
    vs_grid["verdict"] = [A.verdict(r) for _, r in vs_grid.iterrows()]
    last_id = int(dev_ids[-1])
    n_end = int(entries.index[entries["race_id"] == last_id].max()) + 1
    vshare = variance_share(fit_e_at_cfg(entries, lat, best, n_end, float(last_id + 1), extra), [2021, 2022, 2023])
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print("\nPOST-HOC TUNING (refit_every=4)\n", tune1.round(4).to_string(index=False))
    print("\nCONVERGENCE declared\n", conv0.round(6).to_string(index=False))
    print("\nCONVERGENCE post-hoc\n", conv1.round(6).to_string(index=False))
    print("\nPAIRED GAINS post-hoc vs latent_111\n" + full[full["metric"].isin(VERDICT_METRICS)].round(4).to_string(index=False))
    print("\nPER SEASON\n" + by_season.round(4).to_string(index=False))
    print("\nEXACT VS GRID\n" + vs_grid.round(4).to_string(index=False))
    print("\nVARIANCE SHARE end of dev (post-hoc)\n", json.dumps(vshare, indent=1))
    label = "m14_supplier_pool_carry_total_posthoc"
    tmp_lb = res / "_lb_tmp.csv"
    tmp_lb.unlink(missing_ok=True)
    run_cfg = dict(ck_cfg, component=ph["component"], frozen_hyperparameters="plan-1; lam_e tuned on dev; lam_b = lam",
                   status="POST HOC variant; dev only; no gbm variant")
    rid_ = save_run(MDIR, label, run_cfg, p_ck, m_ck, split, "M14", ck_cfg.get("cell", ""), leaderboard=tmp_lb)
    pth = res / rid_ / "config.yaml"
    cc = load_yaml(pth)
    cc["git_commit"] = stamp
    save_yaml(cc, pth)
    tabs["ck"].to_csv(res / rid_ / "outcome_free_table.csv", index=False)
    new_rows = pd.read_csv(tmp_lb)
    tmp_lb.unlink()
    full.to_csv(res / "paired_gains_dev_posthoc.csv", index=False)
    by_season.to_csv(res / "paired_gains_by_season_posthoc.csv", index=False)
    vs_grid.to_csv(res / "exact_vs_grid_posthoc.csv", index=False)
    (res / "posthoc_summary.json").write_text(json.dumps(dict(
        lam_e_chosen=best, configurations_posthoc=n_cfg, configurations_total=len(old) + n_cfg, budget=30, grid=list(ph["lam_e_grid"]),
        gradient_check=gc, variance_share_end_of_dev=vshare, reference_rerun_vs_stored_max_abs_per_race_metric_diff=d_ref[0],
        rederived_declared_tuning_max_abs_rho_diff=float(np.abs(tune0["spearman"].to_numpy() - old["spearman"].to_numpy()).max())), indent=2),
        encoding="utf-8")
    g = full.drop(columns=["component", "reference", "verdict"])
    row = A.attribution_row(ph["component"], "latent_111", g, cfg["verdict_metric"], rid_, len(tabs["ck"]), declared="no")
    append_attribution_text(pd.DataFrame([row]), ROOT / cfg["attribution"])
    U.append_leaderboard(new_rows, ROOT / cfg["leaderboard"])
    log(f"post-hoc verdict (rho): {row['verdict']}; done in {time.time() - t0:.0f}s")


def append_attribution_text(rows: pd.DataFrame, path: Path) -> None:
    """Append text lines only, in one write (existing lines untouched)."""
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
