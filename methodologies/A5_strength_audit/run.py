"""A5 strength audit (PLAN3.md section 4, R1). Usage:
    python methodologies/A5_strength_audit/run.py [--config methodologies/A5_strength_audit/config.yaml] [--quick]

Stages: tune lam_a on dev -> reproduction check -> snapshot strengths + Laplace intervals per season ->
named checks -> identification -> figures -> acceptance -> run folders + leaderboard rows.
Reusable code is in src/pitlane/plan3/audit.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitlane.config import METHODS, RESULTS, ROOT, load_yaml, save_yaml
from pitlane.evaluation.harness import _git_hash, save_run, summarize, walk_forward
from pitlane.models.registry import build_model
from pitlane.plan3 import audit as A

OUT = RESULTS / "plan3"
FIG = OUT / "figures"
MDIR = METHODS / "A5_strength_audit"
TAGS = {"quali_in": "M03 110 (quali in model; `a` = gain beyond grid slot)",
        "h0": "H0 110 (history only; `a` = overall skill)"}


# ------------------------------------------------------------------------------------ provenance
USED_FILES = ["src/pitlane/plan3/audit.py", "src/pitlane/models/latent.py", "src/pitlane/models/registry.py",
              "methodologies/A5_strength_audit/run.py", "methodologies/A5_strength_audit/config.yaml"]


def code_stamp() -> str:
    """`<HEAD>` if the files this run uses are committed and unchanged, else `<HEAD>+dirty-<sha1 of those files>`.
    Only the files the run uses are checked (the tree also holds unrelated concurrent work)."""
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


def stamp_run(run_id: str, stamp: str) -> None:
    """Replace the harness's plain commit hash in the run's config by the dirty-tree stamp."""
    p = MDIR / "results" / run_id / "config.yaml"
    c = load_yaml(p); c["git_commit"] = stamp; save_yaml(c, p)


# ------------------------------------------------------------------------------------ configs
def model_cfgs(cfg: dict) -> dict[str, dict]:
    f1, f2 = load_yaml(ROOT / cfg["frozen_plan1"]), load_yaml(ROOT / cfg["frozen_plan2"])
    return {k: dict((f1 if v["source"] == "plan1" else f2)[v["key"]][v["cell"]]) for k, v in cfg["models"].items()}


def with_lam_a(c: dict, lam_a: float | None) -> dict:
    return dict(c) if lam_a is None else {**c, "lam_a": float(lam_a)}


def paired_ci(x: np.ndarray, b: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (b, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ------------------------------------------------------------------------------------ stage 1: tune lam_a
def tune_lam_a(entries: pd.DataFrame, base: dict, cfg: dict, tag: str) -> tuple[float, pd.DataFrame]:
    grid = [None] + list(cfg["lam_a_grid"])
    per, rows = {}, []
    for la in grid:
        _, pr = walk_forward(entries, build_model(with_lam_a(base, la), entries), "dev",
                             refit_every=cfg["tune_refit_every"], n_samples=200)
        per[la] = pr.set_index("race_id")["spearman"]
        s = summarize(pr)
        rows.append(dict(model=tag, lam_a=("old (=lam)" if la is None else la), lam_a_eff=base["lam"] if la is None else la,
                         spearman=s["spearman"], mae=s["mae"], logloss_win=s["logloss_win"], n_races=s["n_races"]))
    tab = pd.DataFrame(rows)
    best = int(tab["spearman"].idxmax())
    for i, la in enumerate(grid):
        d = (per[la] - per[grid[best]]).to_numpy()
        m, lo, hi = paired_ci(d, cfg["bootstrap"], cfg["seed"])
        tab.loc[i, ["d_vs_best", "d_lo", "d_hi"]] = (m, lo, hi)
        d0 = (per[la] - per[None]).to_numpy()
        m0, lo0, hi0 = paired_ci(d0, cfg["bootstrap"], cfg["seed"])
        tab.loc[i, ["d_vs_old", "d_old_lo", "d_old_hi"]] = (m0, lo0, hi0)
    ok = tab[(tab["d_hi"] >= 0) & (tab["d_lo"] <= 0) | (tab.index == best)]
    ok = ok[ok["lam_a"] != "old (=lam)"]
    rule1 = float(ok["lam_a_eff"].max())
    # rule 2 (used): empirical Bayes from the fit through the end of the last dev season
    n_train, t_ref = {s: (n, t) for s, n, t in A.snapshot_points(entries)}[cfg["eb_snapshot_season"]]
    m = build_model(base, entries)
    A.fit_tight(m, n_train, t_ref)
    sh = A.shrinkage_table(m, entries, n_train, t_ref)
    sd = float(sh[sh["n_starts"] >= cfg["eb_min_starts"]]["a_unpenalised"].std())
    lam_eb = 1.0 / (2 * sd ** 2)
    grid_v = np.array(cfg["lam_a_grid"], dtype=float)
    chosen = float(grid_v[np.abs(np.log(grid_v) - np.log(lam_eb)).argmin()])
    tab["rule1_largest_within_ci"] = rule1
    tab["lam_a_eb"] = lam_eb
    tab["sd_unpenalised_a_2023"] = sd
    tab["is_best"] = tab.index == best
    tab["is_chosen"] = tab["lam_a_eff"] == chosen
    return chosen, tab


# ------------------------------------------------------------------------------------ stage 2: dev runs
def dev_run(entries: pd.DataFrame, c: dict, label: str, stamp: str, method_cell: str = "110"):
    preds, per = walk_forward(entries, build_model(c, entries), "dev", refit_every=1)
    rid = save_run(MDIR, label, c, preds, per, "dev", "A5", method_cell, leaderboard=OUT / "leaderboard.csv")
    stamp_run(rid, stamp)
    return rid, summarize(per), preds, per


def oos_residual_sd(entries: pd.DataFrame, preds: pd.DataFrame) -> float:
    """Out-of-sample sd of y - (0.5 - score) on the dev predictions (y = normalised finishing position)."""
    d = preds.merge(entries[["race_id", "driverId", "y"]], on=["race_id", "driverId"])
    return float((d["y"] - (0.5 - d["score"])).std())


# ------------------------------------------------------------------------------------ stage 3: tables
def build_tables(entries: pd.DataFrame, variants: dict[str, dict], cfg: dict):
    snaps = A.snapshot_points(entries)
    tabs, diags, finals_d, finals_c, models = [], [], [], [], {}
    for tag, c in variants.items():
        for s, n_train, t_ref in snaps:
            t, d, m, se = A.snapshot_table(entries, c, tag, s, n_train, t_ref, z=cfg["interval_z"])
            tabs.append(t)
            diags.append({"model": tag, **d})
        dr, cr = A.final_tables(m, se, entries, tag, cfg["interval_z"])     # last snapshot = fit through the last race
        finals_d.append(dr); finals_c.append(cr); models[tag] = (m, se)
    return (pd.concat(tabs, ignore_index=True), pd.DataFrame(diags), pd.concat(finals_d, ignore_index=True),
            pd.concat(finals_c, ignore_index=True), models)


def hessian_checks(entries: pd.DataFrame, base: dict) -> dict:
    """Analytic vs finite-difference Hessians (full problem, Gaussian) and gradient-FD vs value-FD (small problems)."""
    out = {}
    full_n = len(entries)
    t_ref = float(entries["race_id"].max() + 1)
    m = build_model(base, entries)
    A.fit_tight(m, full_n, t_ref)
    H_fd, H_an = A.hessian_fd(m, full_n, t_ref), A.hessian_analytic_gaussian(m, full_n, t_ref)
    out["full_gaussian_110_max_abs_diff_fd_vs_analytic"] = float(np.abs(H_fd - H_an).max())
    out["full_gaussian_110_max_abs_H"] = float(np.abs(H_an).max())
    out["full_gaussian_110_min_eigenvalue"] = float(np.linalg.eigvalsh(H_an).min())
    small = entries[entries["race_id"] < 50].reset_index(drop=True)
    for cell in ("110", "111"):
        c = {**base, "cell": cell, "kind": "latent"}
        ms = build_model(c, small)
        A.fit_tight(ms, len(small), 50.0)
        H1, H2 = A.hessian_fd(ms, len(small), 50.0), A.hessian_fd_value_only(ms, len(small), 50.0)
        out[f"small_{cell}_params"] = int(len(ms.theta))
        out[f"small_{cell}_rel_max_diff_gradFD_vs_valueFD"] = float(np.abs(H1 - H2).max() / np.abs(H1).max())
    m150 = build_model(base, entries)
    m150.fit(full_n, t_ref)
    out["theta_max_abs_diff_150iter_vs_tight"] = float(np.abs(m150.theta - m.theta).max())
    return out


# ------------------------------------------------------------------------------------ figures
def plot_cars(cr: pd.DataFrame, title: str, path: Path) -> None:
    lins = sorted(cr["id"].unique())
    fig, axs = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
    for ax, l in zip(axs.ravel(), lins):
        d = cr[cr["id"] == l].sort_values("season")
        ax.fill_between(d["season"], d["lo"], d["hi"], alpha=0.25)
        ax.plot(d["season"], d["est"], "o-", ms=3)
        ax.errorbar(d["season"], d["est"], yerr=[d["est"] - d["lo"], d["hi"] - d["est"]], fmt="none", ecolor="C0",
                    capsize=2, lw=0.9)          # error bars also draw single-season lineages (no band possible)
        for yv in (2017, 2022, 2026):
            ax.axvline(yv - 0.5, color="k", ls=":", lw=0.6)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{l} ({d['n_season'].sum()} starts)", fontsize=9)
    for ax in axs.ravel()[len(lins):]:
        ax.axis("off")
    fig.suptitle(title + " - all 13 car lineages, 95% Laplace band (dotted: regulation changes)")
    fig.supxlabel("season"); fig.supylabel("car strength b")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_drivers(variants: dict[str, pd.DataFrame], title: str, path: Path, min_starts: int) -> None:
    """Columns = variants (penalty), rows = >= min_starts (main) and < min_starts (separate)."""
    ncol = len(variants)
    fig, axs = plt.subplots(2, ncol, figsize=(7 * ncol, 9.5), gridspec_kw={"height_ratios": [5, 1.2]}, squeeze=False)
    for j, (name, d) in enumerate(variants.items()):
        for row, sel in ((0, d["n_total"] >= min_starts), (1, d["n_total"] < min_starts)):
            x = d[sel].sort_values("est")
            ax = axs[row, j]
            if x.empty:
                ax.axis("off"); continue
            col = np.where(x["est"] > 0, "#2ca02c", "#d62728")
            ax.barh(x["id"], x["est"], xerr=[x["est"] - x["lo"], x["hi"] - x["est"]], color=col, ecolor="k", capsize=1.5, alpha=0.8)
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(f"{name}\n" + (f">= {min_starts} starts" if row == 0 else f"< {min_starts} starts (shown separately)"), fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle(title + " (drivers active 2024-26; bars = 95% Laplace interval)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_champion_ranks(cr: pd.DataFrame, path: Path) -> None:
    models = list(cr["model"].unique())
    fig, axs = plt.subplots(1, len(models), figsize=(6 * len(models), 4.2), sharey=True, squeeze=False)
    for ax, mname in zip(axs[0], models):
        d = cr[cr["model"] == mname].sort_values("season")
        ax.plot(d["season"], d["champ_rank"], "o-", label="champion / points leader")
        ax.plot(d["season"], d["mate_rank"], "s--", label="champion's teammate")
        ax.invert_yaxis(); ax.set_title(mname, fontsize=9); ax.set_xlabel("season (snapshot at season end)")
        ax.grid(alpha=0.3)
    axs[0, 0].set_ylabel("rank of a among drivers that season (1 = best)"); axs[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_corr(cor: pd.DataFrame, path: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (r, lo, hi, ttl) in zip(axs, [("r_teammate", "r_teammate_lo", "r_teammate_hi", "teammate correlation (plan 1: +0.54 pooled)"),
                                           ("r_car", "r_car_lo", "r_car_hi", "car correlation (plan 1: +0.92 pooled)")]):
        for k, (basis, mk) in enumerate([("final_fit", "o"), ("snapshot", "s")]):
            d = cor[(cor["basis"] == basis) & (cor["model"] == "quali_in")]
            x = d["season"] + (k - 0.5) * 0.25
            ax.errorbar(x, d[r], yerr=[d[r] - d[lo], d[hi] - d[r]], fmt=mk, capsize=2, label=basis)
        ax.axhline(0, color="k", lw=0.6); ax.set_title(ttl, fontsize=9); ax.set_xlabel("season"); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_shrinkage(sh: pd.DataFrame, lam_a: float, lam0: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sh["eff_starts"], sh["a_old"], s=14, label=f"as frozen (lam_a = {lam0:g})", alpha=0.8)
    ax.scatter(sh["eff_starts"], sh["a_new"], s=14, marker="s", label=f"lam_a = {lam_a:g}", alpha=0.8)
    for _, r in sh[sh["n_starts"] < 10].iterrows():
        ax.annotate(r["driverId"], (r["eff_starts"], r["a_old"]), fontsize=6)
    ax.axhline(0, color="k", lw=0.5); ax.set_xscale("log")
    ax.set_xlabel("effective starts (decay-weighted)"); ax.set_ylabel("driver strength a (M03 110 fit through 2026)")
    ax.set_title("Low-count drivers are pulled to 0 only by the penalty"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_ident(ident: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    w = ident["weak"]
    ax.scatter(ident.loc[~w, "n_starts"], ident.loc[~w, "n_teammates"], s=14, label="adequate")
    ax.scatter(ident.loc[w, "n_starts"], ident.loc[w, "n_teammates"], s=18, c="r", label="flagged weak")
    ax.set_xscale("log"); ax.set_xlabel("starts"); ax.set_ylabel("distinct teammates"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Identification of driver effects"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MDIR / "config.yaml"))
    ap.add_argument("--quick", action="store_true", help="2-point grid, no run folders / leaderboard rows")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.quick:
        cfg["lam_a_grid"] = cfg["lam_a_grid"][:2]
    entries = pd.read_parquet(ROOT / cfg["entries"])
    assert (entries.index == np.arange(len(entries))).all()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    base = model_cfgs(cfg)
    stamp = code_stamp()
    metrics: dict = {"git_commit": stamp, "holdout_note": cfg["holdout_note"]}
    print("[code stamp]", stamp, flush=True)

    # 1. tune lam_a on dev only
    chosen, tuning = {}, []
    for tag, c in base.items():
        ch, tab = tune_lam_a(entries, c, cfg, tag)
        chosen[tag] = ch; tuning.append(tab)
        print(f"[tune] {tag}: chosen lam_a={ch}"); print(tab.round(4).to_string(index=False), flush=True)
    tuning = pd.concat(tuning, ignore_index=True)
    tuning.to_csv(OUT / "tuning_lam_a.csv", index=False)
    (MDIR / "results").mkdir(parents=True, exist_ok=True)
    tuning.to_csv(MDIR / "results" / "tuning_lam_a.csv", index=False)
    metrics["lam_a_chosen"] = chosen

    # 2. dev runs: reproduction of the frozen plan-1 / plan-2 numbers, then the tuned variants
    repro, oos_sd, ll_rows = {}, {}, []
    if not args.quick:
        lb1 = pd.read_csv(RESULTS / "leaderboard.csv")
        ref1 = lb1[(lb1["label"] == "latent_110") & (lb1["split"] == "dev")].iloc[-1]
        lb2 = pd.read_csv(RESULTS / "plan2" / "leaderboard.csv")
        ref2 = lb2[(lb2["label"] == "h0_latent_110") & (lb2["split"] == "dev")].iloc[-1]
        for tag, ref in (("quali_in", ref1), ("h0", ref2)):
            rid, s, pr0, per0 = dev_run(entries, base[tag], f"a5_{tag}_repro_lam_a_off", stamp)
            oos_sd[tag] = oos_residual_sd(entries, pr0)
            repro[tag] = {k: {"new": s[k], "leaderboard": float(ref[k]), "abs_diff": abs(s[k] - float(ref[k]))}
                          for k in ("spearman", "mae", "logloss_win", "n_races")}
            print(f"[repro] {tag}", {k: v["abs_diff"] for k, v in repro[tag].items()}, flush=True)
            _, s2, _, per1 = dev_run(entries, with_lam_a(base[tag], chosen[tag]), f"a5_{tag}_lam_a_{chosen[tag]:g}", stamp)
            repro[tag]["tuned_variant_dev"] = s2
            p0, p1 = per0.set_index("race_id"), per1.set_index("race_id")
            assert (p0.index == p1.index).all()
            for met in ("logloss_win", "spearman"):
                m_, lo_, hi_ = paired_ci((p1[met] - p0[met]).to_numpy(), cfg["bootstrap"], cfg["seed"])
                ll_rows.append(dict(model=tag, metric=met, refit_every=1, lam_a_old=base[tag]["lam"], lam_a_new=chosen[tag],
                                    mean_old=float(p0[met].mean()), mean_new=float(p1[met].mean()), diff_new_minus_old=m_, lo=lo_, hi=hi_,
                                    n_races=len(p0)))
        pd.DataFrame(ll_rows).to_csv(OUT / "paired_dev_refit1.csv", index=False)
        print(pd.DataFrame(ll_rows).round(5).to_string(index=False), flush=True)
        metrics["oos_residual_sd_dev"] = oos_sd
        metrics["reproduction"] = repro

    # 3. strengths tables (4 variants): as frozen and with the tuned lam_a, both models
    variants = {}
    for tag, c in base.items():
        variants[tag] = c
        variants[f"{tag}_lam_a"] = with_lam_a(c, chosen[tag])
    strengths, diags, fdr, fcr, models = build_tables(entries, variants, cfg)
    metrics["fit_convergence"] = {"max_grad_over_all_fits": float(diags["grad_max"].max()), "all_success": bool(diags["success"].all()),
                                  "max_iter_used": int(diags["nit"].max())}
    strengths["role"] = strengths["model"].map(lambda t: TAGS.get(t.replace("_lam_a", ""), t))
    strengths.to_csv(OUT / "strengths.csv", index=False)
    fdr.to_csv(OUT / "drivers_final_fit.csv", index=False); fcr.to_csv(OUT / "cars_final_fit.csv", index=False)
    metrics["hessian_checks"] = hessian_checks(entries, base["quali_in"])
    print("[hessian]", json.dumps(metrics["hessian_checks"], indent=1), flush=True)

    # 4. shrinkage diagnosis (M03 110, fit through the last race, exactly the fig7 setup but converged)
    t_ref, N = float(entries["race_id"].max() + 1), len(entries)
    m_old, _ = models["quali_in"]; m_new, _ = models["quali_in_lam_a"]
    sh_old = A.shrinkage_table(m_old, entries, N, t_ref)
    sh_new = A.shrinkage_table(m_new, entries, N, t_ref)
    sh = sh_old.merge(sh_new[["driverId", "a", "data_weight"]].rename(columns={"a": "a_new", "data_weight": "data_weight_new"}), on="driverId")
    sh = sh.rename(columns={"a": "a_old", "data_weight": "data_weight_old"})
    sh.to_csv(OUT / "shrinkage_diagnosis.csv", index=False)
    est = sh[sh["n_starts"] >= 30]
    tau_hat = float(est["a_unpenalised"].std())
    metrics["shrinkage"] = dict(
        lam=base["quali_in"]["lam"], prior_sd_implied_by_lam=float(np.sqrt(1 / (2 * base["quali_in"]["lam"]))),
        sd_a_established_n30=float(est["a_old"].std()), sd_a_unpenalised_established_n30=tau_hat,
        lam_a_empirical_bayes_upper_bound=float(1 / (2 * tau_hat ** 2)),
        max_abs_stationarity_error=float((sh["a_old"] - sh["a_stationary"]).abs().max()),
        one_race_drivers=sh[sh["n_starts"] == 1][["driverId", "a_old", "a_new", "data_weight_old", "data_weight_new", "a_unpenalised"]].round(4).to_dict("records"),
        mean_a_old=float(sh_old["a"].mean()))
    plot_shrinkage(sh, chosen["quali_in"], base["quali_in"]["lam"], FIG / "shrinkage_cause.png")

    # 5. named checks
    champs = A.champions(entries)
    champs.to_csv(OUT / "champions.csv", index=False)
    ranks = pd.concat([A.champion_ranks(strengths, champs, t) for t in variants], ignore_index=True)
    ranks.to_csv(OUT / "champion_ranks.csv", index=False)
    plot_champion_ranks(ranks[ranks["model"].isin(["quali_in", "h0"])], FIG / "champion_ranks.png")
    cor_rows = []
    for tag in variants:
        m_fin, _ = models[tag]
        a_fin = pd.Series(m_fin._split(m_fin.theta)[0], index=list(m_fin.drivers))
        b_fin = {k: m_fin._split(m_fin.theta)[1][i] for k, i in m_fin.car.items()}
        pt, pc = A.teammate_pairs(entries, a_fin, cfg["min_pair_starts"]), A.car_pairs(entries, b_fin)
        pooled = (A.corr_ci(pt["d_a"].to_numpy(), pt["d_y"].to_numpy()), A.corr_ci(pc["b"].to_numpy(), pc["neg_y"].to_numpy()))
        cor_rows.append(dict(model=tag, basis="final_fit", season="pooled", r_teammate=pooled[0][0], r_teammate_lo=pooled[0][1],
                             r_teammate_hi=pooled[0][2], n_pairs=pooled[0][3], r_car=pooled[1][0], r_car_lo=pooled[1][1],
                             r_car_hi=pooled[1][2], n_cars=pooled[1][3]))
        c1 = A.per_season_correlations(pt, pc).assign(model=tag, basis="final_fit")
        # snapshot basis: a and b as estimated at the end of that same season
        pts, pcs = [], []
        for s in sorted(entries["season"].unique()):
            sn = strengths[(strengths["model"] == tag) & (strengths["season"] == s)]
            a_s = sn[sn["kind"] == "driver"].set_index("id")["est"]
            b_s = {(l, s): v for l, v in sn[sn["kind"] == "car"].set_index("id")["est"].items()}
            es = entries[entries["season"] == s]
            pts.append(A.teammate_pairs(es, a_s, cfg["min_pair_starts"]))
            pcs.append(A.car_pairs(es, b_s))
        c2 = A.per_season_correlations(pd.concat(pts), pd.concat(pcs)).assign(model=tag, basis="snapshot")
        cor_rows += c1.to_dict("records") + c2.to_dict("records")
    cor = pd.DataFrame(cor_rows)
    cor.to_csv(OUT / "correlations_per_season.csv", index=False)
    plot_corr(cor[cor["season"] != "pooled"].assign(season=lambda d: d["season"].astype(int)), FIG / "correlations_per_season.png")
    metrics["pooled_correlations"] = cor[cor["season"] == "pooled"][["model", "r_teammate", "n_pairs", "r_car", "n_cars"]].round(3).to_dict("records")

    # 6. identification (uses the Laplace se of the as-frozen quali_in fit)
    se_a = fdr[fdr["model"] == "quali_in"].set_index("id")["se"]
    ident = A.identification(entries, se_a, cfg["min_starts"], cfg["weak_min_teammates"])
    ident.to_csv(OUT / "identification.csv", index=False)
    plot_ident(ident, FIG / "identification.png")
    metrics["identification"] = dict(n_drivers=int(len(ident)), n_weak=int(ident["weak"].sum()),
                                     n_lt10_starts=int((ident["n_starts"] < 10).sum()),
                                     n_not_in_main_component=int((~ident["in_main_component"]).sum()))

    # 7. figures: cars (all 13 lineages) and drivers (as frozen vs tuned penalty)
    nl = fcr[fcr["model"] == "quali_in"]["id"].nunique()
    metrics["n_lineages_plotted"] = int(nl)
    for tag in ("quali_in", "h0"):
        plot_cars(fcr[fcr["model"] == tag], f"Car strength, {tag}", FIG / f"car_strength_{tag}.png")
        # MAIN figure: the model as frozen, with intervals; drivers with < min_starts starts in a separate panel
        main = {f"{tag}: as frozen (lam_a = lam), the model that makes the predictions": fdr[(fdr["model"] == tag) & (fdr["n_2024_26"] > 0)]}
        plot_drivers(main, f"Driver strength a, {TAGS[tag]}", FIG / f"driver_strength_{tag}.png", cfg["min_starts"])
        # SENSITIVITY figure: not the model's explanation (predictions do not use lam_a)
        dd = {f"{tag}: as frozen (lam_a = lam)": main[next(iter(main))],
              f"{tag}: SENSITIVITY lam_a = {chosen[tag]:g} (not used for prediction)": fdr[(fdr["model"] == f"{tag}_lam_a") & (fdr["n_2024_26"] > 0)]}
        plot_drivers(dd, f"SENSITIVITY to the driver penalty, {TAGS[tag]}", FIG / f"driver_strength_{tag}_lam_a_sensitivity.png", cfg["min_starts"])

    # 7b. between-driver variance (corrected noise), resolved pairs, out-of-sample teammate check
    t_ref, N = float(entries["race_id"].max() + 1), len(entries)
    dv_rows, rp_rows, fld_rows = [], [], []
    late = set(entries[entries["season"] >= 2024]["driverId"])
    for tag in ("quali_in", "h0"):
        m_fin, _ = models[tag]
        sig = {"0.25 (model sigma)": 0.25}
        if tag in oos_sd:
            sig[f"{oos_sd[tag]:.3f} (dev out-of-sample residual sd)"] = oos_sd[tag]
        sig["0.20 (referee sensitivity)"] = 0.20
        dv_rows.append(A.driver_variance(m_fin, N, t_ref, sig, tag, cfg["eb_min_starts"]))
        cov = A.laplace_cov(A.hessian_fd(m_fin, N, t_ref))
        a_fin = m_fin._split(m_fin.theta)[0]
        n_tot = entries.groupby("driverId").size()
        ids = [d for d in sorted(late) if n_tot[d] >= cfg["eb_min_starts"]]
        summ, fld = A.resolved_pairs(a_fin, cov, [m_fin.drivers[d] for d in ids], cfg["interval_z"])
        rp_rows.append(summ.assign(model=tag))
        inv = {i: d for d, i in m_fin.drivers.items()}
        fld_rows.append(fld.assign(model=tag, driver=lambda x: x["idx"].map(inv)).drop(columns="idx"))
    dv = pd.concat(dv_rows, ignore_index=True)
    dv.to_csv(OUT / "driver_variance.csv", index=False)
    rp = pd.concat(rp_rows, ignore_index=True)
    rp.to_csv(OUT / "resolved_pairs.csv", index=False)
    pd.concat(fld_rows, ignore_index=True).to_csv(OUT / "driver_vs_field.csv", index=False)
    print(dv.round(5).to_string(index=False)); print(rp.round(4).to_string(index=False), flush=True)
    metrics["driver_variance"] = dv.to_dict("records"); metrics["resolved_pairs"] = rp.to_dict("records")

    oos = pd.concat([A.oos_teammate_correlations(entries, strengths, t, cfg["min_pair_starts"], seed=cfg["seed"]) for t in variants],
                    ignore_index=True)
    oos.to_csv(OUT / "teammate_oos.csv", index=False)
    print(oos[oos["season"] == "pooled"].round(3).to_string(index=False), flush=True)
    metrics["teammate_oos_pooled"] = oos[oos["season"] == "pooled"].round(4).to_dict("records")

    # 8. acceptance: top/bottom 3 with < min_starts starts
    acc = []
    for tag in variants:
        d = fdr[(fdr["model"] == tag) & (fdr["n_2024_26"] > 0)].sort_values("est")
        ext = pd.concat([d.head(3).assign(where="bottom3"), d.tail(3).assign(where="top3")])
        for _, r in ext.iterrows():
            acc.append(dict(model=tag, population="final fit, drivers active 2024-26", season="final", where=r["where"], driver=r["id"],
                            n_total=r["n_total"], low_sample=bool(r["n_total"] < cfg["min_starts"])))
        for s, g in strengths[(strengths["model"] == tag) & (strengths["kind"] == "driver")].groupby("season"):
            g = g.sort_values("est")
            for where, x in (("bottom3", g.head(3)), ("top3", g.tail(3))):
                for _, r in x.iterrows():
                    acc.append(dict(model=tag, population="season-end snapshot, drivers racing that season", season=int(s), where=where,
                                    driver=r["id"], n_total=r["n_total"], low_sample=bool(r["n_total"] < cfg["min_starts"])))
    acc = pd.DataFrame(acc)
    acc.to_csv(OUT / "acceptance_extremes.csv", index=False)
    viol = acc[acc["low_sample"]].groupby(["model", "population"]).size().rename("n_lowsample_in_top_or_bottom3").reset_index()
    full = acc.groupby(["model", "population"]).size().rename("n_slots").reset_index()
    viol = full.merge(viol, how="left", on=["model", "population"]).fillna({"n_lowsample_in_top_or_bottom3": 0})
    viol.to_csv(OUT / "acceptance_summary.csv", index=False)
    print(viol.to_string(index=False))
    fig7 = acc[(acc["season"] == "final") & acc["low_sample"]][["model", "where", "driver", "n_total"]]
    print("[acceptance] low-sample drivers in final-fit top/bottom 3:\n", fig7.to_string(index=False), flush=True)
    metrics["acceptance_final_fit_lowsample_extremes"] = fig7.to_dict("records")

    # acceptance under the final layout: main panel = drivers with >= min_starts starts; the rest are listed separately
    mp = []
    for tag in variants:
        d = fdr[(fdr["model"] == tag) & (fdr["n_2024_26"] > 0)]
        main_d, low_d = d[d["n_total"] >= cfg["min_starts"]].sort_values("est"), d[d["n_total"] < cfg["min_starts"]].sort_values("est")
        for where, x in (("bottom3", main_d.head(3)), ("top3", main_d.tail(3))):
            mp += [dict(model=tag, panel="main (>= min starts)", where=where, driver=r["id"], n_total=int(r["n_total"]), est=r["est"]) for _, r in x.iterrows()]
        mp += [dict(model=tag, panel="separate (< min starts)", where="listed", driver=r["id"], n_total=int(r["n_total"]), est=r["est"]) for _, r in low_d.iterrows()]
    mp = pd.DataFrame(mp)
    mp.to_csv(OUT / "acceptance_main_panel.csv", index=False)
    print(mp.to_string(index=False), flush=True)
    metrics["acceptance_main_panel_lowsample_in_extremes"] = int(((mp["panel"] == "main (>= min starts)") & (mp["n_total"] < cfg["min_starts"])).sum())

    (OUT / "a5_summary.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    sys.exit(main())
