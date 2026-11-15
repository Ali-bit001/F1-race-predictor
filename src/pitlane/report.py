"""Build results/REPORT.md and results/figures/*.png from the saved runs (PLAN.md WP8 + section 5).

Everything in the report is read from run folders listed in results/leaderboard.csv, so each number
traces back to a file.
"""
from __future__ import annotations

import itertools
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from pitlane.config import CONFIGS, ROOT, RESULTS, load_yaml
from pitlane.evaluation.metrics import ece
from pitlane.models.registry import build_model

FIG = RESULTS / "figures"
CELLS = [f"{a}{b}{c}" for a, b, c in itertools.product("01", repeat=3)]
LADDER = {"000": "A Direct", "100": "B Dynamic", "110": "C Decomposed", "111": "D Proposed"}
SPLIT_NAME = {"dev": "Development 2021-23", "test_stable": "Test 2024-25 (stable regs)",
              "test_shift": "Test 2026 (regulation shift)"}
FAMILY = {"L": ("latent", "Family L: latent strength model"), "G": ("gbm", "Family G: gradient-boosted trees")}
rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------------- loading
def load_runs() -> pd.DataFrame:
    lb = pd.read_csv(RESULTS / "leaderboard.csv")
    lb = lb.drop_duplicates(["label", "split"], keep="last").reset_index(drop=True)
    lb["per_race"] = [pd.read_csv(ROOT / d / "per_race_metrics.csv") for d in lb["run_dir"]]
    return lb


def get(lb: pd.DataFrame, label: str, split: str) -> pd.DataFrame | None:
    m = lb[(lb["label"] == label) & (lb["split"] == split)]
    return None if m.empty else m.iloc[0]["per_race"].set_index("race_id")


def boot_ci(x: np.ndarray, b: int = 5000) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(x), (b, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilcoxon_p(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 6 or np.allclose(x, 0):
        return np.nan
    return float(stats.wilcoxon(x).pvalue)


def fmt_ci(t: tuple[float, float, float], d: int = 3) -> str:
    return f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------- tables
def leaderboard_table(lb: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for _, r in lb[lb["split"] == split].iterrows():
        rows.append({"model": f"{r['method']} {r['label']}", "rho": r["spearman"], "MAE": r["mae"],
                     "rho (finishers)": r["spearman_fin"], "NDCG@10": r["ndcg10"], "win hit": r["win_hit"],
                     "podium overlap": r["podium"], "win logloss": r["logloss_win"], "races": int(r["n_races"])})
    t = pd.DataFrame(rows).sort_values("rho", ascending=False)
    for c in ["rho", "rho (finishers)", "NDCG@10", "win hit", "podium overlap", "win logloss"]:
        t[c] = t[c].map("{:.3f}".format)
    t["MAE"] = t["MAE"].map("{:.2f}".format)
    return t


def cell_matrix(lb: pd.DataFrame, kind: str, split: str, metric: str) -> pd.DataFrame | None:
    cols = {}
    for c in CELLS:
        pr = get(lb, f"{kind}_{c}", split)
        if pr is None:
            return None
        cols[c] = pr[metric]
    return pd.DataFrame(cols)


def factor_effects(M: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-race main effects (F1,F2,F3) and two-way interactions from the 2x2x2 per-race matrix."""
    sign = {f: np.array([1 if c[i] == "1" else -1 for c in CELLS]) for i, f in enumerate(["F1", "F2", "F3"])}
    V = M[CELLS].to_numpy()
    out = {}
    for f, s in sign.items():
        out[f] = V[:, s == 1].mean(1) - V[:, s == -1].mean(1)
    for f, g in itertools.combinations(sign, 2):
        s = sign[f] * sign[g]
        out[f"{f}x{g}"] = (V[:, s == 1].mean(1) - V[:, s == -1].mean(1)) / 2
    return out


FACTOR_NAME = {"F1": "F1 dynamic", "F2": "F2 driver-car separation", "F3": "F3 structured ranking",
               "F1xF2": "F1 x F2", "F1xF3": "F1 x F3", "F2xF3": "F2 x F3"}


# ----------------------------------------------------------------------------------- figures
def fig_factorial(lb: pd.DataFrame, splits: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"dev": "#8c8c8c", "test_stable": "#1f77b4", "test_shift": "#d62728"}
    for ax, (metric, title, sgn) in zip(axes, [("spearman", "Spearman rho gain (right = better)", 1),
                                               ("mae", "Position-MAE reduction (right = better)", -1)]):
        ticks, labels, y = [], [], 0.0
        for fam, (kind, fname) in FAMILY.items():
            for f in ["F1", "F2", "F3"]:
                for k, sp in enumerate(splits):
                    M = cell_matrix(lb, kind, sp, metric)
                    if M is None:
                        continue
                    m, lo, hi = boot_ci(sgn * factor_effects(M)[f])
                    ax.errorbar(m, y + k * 0.22, xerr=[[m - lo], [hi - m]], fmt="o", color=colors[sp], capsize=3,
                                label=SPLIT_NAME[sp] if (fam == "L" and f == "F1") else None)
                ticks.append(y + 0.22 * (len(splits) - 1) / 2); labels.append(f"{fam}: {FACTOR_NAME[f]}")
                y += 1
            y += 0.6
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(ticks, labels); ax.invert_yaxis(); ax.set_title(title)
        ax.set_xlabel("main effect of switching the factor ON (95% bootstrap CI over races)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Does each proposed idea help on races the model has never seen?", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_factorial_effects.png", dpi=150)
    plt.close(fig)


def fig_ladder(lb: pd.DataFrame, splits: list[str]) -> None:
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4.2), sharey=True, squeeze=False)
    for ax, sp in zip(axes[0], splits):
        w = 0.38
        for j, (fam, (kind, fname)) in enumerate(FAMILY.items()):
            vals = []
            for c in LADDER:
                pr = get(lb, f"{kind}_{c}", sp)
                vals.append(boot_ci(pr["spearman"].to_numpy()) if pr is not None else (np.nan,) * 3)
            m = np.array([v[0] for v in vals]); lo = np.array([v[1] for v in vals]); hi = np.array([v[2] for v in vals])
            ax.bar(np.arange(4) + (j - 0.5) * w, m, w, yerr=[m - lo, hi - m], capsize=3,
                   label=fname.split(":")[0].replace("Family ", "Family "), color=["#1f77b4", "#ff7f0e"][j])
        for lab, col in [("grid", "k")]:
            pr = get(lb, lab, sp)
            if pr is not None:
                ax.axhline(pr["spearman"].mean(), color=col, ls="--", lw=1, label="grid-order baseline")
        ax.set_xticks(range(4), list(LADDER.values()), rotation=15)
        ax.set_title(SPLIT_NAME[sp], fontsize=10)
        ax.set_ylim(max(0.3, ax.get_ylim()[0]), None)
    axes[0][0].set_ylabel("mean per-race Spearman rho")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Ablation ladder A -> D (95% bootstrap CI over races)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_ablation_ladder.png", dpi=150)
    plt.close(fig)


def fig_rolling(lb: pd.DataFrame, split_list: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    series = [("grid", "Grid order", "#7f7f7f"), ("gbm_000", "G-000 direct GBM", "#ff7f0e"),
              ("gbm_001", "G-001 GBM ranking", "#bcbd22"), ("latent_000", "L-000 latent, all off", "#17becf"),
              ("latent_111", "L-111 proposed", "#d62728")]
    for lab, name, col in series:
        parts = [get(lb, lab, s) for s in split_list]
        if any(p is None for p in parts):
            continue
        pr = pd.concat(parts).reset_index()
        ax.plot(np.arange(len(pr)), pr["spearman"].rolling(10, min_periods=3).mean(), label=name, color=col, lw=2 if lab == "latent_111" else 1.3)
    n = 0
    for s in split_list:
        n += len(get(lb, "grid", s))
        ax.axvline(n - 0.5, color="k", ls=":", lw=0.8)
    ax.set_xlabel("race (chronological, dev -> test)"); ax.set_ylabel("Spearman rho (10-race rolling mean)")
    ax.set_title("Out-of-time ranking quality over time"); ax.legend(fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "fig3_rolling_spearman.png", dpi=150); plt.close(fig)


def fig_calibration(lb: pd.DataFrame, splits: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = [("latent_000", "L-000 (all off)", "#17becf"), ("latent_111", "L-111 (proposed)", "#d62728"),
              ("gbm_000", "G-000 (direct GBM)", "#ff7f0e")]
    for ax, (col, thr, name) in zip(axes, [("p_win", 1, "win"), ("p_podium", 3, "podium"), ("p_points", 10, "points")]):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        for lab, nm, c in labels:
            ps = []
            for s in splits:
                m = _runrow(lab, s)
                if m is not None:
                    ps.append(pd.read_parquet(ROOT / m / "predictions.parquet"))
            if not ps:
                continue
            d = pd.concat(ps)
            p, y = d[col].to_numpy(), (d["pos"] <= thr).to_numpy(float)
            bins = np.quantile(p, np.linspace(0, 1, 11)); bins = np.unique(bins)
            idx = np.clip(np.digitize(p, bins[1:-1]), 0, len(bins) - 2)
            xs = [p[idx == b].mean() for b in range(len(bins) - 1) if (idx == b).any()]
            ys = [y[idx == b].mean() for b in range(len(bins) - 1) if (idx == b).any()]
            ax.plot(xs, ys, "o-", label=f"{nm}  ECE={ece(p, y):.3f}", color=c, ms=4)
        ax.set_title(f"P({name}) reliability (test blocks pooled)"); ax.set_xlabel("predicted"); ax.set_ylabel("observed")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig4_calibration.png", dpi=150); plt.close(fig)


_LB = None


def _runrow(label: str, split: str) -> str | None:
    m = _LB[(_LB["label"] == label) & (_LB["split"] == split)]
    return None if m.empty else m.iloc[0]["run_dir"]


def fig_error_anatomy(lb: pd.DataFrame, splits: list[str]) -> str:
    rows = []
    for lab in ["grid", "latent_000", "latent_111", "gbm_000"]:
        parts = [get(lb, lab, s) for s in splits]
        if any(p is None for p in parts):
            continue
        rows.append((lab, pd.concat(parts)))
    if not rows:
        return ""
    ent = pd.read_parquet(ROOT / "data/processed/entries_feat.parquet")
    dnf_share = ent.groupby("race_id")["dnf"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    out = []
    for lab, pr in rows:
        pr = pr.copy(); pr["dnf_share"] = dnf_share.reindex(pr.index).to_numpy()
        pr["bucket"] = pd.cut(pr["dnf_share"], [-.01, 0.1, 0.2, 1.0], labels=["<=10% DNF", "10-20% DNF", ">20% DNF"])
        g = pr.groupby("bucket", observed=True)["spearman"].mean()
        ax.plot(g.index.astype(str), g.values, "o-", label=lab)
        out.append(g.rename(lab))
    ax.set_ylabel("mean Spearman rho"); ax.set_title("Error anatomy: ranking quality vs. race attrition (test blocks)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "fig5_error_anatomy.png", dpi=150); plt.close(fig)
    tab = pd.concat(out, axis=1).round(3).reset_index().rename(columns={"bucket": "DNF share"})
    tab["DNF share"] = tab["DNF share"].astype(str)
    return md_table(tab)


# ----------------------------------------------------------------------------------- explainability
def explain_section(entries: pd.DataFrame, frozen: dict) -> str:
    """Strength trajectories, additive breakdown, face validity and SHAP, using frozen configs."""
    txt = []
    ent = entries
    cfg_dyn = frozen["M03"]["110"]
    model = build_model(cfg_dyn, ent)
    last = int(ent["race_id"].max())
    model.fit(len(ent), t_ref=float(last + 1))
    a, b, g, beta = model._split(model.theta)

    # --- car strength trajectories
    car = pd.DataFrame([(l, s, b[i]) for (l, s), i in model.car.items()], columns=["lineage", "season", "b"])
    seas_n = ent.groupby(["lineage", "season"]).size().rename("n").reset_index()
    car = car.merge(seas_n, on=["lineage", "season"])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    top = car.groupby("lineage")["n"].sum().nlargest(8).index
    for l in top:
        d = car[car["lineage"] == l].sort_values("season")
        ax.plot(d["season"], d["b"], "o-", label=l.replace("_lineage", ""))
    for yv in (2017, 2022, 2026):
        ax.axvline(yv - 0.5, color="k", ls=":", lw=0.8)
    ax.set_xlabel("season"); ax.set_ylabel("car strength b (log-odds units)")
    ax.set_title("Estimated car strength by season (dotted: regulation changes)"); ax.legend(fontsize=8, ncol=4)
    fig.tight_layout(); fig.savefig(FIG / "fig6_car_strength_trajectories.png", dpi=150); plt.close(fig)

    # --- driver strengths (recent drivers)
    drv = pd.Series(a, index=list(model.drivers)).rename("a")
    recent = ent[ent["season"] >= 2024]["driverId"].unique()
    d = drv[drv.index.isin(recent)].sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(d.index, d.values, color=np.where(d.values > 0, "#2ca02c", "#d62728"))
    ax.set_title("Driver strength a (car-adjusted), drivers active 2024-26"); ax.set_xlabel("a")
    fig.tight_layout(); fig.savefig(FIG / "fig7_driver_strengths.png", dpi=150); plt.close(fig)

    # --- face validity: teammate comparison and car vs results
    ds = ent.groupby(["season", "lineage", "driverId"]).agg(y=("y", "mean"), n=("y", "size")).reset_index()
    ds = ds[ds["n"] >= 8]
    pairs = []
    for (s, l), grp in ds.groupby(["season", "lineage"]):
        if len(grp) == 2:
            x, z = grp.iloc[0], grp.iloc[1]
            pairs.append((drv[x["driverId"]] - drv[z["driverId"]], z["y"] - x["y"]))
    pa = np.array(pairs)
    r_team = np.corrcoef(pa[:, 0], pa[:, 1])[0, 1] if len(pa) > 5 else np.nan
    cs = ent.groupby(["lineage", "season"])["y"].mean().rename("y").reset_index().merge(car, on=["lineage", "season"])
    r_car = np.corrcoef(cs["b"], -cs["y"])[0, 1]
    txt.append(f"- **Teammate face-validity:** across {len(pa)} same-team same-season driver pairs, the difference in "
               f"estimated driver strength `a` correlates **{r_team:+.2f}** with the difference in mean finishing "
               f"position (same car, so this isolates the driver).")
    txt.append(f"- **Car face-validity:** estimated car strength `b` correlates **{r_car:+.2f}** with a constructor's "
               f"mean finishing position in that season ({len(cs)} constructor-seasons).")

    # --- additive breakdown for the final race
    races = ent["race_id"].max()
    cfg111 = frozen["M03"]["111"]
    m2 = build_model(cfg111, ent)
    rows = np.where(ent["race_id"] == races)[0]
    m2.fit(int(rows.min()), t_ref=float(races))
    comp = m2.components(rows)
    comp.index = ent.loc[rows, "driverId"].to_numpy()
    comp = comp.sort_values("u", ascending=False)
    info = ent.loc[rows].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    left_pos = np.zeros(len(comp)); left_neg = np.zeros(len(comp))
    for term, col in [("a_driver", "#1f77b4"), ("b_car", "#ff7f0e"), ("g_circuit", "#9467bd"), ("weekend", "#2ca02c")]:
        v = comp[term].to_numpy()
        left = np.where(v >= 0, left_pos, left_neg)
        ax.barh(range(len(comp)), v, left=left, color=col, label=term)
        left_pos += np.where(v >= 0, v, 0); left_neg += np.where(v < 0, v, 0)
    ax.scatter(comp["u"], range(len(comp)), color="k", zorder=5, s=14, label="total u")
    ax.set_yticks(range(len(comp)), comp.index); ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.6)
    ax.set_title(f"Why the model ranks the field this way: {info['raceName']} {info['season']} (pre-race, L-111)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "fig8_prediction_breakdown.png", dpi=150); plt.close(fig)
    bd = comp.round(2).head(6).reset_index().rename(columns={"index": "driver"})
    txt.append(f"\nTop-6 pre-race breakdown for **{info['raceName']} {info['season']}** "
               f"(each score is the exact sum of its parts):\n\n{md_table(bd)}")
    txt.append("\nFitted weekend-signal coefficients (L-111): " + ", ".join(f"`{k}`={v:+.2f}" for k, v in m2.beta.items()))

    # --- SHAP (LightGBM TreeSHAP) for the proposed GBM cell
    try:
        import lightgbm as lgb  # noqa: F401
        gm = build_model(frozen["M04"]["111"], ent)
        first = int(rows.min())
        gm.fit_predict(ent.iloc[:first], ent.iloc[rows])
        contrib = gm.model.predict(ent.iloc[rows][gm.cols], pred_contrib=True)[:, :-1]
        imp = pd.Series(np.abs(contrib).mean(0), index=gm.cols).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(imp.index, imp.values, color="#1f77b4"); ax.set_title("Family G proposed cell: mean |TreeSHAP| (final race)")
        fig.tight_layout(); fig.savefig(FIG / "fig9_gbm_shap.png", dpi=150); plt.close(fig)
    except Exception as e:  # pragma: no cover
        txt.append(f"(SHAP figure skipped: {e})")
    return "\n".join(txt)


# ----------------------------------------------------------------------------------- main
def make_report() -> None:
    global _LB
    warnings.filterwarnings("ignore")
    FIG.mkdir(parents=True, exist_ok=True)
    lb = load_runs()
    _LB = lb
    splits = [s for s in ["dev", "test_stable", "test_shift"] if (lb["split"] == s).any()]
    test_splits = [s for s in splits if s != "dev"]
    frozen = load_yaml(CONFIGS / "frozen.yaml")
    md = ["# PitLane — results report", "",
          f"_Generated from `results/leaderboard.csv`; configs frozen on dev (hash `{frozen.get('hash')}`) before the test blocks were run. "
          "Every number is a mean over races of per-race metrics; uncertainty is a bootstrap over races._", "",
          "**Primary research question:** can explicit separation of car, driver and circuit effects (with dynamic weighting and "
          "structured ranking) improve out-of-time F1 race-order prediction?", ""]

    # 1. Headline
    md += ["## 1. Leaderboards", "Higher is better for rho / NDCG / hits; lower for MAE and log-loss. "
           "Cell code = F1 dynamic, F2 decomposed, F3 ranking (e.g. `latent_111` = proposed model, `000` = all off).", ""]
    for s in splits:
        md += [f"### {SPLIT_NAME[s]}", md_table(leaderboard_table(lb, s)), ""]

    # 2. Ladder
    md += ["## 2. The deck's ablation ladder (A -> D), paired per race", ""]
    for s in test_splits or splits:
        for fam, (kind, fname) in FAMILY.items():
            rows = []
            prev = None
            base = get(lb, f"{kind}_000", s)
            for c, nm in LADDER.items():
                pr = get(lb, f"{kind}_{c}", s)
                if pr is None or base is None:
                    continue
                r = {"variant": f"{c} {nm}", "rho": f"{pr['spearman'].mean():.3f}", "MAE": f"{pr['mae'].mean():.2f}"}
                if prev is not None:
                    d = (pr["spearman"] - prev["spearman"]).to_numpy(); dm = (prev["mae"] - pr["mae"]).to_numpy()
                    r["d rho vs previous rung"] = fmt_ci(boot_ci(d)); r["p (Wilcoxon)"] = f"{wilcoxon_p(d):.3f}"
                    r["MAE gain vs previous"] = fmt_ci(boot_ci(dm), 2)
                    d0 = (pr["spearman"] - base["spearman"]).to_numpy()
                    r["d rho vs A"] = fmt_ci(boot_ci(d0))
                prev = pr
                rows.append(r)
            md += [f"**{fname} — {SPLIT_NAME[s]}**", "", md_table(pd.DataFrame(rows).fillna("")), ""]

    # 3. Factorial
    md += ["## 3. Factorial effects (all 8 on/off combinations)",
           "Main effect = average gain from switching a factor ON, over all combinations of the other two, "
           "estimated per race then bootstrapped over races. Interactions are half the difference of one factor's "
           "effect across the levels of another (positive = the two help each other).", ""]
    for s in splits:
        for fam, (kind, fname) in FAMILY.items():
            Mr, Mm = cell_matrix(lb, kind, s, "spearman"), cell_matrix(lb, kind, s, "mae")
            if Mr is None:
                continue
            er, em = factor_effects(Mr), factor_effects(Mm)
            rows = [{"effect": FACTOR_NAME[f], "d rho [95% CI]": fmt_ci(boot_ci(er[f])),
                     "p (Wilcoxon)": f"{wilcoxon_p(er[f]):.3f}", "MAE gain [95% CI]": fmt_ci(boot_ci(-em[f]), 2)} for f in er]
            md += [f"**{fname} — {SPLIT_NAME[s]}** ({len(Mr)} races)", "", md_table(pd.DataFrame(rows)), ""]
    md += ["![factorial](figures/fig1_factorial_effects.png)", "", "![ladder](figures/fig2_ablation_ladder.png)", ""]

    # 4. versus baselines
    md += ["## 4. Against the baselines (paired, per race)", ""]
    rows = []
    for s in splits:
        for lab in ["grid", "quali", "standings", "last_race", "season_mean", "rf_000", "gbm_000", "gbm_001"]:
            b, p = get(lb, lab, s), get(lb, "latent_111", s)
            if b is None or p is None:
                continue
            d = (p["spearman"] - b["spearman"]).to_numpy(); dm = (b["mae"] - p["mae"]).to_numpy()
            rows.append({"split": s, "baseline": lab, "base rho": f"{b['spearman'].mean():.3f}",
                         "L-111 rho": f"{p['spearman'].mean():.3f}", "d rho [95% CI]": fmt_ci(boot_ci(d)),
                         "p": f"{wilcoxon_p(d):.3f}", "MAE gain [95% CI]": fmt_ci(boot_ci(dm), 2)})
    md += [md_table(pd.DataFrame(rows)), ""]

    # 5. figures
    fig_factorial(lb, splits); fig_ladder(lb, splits); fig_rolling(lb, splits); fig_calibration(lb, test_splits or splits)
    anat = fig_error_anatomy(lb, test_splits or splits)
    md += ["## 5. Over time, calibration and error anatomy", "![rolling](figures/fig3_rolling_spearman.png)", "",
           "![calibration](figures/fig4_calibration.png)", "", "![anatomy](figures/fig5_error_anatomy.png)", "", anat, ""]

    # 6. explainability
    entries = pd.read_parquet(ROOT / "data/processed/entries_feat.parquet")
    try:
        from pitlane.pipeline import add_latent_strengths
        entries = add_latent_strengths(entries, frozen)
        ex = explain_section(entries, frozen)
    except Exception as e:  # keep the rest of the report if explainability fails
        ex = f"(explainability section failed: {e!r})"
    md += ["## 6. Explainability", ex, "",
           "![cars](figures/fig6_car_strength_trajectories.png)", "", "![drivers](figures/fig7_driver_strengths.png)", "",
           "![breakdown](figures/fig8_prediction_breakdown.png)", "", "![shap](figures/fig9_gbm_shap.png)", ""]

    # 7. chosen configs
    md += ["## 7. Frozen configurations (tuned on dev only)", "```yaml"]
    import yaml
    md += [yaml.safe_dump({k: frozen[k] for k in ("M03", "M04", "M01rf")}, sort_keys=False), "```", ""]
    (RESULTS / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", RESULTS / "REPORT.md")
