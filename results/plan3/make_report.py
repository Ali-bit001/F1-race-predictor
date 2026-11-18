"""PLAN3 R16: build results/plan3/REPORT.md and results/plan3/figures/r16_*.png from files already on disk.

    python results/plan3/make_report.py

Reads only existing result files (results/plan3/*.csv|json, methodologies/*/results/**) and writes REPORT.md plus
figures. Deterministic: fixed seed (0) for the one bootstrap re-derivation, no model is fitted, nothing is fetched,
no existing file is edited. Every number in the report is read or recomputed here; the text around it is fixed.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R3 = ROOT / "results" / "plan3"
FIG = R3 / "figures"
MET = ROOT / "methodologies"
P3R = MET / "P3_final" / "results"
TL_LABEL = "third look, already seen, no decision weight"
SEED, NBOOT = 0, 10_000

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 130, "savefig.dpi": 130})


# ---------------------------------------------------------------- helpers
def sg(x: float, d: int = 4) -> str:
    return f"{x:+.{d}f}"


def iv(g: float, lo: float, hi: float, d: int = 4) -> str:
    return f"{sg(g, d)} [{sg(lo, d)}, {sg(hi, d)}]"


def excl0(lo: float, hi: float) -> bool:
    return bool(lo > 0 or hi < 0)


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def esc(s: object) -> str:
    return str(s).replace("|", "/").replace("\n", " ")


ATT = pd.read_csv(R3 / "attribution.csv")
TL = pd.read_csv(R3 / "third_look.csv")
LB = pd.read_csv(R3 / "leaderboard.csv")
HS = json.loads((R3 / "holdout_status.json").read_text())


def att(comp: str, ref: str, m: str, d: int = 4) -> str:
    """'gain [lo, hi]' for one attribution row and one metric suffix (spearman, mae, exact, rps, ...)."""
    r = ATT[(ATT.component == comp) & (ATT.reference == ref)]
    assert len(r) == 1, (comp, ref, len(r))
    r = r.iloc[0]
    return iv(r[f"gain_{m}"], r[f"lo_{m}"], r[f"hi_{m}"], d)


def att_rows(comp: str, ref: str) -> pd.Series:
    r = ATT[(ATT.component == comp) & (ATT.reference == ref)]
    assert len(r) == 1, (comp, ref)
    return r.iloc[0]


def tl(block: str, name: str, metric: str, d: int = 4) -> str:
    r = TL[(TL.table == "contrast") & (TL.block == block) & (TL.name == name) & (TL.metric == metric)]
    assert len(r) == 1, (block, name, metric, len(r))
    r = r.iloc[0]
    return f"{iv(r.gain, r.lo, r.hi, d)} (n={int(r.n)})"


def tlrow(block: str, name: str, metric: str) -> pd.Series:
    r = TL[(TL.table == "contrast") & (TL.block == block) & (TL.name == name) & (TL.metric == metric)]
    assert len(r) == 1, (block, name, metric)
    return r.iloc[0]


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(esc(r[c]) for c in cols) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- re-derivation from per-race files
def boot_ci(diff: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(NBOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_race(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "per_race_metrics.csv")


def rederive() -> tuple[pd.DataFrame, dict]:
    """Re-derive the headline contrasts from per-race files (point estimate exact, interval by own bootstrap)."""
    fin = per_race(P3R / "20261118-0705_p3_final_model_dev")
    lat = per_race(P3R / "20261118-0705_p3_ref_latent_111_dev")
    grid = per_race(MET / "M17_decoding" / "results" / "20261116-2028_m17_grid_free_dev")
    mer = per_race(P3R / "20261118-0705_p3_loo_r1_minus_reverse_dev")   # final minus reverse = pace_hist only
    rev = per_race(P3R / "20261118-0705_p3_loo_r1_minus_pace_hist_dev")  # final minus pace_hist = reverse only
    for d in (lat, grid, mer, rev):
        assert (d.race_id.values == fin.race_id.values).all()
    rows = []

    def add(label: str, diff: np.ndarray, stored: pd.Series, n: int) -> None:
        lo, hi = boot_ci(diff)
        rows.append({"contrast": label, "n": n, "gain (per-race files)": round(float(diff.mean()), 6),
                     "gain (stored)": round(float(stored.gain), 6),
                     "own 95% bootstrap": f"[{lo:+.4f}, {hi:+.4f}]", "stored interval": f"[{stored.lo:+.4f}, {stored.hi:+.4f}]"})

    def st(comp: str, ref: str, m: str) -> pd.Series:
        r = att_rows(comp, ref)
        return pd.Series({"gain": r[f"gain_{m}"], "lo": r[f"lo_{m}"], "hi": r[f"hi_{m}"]})

    add("dev T1 final - latent_111, rho", (fin.spearman - lat.spearman).values, st("P3_final", "latent_111", "spearman"), len(fin))
    add("dev final - grid, exact rate", (fin.exact - grid.exact).values, st("P3_final", "grid", "exact"), len(fin))
    add("dev T2 final vs final_minus_reverse, MAE (ref - cand)", (mer.mae - fin.mae).values,
        st("P3_LOO_reverse", att_rows("P3_LOO_reverse", ATT[ATT.component == "P3_LOO_reverse"].reference.iloc[0]).reference, "mae"), len(fin))
    add("dev T3 final - reverse_only, rho", (fin.spearman - rev.spearman).values,
        st("P3_LOO_pace_hist", ATT[ATT.component == "P3_LOO_pace_hist"].reference.iloc[0], "spearman"), len(fin))
    # third look, pooled 2024-26 R20
    sc = per_race(P3R / "20261118-1407_p3_R15_score")
    a = sc[sc.model == "final"].set_index("race_id")
    b = sc[sc.model == "latent_111"].set_index("race_id")
    c = sc[sc.model == "reverse_only"].set_index("race_id")
    d = sc[sc.model == "final_minus_reverse"].set_index("race_id")
    g = sc[sc.model == "grid"].set_index("race_id")
    ids = a.index
    assert all((x.index == ids).all() for x in (b, c, d, g))
    add("third look pooled T1 final - latent_111, rho", (a.spearman - b.spearman).values, tlrow("pooled test_stable+test_shift", "T1:final_vs_latent_111", "spearman"), len(ids))
    add("third look pooled T2 MAE (ref - cand)", (d.mae - a.mae).values, tlrow("pooled test_stable+test_shift", "T2:final_vs_final_minus_reverse", "mae"), len(ids))
    add("third look pooled T3 final - reverse_only, rho", (a.spearman - c.spearman).values, tlrow("pooled test_stable+test_shift", "T3:final_vs_reverse_only", "spearman"), len(ids))
    add("third look pooled final - grid, exact rate", (a.exact - g.exact).values, tlrow("pooled test_stable+test_shift", "exact_vs_grid:final", "exact"), len(ids))
    return pd.DataFrame(rows), {"fin": fin, "lat": lat, "grid": grid, "sc": sc}


# ---------------------------------------------------------------- figures
def fig_dev_vs_third(path: Path) -> None:
    pooled, stable, shift = "pooled test_stable+test_shift", "test_stable", "test_shift"
    panels = [
        ("T1: final - latent_111, Spearman rho", att_rows("P3_final", "latent_111"), "spearman", "T1:final_vs_latent_111", "spearman"),
        ("T2: reverse on/off, MAE gain (lower MAE = positive)", att_rows("P3_LOO_reverse", ATT[ATT.component == "P3_LOO_reverse"].reference.iloc[0]), "mae", "T2:final_vs_final_minus_reverse", "mae"),
        ("T3: final - reverse_only, Spearman rho", att_rows("P3_LOO_pace_hist", ATT[ATT.component == "P3_LOO_pace_hist"].reference.iloc[0]), "spearman", "T3:final_vs_reverse_only", "spearman"),
        ("Exact-position rate: final - grid order", att_rows("P3_final", "grid"), "exact", "exact_vs_grid:final", "exact"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), sharey=True)
    labels = ["dev 2021-23\n(n=66, decisions)", "2024-25\n(n=48)", "2026 R1-20\n(n=20)", "2024-26 R20 pooled\n(n=68)"]
    for ax, (title, drow, dm, name, tm) in zip(axes, panels):
        pts = [(drow[f"gain_{dm}"], drow[f"lo_{dm}"], drow[f"hi_{dm}"], "black")]
        for blk in (stable, shift, pooled):
            r = tlrow(blk, name, tm)
            pts.append((r.gain, r.lo, r.hi, "tab:gray"))
        for i, (g, lo, hi, col) in enumerate(pts):
            ax.errorbar(g, i, xerr=[[g - lo], [hi - g]], fmt="o", color=col, capsize=4, lw=2)
            ax.text(g, i - 0.3, f"{g:+.4f} [{lo:+.4f}, {hi:+.4f}]", fontsize=7.5, va="bottom", ha="center", color=col)
        ax.axvline(0, color="k", lw=0.8)
        ax.axhspan(0.5, 3.5, color="0.93", zorder=0)
        ax.set_title(title, fontsize=9)
        ax.set_yticks(range(4))
        ax.set_yticklabels(labels)
        ax.set_ylim(3.6, -0.6)
        lo_all = min(p[1] for p in pts)
        hi_all = max(p[2] for p in pts)
        pad = 0.6 * (hi_all - lo_all)
        span = hi_all - lo_all
        ax.set_xlim(min(lo_all, 0) - 0.4 * span, max(hi_all, 0) + 0.4 * span)
        ax.set_xlabel("paired gain, 95% bootstrap interval (x axis includes 0)")
    fig.suptitle(f"Dev (decision block) against the old test blocks.  Grey band: {TL_LABEL}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def season_means(d: dict) -> pd.DataFrame:
    rows = []
    for nm, df in (("final", d["fin"]), ("latent_111", d["lat"]), ("grid", d["grid"])):
        for s, g in df.groupby("season"):
            rows.append({"season": int(s), "model": nm, "rho": g.spearman.mean(), "n": len(g)})
    sc = d["sc"]
    for nm in ("final", "latent_111", "grid"):
        for s, g in sc[sc.model == nm].groupby("season"):
            rows.append({"season": int(s), "model": nm, "rho": g.spearman.mean(), "n": len(g)})
    return pd.DataFrame(rows)


def fig_per_season(path: Path, sm: pd.DataFrame) -> None:
    seasons = [2021, 2022, 2023, 2024, 2025, 2026]
    models = [("final", "tab:red"), ("latent_111", "tab:blue"), ("grid", "tab:gray")]
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.26
    for j, (nm, col) in enumerate(models):
        vals = [float(sm[(sm.season == s) & (sm.model == nm)].rho.iloc[0]) for s in seasons]
        bars = ax.bar(np.arange(len(seasons)) + (j - 1) * w, vals, w, label=nm, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.axvspan(2.5, 5.5, color="0.92", zorder=0)
    ns = [int(sm[(sm.season == s) & (sm.model == 'final')].n.iloc[0]) for s in seasons]
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(seasons, ns)])
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("mean per-race Spearman rho (axis starts at 0)")
    ax.set_title(f"Per-season rho. 2021-23 = dev (decision block). Grey band 2024-26 R20: {TL_LABEL}", fontsize=10)
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


SHORT_REF = {
    "latent_111+M07 (S3)": "latent_111+M07 (S3)",
    "plan2_H1_latent_111": "plan-2 H1 latent_111",
    "latent_111_S3refit": "latent_111 + S3 re-fit",
    "plan-2 S3 on M06 H3 score (stored run)": "plan-2 S3 (stored H3 run)",
}


def short_ref(r: str) -> str:
    if r in SHORT_REF:
        return SHORT_REF[r]
    if r.startswith("leave-one-out"):
        return "full minus " + r.split("full minus ")[-1]
    return r


def fig_forest(path: Path) -> None:
    a = ATT.reset_index(drop=True)
    n = len(a)
    labels = [f"{c} | vs {short_ref(r)}" + ("" if d == "yes" else " (not declared)") for c, r, d in zip(a.component, a.reference, a.declared)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.27 * n + 2.2), sharey=True)
    for ax, (m, title) in zip(axes, (("spearman", "Spearman rho gain"), ("exact", "Exact-position rate gain"))):
        for i, r in a.iterrows():
            g, lo, hi = r[f"gain_{m}"], r[f"lo_{m}"], r[f"hi_{m}"]
            col = "tab:red" if excl0(lo, hi) else "0.35"
            ax.errorbar(g, i, xerr=[[g - lo], [hi - g]], fmt="o" if r.declared == "yes" else "s", mfc=col if r.declared == "yes" else "white",
                        color=col, capsize=2.5, ms=4)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{title} (dev, 66 races unless n differs; red = interval excludes 0)", fontsize=9)
        ax.set_xlabel("paired gain, 95% bootstrap interval; x axis includes 0")
        ax.grid(axis="x", lw=0.3)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_ylim(n - 0.5, -0.5)
    fig.suptitle("Attribution (results/plan3/attribution.csv): every row, dev block only. Reference differs by row; rows against grid or S3 are not on the same footing as rows against latent_111", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path)
    plt.close(fig)


def latest_cars() -> pd.DataFrame:
    c = pd.read_csv(R3 / "cars_final_fit.csv")
    out = []
    for lin, g in c[c.model.isin(["quali_in", "h0"])].groupby("id"):
        ls = int(g.season.max())
        row = {"lineage": lin, "last_season": ls}
        for m in ("h0", "quali_in"):
            r = g[(g.model == m) & (g.season == ls)].iloc[0]
            row[f"{m}_est"], row[f"{m}_lo"], row[f"{m}_hi"], row[f"{m}_se"] = r.est, r.lo, r.hi, r.se
            row["n_season"] = int(r.n_season)
        out.append(row)
    return pd.DataFrame(out).sort_values("h0_est", ascending=False).reset_index(drop=True)


def fig_cars(path: Path, cars: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    n = len(cars)
    for ax, (m, t) in zip(axes, (("h0", "history-only model (H0)"), ("quali_in", "qualifying-in model"))):
        for i, r in cars.iterrows():
            ax.errorbar(r[f"{m}_est"], i, xerr=[[r[f"{m}_est"] - r[f"{m}_lo"]], [r[f"{m}_hi"] - r[f"{m}_est"]]], fmt="o", color="k", capsize=3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"Car strength, {t}", fontsize=10)
        ax.set_xlabel("strength with 95% Laplace interval (in-sample description)")
        ax.grid(axis="x", lw=0.3)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([f"{r.lineage} ({r.last_season})" for r in cars.itertuples()])
    axes[0].set_ylim(n - 0.5, -0.5)
    fig.suptitle("All 13 car lineages, final fit, last season in the data (cars_final_fit.csv). caterham (2014) and manor (2016) are not comparable to the 2026 cars", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- report text
def holdout_readings_text() -> str:
    """At n < 20 say so; otherwise list each verdict file as written by score.py."""
    rd_ = HS.get("readings") or {}
    if not rd_:
        return "`readings` empty (" + HS["verdict_state"] + ")"
    parts = []
    for t, rel in sorted(rd_.items()):
        f = ROOT / rel
        if f.exists():
            payload = json.loads(f.read_text())
            tests = payload.get("tests", payload)
            parts.append(f"level {t} (`{rel}`): " + json.dumps(tests, sort_keys=True)[:1500])
        else:
            parts.append(f"level {t}: file `{rel}` listed but not found")
    return "readings: " + "; ".join(parts)


def build() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rd, data = rederive()
    sm = season_means(data)
    fig_dev_vs_third(FIG / "r16_dev_vs_third_look.png")
    fig_per_season(FIG / "r16_per_season_rho.png", sm)
    fig_forest(FIG / "r16_attribution_forest.png")
    cars = latest_cars()
    fig_cars(FIG / "r16_all_lineages_car_strength.png", cars)

    A = att
    L = []
    w = L.append
    w("# Plan 3 report (R16): race pace, car development and a clean test")
    w("")
    w(f"Generated by `results/plan3/make_report.py` at repository HEAD {git_head()} (plus uncommitted report files if any). Most numbers are read from files under "
      f"`results/` or `methodologies/*/results/` (path next to the claim) or recomputed in the script. A few are typed values taken from `experiments/LOG.md` or method READMEs, each cited where used: "
      f"Q5 (-0.0039 [-0.0174, +0.0095] and +0.010, LOG R10), Q8 (0.0005-0.0023, LOG R3), Q16 (0.132 and the +0.007 resolution, LOG R8), Q13 (11.8% and 16.4%, M17 README), and the Antonelli finding (A5 README). "
      f"Nothing was re-run: no model fitted, nothing fetched.")
    w("")
    w("## 0. Summary, failures first")
    w("")
    t1d = att_rows("P3_final", "latent_111")
    t3 = tlrow("pooled test_stable+test_shift", "T1:final_vs_latent_111", "spearman")
    w(f"- **There is no verdict on the plan-3 model.** The fresh holdout has n = {HS['n']} races (`holdout_status.json`: `verdict_state` = \"{HS['verdict_state']}\"). "
      f"Nothing can be called confirmed until 20 fresh races exist.")
    w(f"- **No evidence that the final plan-3 model beats `latent_111`.** Dev T1 (final - latent_111, rho) = {iv(t1d.gain_spearman, t1d.lo_spearman, t1d.hi_spearman)}, the lower bound is below 0 "
      f"(`attribution.csv`, component P3_final, reference latent_111). On the already-seen 2024-26 R20 races the contrast is {iv(t3.gain, t3.lo, t3.hi)}, n = {int(t3.n)} "
      f"(`third_look.csv`, {TL_LABEL}). The dev point estimate does not reappear; the dev interval was already not a clean pass and was selected on the same 66 races.")
    ex = att_rows("P3_final", "grid")
    w(f"- **Grid order is not beaten on exact-position rate by any model, on dev or on the third look (no exact-vs-grid interval lies above 0 in either file).** Final minus grid, dev: {iv(ex.gain_exact, ex.lo_exact, ex.hi_exact)} (`attribution.csv`, P3_final vs grid).")
    w("- **Q1 gate failed as declared; Q2 is a null.** The one dev gain that looked like race pace (M09) is credited by the pre-stated control to car form (section Q2).")
    w("- **Not tested:** Q3 (no verdict until 20 fresh races), Q7 (no permitted odds source), Q11 (rookie prior cut), Q17 (neural-network control not run), and parts of Q10 and Q16 (see sections).")
    w("- **Nothing in this report is a finding in the sense of PLAN3 §5**: a secondary result counts only if it also holds on the fresh holdout, and that holdout is empty.")
    w("")
    w("### Verdict table")
    w("")
    vt = [
        ("Q1", "Is race pace relative to qualifying pace persistent?", "Not confirmed (gate failed as declared)"),
        ("Q2", "Does an earlier-race pace offset improve the order?", "Not confirmed (null)"),
        ("Q3", "Do declared upgrades predict car strength?", "Not tested (no verdict until 20 fresh races)"),
        ("Q4", "Does a likelihood that down-weights the back of the field fit better?", "Not confirmed"),
        ("Q5", "Is two-stage (qualifying first) better before qualifying?", "Not confirmed"),
        ("Q6", "Does averaging latent and LightGBM help?", "Not confirmed"),
        ("Q7", "How do we compare with the betting market?", "Not tested (M13 dropped)"),
        ("Q8", "Do hyper-parameters tuned on 2021-23 go stale?", "Not confirmed (no detectable staleness; low power)"),
        ("Q9", "Are strengths believable, is every team shown?", "Every team shown: done. Driver strengths believable: not confirmed"),
        ("Q10", "Do car traits predict where a car is strong?", "Supplier pooling not confirmed; trait x circuit and learned factors not tested"),
        ("Q11", "Does a junior record help rookies?", "Not tested (M15 cut)"),
        ("Q12", "Do better noise models improve simulator probabilities?", "Not confirmed"),
        ("Q13", "Does exact-hit decoding beat score sorting?", "Not confirmed"),
        ("Q14", "Does a pace threshold for passing help?", "Not confirmed (fitted threshold 0, component inactive)"),
        ("Q15", "How fast does accuracy grow in the race?", "Described on dev; the exact-rate target is not confirmed"),
        ("Q16", "Do lap-level driver traits and pit-crew speed add anything?", "Consistency not confirmed; the other traits not tested (failed persistence); crew time not run"),
        ("Q17", "Does a small neural network do better?", "Not tested (needs a rule change; not run)"),
    ]
    w(md_table(pd.DataFrame(vt, columns=["Q", "Question", "Verdict"])))
    w("")
    w("Reading rule used throughout: an interval that includes 0 is a null (\"no measurable difference\"), not a small gain. The plan's decision rule (PLAN3 §5) is a paired per-race 95% bootstrap interval on dev excluding 0; "
      "even then a secondary result is only \"dev only\" until the fresh holdout confirms it.")
    w("")

    # ------------------------------------------------------------ Q1
    w("## Q1 - Is a car's race pace relative to its qualifying pace persistent? (A3 gate)")
    s = json.loads((R3 / "a3_summary.json").read_text())
    g = s["gate"]
    pers = pd.read_csv(R3 / "a3_persistence.csv")
    ps = pers[(pers.level == "car") & (pers.halflife == 50) & (pers.scope == "dev")].set_index("definition")
    sens = pd.read_csv(R3 / "a3_reference_sensitivity.csv")
    q1 = sens[(sens.variant == "rq vs q1_gap (missing Q1 = NaN)")].set_index("definition")
    q1p = sens[(sens.variant.str.startswith("Q1-based offset, partial given q_best, q1 and the decayed"))].set_index("definition")
    w(f"**Verdict: not confirmed.** The gate required a car-level persistence correlation of at least 0.2 on dev; it was {g['med']['car_persistence']:.3f} "
      f"[{ps.loc['med','lo']:.3f}, {ps.loc['med','hi']:.3f}] for the declared median-lap definition D1 and {g['corr']['car_persistence']:.3f} "
      f"[{ps.loc['corr','lo']:.3f}, {ps.loc['corr','hi']:.3f}] for the fuel/stint-corrected D2 (`a3_summary.json` gate; `a3_persistence.csv`, car, half-life 50, dev). Both fail.")
    w(f"The other half of the gate passed: the extended oracle gap on finishers is {g['med']['oracle_gap']:.3f} rho (`a3_oracle_gap.csv`, gap_med: "
      f"{pd.read_csv(R3/'a3_oracle_gap.csv').set_index('quantity').loc['gap_med','lo']:.3f} to {pd.read_csv(R3/'a3_oracle_gap.csv').set_index('quantity').loc['gap_med','hi']:.3f}), threshold 0.03. "
      "The model is below an oracle that knows each driver's race pace, but the offset relative to the best qualifying time does not persist enough to learn it.")
    w("**The null depends on a reference the plan did not fix (post hoc, no decision weight).** "
      f"Against the Q1 gap instead of the best qualifying gap, car persistence is {q1.loc['D1','r']:.3f} [{q1.loc['D1','lo']:.3f}, {q1.loc['D1','hi']:.3f}] (D1) and "
      f"{q1.loc['D2','r']:.3f} [{q1.loc['D2','lo']:.3f}, {q1.loc['D2','hi']:.3f}] (D2) (`a3_reference_sensitivity.csv`, variant \"rq vs q1_gap (missing Q1 = NaN)\"). "
      f"With the car's own earlier race-pace gap controlled, the partial correlation falls to {q1p.loc['D1','r']:.3f} [{q1p.loc['D1','lo']:.3f}, {q1p.loc['D1','hi']:.3f}] (D1) and "
      f"{q1p.loc['D2','r']:.3f} [{q1p.loc['D2','lo']:.3f}, {q1p.loc['D2','hi']:.3f}] (D2). The persistent part is the car's race pace itself, not its offset from qualifying. "
      "These variants were chosen after the gate failed, so the gate outcome stands as declared. Source: `methodologies/A3_pace_persistence/README.md`; `experiments/LOG.md` R4 entry.")
    w("")

    # ------------------------------------------------------------ Q2
    w("## Q2 - Does a race-pace offset learned from earlier races improve the order? (M09, primary hypothesis)")
    w("**Verdict: not confirmed.** The gate (Q1) failed, so M09 was not built as declared. Under PLAN3 Amendment 3 item 6 it was run afterwards as an exploratory component on a definition fixed beforehand "
      "(car race-pace history, then the Q1 offset); that was found after the gate failed, so a dev pass would be weaker evidence than a declared one.")
    h = att_rows("M09_car_pace_hist", "latent_111")
    hg = att_rows("M09_car_pace_hist", "gbm_100")
    q = att_rows("M09_car_q1_offset", "latent_111")
    ctrl = pd.read_csv(MET / "M09_race_pace" / "results" / "control_paired_gains_dev.csv")
    cc = ctrl[(ctrl.metric == "spearman")].set_index("comparison")
    cx = ctrl[(ctrl.metric == "exact")].set_index("comparison")
    w(f"- Dev, latent_111 + car pace history, rho: {iv(h.gain_spearman, h.lo_spearman, h.hi_spearman)}; exact rate {iv(h.gain_exact, h.lo_exact, h.hi_exact)} "
      f"(`attribution.csv`, M09_car_pace_hist vs latent_111). LightGBM, rho: {iv(hg.gain_spearman, hg.lo_spearman, hg.hi_spearman)} (vs gbm_100): null.")
    w(f"- The Q1-offset rung's bundle gain is {iv(q.gain_spearman, q.lo_spearman, q.hi_spearman)}; its own increment over the history rung is not in `attribution.csv` as a separate row. "
      f"The stored verdict text \"passes alone\" on rows M09_car_q1_offset and M09_car_q1_offset_alone refers to the bundle or to a non-declared off-ladder version and must not be read as the offset passing (README of M09 corrects it; the file is append-only).")
    w(f"- **Car-form control** (post hoc and diagnostic, no decision weight, but its decision rule was written to `control_prestatement.md` before its runs; `methodologies/M09_race_pace/results/control_paired_gains_dev.csv`): a column built the same way from the car's centred finishing position gives "
      f"rho {iv(cc.loc['C1_form_vs_ref','gain'], cc.loc['C1_form_vs_ref','lo'], cc.loc['C1_form_vs_ref','hi'])} against the reference; the race-pace column adds "
      f"{iv(cc.loc['hist_vs_C1','gain'], cc.loc['hist_vs_C1','lo'], cc.loc['hist_vs_C1','hi'])} beyond form alone (interval includes 0). "
      f"Form alone lowers the exact rate, {iv(cx.loc['C1_form_vs_ref','gain'], cx.loc['C1_form_vs_ref','lo'], cx.loc['C1_form_vs_ref','hi'])}. By the rule written before the control, the gain is credited to car form, not race pace.")
    w(f"- Holdout evidence so far: third look T3 (final - reverse_only, rho, 2024-26 R20 pooled) {tl('pooled test_stable+test_shift','T3:final_vs_reverse_only','spearman')} ({TL_LABEL}). "
      "The fresh holdout T3 decides what, if anything, M09 contributes.")
    w("")

    # ------------------------------------------------------------ Q3
    w("## Q3 - Do declared car upgrades predict changes in car strength? (M10)")
    w("**Verdict: not tested. M10 has no verdict until 20 fresh holdout races.** There is no dev data (FIA car-presentation documents start at 2024 round 2; `methodologies/M10_upgrades/README.md`, \"Coverage\"). "
      "One rung (total declared-component count, `lam_x` fixed at 1.0) is built and pre-registered; the verdict test is mean rho at H1, with H2 a negative control, and nothing is scored. "
      "Expected gain per the amendment: about 0 at H2, at most 0.01 rho at H0/H1; failure is an interval including 0. PLAN3 §M10's \"covers 2022 onward\" is contradicted by the data and was not edited (M10 README).")
    w("")

    # ------------------------------------------------------------ Q4
    w("## Q4 - Does a likelihood that ignores or down-weights the noisy back of the field fit better? (M11)")
    fin_ = att_rows("M11_finishers", "latent_111")
    top_ = att_rows("M11_topk10", "latent_111")
    rev_ = att_rows("M11_reverse", "latent_111")
    w("**Verdict: not confirmed** for the two variants that match the question (finishers-only, top-10). The third variant, reverse (built from last place upward), is a different mechanism and is carried as a dev-only candidate.")
    w("")
    mt = pd.DataFrame([
        ["finishers-only", A("M11_finishers", "latent_111", "spearman"), A("M11_finishers", "latent_111", "mae", 3), A("M11_finishers", "latent_111", "exact")],
        ["top-10", A("M11_topk10", "latent_111", "spearman"), A("M11_topk10", "latent_111", "mae", 3), A("M11_topk10", "latent_111", "exact")],
        ["reverse", A("M11_reverse", "latent_111", "spearman"), A("M11_reverse", "latent_111", "mae", 3), A("M11_reverse", "latent_111", "exact")],
    ], columns=["variant (vs latent_111, dev, 66 races)", "rho gain", "MAE gain (ref - cand)", "exact-rate gain"])
    w(md_table(mt))
    w("")
    w(f"Source: `attribution.csv`, rows M11_*. For reverse, the MAE and exact intervals exclude 0 while rho does not; LOG R5 records that the overall MAE and exact-rate gains do not survive a Holm correction over the 36 intervals of that package. "
      f"Reverse is inside the final model and holdout test T2 (MAE) tests it; third look T2 pooled: {tl('pooled test_stable+test_shift','T2:final_vs_final_minus_reverse','mae')} ({TL_LABEL}). "
      f"Interaction finishers-only x retirement model (M07): RPS {A('M11_finishers_x_M07','latent_111+M07 (S3)','rps')}, null.")
    w("")

    # ------------------------------------------------------------ Q5
    w("## Q5 - Before qualifying, is \"predict qualifying, then the race\" better than predicting the race directly? (M12)")
    w(f"**Verdict: not confirmed.** Two-stage minus plan-2 H1, rho {A('M12_two_stage','plan2_H1_latent_111','spearman')}, MAE {A('M12_two_stage','plan2_H1_latent_111','mae',3)}, "
      f"n = {int(att_rows('M12_two_stage','plan2_H1_latent_111').n_races)} races of 2023 (practice data exist only from 2023) (`attribution.csv`, M12_two_stage). Both intervals include 0. "
      "LOG R10 records that the point estimate rests on one race (without race 185 it is +0.010) and that the 44 races of 2021-22 (stage 1 from history only) are also null (rho -0.0039 [-0.0174, +0.0095]); that row is not in `attribution.csv` and is not re-derived here.")
    w("")

    # ------------------------------------------------------------ Q6
    w("## Q6 - Does averaging the latent model and LightGBM help? (M05)")
    w(f"**Verdict: not confirmed.** Rank average vs latent_111, rho {A('M05_rank_avg','latent_111','spearman')}; vs gbm_100 {A('M05_rank_avg','gbm_100','spearman')}; "
      f"monotone LightGBM vs gbm_100 {A('M05_gbm_monotone','gbm_100','spearman')} (`attribution.csv`, M05_*). The M05 README records a points log-loss gain over latent_111 that is not there against gbm_100, so the ensemble is not better than LightGBM alone.")
    w("")

    # ------------------------------------------------------------ Q7
    w("## Q7 - How do we compare with the betting market? (M13)")
    w("**Verdict: not tested.** M13 was dropped on 2026-11-18: no source with clearly permitted automated use (Polymarket terms ambiguous, Betfair keys for own betting, Kalshi unreachable, The Odds API has no F1). "
      "Nothing was collected or stored (`methodologies/M13_market_benchmark/README.md`; LOG R17 entry).")
    w("")

    # ------------------------------------------------------------ Q8
    w("## Q8 - Do hyper-parameters tuned once on 2021-23 go stale, and how does accuracy move season by season? (A4)")
    a4 = json.loads((R3 / "a4_summary.json").read_text())["retune"]
    gq = pd.read_csv(R3 / "a4_gain_over_qualifying.csv")

    def gqr(model: str, block: str, bootstrap: str, stat: str = "gain") -> str:
        r = gq[(gq.model == model) & (gq.metric == "spearman") & (gq.block == block) & (gq.bootstrap == bootstrap) & (gq.statistic == stat)].iloc[0]
        return iv(r.estimate, r.ci_lo, r.ci_hi, 3) + f", n={int(r.n_races)}"

    w("**Verdict: not confirmed.** Re-tuning the hyper-parameters on the three seasons before each season does not beat the fixed 2021-23 settings on dev: "
      f"latent_111 rho {iv(a4['latent_111']['spearman']['delta'], a4['latent_111']['spearman']['ci_lo'], a4['latent_111']['spearman']['ci_hi'])}, "
      f"gbm_100 rho {iv(a4['gbm_100']['spearman']['delta'], a4['gbm_100']['spearman']['ci_lo'], a4['gbm_100']['spearman']['ci_hi'])} (delta = rolling - fixed; `a4_summary.json`, retune). "
      f"Decision recorded there: \"{a4['latent_111']['decision']}\". One of the six paired tests excludes 0 and it points against rolling: latent MAE worse by "
      f"{a4['latent_111']['mae']['delta']:+.3f} [{a4['latent_111']['mae']['ci_lo']:+.3f}, {a4['latent_111']['mae']['ci_hi']:+.3f}], all from 2022 (README of A4). "
      "Low power: the two tuned settings differed in one season (latent) or two (LightGBM), and the settings that won their tuning windows did so by 0.0005-0.0023 rho (LOG R3 entry); the test mostly measures noise in picking among near-tied settings, and it cannot speak to staleness under a regime change larger than 2022.")
    w("")
    w("Season by season (every race predicted from earlier races only; seasons before 2021 are labelled \"not clean\" because the hyper-parameters were tuned on later data; model minus qualifying order, rho, paired by race; `a4_gain_over_qualifying.csv`):")
    w("")
    rows = []
    for lab, blk in (("2015-20 (not clean)", "2015-20"), ("dev 2021-23", "dev 2021-23"), ("2024-26 (third look)", "2024-26"), ("all seasons", "all seasons"), ("all seasons except 2017 and 2023", "all except 2017, 2023")):
        bs = "season cluster" if blk == "all except 2017, 2023" else "race"
        rows.append([lab, gqr("latent_111", blk, bs), gqr("gbm_100", blk, bs)])
    w(md_table(pd.DataFrame(rows, columns=["block", "latent_111 - qualifying", "gbm_100 - qualifying"])))
    w("")
    slope = gq[(gq.model == "latent_111") & (gq.statistic == "slope per season")].iloc[0]
    w(f"Reading: the gain over qualifying order is small everywhere, largest on the block the settings were tuned on, and without 2017 and 2023 the pooled gain is not distinguishable from 0 (row \"all except 2017, 2023\"). "
      f"There is no trend over seasons (latent slope {slope.estimate:+.4f} per season [{slope.ci_lo:+.4f}, {slope.ci_hi:+.4f}], p = {slope.p_value:.2f}). The first-half/second-half difference in the README was found after looking and carries no weight. "
      f"2024-26 is a third look ({TL_LABEL}). Season table: `season_table.csv`; figures `figures/a4_season_table.png`, `figures/a4_gain_over_qualifying.png`.")
    w("")

    # ------------------------------------------------------------ Q9
    w("## Q9 - Are the estimated driver and car strengths believable, and is every team shown? (A5)")
    a5 = json.loads((R3 / "a5_summary.json").read_text())
    tm = {d["model"]: d for d in a5["teammate_oos_pooled"]}
    pc = {d["model"]: d for d in a5["pooled_correlations"]}
    rp = {d["model"]: d for d in a5["resolved_pairs"]}
    ident = a5["identification"]
    acc = pd.read_csv(R3 / "acceptance_summary.csv")
    acc_s = acc[(acc.model == "quali_in") & (acc.population.str.startswith("season-end"))].iloc[0]
    acc_f = acc[(acc.model == "quali_in") & (acc.population.str.startswith("final fit"))].iloc[0]
    w("**Verdict, part 1 (is every team shown): done.** All 13 car lineages are in every car table and figure of the audit (`a5_summary.json`: n_lineages_plotted = "
      f"{a5['n_lineages_plotted']}); the plan-1 \"largest 8\" filter was a plotting bug that dropped McLaren, Haas, Cadillac, Manor and Caterham. The table and figure below show all 13.")
    w("**Verdict, part 2 (are the driver strengths believable): not confirmed.** Evidence:")
    tmo = pd.read_csv(R3 / "teammate_oos.csv")
    nneg = int(((tmo.model == "quali_in") & (tmo.season != "pooled") & (tmo.r < 0)).sum())
    w(f"- Out-of-sample teammate check (strength at the end of season S-1 against the teammate finishing gap in S, {tm['quali_in']['n_pairs']} pairs): qualifying-in model r = {tm['quali_in']['r']:+.3f} [{tm['quali_in']['lo']:+.3f}, {tm['quali_in']['hi']:+.3f}], "
      f"history-only (H0) r = {tm['h0']['r']:+.3f} [{tm['h0']['lo']:+.3f}, {tm['h0']['hi']:+.3f}] (`a5_summary.json`, teammate_oos_pooled; per season `teammate_oos.csv`: {nneg} of 12 seasons are negative for the qualifying-in model and almost every single-season interval spans 0, 5-9 pairs each). "
      f"This is well below the in-sample {pc['quali_in']['r_teammate']:+.3f} and {pc['h0']['r_teammate']:+.3f}; in-sample car correlation {pc['quali_in']['r_car']:.3f} (`pooled_correlations`).")
    w(f"- Driver pairs that can be separated: {rp['quali_in']['share_resolved']*100:.1f}% of {rp['quali_in']['n_pairs']} pairs among 26 drivers in the qualifying-in model, {rp['h0']['share_resolved']*100:.1f}% in H0 (`a5_summary.json`, resolved_pairs; `resolved_pairs.csv`). "
      f"With qualifying in the model the driver term `a` is a race-day gain beyond grid slot, not overall skill; the H0 table is the overall-skill one.")
    w(f"- Identification: {ident['n_weak']} of {ident['n_drivers']} drivers are weakly identified (fewer than two teammates or disconnected); {ident['n_lt10_starts']} have fewer than 10 starts (`a5_summary.json`, identification; `identification.csv`).")
    w(f"- One-race drivers at the extremes: the shrinkage cause is the driver penalty (lam 32, implied prior sd {a5['shrinkage']['prior_sd_implied_by_lam']:.3f}) being much wider than the spread of established drivers, not a bug. "
      f"In the unseparated final-fit ranking of the qualifying-in model, {int(acc_f.n_lowsample_in_top_or_bottom3)} of {int(acc_f.n_slots)} extreme slots are one-race drivers "
      f"({', '.join(d['driver'] for d in a5['acceptance_final_fit_lowsample_extremes'])}); across season-end snapshots {int(acc_s.n_lowsample_in_top_or_bottom3)} of {int(acc_s.n_slots)} slots (`acceptance_summary.csv`). "
      f"The acceptance check passes only because low-start drivers are shown in a separate panel (main panel: {a5['acceptance_main_panel_lowsample_in_extremes']} low-sample drivers in the extremes). The stronger penalty (lam_a) variant that removes them was not adopted for prediction (README of A5).")
    cr0 = pd.read_csv(R3 / "champion_ranks.csv")
    nor = cr0[(cr0.model == "quali_in") & (cr0.season == 2025)].iloc[0]
    ant = cr0[(cr0.model == "h0") & (cr0.season == 2026)].iloc[0]
    w(f"- Face validity (A5 README, near line 170, typed from there): Antonelli 2026 is a face-validity failure of the static driver term: 44 starts, H0 rank {int(ant.champ_rank)}/{int(ant.n_drivers)} (`champion_ranks.csv`) and below his teammate (rank {int(ant.mate_rank)}) although he outscored him in 2026 points; the cause given is the static `a` with a 50-race half-life dominated by 2025. The 2026 row is the points leader after R20, season not finished, so it is a check of the model, not of a champion. "
      f"Norris 2025 is ranked {int(nor.champ_rank)}/{int(nor.n_drivers)} in the qualifying-in model (`champion_ranks.csv`). Whether the 2026 order is right is unverified here.")
    w("")
    w("### All 13 car lineages (final fit, last season in the data; `cars_final_fit.csv`)")
    w("")
    ct = pd.DataFrame({
        "lineage": cars.lineage, "last season": cars.last_season,
        "H0 strength [95% interval]": [iv(r.h0_est, r.h0_lo, r.h0_hi, 3) for r in cars.itertuples()],
        "qualifying-in strength [95% interval]": [iv(r.quali_in_est, r.quali_in_lo, r.quali_in_hi, 3) for r in cars.itertuples()],
        "rows that season": cars.n_season,
    })
    w(md_table(ct))
    w("")
    w("Intervals are conditional on sigma = 0.25 and the decay-weighted likelihood; coverage was not verified; all strengths are in-sample descriptions (A5 README). Figure: `figures/r16_all_lineages_car_strength.png`. caterham (2014) and manor_lineage (2016) are shown at their last season and are not comparable to the 2026 cars; all values are in-sample.")
    w("")
    w("### Drivers active in 2026, H0 (overall-skill) fit, all of them (`drivers_final_fit.csv`)")
    w("")
    dr = pd.read_csv(R3 / "drivers_final_fit.csv")
    d0 = dr[(dr.model == "h0") & (dr.last_season == 2026)].sort_values("est", ascending=False)
    dt = pd.DataFrame({"driver": d0.id, "starts (all time in data)": d0.n_total, "H0 skill [95% interval]": [iv(r.est, r.lo, r.hi, 3) for r in d0.itertuples()],
                       "flag": ["fewer than 10 starts" if r.n_total < 10 else "" for r in d0.itertuples()]})
    w(md_table(dt))
    w("")
    w("### Champion (2026: points leader after R20, season not finished) and the champion's teammate, season-end rank (`champion_ranks.csv`)")
    w("")
    cr = pd.read_csv(R3 / "champion_ranks.csv")
    piv = []
    for s in sorted(cr.season.unique()):
        row = {"season": f"{s} (points leader after R20, season not finished)" if s == 2026 else s}
        for m, nm in (("quali_in", "qual-in"), ("h0", "H0")):
            r = cr[(cr.model == m) & (cr.season == s)]
            if len(r):
                r = r.iloc[0]
                row[f"{nm}: champion rank"] = f"{r.champ_id} {int(r.champ_rank)}/{int(r.n_drivers)}"
                row[f"{nm}: teammate rank"] = f"{r.mate_id} {int(r.mate_rank)}/{int(r.n_drivers)}"
        piv.append(row)
    w(md_table(pd.DataFrame(piv)))
    w("")

    # ------------------------------------------------------------ Q10
    w("## Q10 - Do measurable car traits predict where a car is strong? (M14)")
    sp = att_rows("M14_supplier_pool", "latent_111")
    spp = att_rows("M14_supplier_pool_carry_total_posthoc", "latent_111")
    w("**Verdict: not confirmed for engine-supplier pooling (the only part run). Trait x circuit and learned car x circuit factors (R11 parts 2-3): not tested; they were cut** "
      "under the PLAN3 §6 cut order by a main-session decision on 2026-11-18 (`experiments/LOG.md`, R17 entry; the M14 README status line was written earlier and still says they are not cut).")
    w(f"- Supplier pooling, dev: rho {iv(sp.gain_spearman, sp.lo_spearman, sp.hi_spearman)}, exact {iv(sp.gain_exact, sp.lo_exact, sp.hi_exact)} (`attribution.csv`, M14_supplier_pool). The fitted penalty sits at the grid edge, i.e. the data switch the supplier term off. "
      "Dev contains one chassis change (2022) and one engine rebadge, so little could be shown; this was stated in advance.")
    w(f"- Post-hoc variant (form chosen after the declared result; confounded with an own-part ridge): rho {iv(spp.gain_spearman, spp.lo_spearman, spp.hi_spearman)}, worse. It carries no decision weight and is not evidence against pooling.")
    w("- The motivating case (new power-unit rules in 2026) lies in the already-seen block and was not used. The component is not in the final model.")
    w("")

    # ------------------------------------------------------------ Q11
    w("## Q11 - Does a junior-series record give a better starting estimate for a rookie? (M15)")
    w("**Verdict: not tested.** M15 was cut in PLAN3 Amendment 3 item 4: 8 debuts in dev would give an uninformative interval, and mid-season debuts need care to avoid leakage. No folder, run or file exists for it.")
    w("")

    # ------------------------------------------------------------ Q12
    w("## Q12 - Do better noise models improve the simulator's probabilities? (M16)")
    w("**Verdict: not confirmed.** RPS gain over S3 (dev, 66 races; positive = lower RPS):")
    w("")
    rows = []
    for lab, comp in (("S4 Henery stage multipliers", "M16_S4"), ("S5 team shock / normal noise", "M16_S5"), ("S6 Laplace interval width", "M16_S6"), ("S7 AR latent, Spearman-picked (superseded)", "M16_S7"), ("S7 AR latent, RPS-picked (declared)", "M16_S7_rps")):
        rows.append([lab, A(comp, "S3_D0", "rps", 5)])
    w(md_table(pd.DataFrame(rows, columns=["step", "RPS gain vs S3 [95% interval]"])))
    w("")
    w(f"Source: `attribution.csv`, M16_* vs S3_D0. Every interval includes 0. The declared interaction S6 x D2 on exact rate is also null ({A('M16_S6xD2','S6_gain_D0','exact')}). "
      "The M16 README records that the fitted settings sit at or near neutral (the S5 tuning switches every extension off; S7 phi at the grid edge) and that the plan's hoped-for 0.0012-0.0036 RPS gain was not reached. "
      "Secondary intervals that exclude 0 in that package (top-10 and both-cars log-loss, driven by 2023) are not passes. Separately, re-fitting S3 on the final model's scores (not a variant search) gives an RPS gain over the same re-fit of latent_111 "
      f"of {A('P3_final_S3refit','latent_111_S3refit','rps',5)} on dev, in-sample for both fits; on the third look it is {tl('pooled test_stable+test_shift','S3:final_vs_latent_111','rps',5)} ({TL_LABEL}).")
    w("")

    # ------------------------------------------------------------ Q13
    w("## Q13 - Does choosing the predicted order to maximise expected exact hits beat sorting by score? (M17)")
    w(f"**Verdict: not confirmed.** Exact-position rate, dev, 66 races: D2 minus D0 {A('M17_D2','D0','exact')}; D2 minus grid order {A('M17_D2','grid','exact')}; D0 minus grid {A('M17_D0','grid','exact')} (`attribution.csv`, M17_*). "
      f"The pre-stated failure criterion (D2 not above grid) is met. D2 costs ordering quality: rho {A('M17_D2','D0','spearman')}, MAE {A('M17_D2','D0','mae',3)}. "
      "The M17 README locates the cause in the simulator's matrix (11.8% mass on \"finishes in the grid slot\" against 16.4% realised), not in the decoder; the earlier plan statement that sorting is the wrong decoder was corrected there. Grid-slot calibration failed for every later matrix (LOG R12 entry), so D2 was run only for the declared S6 x D2.")
    w("")

    # ------------------------------------------------------------ Q14
    w("## Q14 - Does a pace threshold for passing improve on a linear pace effect? (M18)")
    w(f"**Verdict: not confirmed.** Because Q1's gate failed there is no pace offset, so the dead-band was applied to the model's strength difference inside the plan-2 simulator. The d fitted by RPS on dev is 0, i.e. the component is inactive; the best positive d tried (0.03) gives, against S3 D0, "
      f"rho {A('M18_deadband','S3_D0','spearman')}, exact {A('M18_deadband','S3_D0','exact')}, RPS {A('M18_deadband','S3_D0','rps',5)}: no measurable difference (`attribution.csv`, M18_deadband). "
      f"Against grid order on exact rate: {A('M18_deadband','grid','exact')}. The M18 README gives the mechanism: a dead-band can only pull the model toward grid order, so it cannot beat grid order on exact rate under any objective. "
      "LOG R7 records that the RPS objective could not be verified as chosen before running, and that plan 2 had already fitted the linear grid-attachment term to 0 on the same simulator.")
    w("")

    # ------------------------------------------------------------ Q15
    w("## Q15 - How fast does accuracy grow once the race is running? (M19, reported separately from pre-race results)")
    pg = pd.read_csv(MET / "M19_inrace" / "results" / "paired_gains_dev.csv")
    pg = pg[(pg.subset == "all_66") & (pg.reference == "running_order") & (pg.model.isin(["latent_s2", "gbm_s2"]))]
    rows = []
    for hz, lab in (("H4", "after lap 1"), ("H5", "25% of distance"), ("H6", "50% of distance")):
        for md in ("latent_s2", "gbm_s2"):
            r = pg[(pg.horizon == hz) & (pg.model == md)]
            rr = r[r.metric == "spearman"].iloc[0]
            re_ = r[r.metric == "exact"].iloc[0]
            rows.append([f"{hz} ({lab})", md.replace("_s2", ""), f"{re_.ref_mean:.3f}", f"{re_.cand_mean:.3f}", iv(rr.gain, rr.lo, rr.hi, 3), iv(re_.gain, re_.lo, re_.hi, 3)])
    w("**Verdict: described on dev; the exact-rate target is not confirmed.** The running order itself is the baseline at each horizon. Dev, 66 races, paired gain over the running order "
      "(`methodologies/M19_inrace/results/paired_gains_dev.csv`, subset all_66, models latent_s2 / gbm_s2; this is the literal-reading final version, run ids `20261117-0225_m19v2b_*`):")
    w("")
    w(md_table(pd.DataFrame(rows, columns=["horizon", "model", "running-order exact", "model exact", "rho gain [95%]", "exact-rate gain [95%]"])))
    w("")
    w("Reading: adding in-race state to the pre-race score raises rho over the running order at every horizon (all six intervals exclude 0; dev only, secondary, not tested on the fresh holdout, and LOG R13 records that the H4/H5 gains come from 2022-23, 2021 being null, and the M19 README that H6 is marginal), "
      "but exact-position rate is below the running order at all six pairs (two intervals exclude 0), because sorting by score moves cars off the place they hold. "
      "The Amendment-2 target of 25% by quarter distance is not reached by the model (H5 column above); the running order itself is the better exact-rate predictor. In-race numbers are never to be compared with pre-race ones without the horizon.")
    w("")

    # ------------------------------------------------------------ Q16
    w("## Q16 - Do lap-level driver traits and pit-crew speed add anything? (M20)")
    m20 = json.loads((R3 / "m20_summary.json").read_text())["decision"]
    cons = att_rows("M20_consistency", "latent_111")
    consg = att_rows("M20_consistency", "gbm_100")
    w("**Verdict: not confirmed for driver consistency (the only trait run in a model); tyre degradation, lap-1 gain and slow-stop rate were not tested (they failed the persistence check and were not run); crew time loss was not run.**")
    w("")
    rows = []
    for k, lab in (("consistency", "driver consistency"), ("tyre_degradation", "driver tyre degradation"), ("lap1_gain", "driver lap-1 gain"), ("crew_time_loss", "pit-crew time loss (team)"), ("crew_slow_stop_rate", "pit-crew slow-stop rate (team)")):
        d = m20[k]
        rows.append([lab, f"{d['persistence']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]", "yes" if d["persistent"] else "no", "run" if k == "consistency" else "not run"])
    w(md_table(pd.DataFrame(rows, columns=["feature", "persistence on dev (rule: >= 0.2 on the point estimate) [95%]", "passes rule", "model run?"])))
    w("")
    w(f"Source: `results/plan3/m20_summary.json`, decision. Driver consistency in the model, dev: rho {iv(cons.gain_spearman, cons.lo_spearman, cons.hi_spearman)} vs latent_111 and {iv(consg.gain_spearman, consg.lo_spearman, consg.hi_spearman)} vs gbm_100; "
      "MAE and exact rate null for both (`attribution.csv`, M20_consistency). That is the pre-stated failure. The persistence decisions are fragile (LOG R8 entry, skeptic review): consistency tracks grid position within a race (0.132 after partialling out the grid), and crew time loss fails the 0.2 rule on the point estimate but about half the reasonable variants are at or above it. "
      "Dev cannot distinguish 0 from about +0.007 rho, two to three times the expected effect (LOG R8 phase 2).")
    w("")

    # ------------------------------------------------------------ Q17
    w("## Q17 - Does a small neural network do better than the current models on the same inputs? (M21)")
    w("**Verdict: not tested.** M21 runs only if the \"no neural networks\" rule in `CLAUDE.md` is relaxed for this one control (PLAN3 §4, M21). The rule has not been relaxed and no M21 code, folder or run exists.")
    w("")

    # ------------------------------------------------------------ attribution
    w("## 1. Attribution table (`results/plan3/attribution.csv`, all rows, dev block)")
    w("")
    w("Each row is one component against one reference on the 66 dev races (21 for M12). Gains are candidate minus reference for rho and exact rate; MAE gain is reference minus candidate. "
      "\"Excl. 0\" is computed here from the interval (lower bound above 0 or upper bound below 0 on the rho column); the stored `verdict` is the text written at the time. "
      "Rows for M17, M16, M18 use references D0, S3 or grid, so their rho/exact columns are not comparable with rows against latent_111. "
      "Rows with declared = no are post hoc or diagnostic. The file is append-only; stored verdict texts are shown unedited, and where later work contradicts one it is noted below the table.")
    w("")
    rows = []
    for i, r in ATT.iterrows():
        rps = "" if (pd.isna(r.gain_rps) or (r.gain_rps == 0 and r.lo_rps == 0 and r.hi_rps == 0)) else iv(r.gain_rps, r.lo_rps, r.hi_rps, 5)
        loo = "" if pd.isna(r.loo_gain) else str(r.loo_gain)
        rows.append({"#": i, "component": r.component, "reference": r.reference, "declared": r.declared,
                     "rho gain [95%]": iv(r.gain_spearman, r.lo_spearman, r.hi_spearman), "excl. 0 (rho)": "yes" if excl0(r.lo_spearman, r.hi_spearman) else "no",
                     "MAE gain [95%]": iv(r.gain_mae, r.lo_mae, r.hi_mae, 3), "exact gain [95%]": iv(r.gain_exact, r.lo_exact, r.hi_exact),
                     "RPS gain [95%]": rps, "leave-one-out": loo, "n": int(r.n_races), "stored verdict": r.verdict})
    w(md_table(pd.DataFrame(rows)))
    w("")
    n_ex = int(sum(excl0(r.lo_spearman, r.hi_spearman) for r in ATT.itertuples()))
    w(f"{n_ex} of {len(ATT)} rho intervals in this table exclude 0 (computed from the file). Most of these are structural, not candidate passes: rows against grid order, D2 decoder rows (D2 deliberately costs rho) and leave-one-out rows. The count therefore cannot be compared with the one or two chance passes of PLAN3 §5. Corrections to stored verdict texts: "
      "(a) rows M09_car_q1_offset and M09_car_q1_offset_alone say \"passes alone\"; the offset's own increment over the history rung is null and the gain is credited to car form by the control (see Q2; M09 README). "
      "(b) M17/M16/M18 rows against grid report the exact-rate verdict as the verdict metric; their rho columns (against grid) are about ranking, not exact hits. "
      "(c) The `fresh_holdout_gain` column is empty for every row: nothing has been scored on the fresh holdout. "
      f"Leaderboard rows (`results/plan3/leaderboard.csv`, {len(LB)} rows) include superseded runs (M19 v1, v2; M11 and M16 earlier versions), kept because the file is append-only.")
    w("")
    w("![Attribution forest](figures/r16_attribution_forest.png)")
    w("")
    w("Figure: `figures/r16_attribution_forest.png`. Red = interval excludes 0; hollow squares = not declared. Both panels have a visible zero line and untruncated x axes.")
    w("")

    # ------------------------------------------------------------ final model
    w("## 2. The final plan-3 model against latent_111 and grid order")
    w("")
    w("Final = `latent_111` + reverse likelihood (M11) + car pace history (M09, `lam_x` 3.0), simulator S3 re-fitted on dev, frozen in tag `plan3-freeze` (`methodologies/P3_final/FREEZE.md`, README). "
      "Selected on the same 66 dev races it is scored on, so every dev number below is optimistic.")
    w("")
    tr = [
        ["T1 final - latent_111, rho", "dev 2021-23", A("P3_final", "latent_111", "spearman")],
        ["T1", "2024-25 (third look)", tl("test_stable", "T1:final_vs_latent_111", "spearman")],
        ["T1", "2026 R1-20 (third look)", tl("test_shift", "T1:final_vs_latent_111", "spearman")],
        ["T1", "2024-26 R20 pooled (third look)", tl("pooled test_stable+test_shift", "T1:final_vs_latent_111", "spearman")],
        ["T2 reverse on/off, MAE (ref - cand)", "dev (leave-one-out)", A("P3_LOO_reverse", ATT[ATT.component == "P3_LOO_reverse"].reference.iloc[0], "mae", 3)],
        ["T2", "2024-25 (third look)", tl("test_stable", "T2:final_vs_final_minus_reverse", "mae", 3)],
        ["T2", "2026 R1-20 (third look)", tl("test_shift", "T2:final_vs_final_minus_reverse", "mae", 3)],
        ["T2", "pooled (third look)", tl("pooled test_stable+test_shift", "T2:final_vs_final_minus_reverse", "mae", 3)],
        ["T3 final - reverse_only, rho", "dev (leave-one-out)", A("P3_LOO_pace_hist", ATT[ATT.component == "P3_LOO_pace_hist"].reference.iloc[0], "spearman")],
        ["T3", "2024-25 (third look)", tl("test_stable", "T3:final_vs_reverse_only", "spearman")],
        ["T3", "2026 R1-20 (third look)", tl("test_shift", "T3:final_vs_reverse_only", "spearman")],
        ["T3", "pooled (third look)", tl("pooled test_stable+test_shift", "T3:final_vs_reverse_only", "spearman")],
        ["final - grid, exact rate", "dev 2021-23", A("P3_final", "grid", "exact")],
        ["final - grid, exact rate", "2024-25 (third look)", tl("test_stable", "exact_vs_grid:final", "exact")],
        ["final - grid, exact rate", "2026 R1-20 (third look)", tl("test_shift", "exact_vs_grid:final", "exact")],
        ["final - grid, exact rate", "pooled (third look)", tl("pooled test_stable+test_shift", "exact_vs_grid:final", "exact")],
        ["final - grid, rho", "dev 2021-23", A("P3_final", "grid", "spearman")],
        ["final - grid, rho", "pooled (third look)", tl("pooled test_stable+test_shift", "exact_vs_grid:final", "spearman")],
        ["RPS, final S3 re-fit - latent_111 S3 re-fit (RPS gain)", "dev (both fits in-sample)", A("P3_final_S3refit", "latent_111_S3refit", "rps", 5)],
        ["RPS", "pooled (third look)", tl("pooled test_stable+test_shift", "S3:final_vs_latent_111", "rps", 5)],
    ]
    w(md_table(pd.DataFrame(tr, columns=["contrast (gain = candidate - reference unless stated)", "block", "gain [95% interval]"])))
    w("")
    w("Sources: dev rows `attribution.csv` (P3_final, P3_LOO_reverse, P3_LOO_pace_hist, P3_final_S3refit); third-look rows `results/plan3/third_look.csv` (contrast table). "
      f"**All third-look rows carry the label: {TL_LABEL}.** The dev leave-one-out rows are the same contrasts as T2 and T3 (full minus reverse; full minus pace history).")
    w("")
    w("Per-race re-derivation (this script; `methodologies/P3_final/results/*/per_race_metrics.csv`, `methodologies/M17_decoding/results/20261116-2028_m17_grid_free_dev/per_race_metrics.csv`; own bootstrap 10,000 resamples, seed 0, percentile method - it need not match the stored interval to the last digit):")
    w("")
    w(md_table(rd))
    w("")
    fin_lat = att_rows("P3_final", "latent_111")
    w(f"**Reading.** On dev the final model's rho gain over latent_111 is {iv(fin_lat.gain_spearman, fin_lat.lo_spearman, fin_lat.hi_spearman)}: the interval touches 0 and is not a pass. "
      f"Its MAE gain {A('P3_final','latent_111','mae',3)} and places 1-3 exact-rate gain ({A('P3_final','latent_111','exact_1_3')}) exclude 0 on dev, but they are two of the per-metric intervals of a model chosen on these races. Third-look counterparts for places 1-3 exact rate (already seen, no decision weight): T1 final - latent_111 pooled {tl('pooled test_stable+test_shift','T1:final_vs_latent_111','exact_1_3')}; final - grid order pooled {tl('pooled test_stable+test_shift','exact_vs_grid:final','exact_1_3')}. Neither shows the dev gain. "
      "The pace-history column lowers the full model's exact rate when removed-versus-kept is checked (leave-one-out row P3_LOO_pace_hist, exact gain "
      f"{A('P3_LOO_pace_hist', ATT[ATT.component == 'P3_LOO_pace_hist'].reference.iloc[0], 'exact')}); it was kept under the pre-stated rule, not swapped. "
      f"On the third look nothing replicates (T1 pooled {tl('pooled test_stable+test_shift','T1:final_vs_latent_111','spearman')}; T3 rho 2024-25 {tl('test_stable','T3:final_vs_reverse_only','spearman')}; secondary: MAE of the T3 contrast, 2024-25, against the model (pace column hurt), reference minus candidate, {tl('test_stable','T3:final_vs_reverse_only','mae',3)}). Neither result supports a claim that the final model improves on `latent_111`; the holdout has no race.")
    w("")
    own = rd.iloc[0]
    d1 = (data["fin"].spearman - data["lat"].spearman).values
    from scipy import stats
    se = d1.std(ddof=1) / np.sqrt(len(d1))
    tq = stats.t.ppf(0.975, len(d1) - 1)
    t_lo, t_hi = d1.mean() - tq * se, d1.mean() + tq * se
    cnt = 0
    for sd in range(200):
        r_ = np.random.default_rng(sd)
        m_ = d1[r_.integers(0, len(d1), size=(NBOOT, len(d1)))].mean(axis=1)
        cnt += int(np.percentile(m_, 2.5) > 0)
    frac200 = cnt / 200
    w(f"**Interval sensitivity.** The stored dev T1 interval is {own['stored interval']}; an independent percentile bootstrap of the same 66 per-race differences (seed 0) gives {own['own 95% bootstrap']}. "
      f"The lower bound sits on 0 under either seed stream (the skeptic's bootstrap in LOG R14 agrees with the second). The stored interval is the pre-registered one and is not a pass. Over 200 seeds the bootstrap lower bound is above 0 in {frac200*100:.0f}% of seeds, and the t-interval is [{t_lo:+.4f}, {t_hi:+.4f}]; dev T1 is neither a pass nor a clean null and is not used as a pass here.")
    w("")
    w("![Dev against third look](figures/r16_dev_vs_third_look.png)")
    w("")
    w(f"Figure: `figures/r16_dev_vs_third_look.png`. Black = dev (the decision block); grey band = {TL_LABEL}. Each panel's x axis includes 0.")
    w("")
    w("### Per-season rho, final / latent_111 / grid order")
    w("")
    sp_rows = []
    for s in (2021, 2022, 2023, 2024, 2025, 2026):
        row = {"season": s, "block": "dev (decision)" if s <= 2023 else f"third look ({'2026 R1-20' if s == 2026 else 'full season'}), already seen, no decision weight"}
        for nm in ("final", "latent_111", "grid"):
            x = sm[(sm.season == s) & (sm.model == nm)].iloc[0]
            row[nm] = f"{x.rho:.4f}"
        row["n races"] = int(sm[(sm.season == s) & (sm.model == "final")].n.iloc[0])
        sp_rows.append(row)
    w(md_table(pd.DataFrame(sp_rows)))
    w("")
    w("Sources: final and latent_111, dev: `methodologies/P3_final/results/20261118-0705_p3_{final_model,ref_latent_111}_dev/per_race_metrics.csv`; grid, dev: the M17 grid run named above (its mean 0.5916 equals `methodologies/P3_final/results/means.csv`, grid order, n = 66); "
      "2024-26: `methodologies/P3_final/results/20261118-1407_p3_R15_score/per_race_metrics.csv`. By season the dev contrast (final - latent_111) is positive but each interval includes 0 "
      "(`methodologies/P3_final/results/final_vs_latent_111_by_season.csv`). The dev gain is positive in all three dev seasons; on the old test seasons 2024 and 2025 final is below latent_111 and 2026 R1-20 is above it (table).")
    w("")
    w("![Per-season rho](figures/r16_per_season_rho.png)")
    w("")
    w(f"Figure: `figures/r16_per_season_rho.png` (axis starts at 0; bars are means over the races shown; grey band = {TL_LABEL}).")
    w("")
    w(f"Third-look exact-position rate (pooled 2024-26 R20, n = 68; `third_look.csv`, mean table): "
      + ", ".join(f"{r.name_} {r.cand_mean:.3f}" for r in
                  TL[(TL.table == 'mean') & (TL.metric == 'exact') & (TL.block == 'pooled test_stable+test_shift') & (~TL.block.str.contains('S3'))].rename(columns={"name": "name_"}).itertuples()) + f" ({TL_LABEL}).")
    w("")

    # ------------------------------------------------------------ holdout
    readings_txt = holdout_readings_text()
    w("## 3. Holdout")
    w("")
    w(f"**Status (`results/plan3/holdout_status.json`, run `{HS['last_run']}`, code `{HS['code']}`): n = {HS['n']} fresh races; `verdict_state` = \"{HS['verdict_state']}\"; {readings_txt}.** "
      f"The third look was scored from a dirty working tree (code stamp `{HS['code']}`) and committed afterwards in 0483c35. "
      "The first race eligible is 2026 R21 (Las Vegas; LOG R15 entry). A third look on 2024-26 R20 was scored once (`third_look.csv`); it has no decision weight.")
    w("")
    w("**What will be scored** (`methodologies/P3_final/HOLDOUT_PREREG.md`; not edited here):")
    w("- Every race with race_id > 271 (2026 R21 onward) that has a classified result, at H2, as it is run. No race is removed after its outcome is seen.")
    w("- Point metrics from the harness score order; probabilities (RPS, points log-loss) from the S3 matrix with the frozen re-fitted theta. References: `latent_111`, `final_minus_reverse`, `reverse_only`, `plan2_H2`, `grid` (all frozen, same harness, same races).")
    w("- **Three primary tests**, each a paired per-race bootstrap (10,000 resamples, seed 0, 95%): **T1** `final` - `latent_111`, Spearman rho; **T2** `final` vs `final_minus_reverse`, MAE (reference - candidate); **T3** `final` - `reverse_only`, Spearman rho.")
    w("- **Confirmed** only if the interval excludes 0 in the stated direction after **Holm correction over T1-T3** (familywise 0.05). An uncorrected pass, or the same sign without the interval, is \"not confirmed\". Nothing is \"promising\".")
    w("- **When:** no verdict before **n >= 20** holdout races (before that every number is printed \"no verdict (n < 20)\" with its interval); the verdict is read once at the first n >= 20 and again at **n >= 40** if the data exist; the later reading supersedes the earlier. "
      "No stopping when a result looks good. Whether n = 20 is reachable inside 2026 depends on the calendar, which is not in the files read for this report (unverified); the M10 card says late 2027 at best for its own rung.")
    w("- Secondary, no verdict weight: exact rate against grid order (target counts as met only if the interval excludes 0 at n >= 20), finishers' metrics, place bands, within +-k, RPS and points log-loss through the S3 re-fit, final against `plan2_H2`, per season. "
      "M10 is judged separately (H1 rho, >= 20 fresh races; `methodologies/M10_upgrades/README.md`). If T3 is not confirmed, M09 is reported as \"not confirmed on holdout\" and the frozen model is not swapped.")
    w("")
    w("**Main-session decisions applied where the pre-registration was silent** (`experiments/LOG.md`, R15 entry, made before any holdout race):")
    w("- **Holm:** tests ranked by one-sided bootstrap p = (1 + #{boot <= 0}) / (n_boot + 1), ties broken T1, T2, T3; rank k is confirmed iff the two-sided 1 - 0.05/(m-k+1) percentile lower bound is above 0; stop at the first failure; a wrong-way test ranks last and cannot block.")
    w("- **`plan2_H2`:** the pre-registration's `plan2_H2` is read as the plan-2 frozen H2 model, which is `latent_111`; so the secondary \"final vs plan2_H2\" equals T1 (the two rows are numerically identical in `third_look.csv`). The ambiguity is recorded, not resolved by choosing a better-looking reference.")
    w("- A level-t reading (t = 20, 40) uses data through the t-th holdout race only; verdict files are write-once, carry row digests and the commit hash, and are refused from a dirty tree. Constants (alpha 0.05, levels 20 and 40, 10,000 resamples, seed 0) are hard-coded.")
    w("")
    w("**Command to run after each new race has entered the data:**")
    w("")
    w("```")
    w("python methodologies/P3_final/score.py")
    w("```")
    w("")
    w("It is a no-op if nothing is new; it verifies the config hash and the row hashes of races <= 271 first and stops on any mismatch. `--no-write` computes and prints only.")
    w("")
    w("**Operational gaps (LOG R15 entry, not fixed):**")
    w("- No live pre-race entries builder exists; `predict.py` takes an `--entries` frame, and its simulator D0 order is Monte-Carlo noisy and differs from the harness (descriptive only).")
    w("- The 2027 entry list and lineages are to be added per pre-registration section 6; not yet done.")
    w("- Rebuilding old rows when a new race is appended was not dry-run against a real fetch.")
    w("- The OpenF1 session list needs a refresh.")
    w("- `freeze_rowhash.json` was generated eight minutes after the tag, not inside it (its data were checked against the file hashes in the tag); it is committed and never regenerated.")
    w("")

    # ------------------------------------------------------------ limitations
    tlc = TL[TL.table == "contrast"].dropna(subset=["lo", "hi"])
    n_c = len(tlc)
    n_x = int(((tlc.lo > 0) | (tlc.hi < 0)).sum())
    n_pos, n_neg = int((tlc.lo > 0).sum()), int((tlc.hi < 0).sum())
    n_fz = int(((tlc.hi < 0) & (tlc.hi > -1e-9)).sum())
    w("## 4. Limitations and what did not work")
    w("")
    w("- **No result in this plan is confirmed on clean data.** The fresh holdout is empty. Everything else is dev (used for tuning and selection) or the old test blocks (already seen twice in plans 1-2, a third time here).")
    w(f"- **Multiplicity.** `third_look.csv` holds {n_c} contrast intervals (counting the S3-matrix blocks, per-season blocks and all metrics, which overlap heavily); {n_x} exclude 0 by the literal rule lower bound > 0 or upper bound < 0 ({n_pos} positive, {n_neg} negative); {n_fz} of the negative ones have an upper bound of about -6.9e-19, a floating-point zero (reverse_only exact vs grid, test_shift and season 2026), so with a 1e-9 tolerance the count would be {n_x - n_fz}. The blocks test_shift and season 2026 are identical, so those rows are exact duplicates. "
      f"Several hundred intervals at the 5% level produce many such exclusions by chance, and the signs are mixed. None of them is read as a finding. In `attribution.csv`, {n_ex} of {len(ATT)} rho intervals exclude 0, but most are structural (rows against grid, D2 rows), so that count does not compare with the one or two chance passes PLAN3 §5 expects among the secondary comparisons.")
    w(f"- **The headline order-accuracy gain does not hold up.** Dev T1 is {sg(t1d.gain_spearman)} with a lower bound of {sg(t1d.lo_spearman)}, selected on the same races; the third look gives {sg(t3.gain)}. The component behind most of the dev gain (M09 pace history) is credited to car form by its own control and lowers the exact rate. A change inside the noise is not an improvement.")
    eg = TL[(TL.table == "contrast") & TL.name.str.startswith("exact_vs_grid") & (TL.metric == "exact")]
    below = sorted({f"{r.name.split(':')[1]} ({r.block.replace('test_stable', '2024-25').replace('test_shift', '2026 R1-20')}, {r.gain:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}])" for r in eg.itertuples() if r.hi < -1e-9 and not r.block.startswith("season 2026") and not r.block.startswith("pooled")})
    above = int((eg.lo > 1e-9).sum()) + int((ATT[ATT.reference == "grid"].lo_exact > 1e-9).sum())
    w(f"- **Exact-position rate: grid order is never beaten.** No exact-rate interval against grid order lies above 0: {above} of {len(eg)} third-look rows and of the {int((ATT.reference == 'grid').sum())} dev rows against grid (`attribution.csv`, `third_look.csv`). Final: {iv(ex.gain_exact, ex.lo_exact, ex.hi_exact)} on dev, {tl('pooled test_stable+test_shift','exact_vs_grid:final','exact')} pooled third look, both including 0. "
      f"The intervals that do exclude 0 in the third look are below grid order: {'; '.join(below)} (the 2026 R1-20 and season-2026 blocks are identical and listed once; reverse_only in 2026 has an upper bound of about -6.9e-19, a floating-point zero, so it is not listed). The Amendment-2 pre-race target is not met. The 25% pre-race target was not adopted (it is oracle level).")
    w("- **Q1 gate: failed as declared, and the null depends on a reference the plan did not fix.** The post-hoc Q1-reference analysis shows a persistent car-form signal, which the exploratory M09 run then picked up as car form, not race pace.")
    w("- **Dev power is low.** 66 races (21 for M12, 22 per season) resolve rho differences of roughly 0.007-0.01 and exact-rate differences of about 2 points. Nulls mean \"not shown\", not \"zero\".")
    w("- **Post-hoc and exploratory pieces carry no decision weight:** M09 after the failed gate; the car-form control; the M14 carry-over variant; the M20 robustness variants; the A3 reference sensitivity; M17 blends. Several declared expectations (e.g. M20 pre-statement, M14 pre-statement) cannot be shown to precede the runs (timestamps only; see LOG R8, R11 entries).")
    w("- **Selection optimism.** The S3 re-fit and the final bundle were fitted and scored on the same 66 dev races; four S3 parameters sit at grid edges and the grids were not extended. The re-fit's own RPS gain over the plan-2 theta is null (LOG R14 entry).")
    w("- **Strength audit.** Driver strengths are weakly identified (see Q9); out-of-sample teammate correlation is far below the in-sample one; one-race drivers still sit at the extremes of the unseparated ranking. Car-level strengths are believable in order but intervals are conditional on sigma and unverified for coverage.")
    w("- **Data gaps.** M10 has no dev data (2024 R2 onward only; 2024 R1 missing; parser quality measured in-sample on 67 documents, residual silent error not measured). Pace features have 2021 round 12 missing; fuel coefficients are implausible in a few races (D2). Tyre data exist for 2023-26 only. Jolpica pit time is pit-lane time, not stationary time.")
    w("- **Cut or not run:** M13 (no permitted odds source), M15 (rookie prior), M14 parts 2-3, M20 crew time loss, M21, the lap-level simulator (deferred to plan 4). Q3, Q7, Q11, Q17 and parts of Q10 and Q16 are therefore not tested.")
    w("- **Process.** The plan asks for A3 first; R5 and R10 were committed before it (no number changes). Several reviews found leaks or reinterpretations that were fixed before commit (M19 red-flag leak; race-distance from the race's own last lap; M14 look-ahead). Leakage checks (truncation and scrambling) pass on the tested races; they do not cover every race.")
    w("")

    # ------------------------------------------------------------ files
    w("## 5. Files")
    w("")
    w("- `results/plan3/REPORT.md` (this file), `results/plan3/make_report.py` (regenerates it), `results/plan3/figures/r16_dev_vs_third_look.png`, `r16_per_season_rho.png`, `r16_attribution_forest.png`, `r16_all_lineages_car_strength.png`.")
    w("- Earlier A4/A5 figures remain in `results/plan3/figures/` unchanged.")
    w("")
    (R3 / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("wrote", R3 / "REPORT.md", "lines", len(L))


if __name__ == "__main__":
    build()
