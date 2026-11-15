"""Build results/plan2/REPORT.md and results/plan2/figures/*.png from saved runs (PLAN2.md P8).

Everything is read from run folders listed in results/leaderboard.csv (plan 1) and
results/plan2/leaderboard*.csv (plan 2), so each number traces back to a file. Sections are independent:
a failing section is reported in the document instead of aborting the run.
"""
from __future__ import annotations

import json
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitlane.config import CONFIGS, PLAN2, RESULTS, ROOT, load_yaml
from pitlane.evaluation.metrics import pred_rank
from pitlane.plan2 import dnf as D
from pitlane.plan2 import pipeline2 as Q
from pitlane.plan2 import simulate as S
from pitlane.report import boot_ci, factor_effects, fmt_ci, md_table, wilcoxon_p

FIG = PLAN2 / "figures"
SPLITS = ["dev", "test_stable", "test_shift", "fresh"]
SNAME = {"dev": "Development 2021-23", "test_stable": "Test 2024-25 (second look)",
         "test_shift": "Test 2026 rounds 1-20 (second look)", "fresh": "Fresh holdout (2026 after round 20)"}
GROUP_NAME = {"qrich": "richer qualifying (Q1-Q3 gaps, session reached, teammate gap)",
              "practice": "practice pace (best lap, long run, laps)", "sprint": "sprint result",
              "weather": "weather (wet qualifying, race forecast, mismatch x qualifying)"}
CELLS = Q.CELLS


# ------------------------------------------------------------------------------------ loading
def load_lb() -> pd.DataFrame:
    a = pd.read_csv(RESULTS / "leaderboard.csv")
    a["plan"] = 1
    b = pd.read_csv(Q.LB2) if Q.LB2.exists() else pd.DataFrame()
    b["plan"] = 2
    lb = pd.concat([a, b], ignore_index=True).drop_duplicates(["label", "split"], keep="last").reset_index(drop=True)
    lb["per_race"] = [pd.read_csv(ROOT / d / "per_race_metrics.csv").set_index("race_id") for d in lb["run_dir"]]
    return lb


def pr(lb: pd.DataFrame, label: str, split: str) -> pd.DataFrame | None:
    m = lb[(lb["label"] == label) & (lb["split"] == split)]
    return None if m.empty else m.iloc[0]["per_race"]


def diff(lb, a: str, b: str, split: str, metric: str = "spearman", mask: pd.Series | None = None) -> np.ndarray | None:
    x, y = pr(lb, a, split), pr(lb, b, split)
    if x is None or y is None:
        return None
    d = (x[metric] - y[metric]).dropna()
    if mask is not None:
        d = d[d.index.isin(mask[mask].index)]
    return d.to_numpy()


def flags(entries: pd.DataFrame) -> pd.DataFrame:
    g = entries.groupby("race_id")
    return pd.DataFrame({"first5": g["round"].first() <= 5,
                         "wet": (g["wet_q"].first().fillna(0) > 0) | (g["rain_fc"].first().fillna(0) > 0),
                         "sprint": g["sprint_flag"].first() > 0,
                         "practice": g["fp_missing"].first() == 0})


def sec(title: str, fn, *a) -> str:
    try:
        return fn(*a)
    except Exception as e:                                   # a broken section must not stop the run
        return f"## {title}\n\n_Section failed: `{type(e).__name__}: {e}`_\n\n```\n{traceback.format_exc()[-900:]}\n```\n"


def ci_cell(x: np.ndarray | None, d: int = 3) -> str:
    return "n/a" if x is None or len(x) < 3 else fmt_ci(boot_ci(x), d)


# ------------------------------------------------------------------------------------ sections
def s_overview(lb, frozen2, ceil) -> str:
    return f"""# PitLane plan 2 — results report

_Generated from `results/leaderboard.csv` (plan 1) and `results/plan2/leaderboard*.csv`. Plan-2 configs were
frozen on development data (hash `{frozen2.get('hash')}`) before the test blocks were scored. **The 2024-2026 test
blocks had already been scored once in plan 1, so every test number here is a second look**; only the
"fresh holdout" (races after 2026 round 20) is untouched. Every number is a mean over races of per-race metrics;
intervals are 95% bootstrap CIs over races, comparisons are paired by race._

**Research question 2:** which pre-race information, and which race structure, add predictive value beyond qualifying order?

## 0. Data actually used (deviations from the plan)
| Planned | What happened |
|---|---|
| FastF1 practice laps and session weather, 2018+ | FastF1's timing API now answers 401 without an F1 account, so it could not be used. Practice laps and session weather come from **OpenF1** (no login), which only covers **2023 onward**. Practice features therefore exist for 2023-2026 only; earlier races carry a missing indicator. |
| Race-day forecast archive | Open-Meteo: ERA5 for the qualifying day (observed, all seasons) and its historical-forecast archive for the race window (used from 2020). |
| Sprint results | Jolpica sprint results, 2021+. |
| Race-pace oracle on all of dev | Jolpica allows ~500 requests/hour; race laps were fetched for the 2023 development season only. |
| Practice evaluated on dev 2021-23 | Practice data covers only 2023 in dev (22 races, with an early-2023 training set of 0-21 races). Practice groups are therefore judged on dev-2023 plus the test blocks, and the dev CI for them is wide by construction. |
"""


def s_ceiling(lb, ceil) -> str:
    f = FIG / "f1_ceiling.png"
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    names = ["model\n(latent_111)", "self-consistency\nceiling", "retirement\noracle"]
    vals = [ceil["model_rho"], ceil["self_consistency_rho"], ceil["retirement_oracle_rho"]]
    ax.bar(names, vals, color=["#1f77b4", "#7f7f7f", "#2ca02c"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center")
    ax.set_ylim(0.5, max(vals) + 0.06)
    ax.set_ylabel("mean per-race Spearman rho (dev)")
    ax.set_title("A1: how much order accuracy is left?")
    fig.tight_layout(); fig.savefig(f, dpi=140); plt.close(fig)
    pace = ""
    if ceil.get("pace_races"):
        pace = (f"| Race-pace oracle (finishers ranked by median clean race lap), {ceil['pace_races']} races of 2023 | "
                f"model rho(finishers) {ceil['pace_model_rho_fin']:.3f} vs oracle {ceil['pace_oracle_rho_fin']:.3f} "
                f"(gap {ceil['pace_gap_fin']:+.3f}) |\n")
    gate = ("order accuracy treated as **saturated** (gap < 0.03): probabilities are the headline metric and M06 was "
            "tuned in one round" if ceil["saturated"] else
            "order accuracy is **not saturated** (gap >= 0.03): better race-pace information could still help order, "
            "so M06 used the full tuning round")
    note = ""
    if ceil["self_consistency_rho"] < ceil["model_rho"]:
        note = ("_Note: the self-consistency value is **below** the model's real rho, so it is not an upper bound. The Plackett-Luce temperature is "
                "fitted on whole orders, retirements included, and its sampled orders are noisier than real races are for a model of this "
                "sharpness; read it as the noise level the probability model assumes, not as a limit on accuracy._")
    return f"""## 1. A1 — Ceiling analysis (development block, oracles use race data and are not models)
| Quantity | Value |
|---|---|
| Model: `latent_111` (plan-1 proposed), {ceil['n_races']} races | rho {ceil['model_rho']:.3f} |
| Self-consistency ceiling: model scored against orders sampled from its own calibrated distribution | rho {ceil['self_consistency_rho']:.3f} |
| Retirement oracle: retirees placed at their true positions | rho {ceil['retirement_oracle_rho']:.3f} (gain {ceil['retirement_oracle_gain']:+.3f}) |
{pace}
**Gate decision:** {gate}.

{note}
![ceiling](figures/f1_ceiling.png)
"""


def leaderboard_md(lb, split, models) -> str:
    rows = []
    for label in models:
        m = lb[(lb["label"] == label) & (lb["split"] == split)]
        if m.empty:
            continue
        r = m.iloc[0]
        f = lambda k, d=3: "" if pd.isna(r.get(k)) else f"{r[k]:.{d}f}"
        rows.append({"model": f"{r['method']} {label}", "rho": f("spearman"), "MAE": f("mae", 2), "top-10 overlap": f("points"),
                     "top-5 overlap": f("top5"), "exact": f("exact"), "within 1": f("within1"), "within 2": f("within2"),
                     "RPS": f("rps", 4), "win logloss": f("logloss_win", 4), "points logloss": f("logloss_points", 4),
                     "races": int(r["n_races"])})
    if not rows:
        return "_no runs for this split_\n"
    return md_table(pd.DataFrame(rows).sort_values("rho", ascending=False))


def s_leaderboards(lb) -> str:
    base = ["grid", "quali", "standings", "rf_000", "gbm_000", "gbm_001", "gbm_111", "latent_000", "latent_111"]
    p2 = [l for l in lb[lb["plan"] == 2]["label"].unique() if l.startswith(("m06_L_r", "m06_G", "h0_", "h1_", "m08"))]
    out = ["## 2. Leaderboards (plan-1 reference models and plan-2 models)\n",
           "`top-10 overlap` = how many of the predicted top 10 are in the real top 10; `exact` = share of drivers placed in their exact position; "
           "`RPS` = ranked probability score over the full position distribution (lower is better).\n"]
    for s in SPLITS:
        if not (lb["split"] == s).any():
            out.append(f"### {SNAME[s]}\n_No races yet (the 2026 season data ends at round 20); the pipeline scores them automatically when `scripts/run_plan2.py` is re-run after re-ingesting._\n")
            continue
        out.append(f"### {SNAME[s]}\n" + leaderboard_md(lb, s, base + sorted(p2)) + "\n")
    return "\n".join(out)


def _eff_table(lb, prefix: str, split: str, metric: str = "spearman") -> tuple[str, dict | None]:
    cols = {}
    for c in CELLS:
        p = pr(lb, f"{prefix}{c}", split)
        if p is None:
            return "", None
        cols[c] = p[metric]
    M = pd.DataFrame(cols).dropna()
    eff = factor_effects(M)
    names = {"F1": "F1 dynamic", "F2": "F2 driver-car separation", "F3": "F3 structured ranking",
             "F1xF2": "F1 x F2", "F1xF3": "F1 x F3", "F2xF3": "F2 x F3"}
    rows = [{"effect": names[k], "d rho [95% CI]": fmt_ci(boot_ci(v)), "p (Wilcoxon)": f"{wilcoxon_p(v):.3f}"} for k, v in eff.items()]
    return md_table(pd.DataFrame(rows)), eff


def s_horizon(lb) -> str:
    out = ["## 3. A2 — The three plan-1 factors at different forecast horizons (Family L)\n",
           "H0 = history only (no qualifying, no practice); H2 = plan 1 (history + qualifying position/grid/Q1 gap) from the plan-1 runs; "
           "H1 = history + practice only (practice exists from 2023, so on 2021-22 it equals H0). "
           "Effect = average gain from switching a factor ON over the other two.\n"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, split in zip(axes, ["test_stable", "test_shift"]):
        y = 0
        for tag, prefix, col in [("H0", "h0_latent_", "#d62728"), ("H2 (plan 1)", "latent_", "#1f77b4")]:
            tab, eff = _eff_table(lb, prefix, split)
            if eff is None:
                continue
            for j, f in enumerate(["F1", "F2", "F3"]):
                m, lo, hi = boot_ci(eff[f])
                ax.errorbar(m, j + (0.12 if tag == "H0" else -0.12), xerr=[[m - lo], [hi - m]], fmt="o", color=col, capsize=3,
                            label=tag if j == 0 else None)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(3), ["F1 dynamic", "F2 separation", "F3 ranking"])
        ax.set_title(SNAME[split], fontsize=10)
        ax.set_xlabel("main effect on Spearman rho (95% CI)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Do the plan-1 factors matter more before qualifying is known?", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "f2_horizon_effects.png", dpi=140); plt.close(fig)
    for split in ["dev", "test_stable", "test_shift", "fresh"]:
        t0, e0 = _eff_table(lb, "h0_latent_", split)
        if e0 is None:
            continue
        t2, _ = _eff_table(lb, "latent_", split)
        out.append(f"### {SNAME[split]}\n**H0 (history only)**\n\n{t0}\n\n**H2 (plan 1: history + qualifying)**\n\n{t2}\n")
    rows = []
    for split in ["dev", "test_stable", "test_shift"]:
        row = {"split": SNAME[split]}
        for tag, label in [("H0 A", "h0_latent_000"), ("H0 D", "h0_latent_111"), ("H1 A", "h1_latent_000"), ("H1 D", "h1_latent_111"),
                           ("H2 A", "latent_000"), ("H2 D", "latent_111"), ("grid", "grid"), ("quali", "quali"), ("standings", "standings")]:
            p = pr(lb, label, split)
            row[tag] = "" if p is None else f"{p['spearman'].mean():.3f}"
        rows.append(row)
    out.append("### Mean rho by horizon (A = all factors off, D = all on)\n" + md_table(pd.DataFrame(rows)) + "\n\n![horizon](figures/f2_horizon_effects.png)\n")
    return "\n".join(out)


def s_ladder(lb, entries) -> str:
    fl = flags(entries)
    rungs = [("r0", "H2 = plan 1"), ("r1", "+ richer qualifying"), ("r2", "+ practice"), ("r3", "+ sprint"), ("r4", "+ weather = H3")]
    group_of = {"r1": "qrich", "r2": "practice", "r3": "sprint", "r4": "weather"}
    out = ["## 4. M06 — Does weekend information beyond qualifying position add anything? (Family L)\n",
           "Every rung and leave-one-out variant is tuned separately on dev with the same grid (`r0` is the plan-1 frozen model, reproduced below). "
           "Rule (fixed in `PLAN2.md` §5): a group adds value if its paired dev gain has a 95% CI above 0 **and** the gain has the same sign on test 2024-25. "
           "Practice is judged on dev-2023 only (the only dev season with practice data).\n"]
    # reproduction check
    mx = []
    for cell in ("000", "111"):
        for s in ("dev", "test_stable", "test_shift"):
            a, b = pr(lb, f"latent_{cell}", s), pr(lb, f"m06_L_r0_{cell}", s)
            if a is not None and b is not None:
                mx.append(float((a["spearman"] - b["spearman"]).abs().max()))
    if mx:
        out.append(f"**Reproduction check (P0):** the plan-1 latent models re-run under the plan-2 code differ from the stored plan-1 runs by at most {max(mx):.2e} in per-race rho.\n")
    for cell in ("111", "000"):
        rows = []
        for split in ["dev", "test_stable", "test_shift", "fresh"]:
            for r, name in rungs:
                p = pr(lb, f"m06_L_{r}_{cell}", split)
                if p is None:
                    continue
                prev = f"r{int(r[1]) - 1}"
                d = diff(lb, f"m06_L_{r}_{cell}", f"m06_L_{prev}_{cell}", split) if r != "r0" else None
                dm = diff(lb, f"m06_L_{r}_{cell}", f"m06_L_r0_{cell}", split) if r != "r0" else None
                rows.append({"split": SNAME[split].split(" (")[0], "rung": name, "rho": f"{p['spearman'].mean():.3f}",
                             "MAE": f"{p['mae'].mean():.2f}", "d rho vs previous rung": ci_cell(d), "d rho vs H2": ci_cell(dm)})
        if rows:
            out.append(f"### Information ladder, cell {cell} ({'proposed, all three factors on' if cell == '111' else 'all factors off'})\n" + md_table(pd.DataFrame(rows)) + "\n")
    # LOO + verdicts
    rows, verd = [], []
    for g in ["qrich", "practice", "sprint", "weather"]:
        r = [k for k, v in group_of.items() if v == g][0]
        prev = f"r{int(r[1]) - 1}"
        mask = fl["practice"] if g == "practice" else None
        dd = diff(lb, f"m06_L_{r}_111", f"m06_L_{prev}_111", "dev", mask=mask)
        dt = diff(lb, f"m06_L_{r}_111", f"m06_L_{prev}_111", "test_stable")
        dl = {s: diff(lb, "m06_L_r4_111", f"m06_L_no_{g}_111", s, mask=(fl["practice"] if g == "practice" and s == "dev" else None))
              for s in ("dev", "test_stable", "test_shift")}
        rows.append({"group removed from H3": GROUP_NAME[g], **{f"d rho (H3 - without) {s}": ci_cell(v) for s, v in dl.items()}})
        bd, bt = (boot_ci(dd) if dd is not None else (np.nan,) * 3), (boot_ci(dt) if dt is not None else (np.nan,) * 3)
        if bd[1] > 0 and bt[0] > 0:
            v = "**adds value (rule met)**"
        elif bd[0] > 0 and bt[0] > 0:
            v = "positive on both blocks, rule not met (dev CI includes 0)"
        elif bt[0] > 0:
            v = "positive on test only"
        else:
            v = "no gain"
        verd.append({"group": GROUP_NAME[g], "dev gain (rung)": ci_cell(dd), "test 2024-25 gain (rung)": ci_cell(dt), "verdict": v,
                     "dev races": 0 if dd is None else len(dd)})
    out.append("### Leave-one-group-out from H3 (cell 111)\n" + md_table(pd.DataFrame(rows)) + "\n")
    out.append("### Verdict per information group (decision rule)\n" + md_table(pd.DataFrame(verd)) + "\n")
    # Family G
    rows = []
    for cell in ("000", "111"):
        for split in ["dev", "test_stable", "test_shift", "fresh"]:
            a, b = pr(lb, f"m06_G_h3_{cell}", split), pr(lb, f"gbm_{cell}", split)
            if a is not None and b is not None:
                rows.append({"cell": cell, "split": SNAME[split].split(" (")[0], "H2 (plan 1) rho": f"{b['spearman'].mean():.3f}",
                             "H3 rho": f"{a['spearman'].mean():.3f}", "d rho [CI]": ci_cell(diff(lb, f"m06_G_h3_{cell}", f"gbm_{cell}", split))})
    if rows:
        out.append("### Family G (LightGBM): H3 against the plan-1 features\n" + md_table(pd.DataFrame(rows)) + "\n")
    # figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
    for ax, split in zip(axes, ["dev", "test_stable", "test_shift"]):
        for cell, col in (("111", "#1f77b4"), ("000", "#ff7f0e")):
            ms, los, his = [], [], []
            for r, _ in rungs:
                p = pr(lb, f"m06_L_{r}_{cell}", split)
                b = boot_ci(p["spearman"].to_numpy()) if p is not None else (np.nan,) * 3
                ms.append(b[0]); los.append(b[1]); his.append(b[2])
            x = np.arange(len(rungs)) + (0.07 if cell == "111" else -0.07)
            ax.errorbar(x, ms, yerr=[np.array(ms) - np.array(los), np.array(his) - np.array(ms)], fmt="o-", color=col, capsize=3, label=f"cell {cell}")
        q = pr(lb, "quali", split)
        if q is not None:
            ax.axhline(q["spearman"].mean(), color="k", ls="--", lw=1, label="qualifying order")
        ax.set_xticks(range(len(rungs)), ["H2", "+qual", "+prac", "+sprint", "+wx"], fontsize=9)
        ax.set_title(SNAME[split].split(" (")[0], fontsize=10)
    axes[0].set_ylabel("mean per-race rho"); axes[0].legend(fontsize=8)
    fig.suptitle("Information ladder (Family L)", fontweight="bold"); fig.tight_layout()
    fig.savefig(FIG / "f3_info_ladder.png", dpi=140); plt.close(fig)
    out.append("![ladder](figures/f3_info_ladder.png)\n")
    return "\n".join(out)


def s_dnf(lb, entries) -> str:
    if not D.LB_DNF.exists():
        return "## 5. M07\n_no runs_\n"
    d = pd.read_csv(D.LB_DNF)
    rows, preds_all = [], []
    for _, r in d.iterrows():
        per = pd.read_csv(ROOT / r["run_dir"] / "per_race_metrics.csv")
        ll, lc, lr = per["logloss"].to_numpy(), per["logloss_const"].to_numpy(), per["logloss_rate"].to_numpy()
        rows.append({"split": SNAME[r["split"]], "logloss hazard": f"{ll.mean():.4f}", "logloss constant": f"{lc.mean():.4f}",
                     "logloss decayed-rate": f"{lr.mean():.4f}", "Brier hazard": f"{per['brier'].mean():.4f}",
                     "Brier constant": f"{per['brier_const'].mean():.4f}", "AUC hazard": f"{per['auc'].mean():.3f}",
                     "d logloss vs constant [CI]": fmt_ci(boot_ci(lc - ll), 4), "d logloss vs decayed-rate [CI]": fmt_ci(boot_ci(lr - ll), 4),
                     "races": len(per)})
        preds_all.append(pd.read_parquet(ROOT / r["run_dir"] / "predictions.parquet").assign(split=r["split"]))
    out = ["## 5. M07 — DNF hazard model (logistic, walk-forward)\n",
           "`d logloss` is baseline minus hazard model: positive = the hazard model is better. Target = not classified.\n", md_table(pd.DataFrame(rows)) + "\n"]
    p = pd.concat(preds_all)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for s, col in zip(["dev", "test_stable", "test_shift"], ["#8c8c8c", "#1f77b4", "#d62728"]):
        x = p[p["split"] == s]
        if x.empty:
            continue
        b = pd.cut(x["p_dnf"], [0, .03, .06, .1, .15, .2, .3, 1.0])
        g = x.groupby(b, observed=True).agg(p=("p_dnf", "mean"), y=("dnf", "mean"), n=("dnf", "size"))
        ax.plot(g["p"], g["y"], "o-", color=col, label=SNAME[s].split(" (")[0])
    ax.plot([0, .4], [0, .4], "k--", lw=0.8)
    ax.set_xlabel("predicted P(not classified)"); ax.set_ylabel("observed rate"); ax.legend(fontsize=8)
    ax.set_title("M07 calibration")
    fig.tight_layout(); fig.savefig(FIG / "f4_dnf_calibration.png", dpi=140); plt.close(fig)
    last = d.sort_values("run_id").iloc[-1]
    coef = json.loads((ROOT / last["run_dir"] / "coefficients.json").read_text(encoding="utf-8"))
    out.append("Coefficients of the last refit (standardised features; positive = more retirements):\n\n" +
               md_table(pd.DataFrame([{"feature": k, "coef": v} for k, v in coef.items() if k not in ("upto_race",)])) +
               "\n\n![dnf](figures/f4_dnf_calibration.png)\n")
    return "\n".join(out)


def s_sim(lb, frozen2) -> str:
    out = ["## 6. M08 — Race-level Monte Carlo simulator\n",
           "S0 = Plackett-Luce sampling of the M06 scores (tau fitted on dev by RPS); S1 = + retirements from M07; S2 = + circuit grid stickiness; "
           "S3 = + race-regime mixture. `M06 harness` is M06's own probabilities (online-calibrated tau, no retirement model).\n"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for split in ["dev", "test_stable", "test_shift", "fresh"]:
        rows = []
        for label, name in [("m06_L_r4_111", "M06 harness"), ("m08_S0", "S0 PL only"), ("m08_S1", "S1 + retirements"),
                            ("m08_S2", "S2 + stickiness"), ("m08_S3", "S3 + regime")]:
            p = pr(lb, label, split)
            if p is None:
                continue
            prev = {"m08_S1": "m08_S0", "m08_S2": "m08_S1", "m08_S3": "m08_S2", "m08_S0": "m06_L_r4_111"}.get(label)
            rows.append({"model": name, "RPS": f"{p['rps'].mean():.4f}", "points logloss": f"{p['logloss_points'].mean():.4f}",
                         "win logloss": f"{p['logloss_win'].mean():.4f}", "podium Brier": f"{p['brier_podium'].mean():.4f}",
                         "rho": f"{p['spearman'].mean():.3f}", "MAE": f"{p['mae'].mean():.2f}",
                         "d RPS vs previous (positive = better) [CI]": "" if prev is None or diff(lb, prev, label, split, "rps") is None else fmt_ci(boot_ci(diff(lb, prev, label, split, "rps")), 4),
                         "d points logloss vs previous [CI]": "" if prev is None or diff(lb, prev, label, split, "logloss_points") is None else fmt_ci(boot_ci(diff(lb, prev, label, split, "logloss_points")), 4)})
        if rows:
            out.append(f"### {SNAME[split]}\n" + md_table(pd.DataFrame(rows)) + "\n")
    for ax, (metric, title) in zip(axes, [("rps", "RPS (lower is better)"), ("logloss_points", "P(points) log-loss (lower is better)")]):
        for k, (split, col) in enumerate([("dev", "#8c8c8c"), ("test_stable", "#1f77b4"), ("test_shift", "#d62728")]):
            ys = []
            for label in ["m06_L_r4_111", "m08_S0", "m08_S1", "m08_S2", "m08_S3"]:
                p = pr(lb, label, split)
                ys.append(np.nan if p is None else p[metric].mean())
            ax.plot(range(5), ys, "o-", color=col, label=SNAME[split].split(" (")[0])
        ax.set_xticks(range(5), ["M06", "S0", "S1", "S2", "S3"]); ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "f5_sim_ladder.png", dpi=140); plt.close(fig)
    th = pd.DataFrame([{"variant": v, **{k: round(float(frozen2["M08"][v][k]), 3) for k in S.DEFAULT}, "dev RPS (fit)": round(float(frozen2["M08"][v]["fit_rps"]), 4)}
                       for v in S.SWITCHES if v in frozen2["M08"]])
    out.append("### Parameters fitted on the development block\n" + md_table(th) + "\n\n![sim](figures/f5_sim_ladder.png)\n")
    return "\n".join(out)


def s_subsets(lb, entries) -> str:
    fl = flags(entries)
    models = [("quali", "qualifying order"), ("latent_111", "plan-1 latent_111"), ("m06_L_r4_111", "M06 H3 (latent)"), ("m08_S3", "M08 S3")]
    out = ["## 7. Where does it help? Subsets (mean per-race rho; counts are races)\n"]
    rows = []
    for split in ["dev", "test_stable", "test_shift"]:
        for name, mask in [("first five rounds of a season", fl["first5"]), ("wet-affected weekend", fl["wet"]),
                           ("sprint weekend", fl["sprint"]), ("practice data available", fl["practice"])]:
            row = {"split": SNAME[split].split(" (")[0], "subset": name}
            n = None
            for label, short in models:
                p = pr(lb, label, split)
                if p is None:
                    row[short] = ""
                    continue
                sub = p[p.index.isin(mask[mask].index)]
                n = len(sub)
                row[short] = f"{sub['spearman'].mean():.3f}" if len(sub) else ""
            row["races"] = n or 0
            rows.append(row)
    out.append(md_table(pd.DataFrame(rows)) + "\n\n_Subsets under 10 races are not interpreted (PLAN2.md §7)._\n")
    # entry-level subsets: rookies and drivers new to a team
    rows = []
    ent = entries.set_index(["race_id", "driverId"])
    for split in ["dev", "test_stable", "test_shift"]:
        for flag, name in [("rookie", "rookies (first 20 races)"), ("new_team", "first race with a new team")]:
            row = {"split": SNAME[split].split(" (")[0], "entries": name}
            cnt = 0
            for label, short in models:
                m = lb[(lb["label"] == label) & (lb["split"] == split)]
                if m.empty:
                    row[short] = ""
                    continue
                p = pd.read_parquet(ROOT / m.iloc[0]["run_dir"] / "predictions.parquet")
                p["rank"] = p.groupby("race_id")["score"].transform(lambda s: pred_rank(s.to_numpy()))
                p["f"] = ent.loc[list(zip(p["race_id"], p["driverId"])), flag].to_numpy() > 0
                err = (p["rank"] - p["pos"]).abs()
                cnt = int(p["f"].sum())
                row[short] = f"{err[p['f']].mean():.2f}" if cnt else ""
            row["entries"] = f"{name} (n={cnt})"
            rows.append(row)
    out.append("Mean absolute position error for flagged entries (lower is better):\n\n" + md_table(pd.DataFrame(rows)) + "\n")
    return "\n".join(out)


def s_explain(lb, entries, frozen2) -> str:
    from pitlane.models.registry import build_model
    split = [s for s in ("fresh", "test_shift") if (lb["split"] == s).any() and (lb[(lb["split"] == s)]["label"] == "m06_L_r4_111").any()][0]
    rid = int(pr(lb, "m06_L_r4_111", split).index.max())
    cfg = frozen2["M06L"]["111"]
    m = build_model(cfg, entries)
    rows = entries.index[entries["race_id"] == rid].to_numpy()
    m.fit(int(rows.min()), t_ref=float(rid))
    comp = m.components(rows)
    d = entries.loc[rows, ["driverId", "pos", "season", "round", "raceName"]].reset_index(drop=True).join(comp.reset_index(drop=True))
    d = d.sort_values("u", ascending=False).head(8)
    wk = [c for c in comp.columns if c.startswith("wk_")]
    big = comp[wk].abs().mean().sort_values(ascending=False).head(4).index.tolist()
    t = d[["driverId", "a_driver", "b_car", "g_circuit", "weekend"] + big + ["u", "pos"]].round(2)
    out = [f"## 8. Explainability\n", f"Pre-race breakdown for **{d['raceName'].iloc[0]} {int(d['season'].iloc[0])}** (top 8 by score; `pos` is the actual result). "
           f"Each score is the exact sum of driver + car + circuit + weekend terms; the weekend term is split into its largest signals (`wk_*`).\n\n" + md_table(t) + "\n"]
    th = {k: frozen2["M08"]["S3"][k] for k in S.DEFAULT}
    cols = ["race_id", "driverId", "grid_norm", "stick", "circ_vol", "rain_fc"]
    lbd = pd.read_csv(D.LB_DNF)
    dn_row = lbd[lbd["split"] == split].iloc[-1]
    dn = pd.read_parquet(ROOT / dn_row["run_dir"] / "predictions.parquet")
    dn = dn[dn["race_id"] == rid][["driverId", "p_dnf"]]
    base = load_pred(lb, "m06_L_r4_111", split, rid)
    x = base.merge(dn, on="driverId").merge(entries[entries["race_id"] == rid][cols], on=["race_id", "driverId"])
    consts = {k: frozen2["M08"]["S3"][k] for k in ("stick_ref", "vol_mu", "vol_sd")}
    r = S.prepare(x, consts)[0]
    P = S.simulate_race(r["u"], r["p_dnf"], r["grid"], r["stick_c"], r["zvol"], r["rain"], th, S.SWITCHES["S3"], 3000, np.random.default_rng(0))
    order = np.argsort(P @ np.arange(1, len(P) + 1))
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    im = ax.imshow(P[order][:, :min(20, P.shape[1])], aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(order)), x["driverId"].to_numpy()[order], fontsize=8)
    ax.set_xticks(range(min(20, P.shape[1])), range(1, min(20, P.shape[1]) + 1), fontsize=8)
    ax.set_xlabel("finishing position"); ax.set_title(f"M08 S3 position probabilities, {d['raceName'].iloc[0]} {int(d['season'].iloc[0])}")
    fig.colorbar(im, ax=ax, label="probability")
    for i, o in enumerate(order):
        ax.plot(x["pos"].to_numpy()[o] - 1, i, "r.", ms=7)
    fig.tight_layout(); fig.savefig(FIG / "f6_position_probabilities.png", dpi=140); plt.close(fig)
    p_dnf = x.set_index("driverId")["p_dnf"].round(3)
    out.append("Retirement probability (M07) for the same race: " + ", ".join(f"{k} {v:.2f}" for k, v in p_dnf.sort_values(ascending=False).head(5).items()) +
               ".\n\n![positions](figures/f6_position_probabilities.png)\n\n_Red dots mark the actual finishing position._\n")
    return "\n".join(out)


def load_pred(lb, label, split, rid) -> pd.DataFrame:
    m = lb[(lb["label"] == label) & (lb["split"] == split)].iloc[0]
    p = pd.read_parquet(ROOT / m["run_dir"] / "predictions.parquet")
    return p[p["race_id"] == rid]


def s_summary(lb, ceil) -> str:
    def rho(label, split):
        p = pr(lb, label, split)
        return np.nan if p is None else p["spearman"].mean()
    rows = []
    for split in ["test_stable", "test_shift"]:
        rows.append({"split": SNAME[split].split(" (")[0], "quali order": f"{rho('quali', split):.3f}", "plan-1 latent_111": f"{rho('latent_111', split):.3f}",
                     "M06 H3 latent": f"{rho('m06_L_r4_111', split):.3f}", "M06 H3 LightGBM": f"{rho('m06_G_h3_111', split):.3f}",
                     "M08 S3": f"{rho('m08_S3', split):.3f}"})
    return ("## 9. Summary\n\nQuestions Q1-Q6 of `PLAN2.md` are answered by sections 1 (Q1), 3 (Q2), 4 (Q3, Q6), 5 (Q4) and 6 (Q5). "
            "Headline order accuracy (Spearman rho):\n\n" + md_table(pd.DataFrame(rows)) +
            "\n\nBefore running, `PLAN2.md` §8 predicted gains of 0.00-0.02 rho at H3 and larger gains in probability scores; compare with the tables above.\n")


def s_config(frozen2) -> str:
    import yaml
    return "## 10. Frozen plan-2 configurations (tuned on dev only)\n```yaml\n" + yaml.safe_dump(frozen2, sort_keys=False) + "```\n"


def make_report2(ceil: dict | None = None) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    lb = load_lb()
    frozen2 = load_yaml(CONFIGS / "frozen_plan2.yaml")
    entries = Q.load_all()
    ceil = ceil or json.loads((PLAN2 / "ceiling.json").read_text(encoding="utf-8"))
    parts = [s_overview(lb, frozen2, ceil), sec("1. A1", s_ceiling, lb, ceil), sec("2. Leaderboards", s_leaderboards, lb),
             sec("3. A2", s_horizon, lb), sec("4. M06", s_ladder, lb, entries), sec("5. M07", s_dnf, lb, entries),
             sec("6. M08", s_sim, lb, frozen2), sec("7. Subsets", s_subsets, lb, entries),
             sec("8. Explainability", s_explain, lb, entries, frozen2), sec("9. Summary", s_summary, lb, ceil),
             sec("10. Config", s_config, frozen2)]
    (PLAN2 / "REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    print("wrote", PLAN2 / "REPORT.md")
