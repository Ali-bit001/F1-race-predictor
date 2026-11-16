"""A4 rolling-origin evaluation (PLAN3.md section 4): reusable pieces. The experiment (config, tables, figure,
run folders) lives in methodologies/A4_rolling_origin/run.py.

Nothing here changes a model or a metric: season ranges go through `walk_forward(split="seasons:a-b")`, and the
stored plan-1 predictions are re-scored with the unchanged `race_metrics`.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pitlane import pipeline as P
from pitlane.config import METHODS, RESULTS
from pitlane.evaluation.harness import summarize, walk_forward
from pitlane.evaluation.metrics import race_metrics
from pitlane.models.registry import build_model

COLS = ["spearman", "mae", "exact"]


def season_split(lo: int, hi: int | None = None) -> str:
    return f"seasons:{lo}-{lo if hi is None else hi}"


def stored_predictions(label: str) -> pd.DataFrame:
    """Plan-1 predictions for `label` over dev + test_stable + test_shift (2021-2026 round 20), one row per entry."""
    lb = pd.read_csv(RESULTS / "leaderboard.csv")
    parts = []
    for split in ("dev", "test_stable", "test_shift"):
        row = lb[(lb["label"] == label) & (lb["split"] == split)]
        assert len(row) == 1, (label, split, len(row))
        parts.append(pd.read_parquet(METHODS.parent / row.iloc[0]["run_dir"] / "predictions.parquet"))
    return pd.concat(parts, ignore_index=True)


def rescore(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-race metrics (score-only ones) from a predictions table, with the unchanged `race_metrics`."""
    rows = []
    for rid, g in preds.groupby("race_id", sort=True):
        m = race_metrics(g["pos"].to_numpy(), g["score"].to_numpy(), g["classified"].to_numpy())
        m.update(race_id=int(rid), season=int(g["season"].iloc[0]), round=int(g["round"].iloc[0]), n=len(g))
        rows.append(m)
    return pd.DataFrame(rows)


def add_half(per_race: pd.DataFrame) -> pd.DataFrame:
    """H1 = rounds up to half the races of that season in the table (rounds 1..n//2), H2 = the rest."""
    n = per_race.groupby("season")["round"].transform("max")
    out = per_race.copy()
    out["half"] = np.where(out["round"] <= n // 2, "H1", "H2")
    return out


def season_label(season: int) -> str:
    if season < 2021:
        return "not clean (hyper-parameters tuned on 2021-23, later data)"
    if season <= 2023:
        return "dev (hyper-parameters tuned on these seasons)"
    return "third look, no decision"


def season_table(per_race: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean per-race metrics per model x season x {H1, H2, all}."""
    rows = []
    for model, pr in per_race.items():
        pr = add_half(pr)
        for s, g in pr.groupby("season"):
            for half, h in (("H1", g[g["half"] == "H1"]), ("H2", g[g["half"] == "H2"]), ("all", g)):
                rows.append({"model": model, "season": int(s), "half": half, "n_races": len(h),
                             **{c: float(h[c].mean()) for c in COLS}, "status": season_label(int(s))})
    return pd.DataFrame(rows)


def paired_ci(x: np.ndarray, b: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (b, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ------------------------------------------------------------------------------------ rolling re-tune
def _eval(cfg: dict, entries: pd.DataFrame, split: str, refit_every: int) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _, per = walk_forward(entries, build_model(cfg, entries), split, refit_every=refit_every, n_samples=200)
    return {"cfg": cfg, **summarize(per)}


def tune_window(kind: str, cell: str, entries: pd.DataFrame, lo: int, hi: int, refit_every: int,
                n_jobs: int, max_configs: int) -> tuple[dict, pd.DataFrame]:
    """Plan-1 search space (`pipeline.grid_for`), walk-forward inside seasons lo..hi, best mean Spearman wins
    (first of the grid on exact ties, as `pipeline.tune`)."""
    grid = P.grid_for(kind, cell)
    assert len(grid) <= max_configs, (len(grid), max_configs)
    res = Parallel(n_jobs=n_jobs)(delayed(_eval)(c, entries, season_split(lo, hi), refit_every) for c in grid)
    tab = pd.DataFrame([{**r["cfg"], **{k: v for k, v in r.items() if k != "cfg"}} for r in res])
    best = res[int(tab["spearman"].idxmax())]["cfg"]
    return best, tab


def rolling_predict(entries: pd.DataFrame, cfg: dict, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Final-style scoring of one season: refit before every race (refit_every=1), 1000 samples."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return walk_forward(entries, build_model(cfg, entries), season_split(season), refit_every=1)


# ------------------------------------------------------------------------------------ gain over qualifying
GAIN_MODELS = ("latent_111", "gbm_100")
GAIN_METRICS = ("spearman", "exact")


def _cluster_boot(season_means: np.ndarray, weights: np.ndarray, b: int, seed: int) -> tuple[float, float, float]:
    """Resample whole seasons; statistic = race-weighted mean gain of the resampled seasons."""
    rng = np.random.default_rng(seed)
    k = len(season_means)
    idx = rng.integers(0, k, (b, k))
    w = weights[idx]
    m = (season_means[idx] * w).sum(1) / w.sum(1)
    return (float((season_means * weights).sum() / weights.sum()),
            float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def load_season_table_per_race(run_root, models=("quali",) + GAIN_MODELS) -> dict[str, pd.DataFrame]:
    """Per-race metrics of the season-table run folders (newest folder per model)."""
    out = {}
    for m in models:
        dirs = sorted(run_root.glob(f"*_season_table_{m}_all2015_2026"))
        assert dirs, m
        out[m] = pd.read_csv(dirs[-1] / "per_race_metrics.csv")
    return out


def _gain_frame(per: dict[str, pd.DataFrame], model: str, met: str) -> pd.DataFrame:
    q = per["quali"].set_index("race_id")
    m = per[model].set_index("race_id").loc[q.index]
    df = pd.DataFrame({"season": q["season"], "round": q["round"], "gain": m[met] - q[met]}).reset_index()
    return add_half(df)


def _block_label(name: str) -> str:
    if name == "2015-20":
        return "not clean"
    if name == "dev 2021-23":
        return "dev"
    if name == "2024-26":
        return "third look"
    if name.isdigit():
        return "not clean" if int(name) < 2021 else ("dev" if int(name) <= 2023 else "third look")
    return "mixed"


def gain_over_qualifying(per: dict[str, pd.DataFrame], b: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Paired per-race gain (model - qualifying order) with bootstrap 95% intervals.

    Race-level bootstrap inside a block of seasons; season-cluster bootstrap where marked. Also the H2-minus-H1
    gain difference (by season) and the linear trend of season gain on season (OLS, normal-approx interval)."""
    from scipy import stats
    rows = []

    def add(model, met, block, kind, stat, est, lo, hi, n, label, p=np.nan):
        rows.append({"model": model, "metric": met, "block": block, "bootstrap": kind, "statistic": stat,
                     "estimate": est, "ci_lo": lo, "ci_hi": hi, "p_value": p, "n_races": n, "label": label})

    for model in GAIN_MODELS:
        for met in GAIN_METRICS:
            g = _gain_frame(per, model, met)
            seasons = sorted(g["season"].unique())
            blocks = {str(s): [s] for s in seasons}
            blocks.update({"2015-20": [s for s in seasons if s <= 2020], "dev 2021-23": [2021, 2022, 2023],
                           "2024-26": [s for s in seasons if s >= 2024]})
            for name, ss in blocks.items():
                x = g.loc[g["season"].isin(ss), "gain"].to_numpy()
                add(model, met, name, "race", "gain", *paired_ci(x, b, seed), len(x), _block_label(name))
            add(model, met, "all seasons", "race", "gain", *paired_ci(g["gain"].to_numpy(), b, seed), len(g), "mixed")
            sm = g.groupby("season")["gain"].agg(["mean", "size"])
            for name, drop in (("all seasons", []), ("all except 2017, 2023", [2017, 2023])):
                s = sm.drop(index=drop)
                add(model, met, name, "season cluster", "gain",
                    *_cluster_boot(s["mean"].to_numpy(), s["size"].to_numpy(), b, seed), int(s["size"].sum()), "mixed")
            hs = g.groupby(["season", "half"])["gain"].mean().unstack()
            d = (hs["H2"] - hs["H1"]).dropna()
            for name, sel in (("all seasons", d), ("excluding dev", d[~d.index.isin([2021, 2022, 2023])])):
                n = int(g["season"].isin(sel.index).sum())
                add(model, met, name, "season cluster", "H2 minus H1 gain", *paired_ci(sel.to_numpy(), b, seed), n, "mixed")
            lr = stats.linregress(sm.index.to_numpy(dtype=float), sm["mean"].to_numpy())
            add(model, met, "all seasons", "OLS on season means", "slope per season", float(lr.slope),
                float(lr.slope - 1.96 * lr.stderr), float(lr.slope + 1.96 * lr.stderr), int(sm["size"].sum()), "mixed",
                float(lr.pvalue))
    return pd.DataFrame(rows)


def gain_figure(tab: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    col = {"latent_111": "tab:blue", "gbm_100": "tab:orange"}
    for ax, (met, lab) in zip(axes, [("spearman", "gain in Spearman rho"), ("exact", "gain in exact-position rate")]):
        for k, model in enumerate(GAIN_MODELS):
            t = tab[(tab["model"] == model) & (tab["metric"] == met) & tab["block"].str.fullmatch(r"\d{4}")]
            x = t["block"].astype(int).to_numpy() + (k - 0.5) * 0.25
            ax.errorbar(x, t["estimate"], yerr=[t["estimate"] - t["ci_lo"], t["ci_hi"] - t["estimate"]],
                        fmt="o", ms=3, capsize=2, color=col[model], label=model)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvspan(2014.5, 2020.5, color="0.92", zorder=0)
        ax.axvspan(2023.5, 2026.5, color="#ffe9c9", zorder=0)
        ax.set_ylabel(lab + " minus qualifying", fontsize=9)
        ax.grid(alpha=.3)
    axes[0].legend(loc="lower right", ncol=2)
    axes[1].set_xlabel("season (95% race-level bootstrap interval; grey = not clean, white = dev, orange = third look)")
    fig.suptitle("A4: paired per-race gain over qualifying order, by season")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
