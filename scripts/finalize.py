"""After a full run: append an auto entry to experiments/LOG.md, commit results, push if a remote exists."""
from __future__ import annotations

import subprocess
from datetime import datetime

import pandas as pd

from pitlane.config import CONFIGS, ROOT, RESULTS, load_yaml


def sh(*a: str) -> str:
    r = subprocess.run(a, cwd=ROOT, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def main() -> None:
    lb = pd.read_csv(RESULTS / "leaderboard.csv")
    fz = load_yaml(CONFIGS / "frozen.yaml")
    lines = [f"\n## {datetime.now():%Y-%m-%d} — Full pipeline run (auto-generated)",
             "- Methodology / WP: WP1-WP8 via `scripts/run_all.py`",
             f"- Frozen config hash: `{fz.get('hash')}` (tuned on dev 2021-23 only; test blocks run once afterwards)",
             "- Result (mean per-race Spearman rho, higher is better):"]
    for s in ["dev", "test_stable", "test_shift"]:
        d = lb[lb["split"] == s]
        if d.empty:
            continue
        top = d.sort_values("spearman", ascending=False).head(3)
        grid = d[d["label"] == "grid"]["spearman"]
        lines.append(f"  - {s}: grid baseline {grid.iloc[0]:.3f}; top-3: " +
                     ", ".join(f"{r.label} {r.spearman:.3f}" for r in top.itertuples()))
    lines += ["- Interpretation: see `results/REPORT.md` (paired bootstrap CIs and factorial effects).",
              "- Deviations from PLAN.md: FastF1 practice pace and wet-weather flags not yet used (deferred); "
              "sprint results not used; circuit types are analyst judgement (`data/external/circuits.csv`); "
              "no git tag was created for the freeze, the config hash above identifies it."]
    with open(ROOT / "experiments" / "LOG.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sh("git", "add", "results", "methodologies", "configs", "experiments")
    print(sh("git", "commit", "-m",
             "Add full-run results: leaderboard, report, figures, frozen configs\n\n"
             "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"))
    if sh("git", "remote"):
        print(sh("git", "push", "-u", "origin", "HEAD"))
    else:
        print("no git remote configured; skipping push")


if __name__ == "__main__":
    main()
