"""After a plan-2 run: append an auto entry to experiments/LOG.md, commit the results, push if a remote exists."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime

import pandas as pd

from pitlane.config import CONFIGS, PLAN2, ROOT, load_yaml


def sh(*a: str) -> str:
    r = subprocess.run(a, cwd=ROOT, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def main() -> None:
    lb = pd.concat([pd.read_csv(ROOT / "results" / "leaderboard.csv"), pd.read_csv(PLAN2 / "leaderboard.csv")],
                   ignore_index=True).drop_duplicates(["label", "split"], keep="last")
    fz = load_yaml(CONFIGS / "frozen_plan2.yaml")
    ceil = json.loads((PLAN2 / "ceiling.json").read_text(encoding="utf-8"))
    lines = [f"\n## {datetime.now():%Y-%m-%d} — Plan 2 full run (auto-generated)",
             "- Methodology / WP: P0-P8 via `scripts/run_plan2.py` (A1, A2, M06, M07, M08)",
             f"- Frozen config hash: `{fz.get('hash')}` (tuned on dev 2021-23 only; test blocks are a second look, see PLAN2.md section 5)",
             f"- A1 ceiling (dev): model rho {ceil['model_rho']:.3f}, self-consistency {ceil['self_consistency_rho']:.3f}, "
             f"retirement oracle {ceil['retirement_oracle_rho']:.3f}; pace-oracle gap on finishers "
             f"{ceil['pace_gap_fin'] if ceil['pace_gap_fin'] is None else round(ceil['pace_gap_fin'], 3)} -> "
             f"{'order accuracy saturated (gate met)' if ceil['saturated'] else 'not saturated (gate not met)'}",
             "- Result (mean per-race Spearman rho, higher is better):"]
    for s in ["dev", "test_stable", "test_shift", "fresh"]:
        d = lb[lb["split"] == s]
        if d.empty:
            continue
        top = d.sort_values("spearman", ascending=False).head(4)
        ref = d[d["label"] == "quali"]["spearman"]
        lines.append(f"  - {s}: qualifying order {ref.iloc[0]:.3f}; top-4: " + ", ".join(f"{r.label} {r.spearman:.3f}" for r in top.itertuples()))
    m08 = lb[(lb["label"].str.startswith("m08_")) & (lb["split"] == "test_stable")]
    if len(m08):
        lines.append("  - M08 RPS on test_stable: " + ", ".join(f"{r.label} {r.rps:.4f}" for r in m08.sort_values("label").itertuples()))
    lines += ["- Interpretation: see `results/plan2/REPORT.md` (information ladder, leave-one-out verdicts, DNF model, simulator ladder).",
              "- Deviations from PLAN2.md: FastF1 is blocked (HTTP 401 without an F1 login), so practice laps and session weather come from "
              "OpenF1 and exist for 2023+ only; practice is judged on dev-2023 plus test; race laps for the pace oracle were fetched for 2023 only "
              "(Jolpica rate limit); the race-day forecast comes from Open-Meteo's historical-forecast archive (2020+); every ladder rung is tuned separately "
              "on dev; no git tag was created for the freeze (the config hash identifies it)."]
    with open(ROOT / "experiments" / "LOG.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sh("git", "add", "results", "methodologies", "configs", "experiments")
    print(sh("git", "commit", "-m",
             "Add plan-2 results: ceiling, horizon, information ladder, DNF model, Monte Carlo simulator\n\n"
             "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"))
    if sh("git", "remote"):
        print(sh("git", "push", "-u", "origin", "HEAD"))
    else:
        print("no git remote configured; skipping push")


if __name__ == "__main__":
    main()
