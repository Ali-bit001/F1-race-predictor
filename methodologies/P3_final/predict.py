"""Predict one race with the frozen plan-3 model (PLAN3 R14). Needs nothing but frozen_config.yaml and the data on disk.

    python methodologies/P3_final/predict.py --season 2027 --round 1 [--horizon H2|H1|both] [--entries PATH] [--out PATH] [--no-sim]

What it does, in order:
  1. checks the SHA-256 of frozen_config.yaml against FREEZE.md and stops if they differ (the config was edited after the freeze);
  2. loads the plan-2 entries table (`pipeline2.load_all()`, or `--entries PATH`, a parquet with the same columns) and finds the race;
  3. drops every row after the race and blanks the race's own outcome columns (`final.prepare_pre_race`), builds the car race-pace
     history from the laps of EARLIER races, re-fits the model before each of the last 15 races and the race itself (the plan-1
     walk-forward, warm-started), and predicts the race;
  4. H2 also runs the retirement model (fitted on earlier races) and the re-fitted S3 simulator: win / podium / points probabilities
     and the D0 order (sort by -E[position]).
Nothing is re-tuned. The race's own results, laps and later races are never read for its prediction.

What it does NOT do: build the entries row of a race that has not been ingested. The row needs the pre-race columns (grid,
qualifying, weather, circuit, lineage). `data/processed/entries2.parquet` is built by the plan-1/2 ingest after a race, so for a live
race supply `--entries` with the pre-race row filled in (outcome columns may hold anything; they are blanked).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pitlane.config import ROOT, load_yaml
from pitlane.plan3 import final as F

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "frozen_config.yaml"
FREEZE = HERE / "FREEZE.md"


def load_entries(path: str | None) -> pd.DataFrame:
    if path:
        return pd.read_parquet(path).reset_index(drop=True)
    from pitlane.plan2 import pipeline2 as Q
    return Q.load_all()


def predict(season: int, rnd: int, horizons: list[str], entries: pd.DataFrame, cfg: dict, with_sim: bool = True) -> dict[str, pd.DataFrame]:
    hit = entries[(entries["season"] == season) & (entries["round"] == rnd)]
    if hit.empty:
        raise LookupError(f"no entries rows for season {season} round {rnd}: build the pre-race entries first (see the module docstring)")
    race_id = int(hit["race_id"].iloc[0])
    return {h: F.predict_race(entries, race_id, cfg, h, with_sim=with_sim) for h in horizons}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--horizon", choices=["H2", "H1", "both"], default="H2")
    ap.add_argument("--entries", default=None, help="parquet with the plan-2 entries columns (default: pipeline2.load_all())")
    ap.add_argument("--out", default=None, help="output parquet (default: methodologies/P3_final/predictions/pred_<season>_<round>.parquet)")
    ap.add_argument("--no-sim", action="store_true", help="skip the retirement model and the simulator")
    a = ap.parse_args(argv)
    try:
        sha = F.verify_freeze(CONFIG, FREEZE)
    except (RuntimeError, ValueError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    cfg = load_yaml(CONFIG)
    horizons = ["H2", "H1"] if a.horizon == "both" else [a.horizon]
    try:
        outs = predict(a.season, a.round, horizons, load_entries(a.entries), cfg, with_sim=not a.no_sim)
    except LookupError as e:
        print(f"CANNOT PREDICT: {e}", file=sys.stderr)
        return 3
    res = pd.concat(outs.values(), ignore_index=True)
    res["config_sha256"] = sha
    res["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = Path(a.out) if a.out else HERE / "predictions" / f"pred_{a.season}_{a.round:02d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out, index=False)
    pd.set_option("display.width", 200)
    for h, d in outs.items():
        show = [c for c in ("driverId", "grid", "pred_place", "score", "p_win", "p_podium", "p_points", "p_dnf") if c in d.columns]
        print(f"\n{h}  season {a.season} round {a.round}  (frozen config {sha[:12]})")
        print(d.sort_values("pred_place")[show].round(3).to_string(index=False))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
