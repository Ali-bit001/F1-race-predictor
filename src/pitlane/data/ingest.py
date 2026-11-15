"""Download (cached) Jolpica-F1 results + qualifying and build the tidy entries table.

Run:  python -m pitlane.data.ingest
Output: data/processed/entries.parquet  (one row per driver per race)

Traps handled (PLAN.md section 3): pit-lane starts (grid 0), grid vs qualifying kept distinct,
team rebrands mapped to one lineage, non-starters dropped, DSQ kept at the back of the order.
"""
from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

from pitlane.config import EXTERNAL, FIRST_SEASON, LAST_SEASON, PROCESSED, RAW

BASE = "https://api.jolpi.ca/ergast/f1"
PAGE = 100


def _get(url: str, retries: int = 6) -> dict:
    delay = 5.0
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RuntimeError(f"failed to fetch {url}")


def fetch_pages(kind: str, season: int) -> list[dict]:
    """Return list of Race dicts for a season, merging pages. Cached under data/raw/."""
    cache = RAW / f"{kind}_{season}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    races: dict[str, dict] = {}
    offset, total = 0, None
    while total is None or offset < total:
        d = _get(f"{BASE}/{season}/{kind}.json?limit={PAGE}&offset={offset}")["MRData"]
        total = int(d["total"])
        for race in d["RaceTable"]["Races"]:
            key = race["round"]
            if key not in races:
                races[key] = {k: v for k, v in race.items()
                              if k not in ("Results", "QualifyingResults", "SprintResults")}
                races[key]["rows"] = []
            races[key]["rows"] += race.get("Results") or race.get("QualifyingResults") or race.get("SprintResults") or []
        offset += PAGE
        time.sleep(0.6)
    out = list(races.values())
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def _t(s: str | None) -> float:
    """'1:23.456' -> seconds; missing -> NaN."""
    if not s:
        return np.nan
    m = re.match(r"(?:(\d+):)?(\d+\.\d+)$", s)
    return float(m.group(1) or 0) * 60 + float(m.group(2)) if m else np.nan


def build_entries() -> pd.DataFrame:
    lineage = pd.read_csv(EXTERNAL / "lineage.csv").set_index("constructorId")["lineage"].to_dict()
    circuits = pd.read_csv(EXTERNAL / "circuits.csv").set_index("circuitId")["type"].to_dict()
    rows = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        res = fetch_pages("results", season)
        qual = {q["round"]: q["rows"] for q in fetch_pages("qualifying", season)}
        for race in res:
            rnd = race["round"]
            qrows = qual.get(rnd, [])
            q1 = {x["Driver"]["driverId"]: _t(x.get("Q1")) for x in qrows}
            qpos = {x["Driver"]["driverId"]: int(x["position"]) for x in qrows if x.get("position", "").isdigit()}
            pole_q1 = np.nanmin(list(q1.values())) if q1 and not np.all(np.isnan(list(q1.values()))) else np.nan
            for order, x in enumerate(race["rows"], start=1):
                ptxt = x.get("positionText", "")
                if ptxt in ("W", "F", "E"):  # did not start / failed to qualify / excluded
                    continue
                did, cid = x["Driver"]["driverId"], x["Constructor"]["constructorId"]
                rows.append(dict(
                    season=season, round=int(rnd), date=race["date"], circuitId=race["Circuit"]["circuitId"],
                    raceName=race["raceName"], driverId=did, constructorId=cid,
                    lineage=lineage.get(cid, cid), grid=int(x["grid"]), finish_order=order,
                    positionText=ptxt, classified=ptxt.isdigit(), status=x.get("status", ""),
                    quali_pos=qpos.get(did, np.nan),
                    q1_gap=(q1[did] / pole_q1 - 1) if did in q1 and not np.isnan(pole_q1) else np.nan,
                    points=float(x.get("points", 0)),
                ))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["season", "round", "finish_order"]).reset_index(drop=True)
    # Re-number finishing order 1..n after dropping non-starters.
    df["race_id"] = df.groupby(["season", "round"]).ngroup()
    df["n"] = df.groupby("race_id")["driverId"].transform("size")
    df["pos"] = df.groupby("race_id").cumcount() + 1
    df["y"] = (df["pos"] - 1) / (df["n"] - 1).clip(lower=1)
    df["circuit_type"] = df["circuitId"].map(circuits).fillna("other")
    # Pit-lane start (grid 0) -> back of the grid; keep an explicit flag.
    df["pit_start"] = (df["grid"] == 0).astype(int)
    df.loc[df["grid"] == 0, "grid"] = df["n"]
    df["grid_norm"] = (df["grid"] - 1) / (df["n"] - 1).clip(lower=1)
    df["quali_pos"] = df["quali_pos"].fillna(df["grid"])
    df["quali_norm"] = (df["quali_pos"] - 1) / (df["n"] - 1).clip(lower=1)
    df["q1_gap"] = df["q1_gap"].clip(0, 0.10)
    df["q1_missing"] = df["q1_gap"].isna().astype(int)
    df["q1_gap"] = df["q1_gap"].fillna(0.10)
    df["dnf"] = (~df["classified"]).astype(int)
    df["race_t"] = df["race_id"]  # chronological ordinal
    # Sanity: classified finishers must sit at their official position.
    ok = df[df["classified"]]
    bad = (ok["positionText"].astype(int) != ok["pos"]).mean()
    print(f"classified rows whose official position != derived order: {bad:.3%}")
    return df


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = build_entries()
    df.to_parquet(PROCESSED / "entries.parquet", index=False)
    print(df.groupby("season").agg(races=("race_id", "nunique"), rows=("pos", "size")).T.to_string())
    print("wrote", PROCESSED / "entries.parquet")


if __name__ == "__main__":
    sys.exit(main())
