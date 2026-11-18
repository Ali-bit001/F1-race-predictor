"""PLAN3 R9 part 2: the M10 weekend-term column `m10_declared_x` (declared car-update components per car).

Per (season, round, constructor): the number of components the team declared as updated in the FIA "Car Presentation
Submissions" document of that race, centred within the race and divided by a fixed constant. It enters the LatentModel weekend
term as one extra x_col (lam_x stays at the LatentModel default, no tuning; Amendment 3 item 1).

Availability (the only leakage question; the feature is built from the race's own document and nothing else):
* H2 (with qualifying): the document counts only if it was published before the start of qualifying.
* H1 (before qualifying): only if published before the start of the first practice session (FP1).
* H0 (pre-weekend): the race's own document is never available -> no M10 column (the rung equals its reference).
Publication time = the LATER of the two UTC readings of the FIA 'CET' stamp (fixed UTC+1 / Europe-Paris wall clock). A race
with no session time, no document, an image-only page, an absent team or an NA count has the value MISSING, which becomes 0
(neutral) after centring, exactly as `LatentModel` fills NaN with 0.0 for every x_col (the way M09 left a missing Q1 offset).
No result, grid or later document is read. Missing documents of other races do not change the value of this race.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitlane.config import RAW, ROOT, load_yaml

COL = "m10_declared_x"
HORIZONS = ("H0", "H1", "H2")
CONFIG_PATH = ROOT / "methodologies" / "M10_upgrades" / "config.yaml"
OPENF1_DIR = RAW / "plan2" / "openf1"
SESSION_FOR = {"H1": "Practice 1", "H2": "Qualifying"}   # session whose start is the cutoff


def load_model_cfg(path: Path | str = CONFIG_PATH) -> dict:
    return load_yaml(path)["model"]


# ------------------------------------------------------------------------------------ session start times
def session_starts(races: pd.DataFrame, openf1_dir: Path | str = OPENF1_DIR, years: tuple[int, ...] = (2024, 2025, 2026, 2027)
                   ) -> pd.DataFrame:
    """(season, round, fp1_start, quali_start) as naive UTC timestamps, NaT when unknown.

    `races`: season, round, date (race date). The meeting is matched to the round by the Race session date (within one day),
    the same rule as `features.weekend.practice_features`. Sessions are read from the cached OpenF1 session lists
    (`sessions_<year>.json`); a missing file or a missing session gives NaT, and the document is then not used at that horizon."""
    r = races[["season", "round", "date"]].drop_duplicates(["season", "round"]).copy()
    r["date"] = pd.to_datetime(r["date"])
    rows = []
    for year in years:
        f = Path(openf1_dir) / f"sessions_{year}.json"
        if not f.exists():
            continue
        by_m: dict[int, list[dict]] = {}
        for s in json.loads(f.read_text(encoding="utf-8")) or []:
            by_m.setdefault(s["meeting_key"], []).append(s)
        ry = r[r["season"] == year]
        for ss in by_m.values():
            race_s = [s for s in ss if s["session_name"] == "Race"]
            if not race_s:
                continue
            d0 = _utc(race_s[0]["date_start"]).normalize()
            near = ry[(ry["date"] - d0).abs() <= pd.Timedelta(days=1)]
            if near.empty:
                continue
            row = dict(season=year, round=int(near.iloc[0]["round"]))
            for h, name in SESSION_FOR.items():
                hit = [s for s in ss if s["session_name"] == name and not s.get("is_cancelled", False)]
                row[f"{h}_cut"] = _utc(hit[0]["date_start"]) if hit else pd.NaT
            rows.append(row)
    out = pd.DataFrame(rows, columns=["season", "round", "H1_cut", "H2_cut"])
    return out.drop_duplicates(["season", "round"])


def _utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s).tz_convert("UTC").tz_localize(None)


# ------------------------------------------------------------------------------------ documents
def document_table(df: pd.DataFrame | None = None, emap: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (season, round, constructorId): declared (Int64, NA = missing), how, publication time (later reading).
    Joined through fia_docs.event_map() on event_id, never by name."""
    from pitlane.data import fia_docs as F
    if df is None:
        df = pd.read_parquet(ROOT / "data" / "interim" / "m10_car_presentations.parquet")
    emap = F.event_map() if emap is None else emap
    c = F.declared_counts(df)
    t = c.merge(emap[["event_id", "season", "round", "published_utc_fixed1", "published_utc_paris"]], on="event_id", how="inner",
                suffixes=("_doc", ""))
    t["published_utc"] = publication_time(t["published_utc_fixed1"], t["published_utc_paris"])
    return t[["season", "round", "constructorId", "declared", "how", "published_utc"]].reset_index(drop=True)


def publication_time(fixed1: pd.Series, paris: pd.Series) -> pd.Series:
    """The later of the two UTC readings. If the Paris reading does not exist (DST change), assume the larger possible shift (+1 h
    over fixed1), so that the time is never earlier than any reading."""
    paris = paris.where(paris.notna(), fixed1 + pd.Timedelta(hours=1))
    return pd.concat([fixed1, paris], axis=1).max(axis=1)


# ------------------------------------------------------------------------------------ the feature
def usable_documents(docs: pd.DataFrame, cuts: pd.DataFrame, horizon: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(documents usable at `horizon`, per-race availability table with the reason a race's document was dropped).
    Usable = published strictly before the cutoff of the horizon. A race without a cutoff is not used (conservative)."""
    if horizon not in HORIZONS:
        raise ValueError(horizon)
    if horizon == "H0":
        return docs.iloc[0:0].copy(), _avail(docs, cuts, "H0")
    av = _avail(docs, cuts, horizon)
    ok = av.loc[av["status"] == "used", ["season", "round"]]
    return docs.merge(ok, on=["season", "round"], how="inner"), av


def _avail(docs: pd.DataFrame, cuts: pd.DataFrame, horizon: str) -> pd.DataFrame:
    races = docs[["season", "round", "published_utc"]].drop_duplicates(["season", "round"])
    av = races.merge(cuts, on=["season", "round"], how="left")
    if horizon == "H0":
        av["status"] = "h0_never"
        return av
    cut = av[f"{horizon}_cut"]
    av["status"] = np.where(cut.isna(), "no_session_time", np.where(av["published_utc"] < cut, "used", "published_after_cutoff"))
    return av


def build_column(entries: pd.DataFrame, docs: pd.DataFrame, cuts: pd.DataFrame, horizon: str, scale: float,
                 centre: str = "race_mean") -> pd.Series:
    """`m10_declared_x` aligned to `entries` (needs season, round, constructorId). Missing -> NaN -> 0 in the model.
    centre='race_mean': mean over the constructors of that race with a usable count (cars with no count get 0, neutral)."""
    if centre != "race_mean":
        raise ValueError(f"unknown centring {centre!r}")
    use, _ = usable_documents(docs, cuts, horizon)
    use = use.dropna(subset=["declared"]).copy()
    use["declared"] = use["declared"].astype(float)
    use["x"] = (use["declared"] - use.groupby(["season", "round"])["declared"].transform("mean")) / float(scale)
    keyed = entries[["season", "round", "constructorId"]].merge(use[["season", "round", "constructorId", "x"]],
                                                                on=["season", "round", "constructorId"], how="left")
    return pd.Series(keyed["x"].to_numpy(float), index=entries.index, name=COL)


def add_column(entries: pd.DataFrame, horizon: str, cfg: dict | None = None, docs: pd.DataFrame | None = None,
               cuts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Copy of `entries` with the column set (all NaN at H0: the rung has no own-race information there)."""
    cfg = cfg or load_model_cfg()
    docs = document_table() if docs is None else docs
    cuts = session_starts(entries) if cuts is None else cuts
    out = entries.copy()
    out[COL] = build_column(entries, docs, cuts, horizon, cfg["scale"], cfg["centre"])
    return out


def coverage(entries: pd.DataFrame, docs: pd.DataFrame, cuts: pd.DataFrame, horizon: str, scale: float = 1.0) -> dict:
    """Coverage numbers for the dry run: share of car-races (rows of `entries` within the document era) with a usable count."""
    col = build_column(entries, docs, cuts, horizon, scale)
    _, av = usable_documents(docs, cuts, horizon)
    in_docs = entries[["season", "round"]].merge(docs[["season", "round"]].drop_duplicates().assign(_k=1), how="left")["_k"].eq(1).to_numpy()
    n_cars = int(in_docs.sum())
    return dict(horizon=horizon, car_races_with_document_race=n_cars,
                car_races_usable=int(col.notna().sum()),
                share_usable_of_document_races=float(col.notna().sum() / max(n_cars, 1)),
                share_usable_of_all_2024_26=float(col.notna().mean()),
                documents=int(len(av)),
                documents_used=int((av["status"] == "used").sum()),
                documents_dropped_after_cutoff=int((av["status"] == "published_after_cutoff").sum()),
                documents_dropped_no_session_time=int((av["status"] == "no_session_time").sum()))
