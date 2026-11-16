"""Plan-3 R0 ingestion (PLAN3.md section 3): race laps, pit stops and stints, 2014-2026.

Sources: Jolpica laps 2014-2022 (2023 is reused from data/raw/plan2/laps), Jolpica pit stops 2014-2026,
OpenF1 laps/stints/drivers/pit 2023-2026. One raw JSON file per race per endpoint under data/raw/plan3/.
Every step is resumable (existing files are skipped). A 429, a timeout or any other transient failure is
retried with backoff and, if it still fails, recorded in data/raw/plan3/failures.json and NOT cached, so
a later run retries it. Only a genuine empty answer for a race that has been run is cached as empty,
and it is listed under "empty". Races that have not been run yet are never requested.

Settings: configs/ingest3.yaml. Script: scripts/ingest_plan3.py. Loader: load_tables().
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from pitlane.config import CONFIGS, DATA, RAW, ROOT, load_yaml

RAW3 = RAW / "plan3"
RAW2_LAPS = RAW / "plan2" / "laps"
INTERIM3 = DATA / "interim" / "plan3"
FAIL_FILE = RAW3 / "failures.json"
REQ_FILE = RAW3 / "jolpica_requests.txt"
LOG_FILE = ROOT / "logs" / "ingest3.log"
OPENF1_LAP_FIELDS = ["date_start", "driver_number", "lap_number", "lap_duration", "duration_sector_1",
                     "duration_sector_2", "duration_sector_3", "i1_speed", "i2_speed", "st_speed", "is_pit_out_lap"]
log = logging.getLogger("ingest3")


def cfg() -> dict:
    return load_yaml(CONFIGS / "ingest3.yaml")


def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if log.handlers:
        return
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)


# ------------------------------------------------------------------------------------ files
def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_fail() -> dict:
    return read_json(FAIL_FILE) if FAIL_FILE.exists() else {"failed": {}, "empty": {}}


def record(kind: str, key: str, **info: Any) -> None:
    """kind 'failed' (will be retried on the next run) or 'empty' (cached empty answer for a run race)."""
    d = _load_fail()
    d[kind][key] = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **info}
    write_json(FAIL_FILE, d)


def clear(key: str) -> None:
    d = _load_fail()
    if d["failed"].pop(key, None) is not None:
        write_json(FAIL_FILE, d)


# ------------------------------------------------------------------------------------ HTTP
class Client:
    """GET with per-host pacing and backoff. get() returns (state, json): state 'ok', 'missing' (HTTP 404)
    or 'fail' (all retries used or a non-retryable status). Nothing is cached here."""

    def __init__(self, c: dict):
        import requests
        self.c, self.s = c, requests.Session()
        self.s.headers["User-Agent"] = c["user_agent"]
        self.last: dict[str, float] = {}

    def _jolpica_wait(self, gap: float) -> None:
        """Sliding-window guard over a request log shared by all processes (persists across runs)."""
        cap = int(self.c["jolpica"]["max_per_hour"])
        while True:
            now = time.time()
            ts = []
            if REQ_FILE.exists():
                ts = [float(x) for x in REQ_FILE.read_text().split() if now - float(x) < 3600]
            wait = (max(ts) + gap - now) if ts else 0.0
            if len(ts) >= cap:
                wait = max(wait, sorted(ts)[len(ts) - cap] + 3601 - now)
            if wait <= 0:
                break
            time.sleep(min(wait, 60))
        RAW3.mkdir(parents=True, exist_ok=True)
        with open(REQ_FILE, "a") as f:
            f.write(f"{time.time():.2f}\n")

    def _wait(self, host: str, gap: float) -> None:
        if host == "jolpica":
            self._jolpica_wait(gap)
        else:
            w = self.last.get(host, 0) + gap - time.time()
            if w > 0:
                time.sleep(w)
        self.last[host] = time.time()

    def get(self, url: str, host: str, gap: float) -> tuple[str, Any]:
        import requests
        r_ = self.c["retry"]
        delay = float(r_["first_delay_s"])
        for attempt in range(int(r_["max_tries"])):
            self._wait(host, gap)
            try:
                r = self.s.get(url, timeout=r_["timeout_s"])
            except requests.RequestException as e:
                log.warning("network error %s (try %d): %s", url, attempt + 1, type(e).__name__)
            else:
                if r.status_code == 200:
                    try:
                        return "ok", r.json()
                    except ValueError:
                        log.warning("bad json %s", url)
                elif r.status_code == 404:
                    return "missing", None
                elif r.status_code in (429, 500, 502, 503, 504):
                    ra = r.headers.get("Retry-After")
                    delay = max(delay, float(ra)) if ra and ra.replace(".", "").isdigit() else delay
                    log.warning("HTTP %d %s (try %d), sleeping %.0fs", r.status_code, url, attempt + 1, delay)
                else:
                    log.error("HTTP %d %s (not retried)", r.status_code, url)
                    return "fail", f"HTTP {r.status_code}"
            time.sleep(delay)
            delay = min(delay * 2, float(r_["max_delay_s"]))
        return "fail", "retries exhausted"


# ------------------------------------------------------------------------------------ race lists
def season_races(season: int) -> list[dict]:
    """Races of a season from the cached Jolpica results (data/raw/results_<season>.json)."""
    f = RAW / f"results_{season}.json"
    return read_json(f) if f.exists() else []


def race_start(race: dict) -> datetime:
    t = (race.get("time") or "13:00:00Z").replace("Z", "")
    return datetime.fromisoformat(f"{race['date']}T{t}").replace(tzinfo=timezone.utc)


def is_run(race: dict, c: dict, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    return race_start(race) + timedelta(hours=float(c["race_done_after_h"])) < now


def run_races(season: int, c: dict) -> list[dict]:
    return [r for r in season_races(season) if is_run(r, c)]


# ------------------------------------------------------------------------------------ Jolpica
def jolpica_paged(cl: Client, c: dict, url: str, gap: float, pick: Callable[[dict], list]) -> tuple[str, Any]:
    """Fetch all pages of a Jolpica endpoint; pick(MRData) returns the rows of one page.
    Returns ('ok', rows) or ('fail', reason)."""
    page, rows, offset, total = int(c["jolpica"]["page"]), [], 0, None
    while total is None or offset < total:
        st, d = cl.get(f"{url}{'&' if '?' in url else '?'}limit={page}&offset={offset}", "jolpica", gap)
        if st != "ok":
            return "fail", d if st == "fail" else "HTTP 404"
        d = d["MRData"]
        total = int(d["total"])
        rows += pick(d)
        offset += page
    return "ok", rows


def _laps_rows(d: dict) -> list:
    out = []
    for race in d["RaceTable"]["Races"]:
        for lap in race.get("Laps", []):
            out += [[int(lap["number"]), t["driverId"], t.get("time"), t.get("position")] for t in lap["Timings"]]
    return out


def _pit_rows(d: dict) -> list:
    out = []
    for race in d["RaceTable"]["Races"]:
        out += [[p["driverId"], int(p["lap"]), int(p["stop"]), p.get("time"), p.get("duration")]
                for p in race.get("PitStops", [])]
    return out


def laps_path(season: int, rnd: int) -> Path:
    return RAW3 / "jolpica_laps" / f"laps_{season}_{rnd}.json"


def pits_path(season: int, rnd: int) -> Path:
    return RAW3 / "jolpica_pits" / f"pits_{season}_{rnd}.json"


def _fetch_jolpica(cl: Client, c: dict, kind: str, season: int, rnd: int, path: Path, gap: float) -> str:
    key = f"jolpica_{kind}/{season}/{rnd}"
    if path.exists():
        return "cached"
    ep, pick = ("laps", _laps_rows) if kind == "laps" else ("pitstops", _pit_rows)
    st, rows = jolpica_paged(cl, c, f"{c['jolpica']['base']}/{season}/{rnd}/{ep}.json", gap, pick)
    if st != "ok":
        record("failed", key, reason=str(rows))
        log.error("FAILED %s: %s", key, rows)
        return "failed"
    write_json(path, rows)
    clear(key)
    if not rows:
        record("empty", key, reason="Jolpica returned no rows for a race that has been run")
        log.warning("EMPTY %s", key)
        return "empty"
    log.info("saved %s (%d rows)", key, len(rows))
    return "downloaded"


def ingest_jolpica_laps(cl: Client, c: dict, seasons: list[int] | None = None) -> dict[str, int]:
    seasons = seasons if seasons is not None else c["jolpica"]["laps_seasons"]
    n: dict[str, int] = {}
    for s in seasons:
        if s in c["jolpica"]["reuse_plan2_laps_seasons"]:
            continue
        for race in run_races(s, c):
            rnd = int(race["round"])
            r = _fetch_jolpica(cl, c, "laps", s, rnd, laps_path(s, rnd), c["jolpica"]["laps_gap_s"])
            n[r] = n.get(r, 0) + 1
        log.info("jolpica laps %d done: %s", s, n)
    return n


def ingest_jolpica_pits(cl: Client, c: dict) -> dict[str, int]:
    n: dict[str, int] = {}
    for s in c["jolpica"]["pit_seasons"]:
        for race in run_races(s, c):
            rnd = int(race["round"])
            r = _fetch_jolpica(cl, c, "pits", s, rnd, pits_path(s, rnd), c["jolpica"]["pits_gap_s"])
            n[r] = n.get(r, 0) + 1
        log.info("jolpica pit stops %d done: %s", s, n)
    return n


# ------------------------------------------------------------------------------------ OpenF1
def openf1_path(ep: str, season: int, rnd: int) -> Path:
    return RAW3 / "openf1" / f"{ep}_{season}_{rnd}.json"


def openf1_sessions(cl: Client, c: dict, year: int) -> list[dict] | None:
    f = RAW3 / "openf1" / f"sessions_{year}.json"
    fresh = f.exists() and (year < datetime.now().year or
                            time.time() - f.stat().st_mtime < float(c["openf1"]["sessions_max_age_h"]) * 3600)
    if fresh:
        return read_json(f)
    st, d = cl.get(f"{c['openf1']['base']}/sessions?year={year}", "openf1", c["openf1"]["gap_s"])
    if st != "ok" or not isinstance(d, list):
        record("failed", f"openf1_sessions/{year}", reason=str(d))
        return read_json(f) if f.exists() else None
    write_json(f, d)
    return d


def match_race_sessions(sessions: list[dict], races: list[dict]) -> dict[int, dict]:
    """Jolpica round -> OpenF1 Race session, matched by date (within one day), not by order."""
    out: dict[int, dict] = {}
    for s in sessions:
        if s.get("session_name") != "Race" or s.get("is_cancelled"):
            continue
        d0 = datetime.fromisoformat(s["date_start"]).astimezone(timezone.utc).date()
        near = [r for r in races if abs((datetime.fromisoformat(r["date"]).date() - d0).days) <= 1]
        if near:
            out[int(near[0]["round"])] = s
    return out


def _fetch_openf1_one(cl: Client, c: dict, ep: str, season: int, rnd: int, sk: int) -> str:
    key = f"openf1_{ep}/{season}/{rnd}"
    path = openf1_path(ep, season, rnd)
    if path.exists():
        return "cached"
    st, d = cl.get(f"{c['openf1']['base']}/{ep}?session_key={sk}", "openf1", c["openf1"]["gap_s"])
    if st == "ok" and isinstance(d, dict):                    # e.g. {"detail": "No results found."}
        st, d = ("missing", None) if "no results" in str(d).lower() else ("fail", str(d)[:200])
    if st == "fail":
        record("failed", key, reason=str(d))
        log.error("FAILED %s: %s", key, d)
        return "failed"
    rows = d if st == "ok" else []
    if ep == "laps":
        rows = [{k: r.get(k) for k in OPENF1_LAP_FIELDS} for r in rows]
    write_json(path, rows)
    clear(key)
    if not rows:
        record("empty", key, reason="OpenF1 had no results for a race that has been run (session_key "
                                    f"{sk})")
        log.warning("EMPTY %s", key)
        return "empty"
    log.info("saved %s (%d rows)", key, len(rows))
    return "downloaded"


def ingest_openf1(cl: Client, c: dict, years: list[int] | None = None) -> dict[str, int]:
    n: dict[str, int] = {}
    now = datetime.now(timezone.utc)
    for y in years or c["openf1"]["seasons"]:
        sess = openf1_sessions(cl, c, y)
        if sess is None:
            continue
        races = run_races(y, c)
        match = match_race_sessions(sess, races)
        for race in races:
            rnd = int(race["round"])
            s = match.get(rnd)
            if s is None:
                record("failed", f"openf1_session/{y}/{rnd}", reason="no OpenF1 Race session matched by date")
                continue
            if datetime.fromisoformat(s["date_end"]) > now - timedelta(hours=1):
                continue
            for ep in c["openf1"]["endpoints"]:
                r = _fetch_openf1_one(cl, c, ep, y, rnd, int(s["session_key"]))
                n[r] = n.get(r, 0) + 1
        log.info("openf1 %d done: %s", y, n)
    return n


# ------------------------------------------------------------------------------------ loader
def _t(s: str | None) -> float:
    """'1:23.456' / '22.123' -> seconds; missing or malformed -> NaN."""
    if not s:
        return np.nan
    try:
        parts = [float(x) for x in str(s).split(":")]
        return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    except ValueError:
        return np.nan


def _number_map(season: int) -> dict[int, dict[int, tuple[str, str]]]:
    """round -> {car number: (driverId, code)} from the cached Jolpica results (not permanentNumber)."""
    out: dict[int, dict[int, tuple[str, str]]] = {}
    for race in season_races(season):
        m = {}
        for x in race["rows"]:
            if str(x.get("number", "")).isdigit():
                m[int(x["number"])] = (x["Driver"]["driverId"], (x["Driver"].get("code") or "").upper())
        out[int(race["round"])] = m
    return out


def _jolpica_laps_df(season: int, rnd: int) -> pd.DataFrame | None:
    f = laps_path(season, rnd)
    if not f.exists():
        f2 = RAW2_LAPS / f"laps_{season}_{rnd}.json"          # plan-2 cache (2023): [lap, driverId, time]
        if not f2.exists():
            return None
        f = f2
    rows = read_json(f)
    if not rows:
        return None
    d = pd.DataFrame([r + [None] * (4 - len(r)) for r in rows], columns=["lap", "driverId", "t", "position"])
    d["lap_time_s"] = d["t"].map(_t)
    d["position"] = pd.to_numeric(d["position"], errors="coerce")
    d["season"], d["round"], d["source"] = season, rnd, "jolpica"
    return d.drop(columns="t")


def _openf1_frames(season: int, rnd: int, nums: dict[int, tuple[str, str]], unmatched: list) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for ep in ("laps", "stints", "pit"):
        f = openf1_path(ep, season, rnd)
        if not f.exists():
            continue
        d = pd.DataFrame(read_json(f))
        if d.empty:
            continue
        d["driverId"] = d["driver_number"].map(lambda n: nums.get(int(n), (None, ""))[0])
        for n in sorted(set(d.loc[d["driverId"].isna(), "driver_number"])):
            unmatched.append(dict(season=season, round=rnd, endpoint=ep, driver_number=int(n)))
        d = d.dropna(subset=["driverId"])
        d["season"], d["round"], d["source"] = season, rnd, "openf1"
        out[ep] = d
    f = openf1_path("drivers", season, rnd)
    if f.exists():                                             # acronym vs Jolpica code cross-check
        for r in read_json(f):
            code = nums.get(int(r["driver_number"]), (None, ""))[1]
            if code and str(r.get("name_acronym", "")).upper() != code:
                unmatched.append(dict(season=season, round=rnd, endpoint="drivers", driver_number=int(r["driver_number"]),
                                      note=f"acronym {r.get('name_acronym')} != Jolpica code {code}"))
    return out


def _pits_jolpica_df(season: int, rnd: int, margin: float) -> pd.DataFrame | None:
    f = pits_path(season, rnd)
    if not f.exists():
        return None
    rows = read_json(f)
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=["driverId", "lap", "stop", "time_of_day", "duration"])
    d["duration_s"] = d["duration"].map(_t)
    d["red_flag_suspect"] = d["duration_s"] > d["duration_s"].median() + margin
    d["season"], d["round"], d["source"] = season, rnd, "jolpica"
    return d.drop(columns="duration")


def load_tables(c: dict | None = None, write: bool = True) -> dict[str, Any]:
    """Tidy tables from whatever has been downloaded so far (safe to re-run as files arrive).
    laps: season, round, driverId, lap, lap_time_s, position (Jolpica only), source, plus OpenF1 sector,
    speed, pit-out and start-time columns. A race can have both sources (2023-2026); filter on `source`
    or use dedupe_laps(). pits: one row per stop (Jolpica `duration_s` is total pit-lane time; OpenF1 has
    lane_duration_s / stop_duration_s). stints: OpenF1 only. Keyed by Jolpica driverId."""
    c = c or cfg()
    laps, pits, stints, unmatched = [], [], [], []
    margin = float(c["pits"]["red_flag_margin_s"])
    for season in range(2014, 2027):
        nums = _number_map(season) if season >= 2023 else {}
        for race in season_races(season):
            rnd = int(race["round"])
            d = _jolpica_laps_df(season, rnd)
            if d is not None:
                laps.append(d)
            p = _pits_jolpica_df(season, rnd, margin)
            if p is not None:
                pits.append(p)
            if season >= 2023:
                fr = _openf1_frames(season, rnd, nums.get(rnd, {}), unmatched)
                if "laps" in fr:
                    o = fr["laps"].rename(columns={"lap_number": "lap", "lap_duration": "lap_time_s", "duration_sector_1": "s1_s",
                                                   "duration_sector_2": "s2_s", "duration_sector_3": "s3_s",
                                                   "is_pit_out_lap": "pit_out"})
                    laps.append(o.drop(columns="driver_number"))
                if "stints" in fr:
                    stints.append(fr["stints"].drop(columns=[x for x in ("driver_number", "meeting_key", "session_key") if x in fr["stints"]]))
                if "pit" in fr:
                    o = fr["pit"].rename(columns={"lap_number": "lap", "lane_duration": "lane_duration_s",
                                                  "stop_duration": "stop_duration_s", "date": "time_of_day"})
                    keep = [x for x in ["driverId", "lap", "lane_duration_s", "stop_duration_s", "time_of_day", "season", "round", "source"] if x in o]
                    pits.append(o[keep])
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    laps_df, pits_df, stints_df = cat(laps), cat(pits), cat(stints)
    if len(laps_df):
        laps_df = laps_df.sort_values(["season", "round", "source", "driverId", "lap"]).reset_index(drop=True)
    if len(pits_df):
        pits_df = pits_df.sort_values(["season", "round", "source", "driverId", "lap"]).reset_index(drop=True)
    un = pd.DataFrame(unmatched)
    if write:
        INTERIM3.mkdir(parents=True, exist_ok=True)
        for name, df in (("laps", laps_df), ("pits", pits_df), ("stints", stints_df)):
            if len(df):
                df.to_parquet(INTERIM3 / f"{name}.parquet", index=False)
        write_json(INTERIM3 / "unmatched_drivers.json", unmatched)
    return {"laps": laps_df, "pits": pits_df, "stints": stints_df, "unmatched": un}


def dedupe_laps(laps: pd.DataFrame, prefer: str = "jolpica") -> pd.DataFrame:
    """One source per race: `prefer` where the race has it, the other source otherwise."""
    has = laps.groupby(["season", "round"])["source"].agg(set)
    pick = has.map(lambda s: prefer if prefer in s else sorted(s)[0]).rename("pick").reset_index()
    m = laps.merge(pick, on=["season", "round"])
    return m[m["source"] == m["pick"]].drop(columns="pick").reset_index(drop=True)


def lap_agreement(laps: pd.DataFrame, tol: float = 0.05, season: int = 2023) -> pd.DataFrame:
    """Per race: share of driver-laps with both Jolpica and OpenF1 times that agree within tol seconds."""
    d = laps[laps["season"] == season]
    a = d[d["source"] == "jolpica"][["round", "driverId", "lap", "lap_time_s"]]
    b = d[d["source"] == "openf1"][["round", "driverId", "lap", "lap_time_s"]]
    m = a.merge(b, on=["round", "driverId", "lap"], suffixes=("_j", "_o"))
    m = m.dropna(subset=["lap_time_s_j", "lap_time_s_o"])
    m["agree"] = (m["lap_time_s_j"] - m["lap_time_s_o"]).abs() <= tol
    out = m.groupby("round").agg(n_both=("agree", "size"), share=("agree", "mean")).reset_index()
    out["n_jolpica"] = out["round"].map(a.groupby("round").size())
    out["n_openf1"] = out["round"].map(b.groupby("round").size())
    return out
