"""Plan-2 ingestion (PLAN2.md section 3): sprint results, practice laps, session weather, race-weekend
forecasts and (dev races only) race lap times for the pace oracle. Everything is cached under
data/raw/plan2/ and every function is resumable; a failed request is skipped and logged, never fatal.

Source decisions (recorded in PLAN2.md, amendment 1):
  * FastF1's timing API now answers 401 without an F1 account login, so practice laps and session
    weather come from OpenF1 (free, no login), which only covers 2023 onward.
  * Race-weekend weather comes from Open-Meteo: ERA5 reanalysis for the qualifying day (observed, known
    before the race) and its historical-forecast archive for the race window (forecast, 2020+).
  * Sprint results and race laps come from Jolpica (already used in plan 1).

Run:  python -m pitlane.data.ingest2
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

from pitlane.config import FIRST_SEASON, LAST_SEASON, RAW
from pitlane.data.ingest import BASE, fetch_pages

RAW2 = RAW / "plan2"
OPENF1 = "https://api.openf1.org/v1"
OPENF1_YEARS = (2023, 2024, 2025, 2026)
FAIL_LOG = RAW2 / "failures.txt"
_last: dict[str, float] = {}


def _log_fail(msg: str) -> None:
    RAW2.mkdir(parents=True, exist_ok=True)
    with open(FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def _get(url: str, host: str, gap: float, retries: int = 6):
    """GET json with a per-host minimum gap, backoff on 429/5xx. 404 -> []. Failure -> None."""
    delay = 6.0
    for _ in range(retries):
        wait = _last.get(host, 0) + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, timeout=90)
            _last[host] = time.time()
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return []
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            _log_fail(f"HTTP {r.status_code} {url}")
            return {}                # permanent client error: treated as "no data", cached by callers
        except (requests.RequestException, ValueError):
            time.sleep(delay)
            delay = min(delay * 2, 120)
    _log_fail(url)
    return None


def _cached(path: Path, fn):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    out = fn()
    if out is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out), encoding="utf-8")
    return out


# ------------------------------------------------------------------------------------ Jolpica
def ingest_sprints() -> None:
    for season in range(2021, LAST_SEASON + 1):        # sprints exist from 2021
        fetch_pages("sprint", season)


def fetch_laps(season: int, rnd: int) -> list | None:
    """Race lap times as [lap, driverId, 'm:ss.mmm'] rows (Jolpica caps pages at 100 timings)."""
    def fn():
        rows, offset, total = [], 0, None
        while total is None or offset < total:
            d = _get(f"{BASE}/{season}/{rnd}/laps.json?limit=100&offset={offset}", "jolpi", 0.6)
            if d is None:
                return None
            d = d["MRData"]
            total = int(d["total"])
            for race in d["RaceTable"]["Races"]:
                for lap in race.get("Laps", []):
                    rows += [[int(lap["number"]), t["driverId"], t["time"]] for t in lap["Timings"]]
            offset += 100
        return rows
    return _cached(RAW2 / "laps" / f"laps_{season}_{rnd}.json", fn)


def ingest_dev_laps(seasons=(2023,)) -> None:
    """Race laps for the pace oracle (PLAN2.md A1). Jolpica allows ~500 requests/hour and a race is ~12
    pages, so only the 2023 development season is fetched (22 races)."""
    for season in seasons:
        for race in fetch_pages("results", season):
            fetch_laps(season, int(race["round"]))


# ------------------------------------------------------------------------------------ OpenF1
def openf1_sessions(year: int):
    return _cached(RAW2 / "openf1" / f"sessions_{year}.json",
                   lambda: _get(f"{OPENF1}/sessions?year={year}", "openf1", 2.2))


def _compact_laps(rows: list[dict]) -> list:
    return [[r["driver_number"], r["lap_number"], r["lap_duration"], int(bool(r["is_pit_out_lap"]))] for r in rows]


def _weather_summary(mk: int):
    r = _get(f"{OPENF1}/weather?meeting_key={mk}", "openf1", 2.2)
    if r is None:
        return None
    out: dict[str, dict] = {}
    for k in {x["session_key"] for x in r}:
        rows = [x for x in r if x["session_key"] == k]
        out[str(k)] = {"rain_frac": float(np.mean([x["rainfall"] > 0 for x in rows])),
                       "track_temp": float(np.mean([x["track_temperature"] for x in rows])),
                       "air_temp": float(np.mean([x["air_temperature"] for x in rows]))}
    return out


def ingest_openf1(years=OPENF1_YEARS) -> None:
    now = datetime.now(timezone.utc)
    for y in years:
        sess = openf1_sessions(y) or []
        by_meeting: dict[int, list[dict]] = {}
        for s in sess:
            by_meeting.setdefault(s["meeting_key"], []).append(s)
        for mk, ss in sorted(by_meeting.items()):
            if not any(s["session_name"] == "Race" for s in ss):      # pre-season testing
                continue
            past = [s for s in ss if datetime.fromisoformat(s["date_end"]) < now - timedelta(hours=6)]
            prac = [s for s in past if s["session_type"] == "Practice"]
            if not prac:
                continue
            k0 = prac[0]["session_key"]
            _cached(RAW2 / "openf1" / f"drivers_{mk}.json",
                    lambda: _get(f"{OPENF1}/drivers?session_key={k0}", "openf1", 2.2))
            for s in prac:
                k = s["session_key"]
                _cached(RAW2 / "openf1" / f"laps_{k}.json",
                        lambda: (lambda r: None if r is None else _compact_laps(r))(
                            _get(f"{OPENF1}/laps?session_key={k}", "openf1", 2.2)))
            _cached(RAW2 / "openf1" / f"weather_{mk}.json", lambda: _weather_summary(mk))
        print(f"openf1 {y}: {len(by_meeting)} meetings", flush=True)


# ------------------------------------------------------------------------------------ Open-Meteo
def _window(hourly: dict, key: str, start: datetime, hours: int) -> float | None:
    t = hourly.get("time") or []
    v = hourly.get(key) or []
    lo, hi = start.strftime("%Y-%m-%dT%H:00"), (start + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:00")
    vals = [x for tt, x in zip(t, v) if lo <= tt <= hi and x is not None]
    return float(np.sum(vals)) if vals else None


def race_start_utc(race: dict) -> datetime:
    t = race.get("time") or ""
    try:
        return datetime.fromisoformat(f"{race['date']}T{t.replace('Z', '')}")
    except ValueError:
        lon = float(race["Circuit"]["Location"]["long"])
        return datetime.fromisoformat(f"{race['date']}T14:00:00") - timedelta(hours=lon / 15)


def fetch_weather(race: dict) -> dict | None:
    season, rnd = race["season"], race["round"]
    lat, lon = race["Circuit"]["Location"]["lat"], race["Circuit"]["Location"]["long"]
    start = race_start_utc(race)

    def fn():
        sat = start - timedelta(days=1)
        d1, d2 = sat.strftime("%Y-%m-%d"), start.strftime("%Y-%m-%d")
        a = _get(f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
                 f"&start_date={d1}&end_date={d1}&hourly=precipitation&timezone=GMT", "openmeteo", 0.4)
        f = {} if int(season) < 2020 else _get(
            f"https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&start_date={d2}&end_date={d2}&hourly=precipitation,precipitation_probability&timezone=GMT",
            "openmeteo", 0.4)
        if a is None or f is None:
            return None
        a = {} if isinstance(a, list) else a
        f = {} if isinstance(f, list) else f
        ha, hf = a.get("hourly") or {}, f.get("hourly") or {}
        prob = _window(hf, "precipitation_probability", start, 2)
        return {"sat_mm": _window(ha, "precipitation", sat - timedelta(hours=1), 3),
                "fc_mm": _window(hf, "precipitation", start, 2),
                "fc_prob": None if prob is None else prob / 300.0}   # 3 hourly values in %, summed
    return _cached(RAW2 / "weather" / f"wx_{season}_{rnd}.json", fn)


def ingest_weather() -> None:
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        for race in fetch_pages("results", season):
            fetch_weather(race)
        print(f"weather {season} done", flush=True)


def main() -> None:
    RAW2.mkdir(parents=True, exist_ok=True)
    ingest_sprints()
    print("sprints done", flush=True)
    ingest_weather()
    ingest_openf1()
    ingest_dev_laps()
    print("plan-2 ingestion done; failures:", FAIL_LOG if FAIL_LOG.exists() else "none")


if __name__ == "__main__":
    sys.exit(main())
