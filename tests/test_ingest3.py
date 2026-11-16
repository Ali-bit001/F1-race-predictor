"""R0 (PLAN3 section 3): lap and pit-stop ingestion. No network: `requests` is replaced by a scripted fake.

Parts: parsers, HTTP/caching/retry, pagination, odd rows, driver join, red-flag flag, loader isolation,
and checks on the tables currently on disk (skipped if absent; the Jolpica lap download may be incomplete).
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from pitlane.data import ingest3 as I

INTERIM = I.INTERIM3
LAPS_PQ, PITS_PQ, STINTS_PQ = INTERIM / "laps.parquet", INTERIM / "pits.parquet", INTERIM / "stints.parquet"
IDENTIFYING = ["@", "asus", "muhammad", "ali", "pitlane", "f1-race-predictor", "claude", "anthropic", "gmail"]


# ------------------------------------------------------------------------------------ fakes
class Resp:
    def __init__(self, status=200, body=None, headers=None, bad_json=False):
        self.status_code, self._body, self.headers, self._bad = status, body, headers or {}, bad_json

    def json(self):
        if self._bad:
            raise ValueError("bad json")
        return self._body


class FakeSession:
    """Pops scripted responses (or raises scripted exceptions); records every url and the headers."""

    def __init__(self, script=None, fn=None):
        self.script, self.fn, self.urls, self.headers = list(script or []), fn, [], {}

    def get(self, url, timeout=None):
        self.urls.append(url)
        if self.fn is not None:
            out = self.fn(url)
        else:
            out = self.script.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class FakeClient:
    """Stands in for Client where only get() matters: fn(url) -> (state, json)."""

    def __init__(self, fn):
        self.fn, self.urls = fn, []

    def get(self, url, host, gap):
        self.urls.append(url)
        return self.fn(url)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Redirect every path of the module into tmp_path; no real sleeping."""
    raw = tmp_path / "raw"
    raw3 = raw / "plan3"
    raw3.mkdir(parents=True)
    for name, val in dict(RAW=raw, RAW3=raw3, RAW2_LAPS=raw / "plan2" / "laps", INTERIM3=tmp_path / "interim",
                          FAIL_FILE=raw3 / "failures.json", REQ_FILE=raw3 / "jolpica_requests.txt").items():
        monkeypatch.setattr(I, name, val)
    sleeps: list[float] = []
    monkeypatch.setattr(I.time, "sleep", lambda s: sleeps.append(s))
    c = copy.deepcopy(I.cfg())
    c["jolpica"]["laps_gap_s"] = c["jolpica"]["pits_gap_s"] = 0
    c["openf1"]["gap_s"] = 0
    c["retry"].update(max_tries=3, first_delay_s=1, max_delay_s=4)
    return dict(tmp=tmp_path, raw=raw, raw3=raw3, sleeps=sleeps, c=c)


def make_client(env, script=None, fn=None) -> tuple[I.Client, FakeSession]:
    cl = I.Client(env["c"])
    cl.s = FakeSession(script, fn)
    cl.s.headers = cl.s.headers
    return cl, cl.s


def mr(total, laps=None, races=None, pits=None):
    """Jolpica-shaped MRData page."""
    if laps is not None:
        races = [{"Laps": laps}]
    if pits is not None:
        races = [{"PitStops": pits}]
    return {"MRData": {"total": str(total), "RaceTable": {"Races": races or []}}}


def lap(n, drivers):
    return {"number": str(n), "Timings": [{"driverId": d, "position": str(i + 1), "time": "1:30.000"}
                                          for i, d in enumerate(drivers)]}


def write_results(raw: Path, season: int, races: list[dict]) -> None:
    (raw / f"results_{season}.json").write_text(json.dumps(races))


def race_dict(rnd, date, rows, time="13:00:00Z"):
    return {"season": "2023", "round": str(rnd), "date": date, "time": time, "rows": rows}


def row(number, driver, code):
    return {"number": str(number), "Driver": {"driverId": driver, "code": code, "permanentNumber": "0"}}


# ------------------------------------------------------------------------------------ 1. parsers
@pytest.mark.parametrize("s,exp", [
    ("1:42.038", 102.038), ("59.999", 59.999), ("22.123", 22.123), ("15:25.181", 925.181),
    ("1:02:03.456", 3723.456), ("0:59.000", 59.0), ("1:05.123", 65.123), ("12.092", 12.092),
])
def test_time_parser_valid(s, exp):
    assert I._t(s) == pytest.approx(exp, abs=1e-9)


@pytest.mark.parametrize("s", [None, "", "abc", "1:xx.5", ":", "1:", "--", "1,42.038", " "])
def test_time_parser_malformed_is_nan(s):
    assert np.isnan(I._t(s))


@pytest.mark.xfail(strict=True, reason="_t accepts 'inf', negative values and 4-part strings and returns a "
                                       "number instead of NaN; harmless on real data (none on disk) but a "
                                       "parser that returns inf would pass the 'finite' check silently")
@pytest.mark.parametrize("s", ["inf", "-1:00.000", "1:2:3:4.0"])
def test_time_parser_rejects_non_times(s):
    assert np.isnan(I._t(s))


# ------------------------------------------------------------------------------------ 2. HTTP / caching
def test_user_agent_is_generic_and_nothing_identifying_is_sent(env):
    c = I.cfg()
    ua = c["user_agent"].lower()
    cl = I.Client(c)
    assert cl.s.headers["User-Agent"] == c["user_agent"]
    blob = " ".join(f"{k}={v}" for k, v in cl.s.headers.items()).lower()
    text = (I.CONFIGS / "ingest3.yaml").read_text(encoding="utf-8")
    values = " ".join(str(v) for v in [c["user_agent"], c["jolpica"]["base"], c["openf1"]["base"]]).lower()
    for bad in IDENTIFYING:
        rx = re.compile(rf"(?<![a-z]){re.escape(bad)}(?![a-z])")      # 'ali' must not match 'keep-alive'
        assert not (rx.search(ua) or rx.search(blob) or rx.search(values)), bad
    assert not re.search(r"[\w.]+@[\w.]+", text)
    assert set(cl.s.headers) <= {"User-Agent", "Accept-Encoding", "Accept", "Connection"}


def test_urls_requested_carry_no_identifying_text(env):
    cl, sess = make_client(env, fn=lambda u: Resp(200, mr(0)))
    write_results(env["raw"], 2022, [race_dict(1, "2022-03-20", [])])
    I.ingest_jolpica_laps(cl, env["c"], [2022])
    assert sess.urls
    for u in sess.urls:
        assert u.startswith(env["c"]["jolpica"]["base"])
        assert not any(re.search(rf"(?<![a-z]){re.escape(b)}(?![a-z])", u.lower()) for b in IDENTIFYING)


@pytest.mark.parametrize("bad", [Resp(429, headers={"Retry-After": "7"}), Resp(500), Resp(502), Resp(503), Resp(504),
                                 requests.Timeout("t"), requests.ConnectionError("c"), Resp(200, bad_json=True)])
def test_transient_failure_is_retried_then_succeeds(env, bad):
    cl, sess = make_client(env, [bad, Resp(200, {"a": 1})])
    assert cl.get("http://x/y", "openf1", 0) == ("ok", {"a": 1})
    assert len(sess.urls) == 2
    assert env["sleeps"], "backoff must sleep between tries"


def test_retry_after_header_is_honoured_and_garbage_is_tolerated(env):
    cl, _ = make_client(env, [Resp(429, headers={"Retry-After": "42"}), Resp(200, {})])
    cl.get("http://x", "openf1", 0)
    assert 42.0 in env["sleeps"]
    cl, _ = make_client(env, [Resp(429, headers={"Retry-After": "Wed, 21 Oct 2026"}), Resp(200, {})])
    assert cl.get("http://x", "openf1", 0)[0] == "ok"


def test_exhausted_retries_fail_and_nothing_is_cached(env):
    write_results(env["raw"], 2022, [race_dict(1, "2022-03-20", [])])
    for bad in (Resp(429), Resp(503), requests.Timeout("t")):
        cl, sess = make_client(env, fn=lambda u, b=bad: b)
        out = I.ingest_jolpica_laps(cl, env["c"], [2022])
        assert out == {"failed": 1}
        assert len(sess.urls) == env["c"]["retry"]["max_tries"]
        assert not I.laps_path(2022, 1).exists()
        fl = json.loads(I.FAIL_FILE.read_text())
        assert "jolpica_laps/2022/1" in fl["failed"] and fl["empty"] == {}


def test_non_retryable_status_fails_immediately_and_404_is_missing(env):
    cl, sess = make_client(env, [Resp(403)])
    assert cl.get("http://x", "openf1", 0)[0] == "fail" and len(sess.urls) == 1
    cl, sess = make_client(env, [Resp(404)])
    assert cl.get("http://x", "openf1", 0) == ("missing", None)


def test_failure_then_success_clears_failed_entry(env):
    write_results(env["raw"], 2022, [race_dict(1, "2022-03-20", [])])
    cl, _ = make_client(env, fn=lambda u: Resp(503))
    I.ingest_jolpica_laps(cl, env["c"], [2022])
    assert "jolpica_laps/2022/1" in json.loads(I.FAIL_FILE.read_text())["failed"]
    cl, _ = make_client(env, fn=lambda u: Resp(200, mr(2, laps=[lap(1, ["a", "b"])])))
    assert I.ingest_jolpica_laps(cl, env["c"], [2022]) == {"downloaded": 1}
    assert json.loads(I.FAIL_FILE.read_text())["failed"] == {}
    assert len(json.loads(I.laps_path(2022, 1).read_text())) == 2


def test_genuine_empty_answer_is_cached_only_for_a_race_already_run(env):
    future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
    write_results(env["raw"], 2022, [race_dict(1, "2022-03-20", []), race_dict(2, future, [])])
    cl, sess = make_client(env, fn=lambda u: Resp(200, mr(0)))
    out = I.ingest_jolpica_laps(cl, env["c"], [2022])
    assert out == {"empty": 1}
    assert I.laps_path(2022, 1).exists() and json.loads(I.laps_path(2022, 1).read_text()) == []
    assert "jolpica_laps/2022/1" in json.loads(I.FAIL_FILE.read_text())["empty"]
    assert not I.laps_path(2022, 2).exists()
    assert not any("/2022/2/" in u for u in sess.urls), "a future race must never be requested"


def test_future_race_never_requested_by_any_ingest_function(env):
    future = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
    for s in (2022, 2023, 2026):
        write_results(env["raw"], s, [race_dict(1, future, [])])
    cl, sess = make_client(env, fn=lambda u: Resp(200, mr(0)))
    env["c"]["jolpica"]["pit_seasons"] = [2022, 2023, 2026]
    env["c"]["openf1"]["seasons"] = [2026]
    assert I.ingest_jolpica_laps(cl, env["c"], [2022]) == {}
    assert I.ingest_jolpica_pits(cl, env["c"]) == {}
    sess.fn = lambda u: Resp(200, [{"session_name": "Race", "date_start": future + "T13:00:00+00:00",
                                    "date_end": future + "T15:00:00+00:00", "session_key": 1}])
    I.ingest_openf1(cl, env["c"], [2026])
    assert all("/sessions" in u for u in sess.urls), sess.urls
    assert not list((env["raw3"]).rglob("laps_*")) and not list(env["raw3"].rglob("pits_*"))


def test_is_run_boundary():
    c = I.cfg()
    r = race_dict(1, "2023-03-05", [], "15:00:00Z")
    start = datetime(2023, 3, 5, 15, tzinfo=timezone.utc)
    h = timedelta(hours=float(c["race_done_after_h"]))
    assert I.is_run(r, c, start + h + timedelta(minutes=1))
    assert not I.is_run(r, c, start + h - timedelta(minutes=1))
    assert not I.is_run(r, c, start)
    assert I.race_start(race_dict(1, "2023-03-05", [], None)).hour == 13      # missing time falls back


def test_existing_files_are_skipped_resume(env):
    write_results(env["raw"], 2022, [race_dict(1, "2022-03-20", []), race_dict(2, "2022-03-27", [])])
    I.write_json(I.laps_path(2022, 1), [[1, "a", "1:30.000", "1"]])
    cl, sess = make_client(env, fn=lambda u: Resp(200, mr(1, laps=[lap(1, ["a"])])))
    out = I.ingest_jolpica_laps(cl, env["c"], [2022])
    assert out == {"cached": 1, "downloaded": 1}
    assert all("/2022/2/" in u for u in sess.urls)
    assert json.loads(I.laps_path(2022, 1).read_text()) == [[1, "a", "1:30.000", "1"]], "cached file untouched"
    sess.urls.clear()
    assert I.ingest_jolpica_laps(cl, env["c"], [2022]) == {"cached": 2} and not sess.urls


def test_plan2_season_is_not_redownloaded(env):
    write_results(env["raw"], 2023, [race_dict(1, "2023-03-05", [])])
    cl, sess = make_client(env, fn=lambda u: Resp(200, mr(0)))
    assert I.ingest_jolpica_laps(cl, env["c"], [2023]) == {} and not sess.urls


def test_write_is_atomic(env, monkeypatch):
    p = env["raw3"] / "x" / "f.json"
    I.write_json(p, [1])
    monkeypatch.setattr(I.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        I.write_json(p, [1, 2, 3])
    assert json.loads(p.read_text()) == [1], "old content survives a failed write"
    q = env["raw3"] / "x" / "g.json"
    with pytest.raises(OSError):
        I.write_json(q, [1])
    assert not q.exists(), "no partial target after a failed write"
    monkeypatch.undo()
    with pytest.raises(TypeError):
        I.write_json(q, {"a": object()})
    assert not q.exists()


def test_failed_write_during_ingest_is_not_recorded_as_cached(env, monkeypatch):
    write_results(env["raw"], 2022, [race_dict(1, "2022-03-20", [])])
    cl, _ = make_client(env, fn=lambda u: Resp(200, mr(1, laps=[lap(1, ["a"])])))
    real = I.os.replace
    monkeypatch.setattr(I.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("disk")) if str(b).endswith("laps_2022_1.json") else real(a, b))
    with pytest.raises(OSError):
        I.ingest_jolpica_laps(cl, env["c"], [2022])
    assert not I.laps_path(2022, 1).exists()


def test_hourly_guard_blocks_when_window_full_and_passes_when_not(env, monkeypatch):
    c = env["c"]
    cap = int(c["jolpica"]["max_per_hour"])
    now = I.time.time()
    I.REQ_FILE.write_text("\n".join(f"{now - 10 - i:.2f}" for i in range(cap)) + "\n")
    cl, sess = make_client(env, fn=lambda u: Resp(200, {}))

    class Blocked(Exception):
        pass

    def stop(s):
        raise Blocked(s)

    monkeypatch.setattr(I.time, "sleep", stop)
    with pytest.raises(Blocked) as e:
        cl.get("http://x", "jolpica", 0)
    assert e.value.args[0] == 60 or e.value.args[0] > 0
    assert not sess.urls, "no request may be sent while the window is full"
    # an old request log does not block
    I.REQ_FILE.write_text("\n".join(f"{now - 4000 - i:.2f}" for i in range(cap)) + "\n")
    assert cl.get("http://x", "jolpica", 0)[0] == "ok"
    assert len(I.REQ_FILE.read_text().split()) == cap + 1, "every request is logged"
    # one below the cap does not block either
    I.REQ_FILE.write_text("\n".join(f"{now - 10 - i:.2f}" for i in range(cap - 1)) + "\n")
    assert cl.get("http://x", "jolpica", 0)[0] == "ok"


def test_hourly_guard_counts_every_retry(env):
    cl, sess = make_client(env, [Resp(503), Resp(503), Resp(200, {})])
    cl.get("http://x", "jolpica", 0)
    assert len(I.REQ_FILE.read_text().split()) == 3


def test_openf1_empty_cached_only_when_no_results_and_failure_not_cached(env):
    c = env["c"]
    cl, _ = make_client(env, [Resp(404)])
    assert I._fetch_openf1_one(cl, c, "pit", 2024, 1, 99) == "empty"
    assert json.loads(I.openf1_path("pit", 2024, 1).read_text()) == []
    cl, _ = make_client(env, fn=lambda u: Resp(429))
    assert I._fetch_openf1_one(cl, c, "pit", 2024, 2, 99) == "failed"
    assert not I.openf1_path("pit", 2024, 2).exists()
    cl, _ = make_client(env, [Resp(200, {"detail": "No results found."})])
    assert I._fetch_openf1_one(cl, c, "laps", 2024, 3, 99) == "empty"
    cl, _ = make_client(env, [Resp(200, {"detail": "Rate limit exceeded"})])     # a dict that is not "no results"
    assert I._fetch_openf1_one(cl, c, "laps", 2024, 4, 99) == "failed"
    assert not I.openf1_path("laps", 2024, 4).exists()


def test_openf1_laps_keep_only_whitelisted_fields(env):
    rows = [{"driver_number": 1, "lap_number": 1, "lap_duration": None, "meeting_key": 5, "segments_sector_1": [1]}]
    cl, _ = make_client(env, [Resp(200, rows)])
    I._fetch_openf1_one(cl, env["c"], "laps", 2024, 1, 7)
    got = json.loads(I.openf1_path("laps", 2024, 1).read_text())
    assert set(got[0]) == set(I.OPENF1_LAP_FIELDS) and got[0]["lap_duration"] is None


def test_session_matching_by_date_race_only(env):
    races = [race_dict(1, "2024-03-02", []), race_dict(2, "2024-03-09", [])]
    sess = [
        {"session_name": "Sprint", "date_start": "2024-03-09T11:00:00+00:00", "session_key": 1},
        {"session_name": "Race", "date_start": "2024-03-09T15:00:00+00:00", "session_key": 2},
        {"session_name": "Race", "date_start": "2024-03-02T15:00:00+03:00", "session_key": 3},
        {"session_name": "Race", "date_start": "2024-04-01T15:00:00+00:00", "session_key": 4},
        {"session_name": "Race", "date_start": "2024-03-30T15:00:00+00:00", "session_key": 5, "is_cancelled": True},
    ]
    m = I.match_race_sessions(sess, races)
    assert {k: v["session_key"] for k, v in m.items()} == {1: 3, 2: 2}


# ------------------------------------------------------------------------------------ 3. pagination
def paged_fn(lap_rows_by_page, total):
    def fn(url):
        off = int(re.search(r"offset=(\d+)", url).group(1))
        lim = int(re.search(r"limit=(\d+)", url).group(1))
        return "ok", lap_rows_by_page(off, lim, total)
    return fn


def synth_pages(drivers, n_laps):
    """Flatten (lap, driver) timings like Jolpica does: the limit counts timings, so a lap may be split."""
    flat = [(n, d, i) for n in range(1, n_laps + 1) for i, d in enumerate(drivers)]

    def page(off, lim, total):
        chunk = flat[off:off + lim]
        laps: dict[int, list] = {}
        for n, d, i in chunk:
            laps.setdefault(n, []).append({"driverId": d, "position": str(i + 1), "time": f"1:{30 + i}.{n:03d}"})
        return mr(total, laps=[{"number": str(n), "Timings": t} for n, t in laps.items()])
    return flat, page


def test_pagination_offsets_cover_total_exactly_once(env):
    drivers = [f"d{i}" for i in range(20)]
    flat, page = synth_pages(drivers, 13)                        # 260 timings; page 100 splits laps 5 and 10
    c = env["c"]
    c["jolpica"]["page"] = 100
    seen = []

    def fn(url):
        seen.append(int(re.search(r"offset=(\d+)", url).group(1)))
        return "ok", page(seen[-1], 100, len(flat))
    st, rows = I.jolpica_paged(FakeClient(fn), c, "http://x/laps.json", 0, I._laps_rows)
    assert st == "ok" and seen == [0, 100, 200]
    assert len(rows) == len(flat) == 260
    assert len({(r[0], r[1]) for r in rows}) == 260, "no duplicated (lap, driver)"
    assert {(r[0], r[1]) for r in rows} == {(n, d) for n, d, _ in flat}, "no dropped lap"


@pytest.mark.parametrize("total", [0, 1, 99, 100, 101, 200, 250])
def test_pagination_boundaries(env, total):
    c = env["c"]
    c["jolpica"]["page"] = 100
    items = [{"driverId": f"d{i}", "lap": str(i + 1), "stop": "1", "time": "12:00:00", "duration": "22.1"} for i in range(total)]
    offs = []

    def fn(url):
        off = int(re.search(r"offset=(\d+)", url).group(1))
        offs.append(off)
        return "ok", mr(total, pits=items[off:off + 100])
    st, rows = I.jolpica_paged(FakeClient(fn), c, "http://x/pitstops.json", 0, I._pit_rows)
    assert st == "ok" and len(rows) == total
    assert offs == list(range(0, max(total, 1), 100))


def test_pagination_failure_midway_returns_fail_not_partial(env):
    c = env["c"]
    calls = []

    def fn(url):
        calls.append(url)
        return ("ok", mr(250, laps=[lap(1, ["a"])])) if len(calls) < 2 else ("fail", "retries exhausted")
    st, why = I.jolpica_paged(FakeClient(fn), c, "http://x/laps.json", 0, I._laps_rows)
    assert st == "fail" and why == "retries exhausted"


def test_pagination_url_joins_query_correctly(env):
    urls = []
    I.jolpica_paged(FakeClient(lambda u: (urls.append(u) or ("ok", mr(0)))), env["c"], "http://x/a.json", 0, I._laps_rows)
    I.jolpica_paged(FakeClient(lambda u: (urls.append(u) or ("ok", mr(0)))), env["c"], "http://x/a.json?q=1", 0, I._laps_rows)
    assert urls[0].startswith("http://x/a.json?limit=") and urls[1].startswith("http://x/a.json?q=1&limit=")


def test_full_race_split_across_pages_roundtrips_through_files(env):
    drivers = [f"d{i}" for i in range(20)]
    flat, page = synth_pages(drivers, 13)
    write_results(env["raw"], 2022, [race_dict(1, "2022-03-20", [])])
    cl, _ = make_client(env, fn=lambda u: Resp(200, page(int(re.search(r"offset=(\d+)", u).group(1)),
                                                        int(re.search(r"limit=(\d+)", u).group(1)), len(flat))))
    assert I.ingest_jolpica_laps(cl, env["c"], [2022]) == {"downloaded": 1}
    d = I._jolpica_laps_df(2022, 1)
    assert len(d) == 260 and not d.duplicated(["lap", "driverId"]).any()
    assert d.groupby("lap").size().eq(20).all()


# ------------------------------------------------------------------------------------ 4. odd rows
def test_laps_rows_tolerate_missing_position_and_time():
    page = {"RaceTable": {"Races": [{"Laps": [{"number": "1", "Timings": [{"driverId": "a", "time": "1:30.000"},
                                                                          {"driverId": "b", "position": "2"}]}]}]}}
    assert I._laps_rows(page) == [[1, "a", "1:30.000", None], [1, "b", None, "2"]]
    assert I._laps_rows({"RaceTable": {"Races": []}}) == []
    assert I._laps_rows({"RaceTable": {"Races": [{}]}}) == []                  # race without Laps


def test_jolpica_frame_with_missing_position_short_rows_and_unparseable_times(env):
    I.write_json(I.laps_path(2019, 1), [[1, "a", "1:30.000", None], [1, "b", None, "2"], [2, "a", "xx", "1"], [3, "a", "1:31.5"]])
    d = I._jolpica_laps_df(2019, 1)
    assert len(d) == 4
    assert d["lap_time_s"].isna().tolist() == [False, True, True, False]
    assert d["position"].isna().tolist() == [True, False, False, True]
    # plan-2 cache rows have three columns only
    I.write_json(env["raw"] / "plan2" / "laps" / "laps_2023_1.json", [[1, "a", "1:30.000"], [2, "a", "1:31.000"]])
    d = I._jolpica_laps_df(2023, 1)
    assert len(d) == 2 and d["position"].isna().all() and d["source"].eq("jolpica").all()


def test_driver_with_missing_laps_gets_no_invented_rows(env):
    rows = [[n, "a", "1:30.000", "1"] for n in range(1, 11)] + [[n, "b", "1:31.000", "2"] for n in range(1, 4)]
    I.write_json(I.laps_path(2019, 1), rows)
    d = I._jolpica_laps_df(2019, 1)
    assert d.groupby("driverId").size().to_dict() == {"a": 10, "b": 3}
    assert not d.duplicated(["driverId", "lap"]).any()


def test_empty_cached_race_yields_no_frame(env):
    I.write_json(I.laps_path(2019, 1), [])
    assert I._jolpica_laps_df(2019, 1) is None
    I.write_json(I.pits_path(2019, 1), [])
    assert I._pits_jolpica_df(2019, 1, 20) is None
    assert I._pits_jolpica_df(2019, 2, 20) is None                              # absent file


def test_openf1_null_lap_duration_passes_through_as_nan(env):
    write_results(env["raw"], 2024, [race_dict(1, "2024-03-02", [row(1, "max_verstappen", "VER"), row(11, "perez", "PER")])])
    I.write_json(I.openf1_path("laps", 2024, 1), [
        dict(date_start=None, driver_number=1, lap_number=1, lap_duration=None, duration_sector_1=None, duration_sector_2=None,
             duration_sector_3=None, i1_speed=None, i2_speed=None, st_speed=None, is_pit_out_lap=True),
        dict(date_start="2024-03-02T15:00:00", driver_number=1, lap_number=2, lap_duration=95.5, duration_sector_1=30.1,
             duration_sector_2=35.2, duration_sector_3=30.2, i1_speed=300, i2_speed=None, st_speed=310, is_pit_out_lap=False),
        dict(date_start=None, driver_number=11, lap_number=1, lap_duration=None, duration_sector_1=None, duration_sector_2=None,
             duration_sector_3=None, i1_speed=None, i2_speed=None, st_speed=None, is_pit_out_lap=False),
    ])
    out = I.load_tables(env["c"], write=False)
    laps = out["laps"]
    assert len(laps) == 3 and laps["lap_time_s"].isna().sum() == 2
    assert laps.loc[laps["lap_time_s"].notna(), "lap_time_s"].iloc[0] == 95.5
    assert set(laps["driverId"]) == {"max_verstappen", "perez"} and (laps["source"] == "openf1").all()


def test_pits_with_null_duration(env):
    I.write_json(I.pits_path(2019, 1), [["a", 10, 1, "12:00:00", "22.5"], ["b", 11, 1, "12:01:00", None], ["c", 12, 1, "12:02:00", "bad"]])
    d = I._pits_jolpica_df(2019, 1, 20)
    assert len(d) == 3 and d["duration_s"].isna().tolist() == [False, True, True]
    assert d["red_flag_suspect"].tolist() == [False, False, False]


# ------------------------------------------------------------------------------------ 5. driver join
def test_driver_join_changed_numbers_and_substitutes(env):
    """2023-like: round 1 Verstappen races as #1 (permanent number 33), round 2 a substitute drives #21 and
    round 3 the same number belongs to somebody else. The join is per round on the race's own number."""
    write_results(env["raw"], 2023, [
        race_dict(1, "2023-03-05", [row(1, "max_verstappen", "VER"), row(21, "de_vries", "DEV"), row(3, "ricciardo", "RIC")]),
        race_dict(2, "2023-03-19", [row(1, "max_verstappen", "VER"), row(21, "substitute_x", "SUB"), row(3, "ricciardo", "RIC")]),
        race_dict(3, "2023-04-02", [row(1, "max_verstappen", "VER"), row(21, "lawson", "LAW")]),
    ])
    for rnd in (1, 2, 3):
        I.write_json(I.openf1_path("laps", 2023, rnd), [
            dict(driver_number=n, lap_number=1, lap_duration=90.0 + n / 100) for n in (1, 21, 3, 99)])
        I.write_json(I.openf1_path("drivers", 2023, rnd), [
            dict(driver_number=1, name_acronym="VER"), dict(driver_number=21, name_acronym="DEV" if rnd == 1 else "XXX"),
            dict(driver_number=3, name_acronym="RIC")])
    out = I.load_tables(env["c"], write=False)
    laps = out["laps"]
    got = {(r, n): d for r, n, d in zip(laps["round"], laps["lap_time_s"].round(2), laps["driverId"])}
    assert got[(1, 90.01)] == "max_verstappen" and got[(2, 90.01)] == "max_verstappen"
    assert got[(1, 90.21)] == "de_vries" and got[(2, 90.21)] == "substitute_x" and got[(3, 90.21)] == "lawson"
    assert got[(1, 90.03)] == "ricciardo" and (3, 90.03) not in got, "number 3 absent from round 3's results: dropped, not guessed"
    assert not (laps["driverId"].isna().any())
    un = out["unmatched"]
    assert set(un.loc[un["endpoint"] == "laps", "driver_number"]) == {3, 99}
    assert len(un[(un["endpoint"] == "laps") & (un["driver_number"] == 99)]) == 3
    assert len(un[un["endpoint"] == "laps"]) == 3 + 1, "99 in each of 3 rounds plus #3 in round 3"
    assert len(un[un["endpoint"] == "drivers"]) == 2, "acronym mismatch for #21 in rounds 2 and 3 only"


def test_number_map_ignores_non_numeric_numbers(env):
    write_results(env["raw"], 2023, [race_dict(1, "2023-03-05", [{"number": None, "Driver": {"driverId": "z", "code": "ZZZ"}},
                                                                 {"number": "0", "Driver": {"driverId": "y"}}, row(4, "norris", "nor")])])
    m = I._number_map(2023)
    assert m == {1: {0: ("y", ""), 4: ("norris", "NOR")}}


# ------------------------------------------------------------------------------------ 6. red flag
def test_red_flag_suspect_flag_and_rows_never_dropped(env):
    margin = float(env["c"]["pits"]["red_flag_margin_s"])
    durs = ["22.000", "23.000", "24.000", "21.000", "25.000", "1:45.000", "12:09.250"]
    I.write_json(I.pits_path(2019, 5), [[f"d{i}", 5 + i, 1, "12:00:00", x] for i, x in enumerate(durs)])
    d = I._pits_jolpica_df(2019, 5, margin)
    med = d["duration_s"].median()
    assert len(d) == len(durs)
    assert d["red_flag_suspect"].tolist() == [x > med + margin for x in d["duration_s"]]
    assert d["red_flag_suspect"].tolist() == [False] * 5 + [True, True]
    assert d.loc[d["red_flag_suspect"], "duration_s"].min() == pytest.approx(105.0)


def test_red_flag_threshold_is_strict_and_uses_config_margin(env):
    I.write_json(I.pits_path(2019, 5), [[f"d{i}", 5, i, "t", x] for i, x in enumerate(["20.000", "20.000", "20.000", "40.000", "40.001"])])
    d = I._pits_jolpica_df(2019, 5, 20)
    assert d["red_flag_suspect"].tolist() == [False, False, False, False, True]
    d = I._pits_jolpica_df(2019, 5, 5)
    assert d["red_flag_suspect"].tolist() == [False, False, False, True, True]
    assert I.cfg()["pits"]["red_flag_margin_s"] == 20


def test_red_flag_is_per_race_not_global(env):
    I.write_json(I.pits_path(2019, 1), [["a", 1, 1, "t", "22.0"], ["b", 2, 1, "t", "23.0"], ["c", 3, 1, "t", "70.0"]])
    I.write_json(I.pits_path(2019, 2), [["a", 1, 1, "t", "70.0"], ["b", 2, 1, "t", "71.0"], ["c", 3, 1, "t", "72.0"]])
    assert I._pits_jolpica_df(2019, 1, 20)["red_flag_suspect"].tolist() == [False, False, True]
    assert I._pits_jolpica_df(2019, 2, 20)["red_flag_suspect"].tolist() == [False, False, False]


# ------------------------------------------------------------------------------------ 8. loader isolation
def build_tiny_world(env):
    write_results(env["raw"], 2023, [race_dict(r, f"2023-0{r}-05", [row(1, "max_verstappen", "VER"), row(44, "hamilton", "HAM")])
                                     for r in (1, 2, 3)])
    for r in (1, 2, 3):
        I.write_json(I.laps_path(2023, r), [[n, d, f"1:3{r}.{n:03d}", str(i + 1)] for n in range(1, 6) for i, d in enumerate(["max_verstappen", "hamilton"])])
        I.write_json(I.pits_path(2023, r), [["max_verstappen", 2, 1, "12:00:00", f"2{r}.5"], ["hamilton", 3, 1, "12:01:00", "24.0"]])
        I.write_json(I.openf1_path("laps", 2023, r), [dict(driver_number=n, lap_number=k, lap_duration=90.0 + r + k / 10, is_pit_out_lap=False)
                                                      for n in (1, 44) for k in range(1, 6)])
        I.write_json(I.openf1_path("stints", 2023, r), [dict(driver_number=1, stint_number=1, lap_start=1, lap_end=5, compound="SOFT", tyre_age_at_start=0)])
        I.write_json(I.openf1_path("pit", 2023, r), [dict(driver_number=44, lap_number=3, lane_duration=22.1, stop_duration=2.4, date="x")])


def _sub(df, rnd):
    return df[df["round"] == rnd].reset_index(drop=True)


def test_load_tables_for_a_race_does_not_depend_on_other_races_files(env):
    build_tiny_world(env)
    full = I.load_tables(env["c"], write=False)
    for kind in ("laps", "pits", "stints"):
        assert full[kind]["round"].nunique() == 3
    victim = 2
    for f in list(env["raw3"].rglob(f"*_2023_{victim}.json")) + [env["raw"] / "plan2" / "laps" / f"laps_2023_{victim}.json"]:
        if f.exists():
            f.unlink()
    part = I.load_tables(env["c"], write=False)
    for kind in ("laps", "pits", "stints"):
        assert victim not in set(part[kind]["round"])
        for r in (1, 3):
            pd.testing.assert_frame_equal(_sub(full[kind], r), _sub(part[kind], r))


def test_load_tables_does_not_see_a_later_races_file_for_an_earlier_race(env):
    build_tiny_world(env)
    a = I.load_tables(env["c"], write=False)
    I.write_json(I.laps_path(2023, 3), [[1, "hamilton", "9:99.999", "1"]])        # rewrite race 3 only
    b = I.load_tables(env["c"], write=False)
    for kind in ("laps", "pits", "stints"):
        for r in (1, 2):
            pd.testing.assert_frame_equal(_sub(a[kind], r), _sub(b[kind], r))
    assert len(_sub(b["laps"][b["laps"]["source"] == "jolpica"], 3)) == 1


def test_load_tables_write_false_leaves_disk_alone_and_write_true_goes_to_interim(env):
    build_tiny_world(env)
    I.load_tables(env["c"], write=False)
    assert not I.INTERIM3.exists()
    I.load_tables(env["c"], write=True)
    assert (I.INTERIM3 / "laps.parquet").exists() and (I.INTERIM3 / "unmatched_drivers.json").exists()


def test_dedupe_laps_one_source_per_race():
    d = pd.DataFrame(dict(season=2023, round=[1, 1, 2, 3], source=["jolpica", "openf1", "openf1", "jolpica"], lap=1))
    out = I.dedupe_laps(d)
    assert sorted(zip(out["round"], out["source"])) == [(1, "jolpica"), (2, "openf1"), (3, "jolpica")]
    assert sorted(zip(*[I.dedupe_laps(d, "openf1")[c] for c in ("round", "source")])) == [(1, "openf1"), (2, "openf1"), (3, "jolpica")]


# ------------------------------------------------------------------------------------ 7. data on disk
@pytest.fixture(scope="module")
def disk():
    if not LAPS_PQ.exists() or not PITS_PQ.exists():
        pytest.skip("plan-3 tables not built")
    laps, pits = pd.read_parquet(LAPS_PQ), pd.read_parquet(PITS_PQ)
    stints = pd.read_parquet(STINTS_PQ) if STINTS_PQ.exists() else None
    return laps, pits, stints


@pytest.fixture(scope="module")
def results_ids():
    out = {}
    for s in range(2014, 2027):
        for r in I.season_races(s):
            out[(s, int(r["round"]))] = {x["Driver"]["driverId"] for x in r["rows"]}
    if not out:
        pytest.skip("no cached results")
    return out


def test_disk_no_duplicate_laps(disk):
    laps = disk[0]
    dup = laps[laps.duplicated(["season", "round", "driverId", "lap", "source"], keep=False)]
    assert dup.empty, dup.head()


def test_disk_no_duplicate_pit_stops(disk):
    pits = disk[1]
    j = pits[pits["source"] == "jolpica"]
    assert not j.duplicated(["season", "round", "driverId", "stop"]).any()
    assert not j.duplicated(["season", "round", "driverId", "lap", "stop"]).any()
    o = pits[pits["source"] == "openf1"]
    assert not o.duplicated(["season", "round", "driverId", "lap", "time_of_day"]).any()


def test_disk_lap_times_positive_and_finite(disk):
    x = disk[0]["lap_time_s"].dropna()
    assert np.isfinite(x).all() and (x > 0).all()
    assert x.min() > 30, "no sub-30 s lap exists in F1"
    assert x.max() < 6 * 3600
    j = disk[0][disk[0]["source"] == "jolpica"]
    assert j["lap_time_s"].notna().all(), "Jolpica laps always carry a time"
    o = disk[0][disk[0]["source"] == "openf1"]
    assert o["lap_time_s"].isna().mean() < 0.02, "null OpenF1 lap_duration share unexpectedly high"


def test_disk_lap_numbers_and_positions(disk):
    laps = disk[0]
    assert laps["lap"].min() == 1
    j = laps[laps["source"] == "jolpica"]
    p = j["position"].dropna()
    assert p.between(1, 26).all()
    assert j.groupby(["season", "round", "driverId"])["lap"].apply(lambda s: s.is_monotonic_increasing and s.max() >= len(s) - 0).all()


def test_disk_driver_ids_exist_in_that_races_results(disk, results_ids):
    bad = []
    for name, df in (("laps", disk[0]), ("pits", disk[1]), ("stints", disk[2])):
        if df is None:
            continue
        for (s, r), g in df.groupby(["season", "round"]):
            extra = set(g["driverId"]) - results_ids.get((int(s), int(r)), set())
            if extra:
                bad.append((name, s, r, sorted(extra)))
    assert not bad, bad[:5]


def test_disk_every_lap_race_is_a_known_race(disk, results_ids):
    for kind in disk[:2]:
        keys = set(map(tuple, kind[["season", "round"]].drop_duplicates().to_numpy()))
        assert keys <= set(results_ids), sorted(keys - set(results_ids))[:5]


def test_disk_jolpica_lap_counts_match_results(disk, results_ids):
    """Winner's lap count in the results = highest lap in the lap table (races present on disk only)."""
    j = disk[0][disk[0]["source"] == "jolpica"]
    mx = j.groupby(["season", "round"])["lap"].max()
    bad = []
    for s in range(2014, 2027):
        for r in I.season_races(s):
            k = (s, int(r["round"]))
            if k not in mx.index:
                continue
            laps_won = max(int(x.get("laps", 0) or 0) for x in r["rows"])
            if mx[k] != laps_won:
                bad.append((k, int(mx[k]), laps_won))
    # 2014 round 4 (China): the lap table has 56 laps, the Jolpica results record lap count and race time
    # at lap 54 (winner's 56 lap times minus laps 55-56 equal the results time 1:33:28.338 exactly).
    # The lap table is kept as is; this is the only accepted mismatch (checked over all 182 races, 2026-11-17).
    assert bad in ([], [((2014, 4), 56, 54)]), bad


def test_disk_pit_rows_per_season_vs_race_counts(disk):
    pits = disk[1]
    j = pits[pits["source"] == "jolpica"]
    have = j.groupby("season")["round"].apply(set).to_dict()
    missing = {}
    for s in range(2014, 2027):
        races = {int(r["round"]) for r in I.season_races(s) if I.is_run(r, I.cfg())}
        if not races:
            continue
        m = races - have.get(s, set())
        if m:
            missing[s] = m
        assert have.get(s, set()) <= {int(r["round"]) for r in I.season_races(s)}
    # 2021 round 12 (Belgium, 1 racing lap behind the safety car, nobody pitted) is the only accepted gap
    assert missing in ({}, {2021: {12}}), missing
    assert (j.groupby(["season", "round"]).size() >= 5).all(), "a race with fewer than 5 stops is suspicious"
    per_driver = j.groupby(["season", "round", "driverId"]).size()
    assert per_driver.max() <= 8


def test_disk_pit_values_in_range(disk):
    p = disk[1]
    assert p["lap"].min() >= 1 and p["lap"].max() <= 90
    j = p[p["source"] == "jolpica"]
    assert j["stop"].min() == 1
    d = j["duration_s"]
    assert d.isna().mean() < 0.01
    assert (d.dropna() > 5).all() and (d.dropna() < 3 * 3600).all()
    assert j["red_flag_suspect"].isin([True, False]).all()
    # a quiet season has few suspects: a flag rate above 15% in any season means the margin is wrong
    rate = j.groupby("season")["red_flag_suspect"].mean()
    assert (rate < 0.15).all(), rate.to_dict()


def test_disk_openf1_races_present(disk):
    laps, _, stints = disk
    exp = {2023: 22, 2024: 24, 2025: 24, 2026: 20}
    o = laps[laps["source"] == "openf1"].groupby("season")["round"].nunique().to_dict()
    assert o == exp, o
    if stints is not None:
        assert stints.groupby("season")["round"].nunique().to_dict() == exp
        # OpenF1 lists zero-length stints (lap_end == lap_start - 1: tyre change in the pit lane before the
        # start or under a red flag, 43 of 5256 rows on 2026-11-16). They are kept as delivered; features must
        # use lap_end - lap_start + 1 and drop the zeros. Anything shorter than that is corrupt.
        # One row is worse (2025 R1 lawson stint 3: lap_start 3, lap_end 1, source data); at most 2 such rows allowed.
        assert stints["lap_start"].min() >= 1
        assert (stints["lap_end"] < stints["lap_start"] - 1).sum() <= 2
        assert ((stints["lap_end"] - stints["lap_start"]) == -1).mean() < 0.02
        assert stints["stint_number"].min() >= 1


def test_disk_2023_cross_source_agreement(disk):
    laps = disk[0]
    d = laps[(laps["season"] == 2023) & (laps["lap"] > 1)]
    a = d[d["source"] == "jolpica"][["round", "driverId", "lap", "lap_time_s"]]
    b = d[d["source"] == "openf1"][["round", "driverId", "lap", "lap_time_s"]]
    m = a.merge(b, on=["round", "driverId", "lap"], suffixes=("_j", "_o")).dropna(subset=["lap_time_s_j", "lap_time_s_o"])
    assert len(m) > 15000
    share = ((m["lap_time_s_j"] - m["lap_time_s_o"]).abs() <= float(I.cfg()["agreement"]["tol_s"])).mean()
    assert share >= 0.97, share
    per_race = m.assign(ok=(m["lap_time_s_j"] - m["lap_time_s_o"]).abs() <= 0.05).groupby("round")["ok"].mean()
    assert per_race.min() >= 0.9, per_race.sort_values().head(3).to_dict()
    # coverage: both sources hold (nearly) the same driver-laps
    cov = len(m) / min(len(a), len(b))
    assert cov > 0.95, cov
    # the module's own helper agrees with the independent computation once lap 1 is excluded
    ag = I.lap_agreement(d, 0.05, 2023)
    assert (ag["share"] * ag["n_both"]).sum() / ag["n_both"].sum() == pytest.approx(share, abs=1e-9)


def test_disk_cached_lap_files_vs_race_counts():
    """Informational completeness check: cached files never exceed the race count; no file is an unexplained empty."""
    d = I.RAW3 / "jolpica_laps"
    if not d.exists():
        pytest.skip("no jolpica laps cached yet")
    fl = json.loads(I.FAIL_FILE.read_text()) if I.FAIL_FILE.exists() else {"failed": {}, "empty": {}}
    for s in range(2014, 2023):
        n = len(list(d.glob(f"laps_{s}_*.json")))
        assert n <= len(I.season_races(s))
        for f in d.glob(f"laps_{s}_*.json"):
            if not json.loads(f.read_text()):
                assert f"jolpica_laps/{s}/{int(f.stem.split('_')[2])}" in fl["empty"], f"{f.name} empty but not listed"


def test_disk_failures_json_well_formed():
    if not I.FAIL_FILE.exists():
        pytest.skip("no failures file")
    fl = json.loads(I.FAIL_FILE.read_text())
    assert set(fl) == {"failed", "empty"}
    for k, v in {**fl["failed"], **fl["empty"]}.items():
        assert "time" in v and "reason" in v, k
    for key in fl["empty"]:
        kind, s, r = key.split("/")
        race = next((x for x in I.season_races(int(s)) if int(x["round"]) == int(r)), None)
        assert race is not None and I.is_run(race, I.cfg()), f"{key} cached as empty for a race not yet run"
