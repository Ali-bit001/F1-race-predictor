"""PLAN3 R9 / M10 part 1: FIA crawler (src/pitlane/data/fia_crawl.py) and the document manifest.

No network: every crawler test uses synthetic HTML/JSON strings and a mocked `_get`.
The manifest tests read methodologies/M10_upgrades/documents.csv (committed); the file-hash tests need the
local, git-ignored PDFs in data/raw/fia and skip when they are absent.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request

import pandas as pd
import pytest

from pitlane.config import ROOT
from pitlane.data import fia_crawl as C

# --- synthetic pages ---------------------------------------------------------------------------------------------

SEASON_HTML = """
<html><body>
<div class="event-title active">
   Newest   Grand Prix &amp; Co
</div>
<ul class="document-type-wrapper foo data-id-300"><li>x</li></ul>
<a href="/decision-document-list/nojs/200" rel="nofollow" class="event-title">Middle Grand Prix</a>
<a href="/decision-document-list/nojs/100" class="event-title other">
   Oldest Grand Prix </a>
<a href="/decision-document-list/nojs/300" class="event-title">Newest Grand Prix again</a>
<a href="/decision-document-list/nojs/999" class="not-an-event">Menu link</a>
</body></html>
"""


def ajax(rows: list[tuple[int, str, str, str | None]]) -> bytes:
    """Drupal-style AJAX response: a list of commands, one of which carries an HTML string in 'data'."""
    li = ""
    for key, title, pub, href in rows:
        a = f'<a href="{href}">pdf</a>' if href else ""
        li += (f'<li class="document-row key-{key}"><div class="t">{title}</div>'
               f'<span>Published on {pub}</span>{a}</li>')
    return json.dumps([{"command": "settings", "data": {"not": "a string"}},
                       {"command": "insert", "data": f"<ul>{li}</ul>"}]).encode()


# --- parse_season ------------------------------------------------------------------------------------------------

def test_parse_season_finds_active_div_and_nojs_links_deduplicated_in_page_order():
    got = C.parse_season(SEASON_HTML)
    # the active (expanded) event first, then nojs links in page order; id 300 appears twice and is kept once
    assert [i for i, _ in got] == [300, 200, 100]
    # name: whitespace collapsed and entities unescaped; the first occurrence (the active div) wins
    assert got[0][1] == "Newest Grand Prix & Co"
    assert got[1][1] == "Middle Grand Prix" and got[2][1] == "Oldest Grand Prix"


def test_parse_season_ignores_links_without_event_title_class():
    assert 999 not in [i for i, _ in C.parse_season(SEASON_HTML)]


def test_parse_season_without_active_div_and_empty_page():
    html = '<a href="/decision-document-list/nojs/5" class="event-title">A</a><a href="/decision-document-list/nojs/4" class="event-title">B</a>'
    assert C.parse_season(html) == [(5, "A"), (4, "B")]
    assert C.parse_season("<html></html>") == []


# --- parse_event -------------------------------------------------------------------------------------------------

def test_parse_event_title_published_href_key_and_doc_prefix_stripped():
    rows = C.parse_event(ajax([
        (9, "Doc 9 - Car Presentation Submissions", "07.03.24 12:20", "/sites/default/files/decision-document/a b.pdf"),
        (11, "Doc 12 - Car Presentation and Display Procedure", "01.01.24 00:00", "/x/y.pdf")]))
    assert len(rows) == 2
    r = rows[0]
    assert r["title"] == "Car Presentation Submissions"           # "Doc 9 -" removed
    assert r["published_cet"] == "07.03.24 12:20"
    assert r["href"] == "/sites/default/files/decision-document/a b.pdf"
    assert r["key"] == 9
    assert rows[1]["title"] == "Car Presentation and Display Procedure" and rows[1]["key"] == 11


def test_parse_event_title_without_doc_prefix_and_html_entities_and_missing_fields():
    rows = C.parse_event(ajax([(3, "Car Presentation Submissions", "01.02.25 03:04", "/a.pdf"),
                               (4, "Doc 4 - Sao &amp; Co", "bad date", None)]))
    assert rows[0]["title"] == "Car Presentation Submissions"
    assert rows[1]["title"] == "Sao & Co"
    assert rows[1]["published_cet"] == "" and rows[1]["href"] == ""   # missing, not invented


def test_parse_event_empty_and_non_string_commands():
    assert C.parse_event(json.dumps([{"command": "x", "data": 3}, "text"]).encode()) == []
    with pytest.raises(json.JSONDecodeError):
        C.parse_event(b"<html>not json</html>")


# --- crawl: exact-title filter, skip of known events, no network ---------------------------------------------------

PDF = b"%PDF-1.4 synthetic"


@pytest.fixture
def crawl_env(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MANIFEST", tmp_path / "documents.csv")
    monkeypatch.setattr(C, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(C, "SEASON_PAGES", {2030: "/season-2030"})
    calls: list[str] = []
    pages = {
        "/season-2030": SEASON_HTML.encode(),
        "/decision-document-list/ajax/300": ajax([
            (9, "Doc 9 - Car Presentation Submissions", "07.03.30 12:20", "/files/new.pdf"),
            (8, "Doc 8 - Car Presentation and Display Procedure", "07.03.30 12:00", "/files/display.pdf")]),
        "/decision-document-list/ajax/200": ajax([
            (5, "Doc 5 - Car Presentation Submissions", "01.03.30 10:00", "/files/mid.pdf"),
            (6, "Doc 6 - car presentation submissions", "01.03.30 11:00", "/files/mid2.pdf"),
            (7, "Doc 7 - Car Presentation Submissions (Revised)", "01.03.30 12:00", "/files/mid3.pdf")]),
        "/decision-document-list/ajax/100": ajax([(1, "Doc 1 - Something else", "01.02.30 10:00", "/files/other.pdf")]),
        "/files/new.pdf": PDF + b"new", "/files/mid.pdf": PDF + b"mid", "/files/mid2.pdf": PDF + b"mid2",
        "/files/display.pdf": PDF + b"display", "/files/mid3.pdf": PDF + b"mid3", "/files/other.pdf": PDF + b"other",
    }

    def fake_get(path, ajax=False, tries=3):
        calls.append(path)
        return pages[path]

    monkeypatch.setattr(C, "_get", fake_get)
    return calls, pages, tmp_path


def test_crawl_exact_title_filter_excludes_display_procedure_revised_and_other_titles(crawl_env):
    calls, _, tmp = crawl_env
    df = C.crawl([2030])
    found = df[df["found"] == True]  # noqa: E712
    assert sorted(found["doc_key"].astype(int)) == [5, 6, 9]          # case-insensitive exact match only
    assert "/files/display.pdf" not in calls and "/files/mid3.pdf" not in calls and "/files/other.pdf" not in calls
    # event 100 has no matching document: recorded as not found, nothing downloaded for it
    nf = df[df["found"] == False]  # noqa: E712
    assert nf["event_id"].tolist() == [100]
    # files are named from season, event id and doc key; sha256 recorded matches the bytes written
    for r in found.itertuples():
        f = tmp / "raw" / r.file
        assert f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == r.sha256
        assert r.file == f"2030_{r.event_id}_{int(r.doc_key)}.pdf"


def test_crawl_page_order_is_oldest_first_and_manifest_is_written(crawl_env):
    _, _, tmp = crawl_env
    df = C.crawl([2030])
    order = df.drop_duplicates("event_id").set_index("event_id")["page_order"].to_dict()
    assert order == {100: 1, 200: 2, 300: 3}
    on_disk = pd.read_csv(tmp / "documents.csv")
    assert len(on_disk) == len(df) and on_disk["url"].dropna().str.startswith("https://www.fia.com/").all()


def test_crawl_skips_events_with_a_stored_document_without_any_request(crawl_env):
    calls, pages, tmp = crawl_env
    pd.DataFrame([{"season": 2030, "page_order": 2, "event": "Middle Grand Prix", "event_id": 200, "found": True,
                   "doc_key": 5, "published_cet": "x", "url": "u", "sha256": "s", "bytes": 10, "file": "f.pdf"}]
                 ).to_csv(tmp / "documents.csv", index=False)
    df = C.crawl([2030])
    assert "/decision-document-list/ajax/200" not in calls
    assert "/decision-document-list/ajax/300" in calls
    assert set(df["event_id"]) == {100, 200, 300}
    assert (df["event_id"] == 200).sum() == 1          # the old row is kept once, not duplicated


def test_crawl_polls_events_recorded_as_not_found_again_and_replaces_the_row(crawl_env):
    """R9 round 4: a not-found event is asked again on every run (a document may be published later). Event 100 has no
    matching document in the fake listing: still not found, one row, polled again. Event 200 is recorded as not found in the
    old manifest but now has documents: its old row is replaced by the found rows, no duplicate."""
    calls, _, tmp = crawl_env
    pd.DataFrame([{"season": 2030, "page_order": 1, "event": "Oldest Grand Prix", "event_id": 100, "found": False},
                  {"season": 2030, "page_order": 2, "event": "Middle Grand Prix", "event_id": 200, "found": False}]
                 ).to_csv(tmp / "documents.csv", index=False)
    df = C.crawl([2030])
    assert "/decision-document-list/ajax/100" in calls and "/decision-document-list/ajax/200" in calls
    assert (df["event_id"] == 100).sum() == 1 and not df[df["event_id"] == 100]["found"].any()
    mid = df[df["event_id"] == 200]
    assert len(mid) == 2 and mid["found"].all() and sorted(mid["doc_key"].astype(int)) == [5, 6]
    on_disk = pd.read_csv(tmp / "documents.csv")
    assert on_disk["page_order"].tolist() == sorted(on_disk["page_order"])     # manifest stays in page order


def test_crawl_does_not_poll_a_found_event_again_after_a_second_run(crawl_env):
    calls, _, _ = crawl_env
    C.crawl([2030])
    calls.clear()
    C.crawl([2030])
    assert "/decision-document-list/ajax/200" not in calls and "/decision-document-list/ajax/300" not in calls
    assert "/decision-document-list/ajax/100" in calls     # the one not-found event is asked again


def test_crawl_rejects_a_download_that_is_not_a_pdf(crawl_env):
    _, pages, _ = crawl_env
    pages["/files/new.pdf"] = b"<html>login wall</html>"
    with pytest.raises(RuntimeError, match="not a PDF"):
        C.crawl([2030])


def test_crawl_stops_when_document_limit_is_exceeded(crawl_env):
    with pytest.raises(RuntimeError, match="exceed the agreed limit"):
        C.crawl([2030], max_docs=1)


# --- politeness: User-Agent, delay, no real network -----------------------------------------------------------------

class _Resp:
    def __init__(self, b: bytes):
        self.b = b

    def read(self):
        return self.b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _capture(monkeypatch):
    seen, slept = [], []
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: (seen.append(req), _Resp(b"ok"))[1])
    monkeypatch.setattr(C.time, "sleep", lambda s: slept.append(s))
    return seen, slept


def test_user_agent_is_generic_and_delay_is_at_least_ten_seconds():
    assert C.USER_AGENT == "research-client/0.1"
    assert C.DELAY_S >= 10
    assert not re.search(r"@|\.com|mozilla", C.USER_AGENT, re.I)      # no contact details, no browser spoofing


def test_get_sends_generic_user_agent_ajax_header_and_quotes_the_path(monkeypatch):
    seen, _ = _capture(monkeypatch)
    C._last[0] = 0.0
    C._get("/sites/decision-document/2024 São Paulo Grand Prix - Car Presentation Submissions.pdf", ajax=True)
    req = seen[0]
    assert req.get_header("User-agent") == "research-client/0.1"
    assert req.get_header("X-requested-with") == "XMLHttpRequest"
    assert req.full_url.startswith("https://www.fia.com/") and " " not in req.full_url
    assert "S%C3%A3o" in req.full_url or "S%C3%A3o".lower() in req.full_url.lower()
    C._get("/plain")
    assert seen[1].get_header("X-requested-with") is None


def test_get_waits_out_the_delay_between_requests(monkeypatch):
    _, slept = _capture(monkeypatch)
    C._last[0] = C.time.monotonic()           # a request has just been made
    C._get("/a")
    assert slept and slept[0] > C.DELAY_S - 1.0


def test_get_does_not_double_quote_an_already_encoded_path(monkeypatch):
    seen, _ = _capture(monkeypatch)
    C._last[0] = 0.0
    C._get("/a%20b.pdf")
    assert seen[0].full_url.endswith("/a%20b.pdf")


def test_get_retries_then_raises_and_backs_off(monkeypatch):
    slept: list[float] = []

    def boom(req, timeout=None):
        raise OSError("down")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(C.time, "sleep", lambda s: slept.append(s))
    C._last[0] = 0.0
    with pytest.raises(RuntimeError, match="failed after 3 tries"):
        C._get("/x")
    assert sum(1 for s in slept if s >= C.DELAY_S) >= 2           # back-off sleeps between tries


# --- manifest integrity -----------------------------------------------------------------------------------------------

MAN = pd.read_csv(C.MANIFEST, dtype={"sha256": str, "file": str, "published_cet": str, "url": str})
FOUND = MAN[MAN["found"] == True].copy()  # noqa: E712
LOCAL = ROOT / "data" / "raw" / "fia"
HAVE_ALL = all((LOCAL / f).exists() for f in FOUND["file"])


def test_manifest_counts_67_found_and_3_not_found():
    assert len(FOUND) == 67 and (MAN["found"] == False).sum() == 3  # noqa: E712
    nf = MAN[MAN["found"] == False]  # noqa: E712
    assert [(int(s), e) for s, e in zip(nf["season"], nf["event"])] == [
        (2024, "Bahrain Tests Season"), (2024, "Bahrain Grand Prix"), (2025, "Bahrain Tests Season 2025")]
    assert FOUND.groupby("season").size().to_dict() == {2024: 23, 2025: 24, 2026: 20}


def test_manifest_no_duplicates_and_complete_fields_for_found_rows():
    assert FOUND["event_id"].is_unique and FOUND["sha256"].is_unique and FOUND["file"].is_unique and FOUND["url"].is_unique
    assert FOUND[["doc_key", "published_cet", "url", "sha256", "bytes", "file"]].notna().all().all()
    assert FOUND["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert (FOUND["bytes"] > 10_000).all()
    assert (FOUND["file"] == [f"{s}_{e}_{int(k)}.pdf" for s, e, k in zip(FOUND["season"], FOUND["event_id"], FOUND["doc_key"])]).all()
    # a not-found row carries no document fields
    nf = MAN[MAN["found"] == False]  # noqa: E712
    assert nf[["doc_key", "published_cet", "url", "sha256", "file"]].isna().all().all()


def test_manifest_every_url_is_on_www_fia_com_and_is_a_pdf():
    assert FOUND["url"].str.match(r"^https://www\.fia\.com/[^ ]").all() or FOUND["url"].str.startswith("https://www.fia.com/").all()
    assert FOUND["url"].str.lower().str.endswith(".pdf").all()
    # the exact-title filter held (file names use spaces in 2024 and underscores in 2025-26)
    norm = FOUND["url"].str.lower().str.replace(r"[^a-z]", "", regex=True)
    assert norm.str.contains("carpresentationsubmissions").all()
    assert not norm.str.contains("displayprocedure").any()


def test_manifest_published_dates_are_in_season_and_ordered_by_page_order():
    dt = pd.to_datetime(FOUND["published_cet"], format="%d.%m.%y %H:%M")
    assert (dt.dt.year == FOUND["season"]).all()
    for season, g in FOUND.assign(dt=dt).groupby("season"):
        g = g.sort_values("page_order")
        assert g["page_order"].is_unique
        assert g["dt"].is_monotonic_increasing, f"{season}: published time not ordered by page_order"
        # the first race weekend is in March, the last before the winter
        assert g["dt"].min() >= pd.Timestamp(season, 3, 1) and g["dt"].max() <= pd.Timestamp(season, 12, 31)


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_manifest_sha256_of_every_file_matches_and_folder_holds_exactly_the_manifest_files():
    for r in FOUND.itertuples():
        assert hashlib.sha256((LOCAL / r.file).read_bytes()).hexdigest() == r.sha256, r.file
        assert (LOCAL / r.file).read_bytes().startswith(b"%PDF") and (LOCAL / r.file).stat().st_size == int(r.bytes)
    assert {p.name for p in LOCAL.iterdir()} == set(FOUND["file"])
