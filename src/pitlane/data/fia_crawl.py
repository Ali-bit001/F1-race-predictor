"""Download the FIA "Car Presentation Submissions" PDFs for M10 (PLAN3 Amendment 3 items 1-3).

Listing: season HTML page -> event ids; per event GET /decision-document-list/ajax/<id> (Drupal AJAX
JSON) -> document rows (title, published time, PDF href). Only the exact title "Car Presentation
Submissions" is taken; the href comes from the listing, never built. PDFs go to data/raw/fia/
(git-ignored); a manifest with URL and SHA-256 is written to methodologies/M10_upgrades/documents.csv.
Events recorded as not found are asked again on every run (a document may appear later).
Polite: one request at a time, >= 10 s apart (robots.txt Crawl-delay), generic User-Agent, resumable.

Usage: python -m pitlane.data.fia_crawl [--seasons 2024 2025 2026] [--max-docs 150]
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from pitlane.config import ROOT

BASE = "https://www.fia.com"
SEASON_PAGES = {
    2024: "/documents/championships/fia-formula-one-world-championship-14/season/season-2024-2043",
    2025: "/documents/championships/fia-formula-one-world-championship-14/season/season-2025-2071",
    2026: "/documents/championships/fia-formula-one-world-championship-14/season/season-2026-2072",
}
TITLE = "Car Presentation Submissions"
USER_AGENT = "research-client/0.1"
DELAY_S = 10.0
RAW_DIR = ROOT / "data" / "raw" / "fia"
MANIFEST = ROOT / "methodologies" / "M10_upgrades" / "documents.csv"
MAX_BYTES = 400 * 2**20

_last = [0.0]


def _get(path: str, ajax: bool = False, tries: int = 3) -> bytes:
    url = path if path.startswith("http") else BASE + urllib.parse.quote(path, safe="/:%")
    headers = {"User-Agent": USER_AGENT}
    if ajax:
        headers["X-Requested-With"] = "XMLHttpRequest"
    for k in range(tries):
        wait = DELAY_S - (time.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            _last[0] = time.monotonic()
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return r.read()
        except Exception as e:  # network error or HTTP error: back off, at most two retries
            if k == tries - 1:
                raise RuntimeError(f"failed after {tries} tries: {url}: {e}") from e
            time.sleep(DELAY_S * (k + 2))
    raise AssertionError


def parse_season(page: str) -> list[tuple[int, str]]:
    """(event_id, event name) in page order (newest first)."""
    pat = r'<a href="/decision-document-list/nojs/(\d+)"[^>]*class="event-title[^"]*"[^>]*>\s*(.*?)\s*</a>'
    # the newest event is expanded inline: a div title, its id only on the document wrapper
    act = r'<div class="event-title active">\s*(.*?)\s*</div>\s*<ul class="document-type-wrapper[^"]*data-id-(\d+)'
    out = [(int(i), n) for n, i in re.findall(act, page, re.S)]
    out += [(int(i), n) for i, n in re.findall(pat, page, re.S)]
    out = [(i, html.unescape(re.sub(r"\s+", " ", n)).strip()) for i, n in out]
    seen: set[int] = set()
    return [x for x in out if not (x[0] in seen or seen.add(x[0]))]


def parse_event(ajax_json: bytes) -> list[dict]:
    """Document rows from the AJAX response: title, published (as shown, CET), href."""
    cmds = json.loads(ajax_json)
    h = "".join(c.get("data", "") for c in cmds if isinstance(c, dict) and isinstance(c.get("data"), str))
    rows = []
    for chunk in re.split(r'<li class="document-row', h)[1:]:
        href = re.search(r'href="([^"]+\.pdf)"', chunk, re.I)
        text = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", chunk)))
        pub = re.search(r"Published on (\d\d\.\d\d\.\d\d \d\d:\d\d)", text)
        title = re.sub(r'^\s*key-\d+">\s*', "", text.split("Published on")[0]).strip()
        title = re.sub(r"^Doc\s+\d+\s*-\s*", "", title).strip()
        key = re.match(r'\s*key-(\d+)', chunk)
        rows.append({"title": title, "published_cet": pub.group(1) if pub else "",
                     "href": html.unescape(href.group(1)) if href else "", "key": int(key.group(1)) if key else -1})
    return rows


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def crawl(seasons: list[int], max_docs: int = 150) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame()
    rows = old.to_dict("records") if len(old) else []
    # Only events with a stored document are skipped. An event recorded as found=False is asked again on every run (a document
    # can be published later); its old not-found row is replaced by the new result.
    done = {r["event_id"] for r in rows if str(r.get("found")).lower() == "true"}
    total = sum(int(r["bytes"]) for r in rows if r.get("bytes") == r.get("bytes") and r.get("bytes"))
    for season in seasons:
        events = parse_season(_get(SEASON_PAGES[season]).decode("utf-8", "replace"))[::-1]  # oldest first
        print(f"{season}: {len(events)} events", flush=True)
        for order, (eid, name) in enumerate(events, start=1):
            if eid in done:
                continue
            rows = [r for r in rows if r["event_id"] != eid]   # drop the earlier not-found row of this event, if any
            docs = [d for d in parse_event(_get(f"/decision-document-list/ajax/{eid}", ajax=True))
                    if d["title"].strip().lower() == TITLE.lower()]
            if not docs:
                rows.append({"season": season, "page_order": order, "event": name, "event_id": eid, "found": False})
                print(f"  {name}: none", flush=True)
                _write(rows)
                continue
            for d in docs:
                b = _get(d["href"])
                if not b.startswith(b"%PDF"):
                    raise RuntimeError(f"not a PDF: {d['href']}")
                sha = _sha(b)
                fn = RAW_DIR / f"{season}_{eid}_{d['key']}.pdf"
                fn.write_bytes(b)
                total += len(b)
                rows.append({"season": season, "page_order": order, "event": name, "event_id": eid, "found": True,
                             "doc_key": d["key"], "published_cet": d["published_cet"], "url": BASE + d["href"],
                             "sha256": sha, "bytes": len(b), "file": fn.name})
                print(f"  {name}: {len(b)/1e6:.2f} MB", flush=True)
            _write(rows)
            n_docs = sum(1 for r in rows if r.get("found") is True)
            if n_docs > max_docs or total > MAX_BYTES:
                raise RuntimeError(f"stopping: {n_docs} documents, {total/2**20:.0f} MiB exceed the agreed limit")
    return pd.DataFrame(rows)


def _write(rows: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (int(r["season"]), int(r["page_order"])))   # stable: documents of one event keep their order
    pd.DataFrame(ordered).to_csv(MANIFEST, index=False, lineterminator="\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--max-docs", type=int, default=150)
    a = ap.parse_args()
    df = crawl(a.seasons, a.max_docs)
    f = df[df["found"] == True]  # noqa: E712
    print(f"done: {len(f)} documents, {f['bytes'].sum()/2**20:.1f} MiB; events without one: {(df['found'] == False).sum()}")
