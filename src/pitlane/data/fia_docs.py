"""PLAN3 R9 / M10 part 1: parser for FIA "Car Presentation Submissions" PDFs (Amendment 3 items 1-3).

One PDF = one cover page, then one page per team (a team may also continue over a following page that starts
directly with the table header). Each team page is either a table with one row per updated component, or a
"no updates" line. Rows are counted by component cells: a row is anchored on its reason cell
("<group> - <subtype>") and its component is the text from the nearest row-number token up to that anchor.
The printed row number is never used for counting (2024 Saudi has wrong and repeated numbers).

The FIA files are not stored in git; this module reads them from data/raw/fia and writes
data/interim/m10_car_presentations.parquet (both ignored). Settings and keyword rules are in
methodologies/M10_upgrades/config.yaml. No network access here.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pitlane.config import EXTERNAL, METHODS, ROOT, load_yaml

NO_SUBTYPE = "\x00nosub"   # marker: reason group with no subtype (Reliability, Weight, Cost)
FREE_SUBTYPE = "\x00free"  # R10 parsing fix: group printed without dash, free text after it (subtype not separable)
HDR_MARK ="\x01hdr"       # R10 parsing fix: position of the page-header line (the team name may sit before or after it)
CONFIG_PATH = METHODS / "M10_upgrades" / "config.yaml"
COLUMNS = ["season", "event", "event_key", "team_raw", "team", "constructorId", "lineage", "component",
           "category", "category_secondary", "reason_group", "reason_subtype", "subtype_known", "optional",
           "no_updates", "parse_ok", "parse_status", "raw_text", "page", "doc_file", "doc_sha256", "manifest_season", "manifest_event",
           "event_id", "published_cet"]


def load_cfg(path: Path | str = CONFIG_PATH) -> dict:
    return load_yaml(path)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def _letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _page_lines(text: str) -> list[str]:
    return [ln for ln in (_clean(x) for x in text.split("\n")) if ln]


# --- team names -----------------------------------------------------------------------------------------------

def normalise_team(raw: str) -> str:
    """Lower case, asterisks and punctuation removed, spaces collapsed."""
    return _clean(re.sub(r"[^a-z0-9 ]", " ", raw.lower().replace("*", " ")))


def lineage_map() -> dict[str, str]:
    df = pd.read_csv(EXTERNAL / "lineage.csv")
    return dict(zip(df["constructorId"], df["lineage"]))


def map_team(raw: str, cfg: dict, lineage: dict[str, str] | None = None) -> tuple[str, str | None, str | None]:
    """(normalised name, constructorId or None, lineage or None)."""
    name = normalise_team(raw)
    lin = lineage_map() if lineage is None else lineage
    for pattern, cid in cfg["team_aliases"]:
        if re.search(pattern, name):
            return name, cid, lin.get(cid, cid)
    return name, None, None


# --- component category ---------------------------------------------------------------------------------------

def categorise_all(component: str, cfg: dict) -> list[str]:
    """Every category whose keywords match, in config order (empty if none)."""
    text = _clean(component).lower()
    return [cat for cat, patterns in cfg["component_categories"] if any(re.search(p, text) for p in patterns)]


def categorise(component: str, cfg: dict) -> str:
    """First matching category (unchanged rule), 'other' if none."""
    hits = categorise_all(component, cfg)
    return hits[0] if hits else "other"


def categorise_secondary(component: str, cfg: dict) -> str:
    """R10 parsing fix: second matched category of a combined component ('Rear Wing / Beam Wing'), else ''."""
    hits = categorise_all(component, cfg)
    return hits[1] if len(hits) > 1 else ""


# --- one team table -------------------------------------------------------------------------------------------

@dataclass
class _Anchor:
    start: int                # index of the first line of the row (number token line)
    component: str
    group: str
    subtype: str
    subtype_known: bool
    end: int                  # index of the last line consumed by the reason cell


def _find_anchor_line(line: str, cfg: dict) -> tuple[str, str, str] | None:
    """(text before the group, group, text after the dash) if the line holds '<group> -'."""
    p = cfg["parser"]
    groups = "|".join(re.escape(g) for g in p["reason_groups"])
    m = re.search(rf"^(?P<pre>|\d+\s+.*?\s|\S.*?\s)(?P<g>{groups})\s*[{p['dash_chars']}]\s*(?P<rest>.*)$", line, re.I)
    if not m:
        return _find_anchor_nodash(line, cfg)
    pre = m.group("pre").strip()
    # Prose such as "improves aerodynamic Performance - ..." must not anchor: a prefix is accepted only when
    # the line starts with a row number or the prefix is short (a wrapped component cell).
    if pre and not re.match(r"^\d+(\s|$)", pre) and len(pre.split()) > 4:
        return None
    return pre, m.group("g"), m.group("rest").strip()


def _find_anchor_nodash(line: str, cfg: dict) -> tuple[str, str, str] | None:
    """Reason cells printed without a dash: groups with no subtype (Reliability / Weight / Cost / Cooling) and,
    R10 parsing fix, '<group> <subtype>' with the dash lost ('Performance Local Load', 'Circuit specific Modification ...')."""
    p = cfg["parser"]
    nos = "|".join(re.escape(g) for g in p["reason_groups_no_subtype"])
    strict = set(p.get("reason_groups_no_subtype_strict", []))
    for m in re.finditer(rf"(?<!\S)(?P<g>{nos})(?=\s|$)", line):
        pre = line[:m.start()].strip()
        if re.fullmatch(r"\d+", pre):
            continue   # R10 parsing fix: "2 Weight reduction ..." is a description line; a row needs a component
        if pre and not (re.match(r"^\d+\s", pre) or len(pre.split()) <= 3):
            continue
        if m.group("g") in strict:   # R10 parsing fix: 'Cooling' alone is also a word of prose and of 'Cooling Range'
            if not pre or re.match(r"Range\b", line[m.end():].strip()):
                continue
            if pre.split()[-1] in p.get("reason_group_blocked_after", {}).get(m.group("g"), []):
                continue   # R10 parsing fix: 'Driver Cooling' is another reason cell, not the group 'Cooling'; flagged, not guessed
        return pre, m.group("g"), NO_SUBTYPE
    groups = "|".join(re.escape(g) for g in p["reason_groups"] if g not in p["reason_groups_no_subtype"])
    m = re.search(rf"^(?P<pre>\d+\s+\S.*?)\s(?P<g>{groups})(?:\s+(?P<rest>\S.*))?$", line, re.I)
    if m and len(m.group("pre").split()) <= 6:
        g = _canon_group(m.group("g"), cfg)
        rest = (m.group("rest") or "").strip()
        if g in p.get("dashless_free_text_groups", []) and rest:
            return m.group("pre").strip(), g, FREE_SUBTYPE
        subs = p["reason_subtypes"]
        if any(rest.lower().startswith(k.lower()) or (rest and k.lower().startswith(rest.lower())) for k in subs):
            return m.group("pre").strip(), g, rest
    free = "|".join(re.escape(g) for g in p.get("dashless_free_text_groups", []))
    if free:   # the group (and the free text after it) on a line of its own, the row number on an earlier line
        m = re.search(rf"^(?P<pre>(?:\S+(?:\s+\S+){{0,2}})?)\s*(?P<g>{free})(?:\s+(?P<rest>(?-i:[A-Z]).*))?$", line, re.I)
        if m and not re.search(r"[.!?,]$", m.group("pre")) and not re.match(r"\d", line):
            return m.group("pre").strip(), _canon_group(m.group("g"), cfg), FREE_SUBTYPE
    return None


def _split_subtype(rest: str, following: list[str], cfg: dict) -> tuple[str, bool, int]:
    """Subtype text, whether it is a known subtype, and how many following lines it used (0, 1 or 2).
    R10 parsing fix: a subtype wrapped over two lines ('Local' / 'Load') is joined when the joined text starts with a
    configured subtype and the first part is a proper prefix of it."""
    if rest == NO_SUBTYPE:
        return "", True, 0
    if rest == FREE_SUBTYPE:
        return "", False, 0
    subs = cfg["parser"]["reason_subtypes"]
    text, used = rest, 0
    if not text and following:
        text, used = following[0], 1
    for k in subs:
        if text.lower().startswith(k.lower()):
            return k, True, used
    if text and len(following) > used:
        joined = _clean(text + " " + following[used])
        for k in subs:
            if k.lower().startswith(text.lower()) and joined.lower().startswith(k.lower()):
                return k, True, used + 1
    return text, False, used


def _row_start(lines: list[str], i: int, pre: str, floor: int, cfg: dict) -> tuple[int, str] | None:
    """Row start index and component text for the anchor on line i (pre = text before the group)."""
    back = cfg["parser"]["component_lookback_lines"]
    cand: list[tuple[int, str]] = [(j, lines[j]) for j in range(max(floor, i - back), i)]
    if pre:
        cand.append((i, pre))
    for pos in range(len(cand) - 1, -1, -1):
        j, text = cand[pos]
        m = re.match(r"^\d+(\s+(.*))?$", text)
        if m:
            parts = [(m.group(2) or "").strip()] + [t for _, t in cand[pos + 1:]]
            return j, _clean(" ".join(x for x in parts if x))
    for pos in range(len(cand) - 1, -1, -1):   # R10 parsing fix: row number mid-line after text with no full stop
        j, text = cand[pos]
        mm = list(re.finditer(r"\s(\d{1,2})\s+([A-Z].*)$", text))
        if mm:
            parts = [mm[-1].group(2)] + [t for _, t in cand[pos + 1:]]
            return j, _clean(" ".join(x for x in parts if x))
    return None


def _canon_group(group: str, cfg: dict) -> str:
    """R10 parsing fix: reason group in the configured spelling ('Circuit Specific' -> 'Circuit specific')."""
    p = cfg["parser"]
    return next((g for g in [*p["reason_groups"], *p["reason_groups_no_subtype"]] if g.lower() == group.lower()), group)


def is_cut_row(line: str, cfg: dict) -> bool:
    """R10 parsing fix: a short numbered line with a capitalised component and no reason cell ('3 Beam Wing')."""
    return re.fullmatch(cfg["parser"]["cut_row_regex"], line) is not None


@dataclass
class _Flag:
    """A unit of text that formed no parsed row and becomes one flagged row (R10 parsing fix)."""
    first: int                  # index of the first line (sort key)
    lines: list[str]
    kind: str                   # 'gap' (a printed row number with no reason cell) or 'orphan' (other leftover lines)
    component: str | None = None
    subtype: str | None = None
    text: str = ""              # printed text of a group_missing row (component, subtype and the geometry text)


def _find_anchors(lines: list[str], cfg: dict) -> list[_Anchor]:
    anchors: list[_Anchor] = []
    floor = 0
    i = 0
    while i < len(lines):
        hit = _find_anchor_line(lines[i], cfg)
        start = None
        if hit is not None:
            pre, group, rest = hit
            sub, known, used = _split_subtype(rest, lines[i + 1: i + 3], cfg)
            start = _row_start(lines, i, pre, floor, cfg)
            if start is None and rest != NO_SUBTYPE and re.fullmatch(r"[A-Z][^\d]*", pre) and len(pre.split()) <= 4:
                start = (i, pre)   # R10 parsing fix: the printed row number is missing ('Floor Fences Performance -')
        if hit is not None and start is not None and hit[2] == FREE_SUBTYPE and not re.match(r"\d", lines[i])                 and not _short_cell_lines(lines, start[0], i, cfg):
            start = None   # R10 parsing fix: 'Circuit specific rear wing ...' inside a description is not a reason cell
        if hit is None or start is None:
            i += 1
            continue
        anchors.append(_Anchor(start[0], start[1], group, sub, known, i + used))
        floor = i + used + 1
        i = floor
    return anchors


def _short_cell_lines(lines: list[str], first: int, i: int, cfg: dict) -> bool:
    """True if the lines from the row-number line up to the line before i look like wrapped component cells
    (short, no sentence end), so that a group on line i is the row's reason cell and not prose."""
    p = cfg["parser"]
    return all(len(x.split()) <= p["flag_continuation_max_tokens"] + 1 and not re.search(r"[.!?]$", x) for x in lines[first:i])


def _anchor_rows(lines: list[str], anchors: list[_Anchor], cfg: dict) -> list[dict[str, Any]]:
    rows = []
    for k, a in enumerate(anchors):
        stop = anchors[k + 1].start if k + 1 < len(anchors) else len(lines)
        seg = lines[a.end: stop]  # the reason line itself may carry the geometry text
        opt = any(re.search(cfg["parser"]["optional_regex"], s, re.I) for s in seg)
        rows.append(dict(component=a.component, reason_group=_canon_group(a.group, cfg), reason_subtype=a.subtype,
                         subtype_known=a.subtype_known, optional=opt))
    return rows


def parse_team_body(lines: list[str], cfg: dict, drop_lead: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
    """Rows of one team's table (lines after the header) and the lines that formed no row (dash / cut-row signals only).
    The production path is parse_team_units, which adds the row-number chain check."""
    anchors = _find_anchors(lines, cfg)
    used_lines = {j for a in anchors for j in range(a.start, a.end + 1)}
    return _anchor_rows(lines, anchors, cfg), _unused_prose(lines, anchors, used_lines, cfg, drop_lead)


def _chain_items(lines: list[str], last: int, max_skip: int = 1) -> list[tuple[int, int, int]]:
    """R10 parsing fix, row-number chain: (line, character position, number) of the printed row numbers 'last+1',
    'last+2', ... found in order. A candidate is a number at the start of a line (or alone on it) or after other
    text on a line, followed by a capitalised word. Numbers that do not continue the chain (wrong or repeated
    numbers, '2 Element' prose) are skipped, so the chain is a lower bound of the printed rows."""
    items: list[tuple[int, int, int]] = []
    for j, ln in enumerate(lines):
        for m in re.finditer(r"(?:^|(?<=\s))(\d{1,2})(?=\s+[A-Z]|$)", ln):
            n = int(m.group(1))
            # a number at the start of a line may skip up to chain_max_skip printed numbers (the PDF text of 2025 Australia
            # Red Bull goes 1, 2, 3, 6, 7 ...); a number inside a line must be exactly the next one
            if n == last + 1 or (m.start(1) == 0 and last < n <= last + max_skip):
                items.append((j, m.start(1), n))
                last = n
    return items


def _gap_flags(lines: list[str], anchors: list[_Anchor], items: list[tuple[int, int, int]], cfg: dict) -> list[_Flag]:
    """One flag per chain row number that no anchored row starts on (the reason cell is missing or not recognised)."""
    p = cfg["parser"]
    starts = sorted({a.start for a in anchors})
    flags: list[_Flag] = []
    for k, (j, pos, _) in enumerate(items):
        if j in starts:
            continue
        nxt = items[k + 1] if k + 1 < len(items) else None
        if nxt is not None and nxt[0] == j:
            first, limit = lines[j][pos:nxt[1]], j + 1
        else:
            first = lines[j][pos:]
            limit = min([nxt[0] if nxt else len(lines), *[s_ for s_ in starts if s_ > j]])
        out = [_clean(first)[: p["flag_text_max_chars"]]]
        for q in range(j + 1, min(limit, j + 1 + p["flag_continuation_lines"])):
            if re.match(r"[A-Z]", lines[q]) and len(lines[q].split()) <= p["flag_continuation_max_tokens"] \
                    and not re.search(r"[.!?]$", lines[q]):
                out.append(lines[q])
            else:
                break
        comp, sub, text = _group_missing(out[0], lines[j + 1] if j + 1 < limit else "", cfg)
        flags.append(_Flag(j, out, "gap", comp, sub, text))
    return flags


def _group_missing(first: str, nxt: str, cfg: dict) -> tuple[str | None, str | None, str]:
    """R10 parsing fix: '<n> <component> <known subtype> ...' with no reason group printed -> (component, subtype).
    The group is never inferred from the subtype."""
    subs = "|".join(re.escape(k) for k in sorted(cfg["parser"]["reason_subtypes"], key=len, reverse=True))
    text = _clean(re.sub(r"^\d+\s*", "", first) + " " + nxt)
    m = re.search(rf"\b(?P<s>{subs})\b", text, re.I)
    if not m:
        return None, None, ""
    comp = text[: m.start()].strip()
    if not comp or not re.match(r"[A-Z]", comp) or re.search(r"[.!?:;,\-–—�]", comp) \
            or len(comp.split()) > cfg["parser"]["flag_component_max_words"]:
        return None, None, ""
    sub = next(k for k in cfg["parser"]["reason_subtypes"] if k.lower() == m.group("s").lower())
    return comp, sub, text[: cfg["parser"]["flag_text_max_chars"]]


def parse_team_units(lines: list[str], cfg: dict, drop_lead: bool = False, prev_last: int = 0
                     ) -> tuple[list[dict[str, Any]], list[_Flag], int]:
    """Rows of one team's table, the flagged units, and the last printed row number reached by the chain.
    R10 parsing fix: every printed row number of the chain 1..N is either an anchored row or a flagged unit,
    so a numbered row can no longer vanish with parse_ok True on the team's other rows."""
    anchors = _find_anchors(lines, cfg)
    rows = _anchor_rows(lines, anchors, cfg)
    items = _chain_items(lines, prev_last, cfg["parser"]["chain_max_skip"])
    gaps = _gap_flags(lines, anchors, items, cfg)
    used = {j for a in anchors for j in range(a.start, a.end + 1)}
    for g in gaps:
        used |= set(range(g.first, g.first + len(g.lines)))
    first_anchor = anchors[0].start if anchors else len(lines)
    lead_gap = any(g.first < first_anchor for g in gaps)
    orphan = _unused_prose(lines, anchors, used, cfg, drop_lead or lead_gap, only_dash=not anchors and bool(gaps))
    flags = list(gaps)
    if orphan:
        flags.append(_Flag(next((j for j, x in enumerate(lines) if x == orphan[0]), 0), orphan, "orphan"))
    flags.sort(key=lambda f: f.first)
    return rows, flags, items[-1][2] if items else prev_last


def _unused_prose(lines: list[str], anchors: list[_Anchor], used: set[int], cfg: dict,
                  drop_lead: bool = False, only_dash: bool = False) -> list[str]:
    """Lines that signal a row the parser could not anchor: text before the first row, and any line outside
    the rows that ends in a dash (a reason cell with a group that is not in the config).
    R10 parsing fix: a dash-ended line counts only if it starts with a row number or is short (a reason-cell
    wrap); a prose line that happens to end in a spaced dash is the description of the row above."""
    p = cfg["parser"]
    dash_end = re.compile(rf"\s[{p['dash_chars']}]\s*$")  # spaced dash; "re-" wraps are prose

    def signal(x: str) -> bool:
        if dash_end.search(x):
            return bool(re.match(r"\d+\s", x)) or len(x.split()) <= p["dash_flag_max_tokens"]
        return is_cut_row(x, cfg)

    if not anchors:
        if drop_lead:
            return []
        return [x for j, x in enumerate(lines) if j not in used and (not only_dash or signal(x))]
    out = [] if drop_lead else [lines[j] for j in range(0, anchors[0].start) if j not in used]
    out += [lines[j] for j in range(anchors[0].start, len(lines)) if j not in used and signal(lines[j])]
    return out


def _strip_table_header(lines: list[str], cfg: dict) -> list[str]:
    pats = [re.compile(p, re.I) for p in cfg["parser"]["table_header_regexes"]]
    j = 0
    while j < len(lines) and any(p.search(lines[j]) for p in pats):
        j += 1
    return lines[j:]


# --- one document ---------------------------------------------------------------------------------------------

@dataclass
class ParsedDoc:
    season: int | None
    event: str | None
    event_key: str | None
    is_car_presentation: bool
    rows: list[dict[str, Any]]


def parse_cover(text: str, cfg: dict) -> tuple[bool, int | None, str | None]:
    """(is a Car Presentation Submissions document, season, event name lower-cased)."""
    lines = [_clean(ln.replace("�", " ")) for ln in _page_lines(text)]
    is_cp = any(re.search(rf"^{cfg['parser']['title_regex']}$", ln, re.I) for ln in lines)
    for ln in lines:
        m = re.search(cfg["parser"]["cover_event_regex"], ln, re.I)
        if m:
            return is_cp, int(m.group("season")), _clean(m.group("event")).lower()
    return is_cp, None, None


# --- flat pages -----------------------------------------------------------------------------------------------
# Some PDFs (2024 onwards) come out of pypdf with a whole table, or a team name plus its "no updates" line, on
# one text line. The functions below cut such lines back into the one-cell-per-line shape the parser expects.

def _split_rows(line: str, cfg: dict) -> str:
    """Insert line breaks before each row number and after each '<group> - <known subtype>' in a flat line."""
    p = cfg["parser"]
    groups = "|".join(re.escape(g) for g in p["reason_groups"])
    subs = "|".join(re.escape(k) for k in p["reason_subtypes"])
    strict = set(p.get("reason_groups_no_subtype_strict", []))   # R10 parsing fix: 'Cooling' alone never splits a flat line
    nos = "|".join(re.escape(g) for g in p["reason_groups_no_subtype"] if g not in strict)
    rx = re.compile(rf"(?P<g>{groups})\s*[{p['dash_chars']}](?:\s*(?P<s>{subs}))?|(?P<n>(?-i:{nos}))\b(?!\s*[{p['dash_chars']}])", re.I)
    back = p["flat_component_lookback_tokens"]
    cuts: list[int] = []
    last = 0
    for m in rx.finditer(line):
        toks = list(re.finditer(r"\S+", line[last:m.start()]))
        num = next((t for t in reversed(toks[-back:]) if re.fullmatch(r"\d+", t.group())), None)
        if num is None:
            continue
        cuts += [last + num.start(), m.end()]   # no known subtype: the cell (dash) ends the row header line
        last = m.end()
    for c in sorted(set(cuts), reverse=True):
        line = line[:c] + "\n" + line[c:]
    return line


def expand_flat_lines(lines: list[str], cfg: dict) -> list[str]:
    p = cfg["parser"]
    out: list[str] = []
    for ln in lines:
        for pat in p["flat_split_before"]:           # e.g. a team name running into the table header
            ln = re.sub(rf"(?<=\S)\s+(?={pat})", "\n", ln, count=1)
        ln = re.sub(p["row_after_sentence_regex"], "\n", ln)   # R10 parsing fix: "...control.  2 Front Wing"
        ln = re.sub(p["flat_split_after"], lambda m: m.group(0) + "\n", ln)
        out += [x for x in (_clean(y) for y in _split_rows(ln, cfg).split("\n")) if x]
    return out


def _keep_header_tail(line: str, cfg: dict) -> str:
    """Text after 'Car Presentation ... Grand Prix' on a header line (a team name in flat pages), else ''."""
    return _clean(re.sub(cfg["parser"]["header_strip_regex"], "", line, count=1, flags=re.I))


def _page_body(text: str, cfg: dict, hdr: re.Pattern) -> list[str]:
    """Cell lines of one page. The page-header line becomes the HDR_MARK line (followed by its tail text, if any),
    so that the team name can be found before or after it (R10 parsing fix: pypdf may reverse the page order)."""
    foot = re.compile(cfg["parser"]["footer_regex"], re.I)
    start = re.compile(cfg["parser"]["table_start_regex"], re.I)
    for pat, sub in cfg["parser"]["text_fixes"]:
        text = re.sub(pat, sub, text)
    raw = _page_lines(text)
    first_hdr = next((i for i, ln in enumerate(raw) if hdr.search(ln)), None)
    if first_hdr and any(foot.search(x) for x in raw[:first_hdr]):
        raw = raw[first_hdr:]   # previous page's footer (team name, page number, (c) line) leaked before the header
    elif first_hdr is None and any(foot.search(x) for x in raw):
        last = max(i for i, x in enumerate(raw) if foot.search(x))
        # R10 parsing fix: a footer at the BOTTOM of a headerless page (rows before it) must not eat the rows
        if not any(start.search(x) and not x.isdigit() for x in raw[:last]):   # a lone page number is not a table
            raw = raw[last + 1:]    # footer block with no table before it: drop up to and including the footer
    lines: list[str] = []
    for ln in (_clean(foot.sub(" ", x)) for x in raw):
        if not ln:
            continue
        if hdr.search(ln):
            lines.append(HDR_MARK)
            tail = _keep_header_tail(ln, cfg)
            if tail:
                lines.append(tail)
        else:
            lines.append(ln)
    return expand_flat_lines(lines, cfg)


def _empty_fields() -> dict[str, Any]:
    return dict(component=None, category=None, category_secondary=None, reason_group=None, reason_subtype=None,
                subtype_known=None, optional=None, raw_text=None)


def _flag_fields(f: _Flag, cfg: dict) -> dict[str, Any]:
    """Fields of a flagged row. R10 parsing fix, one row per printed component:
    gap with a known subtype and no group -> 'group_missing' (component and subtype kept, group None, raw_text = the printed
    text without the row number); other gap -> 'reason_missing' ('unparsed' if a reason cell with an unknown group is
    printed, i.e. a spaced dash); orphan lines -> 'reason_missing' if all are cut rows, else 'unparsed'."""
    text = " | ".join(f.lines)
    if f.kind == "gap" and f.component:
        return dict(component=f.component, category=categorise(f.component, cfg),
                    category_secondary=categorise_secondary(f.component, cfg), reason_subtype=f.subtype,
                    subtype_known=True, parse_status="group_missing", raw_text=f.text)
    if f.kind == "gap":
        dash = re.compile(rf"\s[{cfg['parser']['dash_chars']}]\s*$")
        # a reason cell printed with an unknown group ('<n> <component> Novelty -' followed by its subtype line) is 'unparsed';
        # a lone cut-off line, or no dash at all, is 'reason_missing'
        status = "unparsed" if len(f.lines) > 1 and any(dash.search(x) for x in f.lines) else "reason_missing"
    else:
        status = "reason_missing" if f.lines and all(is_cut_row(x, cfg) for x in f.lines) else "unparsed"
    return dict(parse_status=status, raw_text=text)


def _locate_team(lines: list[str], is_body: Any, has_noup: Any) -> tuple[str | None, list[str], bool]:
    """(team name or None, body lines, page had a header line). R10 parsing fix: the team name sits next to the
    header line, before the table (normal order) or after it, with the table / no-updates line first (reversed
    pypdf order: 2025 Japan Mercedes and Haas, 2026 China Williams)."""
    if HDR_MARK not in lines:
        return None, lines, False
    h = lines.index(HDR_MARK)
    pre = [x for x in lines[:h] if x != HDR_MARK]
    post = [x for x in lines[h + 1:] if x != HDR_MARK]
    if (has_noup(pre) or any(is_body(x) for x in pre)) and not (has_noup(post) or any(is_body(x) for x in post)):
        return (_clean(" ".join(post)) or None), pre, True
    k = 0
    while k < len(post) and not is_body(post[k]):
        k += 1
    return (_clean(" ".join(post[:k])) or None), post[k:], True


def _completeness_rows(rows: list[dict], doc: ParsedDoc, cfg: dict, lin: dict[str, str], n_pages: int) -> list[dict]:
    """R10 parsing fix: every team of the season's entry list must appear exactly once per document. A missing team
    gets an explicit marker row (parse_status 'team_missing'); a team seen under two names is marked 'team_duplicate'."""
    for r in rows:
        if r["constructorId"] is None:
            r["parse_status"] = "unmapped_team" if r["parse_ok"] else r["parse_status"] + "+unmapped_team"
    entry = (cfg["parser"].get("expected_teams") or {}).get(doc.season)
    if not entry:
        return []
    names: dict[str, set[str]] = {}
    for r in rows:
        if r["constructorId"]:
            names.setdefault(r["constructorId"], set()).add(r["team_raw"])
    for r in rows:
        if r["constructorId"] and len(names[r["constructorId"]]) > 1:
            r["parse_status"] = "team_duplicate"
        elif r["constructorId"] is not None and r["constructorId"] not in entry:
            r["parse_status"] = "team_unexpected"
    out = []
    for cid in entry:
        if cid not in names:
            out.append(dict(season=doc.season, event=doc.event, event_key=doc.event_key, team_raw=f"<missing {cid}>",
                            team=cid, constructorId=cid, lineage=lin.get(cid, cid), page=max(n_pages, 2),
                            **{**_empty_fields(), "raw_text": "team absent from the document text"},
                            no_updates=False, parse_ok=False, parse_status="team_missing"))
    return out


def parse_pages(pages: list[str], cfg: dict, lineage: dict[str, str] | None = None) -> ParsedDoc:
    """Parse the text of all pages of one document (page 0 is the cover)."""
    p = cfg["parser"]
    is_cp, season, event = parse_cover(pages[0], cfg) if pages else (False, None, None)
    doc = ParsedDoc(season, event, _alnum(event) if event else None, is_cp, [])
    if not is_cp:
        return doc
    lin = lineage_map() if lineage is None else lineage
    hdr = re.compile(p["header_line_regex"], re.I)
    start = re.compile(p["table_start_regex"], re.I)
    noup = p["no_update_regex"]

    def has_noup(ls: list[str]) -> bool:
        return bool(re.search(noup, _letters(" ".join(ls))))

    def is_body(ln: str) -> bool:
        return bool(start.search(ln)) or has_noup([ln])

    team_raw: str | None = None
    team_rows: dict[str, list[dict]] = {}
    no_up: dict[str, bool] = {}
    leftover: dict[str, list[str]] = {}
    flags_of: dict[str, list[tuple[int, _Flag]]] = {}
    last_no: dict[str, int] = {}
    page_of: dict[str, int] = {}

    def register(name: str, pn: int) -> None:
        if name not in team_rows:
            team_rows[name], no_up[name], leftover[name], page_of[name] = [], False, [], pn
            flags_of[name], last_no[name] = [], 0

    for pn, text in enumerate(pages[1:], start=2):
        lines = _page_body(text, cfg, hdr)
        if not [x for x in lines if x != HDR_MARK]:
            continue
        name, rest, had_hdr = _locate_team(lines, is_body, has_noup)
        cont = False
        if had_hdr:
            if name:
                team_raw = name
                register(name, pn)
        elif not start.search(lines[0]):
            k = 0
            while k < len(lines) and not is_body(lines[k]):
                k += 1
            cand = _clean(" ".join(lines[:k]))
            if team_raw is not None and cand not in team_rows and (k >= len(lines) or map_team(cand, cfg, lin)[1] is None):
                cont, rest = True, lines     # page starts with overflow prose of the previous team's table
            else:
                team_raw, rest = cand, lines[k:]
                register(cand, pn)
        if team_raw is None:
            continue
        if cont or (rest and start.search(rest[0])):
            rows, flags, last_no[team_raw] = parse_team_units(_strip_table_header(rest, cfg), cfg, drop_lead=cont,
                                                              prev_last=last_no[team_raw])
            for r in rows:
                r["page"] = pn
            team_rows[team_raw].extend(rows)
            flags_of[team_raw].extend((pn, f) for f in flags)
        elif has_noup(rest):
            no_up[team_raw] = True
        else:
            leftover[team_raw].extend(rest)
    for team_raw, rows in team_rows.items():
        norm, cid, lg = map_team(team_raw, cfg, lin)
        base = dict(season=season, event=event, event_key=doc.event_key, team_raw=team_raw, team=norm,
                    constructorId=cid, lineage=lg, page=page_of[team_raw])
        for r in rows:
            doc.rows.append({**base, **_empty_fields(), **r, "category": categorise(r["component"], cfg),
                             "category_secondary": categorise_secondary(r["component"], cfg),
                             "no_updates": False, "parse_ok": True, "parse_status": "ok"})
        flags = [*flags_of[team_raw], *([(page_of[team_raw], _Flag(0, leftover[team_raw], "orphan"))]
                                          if leftover[team_raw] else [])]
        if not rows and no_up[team_raw] and not flags:
            doc.rows.append({**base, **_empty_fields(), "no_updates": True, "parse_ok": True, "parse_status": "no_updates"})
        elif not rows and not flags:
            # R10 parsing fix: a page with only the header and a team name has no text to parse (image-only page, no OCR):
            # 'page_text_empty'. A long "name" is prose with no table: 'unparsed', the text is kept.
            empty = len(team_raw.split()) <= p["team_name_max_tokens"]
            doc.rows.append({**base, **_empty_fields(), "no_updates": False, "parse_ok": False,
                             "parse_status": "page_text_empty" if empty else "unparsed", "raw_text": "" if empty else team_raw})
        for pn, f in flags:
            doc.rows.append({**base, **_empty_fields(), **_flag_fields(f, cfg), "page": pn, "no_updates": False, "parse_ok": False})
    doc.rows.extend(_completeness_rows(doc.rows, doc, cfg, lin, len(pages)))
    return doc


# --- files ----------------------------------------------------------------------------------------------------

def read_pdf_pages(path: Path | str) -> list[str]:
    from pypdf import PdfReader  # lazy: only real files need it
    return [(pg.extract_text() or "") for pg in PdfReader(str(path)).pages]


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_pdf(path: Path | str, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_cfg()
    doc = parse_pages(read_pdf_pages(path), cfg)
    df = pd.DataFrame(doc.rows, columns=COLUMNS)
    df["doc_file"] = Path(path).name
    df["doc_sha256"] = sha256_file(path)
    return df


MANIFEST_PATH = METHODS / "M10_upgrades" / "documents.csv"


def load_manifest(path: Path | str = MANIFEST_PATH) -> pd.DataFrame:
    """Manifest rows with a document (found == True). The manifest is the only list of source files."""
    m = pd.read_csv(path, dtype={"found": str, "sha256": str, "file": str, "published_cet": str})
    return m[m["found"].str.lower() == "true"].reset_index(drop=True)


def parse_manifest_row(row: pd.Series, src: Path, cfg: dict) -> pd.DataFrame:
    """Parse one manifest file; the SHA-256 must equal the manifest value (mismatch raises ValueError)."""
    f = src / row["file"]
    if not f.exists():
        raise FileNotFoundError(f"manifest file missing: {f}")
    got = sha256_file(f)
    if got != row["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {f.name}: manifest {row['sha256']}, file {got}")
    df = parse_pdf(f, cfg)
    df["manifest_season"] = int(row["season"])
    df["manifest_event"] = row["event"]
    df["event_id"] = int(row["event_id"])
    df["published_cet"] = row["published_cet"]
    return df


def parse_dir(path: Path | str | None = None, out: Path | str | None = None, cfg: dict | None = None,
              manifest: Path | str | None = None) -> pd.DataFrame:
    """Parse exactly the files listed in the manifest (found == True), write the table, return it.
    Files in the directory that are not in the manifest are ignored."""
    cfg = cfg or load_cfg()
    src = Path(path) if path else ROOT / cfg["parser"]["input_dir"]
    man = load_manifest(manifest or MANIFEST_PATH)
    frames = [parse_manifest_row(r, src, cfg) for _, r in man.iterrows()]
    frames = [f for f in frames if len(f)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    target = Path(out) if out else ROOT / cfg["parser"]["output"]
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per document and team: component rows, no-update flag, unparsed rows."""
    g = df.groupby(["doc_file", "team_raw"], sort=False)
    return g.agg(components=("component", lambda s: int(s.notna().sum())), no_updates=("no_updates", "max"),
                 unparsed=("parse_ok", lambda s: int((~s.astype(bool)).sum()))).reset_index()


if __name__ == "__main__":
    print(summarise(parse_dir()).to_string())
