"""PLAN3 R9 third test round of the M10 parser (second revision): synthetic tests for the new statuses and config knobs,
and independent real-PDF checks that do not use the parser's own row logic.

Synthetic cases use the helpers of test_plan3_m10.py. Real-PDF cases skip when data/raw/fia is absent.
"""
from __future__ import annotations

import copy
import re

import pandas as pd
import pytest

from pitlane.data import fia_docs as F
from test_plan3_m10 import (CFG, FER, HAVE_ALL, LIN, LOCAL, MCL, R_FLOOR, R_WING, comp, cover, parse, table, team_page)

NOUP = "No updates submitted for this event. \n"


def cfg_with(**kw) -> dict:
    c = copy.deepcopy(CFG)
    c["parser"].update(kw)
    return c


def parse_cfg(cfg: dict, *pages: str):
    return F.parse_pages(list(pages), cfg, LIN)


def one(body: str, cfg: dict | None = None) -> pd.DataFrame:
    """Rows of one Ferrari page whose table is R_WING row 1 followed by `body`."""
    return comp(parse_cfg(cfg or CFG, cover(), team_page(FER, table(R_WING.format(n=1) + body))))


def cells(df: pd.DataFrame) -> list[tuple]:
    return list(zip(df["component"].where(df["component"].notna(), None), df["reason_group"].where(df["reason_group"].notna(), None),
                    df["reason_subtype"].where(df["reason_subtype"].notna(), None), df["parse_status"]))


# --- reason group 'Other' and 'Cooling' (R9 rule, second revision) ---------------------------------------------------

def test_other_group_is_accepted_as_printed():
    df = one("2 Front Suspension Other - Local flow alignment Made-up text. \nText. \n")
    assert cells(df)[1] == ("Front Suspension", "Other", "Local flow alignment", "ok") and df["subtype_known"].iloc[1]


def test_cooling_group_row_is_read_as_group_cooling_with_empty_subtype():
    df = one("2 Cooling Louvres Cooling Made-up wider panel \nMade up text. \n")
    assert cells(df)[1] == ("Cooling Louvres", "Cooling", "", "ok") and df["category"].iloc[1] == "sidepod_engine_cover"


def test_cooling_range_is_a_subtype_not_the_group():
    """'Coke/Engine Cover  Cooling Range ...' with no group printed: 'Cooling' must not be read before 'Range'."""
    df = one("2 Coke/Engine Cover Cooling Range Made-up smaller exit \nMade up text. \n")
    r = df.iloc[1]
    assert r["parse_status"] == "group_missing" and r["reason_subtype"] == "Cooling Range" and pd.isna(r["reason_group"])
    assert r["component"] == "Coke/Engine Cover"


def test_cooling_alone_at_the_start_of_a_line_is_not_a_reason_group():
    """The group 'Cooling' needs a component text before it on its line; the row is flagged, not read as group Cooling."""
    df = one("2 Louvres \nCooling Wider panels \nText. \n")
    assert df["parse_status"].tolist() == ["ok", "reason_missing"] and "Cooling" not in set(df["reason_group"].dropna())


def test_driver_cooling_is_not_the_group_cooling_and_the_row_is_flagged():
    body = "2 Chassis Scoop Driver Cooling Made-up extra duct for the test \nText. \n"
    df = one(body)
    assert df["parse_status"].tolist() == ["ok", "reason_missing"]
    assert df["raw_text"].iloc[1].startswith("2 Chassis Scoop Driver Cooling") and pd.isna(df["component"].iloc[1])


def test_reason_group_blocked_after_is_a_config_knob():
    body = "2 Chassis Scoop Driver Cooling Made-up extra duct for the test \nText. \n"
    df = one(body, cfg_with(reason_group_blocked_after={}))      # no blocked word: the row is read as group Cooling
    assert df["parse_status"].tolist() == ["ok", "ok"] and df["reason_group"].iloc[1] == "Cooling"
    other = one(body.replace("Driver", "Pilot"), cfg_with(reason_group_blocked_after={"Cooling": ["Pilot"]}))
    assert other["parse_status"].tolist() == ["ok", "reason_missing"]


# --- dash-less reason cells --------------------------------------------------------------------------------------------

def test_dashless_circuit_specific_keeps_group_and_leaves_subtype_empty_unknown():
    df = one("2 Front \nSuspension Circuit specific Made-up change to a test item \nMonaco text. \n")
    r = df.iloc[1]
    assert (r["component"], r["reason_group"], r["reason_subtype"], r["parse_status"]) == ("Front Suspension", "Circuit specific", "", "ok")
    assert not r["subtype_known"]


def test_dashless_performance_with_known_subtype_wrapped_over_two_lines():
    df = one("2 Sidepod Performance Local \nLoad Made-up text line. \nMade up text. \n")
    assert cells(df)[1] == ("Sidepod", "Performance", "Local Load", "ok") and df["subtype_known"].iloc[1]


def test_dashless_performance_in_prose_does_not_make_a_row():
    df = one("3 elements are used here \nBetter Performance Local Load generation. \n")
    assert df["parse_status"].tolist() == ["ok"]
    df = one("3 elements are used here \nBetter Circuit specific wing tuning. \n")
    assert df["parse_status"].tolist() == ["ok"]


# --- group_missing -------------------------------------------------------------------------------------------------------

def test_group_missing_keeps_component_and_subtype_and_never_infers_the_group():
    df = comp(parse(cover(), team_page(FER, table("1 Floor body Local load Made-up smoothing of the test surface \n"
                                                  "2 Mirrors Flow Conditioning Revised mirror housing \nText. \n"))))
    assert df["parse_status"].tolist() == ["group_missing"] * 2 and (~df["parse_ok"]).all()
    assert df["component"].tolist() == ["Floor body", "Mirrors"] and df["reason_subtype"].tolist() == ["Local Load", "Flow Conditioning"]
    assert df["reason_group"].isna().all() and df["subtype_known"].all()
    assert df["category"].tolist() == ["floor", "other"] and df["raw_text"].str.len().gt(0).all()
    assert df["raw_text"].iloc[0].startswith("Floor body Local load") and not df["raw_text"].iloc[0].startswith("1 ")


@pytest.mark.parametrize("body", [
    "2 Floor body Novel thing Made-up smoothing of the test surface \nText. \n",                       # subtype not in the config
    "2 Floor body, left Local Load Subtle revisions \nText. \n",                                 # punctuation in the component text
    "2 A B C D E F G Local Load Subtle revisions \nText. \n",                                    # more than flag_component_max_words
])
def test_group_missing_is_not_used_when_component_or_subtype_is_doubtful(body):
    df = one(body)
    assert df["parse_status"].tolist() == ["ok", "reason_missing"] and pd.isna(df["component"].iloc[1])
    assert df["raw_text"].iloc[1].startswith("2 ")


def test_flagged_rows_never_carry_a_reason_group():
    df = one("2 Floor body Local load Subtle \nText. \n3 Beam Wing \n4 Front Wing Altered made-up thing along the full length now \nmore. \n")
    bad = df[~df["parse_ok"]]
    assert len(bad) == 3 and bad["reason_group"].isna().all()


# --- page_text_empty and unparsed --------------------------------------------------------------------------------------

def test_page_text_empty_for_a_page_with_only_header_and_team_name():
    df = comp(parse(cover(), team_page(FER, ""), team_page(MCL, NOUP)))
    r = df[df["team_raw"].str.contains("Ferrari")].iloc[0]
    assert r["parse_status"] == "page_text_empty" and not r["parse_ok"] and r["raw_text"] == "" and pd.isna(r["component"])
    assert not r["no_updates"]
    assert df[df["team_raw"].str.contains("McLaren")]["parse_status"].tolist() == ["no_updates"]


def test_long_text_without_a_table_is_unparsed_with_text_kept_and_knob_is_config():
    prose = "FERRARI says this is a very long line of prose that has no table at all in it"
    df = comp(parse(cover(), team_page(prose, "")))
    assert df["parse_status"].str.startswith("unparsed").all() and df["raw_text"].iloc[0] == prose
    short = comp(parse_cfg(cfg_with(team_name_max_tokens=30), cover(), team_page(prose, "")))
    assert short["parse_status"].str.startswith("page_text_empty").all()


# --- merged reason cells -----------------------------------------------------------------------------------------------

def test_shared_reason_cell_gives_one_flag_per_component_and_no_reason_is_assigned():
    body = ("1 Front Wing \nPerformance - \nLocal Load \nBig text. \n2 Front Wing Endplate \n3 Nose \n"
            "4 Floor Body \nPerformance - \nLocal Load \nText. \n")
    df = comp(parse(cover(), team_page(FER, table(body))))
    assert df["parse_status"].tolist().count("reason_missing") == 2 and df["parse_status"].tolist().count("ok") == 2
    flags = df[df["parse_status"] == "reason_missing"]
    assert sorted(flags["raw_text"]) == ["2 Front Wing Endplate", "3 Nose"]
    assert flags["reason_group"].isna().all() and flags["component"].isna().all()


# --- chain_max_skip and dash_flag_max_tokens ---------------------------------------------------------------------------

LONGROW = " Beam Wing Altered made-up thing along the full length \nMore text. \n"   # no reason cell, not a cut row


@pytest.mark.parametrize("skip,printed,flagged", [(0, 2, True), (0, 3, False), (1, 2, True), (1, 3, False),
                                                  (3, 2, True), (3, 3, True), (3, 4, True), (3, 5, False), (3, 9, False)])
def test_chain_max_skip_is_the_largest_jump_of_a_line_start_number(skip, printed, flagged):
    """chain_max_skip counts the jump (printed - last number), not the numbers skipped: with 3 the number 4 after 1 is still
    in the chain (2 and 3 skipped) and 5 is not. A row beyond the jump is silently lost: documented limit of the chain."""
    df = comp(parse_cfg(cfg_with(chain_max_skip=skip), cover(),
                        team_page(FER, table(R_WING.format(n=1) + f"{printed}" + LONGROW))))
    assert (df["parse_status"].tolist() == ["ok", "reason_missing"]) is flagged


@pytest.mark.parametrize("tokens,flagged", [(5, False), (6, True), (9, True)])
def test_dash_flag_max_tokens_counts_the_dash_and_is_a_config_knob(tokens, flagged):
    line = "A max downforce rear wing -"                       # 6 tokens with the dash
    df = comp(parse_cfg(cfg_with(dash_flag_max_tokens=tokens), cover(), team_page(FER, table(R_WING.format(n=1) + line + "\nthe camber is raised. \n"))))
    assert (df["parse_status"].tolist() == ["ok", "unparsed"]) is flagged


def test_prose_line_ending_in_a_dash_is_the_description_of_the_row_above_by_default():
    line = "A made-up 'big box' style rear wing setup -"   # 9 tokens, more than the limit of 6 (the real 2024_50070_8 line has 8)
    assert one(line + "\nthe camber is raised. \n")["parse_status"].tolist() == ["ok"]


def test_a_numbered_line_ending_in_a_dash_is_flagged_even_if_long():
    df = one("2 Front Wing Altered made-up thing along the full length of the flap -\nmore. \n")
    assert len(df) == 2 and not df["parse_ok"].iloc[1]


# --- team_missing marker -------------------------------------------------------------------------------------------------

def test_team_missing_marker_has_an_integer_page_and_no_group_or_component():
    df = comp(parse(cover(2024), team_page(FER, table(R_WING.format(n=1)))))
    miss = df[df["parse_status"] == "team_missing"]
    assert len(miss) == 9 and miss["page"].map(lambda p: isinstance(p, int) and p >= 2).all()
    assert miss[["component", "reason_group", "reason_subtype", "category"]].isna().all().all()


# --- false positives of the second revision (fixed in round 4) -------------------------

# fixed in round 4: a row number must be followed by a capital letter ('3 elements are used here' is prose)
@pytest.mark.parametrize("prose,group", [("Improved Cooling flow through the duct. \n", "Cooling"),
                                         ("The wing and other - similar parts \nmore. \n", "Other")])
def test_prose_with_a_new_reason_group_word_after_a_digit_line_is_not_a_row(prose, group):
    df = one("3 elements are used here \n" + prose)
    assert df["parse_status"].tolist() == ["ok"], cells(df)


# --- real PDFs: independent checks --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real(tmp_path_factory):
    if not HAVE_ALL:
        pytest.skip("local FIA PDFs not present")
    return F.parse_dir(out=tmp_path_factory.mktemp("m10r2") / "t.parquet")


def dash(t: str) -> str:
    """En dash, em dash and U+FFFD (how pypdf decodes the FIA en dash) all become a hyphen."""
    return re.sub("[–—�]", "-", t)


@pytest.fixture(scope="module")
def page_text():
    cache: dict[str, list[str]] = {}

    def get(f: str, page: int, raw: bool = False) -> str:
        if f not in cache:
            cache[f] = F.read_pdf_pages(LOCAL / f)
        t = cache[f][page - 1]
        return re.sub(r"\s+", " ", t if raw else dash(t))
    return get


def _rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["component"].notna() | ~df["parse_ok"]]


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_every_team_has_at_least_as_many_rows_as_its_exact_printed_number_chain(real):
    """Independent lower bound: numbers 1, 2, 3, ... (each exactly +1, a capitalised word after) printed in the team's pages.
    A team with fewer parsed + flagged rows than that chain has lost a printed row. Teams with more rows are pinned: the chain
    breaks at skipped, repeated or missing printed numbers (checked by hand, see the report)."""
    more = set()
    for (f, cid), g in real.groupby(["doc_file", "constructorId"]):
        r = _rows(g)
        if r.empty:
            continue
        pages = F.read_pdf_pages(LOCAL / f)
        txt = " ".join(re.sub(r"\s+", " ", pages[i - 1]) for i in range(int(g["page"].min()), int(g["page"].max()) + 1))
        last = 0
        for m in re.finditer(r"(?:(?<=\s)|^)(\d{1,2})(?=\s+[A-Z]|\s*$)", txt):
            if int(m.group(1)) == last + 1:
                last += 1
        assert len(r) >= last, (f, cid, len(r), last)
        if len(r) != last:
            more.add((f, cid))
    assert more == {("2024_50064_9.pdf", "rb"), ("2024_50067_6.pdf", "haas"), ("2024_50084_7.pdf", "williams"),
                    ("2025_55037_7.pdf", "red_bull"), ("2025_56441_9.pdf", "red_bull"), ("2026_60295_9.pdf", "aston_martin"),
                    ("2026_60295_9.pdf", "williams"), ("2026_60303_14.pdf", "red_bull"), ("2026_60304_14.pdf", "red_bull")}, more


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_every_ok_row_is_printed_in_its_page_text_as_component_group_subtype(real, page_text):
    """Precision of the 1163 'ok' rows, read from the pypdf text with a separate regex: component, group and subtype must
    appear in that order on the row's page."""
    ok = real[(real["parse_status"] == "ok") & real["component"].notna()]
    bad = []
    for _, r in ok.iterrows():
        comp_ = re.escape(dash(r["component"])).replace(r"\ ", r"\s+")
        grp = re.escape(r["reason_group"]).replace(r"\ ", r"\s+")
        sub = (r"\s*[-]?\s*" + re.escape(dash(r["reason_subtype"])).replace(r"\ ", r"\s+")) if r["reason_subtype"] else ""
        if not re.search(comp_ + r"\s+" + grp + sub, page_text(r["doc_file"], int(r["page"])), re.I):
            bad.append((r["doc_file"], r["page"], r["component"], r["reason_group"], r["reason_subtype"]))
    assert len(ok) == 1163 and bad == [], bad[:10]


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_rows_found_by_an_independent_regex_but_not_parsed_are_the_three_known_flagged_ones(real, page_text):
    """Recall: '<n> <component> <group>' read with a regex that needs a printed row number and a group word. Everything it
    finds that is not an ok row must be a flagged row (Driver Cooling, and two Mercedes 'Cooling Range' rows with no group)."""
    grp = r"(Performance|Circuit\s+specific|Reliability|Weight|Cost|Other|Cooling)"
    rx = re.compile(r"(?:(?<=\s)|^)(\d{1,2})\s+([A-Z][A-Za-z/&\-\s]{2,55}?)\s+" + grp + r"(?=\s*[-–—�]|\s+[A-Z])", re.I)
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())      # noqa: E731
    extra = []
    for (f, pn), g in real.groupby(["doc_file", "page"]):
        pr = g[(g["parse_status"] == "ok") & g["component"].notna()]
        from collections import Counter
        got = Counter((norm(c), norm(x)) for c, x in zip(pr["component"], pr["reason_group"]))
        found = Counter((norm(m.group(2)), norm(m.group(3))) for m in rx.finditer(page_text(f, int(pn), raw=True)))
        for k in (found - got):
            extra.append((f, int(pn), k))
    assert sorted(extra) == [("2024_50068_10.pdf", 10, ("chassisscoopdriver", "cooling")),
                             ("2026_60304_14.pdf", 4, ("cokeenginecover", "cooling")),
                             ("2026_60307_9.pdf", 4, ("cokeenginecover", "cooling"))], extra
    flagged = real[real["parse_status"].isin(["group_missing", "reason_missing"])]
    assert {"2024_50068_10.pdf", "2026_60304_14.pdf", "2026_60307_9.pdf"} <= set(flagged["doc_file"])


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_ok_row_sample_seed_0_is_row_for_row_in_the_text(real, page_text):
    """The seed-0 sample of 20 ok rows that was read by hand in round 3: all 20 were right."""
    ok = real[(real["parse_status"] == "ok") & real["component"].notna()]
    s = ok.sample(20, random_state=0)
    assert len(s) == 20
    for _, r in s.iterrows():
        t = page_text(r["doc_file"], int(r["page"]))
        assert re.search(re.escape(r["component"]).replace(r"\ ", r"\s+") + r"\s+" + re.escape(r["reason_group"]).replace(r"\ ", r"\s+"), t, re.I)


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_optional_flag_is_a_low_recall_end_of_line_marker_pinned_as_is(real):
    """The 'optional' column is True only when a line of the row ends in the word Optional. Words 'optional' inside prose
    (description text) are not caught: 6 rows are True while 'optional'
    occurs on 17 pages. A model must not read optional == False as 'not optional'. Pin: the six known rows."""
    o = real[real["optional"] == True]       # noqa: E712
    assert sorted(zip(o["doc_file"], o["constructorId"], o["component"])) == [
        ("2024_50064_9.pdf", "rb", "Cooling Louvres"), ("2024_50064_9.pdf", "rb", "Front Wing"), ("2024_50064_9.pdf", "rb", "Rear Wing"),
        ("2025_55037_7.pdf", "williams", "Coke/Engine Cover"), ("2025_56435_8.pdf", "williams", "Beam Wing"),
        ("2026_60307_9.pdf", "ferrari", "Rear Wing")]


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_only_the_rear_wheel_bodywork_component_is_rear_corner_by_the_new_rule(real):
    rc = real[real["category"] == "rear_corner_suspension"]
    rw = rc[rc["component"].str.contains("Wheel", case=False)]
    assert len(rw) >= 1 and (rw["component"] == "Rear Wheel Bodywork").sum() == 1
    assert F.categorise("Rear Corner Winglet", CFG) == "rear_corner_suspension" and F.categorise("Sidepod Inlet", CFG) == "sidepod_engine_cover"
    assert F.categorise("Bodywork", CFG) == "sidepod_engine_cover"        # the rear-corner rule does not steal plain 'bodywork'


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_skipped_printed_numbers_are_not_in_the_pypdf_text(real):
    """Known limit, checked by hand: 2025_55037_7 Red Bull prints rows 1,2,3,6,7,8,9,10,13,15,16,17 (nothing in the text for
    4, 5, 11, 12, 14) and 2026_60295_9 Aston prints 1-5, 7, 8 (nothing for 6). Whether these are unused numbers or rows
    that pypdf cannot read cannot be decided from the text; the parser neither flags nor counts them."""
    rb = real[(real["doc_file"] == "2025_55037_7.pdf") & (real["constructorId"] == "red_bull")]
    assert len(rb) == 12 and rb["parse_ok"].all()
    am = real[(real["doc_file"] == "2026_60295_9.pdf") & (real["constructorId"] == "aston_martin")]
    assert len(am) == 7 and am["parse_ok"].all()
    t = " ".join(re.sub(r"\s+", " ", p) for p in F.read_pdf_pages(LOCAL / "2025_55037_7.pdf")[5:7])
    nums = [int(m.group(1)) for m in re.finditer(r"(?:(?<=\s)|^)(\d{1,2})\s+[A-Z]", t)]
    assert [n for n in nums if n <= 17] == [1, 2, 3, 6, 7, 8, 9, 10, 13, 15, 16, 17]
