"""PLAN3 R10 re-test of the M10 parser: parse_status values, category_secondary, and silent row loss.

Synthetic cases use the helpers of test_plan3_m10.py. Real-PDF cases skip when data/raw/fia is absent.
The defects found by hand-checking the regenerated table against the PDF text (R10 re-test) were fixed in the
second parser revision; the strict xfails that recorded them are now plain tests.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from pitlane.data import fia_docs as F
from test_plan3_m10 import (CFG, FER, HAVE_ALL, LOCAL, MCL, MER, REAL, R_FLOOR, R_WING, comp, cover, parse, table,
                            team_page)

EXPECT_STATUS = {"ok", "no_updates", "unparsed", "reason_missing", "group_missing", "page_text_empty"}   # R10 parsing fix, 2nd revision
FLAGGED = ["unparsed", "reason_missing", "group_missing", "page_text_empty"]
BAD_STATUS = {"team_missing", "team_duplicate", "unmapped_team", "team_unexpected"}
ENTRY = CFG["parser"]["expected_teams"]


# --- synthetic: every parse_status value ----------------------------------------------------------------------------

def test_status_ok_and_no_updates():
    df = comp(parse(cover(), team_page(FER, table(R_WING.format(n=1))), team_page(MCL, "No updates submitted for this event. \n")))
    assert df["parse_status"].tolist() == ["ok", "no_updates"]


NAMES = {"red_bull": "Oracle Red Bull Racing", "mercedes": MER, "ferrari": FER, "mclaren": MCL, "aston_martin": "Aston Martin Aramco F1 Team",
         "alpine": "BWT Alpine F1 Team", "williams": "Williams Racing", "rb": "Visa Cash App Racing Bulls", "sauber": "Kick Sauber F1 Team",
         "haas": "MoneyGram Haas F1 Team", "audi": "Audi Revolut F1 Team", "cadillac": "Cadillac"}
NOUP = "No updates submitted for this event. \n"


def full(season: int, *pages: str, skip: tuple[str, ...] = ()):
    """Doc whose pages are given, plus a no-updates page for every other team of the season's entry list (so none is missing)."""
    named = " ".join(pages)
    extra = [team_page(NAMES[c], NOUP) for c in ENTRY[season] if c not in skip and NAMES[c].split()[0].lower() not in named.lower()]
    return parse(cover(season), *pages, *extra)


def test_team_missing_marker_rows_for_every_absent_team():
    doc = parse(cover(2024), team_page(FER, table(R_WING.format(n=1))), team_page(MCL, NOUP))
    df = comp(doc)
    miss = df[df["parse_status"] == "team_missing"]
    assert set(miss["constructorId"]) == set(ENTRY[2024]) - {"ferrari", "mclaren"} and len(miss) == 8
    assert (~miss["parse_ok"]).all() and miss["component"].isna().all() and not miss["no_updates"].any()
    assert miss["team_raw"].str.startswith("<missing ").all() and (miss["season"] == 2024).all()
    assert (miss["raw_text"] == "team absent from the document text").all()


def test_complete_document_has_no_team_missing_row_and_every_status_ok():
    df = comp(full(2024, team_page(FER, table(R_WING.format(n=1)))))
    assert len(df) == 10 and set(df["constructorId"]) == set(ENTRY[2024])
    assert df["parse_status"].value_counts().to_dict() == {"no_updates": 9, "ok": 1}


def test_no_team_missing_rows_for_a_season_without_an_entry_list():
    assert "team_missing" not in set(comp(parse(cover(2030), team_page(FER, table(R_WING.format(n=1)))))["parse_status"])


def test_team_duplicate_when_one_constructor_appears_under_two_names():
    df = comp(full(2024, team_page(FER, table(R_WING.format(n=1))), team_page("*SCUDERIA FERRARI HP*", table(R_FLOOR.format(n=1)))))
    fer = df[df["constructorId"] == "ferrari"]
    assert len(fer) == 2 and set(fer["parse_status"]) == {"team_duplicate"}
    assert (df["parse_status"] == "team_duplicate").sum() == 2


def test_same_raw_name_twice_is_not_a_duplicate():
    df = comp(full(2024, team_page(FER, table(R_WING.format(n=1))), team_page(FER, table(R_FLOOR.format(n=1)))))
    fer = df.query("constructorId == 'ferrari'")
    assert set(fer["parse_status"]) == {"ok"} and len(fer) == 2


def test_unmapped_team_status_for_rows_noupdate_and_unparsed():
    new = "Brand New Team"
    ok = comp(parse(cover(2030), team_page(new, table(R_WING.format(n=1)))))
    assert ok["constructorId"].isna().all() and ok["parse_status"].tolist() == ["unmapped_team"]
    nu = comp(parse(cover(2030), team_page(new, "No updates submitted for this event. \n")))
    assert nu["parse_status"].tolist() == ["unmapped_team"] and nu["no_updates"].tolist() == [True]
    bad = comp(parse(cover(2030), team_page(new, table("1 Rear Wing Novelty - \nDrag Range \nMade up text. \n"))))
    assert bad["parse_status"].tolist() == ["unparsed+unmapped_team"] and not bad["parse_ok"].any()


def test_unmapped_team_rows_keep_parse_ok_true_so_callers_must_test_the_status():
    """Documented behaviour, not a defect fix: an unmapped team's good rows still have parse_ok True."""
    df = comp(parse(cover(2030), team_page("Brand New Team", table(R_WING.format(n=1)))))
    assert df["parse_ok"].all() and df["parse_status"].iloc[0] == "unmapped_team"


def test_team_unexpected_for_a_constructor_outside_the_entry_list():
    df = comp(full(2024, team_page("Audi Revolut F1 Team", table(R_WING.format(n=1)))))
    assert df.query("constructorId == 'audi'")["parse_status"].tolist() == ["team_unexpected"]


def test_reason_missing_only_when_every_leftover_line_is_a_cut_row():
    cut = comp(parse(cover(), team_page(FER, table(R_WING.format(n=1)).rstrip() + "\n3 Beam Wing \n")))
    assert cut["parse_status"].tolist() == ["ok", "reason_missing"] and cut["raw_text"].iloc[1] == "3 Beam Wing"
    assert cut["component"].iloc[1] is None or pd.isna(cut["component"].iloc[1])
    mixed = comp(parse(cover(), team_page(FER, table(R_WING.format(n=1)).rstrip() + "\n3 Beam Wing \nsome made up prose line -\n")))
    # R10 parsing fix, 2nd revision: one flagged row per printed component; the cut row and the dash-ended prose line are separate rows
    assert mixed["parse_status"].tolist() == ["ok", "reason_missing", "unparsed"] and mixed["raw_text"].iloc[1] == "3 Beam Wing"
    # a numbered unknown-group row ('4 Floor Novelty -') also matches the cut-row regex and is labelled reason_missing, not unparsed (minor)
    nov = comp(parse(cover(), team_page(FER, table(R_WING.format(n=1)).rstrip() + "\n4 Floor Novelty - \n")))
    assert nov["parse_status"].tolist() == ["ok", "reason_missing"]
    only_prose = comp(parse(cover(), team_page(FER, "some made up prose with no table at all \n")))
    assert only_prose["parse_status"].tolist() == ["unparsed"]


def test_cut_row_regex_does_not_fire_on_ordinary_lines():
    for s in ["Revised flap, 3 elements", "3 variations will be made available", "2 Weight reduction made possible by the new part.",
              "1 front wing", "10 Cooling Louvres Cooling Reduced span for the medium levels"]:
        assert not F.is_cut_row(s, CFG), s
    for s in ["3 Beam Wing", "2 Front Wing", "10 Cooling", "7 Floor Edge and Diffuser"]:
        assert F.is_cut_row(s, CFG), s


# --- category_secondary ---------------------------------------------------------------------------------------------

def test_category_secondary_unit_cases():
    assert F.categorise_all("Rear Wing / Beam Wing", CFG) == ["beam_wing", "rear_wing"]
    assert F.categorise_secondary("Rear Wing / Beam Wing", CFG) == "rear_wing"
    assert F.categorise_secondary("Front Wing", CFG) == "" and F.categorise_secondary("", CFG) == ""
    assert F.categorise_secondary("Mirrors", CFG) == "" and F.categorise("Mirrors", CFG) == "other"
    assert F.categorise_secondary("Rear Wing / Beam Wing", {"component_categories": [["a", ["rear"]]]}) == ""


def test_category_secondary_is_filled_in_parsed_rows_only_for_combined_components():
    row = "1 Rear Wing / \nBeam Wing \nPerformance - \nLocal Load \nText. \n"
    df = comp(parse(cover(), team_page(FER, table(row, R_FLOOR.format(n=2)))))
    assert df["category"].tolist() == ["beam_wing", "floor"] and df["category_secondary"].tolist() == ["rear_wing", ""]


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_category_secondary_real_table_equals_second_keyword_hit_and_differs_from_primary(real_table):
    c = real_table[real_table["component"].notna()]
    want = c["component"].map(lambda s: F.categorise_secondary(s, CFG))
    assert (c["category_secondary"].fillna("") == want).all()
    assert (c["category"] == c["component"].map(lambda s: F.categorise(s, CFG))).all()
    filled = c[c["category_secondary"].fillna("") != ""]
    assert (filled["category"] != filled["category_secondary"]).all()
    assert real_table.loc[real_table["component"].isna(), "category_secondary"].isna().all()
    assert len(filled) == 2     # 'Rear Wing / Beam Wing' (2025_55037_7) and 'Rear Wheel Bodywork' (2026_60312_9, now rear_corner + sidepod)


def test_rear_wheel_bodywork_is_a_rear_corner_component():
    assert F.categorise("Rear Wheel Bodywork", CFG) == "rear_corner_suspension"


# --- real parse: statuses and team completeness ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_table(tmp_path_factory):
    if not HAVE_ALL:
        pytest.skip("local FIA PDFs not present")
    return F.parse_dir(out=tmp_path_factory.mktemp("m10s") / "t.parquet")


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_status_counts_and_parse_ok_agree(real_table):
    d = real_table
    assert set(d["parse_status"]) == EXPECT_STATUS and not d["parse_status"].isna().any()
    assert d["parse_status"].value_counts().to_dict() == {"ok": 1163, "no_updates": 274, "reason_missing": 29, "group_missing": 14, "page_text_empty": 2}
    assert (d["parse_ok"] == ~d["parse_status"].isin(FLAGGED)).all()
    assert d.loc[d["parse_status"] == "ok", "component"].notna().all() and d.loc[d["parse_status"] == "no_updates", "no_updates"].all()
    gm = d[d["parse_status"] == "group_missing"]     # component and subtype are kept, the group is never inferred
    assert len(gm) and gm["component"].notna().all() and gm["reason_group"].isna().all() and gm["raw_text"].str.len().gt(0).all()


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_no_row_is_team_missing_duplicate_unmapped_or_unexpected(real_table):
    d = real_table
    assert not d["parse_status"].str.contains("team_|unmapped").any()
    assert d["constructorId"].notna().all() and d["lineage"].notna().all()
    assert not d["team_raw"].str.startswith("<missing").any()


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
@pytest.mark.parametrize("f", sorted(REAL["file"]))
def test_real_document_has_exactly_the_entry_list_teams_once(real_table, f):
    g = real_table[real_table["doc_file"] == f]
    season = int(g["season"].iloc[0])
    cons = g.groupby("constructorId")["team_raw"].nunique()
    assert set(cons.index) == set(ENTRY[season]) and (cons == 1).all(), f
    # each team is either no-updates (one row) or component / flagged rows, never both
    for cid, t in g.groupby("constructorId"):
        assert not (t["no_updates"].any() and t["component"].notna().any()), (f, cid)
        assert len(t) == 1 or not t["no_updates"].any(), (f, cid)


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_flagged_rows_have_raw_text_and_no_component(real_table):
    bad = real_table[~real_table["parse_ok"]]
    assert len(bad) == 45
    # R10 parsing fix: only group_missing rows keep a component (the printed one); every other flagged row has none
    assert bad.loc[bad["parse_status"] != "group_missing", "component"].isna().all()
    assert bad.loc[bad["parse_status"] == "reason_missing", "raw_text"].str.len().gt(0).all()
    blank = bad[bad["parse_status"] == "page_text_empty"]     # the two Williams pages without any text (no OCR)
    assert sorted(blank["doc_file"]) == ["2024_50084_7.pdf", "2026_60295_9.pdf"] and (blank["raw_text"] == "").all()
    assert (blank["team_raw"].str.contains("Williams", case=False)).all()


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_subtype_proper_prefixes_are_only_the_genuine_2026_cooling_subtype(real_table):
    """Stronger than the 'no Local/Flow/Drag' check: a subtype that is a proper prefix of a configured subtype is either
    a wrap cut-off or genuine source text. The only one in the real table is Cadillac 2026 'Performance - Cooling' (hand-checked)."""
    c = real_table[real_table["component"].notna()]
    subs = CFG["parser"]["reason_subtypes"]
    cut = c[c["reason_subtype"].map(lambda s: any(k.lower().startswith(s.lower()) and k.lower() != s.lower() for k in subs if s))]
    assert set(cut["reason_subtype"]) == {"Cooling"} and set(cut["doc_file"]) == {"2026_60301_11.pdf"}
    assert (~cut["subtype_known"]).all()


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_new_repeated_identical_rows_are_distinct_numbered_rows_in_the_pdf(real_table):
    """Mercedes 2024 Italy and Ferrari 2025 Saudi 'Rear Wing / Performance / Drag Reduction' appear twice. Hand-checked: rows 1 and 2
    of each team page carry different descriptions (Monza: 'Subtle change to wing tip detail' vs 'Reduced chord flap'; Jeddah:
    'Decambered / shorter chord top rear wing flap' vs 'Offloaded top rear wing'). Also Mercedes 2024 Monaco rows 1 and 2
    are both printed 'Rear Wing' (row 2 describes a beam wing): counted as printed."""
    d = real_table
    for f, cid, txt in (("2024_50078_8.pdf", "mercedes", ("Subtle change to wing tip", "Reduced chord flap")),
                        ("2025_55041_7.pdf", "ferrari", ("Decambered / shorter chord", "Offloaded top rear wing"))):
        g = d[(d["doc_file"] == f) & (d["constructorId"] == cid) & (d["component"] == "Rear Wing")]
        assert len(g) == 2 and set(g["reason_subtype"]) == {"Drag Reduction"}
        text = re.sub(r"\s+", " ", " ".join(F.read_pdf_pages(LOCAL / f)[1:]))
        assert all(t in text for t in txt), f


# --- silent row loss (defects: parse_ok stays True although a numbered component row is missing) ---------------------
# Expected = numbered component rows printed in the PDF, counted by eye per team block. A row is covered if it is a component row,
# or a numbered component inside a flagged (reason_missing / unparsed) raw_text.

def _covered(real_table, f: str, cid: str) -> int:
    g = real_table[(real_table["doc_file"] == f) & (real_table["constructorId"] == cid)]
    flagged = g.loc[~g["parse_ok"], "raw_text"].fillna("")
    return int(g["component"].notna().sum()) + sum(len(re.findall(r"(?:^|\| )\d{1,2} ", t)) for t in flagged)


SILENT = [
    ("2024_50068_10.pdf", "aston_martin", 5, "row 5 'Chassis Scoop - Driver Cooling' has a reason group that is not configured and no dash"),
    ("2025_55037_7.pdf", "rb", 3, "row 3 'Front Suspension  Other - Local flow alignment': reason group 'Other' is not configured"),
    ("2025_56436_8.pdf", "ferrari", 3, "row 3 'Beam Wing' is cut at the end of the page text mid-line ('...conditions  3 Beam Wing '), not flagged"),
    ("2025_56449_7.pdf", "red_bull", 4, "rows 3 'Floor body' and 4 'Edge wing' print 'Local load' with no group and no dash"),
    ("2026_60304_14.pdf", "ferrari", 4, "row 4 'Mirror stay' has no reason cell (mid-line after the previous description)"),
    ("2026_60308_10.pdf", "ferrari", 5, "rows 4 'Diffuser' and 5 'Beam Wing' have no reason cell; only rows 2, 3 are flagged"),
    ("2026_60309_10.pdf", "red_bull", 4, "row 4 'Front Wing  Flow Conditioning' has no group"),
    ("2026_60311_11.pdf", "red_bull", 5, "rows 4 'Mirrors' and 5 'Halo' print 'Flow Conditioning' with no group"),
]


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
@pytest.mark.parametrize("f,cid,n,why", [pytest.param(*s) for s in SILENT], ids=[f"{s[0]}-{s[1]}" for s in SILENT])
def test_no_numbered_component_row_is_lost_silently(real_table, f, cid, n, why):
    assert _covered(real_table, f, cid) == n


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_no_flagged_row_stands_for_more_than_one_component(real_table):
    bad = real_table[~real_table["parse_ok"]]
    counts = bad["raw_text"].fillna("").map(lambda t: len(re.findall(r"(?:^|\| )\d{1,2} ", t)))
    assert counts.max() <= 1, bad.loc[counts > 1, ["doc_file", "team_raw", "raw_text"]].to_dict("records")
