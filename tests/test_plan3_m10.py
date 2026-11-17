"""PLAN3 R9 / M10 part 1: FIA car-presentation parser (src/pitlane/data/fia_docs.py).

Fixtures are synthetic text in the shape pypdf returns (one string per page); no FIA text is stored here.
One test runs on the three local PDFs when they are present in data/raw/fia and checks team counts only.
"""
from __future__ import annotations

import subprocess

import pandas as pd
import pytest

from pitlane.config import ROOT
from pitlane.data import fia_docs as F

CFG = F.load_cfg()
LIN = {"alphatauri": "rb_lineage", "rb": "rb_lineage", "aston_martin": "aston_lineage", "alfa": "sauber_lineage",
       "audi": "sauber_lineage", "alpine": "alpine_lineage"}
HEADER = "Updated \ncomponent \nPrimary reason \nfor update \nGeometric differences compared to previous \n" \
         "version  Brief description on how the update works  \n"


def cover(season: int = 2030, event: str = "TESTLAND", title: str = "Car Presentation Submissions") -> str:
    return (f"{season} {event} GRAND PRIX\n01 - 03 March {season}\n"
            f"The FIA Formula One Media Delegate\nDocument 9\nDate 01 March {season}\nTime 14:20\n"
            f"Title {title}\nDescription {title}\nEnclosed x.pdf\nSomeone\n")


def team_page(name: str, body: str, head: bool = True) -> str:
    return (f"Car Presentation � Testland Grand Prix \n{name} \n \n" if head else f"{name} \n") + body


def table(*rows: str) -> str:
    return HEADER + "".join(rows)


R_WING = "{n} Rear Wing Circuit specific - \nDrag Range Lower camber version \nExplained in made up words here. \n"
R_COKE = "{n} Coke/Engine \nCover \nCircuit specific - \nCooling Range smaller exit \nMade up explanation text. \n"
R_FLOOR = "{n} Floor Furniture \nPerformance � \nFlow Conditioning New edge \nSome made up words. \n"


def parse(*pages: str, **kw):
    return F.parse_pages(list(pages), CFG, LIN, **kw) if kw else F.parse_pages(list(pages), CFG, LIN)


def comp(doc) -> pd.DataFrame:
    return pd.DataFrame(doc.rows)


# --- cover ------------------------------------------------------------------------------------------------------

def test_cover_season_event_and_title():
    ok, season, event = F.parse_cover(cover(2031, "S�O�TESTO"), CFG)
    assert ok and season == 2031 and event == "s o testo"


def test_display_procedure_title_never_matched():
    for title in ("Car Display Procedure", "Automobile Display Procedure", "Car Presentation Notes"):
        doc = parse(cover(title=title), team_page("TEAM ONE", table(R_WING.format(n=1))))
        assert not doc.is_car_presentation and doc.rows == []


# --- rows -------------------------------------------------------------------------------------------------------

def test_rows_wrapped_cells_and_reason_split():
    doc = parse(cover(), team_page("TEAM ONE", table(R_COKE.format(n=1), R_WING.format(n=2), R_FLOOR.format(n=3))))
    df = comp(doc)
    assert df["component"].tolist() == ["Coke/Engine Cover", "Rear Wing", "Floor Furniture"]
    assert df["reason_group"].tolist() == ["Circuit specific", "Circuit specific", "Performance"]
    assert df["reason_subtype"].tolist() == ["Cooling Range", "Drag Range", "Flow Conditioning"]
    assert df["subtype_known"].all() and df["parse_ok"].all() and not df["no_updates"].any()
    assert df["category"].tolist() == ["sidepod_engine_cover", "rear_wing", "floor"]


def test_wrong_and_repeated_row_numbers_do_not_change_count():
    rows = [R_WING.format(n=3), R_WING.format(n=3), R_FLOOR.format(n=7), R_COKE.format(n=1)]
    df = comp(parse(cover(), team_page("TEAM ONE", table(*rows))))
    assert len(df) == 4 and df["parse_ok"].all()


def test_number_token_alone_on_its_line():
    row = "4 \nFront Wing \nPerformance - \nLocal Load Flap shape \nMade up text. \n"
    df = comp(parse(cover(), team_page("TEAM ONE", table(row))))
    assert df["component"].tolist() == ["Front Wing"] and df["reason_subtype"].tolist() == ["Local Load"]


def test_digit_at_line_start_in_description_is_not_a_row():
    row = R_WING.format(n=1) + "2 variations will be made available \nand another made up line. \n"
    df = comp(parse(cover(), team_page("TEAM ONE", table(row, R_FLOOR.format(n=2)))))
    assert df["component"].tolist() == ["Rear Wing", "Floor Furniture"]


def test_optional_flag_per_row():
    opt = "2 Front Wing Circuit specific - \nBalance Range Flap chord trim to tune balance. Optional. \nMade up text. \n"
    df = comp(parse(cover(), team_page("TEAM ONE", table(R_WING.format(n=1), opt, R_FLOOR.format(n=3)))))
    assert df["optional"].tolist() == [False, True, False]


def test_subtype_on_same_line_as_group():
    row = "1 Rear Wing Circuit specific - Drag Range \nLower camber \nMade up text. \n"
    df = comp(parse(cover(), team_page("TEAM ONE", table(row))))
    assert df["reason_subtype"].tolist() == ["Drag Range"] and df["subtype_known"].all()


def test_continuation_page_without_team_name_joins_previous_team():
    p2 = team_page("TEAM ONE", table(R_WING.format(n=1)))
    p3 = table(R_FLOOR.format(n=2))
    df = comp(parse(cover(), p2, p3))
    assert df["team_raw"].nunique() == 1 and len(df) == 2


def test_unknown_reason_group_is_reported_unparsed():
    row = "1 Rear Wing Novelty - \nDrag Range \nMade up text. \n"
    df = comp(parse(cover(), team_page("TEAM ONE", table(R_FLOOR.format(n=1), row))))
    assert (df["parse_ok"] == False).sum() == 1  # noqa: E712
    assert df.loc[df["parse_ok"], "component"].tolist() == ["Floor Furniture"]


# --- no updates -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["No updates submitted for this event.", "No Updates", "No update for this event",
                                  "No updates for this event", "No updates submitted  for  this even t."])
def test_no_update_wordings(text):
    df = comp(parse(cover(), team_page("TEAM ONE", text + " \n")))
    assert len(df) == 1 and df["no_updates"].iloc[0] and df["component"].isna().iloc[0] and df["parse_ok"].iloc[0]


def test_team_with_rows_is_not_no_update_even_if_prose_says_so():
    row = "1 Rear Wing Circuit specific - \nDrag Range \nThere were no updates to the main plane. \n"
    df = comp(parse(cover(), team_page("TEAM ONE", table(row))))
    assert len(df) == 1 and not df["no_updates"].iloc[0] and df["component"].iloc[0] == "Rear Wing"


# --- team names -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,cid,lineage", [
    ("*SCUDERIA FERRARI HP*", "ferrari", "ferrari"),
    ("ORACLE RED BULL RACING", "red_bull", "red_bull"),
    ("Visa Cash App Racing Bulls", "rb", "rb_lineage"),
    ("VISA CASH APP RB F1 TEAM", "rb", "rb_lineage"),
    ("STAKE F1 TEAM KICK SAUBER", "sauber", "sauber"),
    ("KICK Sauber F1 Team", "sauber", "sauber"),
    ("Audi Revolut F1 Team", "audi", "sauber_lineage"),
    ("Aston Martin Aramco Cognizant F1 Team", "aston_martin", "aston_lineage"),
    ("*Mercedes-AMG PETRONAS F1 Team*", "mercedes", "mercedes"),
    ("TGR HAAS F1 TEAM", "haas", "haas"),
    ("BWT Alpine F1 Team", "alpine", "alpine_lineage"),
    ("*ATLASSIAN WILLIAMS RACING*", "williams", "williams"),
    ("Cadillac", "cadillac", "cadillac"),
])
def test_team_decoration_maps_to_constructor_and_lineage(raw, cid, lineage):
    norm, got, lin = F.map_team(raw, CFG, LIN)
    assert got == cid and lin == lineage and "*" not in norm and norm == norm.lower()


def test_unknown_team_is_kept_without_id():
    norm, cid, lin = F.map_team("Some New Team", CFG, LIN)
    assert norm == "some new team" and cid is None and lin is None


# --- category rules ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,cat", [
    ("Front Wing", "front_wing"), ("Rear Wing", "rear_wing"), ("Beam Wing", "beam_wing"),
    ("Floor Furniture", "floor"), ("Floor Edge", "floor"), ("Coke/Engine Cover", "sidepod_engine_cover"),
    ("Sidepod", "sidepod_engine_cover"), ("Cooling Louvres", "sidepod_engine_cover"),
    ("Front Corner", "front_corner_suspension"), ("Rear Corner", "rear_corner_suspension"),
    ("Steering wheel", "other"), ("Mirrors", "other")])
def test_component_category(text, cat):
    assert F.categorise(text, CFG) == cat


def test_category_rules_come_from_config_not_code():
    cfg = {"component_categories": [["mine", ["mirror"]]]}
    assert F.categorise("Mirrors", cfg) == "mine" and F.categorise("Rear Wing", cfg) == "other"


# --- files ------------------------------------------------------------------------------------------------------

MANIFEST_COLS = "season,page_order,event,event_id,found,doc_key,published_cet,url,sha256,bytes,file"


def write_manifest(path, rows: list[str]):
    path.write_text("\n".join([MANIFEST_COLS, *rows]) + "\n", encoding="utf-8")


def man_row(name: str, data: bytes, found: bool = True, sha: str | None = None) -> str:
    import hashlib
    h = sha or hashlib.sha256(data).hexdigest()
    return f"2030,1,Testland Grand Prix,9001,{found},7,2030-03-01 14:20,http://x,{h if found else ''},{len(data)},{name if found else ''}"


def fake_parse_pdf(path, cfg=None):
    return pd.DataFrame([{c: None for c in F.COLUMNS} | {"doc_file": path.name, "team_raw": "T"}], columns=F.COLUMNS)


def test_parse_dir_empty_manifest_writes_empty_table(tmp_path):
    src, out, man = tmp_path / "in", tmp_path / "out" / "t.parquet", tmp_path / "m.csv"
    src.mkdir()
    write_manifest(man, [man_row("", b"", found=False)])
    df = F.parse_dir(src, out, manifest=man)
    assert df.empty and list(df.columns) == F.COLUMNS and out.exists()


def test_parse_dir_sha_mismatch_is_an_error(tmp_path):
    src, man = tmp_path / "in", tmp_path / "m.csv"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"real bytes")
    write_manifest(man, [man_row("a.pdf", b"real bytes", sha="0" * 64)])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        F.parse_dir(src, tmp_path / "o.parquet", manifest=man)


def test_parse_dir_missing_manifest_file_is_an_error(tmp_path):
    src, man = tmp_path / "in", tmp_path / "m.csv"
    src.mkdir()
    write_manifest(man, [man_row("gone.pdf", b"x")])
    with pytest.raises(FileNotFoundError):
        F.parse_dir(src, tmp_path / "o.parquet", manifest=man)


def test_parse_dir_parses_only_manifest_files_and_carries_manifest_fields(tmp_path, monkeypatch):
    src, man = tmp_path / "in", tmp_path / "m.csv"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"AAA")
    (src / "stray.pdf").write_bytes(b"not in the manifest")
    write_manifest(man, [man_row("a.pdf", b"AAA"), man_row("", b"", found=False)])
    seen: list[str] = []
    monkeypatch.setattr(F, "parse_pdf", lambda path, cfg=None: (seen.append(path.name), fake_parse_pdf(path))[1])
    df = F.parse_dir(src, tmp_path / "o.parquet", manifest=man)
    assert seen == ["a.pdf"] and set(df["doc_file"]) == {"a.pdf"}
    assert df["manifest_season"].iloc[0] == 2030 and df["event_id"].iloc[0] == 9001
    assert df["manifest_event"].iloc[0] == "Testland Grand Prix" and df["published_cet"].iloc[0] == "2030-03-01 14:20"


def test_load_manifest_keeps_found_rows_only(tmp_path):
    man = tmp_path / "m.csv"
    write_manifest(man, [man_row("", b"", found=False), man_row("a.pdf", b"A")])
    assert F.load_manifest(man)["file"].tolist() == ["a.pdf"]


def test_raw_and_interim_paths_are_git_ignored():
    try:
        r = subprocess.run(["git", "check-ignore", "data/raw/fia/x.pdf", "data/interim/m10_car_presentations.parquet"],
                           cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        pytest.skip("git not available")
    assert r.stdout.split() == ["data/raw/fia/x.pdf", "data/interim/m10_car_presentations.parquet"]


LOCAL = ROOT / "data" / "raw" / "fia"
REAL = F.load_manifest()
HAVE_ALL = all((LOCAL / f).exists() for f in REAL["file"])
EXPECT = {"2024_50064_9.pdf": 10, "2025_55060_8.pdf": 10, "2026_60315_12.pdf": 11}  # Saudi 2024, Abu Dhabi 2025, Sao Paulo 2026


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_manifest_files_match_sha256_and_no_duplicates():
    assert REAL["sha256"].is_unique and len(REAL) == 67
    for _, r in REAL.iterrows():
        assert F.sha256_file(LOCAL / r["file"]) == r["sha256"], r["file"]


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_parse_dir_file_set_equals_manifest(tmp_path):
    df = F.parse_dir(out=tmp_path / "t.parquet")
    assert set(df["doc_file"]) == set(REAL["file"])
    assert (df["season"] == df["manifest_season"]).all()
    assert df["event_id"].notna().all() and df["published_cet"].notna().all()


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_local_pdfs_team_counts():
    for name, n_teams in EXPECT.items():
        df = F.parse_pdf(LOCAL / name)
        assert df["team_raw"].nunique() == n_teams, name
        assert df["constructorId"].notna().all(), name
        assert df["lineage"].notna().all(), name
    abu = F.parse_pdf(LOCAL / "2025_55060_8.pdf")
    assert abu["no_updates"].all() and abu["component"].isna().all()


# =====================================================================================================================
# R9 tester additions: one synthetic test per "R9 parsing fix" rule, each with a case where the rule must NOT fire,
# then hand-checked rows from real PDFs, then data-quality tests on the whole parsed table.
# R10: the strict xfails that recorded parser defects were removed after the parsing fixes; the tests now pass.
# =====================================================================================================================

import hashlib  # noqa: E402
import re  # noqa: E402

FER, MCL, MER = "Scuderia Ferrari", "McLaren F1 Team", "Mercedes-AMG PETRONAS F1 Team"   # names that map to a constructor
FOOT = "© 2030. All Rights Reserved. Highly Confidential. \n"


def triples(doc, team_sub: str | None = None) -> list[tuple]:
    df = pd.DataFrame(doc.rows)
    if team_sub is not None:
        df = df[df["team_raw"].str.contains(team_sub, case=False, regex=False)]
    df = df[df["component"].notna()]
    return list(zip(df["component"], df["reason_group"], df["reason_subtype"]))


# --- rule 1: a flat line is cut back into one cell per line ---------------------------------------------------------

FLAT = ("Car Presentation – Testland Grand Prix " + FER + " Updated component Primary reason for update Geometric "
        "differences compared to previous version Brief description on how the update works 1 Front Wing Performance - "
        "Flow Conditioning New flap. Made up words. 2 Rear Wing Circuit specific - Drag Range Lower camber. Optional. "
        "3 Floor Performance - Local Load Edge change.")


def test_flat_page_is_cut_into_rows_team_name_taken_from_header_line():
    doc = parse(cover(), FLAT)
    assert triples(doc) == [("Front Wing", "Performance", "Flow Conditioning"),
                            ("Rear Wing", "Circuit specific", "Drag Range"),
                            ("Floor", "Performance", "Local Load")]
    df = comp(doc)
    assert df["team_raw"].unique().tolist() == [FER] and df["constructorId"].unique().tolist() == ["ferrari"]
    assert df["optional"].tolist() == [False, True, False]      # "Optional." splits the cell but still flags row 2


def test_flat_split_does_not_fire_without_a_row_number_near_the_group():
    # prose mentioning a group, no number in the lookback: left alone
    for s in ["improves Performance - Flow Conditioning in general",
              "7 a b c d e f g h Performance - Flow Conditioning x",       # number further than the 7-token lookback
              "Updated component Primary reason",                          # header words at line start: no split before
              "2 variations of the Front Wing"]:
        assert F.expand_flat_lines([s], CFG) == [s]


def test_flat_split_cuts_before_row_number_and_after_group_dash_even_for_unknown_subtype():
    assert F.expand_flat_lines(["1 Rear Wing Performance - Novel Thing words more words"], CFG) == \
        ["1 Rear Wing Performance -", "Novel Thing words more words"]
    assert F.expand_flat_lines(["intro text 1 Rear Wing Circuit specific - Drag Range then text"], CFG) == \
        ["intro text", "1 Rear Wing Circuit specific - Drag Range", "then text"]


def test_flat_noupdates_line_with_team_name_on_the_header_line():
    doc = parse(cover(), "Car Presentation – Testland Grand Prix " + MCL + " No updates submitted for this event. ")
    df = comp(doc)
    assert len(df) == 1 and df["no_updates"].iloc[0] and df["team_raw"].iloc[0] == MCL


# --- rule 2: header-line team name ---------------------------------------------------------------------------------

def test_header_line_without_tail_yields_no_name_and_the_next_line_is_the_team():
    doc = parse(cover(), "Car Presentation – Testland Grand Prix \n" + FER + " \n" + table(R_WING.format(n=1)))
    assert comp(doc)["team_raw"].tolist() == [FER]


def test_header_tail_is_kept_only_after_grand_prix():
    assert F._keep_header_tail("Car Presentation – Testland Grand Prix", CFG) == ""
    assert F._keep_header_tail("Car Presentation – Testland Grand Prix  TEAM ONE ", CFG) == "TEAM ONE"
    assert F._keep_header_tail("Car Presentation – 2025 Australian Grand Prix *SCUDERIA FERRARI HP*", CFG) == "*SCUDERIA FERRARI HP*"


@pytest.mark.parametrize("hdr", ["Car Presentation – Emilia - Romagna *Scuderia Ferrari*",
                                 "Car Presentation – Testland Grand Prix in Malaysia Scuderia Ferrari"])
def test_team_raw_is_clean_for_odd_header_lines(hdr):
    df = comp(parse(cover(), hdr + " \n" + table(R_WING.format(n=1))))
    assert not df["team_raw"].str.contains("Car Presentation|Malaysia", case=False).any()


# --- rule 3: footer leak ---------------------------------------------------------------------------------------------

def test_leaked_footer_before_header_is_dropped_and_rows_go_to_the_right_team():
    for sign in ("©", "�"):
        leak = f"{FER} \n3 {sign} 2030. All Rights Reserved. Highly Confidential. \n"
        doc = parse(cover(), team_page(FER, table(R_WING.format(n=1))), leak + team_page(MCL, table(R_FLOOR.format(n=1))))
        assert triples(doc, "Ferrari") == [("Rear Wing", "Circuit specific", "Drag Range")]
        assert triples(doc, "McLaren") == [("Floor Furniture", "Performance", "Flow Conditioning")]
        assert comp(doc)["parse_ok"].all()


def test_footer_regex_does_not_fire_on_prose_or_a_page_without_footer():
    foot = re.compile(CFG["parser"]["footer_regex"], re.I)
    assert not foot.search("All Rights Reserved by the team in this made up text.")
    assert not foot.search("copyright 2030 rights reserved")
    for s in ("© 2026. All Rights Reserved.", "� 2025 All Rights Reserved. Highly Confidential.", "12 © 2024 All Rights Reserved"):
        assert foot.search(s)
    doc = parse(cover(), team_page(FER, table(R_WING.format(n=1) + "All Rights Reserved by the team in this made up text. \n")))
    assert triples(doc) == [("Rear Wing", "Circuit specific", "Drag Range")] and comp(doc)["parse_ok"].all()


def test_footer_text_inside_a_row_is_removed_without_losing_the_row():
    doc = parse(cover(), team_page(FER, table(R_WING.format(n=1)) + FOOT + R_FLOOR.format(n=2)))
    assert [t[0] for t in triples(doc)] == ["Rear Wing", "Floor Furniture"]


def test_rows_printed_before_a_footer_on_a_headerless_page_are_kept():
    doc = parse(cover(), team_page(FER, table(R_WING.format(n=1))), table(R_FLOOR.format(n=2)) + FER + " 3 " + FOOT)
    assert [t[0] for t in triples(doc)] == ["Rear Wing", "Floor Furniture"]


# --- rule 4: "No u / pdates" repair ------------------------------------------------------------------------------------

def test_no_u_pdates_split_over_two_lines_is_repaired():
    df = comp(parse(cover(), team_page(FER, "No u\npdates submitted for this event. \n")))
    assert len(df) == 1 and df["no_updates"].iloc[0] and df["parse_ok"].iloc[0]


def test_no_u_pdates_repair_does_not_fire_on_other_text():
    pat, sub = CFG["parser"]["text_fixes"][0]
    assert re.sub(pat, sub, "No u\npdates submitted") == "No updates submitted"
    for s in ["No use of\npdates", "no u\npdates", "No u\nsed parts", "Not u\npdates"]:
        assert re.sub(pat, sub, s) == s
    df = comp(parse(cover(), team_page(FER, "No use of \npdates are not here \n")))
    assert not df["no_updates"].any() and not df["parse_ok"].iloc[0]     # not a no-update team: flagged, not guessed


# --- rule 5: overflow prose belongs to the previous team -------------------------------------------------------------

def test_overflow_prose_page_is_attached_to_previous_team_without_new_rows():
    base = parse(cover(), team_page(FER, table(R_WING.format(n=1))))
    doc = parse(cover(), team_page(FER, table(R_WING.format(n=1))), "which is a plain continuation of the text. \n")
    assert comp(doc)["team_raw"].nunique() == 1 and len(comp(doc)) == len(comp(base)) == 1
    assert comp(doc)["parse_ok"].all() and not comp(doc)["no_updates"].any()


def test_overflow_rule_does_not_fire_for_a_recognised_team_or_the_first_team_page():
    doc = parse(cover(), team_page(FER, table(R_WING.format(n=1))), team_page(MCL, "No updates submitted for this event. \n"))
    df = comp(doc)
    assert df["team_raw"].tolist() == [FER, MCL] and df["no_updates"].tolist() == [False, True]
    first = comp(parse(cover(), "which looks like prose but is the first page \n" + table(R_WING.format(n=1))))
    assert first["team_raw"].iloc[0].startswith("which looks like prose")     # no previous team: it is taken as the name


def test_overflow_prose_naming_a_team_is_not_a_new_team():
    doc = parse(cover(), team_page(FER, table(R_WING.format(n=1))), "which is similar to the Red Bull concept of last year. \n")
    assert comp(doc)["team_raw"].nunique() == 1


def test_unmapped_new_team_with_no_updates_is_not_silently_dropped():
    doc = parse(cover(), team_page(FER, table(R_WING.format(n=1))), team_page("Brand New Team", "No updates submitted for this event. \n"))
    assert comp(doc)["team_raw"].nunique() == 2


# --- rule 6: reason group with no subtype (Reliability / Weight / Cost) ----------------------------------------------

def test_reason_group_without_subtype_rows():
    body = "1 Rear Corner Reliability Enlarged exit duct \nIn preparation. \n2 Floor Weight \nreduced \n3 Sidepod Cost  Cheaper \n"
    doc = parse(cover(), team_page(FER, table(body)))
    assert triples(doc) == [("Rear Corner", "Reliability", ""), ("Floor", "Weight", ""), ("Sidepod", "Cost", "")]
    assert comp(doc)["subtype_known"].all() and comp(doc)["parse_ok"].all()


def test_reason_group_without_subtype_does_not_fire_on_prose_or_lower_case():
    doc = parse(cover(), team_page(FER, table(R_WING.format(n=1) + "This reduces Weight of the part and Cost too. \nGood Reliability gains. \n")))
    assert triples(doc) == [("Rear Wing", "Circuit specific", "Drag Range")]
    low = comp(parse(cover(), team_page(FER, table("1 Rear Corner reliability Enlarged exit duct \nText. \n"))))
    assert low["component"].isna().all() and not low["parse_ok"].iloc[0]      # case-sensitive: not guessed, flagged


def test_numbered_description_line_with_weight_is_not_a_row():
    doc = parse(cover(), team_page(FER, table(R_WING.format(n=1) + "2 Weight reduction made possible by the new part. \n")))
    assert len(triples(doc)) == 1


# --- rule 7: case-insensitive reason groups ------------------------------------------------------------------------------

def test_reason_groups_are_matched_case_insensitively():
    body = "1 Rear Wing CIRCUIT SPECIFIC - Drag Range \nlower \n2 Floor performance - Local Load \nx \n3 Front Wing Circuit Specific - Balance Range \ny\n"
    doc = parse(cover(), team_page(FER, table(body)))
    assert [t[0] for t in triples(doc)] == ["Rear Wing", "Floor", "Front Wing"]
    assert [t[2] for t in triples(doc)] == ["Drag Range", "Local Load", "Balance Range"]


def test_reason_group_is_normalised_to_the_configured_spelling():
    body = "1 Rear Wing CIRCUIT SPECIFIC - Drag Range \nlower \n2 Floor performance - Local Load \nx \n"
    got = {t[1] for t in triples(parse(cover(), team_page(FER, table(body))))}
    assert got <= set(CFG["parser"]["reason_groups"])


# --- rule 8: spaced-dash "unparsed" signal -------------------------------------------------------------------------------

def test_spaced_dash_outside_a_row_is_reported_unparsed():
    rows, left = F.parse_team_body(["1 Rear Wing Novelty -", "Drag Range", "text"], CFG)
    assert rows == [] and left == ["1 Rear Wing Novelty -", "Drag Range", "text"]    # no anchor at all: every line is left over
    rows, left = F.parse_team_body(["1 Rear Wing Performance - Local Load", "text", "2 Floor Novelty -", "more"], CFG)
    assert len(rows) == 1 and left == ["2 Floor Novelty -"]                         # only the dash line is flagged
    doc = parse(cover(), team_page(FER, table(R_WING.format(n=1).replace("Explained", "Explained -\nExplained"))))
    df = comp(doc)
    assert df["parse_ok"].tolist() == [True, False] and df["raw_text"].iloc[1] == "Explained -"


def test_hyphen_wrap_and_unspaced_dash_are_not_unparsed():
    doc = parse(cover(), team_page(FER, table(R_WING.format(n=1).replace("Explained", "re-\nexplained"))))
    assert comp(doc)["parse_ok"].all() and len(comp(doc)) == 1
    rows, left = F.parse_team_body(["1 Rear Wing Performance - Local Load", "a well-known fix", "x-", "text"], CFG)
    assert len(rows) == 1 and left == []


def test_unknown_subtype_is_kept_with_subtype_known_false_not_unparsed():
    df = comp(parse(cover(), team_page(FER, table("1 Rear Wing Circuit specific - Novel Thing \nText. \n"))))
    assert df["reason_subtype"].iloc[0] == "Novel Thing" and not df["subtype_known"].iloc[0] and df["parse_ok"].iloc[0]


# --- defects found by the tester, fixed in R10 (synthetic text mirrors the real pypdf output) ---------------------------------

def test_subtype_wrapped_over_two_lines_is_joined():
    df = comp(parse(cover(), team_page(MER, table("1 Rear Wing  Performance - Local \nLoad \nSmall winglets. \n"))))
    assert df["reason_subtype"].tolist() == ["Local Load"] and df["subtype_known"].all()


def test_row_starting_mid_line_after_previous_description_is_kept():
    body = ("1 Front Wing Performance - Flow \nConditioning \nChange in chord distribution.  Results text control.  2 Front Wing \n"
            "Endplate \nPerformance - Flow \nConditioning \nFront wing endplate adjusted. \n")
    doc = parse(cover(), team_page(MER, table(body)))
    assert [t[0] for t in triples(doc)] == ["Front Wing", "Front Wing Endplate"]


def test_row_cut_off_after_component_is_not_silently_dropped():
    doc = parse(cover(), team_page(FER, table(R_WING.format(n=1)).rstrip() + "  3 Beam Wing "))
    df = comp(doc)
    assert len(df) == 2 and not df["parse_ok"].all()


def test_page_with_team_name_after_the_no_updates_line_keeps_the_team():
    doc = parse(cover(), team_page(FER, table(R_WING.format(n=1))),
                "No updates submitted for this event. \nCar Presentation – Testland Grand Prix \n" + MER + " \n")
    df = comp(doc)
    assert MER in df["team_raw"].tolist() and df.loc[df["team_raw"] == MER, "no_updates"].all()


def test_table_before_the_header_and_team_name_goes_to_the_named_team():
    haas = "MONEYGRAM HAAS F1 TEAM"
    doc = parse(cover(), team_page("BWT Alpine F1 Team", "No updates submitted for this event. \n"),
                table("1 Floor Body Performance - \nLocal Load \nCentral floor. \n") + "Car Presentation – Testland Grand Prix \n" + haas + "  \n")
    df = comp(doc)
    alp = df[df["team_raw"].str.contains("Alpine")]
    assert alp["no_updates"].all() and alp["component"].isna().all()
    assert triples(doc, "HAAS") == [("Floor Body", "Performance", "Local Load")]


# =====================================================================================================================
# Real PDFs: hand-checked rows (compared by the tester against pypdf text, row by row), cross-checks, data quality.
# =====================================================================================================================

FD, FL, LL, DR, CR, BR = "Flow Conditioning", "Local Load", "Local Load", "Drag Range", "Cooling Range", "Balance Range"
P, CS = "Performance", "Circuit specific"

# (file, team substring) -> [(component, reason_group, reason_subtype or None)]. None = subtype free text, not compared.
HAND = {
    ("2024_50068_10.pdf", "MCLAREN"): [("Front Wing", P, FD), ("Front Suspension", P, FD), ("Front Corner", P, FD), ("Floor Body", P, LL),
                                      ("Sidepod Inlet", P, FD), ("Coke/Engine Cover", P, FD), ("Cooling Louvres", P, FD),
                                      ("Rear Suspension", P, FD), ("Rear Corner", P, LL), ("Beam Wing", CS, DR)],
    ("2024_50068_10.pdf", "WILLIAMS"): [("Front Wing", P, LL)],
    ("2024_50072_9.pdf", "FERRARI"): [("Rear Wing", CS, DR), ("Coke/Engine Cover", P, FD), ("Floor Fences", P, FD), ("Floor Body", P, FD),
                                     ("Floor Edge", P, LL), ("Diffuser", P, LL), ("Halo", P, FD)],
    ("2024_50072_9.pdf", "ASTON"): [("Front Suspension", P, LL), ("Front Corner", P, LL), ("Rear Corner", P, LL)],
    ("2024_50072_9.pdf", "VISA"): [("Front Corner", CS, CR), ("Coke/Engine Cover", P, FD), ("Sidepod Inlet", P, FD), ("Floor Body", P, LL),
                                  ("Rear Wing", P, LL), ("Beam Wing", CS, DR)],
    ("2024_50072_9.pdf", "SAUBER"): [("Rear Wing", P, FD), ("Front Corner", P, FD)],
    ("2024_50066_12.pdf", "MCLAREN"): [("Front Corner", CS, CR)],
    ("2024_50066_12.pdf", "ASTON"): [("Floor Body", P, LL), ("Floor Fences", P, LL), ("Floor Edge", P, LL), ("Diffuser", P, LL)],
    ("2024_50066_12.pdf", "SAUBER"): [("Floor Body", P, FD)],
    ("2024_50077_9.pdf", "McLaren"): [("Front Corner", P, FD), ("Front Suspension", P, FD), ("Floor Edge", P, LL), ("Rear Corner", P, FD),
                                     ("Rear Wing", CS, DR), ("Beam Wing", CS, DR)],
    ("2024_50077_9.pdf", "Alpine"): [("Front Suspension", P, FD), ("Rear Corner", P, LL)],
    ("2024_50077_9.pdf", "HAAS"): [("Front Wing", P, FD), ("Nose", P, FD), ("Front Suspension", P, FD), ("Front Corner", P, FD)],
    ("2024_50074_9.pdf", "MCLAREN"): [("Rear Wing", CS, DR), ("Beam Wing", CS, DR), ("Beam Wing", CS, DR), ("Beam Wing", CS, DR),
                                     ("Coke/Engine Cover", CS, CR)],
    ("2025_55037_7.pdf", "FERRARI"): [("Front Suspension", P, FD), ("Sidepod Inlet / Engine cover", P, FD), ("Rear Wing / Beam Wing", P, LL)],
    ("2025_55037_7.pdf", "Mercedes"): [("Sidepod Inlet", P, FD), ("Floor Body", P, LL), ("Coke/Engine Cover", P, FD), ("Beam Wing", P, LL),
                                      ("Front Suspension", P, FD), ("Rear Wing", P, LL)],
    ("2025_56436_8.pdf", "Red Bull"): [("Coke/Engine Cover", P, LL), ("Rear Suspension", P, LL), ("Rear Corner", P, LL)],
    ("2026_60300_8.pdf", "WILLIAMS"): [("Floor", P, None), ("Sidepod", P, None), ("Bodywork – engine cover / coke / cooling exits", P, None),
                                       ("Mirror assembly", P, None), ("Tailpipe bracket", P, None), ("RIS fairings", P, None),
                                       ("Rear brace wing", P, None)],
    ("2026_60300_8.pdf", "Audi"): [("Front Suspension", P, None), ("Floor Edge and Diffuser", P, LL)],
    ("2026_60300_8.pdf", "Cadillac"): [("Front Wing Endplate", P, FD), ("Front Wing Flap", P, FD), ("Mirror Stay", P, FD), ("Forward Floorboard", P, LL),
                                      ("Floor Body", P, LL), ("Diffuser", P, LL), ("Rear Suspension", P, FD), ("Rear Corner", P, LL),
                                      ("Exhaust Tailpipe Bracket", P, LL)],
    ("2025_55039_9.pdf", "Red Bull"): [("Coke/Engine Cover", "Reliability", ""), ("Rear corner", "Reliability", ""), ("Rear Suspension", P, LL)],
    ("2025_55039_9.pdf", "Sauber"): [("Floor Body", P, LL), ("Rear Wing", P, LL), ("Beam Wing", P, LL)],
}
HAND_FILES = sorted({f for f, _ in HAND})
HAVE_HAND = all((LOCAL / f).exists() for f in HAND_FILES)


@pytest.fixture(scope="module")
def parsed_files():
    return {f: F.parse_pdf(LOCAL / f) for f in HAND_FILES}


@pytest.mark.skipif(not HAVE_HAND, reason="local FIA PDFs not present")
@pytest.mark.parametrize("key", sorted(HAND))
def test_hand_checked_rows_component_group_subtype(parsed_files, key):
    f, team = key
    df = parsed_files[f]
    df = df[df["team_raw"].str.contains(team, case=False, regex=False) & df["component"].notna()]
    got = list(zip(df["component"], df["reason_group"], df["reason_subtype"]))
    want = HAND[key]
    assert len(got) == len(want), f"{key}: {len(got)} rows parsed, {len(want)} in the PDF text"
    for g, w in zip(got, want):
        assert g[0] == w[0] and g[1] == w[1] and (w[2] is None or g[2] == w[2]), (key, g, w)
    assert df["parse_ok"].all()


@pytest.mark.skipif(not HAVE_HAND, reason="local FIA PDFs not present")
@pytest.mark.parametrize("component,cat", [("Front Wing Endplate", "front_wing"), ("Floor Edge and Diffuser", "floor"),
                                           ("Forward Floorboard", "floor"), ("Mirror Stay", "other"), ("Exhaust Tailpipe Bracket", "other"),
                                           ("Front Wing Flap", "front_wing"), ("Tailpipe bracket", "other"), ("Halo", "other"),
                                           ("Sidepod Inlet / Engine cover", "sidepod_engine_cover")])
def test_category_of_components_seen_in_real_documents(component, cat):
    assert F.categorise(component, CFG) == cat


def test_combined_rear_wing_beam_wing_row_keeps_first_match_and_gets_a_secondary_category():
    # R10 ruling: the first-match category is unchanged; the second matched category goes to category_secondary.
    assert F.categorise("Rear Wing / Beam Wing", CFG) == "beam_wing"
    assert F.categorise_secondary("Rear Wing / Beam Wing", CFG) == "rear_wing"
    assert F.categorise_secondary("Beam Wing", CFG) == "" and F.categorise_secondary("Mirrors", CFG) == ""


@pytest.fixture(scope="module")
def all_docs(tmp_path_factory):
    return F.parse_dir(out=tmp_path_factory.mktemp("m10") / "t.parquet")


def pdf_text(f: str) -> str:
    return " ".join(F.read_pdf_pages(LOCAL / f)[1:])


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_independent_reason_cell_count_matches_parsed_rows_per_document(all_docs):
    """Count 'Performance -' / 'Circuit specific -' cells straight from the pypdf text; parsed rows + unparsed flags must equal it.
    After the R10 parsing fixes every document matches."""
    rx = re.compile(r"(?:Circuit\s*specific|Performance)\s*[-–—�]", re.I)
    bad = []
    for f, g in all_docs.groupby("doc_file"):
        n_pdf = len(rx.findall(pdf_text(f)))
        n_rows = int(g["reason_group"].str.lower().isin(["performance", "circuit specific"]).sum())
        if n_pdf != n_rows:
            bad.append((f, n_pdf - n_rows, int((~g["parse_ok"]).sum())))      # (file, rows missing, unparsed flags in the file)
    # R10 parsing fix: the seven documents that lost one row (six flagged, 2026 Singapore silent) now match the text.
    # R10 parsing fix, 2nd revision: three documents print a reason group WITHOUT its dash ('Circuit specific Modification ...',
    # 'Performance Local Load'); the parser now accepts these rows, the dash-based count in the text cannot see them (-1 each).
    assert bad == [("2025_56437_11.pdf", -1, 2), ("2026_60302_15.pdf", -1, 0), ("2026_60314_12.pdf", -1, 1)], bad


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_independent_reliability_count_matches(all_docs):
    for f, g in all_docs.groupby("doc_file"):
        n_pdf = len(re.findall(r"\d+ [A-Za-z/ ]{3,40}? (?:Reliability|Weight|Cost)\b", re.sub(r"\s+", " ", pdf_text(f))))
        assert n_pdf == int(g["reason_group"].isin(["Reliability", "Weight", "Cost"]).sum()), f


def _expected_teams(f: str) -> int:
    return 11 if f.startswith("2026") else 10


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
@pytest.mark.parametrize("f", sorted(REAL["file"]))
def test_every_document_has_every_team_exactly_once_and_distinct_constructors(all_docs, f):
    g = all_docs[all_docs["doc_file"] == f]
    assert g["constructorId"].nunique() == _expected_teams(f)
    assert g["constructorId"].notna().all() and g["team_raw"].nunique() == g["constructorId"].nunique()


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_2025_japan_mercedes_and_haas_pages_are_assigned_to_the_named_team():
    """pypdf returns these two pages in reversed order (Mercedes = 'No updates' + header + name, Haas = table + header + name).
    R10 parsing fix: each goes to the team named on its own page; Alpine keeps its own no-updates flag."""
    pages = F.read_pdf_pages(LOCAL / "2025_55039_9.pdf")
    assert "Mercedes-AMG PETRONAS" in pages[5] and "No updates" in pages[5]
    assert "HAAS" in pages[8] and "Floor Body" in pages[8]
    assert [i for i, t in enumerate(pages) if not t.strip()] == [4, 9, 11, 13, 15]     # the truly empty ones: image-only pages
    df = F.parse_pdf(LOCAL / "2025_55039_9.pdf")
    assert df["team_raw"].nunique() == 10 and not (df["parse_status"] == "team_missing").any()
    mer = df[df["team_raw"].str.contains("Mercedes")]
    assert mer["no_updates"].all() and mer["component"].isna().all()
    haas = df[df["team_raw"].str.contains("HAAS", case=False)]
    assert haas["component"].tolist() == ["Floor Body"]
    alp = df[df["team_raw"].str.contains("Alpine")]
    assert alp["no_updates"].all() and alp["component"].isna().all()


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_published_cet_is_carried_on_every_row_and_equals_the_manifest(all_docs):
    assert all_docs["published_cet"].notna().all() and (all_docs["published_cet"].str.len() > 0).all()
    man = REAL.set_index("file")["published_cet"]
    assert (all_docs["published_cet"].values == man.reindex(all_docs["doc_file"]).values).all()
    ts = pd.to_datetime(all_docs["published_cet"], format="%d.%m.%y %H:%M")      # the format a later model must parse (day first)
    assert ts.notna().all() and (ts.dt.year == all_docs["season"]).all()
    assert all_docs["event_id"].notna().all() and all_docs.groupby("doc_file")["published_cet"].nunique().max() == 1


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_table_shape_ranges_and_missing_values(all_docs):
    d = all_docs
    # R10 parsing fix, 2nd revision: 1163 ok + 14 group_missing component rows, 45 flagged rows (was 1453 / 1156 / 23)
    assert len(d) == 1482 and int(d["component"].notna().sum()) == 1177 and int(d["parse_ok"].eq(False).sum()) == 45
    assert d["season"].isin([2024, 2025, 2026]).all() and d.groupby("season")["doc_file"].nunique().to_dict() == {2024: 23, 2025: 24, 2026: 20}
    assert d[["event", "team_raw", "constructorId", "lineage", "doc_file", "doc_sha256", "page"]].notna().all().all()
    comp_rows = d[d["component"].notna() & d["parse_ok"]]     # group_missing rows (parse_ok False) keep component and subtype, group None
    assert comp_rows[["category", "reason_group", "reason_subtype", "subtype_known", "optional"]].notna().all().all()
    gm = d[d["parse_status"] == "group_missing"]
    assert gm[["component", "category", "reason_subtype"]].notna().all().all() and gm["reason_group"].isna().all()
    assert comp_rows["component"].str.len().between(3, 60).all()
    assert set(comp_rows["category"]) <= {"beam_wing", "floor", "front_corner_suspension", "front_wing", "other",
                                          "rear_corner_suspension", "rear_wing", "sidepod_engine_cover"}
    assert comp_rows["reason_group"].str.lower().isin([g.lower() for g in CFG["parser"]["reason_groups"]]).all()
    # a no-update row has no component; a team never has both rows and a no-update flag; every row's sha matches a manifest file
    nu = d[d["no_updates"]]
    assert nu["component"].isna().all() and nu["parse_ok"].all()
    per_team = d.groupby(["doc_file", "team_raw"]).agg(n=("component", lambda s: int(s.notna().sum())), nu=("no_updates", "max"))
    assert not ((per_team["n"] > 0) & per_team["nu"]).any()
    assert set(d["doc_sha256"]) == set(REAL["sha256"])
    # unparsed rows carry their raw text; parsed rows do not
    assert d.loc[~d["parse_ok"] & (d["parse_status"] != "group_missing"), "component"].isna().all()
    assert d.loc[d["parse_ok"], "raw_text"].isna().all()
    assert (d["page"] >= 2).all()


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_repeated_identical_rows_are_genuine_option_variants_not_double_counting(all_docs):
    """23 component rows repeat (19 before R10 parsing fixes) (component, group, subtype) inside a team and document. Checked by hand for 2024 British McLaren:
    three Beam Wing variants (high/mid/low load) are three separate numbered rows in the PDF. Count them as printed."""
    d = all_docs[all_docs["component"].notna()]
    k = ["doc_file", "constructorId", "component", "reason_group", "reason_subtype"]
    assert int(d.duplicated(k, keep=False).sum()) == 23
    mcl = d[(d["doc_file"] == "2024_50074_9.pdf") & (d["constructorId"] == "mclaren") & (d["component"] == "Beam Wing")]
    assert len(mcl) == 3 and "High load Beamwing" in pdf_text("2024_50074_9.pdf") and "Low load Beamwing" in pdf_text("2024_50074_9.pdf")


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_table_reason_group_has_only_the_configured_spellings(all_docs):
    assert set(all_docs["reason_group"].dropna()) <= set(CFG["parser"]["reason_groups"])


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_table_has_no_truncated_one_word_subtypes(all_docs):
    assert not all_docs["reason_subtype"].isin(["Local", "Flow", "Drag"]).any()


def test_new_code_reads_no_test_season_or_results_data():
    """Split discipline: the crawler and the parser are description-only. They must not import models, features, splits or results."""
    from pitlane.data import fia_crawl as crawl_mod
    for mod in (F, crawl_mod):
        src = open(mod.__file__, encoding="utf-8").read()
        assert not re.search(r"pitlane\.(models|features|evaluation|splits)|read_parquet\(.*(processed|results)|leaderboard", src)
