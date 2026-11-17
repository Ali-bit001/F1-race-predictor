"""PLAN3 R9 part 1, parser revision 3: declared-component count, free-text subtype flag, missing entry list, event map and UTC.

Synthetic cases use the helpers of test_plan3_m10.py (made-up text only). Real-file cases skip when the local FIA PDFs or the
races table are absent. No FIA text is stored in this file.
"""
from __future__ import annotations

import copy
import re

import pandas as pd
import pytest

from pitlane.data import fia_docs as F
from test_plan3_m10 import CFG, FER, HAVE_ALL, MCL, R_WING, comp, cover, parse, table, team_page

WIL = "Williams Racing"
NOUP = "No updates submitted for this event. \n"


def frame(doc) -> pd.DataFrame:
    df = comp(doc)
    df["doc_file"] = "d.pdf"
    df["event_id"] = 1
    return df


def decl(df: pd.DataFrame) -> pd.DataFrame:
    return F.declared_counts(df).set_index("team_raw")


# --- declared-component count (item 3) -------------------------------------------------------------------------------

def test_declared_count_counts_ok_group_missing_and_reason_missing_rows_not_the_component_column():
    body = R_WING.format(n=1) + "2 Floor body Local load Subtle \nText. \n3 Beam Wing \n4 Front Wing Altered made-up thing along the full length now \nmore. \n"
    df = frame(parse(cover(), team_page(FER, table(body))))
    assert set(df["parse_status"]) == {"ok", "group_missing", "reason_missing"}
    d = decl(df).loc[FER]
    assert d["declared"] == 4 and d["how"] == "counted"
    assert (d["n_ok"], d["n_group_missing"], d["n_reason_missing"], d["n_unparsed"]) == (1, 1, 2, 0)
    assert df["component"].notna().sum() == 2        # the old summarise() count: it missed the two reason_missing rows


def test_declared_count_is_zero_for_no_updates_and_missing_for_empty_page_and_absent_team():
    doc = parse(cover(2024), team_page(FER, table(R_WING.format(n=1))), team_page(MCL, NOUP), team_page(WIL, ""))
    df = frame(doc)
    d = decl(df)
    assert d.loc[FER, "declared"] == 1
    assert d.loc[MCL, "declared"] == 0 and d.loc[MCL, "how"] == "no_updates"
    assert pd.isna(d.loc[WIL, "declared"]) and d.loc[WIL, "how"] == "missing_page_text"
    miss = d[d["how"] == "missing_team"]
    assert len(miss) == 7 and miss["declared"].isna().all()      # 10 teams expected in 2024: 3 present, 7 absent
    assert str(F.declared_counts(df)["declared"].dtype) == "Int64"


def test_declared_count_team_level_status_is_missing_not_a_guess():
    df = frame(parse(cover(), team_page("Zzz Unknown Team", table(R_WING.format(n=1)))))
    assert df["parse_status"].str.contains("unmapped_team").all()
    assert pd.isna(F.declared_counts(df)["declared"].iloc[0])


def test_summarise_uses_the_same_rule():
    body = R_WING.format(n=1) + "2 Floor body Local load Subtle \nText. \n3 Beam Wing \n"
    df = frame(parse(cover(), team_page(FER, table(body)), team_page(WIL, "")))
    s = F.summarise(df).set_index("team_raw")
    assert s.loc[FER, "components"] == 3 and pd.isna(s.loc[WIL, "components"]) and s.loc[WIL, "unparsed"] == 1


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_declared_counts_total_and_missing(tmp_path):
    df = F.parse_dir(out=tmp_path / "t.parquet")
    d = F.declared_counts(df)
    assert int(d["declared"].sum()) == 1163 + 14 + 29           # ok + group_missing + reason_missing
    assert int(d["declared"].isna().sum()) == 2 and set(d.loc[d["declared"].isna(), "how"]) == {"missing_page_text"}
    assert (d["n_unparsed"] == 0).all() and d["how"].isin(["counted", "no_updates", "missing_page_text"]).all()


# --- no silent skip of the completeness check (item 4) ---------------------------------------------------------------

def test_season_without_entry_list_raises_unless_opted_out():
    cfg = copy.deepcopy(CFG)
    cfg["parser"]["seasons_without_entry_list"] = []
    pages = [cover(2027), team_page(FER, table(R_WING.format(n=1)))]
    with pytest.raises(ValueError, match="expected_teams"):
        F.parse_pages(pages, cfg, {})
    cfg["parser"]["seasons_without_entry_list"] = [2027]
    assert len(F.parse_pages(pages, cfg, {}).rows) == 1           # explicit opt-out: no completeness check, no marker rows
    cfg["parser"]["expected_teams"][2027] = ["ferrari", "mclaren"]
    cfg["parser"]["seasons_without_entry_list"] = []
    rows = F.parse_pages(pages, cfg, {"ferrari": "ferrari"}).rows
    assert [r["parse_status"] for r in rows].count("team_missing") == 1


def test_known_seasons_have_entry_lists_and_nothing_is_opted_out_in_the_shipped_config():
    c = F.load_cfg()["parser"]
    assert set(c["expected_teams"]) == {2024, 2025, 2026} and c["seasons_without_entry_list"] == []


# --- free-text subtype flag (item 2) ---------------------------------------------------------------------------------

def test_long_unknown_subtype_is_flagged_and_not_stored_as_subtype():
    row = "2 Mirror assembly Performance - Made up words that run on into the geometry text of the row \nText. \n"
    df = comp(parse(cover(), team_page(FER, table(R_WING.format(n=1) + row))))
    r = df.iloc[1]
    assert r["parse_status"] == "ok" and r["component"] == "Mirror assembly" and r["reason_group"] == "Performance"
    assert r["subtype_free_text"] is True or bool(r["subtype_free_text"])
    assert r["reason_subtype"] == "" and r["raw_text"] is None and not r["subtype_known"]


def test_short_unknown_subtype_is_kept_as_printed_and_not_flagged():
    row = "2 Mirror assembly Performance - Rear flow & integration \nText. \n"
    df = comp(parse(cover(), team_page(FER, table(R_WING.format(n=1) + row))))
    r = df.iloc[1]
    assert r["reason_subtype"] == "Rear flow & integration" and not r["subtype_free_text"] and not r["subtype_known"]
    assert not comp(parse(cover(), team_page(FER, table(R_WING.format(n=1)))))["subtype_free_text"].any()


def test_free_text_word_limit_is_a_config_knob():
    row = "2 Mirror assembly Performance - Made up words that run on into the geometry text of the row \nText. \n"
    cfg = copy.deepcopy(CFG)
    cfg["parser"]["subtype_free_text_min_words"] = 99
    r = comp(F.parse_pages([cover(), team_page(FER, table(R_WING.format(n=1) + row))], cfg, {})).iloc[1]
    assert not r["subtype_free_text"] and r["reason_subtype"].startswith("Made up words")


@pytest.mark.skipif(not HAVE_ALL, reason="local FIA PDFs not present")
def test_real_exactly_six_rows_are_flagged_as_free_text_subtype(tmp_path):
    df = F.parse_dir(out=tmp_path / "t.parquet")
    f = df[df["subtype_free_text"] == True]    # noqa: E712
    assert sorted(zip(f["doc_file"], f["component"])) == [
        ("2026_60300_8.pdf", "Front Suspension"), ("2026_60300_8.pdf", "Mirror assembly"), ("2026_60300_8.pdf", "Tailpipe bracket"),
        ("2026_60307_9.pdf", "Front Corner"), ("2026_60311_11.pdf", "Front Corner"), ("2026_63507_12.pdf", "Front Corner")]
    assert (f["parse_status"] == "ok").all() and (f["reason_subtype"] == "").all() and f["raw_text"].isna().all()
    left = df[(df["parse_status"] == "ok") & (df["subtype_known"] == False) & (df["subtype_free_text"] == False)]   # noqa: E712
    assert len(left) == 10 and left["reason_subtype"].map(lambda t: len(re.findall(r"[A-Za-z]+", t))).max() <= 5


# --- capital-letter row number (item 1) -----------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="KNOWN LIMIT of round 4: a digit line whose text starts with a capital ('3 Elements are used "
                                       "here') still lets a following 'Cooling' prose line become a row; only lower-case prose is excluded")
def test_capitalised_digit_prose_before_a_group_word_is_still_a_row_known_limit():
    df = comp(parse(cover(), team_page(FER, table(R_WING.format(n=1) + "3 Elements are used here \nImproved Cooling flow through the duct. \n"))))
    assert df["parse_status"].tolist() == ["ok"]


@pytest.mark.parametrize("prose", ["Improved Cooling flow through the duct. \n", "Reliability concerns remain. \n", "Other - similar parts \nmore. \n",
                                   "Cooling \n", "Weight \n"])
def test_lower_case_digit_prose_before_a_group_word_is_not_a_row(prose):
    df = comp(parse(cover(), team_page(FER, table(R_WING.format(n=1) + "3 elements are used here \n" + prose))))
    assert df["parse_status"].tolist() == ["ok"] and df["component"].tolist() == ["Rear Wing"]


# --- event map and UTC (item 8) --------------------------------------------------------------------------------------

def test_published_utc_two_readings():
    s = pd.Series(["07.03.24 12:20", "05.07.24 11:00", "31.03.24 02:30", "27.10.24 02:30", None, "bad"])
    u = F.published_utc(s)
    assert u["published_utc_fixed1"].iloc[0] == pd.Timestamp("2024-03-07 11:20") == u["published_utc_paris"].iloc[0]   # winter: equal
    assert u["published_utc_fixed1"].iloc[1] == pd.Timestamp("2024-07-05 10:00")
    assert u["published_utc_paris"].iloc[1] == pd.Timestamp("2024-07-05 09:00")                                        # summer: 1 h apart
    assert pd.isna(u["published_utc_paris"].iloc[2]) and pd.isna(u["published_utc_paris"].iloc[3])                    # DST gap / overlap
    assert pd.isna(u.iloc[4]).all() and pd.isna(u.iloc[5]).all()


def _tiny(tmp_path, n_races: int):
    man = pd.DataFrame([
        {"season": 2030, "page_order": 1, "event": "Bahrain Tests Season", "event_id": 1, "found": False},
        {"season": 2030, "page_order": 2, "event": "A Grand Prix", "event_id": 2, "found": False},
        {"season": 2030, "page_order": 3, "event": "B Grand Prix", "event_id": 3, "found": True, "published_cet": "15.03.30 10:00",
         "file": "x.pdf"}])
    races = pd.DataFrame([{"season": 2030, "round": k, "date": pd.Timestamp("2030-03-03") + pd.Timedelta(days=7 * k), "raceName": f"R{k}"}
                          for k in range(1, n_races + 1)])
    man.to_csv(tmp_path / "m.csv", index=False)
    races.to_parquet(tmp_path / "r.parquet")
    return tmp_path / "m.csv", tmp_path / "r.parquet"


def test_event_map_skips_test_events_and_uses_page_order(tmp_path):
    m, r = _tiny(tmp_path, 2)
    em = F.event_map(m, r)
    assert em[["event_id", "round", "found"]].values.tolist() == [[2, 1, False], [3, 2, True]]


def test_event_map_raises_if_event_count_and_round_count_differ(tmp_path):
    m, r = _tiny(tmp_path, 3)
    with pytest.raises(ValueError, match="rounds"):
        F.event_map(m, r)


def test_event_map_raises_if_a_document_is_published_after_its_race(tmp_path):
    m, r = _tiny(tmp_path, 2)
    d = pd.read_csv(m)
    d.loc[d["event_id"] == 3, "published_cet"] = "01.06.30 10:00"
    d.to_csv(m, index=False)
    with pytest.raises(ValueError, match="after the race"):
        F.event_map(m, r)


def test_event_map_raises_if_a_document_is_published_long_before_its_mapped_race(tmp_path):
    # a page-order shift keeps the counts equal but maps a document onto a later race: lead of a week or more
    m, r = _tiny(tmp_path, 2)
    d = pd.read_csv(m)
    d.loc[d["event_id"] == 3, "published_cet"] = "08.03.30 10:00"     # 9 days before round 2 (2030-03-17)
    d.to_csv(m, index=False)
    with pytest.raises(ValueError, match="before the mapped race"):
        F.event_map(m, r)


def test_declared_count_is_missing_when_a_team_has_only_unparsed_rows():
    g = pd.DataFrame({"parse_status": ["unparsed", "unparsed"], "no_updates": [False, False]})
    n, how = F._declared_of(g)
    assert pd.isna(n) and how == "has_unparsed"
    g2 = pd.DataFrame({"parse_status": ["ok", "unparsed"], "no_updates": [False, False]})
    assert F._declared_of(g2) == (1, "has_unparsed")


RACES_OK = F.RACES_PATH.exists()


@pytest.mark.skipif(not RACES_OK, reason="races table (data/processed/entries.parquet) not present")
def test_real_all_67_found_documents_map_to_exactly_one_race_and_no_two_to_the_same_race():
    em = F.event_map()
    man = F.load_manifest()
    f = em[em["found"]]
    assert len(em) == 68 and len(f) == 67 == len(man)
    assert f["event_id"].is_unique and not f.duplicated(["season", "round"]).any()
    assert set(f["event_id"]) == set(man["event_id"].astype(int))
    assert sorted(f["doc_file"]) == sorted(man["file"])
    assert f[["season", "round"]].notna().all().all()
    miss = em[~em["found"]]
    assert miss[["season", "round"]].values.tolist() == [[2024, 1]]                     # 2024 Bahrain GP: no document
    # published times rise with the round inside a season (page order and race order agree), and precede the race date
    for _, g in f.groupby("season"):
        assert g.sort_values("round")["published_utc_fixed1"].is_monotonic_increasing
    assert (f["published_utc_fixed1"] < f["race_date"] + pd.Timedelta(days=1)).all()
    assert (f["race_date"] - f["published_utc_fixed1"]).max() < pd.Timedelta(days=12)


@pytest.mark.skipif(not RACES_OK, reason="races table (data/processed/entries.parquet) not present")
def test_real_event_names_differ_only_for_the_three_known_cases():
    em = F.event_map()
    key = lambda s: re.sub(r"[^a-z]", "", re.sub(r"grand prix", "", s.lower()))      # noqa: E731
    diff = em[em["manifest_event"].map(key) != em["race_name"].map(key)]
    assert sorted(zip(diff["season"], diff["round"])) == [(2026, 7), (2026, 16), (2026, 20)]


@pytest.mark.skipif(not RACES_OK, reason="races table (data/processed/entries.parquet) not present")
def test_real_utc_columns_differ_by_one_hour_exactly_in_summer():
    em = F.event_map()
    f = em[em["found"]]
    d = (f["published_utc_fixed1"] - f["published_utc_paris"]).dt.total_seconds() / 3600
    assert set(d.dropna().unique()) == {0.0, 1.0}
    summer = f["published_utc_fixed1"].dt.month.between(4, 9)
    assert (d[summer] == 1.0).all() and (d[f["published_utc_fixed1"].dt.month.isin([1, 2, 11, 12])] == 0.0).all()
