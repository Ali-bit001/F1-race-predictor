"""Independent checks of the hand table data/external/power_units.csv (PLAN3 Amendment 3 item 4, M14 supplier pooling).

Sections: coverage against data/processed/entries.parquet, vocabulary and source, internal consistency (supplier changes only in
listed seasons, md notes agree with csv), works flag, and a spot-check against F1 history from the tester's own memory.
Spot-checks are NOT verified against a source (no browsing); they are recalled facts. Nothing here edits the table.
The coverage tests skip when the git-ignored entries parquet is missing.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from pitlane.config import PROCESSED, ROOT

CSV = ROOT / "data/external/power_units.csv"
MD = ROOT / "data/external/power_units.md"
ALLOWED = {"mercedes", "ferrari", "renault", "honda", "red_bull_powertrains", "audi"}
SUPPLIER_SEASONS = {  # first/last season a supplier appears, as the md vocabulary table states
    "renault": (2014, 2025), "honda": (2015, 2026), "red_bull_powertrains": (2026, 2026), "audi": (2026, 2026),
    "mercedes": (2014, 2026), "ferrari": (2014, 2026)}
# seasons in which a constructor's supplier is allowed to differ from its previous season (listed from the notes' "switched" text)
CHANGE_SEASONS = {
    "marussia": set(), "toro_rosso": {2015 + 1, 2017, 2018, 2019}, "williams": set(), "mclaren": {2015, 2018, 2021},
    "red_bull": {2019, 2026}, "lotus_f1": {2015}, "manor": {2016}, "alpine": {2026}, "force_india": set(), "aston_martin": {2026}, "rb": {2026},
}
# Well-known pairs recalled from memory (UNVERIFIED against a source)
SPOT = {
    (2014, "mercedes"): "mercedes", (2014, "williams"): "mercedes", (2014, "force_india"): "mercedes",
    (2014, "mclaren"): "mercedes", (2014, "red_bull"): "renault", (2014, "toro_rosso"): "renault",
    (2014, "lotus_f1"): "renault", (2014, "caterham"): "renault", (2014, "sauber"): "ferrari", (2014, "marussia"): "ferrari",
    (2015, "mclaren"): "honda", (2016, "mclaren"): "honda", (2017, "mclaren"): "honda", (2015, "lotus_f1"): "mercedes",
    (2015, "toro_rosso"): "renault", (2015, "manor"): "ferrari", (2016, "manor"): "mercedes", (2016, "haas"): "ferrari",
    (2016, "toro_rosso"): "ferrari", (2017, "toro_rosso"): "renault", (2016, "renault"): "renault",
    (2015, "red_bull"): "renault", (2016, "red_bull"): "renault", (2017, "red_bull"): "renault", (2018, "red_bull"): "renault",
    (2018, "mclaren"): "renault", (2019, "mclaren"): "renault", (2020, "mclaren"): "renault",
    (2018, "toro_rosso"): "honda", (2019, "toro_rosso"): "honda", (2019, "red_bull"): "honda", (2021, "red_bull"): "honda",
    (2021, "mclaren"): "mercedes", (2025, "mclaren"): "mercedes", (2026, "mclaren"): "mercedes",
    (2018, "sauber"): "ferrari", (2019, "alfa"): "ferrari", (2024, "sauber"): "ferrari", (2025, "sauber"): "ferrari",
    (2016, "sauber"): "ferrari", (2021, "haas"): "ferrari", (2025, "haas"): "ferrari", (2026, "haas"): "ferrari",
    (2019, "racing_point"): "mercedes", (2020, "racing_point"): "mercedes",
    (2021, "aston_martin"): "mercedes", (2025, "aston_martin"): "mercedes", (2026, "aston_martin"): "honda",
    (2021, "alpine"): "renault", (2025, "alpine"): "renault", (2026, "alpine"): "mercedes",
    (2020, "alphatauri"): "honda", (2022, "alphatauri"): "honda", (2024, "rb"): "honda", (2025, "rb"): "honda",
    (2026, "rb"): "red_bull_powertrains", (2026, "red_bull"): "red_bull_powertrains",
    (2026, "audi"): "audi", (2026, "cadillac"): "ferrari", (2026, "williams"): "mercedes", (2022, "red_bull"): "honda",
}


@pytest.fixture(scope="module")
def pu() -> pd.DataFrame:
    return pd.read_csv(CSV, dtype={"constructorId": str, "supplier": str})


@pytest.fixture(scope="module")
def entries_pairs() -> set[tuple[int, str]]:
    f = PROCESSED / "entries.parquet"
    if not f.exists():
        pytest.skip("entries.parquet missing")
    e = pd.read_parquet(f, columns=["season", "constructorId"])
    e = e[(e.season >= 2014) & (e.season <= 2026)]
    return set(map(tuple, e.drop_duplicates().itertuples(index=False, name=None)))


# 1. coverage -------------------------------------------------------------------------------------------------------
def test_columns_and_row_count(pu):
    assert list(pu.columns) == ["season", "constructorId", "supplier", "works", "source", "notes"]
    assert len(pu) == 133


def test_no_duplicate_keys(pu):
    assert not pu.duplicated(["season", "constructorId"]).any()


def test_coverage_exact(pu, entries_pairs):
    have = set(zip(pu.season.astype(int), pu.constructorId))
    assert have - entries_pairs == set(), f"extra rows: {sorted(have - entries_pairs)}"
    assert entries_pairs - have == set(), f"missing rows: {sorted(entries_pairs - have)}"


def test_season_range(pu):
    assert pu.season.min() == 2014 and pu.season.max() == 2026
    assert set(pu.season) == set(range(2014, 2027))


def test_ids_have_no_whitespace_or_case_noise(pu):
    assert (pu.constructorId == pu.constructorId.str.strip().str.lower()).all()
    assert (pu.supplier == pu.supplier.str.strip().str.lower()).all()


# 2. vocabulary, blanks, source ---------------------------------------------------------------------------------------
def test_supplier_in_allowed_set(pu):
    assert pu.supplier.notna().all()
    assert set(pu.supplier) <= ALLOWED, set(pu.supplier) - ALLOWED


def test_no_blank_cells(pu):
    assert not pu[["season", "constructorId", "supplier", "works", "source"]].isna().any().any()
    assert (pu.source.str.strip() != "").all()


def test_source_url_matches_season(pu):
    pat = re.compile(r"^https://en\.wikipedia\.org/wiki/(\d{4})_Formula_One_World_Championship$")
    for r in pu.itertuples():
        m = pat.match(r.source)
        assert m, f"unexpected source {r.source} ({r.season},{r.constructorId})"
        assert int(m.group(1)) == r.season, f"source season mismatch {r.season},{r.constructorId}"


def test_notes_non_empty(pu):
    assert pu.notes.notna().all() and (pu.notes.str.strip() != "").all()


def test_supplier_active_years_match_md_vocabulary(pu):
    for sup, (lo, hi) in SUPPLIER_SEASONS.items():
        s = pu[pu.supplier == sup].season
        assert s.min() >= lo and s.max() <= hi, (sup, s.min(), s.max())
    # renault and honda each appear in every year of their stated range
    assert set(pu[pu.supplier == "renault"].season) == set(range(2014, 2026))
    # md vocabulary table said Honda 2015-2025 (stale); fixed to 2015-2026 to include aston_martin 2026
    assert set(pu[pu.supplier == "honda"].season) == set(range(2015, 2027))


# 3. internal consistency ---------------------------------------------------------------------------------------------
def test_supplier_changes_only_in_listed_seasons(pu):
    bad = []
    for cid, g in pu.sort_values("season").groupby("constructorId"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            if g.supplier[i] != g.supplier[i - 1] and int(g.season[i]) not in CHANGE_SEASONS.get(cid, set()):
                bad.append((cid, int(g.season[i]), g.supplier[i - 1], g.supplier[i]))
    assert not bad, f"unlisted supplier changes: {bad}"


def test_every_change_is_noted_as_switched_in_notes(pu):
    """A supplier change must be flagged in that row's notes (switched / back to / Renault bought / new)."""
    bad = []
    for cid, g in pu.sort_values("season").groupby("constructorId"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            if g.supplier[i] != g.supplier[i - 1] and not re.search(r"switched|back to|formerly", g.notes[i], re.I):
                bad.append((cid, int(g.season[i])))
    # known gaps in the notes (the change is true but the row's notes do not say 'switched'); reported in the hand-back
    known = {("aston_martin", 2026), ("mclaren", 2015), ("rb", 2026), ("red_bull", 2026), ("toro_rosso", 2018)}
    assert set(bad) == known, f"changed set of un-noted supplier changes: {bad}"


def test_consecutive_seasons_for_each_constructor(pu):
    """No constructor has a gap (a gap would hide a possible unlisted change)."""
    for cid, g in pu.groupby("constructorId"):
        s = sorted(g.season)
        if cid == "sauber":  # 2019-2023 is the alfa id (same lineage); check the bridge instead
            assert set(range(s[-1] if False else 2019, 2024)) == set(pu[pu.constructorId == "alfa"].season)
            continue
        assert s == list(range(s[0], s[-1] + 1)), (cid, s)


def test_red_bull_family_labelling_consistent(pu):
    """2022 RBPT rebadge mapped to honda in csv and md; 2026 is the only red_bull_powertrains season."""
    md = MD.read_text(encoding="utf-8")
    assert re.search(r"RBPT.*\|\s*honda", md), "md vocabulary table does not map RBPT to honda"
    for cid in ("red_bull", "alphatauri"):
        r = pu[(pu.season == 2022) & (pu.constructorId == cid)].iloc[0]
        assert r.supplier == "honda"
        assert "RBPT" in r.notes
    rb = pu[pu.supplier == "red_bull_powertrains"]
    assert set(rb.season) == {2026} and set(rb.constructorId) == {"red_bull", "rb"}
    # red_bull and its sister team share a supplier in every season they coexist (2020-2026 alphatauri/rb)
    sister = {2020: "alphatauri", 2021: "alphatauri", 2022: "alphatauri", 2023: "alphatauri", 2024: "rb", 2025: "rb", 2026: "rb",
              2019: "toro_rosso", 2018: "toro_rosso"}
    for s, t in sister.items():
        a = pu[(pu.season == s) & (pu.constructorId == "red_bull")].supplier.iloc[0]
        b = pu[(pu.season == s) & (pu.constructorId == t)].supplier.iloc[0]
        if s == 2018:
            assert (a, b) == ("renault", "honda")  # the only year the sister teams differ
        else:
            assert a == b, (s, a, b)


def test_notes_badge_text_agrees_with_supplier(pu):
    """Where the notes name a power unit family keyword, it must be consistent with the supplier column."""
    kw = {"Mercedes": "mercedes", "Ferrari": "ferrari", "Renault": "renault", "Honda": "honda"}
    bad = []
    for r in pu.itertuples():
        head = r.notes.split(";")[0]
        for k, sup in kw.items():
            # only the first clause names this season's unit; skip clauses that talk about rebadging or a switch
            if k in head and not re.search(r"rebadged|switched|from ", head) and r.supplier != sup:
                if not (r.supplier == "honda" and "RBPT" in head) and not (r.supplier == "renault" and "TAG" in head):
                    bad.append((r.season, r.constructorId, r.supplier, head))
    assert not bad, bad


def test_teammates_share_supplier_by_construction(pu, entries_pairs):
    """One row per constructor-season means all drivers of that constructor share the supplier. Check no constructor has
    two suppliers recorded (duplicates already excluded) and that mid-season changes are absent per md."""
    assert pu.groupby(["season", "constructorId"]).supplier.nunique().max() == 1
    assert "No mid-season power-unit change was found" in MD.read_text(encoding="utf-8")


# 4. works flag -------------------------------------------------------------------------------------------------------
def test_works_is_boolean(pu):
    assert set(pu.works.unique()) <= {0, 1}
    assert pu.works.dtype.kind in "iub"


def test_at_most_one_works_team_per_supplier_season(pu):
    cnt = pu.groupby(["season", "supplier"]).works.sum()
    assert cnt.max() <= 1, cnt[cnt > 1]


def test_works_only_flag_when_owner_or_lead_rule_holds(pu):
    """Maker-owned teams must be works; maker-name teams: mercedes/ferrari own team always works=1 and customers 0."""
    for sup in ("mercedes", "ferrari"):
        own = pu[(pu.constructorId == sup) & (pu.supplier == sup)]
        assert (own.works == 1).all()
        cust = pu[(pu.constructorId != sup) & (pu.supplier == sup)]
        assert (cust.works == 0).all(), cust[cust.works == 1]
    # Renault works team years 2016-2020 (constructor renault) and Alpine 2021-2025
    r = pu[(pu.constructorId.isin(["renault", "alpine"])) & (pu.supplier == "renault")]
    assert (r.works == 1).all()
    # Renault 2014-15 had no works team, per md
    assert pu[(pu.supplier == "renault") & pu.season.isin([2014, 2015])].works.sum() == 0
    # alpine 2026 is a Mercedes customer
    assert pu[(pu.season == 2026) & (pu.constructorId == "alpine")].works.iloc[0] == 0


def test_every_season_supplier_with_a_works_team_is_listed(pu):
    """List the (season, supplier) pairs with no works team (informational; asserts only the md claim for renault 2014-15)."""
    g = pu.groupby(["season", "supplier"]).works.sum()
    none = sorted(g[g == 0].index.tolist())
    # md: 'each supplier has at most one works team per season; Renault 2014-15 has none'. Others without works team are
    # customer-only suppliers; print so the report can list them.
    print("supplier-seasons with no works team:", none)
    assert (2014, "renault") in none and (2015, "renault") in none


def test_borderline_rows_listed_in_notes_match_md(pu):
    md = MD.read_text(encoding="utf-8")
    b = pu[pu.notes.str.contains("BORDERLINE")]
    print("BORDERLINE rows:", [(int(r.season), r.constructorId, int(r.works)) for r in b.itertuples()])
    assert len(b) > 0
    # md-declared borderline sets
    exp = {(2014, "red_bull", 0), (2015, "red_bull", 0)}
    exp |= {(s, "red_bull", 1) for s in (2019, 2020, 2021)}
    exp |= {(s, t, 0) for s, t in [(2019, "toro_rosso"), (2020, "alphatauri"), (2021, "alphatauri"), (2022, "alphatauri"),
                                   (2023, "alphatauri"), (2024, "rb"), (2025, "rb"), (2026, "rb")]}
    got = {(int(r.season), r.constructorId, int(r.works)) for r in b.itertuples()}
    assert exp <= got, f"md-described borderline rows without BORDERLINE flag: {sorted(exp - got)}"
    assert "BORDERLINE" in md


def test_md_works_claims_match_csv(pu):
    def w(s, c):
        return int(pu[(pu.season == s) & (pu.constructorId == c)].works.iloc[0])
    assert all(w(s, "mclaren") == 1 for s in (2015, 2016, 2017))
    assert w(2018, "toro_rosso") == 1
    assert all(w(s, "red_bull") == 1 for s in range(2019, 2026))
    assert w(2026, "aston_martin") == 1 and w(2026, "audi") == 1 and w(2026, "red_bull") == 1
    assert all(w(s, "aston_martin") == 0 for s in range(2021, 2026))
    assert all(w(s, "racing_point") == 0 for s in (2019, 2020))
    assert w(2014, "mclaren") == 0
    assert all(w(s, "alpine") == 1 for s in range(2021, 2026))


# 5. spot-check against recalled history (UNVERIFIED against a source) --------------------------------------------------
@pytest.mark.parametrize("key,sup", sorted(SPOT.items()))
def test_spot_check_recalled_history(pu, key, sup):
    r = pu[(pu.season == key[0]) & (pu.constructorId == key[1])]
    assert len(r) == 1, f"{key} not in table"
    assert r.supplier.iloc[0] == sup


def test_no_value_known_only_after_event(pu):
    """Season-start values: the source notes must not describe a mid-season change for the row's supplier."""
    bad = pu[pu.notes.str.contains(r"mid-season|from round|after round", case=False, regex=True)
             & ~pu.notes.str.contains("entry change", case=False)]
    assert bad.empty, bad[["season", "constructorId", "notes"]]
