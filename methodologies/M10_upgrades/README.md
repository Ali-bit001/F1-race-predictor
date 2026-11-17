# M10 — Upgrade signal (FIA "Car Presentation Submissions")

**Status:** in progress (R9 part 1: parser only; no model rung yet)
**Plan reference:** PLAN3.md R9 / Amendment 3 items 1-3
**Factors:** n/a (data layer; the model rung comes later)

## Hypothesis
Not defined yet. Part 1 only turns the 67 FIA PDFs (2024-2026) into one table
(`data/interim/m10_car_presentations.parquet`, git-ignored, regenerable).

## Method (parsing)
Code: `src/pitlane/data/fia_docs.py`. Rules and keyword lists: `config.yaml` (every R10 change is commented
"R10 parsing fix"). Source list: `documents.csv` (SHA-256 checked on every run). Run:
`python -m pitlane.data.fia_docs` (writes the parquet and prints a per-team summary).

One PDF = cover page + one page per team (a table, or a "no updates" line). A row is anchored on its reason cell
("<group> - <subtype>", or Reliability / Weight / Cost with no subtype); its component is the text from the nearest
row-number token up to the anchor. The printed row number is never used for counting.

### R10 parsing fixes (after a hand check of 8 documents found an error rate of about 3-5% of rows)
1. **Reversed pypdf page order.** The team name is read from next to the page-header line, before or after it. If the
   table / "No updates" line comes before the header and the name after it, it belongs to that name
   (2025 Japan Mercedes and Haas, 2026 China Williams). Haas's Floor Body row is no longer given to Alpine,
   and Alpine keeps its own no-updates flag.
2. **Completeness check.** `expected_teams` per season in the config (10 for 2024/2025, 11 for 2026). A team absent
   from a document gets a marker row, `parse_status = "team_missing"`; a team seen under two names is marked
   `team_duplicate`; a name not mapped to a constructor is marked `unmapped_team`. Result on the 67 files: no
   marker rows (before the fixes three teams were omitted silently).
3. **Row number mid-line.** A number after sentence-ending text starts a new row (`row_after_sentence_regex`), and a
   number mid-line before an anchor is used when no line starts with one (2026 Singapore Mercedes).
   All six documents that were one row short and flagged are now complete. Rows with a missing printed
   number but a clean "Component Performance -" start are kept.
4. **Reason cell lost.** A short numbered line with no reason cell after it (`cut_row_regex`) is flagged
   (`parse_ok = False`, `parse_status = "reason_missing"`, text in `raw_text`), not dropped. These flags also catch
   real rows whose reason is not written as "<group> - <subtype>" (e.g. "Cooling Range" alone, "Other -"): the
   component is not parsed for them.
5. **Wrapped subtype.** "Local" / "Load", "Flow" / "Conditioning", "Drag" / "Range" are joined when the joined text
   starts with a configured subtype. "Drag Reduction" was added to the list. `subtype_known == False`: 111 -> 14.
6. **Reason group case** is normalised to the configured spelling ("Circuit Specific" -> "Circuit specific").
7. **team_raw** no longer carries header residue ("Car Presentation - Emilia - Romagna", "in Malaysia").
8. **Latent hazards (synthetic cases) fixed:** rows before a footer on a headerless continuation page are kept; an
   unmapped team with only "No updates" is its own team (status `unmapped_team`) instead of being merged into the
   previous team; "2 Weight reduction ..." in a description is not a Weight row; prose that names a team on a page with
   no table and no header is overflow of the previous team, not a new team.
9. **Combined components.** `category` keeps the first-match rule unchanged; the new column `category_secondary` holds
   the second matched category ("Rear Wing / Beam Wing": beam_wing + rear_wing; "Rear Wheel Bodywork":
   rear_corner_suspension + sidepod_engine_cover after the second revision), empty otherwise.

New columns: `category_secondary`, `parse_status` (ok, no_updates, unparsed, reason_missing, group_missing, page_text_empty, team_missing,
team_duplicate, unmapped_team).

### Second revision (R10 parsing fix, after the tester's re-check)
The re-check found a crash and ten printed rows that vanished with `parse_ok = True` on the team's other rows. Changes:
1. **Crash.** A team absent from a document made `parse_pages` raise `TypeError`; the `team_missing` marker row is now built correctly.
2. **No silent row loss.** Per team block the parser now follows the printed row numbers 1, 2, 3, ... (the "chain"). Every
   number in the chain is either an anchored row or a flagged row (`parse_ok = False`, `raw_text` kept). A number at the start
   of a line may skip up to 3 numbers. A number the PDF text never prints cannot be flagged (see limits). All ten rows from
   the re-test are now rows or flags.
3. **Reason groups as printed.** "Other" and "Cooling" are accepted as printed values (config comment: observed in documents,
   not confirmed against the form definition). A group printed without its dash ("Circuit specific Modification ...",
   "Performance Local Load") is accepted; when the dash is missing and free text follows "Circuit specific", the subtype
   is left empty with `subtype_known = False`. A row with a known subtype and NO group ("Front Wing  Flow Conditioning") keeps
   its component and subtype, `reason_group` is empty and `parse_status = "group_missing"` (`parse_ok = False`); the group
   is not guessed from the subtype. "Driver Cooling" (2024_50068_10) is not read as the group "Cooling": that row is flagged.
4. **One flag per component.** Rows that share one printed reason cell (Ferrari 2026, Audi 2026) give one flagged row per
   component (`reason_missing`, text "9 Coke/Engine | Cover"); a flag never stands for several components. The reason cannot be
   assigned to the rows above or below the shared cell, so none is assigned.
5. **Rear Wheel Bodywork** (2026_60312_9) is now `rear_corner_suspension` (the rear-corner rule is tested before the sidepod rule);
   it was the only row whose category changed.
6. **False flag removed.** A prose line that ends in a spaced dash is flagged only if it starts with a row number or has at most
   6 tokens (2024_50070_8 Visa RB was a description line).
7. **Blank pages.** The two Williams pages with no text (2024_50084_7, 2026_60295_9) have `parse_status = "page_text_empty"`. No OCR.

### Numbers (67 documents)
| | first revision | second revision |
|---|---|---|
| table rows | 1453 | 1482 |
| component rows (`component` set) | 1156 | 1177 (1163 `ok` + 14 `group_missing`) |
| flagged rows (`parse_ok == False`) | 23 | 45: 29 `reason_missing`, 14 `group_missing`, 2 `page_text_empty` (no `unparsed` left) |
| `subtype_known == False` among `ok` rows | 14 | 16 |
| "other" category share, 2024 / 2025 / 2026 | 6.8% / 2.2% / 11.1% | 6.8% / 2.2% / 11.1% (unchanged) |

Plain reading: 1163 rows are complete (component, group, subtype). 14 more have a component and subtype but the PDF prints no
group. 29 rows are components whose reason cell is missing or shared, so they carry no reason. The estimated share of
rows that are still wrong without being flagged is below about 0.5% (at most 7 rows of 1482). This is an estimate: all ten re-test
rows, the 14 group-missing rows and the 29 reason-missing rows were read against the page text, the three dash-less documents
explain the only differences from the independent Performance / Circuit-specific count, but the unflagged rows were not hand
checked again. The chain cannot see a lost row in a block whose printed numbers do not run in order; in the 67 files that
affects 4 anchored rows (2024_50064_9, 2024_50067_6, 2025_56441_9, 2026_60304_14), each of which was parsed normally.
Five printed numbers are never in the text of 2025_55037_7 Red Bull (4, 5, 11, 12, 14: the table runs 1, 2, 3, 6, ... 10, 13, 15)
and one in 2026_60295_9 Aston (6). They may be unused numbers or rows lost by pypdf; they are not flagged, because wrong and skipped
numbers elsewhere (2024 Saudi) are normal and the plan counts rows by reason cell, not by number.

Rows that changed against the first revision: the flagged 23 rows became rows or per-component flags (list in the run report),
24 old rows were replaced by 53 new ones (components that were missing, per-component flags, and the 2 blank pages); the
category of one component row changed (Rear Wheel Bodywork).

### First revision numbers (67 documents, superseded)
| | before | after |
|---|---|---|
| table rows | 1435 | 1453 |
| component rows | 1149 | 1156 |
| unparsed entries (`parse_ok == False`) | 15 | 23 (9 `unparsed`, 14 `reason_missing`; none are silent now) |
| `team_missing` markers | n/a (3 teams absent silently) | 0 |
| `subtype_known == False` | 111 | 14 |
| "other" category share, 2024 / 2025 / 2026 | 6.8% / 2.2% / 11.3% | 6.8% / 2.2% / 11.1% |
| documents whose Performance / Circuit-specific cell count differs from the independent pypdf count | 7 | 0 |

### Known limits (not fixed)
- Rows with a shared or missing reason cell (29) have no component; 14 rows have a component and subtype but no group.
  A later model must test `parse_ok` and `parse_status`, not only `component`.
- 2026_60303_14 Red Bull row 1 prints "Local load and flow conditioning" as the reason; only the first subtype is recorded
  (`raw_text` has the full text). Rows 3 and 4 of that table are printed blank and are flagged.
- "Other" and "Cooling" are recorded as printed; they are not confirmed against the form definition. "Local flow alignment" was
  added to the subtype list for the same reason.
- The "Weight" and "Cost" no-subtype rules never fire in the 67 files (only Reliability does); they are untested on real text.
- Repeated identical rows inside one team and document (23 component rows) are genuine option variants (e.g. three
  Beam Wing loads in one document), not double counting. Counted as printed.
- The 2026 subtype "Cooling" alone (one row) and free-text subtypes stay `subtype_known = False`.
- Category keywords are first-match; a free-text component such as "Bodywork" is "other" or sidepod by keyword only.
- `published_cet` is a day-first CET string (`%d.%m.%y %H:%M`), not a UTC timestamp; a later model must parse it and
  treat a document as known only after that time.
- A page that pypdf returns empty (image-only) cannot be read: two Williams pages (`page_text_empty`). No OCR.

## Leakage notes
Description-only data. A document is usable for a race only if `published_cet` is before that race starts.

## How to run
```
python -m pitlane.data.fia_docs
python -m pytest tests/test_plan3_m10.py tests/test_fia_crawl.py -q
```

## Results
No model run yet.

## What we learned
Parser error rate before the fixes was 3-5% of rows by hand check of 8 documents. After the second revision no printed
row number can vanish silently; 45 flagged rows (3.0% of table rows) are listed in the parquet with their raw text, and the
unflagged error rate is estimated below 0.5%.

## Open questions / next tries
Model rung (R9 part 2) is not started.
