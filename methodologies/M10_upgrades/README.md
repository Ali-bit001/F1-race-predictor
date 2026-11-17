# M10 — Upgrade signal (FIA "Car Presentation Submissions")

**Status:** in progress. R9 part 1: parser and data layer only (third revision). No model rung has been run.
**Plan reference:** PLAN3.md R9 and Amendment 3 items 1-3
**Factors:** car (the signal is the number of components a team declares as updated at an event)

## Hypothesis (copied from PLAN3 Amendment 3 item 1; the amendment rules)
"**M10 (R9) is holdout-only.** The FIA car-presentation documents start at 2024 round 2, so dev has no data. One rung only,
fixed now: total declared-component count per car in the weekend term, at H2 and H0/H1, `lam_x` fixed at the `LatentModel`
default (1.0), no tuning. Category and reason are parsed and reported descriptively, not as rungs. Judged only on the fresh
holdout; the 2024–26 test block is used to check the parser, not for any decision. Expected gain on the holdout: about 0 at H2,
at most 0.01 ρ at H0/H1; failure is an interval including 0."

The PLAN3 R9 table row says the rung "runs at H2 and H0/H1 on dev". That conflicts with the amendment (dev has no documents).
The amendment rules, so nothing is run on dev. Nothing here has been fitted or scored.

## Coverage and what is missing (read before using the data)
- 67 of the 68 Grands Prix of 2024-2026 have a document. The manifest (`documents.csv`) has 70 rows: 68 Grands Prix and 2
  test events ("Tests" events are not races). The missing one is the 2024 Bahrain Grand Prix (round 1).
- The first document is 2024 round 2. A car's 2024 round-1 count is MISSING, not zero. Likewise every 2022-2023 race.
  2022-23 have no documents on the FIA site as far as checked: 5 events were looked at, the rest is unverified.
- Dev (2021-2023) therefore has no M10 data, so `lam_x` cannot be tuned and nothing can be judged on dev. The fresh holdout will
  be small, so an interval that includes 0 is the likely outcome whatever the true effect.
- **PLAN3 §M10 says the signal "covers 2022 onward". That is wrong by the evidence above** (it starts at 2024 round 2). Raised
  here; PLAN3 is not edited by this package.

## Method (parsing)
Code: `src/pitlane/data/fia_docs.py` (parser, counts, event map), `src/pitlane/data/fia_crawl.py` (download, polite, resumable).
Rules and keyword lists: `config.yaml` (every parsing change is commented "R9 parsing fix"). Source list: `documents.csv`
(URL and SHA-256, checked on every run). Run: `python -m pitlane.data.fia_docs` (writes `data/interim/m10_car_presentations.parquet`,
git-ignored, and prints a per-team summary). No FIA file or text is committed.

One PDF = a cover page, then one page per team (a table, or a "no updates" line). A row is anchored on its reason cell
("<group> - <subtype>", or Reliability / Weight / Cost / Cooling with no subtype). Its component is the text from the nearest
row-number token up to the anchor. The printed row number is not used for counting, except by the "chain" check below.

### What the parser does beyond the basics (all rules fitted on these 67 documents)
1. Team name is read next to the page-header line, before or after it (pypdf sometimes reverses the page order).
2. Completeness: each document must hold every team of the season's entry list (`expected_teams`: 10 teams for 2024-25, 11 for
   2026). A missing team gets a `team_missing` row, a team under two names `team_duplicate`, an unknown name `unmapped_team`.
   **A season with no entry list now raises** (`ValueError`), unless it is listed in `seasons_without_entry_list` on purpose.
   There is no 2027 list yet.
3. Row-number chain: per team the printed numbers 1, 2, 3, ... are followed. A printed number with no recognised reason cell
   becomes a flagged row (`parse_ok = False`, `parse_status` reason_missing / group_missing, text in `raw_text`).
   `chain_max_skip: 3` means a number at the start of a line may be at most 3 above the last one, i.e. at most 2 numbers skipped.
4. A row number must be followed by a capital letter: a line such as "3 elements are used here" is description text, not a row.
   This fixed two false rows the tester found (a following line with "Cooling", "Reliability" or "Other -"). Residual limit:
   a description line that starts with a digit AND a capital ("3 Elements ...") followed by such a line still makes a row
   (strict xfail test). No real row changed with this fix.
5. A reason group printed without its dash, wrapped subtypes ("Local" / "Load"), reason group case, combined components
   (`category_secondary`), shared reason cells (one flagged row per component), blank pages (`page_text_empty`).
6. **Free-text subtype flag (new column `subtype_free_text`).** Six `ok` rows (2026_60300_8 Williams x2, Audi, and one each of
   2026_60307_9, 2026_60311_11 Cadillac and 2026_63507_12 Alpine) had the subtype cell and the start of the geometry text on one
   line, so `reason_subtype` held description text. For an unknown subtype of 6 or more words (letters only) `reason_subtype` is
   now empty and `subtype_free_text = True` (the text is not kept: parsed rows carry no `raw_text`; read it again from the PDF); `parse_status` stays `ok` (component and group are right).
   The cut cannot be done by position, so it is a word-count rule fitted on these rows. Ten `ok` rows still have an unknown but
   short subtype written by the team (e.g. a 4-word phrase on one line of its own), kept as printed with `subtype_known = False`;
   two of them are the dash-less "Circuit specific" rows whose subtype is empty by design.

### Counting rule: declared components per team per event
`fia_docs.declared_counts(df)` (and `summarise`, which now uses the same rule):
- count rows with `parse_status` in {ok, group_missing, reason_missing}. A reason_missing row has no component text but is one
  printed row, so counting `component.notna()` (the old rule) undercounts it;
- a team with "no updates" has 0;
- `page_text_empty` (image-only page, no OCR) and `team_missing` give MISSING, not 0; so does a team-level status;
- `unparsed` rows are not counted and are reported in `n_unparsed` (none in the 67 files).
Result on the 67 documents: 1206 declared components (1163 ok + 14 group_missing + 29 reason_missing) in 690 team-documents:
414 counted, 274 no updates (0), 2 missing (the two Williams pages with no text: 2024_50084_7 and 2026_60295_9).

### Event map and publication time
`fia_docs.event_map()` gives `event_id -> (season, round)` for joins. Names differ between the FIA list and the race table
("Barcelona-Catalunya" / "Barcelona", "Bahrain Grand Prix" / "Bahrain Grand Prix in Malaysia", "Sao Paulo" / "Brazilian"; the
event key for Sao Paulo is "sopaulo"), so the join uses page order: the k-th race event of a season (oldest first, "Tests" events
removed) is round k of `data/processed/entries.parquet`. It raises if the counts differ, if a document is dated more than a day after its race date, or if it is dated more than `event_lead_max_days` (5) before it (observed lead 1.0-2.0 days).
Tests: all 67 found documents map to exactly one race and no two documents to the same race; names differ only for 2026 rounds
7, 16, 20; publication times rise with the round.
Two UTC columns are given because the FIA page says "CET" and the reading is UNVERIFIED: `published_utc_fixed1` (time minus 1 h)
and `published_utc_paris` (Europe/Paris wall clock converted, i.e. CEST in summer). They differ by 1 hour from late March to
late October and agree otherwise.

### Which documents a prediction may use (per horizon)
- H0 (before the weekend): only documents of EARLIER races.
- H1: additionally the race's own document, only if it was published before the H1 cutoff. Checked for 3 documents only; the
  others are unverified. The CET / CEST question moves a time by at most 1 hour.
- The crawler now asks again, on every run, for events recorded as "not found" (a document may appear later); events with a
  stored document are never asked again.

## Numbers on the 67 documents (regenerated parquet)
| | first revision | second revision | third revision |
|---|---|---|---|
| table rows | 1453 | 1482 | 1482 |
| rows with `component` set | 1156 | 1177 | 1177 |
| `ok` rows | n/a | 1163 | 1163 (6 of them `subtype_free_text`) |
| flagged rows (`parse_ok == False`) | 23 | 45 (29 reason_missing, 14 group_missing, 2 page_text_empty) | 45 (same) |
| `subtype_known == False` among `ok` rows | 14 | 16 | 16 |
Third revision against the second: the only changed rows are the 6 free-text subtype rows (`reason_subtype` emptied, new column). Category, group, status and row count are unchanged.

"Other" category share (denominator: `ok` rows, 411 / 318 / 434 for 2024 / 2025 / 2026): 6.8% / 2.2% / 11.1%.

### Share of component units not fully parsed, per team-season
Units = declared components (ok + group_missing + reason_missing). Flagged = group_missing + reason_missing (component, group
or both missing); the two image-only pages are missing, not counted as units.
| team, season | flagged / units |
|---|---|
| Ferrari 2026 | 15 / 52 (28.8%) |
| Audi 2026 | 8 / 43 (18.6%) |
| Red Bull 2026 | 8 / 58 (13.8%) |
| Ferrari 2025 | 3 / 33 (9.1%) |
| Red Bull 2025 | 4 / 51 (7.8%) |
| Mercedes 2026 | 2 / 38 (5.3%) |
| all others | at most 1 / 41 (2.4%) |
By season: 2024 2 / 413, 2025 7 / 325, 2026 34 / 468 flagged units (these become 3 / 413 and 35 / 468 if each image-only
Williams page is counted as one lost unit). The loss is concentrated in a few team-seasons. The declared count itself includes the
flagged rows, so it is usable for them; category and reason shares are not reliable for Ferrari, Audi and Red Bull 2026.

## Parser quality: what was measured and what was not
Measured by the tester on the regenerated table:
- Whole-table scans found 0 errors in the `ok` rows. Precision: for every `ok` row, component, group and subtype appear in
  that order in the page text. Recall: an independent regex over all pages found nothing that is not already an `ok` row or one of
  3 known flagged rows. A chain scan (tester's count: 416 team blocks across pages; the table has 414 counted
  team-documents plus 274 no-update and 2 missing) found no team with fewer rows than its printed chain.
- A random sample of 20 `ok` rows was read by hand: 20 of 20 correct. The 95% upper bound on the error rate from 20 of 20 is
  about 14%, so the sample alone says little.
Not measured: the residual silent error rate (rows that are wrong or missing and not flagged). No percentage is claimed.
Known blind spots:
- 6 printed numbers are absent from the page text and unflagged: 2025_55037_7 Red Bull (numbers 4, 5, 11, 12, 14) and
  2026_60295_9 Aston Martin (number 6). They may be unused numbers or rows that exist only as images; there is no renderer here to check.
- The chain cannot see a lost LAST row of a team, or a row inside a block whose printed numbers are out of order (4 anchored
  rows: 2024_50064_9, 2024_50067_6, 2025_56441_9, 2026_60304_14; each parsed normally).
- Every fix was fitted on these 67 documents, so the quality above is in-sample. Holdout documents are parsed out-of-sample and
  will probably show some new layouts; the parser flags what it cannot place, it does not guarantee it.

### Known limits (not fixed)
- `parse_ok == False` rows (45) have no reason or no group; a model must test `parse_status`, not only `component`.
- `optional` is low-recall: 6 rows are True, while the word "optional" occurs on 17 pages. `optional == False` does NOT mean
  "not optional".
- "Other" and "Cooling" are recorded as printed; they are not confirmed against the form definition.
- The "Weight" and "Cost" no-subtype rules never fire in the 67 files; untested on real text.
- 43 component rows sit in groups that repeat (same document, team and component), i.e. 22 surplus rows (the tester's
  test counts 23 repeated rows on a different key); the ones checked are genuine variants, counted as printed.
- Category keywords are first-match; a free-text component name is "other" or sidepod by keyword only.
- Two Williams pages are image-only (no OCR).

## Leakage notes
Description-only data. A document may be used for a race only if it was published before that race's cutoff (see above).

## How to run
```
python -m pitlane.data.fia_docs
python -m pytest tests/test_plan3_m10.py tests/test_plan3_m10_status.py tests/test_plan3_m10_rev2.py tests/test_plan3_m10_rev3.py tests/test_fia_crawl.py -q
```

## Results
No model run yet.

## What we learned
After a hand check of 8 documents the first parser had errors on 3-5% of rows; the later revisions removed the errors found.
The measured state is in the section above: 0 errors in the whole-table scans of `ok` rows, 20 of 20 in a random sample, and an
unmeasured residual. The data cover only 2024 round 2 to 2026; there is no dev data for M10.

## Open questions / next tries
Model rung (R9 part 2) is not started and can only be judged on the fresh holdout. Whether the FIA "CET" is fixed UTC+1 or the
Paris clock is unverified.
