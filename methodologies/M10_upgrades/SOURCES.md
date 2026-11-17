# M10 sources (checked 2026-11-17 by pitlane-researcher)

All notes are in our own words. No FIA text or data is stored here (Amendment 3 item 3).
Status: V = opened and checked, U = not opened / not verified.

## Listing
- V `https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-<YYYY>-<id>`
  Season page (HTML). Ids seen: 2022-2005, 2023-2042, 2024-2043, 2025-2071, 2026-2072. It lists every
  event; only the latest event is expanded with its documents. Other events are links
  `/decision-document-list/nojs/<event_id>` (a plain GET of the nojs link redirects to the home page).
- V `https://www.fia.com/decision-document-list/ajax/<event_id>` (GET, header X-Requested-With: XMLHttpRequest)
  returns a Drupal AJAX JSON array; the `insert` command holds an HTML fragment with, per document,
  "Doc N - <title>", "Published on DD.MM.YY HH:MM CET" and the PDF href. No clean JSON API seen.
  Event ids used: 2024 Bahrain 50063, 2024 Saudi 50064, 2023 Spanish 44624, 2023 Hungarian 44628,
  2022 Spanish 39242.
- V Document title on the index: "Car Presentation Submissions". Distinct from "Car Display Procedure"
  (2022 Spanish: "Automobile Display Procedure"), which is a procedural note, not a team declaration.
- V PDF href changes: 2024 `/sites/default/files/decision-document/<YYYY> <Event Name> - Car Presentation Submissions.pdf`
  (spaces); 2025 and 2026 `/system/files/decision-document/<yyyy>_<event_slug>_-_car_presentation_submissions.pdf`
  (the 2025/26 file is 404 under `/sites/default/files/`). Take the href from the listing; do not build it.

## Coverage
- V 2024 R1 Bahrain: no Car Presentation document. 2024 R2 Saudi: present (Doc 9).
- V 2022 Spanish, 2022 Abu Dhabi, 2023 Spanish, 2023 Hungarian, 2023 Abu Dhabi: none, under any similar title.
- U The other 2022-23 events were not checked one by one. U Whether the documents exist elsewhere for 2022-23.

## Documents opened (SHA-256 of the file as downloaded)
- V 2024 Saudi Arabian GP, 11 pages, 9a9da0578b88e9e6b0777a6245f07428f472567dbc70fb2f3b802d08a4ad5a0b
- V 2025 Abu Dhabi GP, 11 pages, 786d9b3020d8d2293539fa3583acb5d97614f7811b98745bbd873523f5afeb43 (all teams: no updates)
- V 2026 Sao Paulo GP, 12 pages, eaf9202e01277f4e051bc97165164078a72505f73d9472a71c69b84412809f56
- Layout: one cover page (event, document number, date and local time), then one page per team. Each team
  page is a table, one row per updated part: row number, component (e.g. front wing, rear wing, beam wing,
  floor furniture, front corner, coke/engine cover), primary reason written "Group - subtype" (seen:
  Performance - Flow Conditioning / Local Load; Circuit specific - Drag Range / Cooling Range / Balance Range),
  a short geometry note, and a free-text explanation (2026 form asks for 20-100 words).
  Teams with nothing say so in free text with varying wording ("No updates submitted for this event",
  "No Updates", "No update for this event", "No updates for this event").
- V Text-based, not scanned (pdftotext from Git Bash extracted all rows; used for this check only).
  U Extraction quality with pypdf was not tested (pypdf not installed in this environment).
- Parser hazards seen: team names vary in case and decoration (asterisks, sponsor names); event name on the
  team pages varies; row numbers can be wrong or repeated (2024 Saudi, VCARB page), so count rows by
  component cells, not by the number; table cells wrap across lines; some rows flagged "Optional".

## Timing
- V Index times vs in-document local time: 2024 Saudi 12:20 CET / 14:20 local Thu; 2025 Abu Dhabi
  08:00 CET / 10:53 local Fri; 2026 Sao Paulo 15:32 CET / 10:05 local Fri.
- V In all three, publication came before the first-practice classification was posted (2024 Saudi P1
  classification 15:55 CET; 2025 Abu Dhabi 11:46 CET; 2026 Sao Paulo 17:49 CET) and a day or more before qualifying.
- V 2026 Sao Paulo Timetable V3 (Doc 5) shows first practice on Friday at 12:30 local (read from a
  column-scrambled text extraction, so treat the exact time as approximate).
- U Whether the index "CET" label is fixed UTC+1 or Paris wall-clock time in summer.

## Terms
- V `https://www.fia.com/robots.txt`: `Crawl-delay: 10` for all agents; no Disallow on `/documents`,
  `/decision-document-list/`, `/sites/default/files/` or `/system/files/`.
- V `https://www.fia.com/file/83930/download` (site terms, PDF, (c) 2019): all content reserved; single copies
  for private non-commercial use; short referenced extracts may be quoted; clause 3.9 says no championship
  information or data may be reproduced or used without FIA consent. No clause on automated access found.

## Parser spot-check (R9 part 1, pitlane-coder, 2026-11-17)
Own words, counts and categories only. Parser: `src/pitlane/data/fia_docs.py`, rules in `config.yaml`.
Checked by eye against the pypdf text extraction (default and layout modes) of the three local files; SHA-256 of
all three equals the values listed above. Nothing here was checked against the rendered PDF page images (U).
- 2024 Saudi: 10 team pages (V, read all). Component rows per team in page order: team 1: 3, team 2: 1, team 3: 2,
  team 4: 2, team 5: 2, team 6: none, team 7: 2, team 8: 4, team 9: none, team 10: none; 18 rows in all (V, every
  row read). The three teams without rows carry a "no updates" line (V). The eight rows with a rear or beam wing
  fall in the rear/beam categories (V). The 4-row team prints its numbers as 1, 2, 3, 3; the parser counts 4 by
  component cells (V). Three of its four rows carry the Optional marker (V; the first row does not).
- 2025 Abu Dhabi: 10 team pages, all "no updates" (V, read all). Names carry asterisks on 4 pages (V).
- 2026 Sao Paulo: 11 team pages (V). Only the first team has rows (4: front wing, rear wing, beam wing, floor
  furniture; all four reason groups read as "Performance"; three subtypes Flow Conditioning, one Local Load), the other 10 are "no updates" (V). Four different "no updates" wordings occur in this file (V).
- Reason cells: the en dash between group and subtype decodes as U+FFFD in pypdf; the parser treats it as a dash.
- Page count: pypdf reports 18 pages for the 2024 file (blank pages between teams), against 11 listed above;
  pypdf reports 11 for 2025 (same) and 13 for 2026 (12 listed above).
- Unverified: team name to constructor mapping beyond these three files; the category rules on component
  names that do not occur in these files (they fall to "other"); the Optional rule on wordings not seen here.
