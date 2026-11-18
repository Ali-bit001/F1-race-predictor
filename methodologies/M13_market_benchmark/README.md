# M13 — Market benchmark (dropped: no clearly permitted odds source)

**Status:** dropped on 2026-11-18 under the PLAN3 risk row "Odds source terms of use: checked before any collection;
M13 is dropped if not permitted". Decision by the user after the source check below. Q7 ("How do we compare with the
betting market?") is recorded as **not tested**. No script was built and no price was collected or stored.

## Source check (pitlane-researcher, 2026-11-18; own words; V = opened, U = not verified)
- **Polymarket** (V): a public read API with no key has per-race F1 winner and podium markets and hourly price history
  for closed markets (about 43 winner events 2024-26, 13 podium events from late 2025). robots.txt allows everything.
  But the Terms of Service (§4.2, prohibited conduct) forbid automated tools that scrape or extract data from the
  site "or any other Interface", and limit using its data "directly or through an API" to professional capital-markets
  clients or data distributors unless agreed in writing. A non-commercial research script is not in that group.
  Judged ambiguous, so not "permitted". This also contradicts PLAN3 §3's "no free historical archive" for this source.
- **Betfair Exchange API** (partly V): keys are tied to a betting account and meant for the holder's own betting; the
  delayed key is for development. Not a research data source. Fee and storage terms U.
- **Kalshi** (U): has F1 podium / race-winner contracts (contract rules PDF opened), but the API and terms could not be reached.
- **The Odds API** (V for terms): terms allow storing data, but no Formula 1 is listed. Not usable.
- **Bookmakers / odds comparison sites** (U): terms unreadable from here, sites block automated access; excluded.

## What would revive it
Written permission from Polymarket (or a reachable, permitting Kalshi API). Any snapshot must be taken strictly before
lights-out; closed-market history ends at 0.99 after the race, so a late timestamp leaks the result.
