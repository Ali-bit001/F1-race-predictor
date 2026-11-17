# power_units.csv - engine supplier per constructor per season (2014-2026)

Hand table for M14 supplier pooling (PLAN3.md Amendment 3 item 4, section 3, section 4 M14).
Built 2026-11-17. 133 rows, one per (season, constructorId) present in data/processed/entries.parquet;
the join covers all 5503 entry rows with no missing and no extra keys.

## Columns
- season, constructorId: Jolpica ids exactly as in the processed data (for example force_india 2018 is one id
  across both 2018 entries; rb covers RB 2024 and Racing Bulls 2025-26; sauber covers Kick Sauber 2024-25).
- supplier: the company that builds the power unit, not the badge (vocabulary below).
- works: 1 if the supplier is that team's own works partner, else 0 (rule below).
- source: the Wikipedia season article whose Entries table lists the power unit (fetched 2026-11-17, current
  revision, not a permalink).
- notes: the power-unit name as listed, badge mapping, borderline flags, and any change during the season.

## Supplier normalisation (badge -> builder)
| Badge in entry list | supplier | Seasons |
|---|---|---|
| Mercedes, Mercedes-AMG, BWT Mercedes (Racing Point) | mercedes | all |
| Ferrari | ferrari | all |
| Renault, TAG Heuer (Red Bull 2016-18), Toro Rosso (Toro Rosso 2017), Renault E-Tech (Alpine 2021-25) | renault | 2014-2025 |
| Honda, RBPT / Red Bull RBPTH001 (2022), Honda RBPT (2023-25), Honda (Aston Martin 2026) | honda | 2015-2026 |
| Red Bull Ford DM01 | red_bull_powertrains | 2026 |
| Audi | audi | 2026 |

Reasons for the less obvious mappings:
- TAG Heuer 2016-18 and Toro Rosso 2017: the season articles describe these as rebadged Renault units.
- RBPT 2022 and Honda RBPT 2023-25: Honda kept building and supplying the unit; the 2022 article says the
  engines were rebadged Red Bull Powertrains, the 2023 article says Honda returned as the named supplier
  with Honda support to the end of 2025. Mapped to honda so 2021 and 2022-25 pool together.
- Red Bull Ford 2026: a new unit built by Red Bull Powertrains with Ford as partner; it is not a Honda
  continuation, so it gets its own level. Ford is not used as a level (Ford is a partner, not the builder;
  this attribution is from the 2026 season article, not from a technical source).
- Cadillac 2026 runs Ferrari customer units (GM unit planned for 2029, not in scope).

## Works rule
works = 1 when the power-unit maker owns the team (Mercedes, Ferrari, Renault 2016-20, Alpine 2021-25 as a
Renault group brand, Audi 2026, Red Bull 2026 with its own Red Bull Powertrains unit), or when the team is the
maker's sole or designated lead partner under a works deal (McLaren-Honda 2015-17, Toro Rosso-Honda 2018,
Red Bull-Honda 2019-25, Aston Martin-Honda 2026). Everything else is 0.
With this rule each supplier has at most one works team per season; Renault 2014-15 has none.

Borderline decisions (flagged BORDERLINE in notes):
- Red Bull-Renault 2014-15: works = 0. Renault owned no team and Red Bull was its lead partner, so a case
  for 1 exists. Chosen 0 because Renault did not own the team and Red Bull is not described as a works
  team in the sources opened. Unverified: the commercial terms (paid supply or not) were not checked.
- Red Bull-Honda 2019-21 and Red Bull with Honda RBPT 2022-25: works = 1 (lead Honda partner; from 2022
  Red Bull Powertrains, owned by Red Bull, holds the programme). Honda never owned the team.
- Toro Rosso / AlphaTauri / RB on Honda 2019-25 and RB on Red Bull Ford 2026: works = 0 (sister team, second
  customer of the same unit). Toro Rosso 2018 = 1 because it was then the only Honda team and the 2018 article
  calls it full-works Honda.
- McLaren-Honda 2015-17: works = 1 (only Honda team all three years).
- Alpine-Renault 2021-25: works = 1 (same company as the supplier).
- Aston Martin-Mercedes 2021-25 and Racing Point BWT Mercedes 2019-20: works = 0 (customer units).
- McLaren-Mercedes 2014: works = 0 (Mercedes had its own works team by then).
- Aston Martin-Honda 2026: works = 1 (Honda supplies only Aston Martin per the 2026 article).
If M14 is sensitive to the works flag, the BORDERLINE rows are the ones to flip in a robustness check.

## Known before the season
Every supplier value is the season-start entry-list value. No mid-season power-unit change was found in any
of the 13 season articles. One mid-season entry change: Force India 2018 (Sahara Force India rounds 1-12,
Racing Point Force India rounds 13-21), same Mercedes unit, one constructorId in the data.
Customer units of an older specification are noted but not modelled (Manor 2015 Ferrari 059/3, Toro Rosso
2016 Ferrari 060, Sauber 2017 Ferrari 061). Haas 2024 is listed with Ferrari 066/10 while other Ferrari
teams show 066/12; this label was not cross-checked and does not affect the supplier.

## Second-source check
Every row was compared with the team name on the formula1.com constructors standings page for that season
(https://www.formula1.com/en/results/YEAR/team; for 2025 and 2026 the names omit the engine and the
engine is read from the team link path, e.g. Red-Bull-Racing-Red-Bull-Ford, Aston-Martin-Aramco-Honda).
All 133 rows agree on the badge. Rows singled out as spot checks of the tricky cases:
1. 2014 marussia - Marussia Ferrari
2. 2014 caterham - Caterham Renault
3. 2015 lotus_f1 - Lotus Mercedes
4. 2015 mclaren - McLaren Honda
5. 2016 red_bull - Red Bull Racing TAG Heuer
6. 2016 toro_rosso - Toro Rosso Ferrari
7. 2016 manor - MRT Mercedes
8. 2017 toro_rosso - Toro Rosso (no engine name; consistent with the Renault rebadge in the season article)
9. 2018 toro_rosso - Scuderia Toro Rosso Honda
10. 2018 mclaren - McLaren Renault
11. 2019 racing_point - Racing Point BWT Mercedes
12. 2022 red_bull and alphatauri - Red Bull Racing RBPT, AlphaTauri RBPT
13. 2024 rb - RB Honda RBPT
14. 2026 alpine - Alpine-Mercedes; 2026 cadillac - Cadillac-Ferrari; 2026 audi - Audi; 2026 red_bull - Red-Bull-Racing-Red-Bull-Ford
The second source confirms badges only. It does not confirm the builder behind a badge (TAG Heuer, Toro
Rosso 2017, RBPT 2022) or the works flag; those rest on the season articles alone.

## Not verified
- The works flag is a judgement under the rule above; no source states works or customer status in a
  uniform way. The BORDERLINE rows are unverified judgements.
- The builder behind the Toro Rosso 2017 badge, the TAG Heuer badge and the 2022 RBPT badge rests on one
  source (Wikipedia season articles), not on an FIA document.
- FIA entry lists were not opened.
- Wikipedia text is CC BY-SA; the table records facts only, in our own words.
