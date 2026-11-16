"""R0: download race laps, pit stops and stints (PLAN3.md section 3). Resumable; re-run to continue.

  python scripts/ingest_plan3.py --openf1          # OpenF1 laps/stints/drivers/pit 2023-2026
  python scripts/ingest_plan3.py --pits            # Jolpica pit stops 2014-2026
  python scripts/ingest_plan3.py --laps            # Jolpica laps 2022 down to 2014 (hours; paced)
  python scripts/ingest_plan3.py --laps-extra      # optional: Jolpica laps 2024-2026 (cross-check)
  python scripts/ingest_plan3.py --tables          # rebuild data/interim/plan3/*.parquet from what exists
  python scripts/ingest_plan3.py --check           # files per season vs race counts, failures
No flag selects --openf1 --pits --laps in that order.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pitlane.data import ingest3 as I  # noqa: E402


def check(c: dict) -> int:
    """Files per season against the number of races that have been run. Returns the number of problems."""
    spec = [("jolpica_laps", I.laps_path, c["jolpica"]["laps_seasons"] + c["jolpica"]["reuse_plan2_laps_seasons"]),
            ("jolpica_pits", I.pits_path, c["jolpica"]["pit_seasons"])]
    spec += [(f"openf1_{ep}", (lambda s, r, ep=ep: I.openf1_path(ep, s, r)), c["openf1"]["seasons"])
             for ep in c["openf1"]["endpoints"]]
    empty = I._load_fail()["empty"]
    problems = 0
    print(f"{'endpoint':<16}{'season':>7}{'races':>7}{'files':>7}{'empty':>7}  status")
    for name, pathfn, seasons in spec:
        for s in sorted(seasons):
            races = I.run_races(s, c)
            files, emp = 0, 0
            for r in races:
                rnd = int(r["round"])
                p = pathfn(s, rnd)
                if not p.exists() and name == "jolpica_laps" and s in c["jolpica"]["reuse_plan2_laps_seasons"]:
                    p = I.RAW2_LAPS / f"laps_{s}_{rnd}.json"
                if p.exists():
                    files += 1
                    emp += f"{name}/{s}/{rnd}" in empty
            status = "ok" if files == len(races) else f"MISSING {len(races) - files}"
            problems += files != len(races)
            print(f"{name:<16}{s:>7}{len(races):>7}{files:>7}{emp:>7}  {status}")
    d = I._load_fail()
    print(f"\nfailed (will be retried on re-run): {len(d['failed'])}")
    for k, v in sorted(d["failed"].items()):
        print(f"  {k}: {v.get('reason')} ({v['time']})")
    print(f"empty answers cached for races that were run: {len(d['empty'])}")
    for k, v in sorted(d["empty"].items()):
        print(f"  {k}")
    return problems + len(d["failed"])


def main() -> int:
    ap = argparse.ArgumentParser()
    for f in ("openf1", "pits", "laps", "laps-extra", "tables", "check"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args()
    c = I.cfg()
    I.setup_logging()
    if a.check:
        return 1 if check(c) else 0
    any_dl = a.openf1 or a.pits or a.laps or a.laps_extra
    cl = I.Client(c)
    if a.openf1 or not (any_dl or a.tables):
        I.log.info("openf1: %s", I.ingest_openf1(cl, c))
    if a.pits or not (any_dl or a.tables):
        I.log.info("pit stops: %s", I.ingest_jolpica_pits(cl, c))
    if a.tables or not (a.laps or a.laps_extra):
        t = I.load_tables(c)
        I.log.info("tables written: laps=%d pits=%d stints=%d", len(t["laps"]), len(t["pits"]), len(t["stints"]))
    if a.laps or not (any_dl or a.tables):
        I.log.info("jolpica laps: %s", I.ingest_jolpica_laps(cl, c))
    if a.laps_extra:
        I.log.info("jolpica laps extra: %s", I.ingest_jolpica_laps(cl, c, c["jolpica"]["extra_laps_seasons"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
