"""PLAN3 R15 helpers: data-integrity check against the freeze, Holm step-down with bootstrap intervals, minimum-n gating,
write-once verdict files. No experiment logic here; `methodologies/P3_final/score.py` uses these.

Integrity (HOLDOUT_PREREG section 6): the file hashes in FREEZE.md only match while no row is appended. After the first holdout
race is appended, the check is made on row hashes of the rows that existed at the freeze:
- parquet tables: one 64-bit hash per row over the columns sorted by name, rows of race_id <= last (or season/round <= last for
  laps and pits); a digest per group (race, or season+round) and one per file, both independent of row order;
- external CSV tables: every row at the freeze must still be the same row at the same position (appended rows are allowed).
"""
from __future__ import annotations

import hashlib
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

PROCEDURE = (
    "Parquet tables: restrict to rows with race_id <= last_race_id (tables with a race_id column) or (season, round) <= "
    "(last_season, last_round) (laps, pits). Columns sorted by name (the columns listed here must all still exist; extra "
    "columns are ignored). Row hash = pandas.util.hash_pandas_object(df[columns], index=False) (uint64). Group digest = sha256 of the "
    "sorted uint64 row hashes of the group (little-endian bytes); groups = race_id, or season+round. File digest = same over all "
    "restricted rows. CSV tables: read with dtype=str, keep_default_na=False; row hash = sha256 of the cell values joined by U+001F; "
    "the first n_rows rows must be unchanged and in the same order (rows may be appended). Generated once, from files whose raw "
    "SHA-256 equals the FREEZE.md value (checked at generation). The check passes only if every group digest, the file digest and "
    "every frozen CSV row match."
)


class DataIntegrityError(RuntimeError):
    pass


# Pre-registered constants (HOLDOUT_PREREG sections 4 and 6, FREEZE.md). Not read from any config file.
ALPHA = 0.05
LEVELS = (20, 40)
N_BOOT = 10_000
SEED = 0
PARQUET_FILES = ["data/processed/entries2.parquet", "data/processed/entries.parquet", "data/interim/plan3/laps.parquet",
                 "data/interim/plan3/pits.parquet"]
CSV_FILES = ["data/external/lineage.csv", "data/external/circuits.csv", "data/external/power_units.csv", "data/external/scheduled_laps.csv"]


# ------------------------------------------------------------------------------------ row hashes
def sha256_bytes(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_data_hashes(freeze_md: Path | str) -> dict[str, str]:
    """`path -> sha256` for the data files listed in FREEZE.md (lines of the form `data/...` `<64 hex>`)."""
    txt = Path(freeze_md).read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in re.finditer(r"`(data/[^`]+)`\s+`([0-9a-f]{64})`", txt)}


def restrict(df: pd.DataFrame, last: dict) -> pd.DataFrame:
    """Rows that existed at the freeze."""
    if "race_id" in df.columns:
        return df[df["race_id"] <= int(last["race_id"])]
    s, r = df["season"], df["round"]
    return df[(s < int(last["season"])) | ((s == int(last["season"])) & (r <= int(last["round"])))]


def _digest(h: np.ndarray) -> str:
    return hashlib.sha256(np.sort(h.astype("<u8")).tobytes()).hexdigest()


def parquet_digests(df: pd.DataFrame, columns: list[str], group_cols: list[str]) -> dict:
    h = pd.util.hash_pandas_object(df[columns], index=False).to_numpy()
    groups = {}
    for key, ix in df.groupby(group_cols, sort=True).indices.items():
        k = "-".join(str(int(x)) for x in (key if isinstance(key, tuple) else (key,)))
        groups[k] = _digest(h[ix])
    return dict(n_rows=int(len(df)), digest=_digest(h), groups=groups)


def csv_row_hashes(path: Path | str) -> tuple[list[str], list[str]]:
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = ["\x1f".join(r) for r in d.itertuples(index=False, name=None)]
    return list(d.columns), [hashlib.sha256(r.encode("utf-8")).hexdigest() for r in rows]


# ------------------------------------------------------------------------------------ spec
def group_cols_for(df: pd.DataFrame) -> list[str]:
    return ["race_id"] if "race_id" in df.columns else ["season", "round"]


def build_spec(root: Path, parquet_files: list[str], csv_files: list[str], last: dict, versions: dict | None = None) -> dict:
    spec: dict = dict(procedure=PROCEDURE, last=dict(last), versions=versions or {}, parquet={}, csv={})
    for rel in parquet_files:
        df = restrict(pd.read_parquet(root / rel), last)
        cols = sorted(df.columns)
        spec["parquet"][rel] = dict(columns=cols, group_cols=group_cols_for(df), **parquet_digests(df, cols, group_cols_for(df)))
    for rel in csv_files:
        cols, hs = csv_row_hashes(root / rel)
        spec["csv"][rel] = dict(columns=cols, n_rows=len(hs), row_hashes=hs)
    return spec


def check_spec(spec: dict, root: Path) -> list[str]:
    """Problems found (empty list = every frozen row is unchanged)."""
    problems: list[str] = []
    last = spec["last"]
    for rel, s in spec["parquet"].items():
        p = root / rel
        if not p.exists():
            problems.append(f"{rel}: file missing")
            continue
        df = restrict(pd.read_parquet(p), last)
        missing = [c for c in s["columns"] if c not in df.columns]
        if missing:
            problems.append(f"{rel}: columns missing {missing}")
            continue
        now = parquet_digests(df, s["columns"], s["group_cols"])
        bad = sorted(k for k in set(now["groups"]) | set(s["groups"]) if now["groups"].get(k) != s["groups"].get(k))
        if bad:
            problems.append(f"{rel}: rows changed or missing in groups {bad[:12]}{' ...' if len(bad) > 12 else ''} "
                            f"({len(bad)} of {len(s['groups'])} groups; rows now {now['n_rows']}, at freeze {s['n_rows']})")
        elif now["digest"] != s["digest"] or now["n_rows"] != s["n_rows"]:
            problems.append(f"{rel}: file digest differs although every group matches")
    for rel, s in spec["csv"].items():
        p = root / rel
        if not p.exists():
            problems.append(f"{rel}: file missing")
            continue
        cols, hs = csv_row_hashes(p)
        if cols != s["columns"]:
            problems.append(f"{rel}: columns changed {cols} vs {s['columns']} (every cell of a row is hashed, so a column change alters every row)")
            continue
        n = s["n_rows"]
        if len(hs) < n:
            problems.append(f"{rel}: {n - len(hs)} rows removed (now {len(hs)}, at freeze {n})")
            continue
        diff = [i + 2 for i in range(n) if hs[i] != s["row_hashes"][i]]       # +2: file line number (header = line 1)
        if diff:
            problems.append(f"{rel}: existing rows edited at file lines {diff[:12]}")
    return problems


def assert_unchanged(spec: dict, root: Path) -> None:
    problems = check_spec(spec, root)
    if problems:
        raise DataIntegrityError("rows that existed at the plan-3 freeze have changed; refusing to score:\n  " + "\n  ".join(problems))


def write_spec_once(spec: dict, path: Path) -> None:
    """Exclusive create: an existing file is never rewritten."""
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        json.dump(spec, f, indent=1, sort_keys=True)
        f.write("\n")


# ------------------------------------------------------------------------------------ bootstrap, Holm, gating
def paired_diffs(ref: pd.DataFrame, cand: pd.DataFrame, metric: str, higher_better: bool) -> np.ndarray:
    """Per-race paired differences, positive = candidate better (same construction as `attribution.paired_gain`)."""
    m = ref.merge(cand, on="race_id", suffixes=("_r", "_c"))
    d = (1.0 if higher_better else -1.0) * (m[f"{metric}_c"].to_numpy(float) - m[f"{metric}_r"].to_numpy(float))
    return d[~np.isnan(d)]


def boot_means(d: np.ndarray, metric: str, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """The bootstrap means behind `attribution.paired_gain` (same random stream per metric)."""
    rng = np.random.default_rng([seed, zlib.crc32(metric.encode())])
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    return d[idx].mean(1)


def holm_stepdown(boots: dict[str, np.ndarray], alpha: float = ALPHA) -> pd.DataFrame:
    """Holm step-down over named tests, each given as its bootstrap means of (candidate better = positive) paired gains.

    Main-session decision 2026-11-18 (before any holdout race): the hypotheses are one-sided, "in the stated direction". Tests are ranked by
    the one-sided bootstrap p in the stated direction, p = (1 + #{boot gain <= 0}) / (n_boot + 1); ties are broken by the pre-registered
    order T1, T2, T3 (the name). The test of rank k (1 = smallest p) is confirmed if the two-sided percentile interval at level
    1 - alpha/(m-k+1) has its lower bound above 0, and every earlier test was confirmed; the first failure stops the procedure. A test whose
    gain is in the wrong direction has p near 1, ranks last and cannot block a right-way test.
    """
    m = len(boots)
    rows = []
    for name, b in boots.items():
        rows.append(dict(test=name, gain=float(b.mean()), p_boot=(1.0 + int((b <= 0).sum())) / (len(b) + 1)))
    order = pd.DataFrame(rows).sort_values(["p_boot", "test"], kind="stable").reset_index(drop=True)
    out, alive = [], True
    for k, r in order.iterrows():
        a_k = alpha / (m - k)
        lo, hi = np.percentile(boots[r["test"]], [100 * a_k / 2, 100 * (1 - a_k / 2)])
        ok = bool(alive and lo > 0)
        alive = ok
        out.append(dict(test=r["test"], rank=int(k + 1), alpha_k=a_k, p_boot=r["p_boot"], lo_holm=float(lo), hi_holm=float(hi), confirmed=ok))
    return pd.DataFrame(out)


NO_VERDICT = "no verdict (n < 20)"


def verdict_state(n: int, min_n: int = 20) -> str:
    return NO_VERDICT if n < min_n else f"verdict allowed (n = {n} >= {min_n})"


def read_levels(n: int, levels: tuple[int, ...] = LEVELS) -> list[int]:
    """Read-outs that are due at n races: the verdict is read at the first n >= 20 and again at the first n >= 40."""
    return [t for t in levels if n >= t]


def write_verdict_once(path: Path, payload: dict) -> bool:
    """Exclusive create. Returns True if written, False if a verdict is already there (the first one is never overwritten)."""
    try:
        with open(path, "x", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=1, sort_keys=True, default=str)
            f.write("\n")
        return True
    except FileExistsError:
        return False


# ------------------------------------------------------------------------------------ pre-registration hash, holdout row digests
def prereg_hash(freeze_md: Path | str) -> str:
    """The SHA-256 of HOLDOUT_PREREG.md recorded in FREEZE.md."""
    m = re.search(r"HOLDOUT_PREREG\.md`,\s*SHA-256 `([0-9a-f]{64})`", Path(freeze_md).read_text(encoding="utf-8"))
    if not m:
        raise ValueError("FREEZE.md has no HOLDOUT_PREREG.md SHA-256 line")
    return m.group(1)


def verify_prereg(prereg: Path | str, freeze_md: Path | str) -> str:
    """Raise if HOLDOUT_PREREG.md differs from the freeze (hash of the file with CRLF turned into LF, as for frozen_config.yaml)."""
    got = hashlib.sha256(Path(prereg).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()
    want = prereg_hash(freeze_md)
    if got != want:
        raise DataIntegrityError(f"HOLDOUT_PREREG.md was edited after the freeze: sha256 {got} != recorded {want}")
    return got


def holdout_digests(root: Path, race_keys: dict[int, tuple[int, int]], columns: dict[str, list[str]] | None = None) -> dict:
    """Row digests of the rows of the given holdout races (race_id -> (season, round)): entries2 and entries per race_id, laps and pits per
    season-round. All columns of the file at the time of writing are hashed (column list stored)."""
    out: dict = {}
    for rel in PARQUET_FILES:
        df = pd.read_parquet(root / rel)
        if "race_id" in df.columns:
            sub = df[df["race_id"].isin(list(race_keys))]
        else:
            want = set(race_keys.values())
            sub = df[[(int(a), int(b)) in want for a, b in zip(df["season"], df["round"])]]
        cols = list(columns[rel]) if columns else sorted(df.columns)
        out[rel] = dict(columns=cols, group_cols=group_cols_for(df), **parquet_digests(sub, cols, group_cols_for(df)))
    return out


def check_holdout_digests(stored: dict, root: Path, race_keys: dict[int, tuple[int, int]]) -> list[str]:
    problems = []
    now = holdout_digests(root, race_keys, {rel: s_["columns"] for rel, s_ in stored.items()})
    for rel, s in stored.items():
        n = now[rel]
        if n["groups"] != s["groups"] or n["n_rows"] != s["n_rows"]:
            bad = sorted(k for k in set(n["groups"]) | set(s["groups"]) if n["groups"].get(k) != s["groups"].get(k))
            problems.append(f"{rel}: holdout rows changed in groups {bad[:12]}")
    return problems
