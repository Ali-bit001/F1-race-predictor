"""Paths and the locked temporal split definition (PLAN.md section 2.3)."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
RESULTS = ROOT / "results"
METHODS = ROOT / "methodologies"
CONFIGS = ROOT / "configs"

FIRST_SEASON = 2014
LAST_SEASON = 2026

# Seasons (inclusive) per block.
SPLITS = {
    "burn_in": (2014, 2020),
    "dev": (2021, 2023),
    "test_stable": (2024, 2025),
    "test_shift": (2026, 2026),
}
WARMUP_RACES = 15  # races scored (but not counted) before a split, used only for online calibration


def load_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: Path | str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
