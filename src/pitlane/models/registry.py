"""Build a model from a plain config dict, so configs are picklable (parallel tuning) and saved with runs."""
from __future__ import annotations

import pandas as pd

from pitlane.models.baselines import Heuristic
from pitlane.models.gbm import GBMModel, RFModel
from pitlane.models.latent import LatentModel


def cell_flags(cell: str) -> tuple[bool, bool, bool]:
    """'110' -> (dynamic, decomp, ranking)."""
    return cell[0] == "1", cell[1] == "1", cell[2] == "1"


def build_model(cfg: dict, entries: pd.DataFrame):
    kind = cfg["kind"]
    p = {k: v for k, v in cfg.items() if k not in ("kind", "cell")}
    if kind == "heuristic":
        return Heuristic(**p)
    if kind == "rf":
        return RFModel(**p)
    dyn, dec, rank = cell_flags(cfg["cell"])
    if kind == "latent":
        return LatentModel(entries, dynamic=dyn, decomp=dec, ranking=rank, **p)
    if kind == "gbm":
        return GBMModel(dynamic=dyn, decomp=dec, ranking=rank, **p)
    raise ValueError(kind)
