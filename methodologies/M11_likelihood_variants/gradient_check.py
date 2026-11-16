"""Finite-difference check of the analytic gradient of the M11 objectives on the real training window.

    python methodologies/M11_likelihood_variants/gradient_check.py

Training window = every race before the first 2021 race (2808 rows, 254 parameters).
40 random coordinates, random parameter vector (seed 0), central differences with step 1e-6, relative error
|fd - g| / max(|g|, 1e-3 * max|g|). Writes results/gradient_check.csv. Fixed seed; nothing is scored.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

from pitlane import pipeline as P
from pitlane.config import ROOT, load_yaml
from pitlane.models.latent import LatentModel

MDIR = Path(__file__).resolve().parent
SW = {"off": {}, "finishers": dict(finishers_only=True), "top10": dict(topk=10), "reverse": dict(reverse=True),
      "finishers+reverse (not a component)": dict(finishers_only=True, reverse=True)}


def main() -> None:
    entries = P.load_entries()
    base = {k: v for k, v in load_yaml(ROOT / "configs/frozen.yaml")["M03"]["111"].items() if k not in ("kind", "cell")}
    first = int(entries.loc[entries["season"] == 2021, "race_id"].min())
    rows = []
    for name, sw in SW.items():
        m = LatentModel(entries, dynamic=True, decomp=True, ranking=True, **base, **sw)
        n = int(m.race_rows[first].min())
        w, pad = m.fit_inputs(n, float(first))
        th = np.random.default_rng(0).normal(0, 0.3, m.theta.size)
        f0, g = m._objective(th, n, w, pad)
        js = np.random.default_rng(1).choice(th.size, 40, replace=False)
        floor = 1e-3 * np.abs(g).max()
        errs = []
        for j in js:
            e = np.zeros_like(th)
            e[j] = 1e-6
            fd = (m._objective(th + e, n, w, pad)[0] - m._objective(th - e, n, w, pad)[0]) / 2e-6
            errs.append(abs(fd - g[j]) / max(abs(g[j]), floor))
        rows.append(dict(variant=name, n_train_rows=n, n_params=th.size, coords_checked=40, max_rel_error=float(max(errs))))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(MDIR / "results" / "gradient_check.csv", index=False)


if __name__ == "__main__":
    main()
