"""M10 model rung (PLAN3 R9 part 2, Amendment 3 item 1). Holdout-only.

    python methodologies/M10_upgrades/run.py --dry-run          # coverage on 2024-26, no model, no metric
    python methodologies/M10_upgrades/run.py                    # score every fresh-holdout race (called by R15)
    python methodologies/M10_upgrades/run.py --races 2027:1,2027:2

One rung: `m10_declared_x` added to the LatentModel weekend term (lam_x = 1.0, no tuning) at H2 and at H1, each against its
same-harness reference (H2: frozen latent_111; H1: frozen plan-2 h1_latent_111). H0 has no own-race document: no M10 column, the rung
is the reference, nothing is run. The run REFUSES to score or write a metric for any race at or before config.FRESH_AFTER
(2026 round 20), i.e. for any race of 2021-2026 round 20. The dry run builds features only and prints coverage.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import pandas as pd

from pitlane.config import FRESH_AFTER, ROOT, load_yaml
from pitlane.evaluation.harness import save_run, split_races, walk_forward
from pitlane.models.latent import LatentModel
from pitlane.models.registry import build_model
from pitlane.plan3 import attribution as A
from pitlane.plan3 import m10_feature as M
from pitlane.plan3 import runutil as U

MDIR = Path(__file__).resolve().parent
CONFIG = MDIR / "config.yaml"
VERDICT_METRICS = ["spearman", "mae", "exact", "exact_fin", "spearman_fin", "mae_fin", "logloss_points"]


class FreezeError(RuntimeError):
    """A race at or before the plan-3 freeze date was asked for a metric."""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------ guards
def after_freeze(season: int, rnd: int) -> bool:
    return (int(season), int(rnd)) > tuple(FRESH_AFTER)


def assert_after_freeze(pairs: list[tuple[int, int]]) -> None:
    """Refuse (raise FreezeError) if any (season, round) is not strictly after FRESH_AFTER."""
    bad = [p for p in pairs if not after_freeze(*p)]
    if bad:
        raise FreezeError(f"refusing to output metrics for {len(bad)} race(s) not after the freeze {tuple(FRESH_AFTER)}: "
                          f"{sorted(bad)[:5]}...; M10 is judged on the fresh holdout only")


def resolve_lam_x(model_cfg: dict) -> float:
    """lam_x of the rung. Must equal the LatentModel default unless `unlock_lam_x` is true (a deliberate deviation)."""
    default = float(inspect.signature(LatentModel.__init__).parameters["lam_x"].default)
    lam = float(model_cfg["lam_x"])
    if lam != default and not bool(model_cfg.get("unlock_lam_x", False)):
        raise ValueError(f"lam_x={lam} differs from the LatentModel default {default}; Amendment 3 item 1 fixes it "
                         f"(set model.unlock_lam_x: true to override on purpose)")
    return lam


def parse_races(s: str | None) -> list[tuple[int, int]] | None:
    if not s:
        return None
    return [(int(a), int(b)) for a, b in (t.split(":") for t in s.split(","))]


# ------------------------------------------------------------------------------------ configs
def reference_cfg(model_cfg: dict, horizon: str) -> dict:
    ref = model_cfg["references"][horizon]
    return dict(load_yaml(ROOT / ref["file"])[ref["key"]][ref["cell"]])


def rung_cfg(model_cfg: dict, horizon: str) -> dict:
    """Reference + the M10 column as one more weekend-term column; lam_x fixed (resolve_lam_x)."""
    if horizon == "H0":
        raise ValueError("H0 has no M10 column: the rung equals the reference")
    base = reference_cfg(model_cfg, horizon)
    return dict(base, x_cols=list(base.get("x_cols", [])) + [model_cfg["column"]], lam_x=resolve_lam_x(model_cfg))


# ------------------------------------------------------------------------------------ dry run
def dry_run(model_cfg: dict) -> dict:
    """Features for 2024-26 and coverage only. No model is built, no prediction, no metric."""
    from pitlane.plan3.m10_feature import coverage, document_table, session_starts
    ent = pd.read_parquet(ROOT / "data" / "processed" / "entries.parquet", columns=["season", "round", "date", "constructorId"])
    ent = ent[ent["season"] >= 2024].reset_index(drop=True)
    docs, cuts = document_table(), session_starts(ent)
    out = {}
    for h in model_cfg["horizons"]:
        c = coverage(ent, docs, cuts, h, model_cfg["scale"])
        col = M.build_column(ent, docs, cuts, h, model_cfg["scale"], model_cfg["centre"])
        c["share_nonzero_x_of_document_races"] = float((col.abs() > 1e-12).sum() / max(c["car_races_with_document_race"], 1))
        out[h] = c
    out["H0"] = dict(note="own-race document never available before the weekend; no M10 column; rung == reference")
    return out


# ------------------------------------------------------------------------------------ scoring (fresh holdout only)
def score(entries: pd.DataFrame, model_cfg: dict, races: list[tuple[int, int]] | None = None, write: bool = True) -> dict:
    """Score rung vs reference at each horizon on the fresh races of `entries` (optionally a subset `races`)."""
    if races is not None:
        assert_after_freeze(races)                                     # refuse before anything is built
    horizons = list(model_cfg["horizons"])
    docs, cuts = M.document_table(), M.session_starts(entries)
    first_h = M.add_column(entries, horizons[0], model_cfg, docs, cuts)
    _, counted = split_races(first_h, model_cfg["split"])
    key = first_h.groupby("race_id")[["season", "round"]].first()
    assert_after_freeze([tuple(map(int, key.loc[r])) for r in counted])    # the harness split must agree with the guard
    if not counted:
        log("no fresh-holdout race in the entries table yet: nothing to score")
        return {}
    keep = None
    if races is not None:
        keep = {r for r in counted if tuple(map(int, key.loc[r])) in set(races)}
        missing = set(races) - {tuple(map(int, key.loc[r])) for r in keep}
        if missing:
            log(f"WARNING: requested races not in the fresh split, not scored: {sorted(missing)}")
    summary, tabs = {}, {}
    for h in horizons:
        ent = M.add_column(entries, h, model_cfg, docs, cuts)
        for tag, cfg in (("ref", reference_cfg(model_cfg, h)), ("m10", rung_cfg(model_cfg, h))):
            p, m = walk_forward(ent, build_model(cfg, ent), model_cfg["split"], refit_every=1)
            if keep is not None:
                p, m = p[p["race_id"].isin(keep)], m[m["race_id"].isin(keep)]
            assert_after_freeze(list(zip(m["season"], m["round"])))
            tabs[(h, tag)] = (cfg, p, m)
            log(f"{h} {tag}: rho={m['spearman'].mean():.4f} mae={m['mae'].mean():.3f} races={len(m)}")
    nb = model_cfg["bootstrap"]
    gains = []
    for h in horizons:
        a = A.race_table(U.outcome_free(tabs[(h, "ref")][1]))
        b = A.race_table(U.outcome_free(tabs[(h, "m10")][1]))
        g = A.paired_gain(a, b, [k for k in VERDICT_METRICS], n_boot=nb["n_boot"], seed=nb["seed"])
        gains.append(g.assign(horizon=h, reference=model_cfg["references"][h]["label"]))
    gain = pd.concat(gains, ignore_index=True)
    print("\nPAIRED GAIN, M10 rung minus reference (positive = rung better; 95% bootstrap CI over fresh races)\n"
          + gain.round(4).to_string(index=False))
    if write:
        lb = ROOT / model_cfg["leaderboard"]
        ids = {}
        for (h, tag), (cfg, p, m) in tabs.items():
            lab = f"m10_{h.lower()}_" + ("ref_" + model_cfg["references"][h]["label"] if tag == "ref" else "declared_x")
            ids[lab] = save_run(MDIR, lab, dict(cfg, note="R9 part 2; fresh holdout only"), p, m, model_cfg["split"], "M10",
                                cfg.get("cell", ""), leaderboard=lb)
        res = MDIR / "results"
        gain.to_csv(res / f"paired_gain_fresh_{time.strftime('%Y%m%d-%H%M')}.csv", index=False)
        (res / "last_run_ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    summary["gain"] = gain
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--dry-run", action="store_true", help="features for 2024-26, coverage only; no model, no metric")
    ap.add_argument("--races", default=None, help="comma list season:round, all strictly after the freeze")
    args = ap.parse_args()
    model_cfg = load_yaml(args.config)["model"]
    resolve_lam_x(model_cfg)
    if args.dry_run:
        print(json.dumps(dry_run(model_cfg), indent=2))
        return
    from pitlane.plan2.pipeline2 import load_all
    score(load_all(), model_cfg, parse_races(args.races))


if __name__ == "__main__":
    main()
