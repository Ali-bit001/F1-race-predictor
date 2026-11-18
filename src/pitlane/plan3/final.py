"""PLAN3 R14: the final pre-race model, shared by the run script, `predict.py` and the tests.

Contents
- the car race-pace history column `pace_hist_x` (M09 `M09_car_pace_hist`, D1 median clean lap) for every season with laps;
- model configs from the frozen file, with every new switch optional so that "all off" is `latent_111` bit for bit;
- `forward_scores`: the harness's walk-forward loop (refit before every target race) without any metric, so it also works for
  a race whose outcome is not known;
- `predict_race`: everything needed to predict one race from the frozen config and the data up to that race.

Leakage rule enforced here: a race is predicted from rows with race_id <= r only, and the outcome columns of race r are blanked
before anything is built.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pitlane.config import ROOT
from pitlane.features import pace as PACE
from pitlane.models.registry import build_model
from pitlane.plan2 import dnf as D
from pitlane.plan2 import simulate as S
from pitlane.plan3 import decode as DEC

PACE_COL = "pace_hist_x"
# Columns that are outcomes of the race itself (known only after it). Blanked for the race being predicted.
OUTCOME_COLS = ["finish_order", "positionText", "classified", "status", "points", "pos", "y", "dnf", "inc", "mech"]


# ------------------------------------------------------------------------------------ freeze file
def sha256_file(path: Path | str) -> str:
    """SHA-256 of the file with CRLF turned into LF first, so a Windows checkout (autocrlf) gives the same hash."""
    return hashlib.sha256(Path(path).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()


def freeze_hash(freeze_md: Path | str) -> str:
    """The SHA-256 recorded in FREEZE.md (line `SHA-256 of frozen_config.yaml: <64 hex>`)."""
    m = re.search(r"SHA-256 of frozen_config\.yaml:\s*`?([0-9a-f]{64})`?", Path(freeze_md).read_text(encoding="utf-8"))
    if not m:
        raise ValueError("FREEZE.md has no SHA-256 line")
    return m.group(1)


def verify_freeze(config_path: Path | str, freeze_md: Path | str) -> str:
    """Raise if the config file does not match the hash recorded at the freeze. Returns the hash."""
    got, want = sha256_file(config_path), freeze_hash(freeze_md)
    if got != want:
        raise RuntimeError(f"frozen_config.yaml was edited after the freeze: sha256 {got} != recorded {want}")
    return got


# ------------------------------------------------------------------------------------ pace history column
def load_pace_inputs(spec: dict, interim: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Laps and pit stops, one source per season range: `spec['lap_source']` maps 'le_2023' -> jolpica, 'ge_2024' -> openf1."""
    interim = interim or ROOT / "data" / "interim" / "plan3"
    laps, pits = pd.read_parquet(interim / "laps.parquet"), pd.read_parquet(interim / "pits.parquet")
    split = int(spec["source_split_season"])
    src = spec["lap_source"]

    def pick(df: pd.DataFrame) -> pd.DataFrame:
        return df[((df["source"] == src["le"]) & (df["season"] <= split)) | ((df["source"] == src["gt"]) & (df["season"] > split))]
    return pick(laps), pick(pits)


def build_pace_table(spec: dict, interim: Path | None = None) -> pd.DataFrame:
    laps, pits = load_pace_inputs(spec, interim)
    return PACE.build_pace_table(laps, pits, float(spec["thresh"]), int(spec["min_clean"]))


def pace_hist_raw(entries: pd.DataFrame, pace: pd.DataFrame, spec: dict) -> np.ndarray:
    """Unscaled decayed mean (earlier races only, lineage map) of the car's field-median-relative race-pace gap, aligned to
    `entries` rows. NaN where the car has fewer than `min_prior_obs` earlier valid observations. Same arithmetic as the M09
    run (`feature_columns`), for every season with laps. `entries` needs race_id, season, round, driverId, lineage."""
    base = entries[["race_id", "season", "round", "driverId", "lineage"]].assign(q_best_gap=np.nan)
    rq = PACE.rq_table(pace, base, spec["pace_col"])
    gp = rq[["race_id", "season", "driverId", "lineage", "race_gap_pct"]].copy()
    gp["rq_c"] = gp["race_gap_pct"] - gp.groupby("race_id")["race_gap_pct"].transform("median")
    h = PACE.rq_history_features(gp.assign(rq_tm=np.nan), (int(spec["halflife"]),), None)
    h = h[["race_id", "driverId", f"con_rq_ew{int(spec['halflife'])}", "con_n"]]
    h = h.rename(columns={f"con_rq_ew{int(spec['halflife'])}": "v"})
    h["v"] = h["v"].where(h["con_n"] >= int(spec["min_prior_obs"]))
    out = entries[["race_id", "driverId"]].merge(h[["race_id", "driverId", "v"]], on=["race_id", "driverId"], how="left")
    return out["v"].to_numpy(float)


def fit_scale_sd(raw: np.ndarray, season: np.ndarray, max_season: int) -> float:
    """SD of the raw column over seasons <= max_season (pandas ddof=1, NaN skipped): a constant fixed before dev."""
    return float(pd.Series(raw)[np.asarray(season) <= max_season].std())


def add_pace_hist(entries: pd.DataFrame, pace: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """`entries` with PACE_COL added (raw / spec['scale_sd']). The scale is a constant stored in the config."""
    out = entries.copy()
    out[PACE_COL] = pace_hist_raw(entries, pace, spec) / float(spec["scale_sd"])
    return out


# ------------------------------------------------------------------------------------ models
def latent_cfg(base: dict, reverse: bool = False, pace_hist: bool = False, lam_x: float | None = None,
               car_lam: float | None = None) -> dict:
    """Model config for `build_model`. `base` is the plan-1 frozen `latent_111` dict. All switches off returns `base`
    unchanged (so the model is `latent_111` bit for bit). A car penalty `car_lam` keeps the driver penalty at the base value."""
    c = dict(base)
    if reverse:
        c["reverse"] = True
    if pace_hist:
        c["x_cols"] = [PACE_COL]
        c["lam_x"] = float(lam_x)
    if car_lam is not None and float(car_lam) != float(base["lam"]):
        c["lam_a"] = float(base["lam"])
        c["lam"] = float(car_lam)
    return c


def forward_scores(entries: pd.DataFrame, model, target_ids: list[int], refit_every: int = 1) -> np.ndarray:
    """The harness walk-forward loop: refit on all rows before each target race (or chunk), predict it. Returns one score
    per row of `entries` (NaN for rows of non-target races). Identical to `walk_forward`'s first loop."""
    assert (entries.index == np.arange(len(entries))).all(), "entries must have a RangeIndex"
    race_rows = entries.groupby("race_id").indices
    scores = np.full(len(entries), np.nan)
    for i in range(0, len(target_ids), refit_every):
        chunk = target_ids[i:i + refit_every]
        test_rows = np.concatenate([race_rows[r] for r in chunk])
        first_row = test_rows.min()
        scores[test_rows] = model.fit_predict(entries.iloc[:first_row], entries.iloc[test_rows])
    return scores


# ------------------------------------------------------------------------------------ one race
def prepare_pre_race(entries: pd.DataFrame, race_id: int) -> pd.DataFrame:
    """Rows of races <= race_id only; the outcome columns of race_id blanked. Row order and RangeIndex kept."""
    e = entries[entries["race_id"] <= race_id].reset_index(drop=True).copy()
    m = (e["race_id"] == race_id).to_numpy()
    for c in OUTCOME_COLS:
        if c not in e.columns:
            continue
        if e[c].dtype == bool:
            e.loc[m, c] = False
        elif e[c].dtype.kind in "iuf":
            e[c] = e[c].astype(float)
            e.loc[m, c] = np.nan
        else:
            e[c] = e[c].astype(object)
            e.loc[m, c] = None
    return e


def p_dnf_for_race(entries: pd.DataFrame, race_id: int, C: float) -> np.ndarray:
    """M07 retirement probability for the entries of `race_id`, fitted on races before it (as `walk_dnf`)."""
    X, y = D.design(entries), entries["dnf"].to_numpy(float)
    race = entries["race_id"].to_numpy()
    tr, te = (race < race_id) & ~np.isnan(y), race == race_id
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=C, max_iter=500).fit(sc.transform(X[tr]), y[tr])
    return m.predict_proba(sc.transform(X[te]))[:, 1]


def simulate_race_matrix(frame: pd.DataFrame, sim: dict) -> np.ndarray:
    """Driver x position probability matrix of the frozen simulator for one race. `frame`: score, p_dnf, grid_norm, stick,
    circ_vol, rain_fc (+ race_id, pos, classified, which are not read by the simulation)."""
    races = S.prepare(frame, {k: sim[k] for k in ("stick_ref", "vol_mu", "vol_sd")})
    theta = {k: float(sim["theta"][k]) for k in S.DEFAULT}
    return DEC.simulate_matrices(races, theta, sim.get("variant", "S3"), int(sim["draws"]), int(sim["seed"]))[0]


def predict_race(entries: pd.DataFrame, race_id: int, cfg: dict, horizon: str = "H2", with_sim: bool = True,
                 pace: pd.DataFrame | None = None, n_warm: int | None = None) -> pd.DataFrame:
    """Predict one race from the frozen config `cfg` (the parsed frozen_config.yaml) and the data up to that race.

    `entries`: the plan-2 entries table (`pipeline2.load_all()`), which must contain the rows of `race_id` with their
    pre-race columns (grid, qualifying, features). Anything after `race_id` is dropped and the outcome columns of `race_id`
    are blanked before use, so nothing from the race or later is read. `pace`: the pace table (default: built from the laps
    on disk; races >= race_id are not read for the column of race_id even if present).
    Returns one row per entry: model score, and for H2 with the simulator the position probabilities and the D0 order.
    """
    hz = cfg["horizons"][horizon]
    warm_n = int(cfg["walk_forward"]["n_warm"] if n_warm is None else n_warm)
    ent = prepare_pre_race(entries, race_id)
    if hz.get("uses_pace_hist"):
        pt = build_pace_table(cfg["pace_feature"]) if pace is None else pace
        ent = add_pace_hist(ent, pt, cfg["pace_feature"])
    model = build_model(hz["model"], ent)
    races = sorted(ent["race_id"].unique())
    targets = [r for r in races if r < race_id][-warm_n:] + [race_id]
    score = forward_scores(ent, model, targets, int(cfg["walk_forward"]["refit_every"]))
    te = (ent["race_id"] == race_id).to_numpy()
    out = ent.loc[te, ["race_id", "season", "round", "driverId", "lineage", "grid"]].copy()
    out["score"] = score[te]
    out["horizon"] = horizon
    if with_sim and "simulator" in hz:
        sim = hz["simulator"]
        fr = ent.loc[te].copy()
        fr["score"] = score[te]
        fr["p_dnf"] = p_dnf_for_race(ent, race_id, float(cfg["retirement_model"]["C"]))
        P_ = simulate_race_matrix(fr, sim)
        n = len(fr)
        out["p_dnf"] = fr["p_dnf"].to_numpy()
        out["e_pos"] = DEC.expected_position(P_)
        out["p_win"], out["p_podium"], out["p_points"] = P_[:, 0], P_[:, :3].sum(1), P_[:, :min(10, n)].sum(1)
        util = DEC.break_ties(-out["e_pos"].to_numpy(), out["driverId"].to_numpy())      # decoder D0, outcome-free tie-break
        order = np.argsort(-util, kind="stable")
        pred = np.empty(n)
        pred[order] = np.arange(1, n + 1)
        out["pred_place"] = pred.astype(int)
    else:
        util = DEC.break_ties(out["score"].to_numpy(), out["driverId"].to_numpy())
        order = np.argsort(-util, kind="stable")
        pred = np.empty(len(out))
        pred[order] = np.arange(1, len(out) + 1)
        out["pred_place"] = pred.astype(int)
    return out.reset_index(drop=True)
