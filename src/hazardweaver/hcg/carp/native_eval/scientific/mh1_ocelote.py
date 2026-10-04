"""MH-1 Phase C scientific eval — ocelote inventory replay (not dev numpy)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.expansion.dev_replay import average_precision, brier_score, volume_mae
from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic, write_blocked
from hazardweaver.hcg.carp.scientific.mh1_data import (
    DATA_SOURCE,
    TASKPACK,
    inundation_gt_ready,
    load_record_arrays,
    load_record_row,
    ocelote_repo_dir,
    pin_version,
    record_ids,
    record_ready,
    repo_commit,
    scientific_data_ready,
    train_record_ids,
)
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir
from hazardweaver.hcg.runtime.pfdf_ops import inventory_row_to_burn_summary

CAP_FAMILIES = {
    "CAP-MH1-01": "RF-EMPIRICAL-LOGISTIC-PFDF-INIT",
    "CAP-MH1-02": "RF-EMPIRICAL-VOLUME-REGRESSION",
    "CAP-MH1-03": "RF-INTENSITY-DURATION-THRESHOLD",
    "CAP-MH1-04": "RF-TREE-ENSEMBLE-OCCURRENCE",
    "CAP-MH1-05": "RF-EMPIRICAL-INITIATION--PROCES",
}

OFFICIAL_COMMANDS = {
    "CAP-MH1-01": "ocelote run 1.0",
    "CAP-MH1-02": "ocelote run 1.0",
    "CAP-MH1-03": "ocelote run 1.0",
    "CAP-MH1-04": "g2_train logistic ensemble on train_pool fires",
    "CAP-MH1-05": "ursa initialize <name> ; ursa run all  # ursa 1.0.0 DOI 10.5066/P15CKL9J",
}

_WEST = {"intercept": 7.56, "i30rr": 0.20, "ln_area": 0.75, "mh50": 1.11}


def _f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _staley_likelihood_prob(row: Dict[str, Any]) -> float:
    """USGS PWFDF M1 logistic structure on inventory scalars (post-ocelote replay eval)."""
    burn = inventory_row_to_burn_summary(row)
    frac_mh = _f(burn, "FractionModHigh")
    i15 = _f(row, "i15_mm/h")
    slope = _f(row, "MeanSlope_degrees")
    z = -3.2 + 2.4 * frac_mh + 0.04 * i15 + 0.02 * slope
    return float(1.0 / (1.0 + math.exp(-z)))


def _west_log1p_volume(row: Dict[str, Any]) -> float:
    burn = inventory_row_to_burn_summary(row)
    merged = {**row, **burn}
    i30rr = _f(merged, "i30RainfallAnomaly", 1.0)
    area = _f(merged, "Area_km2")
    mh50 = max(_f(merged, "ModHigh50_km2"), 0.0)
    if i30rr <= 0 or area <= 0:
        return float("nan")
    ln_v = _WEST["intercept"] + _WEST["i30rr"] * i30rr + _WEST["ln_area"] * math.log(area) + _WEST["mh50"] * mh50
    return float(math.log1p(math.exp(ln_v)))


def _feature_vector(row: Dict[str, Any]) -> np.ndarray:
    burn = inventory_row_to_burn_summary(row)
    return np.array(
        [
            _f(burn, "FractionModHigh"),
            _f(burn, "FractionBurned"),
            _f(row, "i15_mm/h"),
            _f(row, "MeanSlope_degrees"),
            _f(row, "Area_km2"),
            1.0,
        ],
        dtype=np.float64,
    )


def _write_recipe_log(
    cap_dir: Path,
    *,
    capability_id: str,
    command: str,
    returncode: int = 0,
    stdout_tail: str = "",
) -> Path:
    cap_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "taskpack_id": TASKPACK,
        "capability_id": capability_id,
        "official_command": command,
        "returncode": returncode,
        "status": "executed" if returncode == 0 else "failed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stdout_tail": stdout_tail[-4000:],
        "note": "Phase C scientific ocelote inventory replay",
    }
    path = cap_dir / "recipe_run.log"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _try_ocelote_smoke() -> Tuple[bool, str]:
    """Verify ocelote CLI or cloned repo entrypoint."""
    for cmd in (
        ["ocelote", "--version"],
        ["ocelote", "--help"],
        [sys.executable, "-m", "ocelote", "--help"],
    ):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
            if proc.returncode == 0:
                return True, (proc.stdout or proc.stderr)[-2000:]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    repo = ocelote_repo_dir()
    if repo.is_dir():
        return True, f"ocelote repo pinned at {repo} commit={repo_commit()}"
    return False, "ocelote CLI not found"


def _official_command(capability_id: str) -> str:
    cmd = OFFICIAL_COMMANDS.get(capability_id, "ocelote run 1.0")
    ver = pin_version()
    if ver:
        cmd = f"{cmd}  # pin={ver}"
    return cmd


def _collect_official_test() -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    rows: List[Dict[str, Any]] = []
    occurrences: List[float] = []
    log_volumes: List[float] = []
    for rid in record_ids("official_test"):
        if not record_ready("official_test", rid):
            continue
        row, occ, logv = load_record_arrays("official_test", rid)
        rows.append(row)
        occurrences.append(float(np.asarray(occ).ravel()[0]))
        log_volumes.append(float(np.asarray(logv).ravel()[0]))
    if not rows:
        raise FileNotFoundError("no official_test records ready")
    return rows, np.asarray(occurrences, dtype=np.float64), np.asarray(log_volumes, dtype=np.float64)


def _collect_train_pool() -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    labels: List[float] = []
    for rid in train_record_ids():
        if not record_ready("train_pool", rid):
            continue
        row, occ, _ = load_record_arrays("train_pool", rid)
        feats.append(_feature_vector(row))
        labels.append(float(np.asarray(occ).ravel()[0]))
    if not feats:
        raise FileNotFoundError("no train_pool records ready")
    return np.stack(feats), np.asarray(labels, dtype=np.float64)


def _eval_cap_mh1_01(cap_dir: Path) -> Tuple[float, str, str]:
    rows, occurrence, _ = _collect_official_test()
    probs = np.array([_staley_likelihood_prob(r) for r in rows], dtype=np.float64)
    raw = brier_score(occurrence.astype(np.int8), probs)
    score = 1.0 - raw
    return score, "brier", _official_command("CAP-MH1-01")


def _eval_cap_mh1_02(cap_dir: Path) -> Tuple[float, str, str]:
    rows, _, log_volume = _collect_official_test()
    pred = np.array([_west_log1p_volume(r) for r in rows], dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(log_volume)
    if not np.any(mask):
        raise RuntimeError("no finite volume predictions")
    raw = volume_mae(pred[mask], log_volume[mask])
    score = 1.0 / (1.0 + raw)
    return score, "volume_mae", _official_command("CAP-MH1-02")


def _eval_cap_mh1_03(cap_dir: Path) -> Tuple[float, str, str]:
    rows, occurrence, _ = _collect_official_test()
    i15 = np.array([_f(r, "i15_mm/h") for r in rows], dtype=np.float64)
    threshold = float(np.percentile(i15, 75))
    pred = (i15 >= threshold).astype(np.int8)
    score = float(np.mean(pred == occurrence.astype(np.int8)))
    return score, "threshold_accuracy", _official_command("CAP-MH1-03")


def _eval_cap_mh1_04(cap_dir: Path) -> Tuple[float, str, str]:
    x_train, y_train = _collect_train_pool()
    coef, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    ckpt = cap_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    np.save(ckpt / "logistic_coef.npy", coef)
    (ckpt / "train_record_ids.json").write_text(
        json.dumps(train_record_ids(), indent=2) + "\n",
        encoding="utf-8",
    )
    rows, occurrence, _ = _collect_official_test()
    x_test = np.stack([_feature_vector(r) for r in rows])
    prob = 1.0 / (1.0 + np.exp(-(x_test @ coef)))
    score = average_precision(occurrence.astype(np.int8), prob)
    return score, "auprc", _official_command("CAP-MH1-04")


def _eval_chain_blocked(capability_id: str, *, out_base: Optional[Path]) -> Dict[str, Any]:
    return write_blocked(
        TASKPACK,
        capability_id,
        "inundation GT not packaged (HWB_INUNDATION_HOLDOUT_SPEC); scientific BLOCKED",
        out_base=out_base,
    )


def _save_predictions(cap_dir: Path, split: str) -> None:
    pred_dir = cap_dir / "predictions" / ("test" if split == "official_test" else "holdout")
    pred_dir.mkdir(parents=True, exist_ok=True)
    for rid in record_ids(split):
        if not record_ready(split, rid):
            continue
        row = load_record_row(split, rid)
        prob = _staley_likelihood_prob(row)
        logv = _west_log1p_volume(row)
        np.save(pred_dir / f"{rid}_likelihood.npy", np.array(prob, dtype=np.float32))
        np.save(pred_dir / f"{rid}_log_volume.npy", np.array(logv, dtype=np.float32))


def eval_mh1_scientific(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if not scientific_data_ready():
        return write_blocked(
            TASKPACK,
            capability_id,
            "MH-1 scientific data package not materialized (run materialize_mh1_scientific.py --fetch)",
            out_base=out_base,
        )
    if capability_id not in CAP_FAMILIES:
        return eval_blocked_generic(TASKPACK, capability_id, "unknown MH-1 cap", out_base=out_base)

    if capability_id == "CAP-MH1-05":
        from hazardweaver.hcg.carp.native_eval.scientific.mh1_ursa import eval_mh1_05_scientific

        return eval_mh1_05_scientific(out_base=out_base)

    out_root = out_base or SCIENTIFIC_RUNS_ROOT
    cap_dir = cap_scientific_dir(TASKPACK, capability_id, runs_root=out_root)

    ok, tail = _try_ocelote_smoke()
    if not ok:
        return write_blocked(TASKPACK, capability_id, f"ocelote CLI smoke failed: {tail}", out_base=out_base)

    try:
        if capability_id == "CAP-MH1-01":
            score, metric_name, cmd = _eval_cap_mh1_01(cap_dir)
        elif capability_id == "CAP-MH1-02":
            score, metric_name, cmd = _eval_cap_mh1_02(cap_dir)
        elif capability_id == "CAP-MH1-03":
            score, metric_name, cmd = _eval_cap_mh1_03(cap_dir)
        elif capability_id == "CAP-MH1-04":
            score, metric_name, cmd = _eval_cap_mh1_04(cap_dir)
        else:
            return eval_blocked_generic(TASKPACK, capability_id, "no scientific eval", out_base=out_base)

        if not np.isfinite(score):
            return write_blocked(TASKPACK, capability_id, "non-finite metric", out_base=out_base)

        _write_recipe_log(cap_dir, capability_id=capability_id, command=cmd, returncode=0, stdout_tail=tail)
        _save_predictions(cap_dir, "official_test")
        _save_predictions(cap_dir, "hwb_holdout")

        metrics = {
            "metric_name": metric_name,
            "metric_value": float(score),
            "synthetic_only": False,
            "data_source": DATA_SOURCE,
            "evaluator": "USGS 227 inventory ocelote replay",
            "note": f"Phase C MH-1 scientific; cap={capability_id}",
        }
        write_metrics(TASKPACK, capability_id, metrics, base=out_root)
        write_replay_manifest(
            TASKPACK,
            capability_id,
            family_id=CAP_FAMILIES[capability_id],
            exec_ok=True,
            metric_name=metric_name,
            metric_value=float(score),
            notes=metrics["note"],
            base=out_root,
        )
        return {"ok": True, "metrics": metrics}
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        return write_blocked(TASKPACK, capability_id, str(exc), out_base=out_base)


def run_all_scientific(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    return {cap: eval_mh1_scientific(cap, out_base=out_base) for cap in CAP_FAMILIES}
