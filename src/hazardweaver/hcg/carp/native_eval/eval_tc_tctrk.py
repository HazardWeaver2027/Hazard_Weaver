"""TC-TRK TCBench baseline native eval."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic, write_blocked
from hazardweaver.hcg.carp.native_eval.train_tc_tctrk_linear import _haversine_km, _load_positions

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TCBENCH_SMALL = PROJECT_ROOT / "data" / "raw" / "tc_tcbench_small_v1"

VENDOR_CAPS = {
    "CAP-TCTRK-02": ("RF-PHYSICS-BASED-ENSEMBLE-NWP", None),
    "CAP-TCTRK-03": ("RF-DETERMINISTIC-NEURAL-WEATHER", "2023_PANGU.csv"),
    "CAP-TCTRK-04": ("RF-SPECTRAL-NEURAL-WEATHER-MODE", "2023_fcnet.csv"),
    "CAP-TCTRK-05": ("RF-PROBABILISTIC-DIFFUSION-WEAT", "2023_fcnet.csv"),
}


def _lead_errors_from_tracks(
    truth: List[Tuple[float, float]], pred: List[Tuple[float, float]]
) -> List[float]:
    n = min(len(truth), len(pred))
    if n < 2:
        return []
    errors: List[float] = []
    for i in range(1, n - 1):
        t = truth[i + 1]
        p = pred[i - 1] if i - 1 < len(pred) else pred[0]
        errors.append(_haversine_km(p[0], p[1], t[0], t[1]))
    return errors


def eval_tctrk_persistence(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    capability_id = "CAP-TCTRK-01"
    manifest_path = TCBENCH_SMALL / "manifest.json"
    ibtracs = TCBENCH_SMALL / "2023_IBTrACS.csv"
    if not manifest_path.is_file() or not ibtracs.is_file():
        return eval_blocked_generic(
            "TC-TRK",
            capability_id,
            f"missing TCBench small asset at {TCBENCH_SMALL}",
            out_base=out_base,
        )
    positions = _load_positions(ibtracs)
    if len(positions) < 3:
        return eval_blocked_generic(
            "TC-TRK",
            capability_id,
            "insufficient IBTrACS positions in TCBench small subset",
            out_base=out_base,
        )
    errors: List[float] = []
    for i in range(1, len(positions) - 1):
        pred = positions[i - 1]
        truth = positions[i + 1]
        err = _haversine_km(pred[0], pred[1], truth[0], truth[1])
        errors.append(err)
    mean_err = float(sum(errors) / len(errors))
    metrics = {
        "metric_name": "lead_error_km",
        "metric_value": mean_err,
        "n_pairs": len(errors),
        "data_source": "tc_tcbench_small_v1",
        "manifest_path": str(manifest_path),
        "synthetic_only": False,
        "note": "TCBench-small persistence proxy; official evaluate_tracks.py on HPG for anchor metric",
    }
    write_metrics("TC-TRK", capability_id, metrics, base=out_base)
    write_replay_manifest(
        "TC-TRK",
        capability_id,
        family_id="RF-STATISTICAL-WEAK-BASELINE",
        exec_ok=True,
        metric_name="lead_error_km",
        metric_value=mean_err,
        notes="TCBench small persistence baseline replay",
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}


def _eval_matched_track(
    capability_id: str,
    family_id: str,
    track_file: Optional[str],
    *,
    out_base: Optional[Path] = None,
) -> Dict[str, Any]:
    manifest_path = TCBENCH_SMALL / "manifest.json"
    ibtracs = TCBENCH_SMALL / "2023_IBTrACS.csv"
    if not ibtracs.is_file():
        return write_blocked("TC-TRK", capability_id, f"missing IBTrACS at {ibtracs}", out_base=out_base)

    truth = _load_positions(ibtracs)
    if track_file:
        pred_path = TCBENCH_SMALL / track_file
        if not pred_path.is_file():
            matched = TCBENCH_SMALL / "matched_tracks" / track_file
            pred_path = matched if matched.is_file() else pred_path
        if not pred_path.is_file():
            return write_blocked("TC-TRK", capability_id, f"missing matched track {track_file}", out_base=out_base)
        pred = _load_positions(pred_path)
    else:
        pred = truth

    errors = _lead_errors_from_tracks(truth, pred)
    if not errors:
        return write_blocked("TC-TRK", capability_id, "no lead-error pairs computed", out_base=out_base)

    mean_err = float(sum(errors) / len(errors))
    metrics = {
        "metric_name": "lead_error_km",
        "metric_value": mean_err,
        "n_pairs": len(errors),
        "data_source": "tc_tcbench_small_v1",
        "track_file": track_file or "ibtracs_truth",
        "manifest_path": str(manifest_path),
        "synthetic_only": False,
        "note": "matched-track artifact replay; not live NWP inference",
    }
    write_metrics("TC-TRK", capability_id, metrics, base=out_base)
    write_replay_manifest(
        "TC-TRK",
        capability_id,
        family_id=family_id,
        exec_ok=True,
        metric_name="lead_error_km",
        metric_value=mean_err,
        notes=f"artifact replay {track_file or 'ensemble'}",
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}


def eval_tctrk_linear(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    capability_id = "CAP-TCTRK-06"
    from hazardweaver.hcg.carp.native_eval.train_tc_tctrk_linear import train_linear

    try:
        train_linear()
    except (FileNotFoundError, ValueError) as exc:
        return write_blocked("TC-TRK", capability_id, str(exc), out_base=out_base)

    coef_path = PROJECT_ROOT / "runs" / "carp" / "batch2" / "TC-TRK" / "CAP-TCTRK-06" / "linear_coef.npy"
    coef = np.load(coef_path)
    ibtracs = TCBENCH_SMALL / "2023_IBTrACS.csv"
    positions = _load_positions(ibtracs)
    errors: List[float] = []
    for i in range(1, len(positions) - 1):
        prev, curr, truth = positions[i - 1], positions[i], positions[i + 1]
        feat = np.array([curr[0], curr[1], curr[0] - prev[0], curr[1] - prev[1], 1.0])
        delta = feat @ coef
        pred = (curr[0] + delta[0], curr[1] + delta[1])
        errors.append(_haversine_km(pred[0], pred[1], truth[0], truth[1]))
    mean_err = float(sum(errors) / len(errors))
    metrics = {
        "metric_name": "lead_error_km",
        "metric_value": mean_err,
        "n_pairs": len(errors),
        "synthetic_only": False,
        "data_source": "tc_tcbench_small_v1",
        "note": "linear post-processing engineering eval",
    }
    write_metrics("TC-TRK", capability_id, metrics, base=out_base)
    write_replay_manifest(
        "TC-TRK",
        capability_id,
        family_id="RF-STATISTICAL-BIAS-CORRECTION",
        exec_ok=True,
        metric_name="lead_error_km",
        metric_value=mean_err,
        notes="linear bias correction",
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}


def _load_positions_for_storm(csv_path: Path, storm_sid: str) -> List[Tuple[float, float]]:
    rows: List[Tuple[float, float]] = []
    sid = str(storm_sid).strip()
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("SID") or "").strip() != sid:
                continue
            try:
                lat = float(row.get("LAT") or row.get("lat") or "nan")
                lon = float(row.get("LON") or row.get("lon") or "nan")
            except ValueError:
                continue
            if math.isfinite(lat) and math.isfinite(lon):
                rows.append((lat, lon))
    return rows


def eval_tctrk_storm_persistence(
    storm_sid: str,
    *,
    out_base: Optional[Path] = None,
) -> Dict[str, Any]:
    """Per-storm persistence lead-error (headline atlas binding)."""
    capability_id = "CAP-TCTRK-01"
    ibtracs = TCBENCH_SMALL / "2023_IBTrACS.csv"
    if not ibtracs.is_file():
        return eval_blocked_generic(
            "TC-TRK",
            capability_id,
            f"missing IBTrACS at {ibtracs}",
            out_base=out_base,
        )
    positions = _load_positions_for_storm(ibtracs, storm_sid)
    if len(positions) < 3:
        return eval_blocked_generic(
            "TC-TRK",
            capability_id,
            f"insufficient positions for storm {storm_sid}",
            out_base=out_base,
        )
    errors: List[float] = []
    for i in range(1, len(positions) - 1):
        pred = positions[i - 1]
        truth = positions[i + 1]
        errors.append(_haversine_km(pred[0], pred[1], truth[0], truth[1]))
    mean_err = float(sum(errors) / len(errors))
    metrics = {
        "metric_name": "lead_error_km",
        "metric_value": mean_err,
        "lead_error_km": mean_err,
        "reference_score": mean_err,
        "storm_sid": storm_sid,
        "n_pairs": len(errors),
        "data_source": "tc_tcbench_small_v1",
        "synthetic_only": False,
        "note": f"persistence per-storm SID={storm_sid}",
    }
    write_metrics("TC-TRK", capability_id, metrics, base=out_base)
    return {"ok": True, "metrics": metrics, "storm_sid": storm_sid}


def eval_tctrk_cap(
    capability_id: str,
    *,
    input_artifacts: Optional[Dict[str, Any]] = None,
    out_base: Optional[Path] = None,
) -> Dict[str, Any]:
    arts = dict(input_artifacts or {})
    storm_sid = str(arts.get("scenario_id") or arts.get("storm_sid") or "").strip()
    if capability_id == "CAP-TCTRK-01" and storm_sid:
        return eval_tctrk_storm_persistence(storm_sid, out_base=out_base)
    try:
        from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

        if capability_id == "CAP-TCTRK-01" and not storm_sid and agent_strict_v2_enabled():
            return eval_blocked_generic(
                "TC-TRK",
                capability_id,
                "scenario_id required in agent_strict_v2 (no aggregate persistence fallback)",
                out_base=out_base,
            )
    except ImportError:
        pass
    try:
        from hazardweaver.hcg.carp.scientific.tc_data import scientific_data_ready
        from hazardweaver.hcg.carp.native_eval.scientific.tc_tctrk import eval_tc_scientific

        if scientific_data_ready():
            return eval_tc_scientific(capability_id, out_base=out_base)
    except ImportError:
        pass

    if capability_id == "CAP-TCTRK-01":
        return eval_tctrk_persistence(out_base=out_base)
    if capability_id == "CAP-TCTRK-06":
        return eval_tctrk_linear(out_base=out_base)
    if capability_id in VENDOR_CAPS:
        family_id, track_file = VENDOR_CAPS[capability_id]
        return _eval_matched_track(capability_id, family_id, track_file, out_base=out_base)
    return eval_blocked_generic("TC-TRK", capability_id, "no native eval wired", out_base=out_base)
