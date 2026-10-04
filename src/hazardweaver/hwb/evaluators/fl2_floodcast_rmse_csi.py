"""FL-2 FloodCastBench native evaluator: rmse_depth + CSI@0.01m."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from hazardweaver.hcg.carp.native_eval.fl2_subset import csi_at_threshold, rmse

EVALUATOR_ID = "fl2_floodcast_rmse_csi"
CSI_THRESHOLD = 0.01


def score_depth_artifact(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Array-level scoring for depth sequence predictions."""
    return {
        "rmse_depth": rmse(pred, truth),
        "csi_0.01": csi_at_threshold(pred, truth, CSI_THRESHOLD),
    }


def score_scalar_submission(
    submitted: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fixture-level scalar scoring (HWB dual-gate taskpacks)."""
    pred_rmse = float(submitted.get("rmse_depth", submitted.get("reference_score", 0.0)))
    ref_rmse = float(reference.get("rmse_depth", reference.get("reference_score", 0.0)))
    pred_csi = float(submitted.get("csi_0.01", 0.0))
    ref_csi = float(reference.get("csi_0.01", 0.0))
    return {
        "metric_name": "rmse_depth",
        "score": abs(pred_rmse - ref_rmse),
        "rmse_depth": pred_rmse,
        "csi_0.01": pred_csi,
        "reference_rmse_depth": ref_rmse,
        "reference_csi_0.01": ref_csi,
        "higher_is_better": False,
    }


def _resolved_reference_view(
    taskpack: Mapping[str, Any],
    submitted: Mapping[str, Any],
) -> Mapping[str, Any]:
    ref = taskpack.get("reference_view") or {}
    taskpack_id = str(taskpack.get("taskpack_id") or "")
    if taskpack_id not in ("hwb_fl2_parametric_v1", "hwb_fl2_solver_parametric_v1"):
        return ref
    from hazardweaver.hwb.registry.fl2_parametric_resolver import resolve_reference_view

    sid = submitted.get("scenario_id")
    return resolve_reference_view(taskpack, scenario_id=str(sid) if sid else None)


def passes_fl2_tolerance(taskpack: Mapping[str, Any], submitted: Mapping[str, Any]) -> bool:
    ref = _resolved_reference_view(taskpack, submitted)
    outputs = ref.get("outputs") or {}
    tol = ref.get("tolerance") or {}
    sec = ref.get("secondary_tolerance") or tol.get("secondary") or {}
    gate = ref.get("gate_policy") or {}

    csi_val = float(submitted.get("csi_0.01", 0.0))
    min_csi = float(sec.get("min_score", 0.5)) if "min_score" in sec else None
    primary = str(gate.get("primary_metric") or "csi_0.01")
    rmse_mode = str(gate.get("rmse_mode") or "advisory")

    if gate.get("csi_primary_gate") == "not_applicable":
        return True

    if primary == "csi_0.01" and min_csi is not None:
        if csi_val < min_csi:
            return False
        if rmse_mode in ("advisory", "advisory_record_only"):
            return True

    rmse_err = abs(float(submitted.get("rmse_depth", 0.0)) - float(outputs.get("rmse_depth", 0.0)))
    if "max_abs_error" in tol and rmse_err > float(tol["max_abs_error"]):
        return False

    if min_csi is not None and csi_val < min_csi:
        return False
    return True
