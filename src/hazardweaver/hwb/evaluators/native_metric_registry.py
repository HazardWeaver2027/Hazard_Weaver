"""Per-task native scientific metric registry (no cross-hazard macro average)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Union

import numpy as np

from hazardweaver.hwb.evaluators.fl2_floodcast_rmse_csi import (
    passes_fl2_tolerance,
    score_scalar_submission,
)
from hazardweaver.hwb.evaluators.mh4_tc_building_loss import LOSS_VARIABLE, score_event_loss as _mh4_score_event_loss
from hazardweaver.hwb.synthetic.evaluators import metrics as synth_metrics

Number = Union[int, float]

MH1_HIGHER_IS_BETTER = frozenset(
    {"brier", "volume_mae", "threshold_accuracy", "auprc", "threat_score", "sdo_skill", "iou", "accuracy"}
)


@dataclass(frozen=True)
class MetricSpec:
    metric_name: str
    higher_is_better: bool
    score_fn: Callable[[Any, Any], float]


def _scalar_mae(pred: Number, ref: Number) -> float:
    return float(abs(float(pred) - float(ref)))


def _scalar_iou_proxy(pred: Number, ref: Number) -> float:
    """Scalar IoU proxy for fixture tasks (pred/ref in [0,1])."""
    p = float(pred)
    r = float(ref)
    inter = min(p, r)
    union = max(p, r)
    if union <= 0:
        return 1.0 if p == r else 0.0
    return float(inter / union)


def _mh4_log_mae(pred: Number, ref: Number) -> float:
    return float(_mh4_score_event_loss(float(pred), float(ref))["log_mae"])


def _extract_prediction_value(final_artifact: Mapping[str, Any]) -> Number:
    value = final_artifact.get("value")
    if not isinstance(value, Mapping):
        if isinstance(value, (int, float)):
            return value
        raise ValueError("final_artifact.value must contain a scalar score")
    metric_name = str(value.get("metric_name") or "")
    if metric_name == "rmse_depth" or (metric_name == "" and "rmse_depth" in value):
        if isinstance(value.get("rmse_depth"), Mapping):
            return value
        if value.get("rmse_depth") is not None:
            return value["rmse_depth"]
    if LOSS_VARIABLE in value:
        return value[LOSS_VARIABLE]
    if value.get("vendor_final_answer_eval_v1") and "metric_value" in value:
        return value["metric_value"]
    if "reference_score" in value:
        return value["reference_score"]
    if "reference_label" in value:
        return value["reference_label"]
    if "iou" in value:
        return value["iou"]
    if "score" in value:
        return value["score"]
    if "metric_value" in value:
        return value["metric_value"]
    raise ValueError("final_artifact.value must contain a scalar score")


def _extract_reference_value(reference_view: Mapping[str, Any]) -> Number:
    outputs = reference_view.get("outputs") or {}
    if "rmse_depth" in outputs:
        return outputs
    if LOSS_VARIABLE in outputs:
        return outputs[LOSS_VARIABLE]
    if "reference_score" in outputs:
        return outputs["reference_score"]
    if "reference_label" in outputs:
        return outputs["reference_label"]
    raise ValueError("reference_view.outputs.reference_score or reference_label required")


def _fl2_rmse_depth(pred: Number, ref: Number) -> float:
    submitted = pred if isinstance(pred, Mapping) else {"rmse_depth": pred}
    reference = ref if isinstance(ref, Mapping) else {"rmse_depth": ref}
    return float(score_scalar_submission(submitted, reference)["score"])


NATIVE_METRIC_REGISTRY: Dict[str, MetricSpec] = {
    "hwb_wf3_spread_fixture_v1": MetricSpec(
        metric_name="iou",
        higher_is_better=True,
        score_fn=_scalar_iou_proxy,
    ),
    "hwb_mh3_compound_fixture_v1": MetricSpec(
        metric_name="mae",
        higher_is_better=False,
        score_fn=_scalar_mae,
    ),
    "hwb_fl2_fixture_v1": MetricSpec(
        metric_name="rmse_depth",
        higher_is_better=False,
        score_fn=_fl2_rmse_depth,
    ),
    "hwb_fl2_parametric_v1": MetricSpec(
        metric_name="rmse_depth",
        higher_is_better=False,
        score_fn=_fl2_rmse_depth,
    ),
    "hwb_fl2_solver_parametric_v1": MetricSpec(
        metric_name="rmse_depth",
        higher_is_better=False,
        score_fn=_fl2_rmse_depth,
    ),
    "fl2_floodcast_rmse_csi": MetricSpec(
        metric_name="rmse_depth",
        higher_is_better=False,
        score_fn=_fl2_rmse_depth,
    ),
    "l2_lhasa_auprc": MetricSpec(
        metric_name="auprc",
        higher_is_better=True,
        score_fn=lambda p, r: _scalar_iou_proxy(p, r),
    ),
    "dr_out_sdo_skill": MetricSpec(
        metric_name="sdo_skill",
        higher_is_better=True,
        score_fn=lambda p, r: _scalar_iou_proxy(p, r),
    ),
    "hwb_pfdf_parametric_v1": MetricSpec(
        metric_name="log_volume_v1",
        higher_is_better=False,
        score_fn=lambda p, r: float(p),
    ),
    "hwb_drout_parametric_v1": MetricSpec(
        metric_name="sdo_skill",
        higher_is_better=True,
        score_fn=lambda p, r: float(p),
    ),
    "hwb_tctrk_parametric_v1": MetricSpec(
        metric_name="lead_error_km",
        higher_is_better=False,
        score_fn=lambda p, r: float(p),
    ),
    "hwb_mh1_parametric_v1": MetricSpec(
        metric_name="brier",
        higher_is_better=True,
        score_fn=lambda p, r: float(p),
    ),
    "hwb_mh1_atlas_state_variant_v1": MetricSpec(
        metric_name="log_volume_v1",
        higher_is_better=False,
        score_fn=lambda p, r: float(p),
    ),
    "mh1_volume_mae": MetricSpec(
        metric_name="volume_mae",
        higher_is_better=False,
        score_fn=_scalar_mae,
    ),
    "mh2_lifeline_chain": MetricSpec(
        metric_name="lifeline_chain_iou",
        higher_is_better=True,
        score_fn=_scalar_iou_proxy,
    ),
    "mh4_tc_building_loss_v1": MetricSpec(
        metric_name="log_mae",
        higher_is_better=False,
        score_fn=_mh4_log_mae,
    ),
    "MH-4": MetricSpec(
        metric_name="log_mae",
        higher_is_better=False,
        score_fn=_mh4_log_mae,
    ),
}


def get_metric_spec(taskpack: Mapping[str, Any], *, reference_view: Optional[Mapping[str, Any]] = None) -> MetricSpec:
    ref = reference_view if reference_view is not None else (taskpack.get("reference_view") or {})
    taskpack_id = str(taskpack.get("taskpack_id") or "")
    if taskpack_id == "hwb_mh1_parametric_v1" or (
        taskpack_id.startswith("hwb_") and taskpack_id.endswith("_parametric_v1")
    ):
        metric_name = str(
            (ref.get("tolerance") or {}).get("metric")
            or (ref.get("outputs") or {}).get("metric_name")
            or "brier"
        )
        higher = metric_name in MH1_HIGHER_IS_BETTER or metric_name in {
            "average_precision",
            "auprc",
            "sdo_skill",
            "lifeline_chain_iou",
            "threat_score",
        }
        return MetricSpec(
            metric_name,
            higher,
            lambda p, r: float(p),
        )
    taskpack_id = str(taskpack.get("taskpack_id") or "")
    if taskpack_id not in NATIVE_METRIC_REGISTRY:
        ref = taskpack.get("reference_view") or {}
        spec = ref.get("final_output_spec") or {}
        metric_name = str(spec.get("metric") or (ref.get("tolerance") or {}).get("metric") or "mae")
        if metric_name == "iou":
            return MetricSpec("iou", True, _scalar_iou_proxy)
        if metric_name == "accuracy":
            return MetricSpec("accuracy", True, lambda p, r: abs(float(p) - float(r)))
        if metric_name == "rmse_depth":
            return MetricSpec("rmse_depth", False, _fl2_rmse_depth)
        return MetricSpec("mae", False, _scalar_mae)
    return NATIVE_METRIC_REGISTRY[taskpack_id]


def score_artifact(
    taskpack: Mapping[str, Any],
    final_artifact: Mapping[str, Any],
    *,
    reference_view: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Score artifact using per-task native metric only (evaluator view)."""
    ref_view = reference_view if reference_view is not None else (taskpack.get("reference_view") or {})
    taskpack_id = str(taskpack.get("taskpack_id") or "")
    _parametric_ids = {
        "hwb_fl2_parametric_v1",
        "hwb_fl2_solver_parametric_v1",
        "hwb_pfdf_parametric_v1",
        "hwb_drout_parametric_v1",
        "hwb_tctrk_parametric_v1",
        "hwb_mh1_parametric_v1",
        "hwb_mh1_atlas_state_variant_v1",
    }
    try:
        from hazardweaver.hwb.registry.seven_track_parametric_v1 import SEVEN_TRACK_PARAMETRIC_IDS

        _parametric_ids.update(SEVEN_TRACK_PARAMETRIC_IDS)
    except ImportError:
        pass
    from hazardweaver.hwb.evaluators.artifact_checker import _parametric_ref_locked_to_executed_capability

    if (
        reference_view is None
        and taskpack_id in _parametric_ids
        and not _parametric_ref_locked_to_executed_capability(ref_view, final_artifact)
    ):
        meta = taskpack.get("metadata") or {}
        overlay = str(meta.get("hieraplan_eval_view") or "")
        native_outcome = str(
            meta.get("vendor_final_answer_eval_v1") or meta.get("native_outcome_eval_v1") or ""
        )
        if not overlay or native_outcome in {"native_outcome_eval_v1", "vendor_final_answer_eval_v1"}:
            if taskpack_id.startswith("hwb_fl2"):
                from hazardweaver.hwb.registry.fl2_parametric_resolver import resolve_reference_view
            else:
                from hazardweaver.hwb.registry.track_parametric_resolver import resolve_reference_view

            value = final_artifact.get("value") or {}
            sid = value.get("scenario_id") if isinstance(value, Mapping) else None
            ref_view = resolve_reference_view(taskpack, scenario_id=str(sid) if sid else None)
    spec = get_metric_spec(taskpack, reference_view=ref_view)
    tol_metric = str((ref_view.get("tolerance") or {}).get("metric") or spec.metric_name or "")
    value = final_artifact.get("value") or {}
    if isinstance(value, Mapping) and tol_metric and value.get(tol_metric) is not None:
        pred_val = value[tol_metric]
    else:
        pred_val = _extract_prediction_value(final_artifact)
    ref_val = _extract_reference_value(ref_view)
    raw = spec.score_fn(pred_val, ref_val)
    if spec.metric_name == "iou":
        # IoU proxy on submitted scalar vs reference scalar
        score = _scalar_iou_proxy(pred_val, ref_val) if pred_val != ref_val else float(pred_val)
        if isinstance(final_artifact.get("value"), Mapping) and "reference_score" in final_artifact["value"]:
            score = float(final_artifact["value"]["reference_score"])
    elif spec.metric_name == "mae":
        score = _scalar_mae(pred_val, ref_val)
        if isinstance(final_artifact.get("value"), Mapping) and "reference_score" in final_artifact["value"]:
            submitted = float(final_artifact["value"]["reference_score"])
            score = _scalar_mae(submitted, ref_val)
    elif spec.metric_name == "rmse_depth":
        value = final_artifact.get("value") or {}
        if isinstance(value, Mapping):
            score = float(score_scalar_submission(value, ref_view.get("outputs") or {})["score"])
    else:
        score = raw
    return {
        "metric_name": spec.metric_name,
        "score": float(score),
        "higher_is_better": spec.higher_is_better,
        "prediction": pred_val,
        "reference": ref_val,
    }


def passes_tolerance(taskpack: Mapping[str, Any], score: float, metric_name: str) -> bool:
    ref = taskpack.get("reference_view") or {}
    tol = ref.get("tolerance") or {}
    if tol.get("metric") and str(tol["metric"]) != metric_name:
        return False
    if "min_score" in tol:
        return score >= float(tol["min_score"])
    if "max_abs_error" in tol:
        ref_score = float((ref.get("outputs") or {}).get("reference_score", 0))
        return abs(ref_score - score) <= float(tol["max_abs_error"])
    return True


def macro_average_across_hazards(*_args: Any, **_kwargs: Any) -> None:
    """Explicitly forbidden in Phase A — RQ1 aggregation belongs in Phase C."""
    raise NotImplementedError(
        "macro_average_across_hazards is forbidden in per-task native metric registry"
    )


def array_metric(metric_name: str, pred: np.ndarray, target: np.ndarray) -> float:
    """Optional array-level metrics for bundle graders (not cross-hazard)."""
    fn = getattr(synth_metrics, metric_name, None)
    if fn is None:
        raise ValueError(f"unknown array metric: {metric_name}")
    return float(fn(pred, target))
