"""E_q — scientific artifact checker (route-neutral)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from hazardweaver.hwb.bridge.submission_metric_v1 import metrics_compatible
from hazardweaver.hwb.evaluators.fl2_floodcast_rmse_csi import passes_fl2_tolerance
from hazardweaver.hwb.evaluators.native_metric_registry import (
    passes_tolerance,
    score_artifact,
)
from hazardweaver.hwb.registry.unified_benchmark_tolerance_v1 import (
    is_unified_benchmark_row,
    normalize_l2_metric_name,
)


@dataclass
class ArtifactEqResult:
    eq: bool
    metric_name: str
    score: float
    threshold_met: bool
    errors: List[str] = field(default_factory=list)
    route_neutral: bool = True


def _validate_canonical_output(
    final_artifact: Mapping[str, Any],
    final_output_spec: Mapping[str, Any],
) -> List[str]:
    errors: List[str] = []
    expected_schema = str(final_output_spec.get("schema_id") or "hwa.final_artifact/v1")
    if final_artifact.get("schema_id") != expected_schema:
        errors.append(
            f"schema_mismatch: expected {expected_schema!r}, got {final_artifact.get('schema_id')!r}"
        )
    value = final_artifact.get("value")
    if value is None:
        errors.append("missing_final_artifact_value")
    output_keys = final_output_spec.get("output_keys") or []
    if isinstance(value, Mapping):
        for key in output_keys:
            if key not in value:
                errors.append(f"missing_output_key:{key}")
    elif output_keys:
        errors.append("value_must_be_object_for_output_keys")
    return errors


def _parametric_ref_locked_to_executed_capability(
    ref_view: Mapping[str, Any],
    final_artifact: Mapping[str, Any],
) -> bool:
    """When gold is already bound to executed capability, ignore hazard scenario_id in artifact."""
    bound_cap = str(ref_view.get("capability_id") or "").strip()
    if not bound_cap:
        return False
    value = final_artifact.get("value") or {}
    prov = final_artifact.get("provenance") or {}
    exec_cap = ""
    if isinstance(value, Mapping):
        exec_cap = str(value.get("capability_id") or "").strip()
    if not exec_cap and isinstance(prov, Mapping):
        exec_cap = str(prov.get("capability_id") or "").strip()
    if bound_cap:
        if exec_cap and exec_cap != bound_cap:
            return False
        return True

    if exec_cap:
        tol_metric = str((ref_view.get("tolerance") or {}).get("metric") or "")
        submitted = str(value.get("metric_name") or "") if isinstance(value, Mapping) else ""
        if tol_metric and submitted and _artifact_metric_names_compatible(
            submitted, tol_metric, {"reference_view": ref_view}
        ):
            return True
    return False


def _artifact_metric_names_compatible(
    submitted_metric: str,
    tol_metric: str,
    taskpack: Mapping[str, Any],
) -> bool:
    if not submitted_metric or not tol_metric:
        return True
    if submitted_metric == tol_metric:
        return True
    if metrics_compatible(submitted_metric, tol_metric):
        return True
    inv_row = taskpack.get("inventory_row") or taskpack
    if is_unified_benchmark_row(inv_row):
        if normalize_l2_metric_name(submitted_metric) == normalize_l2_metric_name(tol_metric):
            return True
        if {submitted_metric, tol_metric} <= {"substrate_coverage", "log_mae"}:
            return True
    return False


def check_artifact_eq(
    taskpack: Mapping[str, Any],
    final_artifact: Mapping[str, Any],
    *,
    reference_view: Optional[Mapping[str, Any]] = None,
) -> ArtifactEqResult:
    """
    E_q(y)=1 iff canonical output contract holds and native metric passes tolerance.

    Route-neutral: does not require matching selected_route_id or a single gold path.
    """
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
    final_spec = ref_view.get("final_output_spec") or {}
    errors = _validate_canonical_output(final_artifact, final_spec)

    if errors:
        return ArtifactEqResult(
            eq=False,
            metric_name=str(final_spec.get("metric") or "unknown"),
            score=float("nan"),
            threshold_met=False,
            errors=errors,
        )

    value = final_artifact.get("value") or {}
    tol_metric = str((ref_view.get("tolerance") or {}).get("metric") or "")
    if isinstance(value, Mapping):
        submitted_metric = str(value.get("metric_name") or "")
        if value.get("vendor_final_answer_eval_v1"):
            if value.get("failure_class") or submitted_metric in {"", "execution_failed"}:
                return ArtifactEqResult(
                    eq=False,
                    metric_name=tol_metric or submitted_metric or "unknown",
                    score=float("nan"),
                    threshold_met=False,
                    errors=["no_vendor_final_answer"],
                )
        if tol_metric and submitted_metric and not _artifact_metric_names_compatible(
            submitted_metric, tol_metric, taskpack
        ):
            return ArtifactEqResult(
                eq=False,
                metric_name=tol_metric,
                score=float("nan"),
                threshold_met=False,
                errors=[f"metric_mismatch:{submitted_metric}!={tol_metric}"],
            )
        if tol_metric and not submitted_metric:
            wrong_key = next(
                (
                    k
                    for k in value.keys()
                    if k.endswith("_f1")
                    or k.endswith("_score")
                    or k in {"pick_f1", "association_score", "gmpe_coverage", "auprc", "brier"}
                ),
                None,
            )
            if wrong_key and wrong_key != tol_metric and tol_metric not in value:
                return ArtifactEqResult(
                    eq=False,
                    metric_name=tol_metric,
                    score=float("nan"),
                    threshold_met=False,
                    errors=[f"metric_mismatch:missing_{tol_metric},found_{wrong_key}"],
                )

    try:
        scored = score_artifact(taskpack, final_artifact, reference_view=ref_view)
    except (ValueError, TypeError) as exc:
        return ArtifactEqResult(
            eq=False,
            metric_name=str(final_spec.get("metric") or "unknown"),
            score=float("nan"),
            threshold_met=False,
            errors=[f"scoring_error:{exc}"],
        )

    metric_name = str(scored["metric_name"])
    score = float(scored["score"])

    # For IoU fixture: score is the submitted reference_score directly when higher is better
    if metric_name == "iou" and isinstance(final_artifact.get("value"), Mapping):
        submitted = final_artifact["value"].get("reference_score")
        if submitted is not None:
            score = float(submitted)

    # For MAE: use absolute error as score when tolerance uses max_abs_error
    if metric_name == "mae" and isinstance(final_artifact.get("value"), Mapping):
        submitted = final_artifact["value"].get("reference_score")
        ref_score = float((ref_view.get("outputs") or {}).get("reference_score", 0))
        if submitted is not None:
            score = abs(float(submitted) - ref_score)

    threshold_met = passes_tolerance(taskpack, score, metric_name)
    if metric_name == "iou":
        threshold_met = passes_tolerance(taskpack, score, metric_name)
    elif metric_name == "mae":
        tol = ref_view.get("tolerance") or {}
        submitted = (final_artifact.get("value") or {}).get("reference_score")
        if submitted is not None and "max_abs_error" in tol:
            ref_score = float((ref_view.get("outputs") or {}).get("reference_score", 0))
            threshold_met = abs(float(submitted) - ref_score) <= float(tol["max_abs_error"])
    elif metric_name == "rmse_depth":
        value = final_artifact.get("value") or {}
        if isinstance(value, Mapping):
            threshold_met = passes_fl2_tolerance(taskpack, value)
    elif metric_name == "accuracy":
        tol = ref_view.get("tolerance") or {}
        value = final_artifact.get("value") or {}
        submitted = value.get("reference_label")
        ref_acc = (ref_view.get("outputs") or {}).get("reference_label")
        if submitted is not None and ref_acc is not None:
            score = abs(float(submitted) - float(ref_acc))
            if "max_abs_error" in tol:
                threshold_met = score <= float(tol["max_abs_error"])
            else:
                # Tabular baseline slack aligned with MAE tasks (2× distance from perfect).
                slack = 2.0 * max(1.0 - float(ref_acc), 0.0)
                threshold_met = score <= slack + 1e-9
            eq = len(errors) == 0 and threshold_met
            return ArtifactEqResult(
                eq=eq,
                metric_name=metric_name,
                score=score,
                threshold_met=threshold_met,
                errors=errors,
                route_neutral=True,
            )

    eq = len(errors) == 0 and threshold_met
    return ArtifactEqResult(
        eq=eq,
        metric_name=metric_name,
        score=score,
        threshold_met=threshold_met,
        errors=errors,
        route_neutral=True,
    )
