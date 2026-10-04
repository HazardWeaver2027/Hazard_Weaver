"""VCE pre-submit gate — delegates to ``build_hwb_submission`` (same path as batch eval)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwb.bridge.hwa_workdir import build_hwb_submission
from hazardweaver.hwb.bridge.submission_metric_v1 import (
    artifact_satisfies_taskpack,
    normalize_submission_value,
    taskpack_tolerance_metric,
    value_has_score,
)
from hazardweaver.hwb.registry.track_parametric_resolver import align_reference_view_to_executed_capability


def _taskpack_for_artifact_metric_check(
    taskpack: Mapping[str, Any],
    *,
    value: Mapping[str, Any],
    inventory_capability_id: str = "",
) -> Mapping[str, Any]:
    """Bind tolerance to the capability that produced the artifact (DL-229).

    Scenario-level taskpacks (e.g. MH-1 shock on CAP-MH1-02) may expect volume_mae
    while the witness route CAP-MH1-01 emits brier. ``dca_scorer`` already aligns via
    ``align_reference_view_to_executed_capability``; submission_prepare must match.
    """
    executed = str(value.get("capability_id") or inventory_capability_id or "").strip()
    if not executed:
        return taskpack
    return align_reference_view_to_executed_capability(taskpack, executed_capability_id=executed)


def _normalize_submission_metric_name(
    value: Dict[str, Any],
    *,
    taskpack: Optional[Mapping[str, Any]] = None,
    inventory_row: Optional[Mapping[str, Any]] = None,  # noqa: ARG001 — compat
) -> Dict[str, Any]:
    return normalize_submission_value(value, taskpack=taskpack)


def prepare_solve_submission(
    *,
    workdir: Path,
    route_id: str,
    execution_id: str,
    final_artifact_id: str,
    inventory_row: Optional[Mapping[str, Any]] = None,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    ready = ensure_execution_submission_ready(
        workdir=workdir,
        route_id=route_id,
        execution_id=execution_id,
        final_artifact_id=final_artifact_id,
        inventory_row=inventory_row,
        taskpack=taskpack,
    )
    if not ready.get("ok"):
        raise ValueError(str(ready.get("error") or "submission_not_ready"))
    return dict(ready["submission"])


def ensure_execution_submission_ready(
    *,
    workdir: Path,
    route_id: str,
    execution_id: str,
    final_artifact_id: str,
    inventory_row: Optional[Mapping[str, Any]] = None,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """VCE VERIFY: build submission exactly like batch eval, then check contract."""
    workdir = Path(workdir)
    eid = str(execution_id).strip()
    aid = str(final_artifact_id).strip()
    cap = str((inventory_row or {}).get("capability_id") or "").strip()

    answer_body: Dict[str, Any] = {
        "action": "solve",
        "execution_id": eid,
        "final_artifact_id": aid,
        "route_id": str(route_id).strip(),
    }
    if cap:
        answer_body["capability_id"] = cap

    try:
        submission = build_hwb_submission(
            workdir,
            answer_body=answer_body,
            taskpack=taskpack,
            route_id=str(route_id).strip(),
        )
    except ValueError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "verify_layer": "submission_prepare",
            "execution_id": eid,
        }

    value = (submission.get("final_artifact") or {}).get("value") or {}
    if not value_has_score(value):
        return {
            "ok": False,
            "error": "final_artifact_missing_score",
            "verify_layer": "submission_prepare",
            "execution_id": eid,
        }
    if taskpack is not None:
        eval_taskpack = _taskpack_for_artifact_metric_check(
            taskpack,
            value=value,
            inventory_capability_id=cap,
        )
        if not artifact_satisfies_taskpack(value, taskpack=eval_taskpack):
            return {
                "ok": False,
                "error": "task_metric_mismatch",
                "verify_layer": "submission_prepare",
                "expected_metric": taskpack_tolerance_metric(eval_taskpack),
                "got_metric": value.get("metric_name"),
                "execution_id": eid,
            }

    return {
        "ok": True,
        "verify_layer": "submission_prepare",
        "submission": submission,
        "route_id": str(route_id).strip(),
        "capability_id": cap or str(value.get("capability_id") or ""),
    }
