"""Honest BLOCKED replay helpers for native eval."""

from __future__ import annotations

from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.portfolio import load_portfolio


def _family_id(taskpack_id: str, capability_id: str) -> str:
    port = load_portfolio(taskpack_id)
    cap = port.capability_by_id(capability_id)
    return (cap.family_id if cap else "unknown") or "unknown"


def write_blocked(
    taskpack_id: str,
    capability_id: str,
    reason: str,
    *,
    out_base=None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    write_metrics(
        taskpack_id,
        capability_id,
        {
            "metric_name": "blocked",
            "metric_value": None,
            "blocked": True,
            "validation_tier": "blocked",
            "note": reason,
        },
        base=out_base,
    )
    write_replay_manifest(
        taskpack_id,
        capability_id,
        family_id=_family_id(taskpack_id, capability_id),
        exec_ok=False,
        notes=f"BLOCKED: {reason}",
        extra=extra,
        base=out_base,
    )
    return {"ok": False, "blocked": True, "reason": reason}


def write_external_acquisition_required(
    taskpack_id: str,
    capability_id: str,
    reason: str,
    *,
    acquisition_plan: Optional[Dict[str, Any]] = None,
    out_base=None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = dict(acquisition_plan or {})
    payload.setdefault("status", "EXTERNAL_ACQUISITION_REQUIRED")
    merged_extra = {"acquisition_plan": payload, **(extra or {})}
    write_metrics(
        taskpack_id,
        capability_id,
        {
            "metric_name": "external_acquisition",
            "metric_value": None,
            "blocked": True,
            "validation_tier": "blocked",
            "external_acquisition_required": True,
            "note": reason,
        },
        base=out_base,
    )
    write_replay_manifest(
        taskpack_id,
        capability_id,
        family_id=_family_id(taskpack_id, capability_id),
        exec_ok=False,
        notes=f"EXTERNAL_ACQUISITION_REQUIRED: {reason}",
        extra=merged_extra,
        base=out_base,
    )
    return {
        "ok": False,
        "blocked": True,
        "external_acquisition_required": True,
        "acquisition_status": "EXTERNAL_ACQUISITION_REQUIRED",
        "reason": reason,
        "acquisition_plan": payload,
    }


def eval_blocked_generic(
    taskpack_id: str,
    capability_id: str,
    reason: str,
    *,
    out_base=None,
) -> Dict[str, Any]:
    return write_blocked(taskpack_id, capability_id, reason, out_base=out_base)
