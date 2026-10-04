"""Execution certificates for non-HCG run_capability dispatches (g2 / registry replay)."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hcg.contracts.execution_event import CapabilityExecutionCertificate, ExecutionEvent


def _metric_payload(raw: Mapping[str, Any]) -> Dict[str, Any]:
    inner = raw.get("result") if isinstance(raw.get("result"), Mapping) else raw
    metrics = inner.get("metrics") if isinstance(inner.get("metrics"), Mapping) else {}
    prediction = inner.get("prediction") if isinstance(inner.get("prediction"), Mapping) else {}
    if not metrics and prediction:
        metrics = prediction
    if not metrics:
        return {}
    metric_name = str(metrics.get("metric_name") or prediction.get("metric_name") or "mae")
    score = metrics.get("val_metric")
    if score is None:
        score = metrics.get("validation_metric")
    if score is None:
        score = prediction.get("reported_val_metric")
    if score is None:
        score = metrics.get("reference_score")
    if score is None:
        return {}
    return {
        "metric_name": metric_name,
        "val_metric": float(score),
        "reference_score": float(score),
    }


def build_legacy_execution_bundle(
    *,
    capability_id: str,
    dispatch: str,
    payload: Mapping[str, Any],
    ok: bool = True,
    execution_id: str = "",
    lease_id: str = "",
    route_id: str = "",
) -> Dict[str, Any]:
    """Mint execution_id + certificates from a successful legacy run_capability result."""
    eid = str(execution_id or payload.get("execution_id") or f"E_legacy_{uuid.uuid4().hex[:12]}")
    metrics = _metric_payload(payload)
    event = ExecutionEvent(
        execution_id=eid,
        capability_id=str(capability_id),
        status="completed" if ok else "failed",
        produced_artifacts={"output": {"metrics": metrics}} if metrics else {},
        contract_checks=[{"check": "legacy_dispatch", "passed": bool(ok)}],
        provenance={
            "dispatch": dispatch,
            "route_id": route_id or payload.get("route_id"),
            "smoke_mode": False,
        },
        lease_id=str(lease_id or payload.get("lease_id") or ""),
    )
    cert = CapabilityExecutionCertificate(
        execution_id=eid,
        capability_id=str(capability_id),
        lease_id=event.lease_id,
        route_id=str(route_id or payload.get("route_id") or ""),
        ok=bool(ok),
        contract_checks=event.contract_checks,
        adapters_explicit=True,
        inputs_from_allowed_state=True,
    )
    return {
        "execution_id": eid,
        "execution_event": event.model_dump(mode="json"),
        "execution_certificate": cert.model_dump(mode="json"),
    }


def record_legacy_run_capability_ledger(
    workdir: Any,
    *,
    capability_id: str,
    dispatch: str,
    payload: Mapping[str, Any],
    ok: bool = True,
    lease_id: str = "",
    route_id: str = "",
) -> None:
    """Append native ledger step with certificates for react / g2 replay paths."""
    from pathlib import Path

    from hazardweaver.hwa.runtime.trajectory_ledger import record_tool_execution

    bundle = build_legacy_execution_bundle(
        capability_id=capability_id,
        dispatch=dispatch,
        payload=payload,
        ok=ok,
        execution_id=str(payload.get("execution_id") or ""),
        lease_id=lease_id,
        route_id=route_id,
    )
    record_tool_execution(
        Path(workdir),
        tool="run_capability",
        capability_id=str(capability_id),
        ok=bool(ok),
        execution_id=bundle["execution_id"],
        lease_id=lease_id or None,
        route_id=route_id or None,
        execution_certificate=bundle["execution_certificate"],
        execution_event=bundle["execution_event"],
        error=str(payload.get("error") or ""),
    )
