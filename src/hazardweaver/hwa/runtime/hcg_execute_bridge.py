"""Headline capability execution via hcg.api.execute_capability (Phase 4)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from hazardweaver.hcg.api import execute_capability
from hazardweaver.hcg.contracts.execution_event import ExecutionContext, ExecutionEvent, ScopedExecutionLease
from hazardweaver.hcg.registry.canonical_ids import is_headline_capability
from hazardweaver.hcg.runtime.execute_capability import (
    ExecuteCapabilityBundle,
    ExecutionLeaseError,
    execute_capability_bundle,
)


def run_headline_capability(
    capability_id: str,
    input_artifacts: Mapping[str, Any],
    lease: ScopedExecutionLease,
    *,
    execution_id: str = "",
    smoke_mode: bool = False,
    full_certificates: bool = False,
) -> ExecutionEvent | ExecuteCapabilityBundle:
    cid = str(capability_id).strip()
    if not is_headline_capability(cid):
        raise ExecutionLeaseError(f"not_headline_capability:{cid}")
    ctx = ExecutionContext(
        lease=lease,
        execution_id=execution_id or "",
    )
    if full_certificates:
        return execute_capability_bundle(
            cid,
            dict(input_artifacts),
            ctx,
            smoke_mode=smoke_mode,
        )
    return execute_capability(
        cid,
        dict(input_artifacts),
        ctx,
        smoke_mode=smoke_mode,
    )


def event_to_run_capability_result(
    event: ExecutionEvent,
    *,
    capability_id: str,
    bundle: Optional[ExecuteCapabilityBundle] = None,
) -> Dict[str, Any]:
    produced = event.produced_artifacts or {}
    output = produced.get("output") if isinstance(produced, Mapping) else None
    blocked = (
        isinstance(output, Mapping)
        and (output.get("blocked") is True or (output.get("ok") is False and output.get("reason")))
    )
    ok = event.status == "completed" and not blocked
    out: Dict[str, Any] = {
        "ok": ok,
        "capability_id": capability_id,
        "dispatch": "hcg_execute_capability",
        "tool": "run_capability",
        "execution_id": event.execution_id,
        "lease_id": event.lease_id,
        "result": {
            "produced_artifacts": event.produced_artifacts,
            "consumed_artifacts": event.consumed_artifacts,
            "contract_checks": event.contract_checks,
            "provenance": event.provenance,
        },
        "trained_in_tool": False,
    }
    if blocked and isinstance(output, Mapping):
        out["error"] = str(output.get("reason") or "execution_blocked")
        out["blocked"] = True
    if bundle is not None:
        out["execution_event"] = bundle.event.model_dump(mode="json")
        out["execution_certificate"] = bundle.execution_certificate.model_dump(mode="json")
        out["reachability_certificate"] = bundle.reachability_certificate.model_dump(mode="json")
    return out
