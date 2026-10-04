"""Independent verification of HCG execution certificates for V_cap/V_prov ().

Certificate is evidence, not ground truth — HWB re-checks underlying facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set

from hazardweaver.hwb.evaluators.causality_firewall import check_v_causal


@dataclass
class CertificateVerifyResult:
    passed: bool
    errors: List[str] = field(default_factory=list)

    def extend(self, other: "CertificateVerifyResult") -> None:
        self.errors.extend(other.errors)
        self.passed = self.passed and other.passed


def _cert_index(trajectory: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    index: Dict[str, Mapping[str, Any]] = {}
    for cert in trajectory.get("execution_certificates") or []:
        if isinstance(cert, Mapping) and cert.get("execution_id"):
            index[str(cert["execution_id"])] = cert
    return index


def verify_execution_certificate(
    cert: Mapping[str, Any],
    *,
    step: Mapping[str, Any],
    allowed_capabilities: Set[str],
    cutoff_manifest: Optional[Mapping[str, Any]] = None,
    lease_allowed: Optional[Set[str]] = None,
) -> CertificateVerifyResult:
    """
    Six independent checks (Fusion / ):
    execution_id exists; capability allowed; contract_checks valid;
    adapters explicit; inputs from allowed state; timestamps legal.
    """
    errors: List[str] = []
    eid = str(cert.get("execution_id") or "")
    if not eid:
        errors.append("certificate:missing_execution_id")
    step_eid = str(step.get("execution_id") or "")
    if step_eid and eid and step_eid != eid:
        errors.append("certificate:execution_id_mismatch")

    cap = str(cert.get("capability_id") or step.get("capability_id") or "")
    if cap and allowed_capabilities and cap not in allowed_capabilities:
        errors.append(f"certificate:capability_not_allowed:{cap}")
    if lease_allowed and cap and cap not in lease_allowed:
        errors.append(f"certificate:capability_not_in_lease:{cap}")

    if cert.get("ok") is False:
        errors.append("certificate:verdict_ok_false")

    checks = cert.get("contract_checks") or []
    for chk in checks:
        if isinstance(chk, Mapping) and chk.get("passed") is False:
            errors.append(f"certificate:contract_check_failed:{chk.get('check')}")

    if cert.get("adapters_explicit") is False:
        errors.append("certificate:adapter_not_explicit")
    if cert.get("inputs_from_allowed_state") is False:
        errors.append("certificate:input_lineage_invalid")

    event_raw = cert.get("execution_event")
    if isinstance(event_raw, Mapping) and event_raw:
        try:
            from pydantic import ValidationError

            from hazardweaver.hcg.contracts.execution_event import ExecutionEvent

            event = ExecutionEvent.model_validate(dict(event_raw))
            if cert.get("lease_id") and event.lease_id and str(cert["lease_id"]) != event.lease_id:
                errors.append("certificate:lease_id_mismatch")
            if cap and event.capability_id and cap != event.capability_id:
                errors.append("certificate:capability_id_mismatch")
            if event.status and str(event.status).lower() in {"failed", "error"}:
                errors.append("certificate:event_status_failed")
            prov = event.provenance or {}
            if isinstance(prov, Mapping):
                t_avail = prov.get("t_available")
                t_dec = prov.get("decision_time") or prov.get("t_decision")
                if t_avail is not None and t_dec is not None and str(t_avail) > str(t_dec):
                    errors.append("certificate:timestamps_illegal")
        except ValidationError:
            errors.append("certificate:invalid_execution_event")

    content_hash = cert.get("content_hash")
    if content_hash:
        from hazardweaver.hcg.certification.certificate_hash import compute_certificate_content_hash

        recomputed = compute_certificate_content_hash(cert)
        if recomputed != content_hash:
            errors.append("certificate:content_hash_mismatch")

    event = cert.get("execution_event") or {}
    if isinstance(event, Mapping):
        prov = event.get("provenance") if isinstance(event, Mapping) else {}
    else:
        prov = {}
    if isinstance(prov, Mapping) and cutoff_manifest:
        causal = check_v_causal(
            [{"step_index": step.get("step_index"), "provenance": prov}],
            cutoff_manifest=cutoff_manifest,
        )
        errors.extend(causal)

    produced = event.get("produced_artifacts") if isinstance(event, Mapping) else None
    if isinstance(produced, Mapping):
        for _aid, handle in produced.items():
            if isinstance(handle, Mapping) and not handle.get("hash") and handle.get("require_hash"):
                errors.append("certificate:missing_artifact_hash")

    return CertificateVerifyResult(passed=len(errors) == 0, errors=errors)


def _scored_execution_id(trajectory: Mapping[str, Any]) -> Optional[str]:
    """Execution id credited by a solve submission, when present."""
    fa = trajectory.get("final_artifact") or {}
    if isinstance(fa, Mapping):
        eid = fa.get("execution_id")
        if eid:
            return str(eid).strip()
    body = trajectory.get("answer") or {}
    if isinstance(body, Mapping):
        eid = body.get("execution_id")
        if eid:
            return str(eid).strip()
    return None


def verify_trajectory_certificates(
    taskpack: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    cutoff_manifest: Optional[Mapping[str, Any]] = None,
    scored_execution_id: Optional[str] = None,
) -> CertificateVerifyResult:
    """Verify step-linked execution certificates on a trajectory.

    When ``scored_execution_id`` is set (or inferred from ``final_artifact``),
    only that execution's certificate is verified. Intermediate failed attempts
    in a long agent loop must not void an otherwise valid scored submission.
    """
    allowed = {str(e) for e in (taskpack.get("solver_view") or {}).get("allowed_edge_ids") or []}
    lease_caps: Optional[Set[str]] = None
    lease = trajectory.get("execution_lease") or {}
    if isinstance(lease, Mapping) and lease.get("allowed_capability_ids"):
        lease_caps = {str(c) for c in lease["allowed_capability_ids"]}

    cutoff = cutoff_manifest or trajectory.get("cutoff_manifest") or taskpack.get("cutoff_manifest") or {}
    cert_by_id = _cert_index(trajectory)
    aggregate = CertificateVerifyResult(passed=True)

    scored_eid = str(scored_execution_id or _scored_execution_id(trajectory) or "").strip() or None

    steps = [s for s in (trajectory.get("steps") or []) if isinstance(s, Mapping)]
    for step in steps:
        if step.get("kind") == "submit":
            continue
        cert = step.get("execution_certificate")
        if not isinstance(cert, Mapping):
            eid = step.get("execution_id")
            if eid and str(eid) in cert_by_id:
                cert = cert_by_id[str(eid)]
        if not isinstance(cert, Mapping):
            requires_cert = (
                str(trajectory.get("schema_version") or "") == "HWA_TRAJECTORY_v2"
                or step.get("execution_id")
                or cert_by_id
            )
            if requires_cert and step.get("capability_id"):
                step_eid = str(step.get("execution_id") or "").strip()
                if scored_eid and step_eid and step_eid != scored_eid:
                    continue
                aggregate.errors.append(f"certificate:missing_for_step:{step.get('step_index')}")
                aggregate.passed = False
            continue
        cert_eid = str(cert.get("execution_id") or step.get("execution_id") or "").strip()
        if scored_eid and cert_eid and cert_eid != scored_eid:
            continue
        result = verify_execution_certificate(
            cert,
            step=step,
            allowed_capabilities=allowed,
            cutoff_manifest=cutoff,
            lease_allowed=lease_caps,
        )
        aggregate.extend(result)

    return aggregate
