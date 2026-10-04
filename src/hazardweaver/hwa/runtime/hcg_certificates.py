"""Map HCG CapabilityReachabilityCertificate / ExecutionCertificate for HWA routes."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from hazardweaver.hcg.contracts.reachability_certificate import (
    CapabilityReachabilityCertificate,
    ReachabilityVerdict,
)
from hazardweaver.hwa.scientific_controller.reason_codes import ACapVerdict


_VERDICT_MAP = {
    ReachabilityVerdict.REACHABLE.value: ACapVerdict.REACHABLE.value,
    ReachabilityVerdict.UNRESOLVED.value: ACapVerdict.UNRESOLVED.value,
    ReachabilityVerdict.UNREACHABLE.value: ACapVerdict.UNREACHABLE_MISSING_ARTIFACT.value,
}


def reachability_cert_to_acap(cert: CapabilityReachabilityCertificate) -> Dict[str, Any]:
    """Derive legacy A_cap dict view from authoritative HCG certificate."""
    verdict_raw = cert.verdict.value if hasattr(cert.verdict, "value") else str(cert.verdict)
    codes = [str(b.get("code") or b) for b in (cert.blockers or [])]
    for chk in cert.runtime_checks or []:
        if chk.get("status") == "unresolved":
            code = str(chk.get("code") or chk.get("probe") or "RUNTIME_UNRESOLVED")
            if code not in codes:
                codes.append(code)
    missing = [
        str(b.get("artifact_id"))
        for b in (cert.blockers or [])
        if str(b.get("code") or "").upper() in {"MISSING_ARTIFACT", "MISSING_CAPABILITY"}
        and b.get("artifact_id")
    ]
    return {
        "verdict": _VERDICT_MAP.get(verdict_raw, ACapVerdict.UNRESOLVED.value),
        "codes": codes,
        "missing_artifacts": missing,
        "hcg_verdict": verdict_raw,
        "certificate_source": "hcg.api.evaluate_reachability",
    }


def attach_reachability_certificate(
    route: Mapping[str, Any],
    cert: CapabilityReachabilityCertificate,
) -> Dict[str, Any]:
    out = dict(route)
    dumped = cert.model_dump(mode="json")
    out["reachability_certificate"] = dumped
    out["A_cap"] = reachability_cert_to_acap(cert)
    return out
