"""Unify DCA with HKC/HCG mechanism violations (single validity gate)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from hazardweaver.hwa.benchmark.mechanism_metrics_v1 import extract_mechanism_metrics

MECHANISM_DCA_REASON = "scientific_mechanism_violation"


def mechanism_dca_violations(workdir: Path) -> List[str]:
    """Return mechanism flags that must invalidate DCA for this workdir."""
    mech = extract_mechanism_metrics(workdir)
    out: List[str] = []
    if mech.get("scientifically_invalid_route_commitment"):
        out.append("scientifically_invalid_route_commitment")
    if mech.get("invalid_near_miss_binding"):
        out.append("invalid_near_miss_binding")
    return out


def apply_mechanism_veto_to_dca(dca_dict: Mapping[str, Any], workdir: Path) -> Dict[str, Any]:
    """If HKC/HCG mechanism audit fails, DCA is invalid (contract_violated)."""
    if dca_dict.get("skipped") or dca_dict.get("ok") is False:
        return dict(dca_dict)
    violations = mechanism_dca_violations(workdir)
    if not violations:
        return dict(dca_dict)
    out = dict(dca_dict)
    out["valid"] = False
    out["contract_violated"] = True
    out["counted"] = False
    out["dca_score"] = 0.0
    out["outcome"] = "invalid"
    out["reason_code"] = MECHANISM_DCA_REASON
    out["mechanism_violations"] = violations
    return out
