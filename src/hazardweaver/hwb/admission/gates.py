"""HWB-Core twelve-gate admission pipeline ()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from hazardweaver.hwb.build.witness_solvability import check_witness_solvability
from hazardweaver.hwb.evaluators.leakage_audit import audit_taskpack_leakage

GATE_NAMES = [
    "scientific_realism",
    "decision_time_validity",
    "independent_reference",
    "solvability",
    "reproducibility",
    "route_multiplicity",
    "route_neutrality",
    "leakage",
    "causality",
    "holdout",
    "evaluator_determinism",
    "no_benchmark_conditioned_tuning",
]


@dataclass
class GateResult:
    gate: str
    passed: bool
    reason: str = ""


@dataclass
class AdmissionReport:
    instance_id: str
    core_eligible: bool
    gates: List[GateResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "core_eligible": self.core_eligible,
            "gates": [{"gate": g.gate, "passed": g.passed, "reason": g.reason} for g in self.gates],
        }


def _is_fixture_only(candidate: Mapping[str, Any]) -> bool:
    meta = candidate.get("metadata") or {}
    if meta.get("fixture"):
        return True
    tier = str(candidate.get("tier") or candidate.get("coverage_tier") or "")
    if tier == "legacy" and not candidate.get("decision_time"):
        return True
    return False


def evaluate_gates(candidate: Mapping[str, Any], *, taskpack: Optional[Mapping[str, Any]] = None) -> AdmissionReport:
    """Evaluate machine-checkable subset of HWB-Core admission gates."""
    instance_id = str(candidate.get("instance_id") or candidate.get("scenario_id") or "unknown")
    gates: List[GateResult] = []

    decision_time = candidate.get("decision_time") or (candidate.get("cutoff_manifest") or {}).get("decision_time")
    gates.append(
        GateResult(
            "scientific_realism",
            not _is_fixture_only(candidate),
            "fixture-only instances are Atlas, not Core",
        )
    )
    gates.append(
        GateResult(
            "decision_time_validity",
            bool(decision_time),
            "decision_time or cutoff_manifest.decision_time required",
        )
    )
    gates.append(
        GateResult(
            "independent_reference",
            bool(candidate.get("evaluator_bundle_ref") or (taskpack and taskpack.get("reference_view"))),
            "hidden evaluator reference required",
        )
    )
    if taskpack:
        solv = check_witness_solvability(taskpack)
        gates.append(GateResult("solvability", solv.passed, solv.reason))
    else:
        gates.append(GateResult("solvability", False, "no taskpack"))

    gates.append(
        GateResult(
            "reproducibility",
            bool(candidate.get("materialized", True)),
            "scenario must be materialized on disk",
        )
    )
    witnesses = []
    if taskpack:
        witnesses = (taskpack.get("reference_view") or {}).get("accepted_witnesses") or []
    gates.append(
        GateResult(
            "route_multiplicity",
            len(witnesses) >= 2 or str((taskpack or {}).get("reference_view", {}).get("expected_action")) in ("abstain", "clarify"),
            "≥2 witness routes or abstention/clarify task",
        )
    )
    gates.append(
        GateResult(
            "route_neutrality",
            True,
            "property-based V_q (no witness-edge pass criterion)",
        )
    )
    if taskpack:
        leak = audit_taskpack_leakage(taskpack)
        gates.append(GateResult("leakage", leak.critical_count == 0, f"critical={leak.critical_count}"))
    else:
        gates.append(GateResult("leakage", False, "no taskpack for leakage audit"))

    cutoff = candidate.get("cutoff_manifest") or {}
    gates.append(
        GateResult(
            "causality",
            bool(cutoff.get("decision_time") or cutoff.get("issue_yyyymm")),
            "cutoff_manifest must declare decision boundary",
        )
    )
    holdout_seal = candidate.get("holdout_seal") or {}
    gates.append(
        GateResult(
            "holdout",
            holdout_seal.get("status") == "VERIFIED" or candidate.get("split_role") != "hwb_holdout",
            "holdout requires VERIFIED seal or non-holdout split",
        )
    )
    gates.append(
        GateResult(
            "evaluator_determinism",
            bool(taskpack and (taskpack.get("reference_view") or {}).get("tolerance")),
            "frozen tolerance/checker on taskpack",
        )
    )
    gates.append(
        GateResult(
            "no_benchmark_conditioned_tuning",
            not candidate.get("used_in_tuning", False),
            "instance must not be used for system tuning",
        )
    )

    core_eligible = all(g.passed for g in gates)
    return AdmissionReport(instance_id=instance_id, core_eligible=core_eligible, gates=gates)
