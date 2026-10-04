"""Tri-state A_sci matcher + Scientific Admission Certificate (DL-061)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from .schema import Predicate, ScientificRouteContract


class MatchState(str, Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNRESOLVED = "UNRESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ASciVerdict(str, Enum):
    ADMIT = "ADMIT"
    REJECT = "REJECT"
    UNKNOWN = "UNKNOWN"


REASON_MISSING_CONTEXT = "MISSING_TASK_CONTEXT"
REASON_VIOLATED_PREDICATE = "VIOLATED_STRICT_PREDICATE"
REASON_UNRESOLVED_PREDICATE = "UNRESOLVED_STRICT_PREDICATE"


def _task_constraints(task: Mapping[str, Any]) -> Set[str]:
    sv = task.get("solver_visible") or {}
    inputs = sv.get("inputs") or {}
    raw = inputs.get("constraints") or []
    return {str(c).strip().lower() for c in raw if str(c).strip()}


def match_predicate_to_task(pred: Predicate, task: Mapping[str, Any]) -> MatchState:
    """Deterministic matcher — no runtime LLM (ChatGPT §7)."""
    if pred.gate_role != "hard":
        return MatchState.NOT_APPLICABLE

    constraints = _task_constraints(task)
    canonical = str(pred.canonical_variable or "").strip().lower()
    require_key = f"require:{canonical}"

    if pred.predicate_type == "REQUIRES_INPUT":
        if not constraints:
            return MatchState.UNRESOLVED
        if require_key in constraints or canonical in constraints:
            return MatchState.SATISFIED
        if any(require_key in c or canonical in c for c in constraints):
            return MatchState.SATISFIED
        return MatchState.VIOLATED

    if pred.predicate_type in {"APPLICABLE_UNDER", "VALID_AT_SCALE", "CALIBRATION_CONDITION"}:
        if constraints:
            return MatchState.SATISFIED
        return MatchState.UNRESOLVED

    if pred.predicate_type == "FAILS_UNDER":
        return MatchState.NOT_APPLICABLE

    return MatchState.UNRESOLVED


@dataclass
class ScientificAdmissionCertificate:
    certificate_id: str
    task_id: str
    route_id: str
    family_id: str
    verdict: str
    satisfied_predicates: List[str] = field(default_factory=list)
    violated_predicates: List[str] = field(default_factory=list)
    unresolved_predicates: List[str] = field(default_factory=list)
    advisory_evidence: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "task_id": self.task_id,
            "route_id": self.route_id,
            "family_id": self.family_id,
            "verdict": self.verdict,
            "satisfied_predicates": list(self.satisfied_predicates),
            "violated_predicates": list(self.violated_predicates),
            "unresolved_predicates": list(self.unresolved_predicates),
            "advisory_evidence": list(self.advisory_evidence),
            "evidence_refs": list(self.evidence_refs),
            "reason_codes": list(self.reason_codes),
        }


def evaluate_contract(
    contract: ScientificRouteContract,
    task: Mapping[str, Any],
) -> ScientificAdmissionCertificate:
    satisfied: List[str] = []
    violated: List[str] = []
    unresolved: List[str] = []
    advisory_ids: List[str] = []
    refs: List[str] = []
    reasons: List[str] = []

    for pred in contract.strict_predicates:
        state = match_predicate_to_task(pred, task)
        if state == MatchState.SATISFIED:
            satisfied.append(pred.predicate_id)
        elif state == MatchState.VIOLATED:
            violated.append(pred.predicate_id)
            reasons.append(REASON_VIOLATED_PREDICATE)
        elif state == MatchState.UNRESOLVED:
            unresolved.append(pred.predicate_id)
            reasons.append(REASON_UNRESOLVED_PREDICATE)
        for ev in pred.evidence:
            if ev.paper_id and ev.text:
                refs.append(f"{ev.paper_id}:{ev.start}-{ev.end}")

    for pred in contract.advisory_predicates:
        advisory_ids.append(pred.predicate_id)

    if violated:
        verdict = ASciVerdict.REJECT.value
    elif unresolved:
        verdict = ASciVerdict.UNKNOWN.value
    else:
        verdict = ASciVerdict.ADMIT.value

    cert_id = f"cert_{contract.contract_id}_{task.get('task_id', 'task')}"
    binding = contract.route_binding
    return ScientificAdmissionCertificate(
        certificate_id=cert_id,
        task_id=str(task.get("task_id") or binding.get("task_id") or ""),
        route_id=str(binding.get("route_id") or ""),
        family_id=str(binding.get("family_id") or ""),
        verdict=verdict,
        satisfied_predicates=satisfied,
        violated_predicates=violated,
        unresolved_predicates=unresolved,
        advisory_evidence=advisory_ids,
        evidence_refs=refs,
        reason_codes=sorted(set(reasons)),
    )


def certificate_to_hwa_verdict(cert: ScientificAdmissionCertificate) -> Dict[str, Any]:
    """Map certificate to HWA A_sci dict."""
    from hwa.scientific_controller.reason_codes import ASciVerdict as HWAVerdict

    mapping = {
        ASciVerdict.ADMIT.value: HWAVerdict.APPLICABLE.value,
        ASciVerdict.REJECT.value: "SCI_INAPPLICABLE_THEORY",
        ASciVerdict.UNKNOWN.value: HWAVerdict.UNKNOWN_PENDING_THEORY.value,
    }
    return {
        "verdict": mapping.get(cert.verdict, HWAVerdict.UNKNOWN_PENDING_THEORY.value),
        "codes": list(cert.reason_codes),
        "refs": list(cert.evidence_refs),
        "certificate_id": cert.certificate_id,
        "hkc_strict_satisfied": cert.satisfied_predicates,
        "hkc_strict_violated": cert.violated_predicates,
        "hkc_strict_unresolved": cert.unresolved_predicates,
        "hkc_advisory_evidence": cert.advisory_evidence,
    }
