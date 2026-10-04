"""Scientific Route Contract schema (DL-061 fused)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "scientific_route_contract/v1"
COMPILER_VERSION = "route_contract_compiler_v1"

SLOT_TO_PREDICATE_TYPE: Dict[str, str] = {
    "RequiresInput": "REQUIRES_INPUT",
    "ApplicableUnder": "APPLICABLE_UNDER",
    "ValidAtScale": "VALID_AT_SCALE",
    "CalibrationCondition": "CALIBRATION_CONDITION",
    "FailureCondition": "FAILS_UNDER",
    "Assumption": "ASSUMPTION",
    "Mechanism": "MECHANISM",
    "Uncertainty": "UNCERTAINTY",
    "UsesEquation": "USES_EQUATION",
    "Provenance": "PROVENANCE",
}

STRICT_DEFAULT_TYPES = frozenset(
    {"REQUIRES_INPUT", "APPLICABLE_UNDER", "VALID_AT_SCALE", "CALIBRATION_CONDITION"}
)
ADVISORY_DEFAULT_TYPES = frozenset(
    {"ASSUMPTION", "MECHANISM", "UNCERTAINTY", "USES_EQUATION", "PROVENANCE"}
)


@dataclass
class EvidenceSpan:
    text: str
    start: int
    end: int
    paper_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": int(self.start),
            "end": int(self.end),
            "paper_id": self.paper_id,
        }


@dataclass
class Predicate:
    predicate_id: str
    predicate_type: str
    canonical_variable: str
    relation: str = "required"
    operator: str = "present"
    expected_value: Any = True
    unit: Optional[str] = None
    scope: Dict[str, Any] = field(default_factory=dict)
    evidence: List[EvidenceSpan] = field(default_factory=list)
    support: Dict[str, Any] = field(default_factory=dict)
    gate_role: str = "advisory"
    promotion_stage: str = "candidate"
    qualification_checks: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate_id": self.predicate_id,
            "predicate_type": self.predicate_type,
            "canonical_variable": self.canonical_variable,
            "relation": self.relation,
            "operator": self.operator,
            "expected_value": self.expected_value,
            "unit": self.unit,
            "scope": dict(self.scope),
            "evidence": [e.to_dict() for e in self.evidence],
            "support": dict(self.support),
            "gate_role": self.gate_role,
            "promotion_stage": self.promotion_stage,
            "qualification_checks": dict(self.qualification_checks),
        }


@dataclass
class ScientificRouteContract:
    contract_id: str
    route_binding: Dict[str, Any]
    strict_predicates: List[Predicate] = field(default_factory=list)
    advisory_predicates: List[Predicate] = field(default_factory=list)
    coverage: Dict[str, Any] = field(default_factory=dict)
    unresolved: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    compiler_metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "route_binding": dict(self.route_binding),
            "strict_predicates": [p.to_dict() for p in self.strict_predicates],
            "advisory_predicates": [p.to_dict() for p in self.advisory_predicates],
            "coverage": dict(self.coverage),
            "unresolved": list(self.unresolved),
            "provenance": dict(self.provenance),
            "compiler_metadata": dict(self.compiler_metadata),
        }
