"""Scientific Route Card schema (DL-052 production artifact)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .slot_schema import SLOT_FAMILIES, normalize_slot_type

ROUTE_CARD_SCHEMA_VERSION = "route_card_v1"
COMPILER_VERSION = "route_card_compiler_v1"

# Route-applicability slot families → Route Card JSON fields.
SLOT_TYPE_TO_FIELD: Dict[str, str] = {
    "RequiresInput": "required_inputs",
    "Assumption": "assumptions",
    "ApplicableUnder": "applicability_conditions",
    "ValidAtScale": "valid_scales",
    "Mechanism": "mechanisms",
    "UsesEquation": "uses_equations",
    "CalibrationCondition": "calibration_thresholds",
    "Uncertainty": "uncertainty",
    "FailureCondition": "failure_conditions",
    "Provenance": "provenance_slots",
}

ROUTE_CARD_SLOT_FIELDS = tuple(sorted(set(SLOT_TYPE_TO_FIELD.values())))

ADMISSION_PRECISION_THRESHOLD = 0.90
ADMISSION_PRECISION_STRETCH = 0.95
REQUIRED_SLOT_RECALL_THRESHOLD = 0.90
EVIDENCE_RECALL_THRESHOLD = 0.90
PROVENANCE_CORRECTNESS_THRESHOLD = 0.99

# Slots excluded from production Route Cards (DL-052 diagnostic-only / low admitted precision).
# Authority: ROUTE_CARD_ADMISSION_EVAL_v1 per-slot audit (mini verifier, paper-disjoint test).
PRODUCTION_SLOT_DENYLIST = frozenset(
    {
        "Provenance",  # admitted precision ≈0.00 on test gold
        "Assumption",  # ≈0.29
        "FailureCondition",  # ≈0.22
    }
)


@dataclass(frozen=True)
class EvidenceSpan:
    text: str
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "start": int(self.start), "end": int(self.end)}


@dataclass
class AdmittedSlotRecord:
    slot_type: str
    canonical_value: str
    evidence_span: EvidenceSpan
    confidence: float
    ambiguity_state: str
    provenance: Dict[str, Any]
    admitted: bool = True
    cluster_ids: List[str] = field(default_factory=list)
    verification: str = ""
    admission_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_type": self.slot_type,
            "canonical_value": self.canonical_value,
            "evidence_span": self.evidence_span.to_dict(),
            "confidence": round(float(self.confidence), 4),
            "ambiguity_state": self.ambiguity_state,
            "provenance": dict(self.provenance),
            "admitted": bool(self.admitted),
            "cluster_ids": list(self.cluster_ids),
            "verification": self.verification,
            "admission_reason": self.admission_reason,
        }


@dataclass
class RouteBinding:
    task_id: str
    family_id: str
    route_id: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family_id": self.family_id,
            "route_id": self.route_id,
            "description": self.description,
        }


@dataclass
class RouteCard:
    route_card_id: str
    schema_version: str = ROUTE_CARD_SCHEMA_VERSION
    compiler_version: str = COMPILER_VERSION
    route_binding: RouteBinding = field(default_factory=lambda: RouteBinding("", "", ""))
    paper_id: str = ""
    item_id: str = ""
    evidence_sha256: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    required_inputs: List[AdmittedSlotRecord] = field(default_factory=list)
    assumptions: List[AdmittedSlotRecord] = field(default_factory=list)
    applicability_conditions: List[AdmittedSlotRecord] = field(default_factory=list)
    valid_scales: List[AdmittedSlotRecord] = field(default_factory=list)
    mechanisms: List[AdmittedSlotRecord] = field(default_factory=list)
    uses_equations: List[AdmittedSlotRecord] = field(default_factory=list)
    calibration_thresholds: List[AdmittedSlotRecord] = field(default_factory=list)
    uncertainty: List[AdmittedSlotRecord] = field(default_factory=list)
    failure_conditions: List[AdmittedSlotRecord] = field(default_factory=list)
    provenance_slots: List[AdmittedSlotRecord] = field(default_factory=list)
    optional_directional_edges: List[Dict[str, Any]] = field(default_factory=list)
    admission_summary: Dict[str, Any] = field(default_factory=dict)
    status: str = "compiled"

    def slot_field(self, slot_type: str) -> str:
        return SLOT_TYPE_TO_FIELD.get(normalize_slot_type(slot_type), "")

    def add_slot(self, record: AdmittedSlotRecord) -> None:
        field_name = self.slot_field(record.slot_type)
        if not field_name:
            return
        bucket = getattr(self, field_name)
        bucket.append(record)

    def all_admitted_slots(self) -> List[AdmittedSlotRecord]:
        out: List[AdmittedSlotRecord] = []
        for field_name in ROUTE_CARD_SLOT_FIELDS:
            out.extend(getattr(self, field_name))
        return out

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "route_card_id": self.route_card_id,
            "schema_version": self.schema_version,
            "compiler_version": self.compiler_version,
            "route_binding": self.route_binding.to_dict(),
            "paper_id": self.paper_id,
            "item_id": self.item_id,
            "evidence_sha256": self.evidence_sha256,
            "provenance": dict(self.provenance),
            "optional_directional_edges": list(self.optional_directional_edges),
            "admission_summary": dict(self.admission_summary),
            "status": self.status,
        }
        for field_name in ROUTE_CARD_SLOT_FIELDS:
            bucket = getattr(self, field_name)
            payload[field_name] = [s.to_dict() for s in bucket]
        return payload


def span_offsets(evidence: str, span_text: str) -> Optional[EvidenceSpan]:
    text = str(span_text or "").strip()
    if not text or text not in evidence:
        return None
    start = evidence.index(text)
    return EvidenceSpan(text=text, start=start, end=start + len(text))


def validate_slot_types_for_route_card(slots: Sequence[str]) -> List[str]:
    errors: List[str] = []
    for s in slots:
        norm = normalize_slot_type(s)
        if norm not in SLOT_FAMILIES:
            errors.append(f"invalid_slot:{s}")
        elif norm not in SLOT_TYPE_TO_FIELD:
            errors.append(f"unmapped_slot:{s}")
    return errors


def route_card_from_dict(payload: Mapping[str, Any]) -> RouteCard:
    binding_raw = payload.get("route_binding") or {}
    binding = RouteBinding(
        task_id=str(binding_raw.get("task_id") or ""),
        family_id=str(binding_raw.get("family_id") or ""),
        route_id=str(binding_raw.get("route_id") or binding_raw.get("family_id") or ""),
        description=str(binding_raw.get("description") or ""),
    )
    card = RouteCard(
        route_card_id=str(payload.get("route_card_id") or ""),
        schema_version=str(payload.get("schema_version") or ROUTE_CARD_SCHEMA_VERSION),
        compiler_version=str(payload.get("compiler_version") or COMPILER_VERSION),
        route_binding=binding,
        paper_id=str(payload.get("paper_id") or ""),
        item_id=str(payload.get("item_id") or ""),
        evidence_sha256=str(payload.get("evidence_sha256") or ""),
        provenance=dict(payload.get("provenance") or {}),
        optional_directional_edges=list(payload.get("optional_directional_edges") or []),
        admission_summary=dict(payload.get("admission_summary") or {}),
        status=str(payload.get("status") or "compiled"),
    )
    for slot_type, field_name in SLOT_TYPE_TO_FIELD.items():
        for row in payload.get(field_name) or []:
            span_raw = row.get("evidence_span") or {}
            span = EvidenceSpan(
                text=str(span_raw.get("text") or ""),
                start=int(span_raw.get("start") or 0),
                end=int(span_raw.get("end") or 0),
            )
            card.add_slot(
                AdmittedSlotRecord(
                    slot_type=str(row.get("slot_type") or slot_type),
                    canonical_value=str(row.get("canonical_value") or ""),
                    evidence_span=span,
                    confidence=float(row.get("confidence") or 0.0),
                    ambiguity_state=str(row.get("ambiguity_state") or "unknown"),
                    provenance=dict(row.get("provenance") or {}),
                    admitted=bool(row.get("admitted", True)),
                    cluster_ids=list(row.get("cluster_ids") or []),
                    verification=str(row.get("verification") or ""),
                    admission_reason=str(row.get("admission_reason") or ""),
                )
            )
    return card
