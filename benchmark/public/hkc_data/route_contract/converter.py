"""RouteCard → Scientific Route Contract converter (DL-061)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .qualification import load_policy, predicate_from_route_card_slot, qualify_predicates
from .schema import COMPILER_VERSION, ScientificRouteContract


def _all_admitted_slots(route_card: Mapping[str, Any]) -> List[Dict[str, Any]]:
    buckets = (
        "required_inputs",
        "assumptions",
        "applicability_conditions",
        "valid_scales",
        "mechanisms",
        "uses_equations",
        "calibration_thresholds",
        "uncertainty",
        "failure_conditions",
        "provenance_slots",
    )
    out: List[Dict[str, Any]] = []
    for b in buckets:
        for slot in route_card.get(b) or []:
            if slot.get("admitted", True):
                out.append(dict(slot))
    return out


def compile_contract_from_route_card(
    route_card: Mapping[str, Any],
    evidence_text: str = "",
    *,
    policy_path: Optional[str] = None,
) -> ScientificRouteContract:
    from pathlib import Path

    policy = load_policy(Path(policy_path) if policy_path else None)
    binding = dict(route_card.get("route_binding") or {})
    paper_id = str(route_card.get("paper_id") or "")
    family_id = str(binding.get("family_id") or "")
    contract_id = f"src_{family_id}_{route_card.get('item_id') or route_card.get('route_card_id')}"

    route_binding = {
        "hazard_id": str(binding.get("task_id") or "").split("-")[0].lower(),
        "task_id": str(binding.get("task_id") or ""),
        "family_id": family_id,
        "route_id": str(binding.get("route_id") or family_id),
        "description": str(binding.get("description") or ""),
    }

    slots = _all_admitted_slots(route_card)
    predicates = [
        predicate_from_route_card_slot(s, paper_id=paper_id, index=i, policy=policy)
        for i, s in enumerate(slots)
    ]
    scores = [float(s.get("confidence") or 0.5) for s in slots]

    strict, advisory, qual_meta = qualify_predicates(
        predicates,
        evidence_text,
        policy=policy,
        heuristic_scores=scores,
    )

    unresolved = [
        p.predicate_id
        for p in predicates
        if p.promotion_stage not in ("strict", "advisory") and p not in strict and p not in advisory
    ]

    admission = route_card.get("admission_summary") or {}
    coverage = {
        "n_route_card_admitted": admission.get("n_admitted"),
        "n_strict": len(strict),
        "n_advisory": len(advisory),
        "n_unresolved": len(unresolved),
        **qual_meta,
    }

    return ScientificRouteContract(
        contract_id=contract_id,
        route_binding=route_binding,
        strict_predicates=strict,
        advisory_predicates=advisory,
        coverage=coverage,
        unresolved=unresolved,
        provenance=dict(route_card.get("provenance") or {}),
        compiler_metadata={
            "compiler_version": COMPILER_VERSION,
            "source_route_card_id": route_card.get("route_card_id"),
            "policy_id": policy.get("policy_id"),
        },
    )
