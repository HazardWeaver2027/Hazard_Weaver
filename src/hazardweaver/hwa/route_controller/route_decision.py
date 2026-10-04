"""RouteDecisionPacket + RouteIntent ()."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwa.route_controller.pareto_prune import frontier_only


def build_route_decision_packet(
    routes: Sequence[Mapping[str, Any]],
    *,
    task_id: str = "",
    checkpoint: str = "",
) -> Dict[str, Any]:
    """Structured packet for LLM route selection within Pareto frontier."""
    frontier = frontier_only(routes)
    candidates = []
    for r in frontier:
        candidates.append({
            "route_id": r.get("route_id"),
            "A_sci": r.get("A_sci"),
            "A_cap": r.get("A_cap"),
            "validation_utility": r.get("validation_utility"),
            "estimated_cost": r.get("estimated_cost"),
            "pareto_dominated": False,
            "capability_ids": r.get("capability_ids") or r.get("edges"),
        })
    return {
        "schema_version": "HWA_ROUTE_DECISION_PACKET_v1",
        "task_id": task_id,
        "checkpoint": checkpoint,
        "candidates": candidates,
        "pareto_frontier_ids": [str(c["route_id"]) for c in candidates if c.get("route_id")],
        "n_candidates": len(candidates),
    }


def parse_route_intent(intent: Mapping[str, Any]) -> Optional[str]:
    """Extract chosen route_id from structured RouteIntent."""
    rid = intent.get("route_id") or intent.get("chosen_route_id")
    if rid:
        return str(rid)
    choice = intent.get("choice")
    if isinstance(choice, Mapping) and choice.get("route_id"):
        return str(choice["route_id"])
    return None
