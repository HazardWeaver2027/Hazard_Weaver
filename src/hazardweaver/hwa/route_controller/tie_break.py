"""TIE state — REMSA soft-rank then deterministic tie-break."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwa.route_controller.remsa_rank import remsa_soft_rank


def tie_break_routes(
    routes: Sequence[Mapping[str, Any]],
    *,
    task: Optional[Mapping[str, Any]] = None,
    state: Any = None,
    committed_route: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Pick top REMSA-ranked admissible route."""
    ranked = remsa_soft_rank(
        routes, task=task, state=state, committed_route=committed_route
    )
    return ranked[0] if ranked else None


def needs_clarification(routes: Sequence[Mapping[str, Any]]) -> bool:
    """True when multiple admissible routes tie on utility vector."""
    adm = [r for r in routes if r.get("admissible")]
    if len(adm) <= 1:
        return False
    utils = set()
    for r in adm:
        key = (
            r.get("validation_utility"),
            r.get("estimated_cost"),
            str((r.get("A_sci") or {}).get("verdict")),
            str((r.get("A_cap") or {}).get("verdict")),
        )
        utils.add(key)
    return len(adm) > 1 and len(utils) == 1
