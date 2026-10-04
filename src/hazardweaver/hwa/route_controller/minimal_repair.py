"""Minimal Scientific Repair R0–R6 dispatch ()."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwa.route_controller.adaptation import agent_debug_repair_branch
from hazardweaver.hwa.route_controller.remsa_rank import remsa_soft_rank
from hazardweaver.hwa.runtime.artifact_frontier import frontier_snapshot
from hazardweaver.hwa.runtime.blocker_repair_map import repair_level_for_blocker
from hazardweaver.hwa.scientific_controller.reinstantiate import (
    invalidate_capability,
    reinstantiate_routes,
)


def _blocker_from_reason(reason: str) -> str:
    text = (reason or "").strip().upper()
    for token in (
        "MISSING_ARTIFACT",
        "CHECKPOINT_MISSING",
        "LICENSE_UNAVAILABLE",
        "CAPABILITY_INVALIDATED",
        "VALIDATION_POLICY_UNMET",
        "UNIT_MISMATCH",
    ):
        if token in text:
            return token
    return "CAPABILITY_INVALIDATED"


def dispatch_minimal_repair(
    controller: Any,
    route: Mapping[str, Any],
    *,
    failed_capability_id: str = "",
    reason: str = "",
    blocker_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Map blocker → R0–R6 action; reinstantiate with P_k sources; REMSA-rank survivors."""
    code = blocker_code or _blocker_from_reason(reason)
    level = repair_level_for_blocker(code)
    state = controller.state
    task = controller.task
    store = getattr(controller, "store", None)
    cid = str(failed_capability_id or "").strip()
    edges = list(route.get("edges") or route.get("capability_ids") or [])

    result: Dict[str, Any] = {
        "repair_level": level,
        "blocker_code": code,
        "P_k": frontier_snapshot(state),
    }

    if level == "R0":
        result["action"] = "continue_segment"
        result["ok"] = True
        return result

    if level == "R1":
        invalidate_capability(
            state,
            cid or (edges[0] if edges else "unknown"),
            reason=reason,
            also_invalidate_edges=[cid] if cid else None,
        )
        result["action"] = "same_family_impl_swap"
    elif level == "R2":
        invalidate_capability(
            state,
            cid or (edges[0] if edges else "unknown"),
            reason=reason,
            also_invalidate_edges=edges,
        )
        result["action"] = "suffix_branch_swap"
    elif level in ("R3", "pre_execution_cert_bug"):
        invalidate_capability(
            state,
            cid or (edges[0] if edges else "unknown"),
            reason=reason,
            also_invalidate_edges=edges,
        )
        result["action"] = "family_switch" if level == "R3" else "pre_execution_cert_bug"
    elif level == "R4":
        dbg = agent_debug_repair_branch("MISSING_ARTIFACT", controller)
        result["action"] = "clarify_missing_derivable"
        result["agent_debug"] = dbg
        return result
    elif level == "R5":
        dbg = agent_debug_repair_branch(code, controller)
        result["action"] = "user_recoverable_clarify"
        result["agent_debug"] = dbg
        return result
    else:
        invalidate_capability(
            state,
            cid or (edges[0] if edges else "unknown"),
            reason=reason,
            also_invalidate_edges=edges,
        )
        result["action"] = "reinstantiate_default"

    from hazardweaver.hwa.runtime.artifact_frontier import artifact_frontier

    pk_sources = artifact_frontier(state)
    routes = reinstantiate_routes(
        task,
        state,
        store=store,
        sources=pk_sources or state.sources,
        admissible_only=False,
    )
    ranked = remsa_soft_rank(
        routes,
        task=task,
        state=state,
        committed_route=route,
    )
    controller._cache_routes(ranked)
    result.update(
        {
            "ok": False,
            "invalidated": cid,
            "reason": reason,
            "n_reinstantiated_routes": len(ranked),
            "routes": ranked,
            "n_admissible": sum(1 for r in ranked if r.get("admissible")),
        }
    )
    if level == "R6":
        result["action"] = "certified_abstention_recommended"
        result["abstention_recommended"] = True
    return result
