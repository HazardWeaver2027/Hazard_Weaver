"""Adaptation: reinstantiate + AgentDebug branches + counterfactual reachability."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from hazardweaver.hcg.api import counterfactual_reachability
from hazardweaver.hwa.runtime.blocker_repair_map import (
    agent_debug_class_for_blocker,
    should_recheck_a_cap,
    should_recheck_a_sci,
)
from hazardweaver.hwa.runtime.session_state import SessionState


def handle_execution_failure(
    controller: Any,
    *,
    capability_id: str,
    error: str,
) -> Dict[str, Any]:
    """Invalidate failed capability and reinstantiate admissible set."""
    rid = str(getattr(controller.state, "active_route_id", None) or "")
    route = controller._route_cache.get(rid) or {}
    result = controller.handle_execution_failure(
        route,
        failed_capability_id=capability_id,
        reason=error,
    )
    if hasattr(controller, "gate"):
        routes = result.get("routes") or []
        annotated = controller.gate.admissible_routes(
            routes,
            controller.state,
            graph=controller.graph,
        )
        result["admissible_routes"] = [r for r in annotated if r.get("admissible")]
    return result


def agent_debug_repair_branch(
    blocker_code: str,
    controller: Any,
) -> Dict[str, Any]:
    """System → A_cap recheck; Planning → A_sci recheck."""
    cls = agent_debug_class_for_blocker(blocker_code)
    result: Dict[str, Any] = {"blocker": blocker_code, "agent_debug_class": cls}
    if should_recheck_a_cap(cls):
        result["action"] = "recheck_a_cap"
        routes = controller.enumerate_routes(admissible_only=True)
        result["n_admissible"] = routes.get("n_routes", 0)
    elif should_recheck_a_sci(cls):
        result["action"] = "recheck_a_sci"
        routes = controller.enumerate_routes(admissible_only=True)
        result["n_admissible"] = routes.get("n_routes", 0)
    else:
        result["action"] = "reinstantiate"
        handle_execution_failure(controller, capability_id="", error=blocker_code)
    return result


def counterfactual_recovery(
    routes: list,
    state: SessionState,
    state_delta: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    graph: Any = None,
) -> Dict[str, Any]:
    return counterfactual_reachability(routes, state, state_delta, task=task, graph=graph)
