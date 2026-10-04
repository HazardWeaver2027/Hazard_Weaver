"""ReAct headline: controlled direct execute without controller commit."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from hazardweaver.hcg.contracts.execution_event import ScopedExecutionLease
from hazardweaver.hwa.runtime.lease_manager import mint_lease, require_lease, store_lease_on_host


def _allowed_capability_ids(task: Mapping[str, Any]) -> List[str]:
    sv = task.get("solver_visible") or {}
    inputs = sv.get("inputs") or {}
    allowed = [str(e) for e in (inputs.get("allowed_edge_ids") or [])]
    return [e for e in allowed if not e.startswith("schema_map")]


def is_react_headline_task(task: Mapping[str, Any]) -> bool:
    meta = task.get("metadata") or {}
    if meta.get("same_llm_condition") != "react":
        return False
    return bool(meta.get("hwb_headline_inventory") or meta.get("same_llm_g6_coreexec"))


def synthesize_g6_route(capability_id: str, *, admissible: bool = True) -> Dict[str, Any]:
    rid = f"route:cap:{capability_id}"
    return {
        "route_id": rid,
        "edges": [capability_id],
        "capability_ids": [capability_id],
        "admissible": admissible,
        "executable": True,
        "g6_coreexec": True,
        "route_source": "react_headline_direct",
    }


def _session_state_for_task(task: Mapping[str, Any]) -> Any:
    from hazardweaver.hwa.scientific_controller.reachability import init_state_from_task

    state = init_state_from_task(task)
    from hazardweaver.hwa.runtime.runtime_closure import hydrate_session_graph_from_task

    hydrate_session_graph_from_task(task, state)
    return state


def build_react_route(
    task: Mapping[str, Any],
    capability_id: str,
    *,
    host: Any = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Build route for ReAct execute; optional Π_adm gate (R2)."""
    from hazardweaver.hwa.experiments.agent_strict_v2 import react_admissibility_gate_enabled

    cid = str(capability_id).strip()
    if not react_admissibility_gate_enabled():
        return synthesize_g6_route(cid, admissible=True), None

    from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import (
        prepare_route_for_controller,
        routes_from_allowed_edges,
    )
    from hazardweaver.hwa.scientific_controller.admissibility import (
        admissibility_reason_code,
        annotate_route,
        is_admissible,
    )

    state = _session_state_for_task(task)
    graph = getattr(host, "graph", None) if host is not None else None
    candidates = routes_from_allowed_edges(task)
    base = synthesize_g6_route(cid, admissible=False)
    matched = next(
        (r for r in candidates if cid in [str(c) for c in (r.get("capability_ids") or [])]),
        base,
    )
    # Option-B allowed_edge routes already carry admissible A_sci/A_cap; re-annotate
    # applies task_require grounding that rejects curated headline cells (→ Π false reject).
    if matched.get("g6_coreexec") and str(matched.get("route_source") or "") == "allowed_edge_ids":
        route = prepare_route_for_controller(matched, task, state, graph=graph)
    else:
        route = annotate_route(matched, task, state, graph=graph)
        route = prepare_route_for_controller(route, task, state, graph=graph)
    if not is_admissible(route.get("A_sci") or {}, route.get("A_cap") or {}):
        return None, {
            "ok": False,
            "error": "react_route_not_admissible",
            "reason_code": admissibility_reason_code(
                route.get("A_sci") or {},
                route.get("A_cap") or {},
            ),
            "capability_id": cid,
            "route_id": route.get("route_id"),
            "A_sci": route.get("A_sci"),
            "A_cap": route.get("A_cap"),
            "tool": "run_capability",
        }
    route["admissible"] = True
    return route, None


def prepare_react_headline_execute(
    host: Any,
    capability_id: str,
) -> Tuple[Optional[ScopedExecutionLease], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Mint ephemeral lease + routes for ReAct headline allowed-edge execute.

    Returns (lease, routes, error_dict). error_dict is set when cap is not allowed.
    """
    from hazardweaver.hwa.experiments.agent_strict_v2 import (
        agent_strict_v2_enabled,
        agent_strict_v2_react_ablation_enabled,
    )

    task = getattr(host, "task", None) or {}
    if agent_strict_v2_enabled() and not agent_strict_v2_react_ablation_enabled(task):
        return None, [], {
            "ok": False,
            "error": "react_ephemeral_lease_disabled_in_agent_strict_v2",
            "capability_id": str(capability_id or "").strip(),
            "tool": "run_capability",
        }
    if not is_react_headline_task(task):
        return None, [], None

    cid = str(capability_id).strip()
    allowed = _allowed_capability_ids(task)
    if cid not in allowed:
        return None, [], {
            "ok": False,
            "error": "react_headline_capability_not_allowed",
            "capability_id": cid,
            "allowed_capability_ids": allowed,
            "tool": "run_capability",
        }

    route, adm_err = build_react_route(task, cid, host=host)
    if adm_err is not None:
        return None, [], adm_err
    assert route is not None

    existing = require_lease(host)
    if existing is not None and cid in list(existing.allowed_capability_ids or []):
        return existing, [route], None

    lease = mint_lease(
        route_id=route["route_id"],
        allowed_capability_ids=[cid],
    )
    store_lease_on_host(host, lease)
    return lease, [route], None
