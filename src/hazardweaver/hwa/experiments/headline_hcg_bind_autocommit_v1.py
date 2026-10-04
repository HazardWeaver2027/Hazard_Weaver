"""HCG bind + propose + commit for near-miss mechanism cells (RQ2 v3 HWA-loop)."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import (
    routes_from_allowed_edges,
    uses_allowed_edge_route_synthesis,
)
from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import hcg_untyped
from hazardweaver.hwa.experiments.inventory_mechanism_overlay_v1 import hcg_baseline_support
from hazardweaver.hwa.scientific_controller.admissibility import is_admissible


def is_hcg_near_miss_mechanism_cell(task: Mapping[str, Any]) -> bool:
    meta = task.get("metadata") or {}
    return (
        str(meta.get("hcg_near_miss_role") or "") == "pi_near_miss"
        and bool(meta.get("hcg_typed_probe"))
        and uses_allowed_edge_route_synthesis(task)
    )


def hcg_bind_autocommit_enabled(task: Mapping[str, Any]) -> bool:
    if not is_hcg_near_miss_mechanism_cell(task):
        return False
    return str(os.environ.get("HWA_HCG_BIND_AUTOCOMMIT", "1")).lower() in (
        "1",
        "true",
        "yes",
    )


def build_pi_valid_recovery_route(
    primary: Mapping[str, Any],
    task: Mapping[str, Any],
) -> Dict[str, Any]:
    """Full-mode recovery: bind baseline contract (π_valid) after near-miss rejection."""
    meta = task.get("metadata") or {}
    recovery = dict(primary)
    recovery["route_id"] = str(primary.get("route_id") or "route")
    recovery["input_contract"] = hcg_baseline_support()
    avail = meta.get("hcg_available_support")
    if isinstance(avail, Mapping) and avail:
        recovery["available_support"] = dict(avail)
    else:
        recovery["available_support"] = hcg_baseline_support()
    recovery["hcg_binding_role"] = "pi_valid_recovery"
    recovery["hcg_bind_recovery_from"] = recovery["route_id"]
    return recovery


def bind_route(controller: Any, route: Mapping[str, Any]) -> Dict[str, Any]:
    """HCG.bind(route, state) — annotate typed compatibility on a concrete workflow."""
    gate = getattr(controller, "gate", None)
    if gate is not None:
        # Match g6 enumerate: allowed-edge tabular caps are not in the headline graph store.
        bound = gate.annotate_route(route, controller.state)
    else:
        from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import prepare_route_for_controller

        bound = prepare_route_for_controller(route, controller.task, controller.state)
    out = dict(bound)
    out["hcg_bind_applied"] = True
    out["binding_decision_made"] = True
    return out


def _route_admissible(route: Mapping[str, Any]) -> bool:
    return is_admissible(route.get("A_sci") or {}, route.get("A_cap") or {})


def resolve_hcg_bind_commit_route(controller: Any) -> Optional[Dict[str, Any]]:
    """Pick commit target: full → π_valid recovery; untyped → π_near_miss."""
    task = controller.task
    routes = routes_from_allowed_edges(task)
    if not routes:
        return None
    primary = dict(routes[0])

    if hcg_untyped():
        bound = bind_route(controller, primary)
        if _route_admissible(bound):
            bound["hcg_binding_role"] = "pi_near_miss"
            return bound
        return None

    bound_bad = bind_route(controller, primary)
    if _route_admissible(bound_bad):
        bound_bad["hcg_binding_role"] = "pi_near_miss"
        return bound_bad

    recovery = build_pi_valid_recovery_route(primary, task)
    bound_valid = bind_route(controller, recovery)
    if _route_admissible(bound_valid):
        return bound_valid
    return None


def _ensure_route_on_frontier(controller: Any, route: Mapping[str, Any]) -> None:
    rid = str(route.get("route_id") or "")
    if not rid:
        return
    controller._cache_routes([route])  # noqa: SLF001
    packet = dict(getattr(controller, "_last_decision_packet", None) or {})
    frontier_ids = [str(x) for x in (packet.get("pareto_frontier_ids") or []) if x]
    if rid not in frontier_ids:
        frontier_ids.append(rid)
        packet["pareto_frontier_ids"] = frontier_ids
        candidates = list(packet.get("candidates") or [])
        candidates.append(
            {
                "route_id": rid,
                "A_sci": route.get("A_sci"),
                "A_cap": route.get("A_cap"),
                "pareto_dominated": False,
                "capability_ids": route.get("capability_ids") or route.get("edges"),
            }
        )
        packet["candidates"] = candidates
        packet["n_candidates"] = len(candidates)
        controller._last_decision_packet = packet  # noqa: SLF001


def apply_hcg_bind_autocommit_if_eligible(
    controller: Any,
    enum_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """After enumerate: HCG.bind → propose → commit once (mechanism near-miss cells)."""
    out = dict(enum_result)
    task = getattr(controller, "task", {}) or {}
    if not hcg_bind_autocommit_enabled(task):
        return out
    if getattr(controller, "_hcg_bind_autocommit_done", False):
        return out
    state = getattr(controller, "state", None)
    if state is not None and str(getattr(state, "active_route_id", "") or "").strip():
        return out
    if out.get("g1_autocommit_applied"):
        return out

    route = resolve_hcg_bind_commit_route(controller)
    if route is None:
        out["hcg_bind_autocommit"] = {"ok": False, "error": "no_bindable_route"}
        return out

    rid = str(route.get("route_id") or "")
    controller._hcg_bind_autocommit_done = True  # noqa: SLF001
    _ensure_route_on_frontier(controller, route)

    controller._log(  # noqa: SLF001
        "bind",
        route=route,
        notes="hcg_bind_autocommit",
        extra={
            "binding_decision_made": True,
            "hcg_binding_role": route.get("hcg_binding_role"),
        },
    )

    proposed = controller.propose_route(rid, rationale="hcg_bind_autocommit")
    if not proposed.get("ok"):
        out["hcg_bind_autocommit"] = {"ok": False, "stage": "propose", **proposed}
        return out

    committed = controller.commit_route(route_id=rid)
    out["hcg_bind_autocommit"] = committed
    out["hcg_bind_autocommit_applied"] = bool(committed.get("ok"))
    return out
