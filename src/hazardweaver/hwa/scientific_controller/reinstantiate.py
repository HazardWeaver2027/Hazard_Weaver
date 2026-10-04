"""HWA-layer re-instantiate shim (until HCG core PR lands)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwa.runtime.artifact_frontier import artifact_frontier
from hazardweaver.hcg import path_edge_ids

from hazardweaver.hwa.scientific_controller.admissibility import admissible_routes, route_from_hcg_path
from hazardweaver.hwa.scientific_controller.reachability import find_reachable_paths
from hazardweaver.hwa.scientific_controller.state import SessionState


def invalidate_capability(
    state: SessionState,
    capability_id: str,
    *,
    reason: str = "",
    also_invalidate_edges: Optional[Sequence[str]] = None,
) -> None:
    """Mark capability/edges invalid in s_k."""
    cid = str(capability_id).strip()
    if cid:
        state.invalidated_capabilities.add(cid)
        state.invalidated_edges.add(cid)
    for eid in also_invalidate_edges or ():
        state.invalidated_edges.add(str(eid))
    state.bump_checkpoint()
    state.active_route_id = None
    state.pending_route_id = None


def invalidate_edge(state: SessionState, edge_id: str, *, reason: str = "") -> None:
    state.invalidated_edges.add(str(edge_id))
    state.bump_checkpoint()


def reinstantiate_routes(
    task: Mapping[str, Any],
    state: SessionState,
    *,
    store: Any = None,
    sources: Optional[Sequence[str]] = None,
    target: Optional[str] = None,
    max_paths: int = 8,
    admissible_only: bool = True,
) -> List[Dict[str, Any]]:
    """Re-search Π(q) after invalidation, excluding blocked edges."""
    from hazardweaver.hwa.scientific_controller.reachability import resolve_graph_store

    resolved_store = resolve_graph_store(store, state, packs=state.packs)
    src = list(sources or artifact_frontier(state) or state.sources)
    tgt = str(target or state.target or "")
    if not src or not tgt:
        return []

    routes: List[Dict[str, Any]] = []
    if state.invalidated_capabilities and resolved_store is not None:
        from hazardweaver.hcg.search.invalidate_reroute import invalidate_and_reroute

        seen: set[tuple[str, ...]] = set()
        for cid in sorted(state.invalidated_capabilities):
            rr = invalidate_and_reroute(
                resolved_store.graph,
                src,
                tgt,
                invalidated_capability=cid,
            )
            for path in rr.get("paths") or []:
                pinfo = {
                    "edges": path_edge_ids(path),
                    "available": sorted(path.available),
                    "valid": True,
                    "rerouted": True,
                }
                key = tuple(pinfo["edges"])
                if key in seen:
                    continue
                seen.add(key)
                route = route_from_hcg_path(pinfo)
                route["rerouted"] = True
                routes.append(route)

    if len(routes) < max_paths:
        seen = {tuple(r.get("edges") or []) for r in routes}
        for route in find_reachable_paths(
            src,
            tgt,
            state,
            store=resolved_store,
            max_paths=max_paths,
            packs=state.packs,
            executable_only=True,
        ):
            key = tuple(route.get("edges") or [])
            if key in seen:
                continue
            seen.add(key)
            routes.append(route)
            if len(routes) >= max_paths:
                break
    else:
        routes = routes[:max_paths]
    graph = resolved_store.graph if resolved_store is not None else None
    return admissible_routes(
        routes,
        task,
        state,
        graph=graph,
        admissible_only=admissible_only,
    )


def tool_hcg_reinstantiate(
    sources: Sequence[str],
    target: str,
    *,
    state: SessionState,
    task: Mapping[str, Any],
    store: Any = None,
    max_paths: int = 8,
) -> Dict[str, Any]:
    """Internal-only reinstantiate API for ScientificController (not LLM-facing)."""
    state.sources = list(sources) or state.sources
    state.target = str(target) or state.target
    routes = reinstantiate_routes(
        task,
        state,
        store=store,
        sources=sources,
        target=target,
        max_paths=max_paths,
        admissible_only=False,
    )
    admissible = [r for r in routes if r.get("admissible")]
    return {
        "ok": True,
        "n_routes": len(routes),
        "n_admissible": len(admissible),
        "routes": routes,
        "admissible_routes": admissible,
        "invalidated_edges": sorted(state.invalidated_edges),
        "invalidated_capabilities": sorted(state.invalidated_capabilities),
        "checkpoint": state.checkpoint,
    }
