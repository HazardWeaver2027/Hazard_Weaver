"""Static whitelist-s0 ablation: freeze curator allowed_edge_ids at s0; no post-shock expand/reinstantiate."""

from __future__ import annotations

from typing import Any, List, Mapping, MutableMapping, Sequence, Tuple

from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import route_eligibility_static_whitelist_s0


def static_whitelist_s0_frozen_edge_ids(task: Mapping[str, Any]) -> list[str]:
    meta = task.get("metadata") or {}
    frozen = meta.get("static_whitelist_s0_frozen_edge_ids")
    if isinstance(frozen, Sequence) and not isinstance(frozen, (str, bytes)):
        return [str(e).strip() for e in frozen if str(e).strip()]
    full = meta.get("rq4_full_allowed_edge_ids")
    if isinstance(full, Sequence) and not isinstance(full, (str, bytes)):
        return [str(e).strip() for e in full if str(e).strip()]
    return []


def apply_static_whitelist_s0_freeze(task: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """After E12 rq4 narrow: pin allowed_edge_ids to full curator whitelist for static arm."""
    if not route_eligibility_static_whitelist_s0():
        return task
    meta = dict(task.get("metadata") or {})
    full_allowed = static_whitelist_s0_frozen_edge_ids(task) or [
        str(e).strip()
        for e in (meta.get("rq4_full_allowed_edge_ids") or [])
        if str(e).strip()
    ]
    if not full_allowed:
        from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import unified_inventory_allowed_caps

        full_allowed = sorted(unified_inventory_allowed_caps(task))
    if not full_allowed:
        return task
    meta["static_whitelist_s0_frozen_edge_ids"] = list(full_allowed)
    meta["rq4_static_control"] = True
    meta["static_whitelist_s0_mode"] = True
    task["metadata"] = meta
    inputs = task.setdefault("solver_visible", {}).setdefault("inputs", {})
    inputs["allowed_edge_ids"] = list(full_allowed)
    return task


def static_whitelist_s0_blocks_post_shock_expand(task: Mapping[str, Any]) -> bool:
    if not route_eligibility_static_whitelist_s0():
        return False
    meta = task.get("metadata") or {}
    return bool(meta.get("static_whitelist_s0_mode"))


def clear_e12_defer_s0_narrow(task: MutableMapping[str, Any]) -> None:
    """End E12 s0 shock-only commit phase; enable post-shock recovery on frozen whitelist."""
    meta = dict(task.get("metadata") or {})
    rq4 = dict(meta.get("rq4_intervention") or {})
    rq4["defer_s0_narrow_active"] = False
    meta["rq4_intervention"] = rq4
    meta["rq4_defer_s0_narrow_active"] = False
    task["metadata"] = meta
    constraints = task.setdefault("solver_visible", {}).setdefault("constraints", {})
    crq4 = dict(constraints.get("rq4_intervention") or {})
    crq4["defer_s0_narrow_active"] = False
    constraints["rq4_intervention"] = crq4


def apply_static_whitelist_s0_post_shock_failure(
    task: MutableMapping[str, Any],
    state: Any,
    *,
    frozen_routes: Sequence[Mapping[str, Any]] | None,
) -> Tuple[List[str], List[dict[str, Any]]]:
    """After shock invalidation: clear s0 narrow, shrink allowed surface, replay frozen routes."""
    from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import unified_e12_post_shock_allowed_edge_ids

    clear_e12_defer_s0_narrow(task)
    invalidated = set(getattr(state, "invalidated_capabilities", None) or ())
    frozen_edges = static_whitelist_s0_frozen_edge_ids(task)
    post_allowed = unified_e12_post_shock_allowed_edge_ids(
        task,
        full_allowed=frozen_edges,
        invalidated=invalidated,
    )
    inputs = task.setdefault("solver_visible", {}).setdefault("inputs", {})
    inputs["allowed_edge_ids"] = list(post_allowed)
    filtered = filter_routes_by_invalidated(list(frozen_routes or []), invalidated)
    return post_allowed, filtered


def filter_routes_by_invalidated(
    routes: Sequence[Mapping[str, Any]],
    invalidated: set[str],
) -> list[dict[str, Any]]:
    if not invalidated:
        return [dict(r) for r in routes]
    out: list[dict[str, Any]] = []
    for route in routes:
        caps = [
            str(c).strip()
            for c in (route.get("capability_ids") or route.get("edges") or [])
            if str(c).strip()
        ]
        if any(cap in invalidated for cap in caps):
            continue
        out.append(dict(route))
    return out
