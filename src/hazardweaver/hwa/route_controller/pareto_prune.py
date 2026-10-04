"""Pareto prune on solver-visible utility vector u(ρ) — deterministic, no LLM."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _utility_vector(route: Mapping[str, Any]) -> Tuple[float, float, float]:
    """Higher is better for all dimensions: validation_utility, -cost, admissible."""
    vu = route.get("validation_utility")
    val = float(vu) if vu is not None else 0.0
    cost = route.get("estimated_cost")
    cost_f = float(cost) if cost is not None else 0.0
    adm = 1.0 if route.get("admissible") else 0.0
    return (adm, val, -cost_f)


def dominates(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    ua = _utility_vector(a)
    ub = _utility_vector(b)
    if ua == ub:
        return False
    return all(x >= y for x, y in zip(ua, ub)) and any(x > y for x, y in zip(ua, ub))


def pareto_prune(routes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return non-dominated routes; mark dominated routes with pareto_dominated=True."""
    items = [dict(r) for r in routes]
    frontier_ids: List[str] = []
    for i, ri in enumerate(items):
        dominated = False
        for j, rj in enumerate(items):
            if i != j and dominates(rj, ri):
                dominated = True
                break
        ri["pareto_dominated"] = dominated
        if not dominated:
            rid = str(ri.get("route_id") or f"idx:{i}")
            frontier_ids.append(rid)
    return items


def frontier_only(routes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    pruned = pareto_prune(routes)
    return [r for r in pruned if not r.get("pareto_dominated")]
