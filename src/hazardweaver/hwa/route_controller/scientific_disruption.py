"""Scientific disruption distance d_sci for minimal repair ordering ()."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _family_id(route: Mapping[str, Any]) -> str:
    return str(
        route.get("route_family_id")
        or route.get("hkc_family_id")
        or route.get("family_id")
        or ""
    )


def _impl_family(route: Mapping[str, Any]) -> str:
    caps = route.get("capability_ids") or route.get("edges") or []
    if caps:
        return str(caps[0])
    return _family_id(route)


def scientific_disruption(
    candidate: Mapping[str, Any],
    committed: Optional[Mapping[str, Any]],
) -> float:
    """Lower is less disruptive. Same impl family < same route family < different family."""
    if committed is None:
        return 0.0
    cand_impl = _impl_family(candidate)
    comm_impl = _impl_family(committed)
    if cand_impl and comm_impl and cand_impl == comm_impl:
        return 0.0
    cand_fam = _family_id(candidate)
    comm_fam = _family_id(committed)
    if cand_fam and comm_fam and cand_fam == comm_fam:
        return 1.0
    if cand_fam and comm_fam:
        return 2.0
    # edge overlap proxy when family metadata absent
    ce = set(candidate.get("edges") or [])
    pe = set(committed.get("edges") or [])
    if ce and pe:
        jaccard = len(ce & pe) / max(1, len(ce | pe))
        return 3.0 * (1.0 - jaccard)
    return 3.0
