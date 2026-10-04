"""P_k — valid reusable artifact frontier for minimal repair ()."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Set

from hazardweaver.hwa.runtime.session_state import SessionState


def artifact_frontier(state: SessionState) -> List[str]:
    """Artifact/handle ids reusable as HCG sources after partial execution."""
    seen: Set[str] = set()
    ordered: List[str] = []
    for src in state.sources:
        s = str(src).strip()
        if s and s not in seen:
            seen.add(s)
            ordered.append(s)
    for aid in sorted(state.available_artifacts):
        s = str(aid).strip()
        if s and s not in seen:
            seen.add(s)
            ordered.append(s)
    for hid in sorted(state.produced_handles):
        s = str(hid).strip()
        if s and s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def artifact_reuse_score(route: Mapping[str, Any], state: SessionState) -> float:
    """Fraction of route edges satisfied by P_k (0..1)."""
    pk = set(artifact_frontier(state))
    if not pk:
        return 0.0
    edges = list(route.get("edges") or route.get("capability_ids") or [])
    available = set(route.get("available") or []) | pk
    if not edges:
        return 0.0
    hit = sum(1 for e in edges if str(e) in available)
    return hit / len(edges)


def frontier_snapshot(state: SessionState) -> Dict[str, Any]:
    pk = artifact_frontier(state)
    return {
        "P_k": pk,
        "n_artifacts": len(pk),
        "produced_handles": dict(state.produced_handles),
        "checkpoint": state.checkpoint,
    }
