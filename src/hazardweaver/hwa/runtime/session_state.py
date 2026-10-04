"""Shared session state s_k (HWA + HCG api boundary)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class SessionState:
    """Runtime session state s_k for certificate-grounded route control."""

    available_artifacts: Set[str] = field(default_factory=set)
    produced_handles: Dict[str, Any] = field(default_factory=dict)
    invalidated_edges: Set[str] = field(default_factory=set)
    invalidated_capabilities: Set[str] = field(default_factory=set)
    checkpoint: int = 0
    active_route_id: Optional[str] = None
    pending_route_id: Optional[str] = None
    theory_arm: str = "verified"
    sources: List[str] = field(default_factory=list)
    target: str = ""
    packs: List[str] = field(default_factory=list)
    state_version: str = "s0"
    certificate_version: str = ""
    active_lease: Optional[Dict[str, Any]] = None
    segment_id: str = ""
    legacy_bypass_count: int = 0

    def bump_checkpoint(self) -> int:
        self.checkpoint += 1
        self.state_version = f"s{self.checkpoint}"
        return self.checkpoint

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available_artifacts": sorted(self.available_artifacts),
            "invalidated_edges": sorted(self.invalidated_edges),
            "invalidated_capabilities": sorted(self.invalidated_capabilities),
            "checkpoint": self.checkpoint,
            "active_route_id": self.active_route_id,
            "pending_route_id": self.pending_route_id,
            "theory_arm": self.theory_arm,
            "sources": list(self.sources),
            "target": self.target,
            "packs": list(self.packs),
            "state_version": self.state_version,
            "certificate_version": self.certificate_version,
            "active_lease": self.active_lease,
            "segment_id": self.segment_id,
            "legacy_bypass_count": self.legacy_bypass_count,
        }

    def copy(self) -> "SessionState":
        return SessionState(
            available_artifacts=set(self.available_artifacts),
            produced_handles=dict(self.produced_handles),
            invalidated_edges=set(self.invalidated_edges),
            invalidated_capabilities=set(self.invalidated_capabilities),
            checkpoint=self.checkpoint,
            active_route_id=self.active_route_id,
            pending_route_id=self.pending_route_id,
            theory_arm=self.theory_arm,
            sources=list(self.sources),
            target=self.target,
            packs=list(self.packs),
            state_version=self.state_version,
            certificate_version=self.certificate_version,
            active_lease=dict(self.active_lease) if self.active_lease else None,
            segment_id=self.segment_id,
            legacy_bypass_count=self.legacy_bypass_count,
        )
