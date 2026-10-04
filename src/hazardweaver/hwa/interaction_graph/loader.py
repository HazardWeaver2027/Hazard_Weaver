"""Interaction Evidence Graph loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class EvidenceSource(BaseModel):
    citation_key: str
    url: str = ""
    evidence_type: str = "reviewed_literature"


class InteractionEdge(BaseModel):
    edge_id: str
    source: str
    target: str
    relation: str
    conditions: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[EvidenceSource] = Field(default_factory=list)
    verification_status: str = "candidate"


class InteractionNode(BaseModel):
    id: str
    type: str


class InteractionEvidenceGraph(BaseModel):
    nodes: List[InteractionNode] = Field(default_factory=list)
    edges: List[InteractionEdge] = Field(default_factory=list)

    def get_edge(self, edge_id: str) -> Optional[InteractionEdge]:
        for e in self.edges:
            if e.edge_id == edge_id:
                return e
        return None

    def verified_edges(self) -> List[InteractionEdge]:
        return [e for e in self.edges if e.verification_status == "human_verified"]

    def admissible_path(self, source: str, target: str) -> bool:
        """Check if a directed path exists from source to target."""
        adj: Dict[str, List[str]] = {}
        for e in self.edges:
            if e.verification_status != "human_verified":
                continue
            adj.setdefault(e.source, []).append(e.target)
        stack = [source]
        seen = set()
        while stack:
            n = stack.pop()
            if n == target:
                return True
            if n in seen:
                continue
            seen.add(n)
            stack.extend(adj.get(n, []))
        return False


def load_ieg(path: Path | str) -> InteractionEvidenceGraph:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return InteractionEvidenceGraph.model_validate(raw)


def default_pfdf_ieg() -> InteractionEvidenceGraph:
    root = Path(__file__).resolve().parent
    return load_ieg(root / "pfdf_minimal.yaml")
