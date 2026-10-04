"""WorkflowSpec: explicit DAG with validation and trace."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ValidationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"


class WorkflowNode(BaseModel):
    id: str
    capability_id: str
    params: Dict[str, Any] = Field(default_factory=dict)


class WorkflowEdge(BaseModel):
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    mapping: str
    schema_id: Optional[str] = None

    model_config = {"populate_by_name": True}


class ValidationResult(BaseModel):
    spatial_relation: ValidationStatus = ValidationStatus.PASS
    temporal_relation: ValidationStatus = ValidationStatus.PASS
    interface_compatibility: ValidationStatus = ValidationStatus.PASS
    interaction_conformance: ValidationStatus = ValidationStatus.PASS
    messages: List[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return all(
            s == ValidationStatus.PASS
            for s in (
                self.spatial_relation,
                self.temporal_relation,
                self.interface_compatibility,
                self.interaction_conformance,
            )
        )


class WorkflowTrace(BaseModel):
    """Execution trace for provenance ledger."""

    node_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    timings_sec: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class WorkflowSpec(BaseModel):
    workflow_id: str
    request_id: str
    interaction_spec_id: str
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]
    validation: ValidationResult = Field(default_factory=ValidationResult)
    trace: Optional[WorkflowTrace] = None
    prediction: Optional[float] = None
    abstained: bool = False
    provenance: Dict[str, Any] = Field(default_factory=dict)
