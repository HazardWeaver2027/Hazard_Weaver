"""CapabilityCard: registry contract for datasets, models, and operators."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CapabilityKind(str, Enum):
    DATASET = "dataset"
    PREDICTIVE_MODEL = "predictive_model"
    FEATURE_OPERATOR = "feature_operator"
    DOMAIN_FORMULA = "domain_formula"
    SIMULATOR = "simulator"


class ContractField(BaseModel):
    """Typed input/output field declaration."""

    schema_id: str
    spatial_support: Optional[str] = None
    temporal_support: Optional[str] = None
    units: Optional[str] = None
    dtype: str = "float32"
    description: str = ""


class ValidityCondition(BaseModel):
    """Conditions under which a capability may be invoked."""

    condition_id: str
    description: str
    required: bool = True


class ArtifactSpec(BaseModel):
    checkpoint_uri: Optional[str] = None
    config_hash: Optional[str] = None


class RuntimeSpec(BaseModel):
    env: str = "envs/pyhazards"
    device: str = "cpu"


class ExecutionSpec(BaseModel):
    """How to invoke the capability."""

    entrypoint: str
    config_path: Optional[str] = None
    checkpoint_uri: Optional[str] = None  # legacy; prefer artifact.checkpoint_uri


class ProvenanceSpec(BaseModel):
    """Training/data provenance for audit."""

    source_paper: Optional[str] = None
    source_url: Optional[str] = None
    training_data_manifest: Optional[str] = None
    verification_status: str = "human_verified"


class CapabilityCard(BaseModel):
    """Permanent capability registry entry."""

    capability_id: str
    kind: CapabilityKind
    hazard_scope: str
    description: str = ""
    input_contract: List[ContractField] = Field(default_factory=list)
    output_contract: List[ContractField] = Field(default_factory=list)
    validity_conditions: List[ValidityCondition] = Field(default_factory=list)
    execution: ExecutionSpec
    artifact: ArtifactSpec = Field(default_factory=ArtifactSpec)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    provenance: ProvenanceSpec = Field(default_factory=ProvenanceSpec)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def resolved_checkpoint(self) -> Optional[str]:
        return self.artifact.checkpoint_uri or self.execution.checkpoint_uri

    def accepts_schema(self, schema_id: str, direction: str = "input") -> bool:
        fields = self.input_contract if direction == "input" else self.output_contract
        return any(f.schema_id == schema_id for f in fields)

    def to_yaml_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")
