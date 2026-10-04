"""Hazard-native capability contract v2."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from hazardweaver.hwa.contracts.capability_card import (
    ArtifactSpec,
    CapabilityKind,
    ContractField,
    ExecutionSpec,
    ProvenanceSpec,
    RuntimeSpec,
    ValidityCondition,
)


class HazardRole(str, Enum):
    FORCING = "forcing"
    STATE = "state"
    CONDITION = "condition"
    OUTCOME = "outcome"


class CapabilityType(str, Enum):
    PREDICTOR = "predictor"
    ADAPTER = "adapter"
    DATA_SOURCE = "data_source"
    TRAINER = "trainer"
    VALIDATOR = "validator"
    DOMAIN_FORMULA = "domain_formula"


class ProfileStatus(str, Enum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class DecisionTimeAvailability(BaseModel):
    available_before_decision_time: bool = True
    cutoff_description: str = ""
    latency: Optional[str] = None


class ValidityDomain(BaseModel):
    spatial_domain: List[str] = Field(default_factory=list)
    temporal_domain: Optional[str] = None
    hazard_context: List[str] = Field(default_factory=list)
    exclusions: List[str] = Field(default_factory=list)


class UncertaintyInterface(BaseModel):
    kind: str = "none"
    output_fields: List[str] = Field(default_factory=list)
    calibration_source: Optional[str] = None
    notes: str = ""


class PerformanceProfile(BaseModel):
    status: ProfileStatus = ProfileStatus.UNKNOWN
    source_artifact: Optional[str] = None
    held_out_error_by_context_slice: Dict[str, Any] = Field(default_factory=dict)
    calibration_by_context_slice: Dict[str, Any] = Field(default_factory=dict)
    coverage_and_missingness_statistics: Dict[str, Any] = Field(default_factory=dict)
    ood_risk_score: Optional[float] = None
    latency_and_cost_distribution: Dict[str, Any] = Field(default_factory=dict)


class CapabilityContract(BaseModel):
    """V2 capability contract while preserving `CapabilityCard` fields."""

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

    capability_type: Optional[CapabilityType] = None
    hazard_roles: List[HazardRole] = Field(default_factory=list)
    decision_time_availability: DecisionTimeAvailability = Field(
        default_factory=DecisionTimeAvailability
    )
    validity_domain: ValidityDomain = Field(default_factory=ValidityDomain)
    training_population: Optional[str] = None
    uncertainty_interface: UncertaintyInterface = Field(default_factory=UncertaintyInterface)
    performance_profile: PerformanceProfile = Field(default_factory=PerformanceProfile)
    known_failure_modes: List[str] = Field(default_factory=list)
    required_adapters: List[str] = Field(default_factory=list)
    version: Optional[str] = None
    checkpoint_or_formula_hash: Optional[str] = None

    def resolved_checkpoint(self) -> Optional[str]:
        return self.artifact.checkpoint_uri or self.execution.checkpoint_uri

    def accepts_schema(self, schema_id: str, direction: str = "input") -> bool:
        fields = self.input_contract if direction == "input" else self.output_contract
        return any(f.schema_id == schema_id for f in fields)

    def to_legacy_card_dict(self) -> Dict[str, Any]:
        """Return the subset expected by the legacy `CapabilityCard` parser."""

        return {
            "capability_id": self.capability_id,
            "kind": self.kind.value,
            "hazard_scope": self.hazard_scope,
            "description": self.description,
            "input_contract": [f.model_dump(mode="json") for f in self.input_contract],
            "output_contract": [f.model_dump(mode="json") for f in self.output_contract],
            "validity_conditions": [
                v.model_dump(mode="json") for v in self.validity_conditions
            ],
            "execution": self.execution.model_dump(mode="json"),
            "artifact": self.artifact.model_dump(mode="json"),
            "runtime": self.runtime.model_dump(mode="json"),
            "provenance": self.provenance.model_dump(mode="json"),
            "metadata": self.metadata,
        }
