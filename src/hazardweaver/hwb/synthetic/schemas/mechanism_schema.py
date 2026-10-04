"""Pydantic schemas for mechanism cards and scenario specs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class RelationType(str, Enum):
    TRIGGERING = "triggering"
    CONDITIONING = "conditioning"
    COMMON_FORCING = "common_forcing"
    COMPOUND_OVERLAP = "compound_overlap"
    EXPOSURE_COUPLING = "exposure_coupling"
    OTHER = "other"


class CompoundStructure(str, Enum):
    SEQUENTIAL = "sequential"
    SIMULTANEOUS = "simultaneous"
    MULTIVARIATE = "multivariate"
    SPATIALLY_COMPOUNDING = "spatially_compounding"
    TEMPORALLY_COMPOUNDING = "temporally_compounding"


class EvidenceStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class OutputModality(str, Enum):
    GRAPH = "graph"
    RASTER = "raster"
    TABULAR = "tabular"
    TRACK_SEQUENCE = "track_sequence"
    MIXED = "mixed"


class ParameterBound(BaseModel):
    min: float
    max: float


class DependencyComparePair(BaseModel):
    left: str
    right: str
    min_gap: float = 0.05


class DependencyTestSpec(BaseModel):
    metric: str = "source_target_correlation"
    compare_pairs: List[DependencyComparePair] = Field(default_factory=list)


class NegativeControlSpec(BaseModel):
    coupling_strength_override: float = 0.0
    description: str = "zero or weak coupling control"


class MechanismCard(BaseModel):
    mechanism_id: str
    name: str
    source_hazard: str
    target_hazard: str
    relation_type: RelationType
    compound_structure: CompoundStructure
    mediators: List[str] = Field(default_factory=list)
    source_variables: List[str] = Field(default_factory=list)
    target_variables: List[str] = Field(default_factory=list)
    context_variables: List[str] = Field(default_factory=list)
    temporal_window: Optional[str] = None
    spatial_operator: Optional[str] = None
    allowed_coupling_types: List[str] = Field(default_factory=list)
    counterfactual_types: List[str] = Field(min_length=1)
    citations: List[str] = Field(default_factory=list)
    evidence_strength: EvidenceStrength = EvidenceStrength.MODERATE
    parameter_bounds: Dict[str, ParameterBound] = Field(default_factory=dict)
    dependency_test_spec: DependencyTestSpec = Field(default_factory=DependencyTestSpec)
    negative_control_spec: NegativeControlSpec = Field(default_factory=NegativeControlSpec)
    notes: str = ""
    proxy_only: bool = False


class FieldSpecEntry(BaseModel):
    name: str
    shape: List[int]
    dtype: str = "float32"


class CouplingSpec(BaseModel):
    type: str
    strength: float
    monotonicity: Literal["increasing", "decreasing", "none"] = "increasing"
    interaction_terms: List[str] = Field(default_factory=list)


class TemporalOperatorSpec(BaseModel):
    lag_steps: int = 1
    window: int = 1
    alignment_mode: str = "sequential"


class SpatialOperatorSpec(BaseModel):
    type: str = "distance_decay"
    distance_decay: float = 1.0
    mask: Optional[str] = None
    shift: Optional[List[int]] = None
    resample: Optional[List[int]] = None


class LabelRuleSpec(BaseModel):
    type: Literal["regression", "binary", "raster_mask"]
    target: str
    thresholds: Dict[str, float] = Field(default_factory=dict)
    aggregation: str = "none"
    weights: Dict[str, float] = Field(default_factory=dict)


class CounterfactualSpec(BaseModel):
    name: str
    operator: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ExposedFieldManifestEntry(BaseModel):
    name: str
    shape: List[int]
    dtype: str = "float32"
    description: str = ""


class ScenarioSpec(BaseModel):
    scenario_id: str
    mechanism_id: str
    seed: int
    n_samples: int = Field(ge=1, le=1024)
    output_modality: OutputModality
    source_field_spec: List[FieldSpecEntry] = Field(default_factory=list)
    target_field_spec: List[FieldSpecEntry] = Field(default_factory=list)
    context_field_spec: List[FieldSpecEntry] = Field(default_factory=list)
    coupling: CouplingSpec
    temporal_operator: TemporalOperatorSpec = Field(default_factory=TemporalOperatorSpec)
    spatial_operator: SpatialOperatorSpec = Field(default_factory=SpatialOperatorSpec)
    label_rule: LabelRuleSpec
    counterfactuals: List[CounterfactualSpec] = Field(min_length=1)
    exposed_fields: List[str] = Field(default_factory=list)
    exposed_field_manifest: List[ExposedFieldManifestEntry] = Field(default_factory=list)
    hidden_fields: List[str] = Field(default_factory=list)
    evaluation_metrics: List[str] = Field(default_factory=list)
    parameter_bounds: Dict[str, ParameterBound] = Field(default_factory=dict)
    is_negative_control: bool = False

    @field_validator("n_samples")
    @classmethod
    def validate_n_samples(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_samples must be >= 1")
        return v


class ScenarioBundleManifest(BaseModel):
    bundle_version: str = "0.1.0"
    scenario_id: str
    mechanism_id: str
    seed: int
    generator_version: str
    mechanism_spec_hash: str
    proxy_only: bool = False
    counterfactual_names: List[str] = Field(default_factory=list)
    exposed_field_names: List[str] = Field(default_factory=list)
    label_type: str
    n_samples: int


class GenerationResult(BaseModel):
    """In-memory result before bundle write."""

    labels: Any
    exposed: Dict[str, Any]
    hidden: Dict[str, Any]
    fields: Dict[str, Any]
    counterfactuals: Dict[str, Dict[str, Any]]
    dependency_scores: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
