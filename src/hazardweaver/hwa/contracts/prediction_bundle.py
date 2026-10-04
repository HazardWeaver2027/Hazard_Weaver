"""Prediction bundles returned by Hazard Weaver Agent workflows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from hazardweaver.hwa.contracts.workflow_certificate import AbstainCode, WorkflowValidityCertificate


class UncertaintySpec(BaseModel):
    kind: str = "none"
    lower: Optional[float] = None
    upper: Optional[float] = None
    coverage: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PredictionBundle(BaseModel):
    value: Optional[float] = None
    uncertainty: UncertaintySpec = Field(default_factory=UncertaintySpec)
    spatial_support: str
    temporal_support: str
    units: str
    quality_flags: List[str] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    capability_version: Optional[str] = None
    decision_time_cutoff: Optional[datetime] = None


class DecisionReadyPredictionBundle(BaseModel):
    estimate: Optional[PredictionBundle] = None
    uncertainty: Optional[UncertaintySpec] = None
    workflow_certificate: Optional[WorkflowValidityCertificate] = None
    applicability_range: List[str] = Field(default_factory=list)
    data_quality_flags: List[str] = Field(default_factory=list)
    interaction_assumptions: List[str] = Field(default_factory=list)
    alternative_valid_workflows: List[Dict[str, Any]] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    human_readable_summary: str = ""
    abstain: bool = False
    violated_contracts: List[str] = Field(default_factory=list)
    failed_checks: List[str] = Field(default_factory=list)
    minimal_missing_inputs: List[str] = Field(default_factory=list)
    valid_fallbacks: List[str] = Field(default_factory=list)
    abstain_codes: List[AbstainCode] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_bundle_mode(self) -> "DecisionReadyPredictionBundle":
        if self.abstain:
            if not (self.failed_checks or self.violated_contracts or self.abstain_codes):
                raise ValueError("abstain_bundle_requires_failed_checks_or_codes")
        elif self.estimate is None:
            raise ValueError("non_abstain_bundle_requires_estimate")
        return self
