"""Curated interaction rule packs for workflow synthesis constraints."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from hazardweaver.hwa.contracts.capability_contract import HazardRole


class CuratorStatus(str, Enum):
    PROPOSED = "proposed"
    REVIEWED = "reviewed"
    TRUSTED = "trusted"
    DEPRECATED = "deprecated"


class ConflictStatus(str, Enum):
    NONE = "none"
    CONFLICTING = "conflicting"
    SUPERSEDED = "superseded"


class EvidenceCitation(BaseModel):
    citation_key: str
    url: str = ""
    evidence_type: str = "reviewed_literature"
    notes: str = ""


class RoleBinding(BaseModel):
    variable: str
    role: HazardRole
    schema_id: Optional[str] = None


class InteractionTarget(BaseModel):
    variable: str
    role: HazardRole = HazardRole.OUTCOME
    unit: Optional[str] = None
    spatial_support: Optional[str] = None
    temporal_support: Optional[str] = None


class InteractionTemplate(BaseModel):
    template_id: str
    topology_id: str
    target: InteractionTarget
    source_roles: Dict[str, RoleBinding] = Field(default_factory=dict)
    temporal_order: List[str] = Field(default_factory=list)
    spatial_requirements: List[str] = Field(default_factory=list)
    required_transforms: List[str] = Field(default_factory=list)
    forbidden_routes: List[str] = Field(default_factory=list)
    evidence: List[EvidenceCitation] = Field(default_factory=list)
    applicability_domain: List[str] = Field(default_factory=list)
    confidence: str = "medium"

    @model_validator(mode="after")
    def validate_template(self) -> "InteractionTemplate":
        if self.target.role != HazardRole.OUTCOME:
            raise ValueError("interaction_template_target_must_be_outcome")
        if not self.source_roles:
            raise ValueError("interaction_template_requires_source_roles")
        if not self.forbidden_routes:
            raise ValueError("interaction_template_requires_forbidden_routes")
        return self


class RulePack(BaseModel):
    rulepack_id: str
    version: str
    curator_status: CuratorStatus
    conflict_status: ConflictStatus = ConflictStatus.NONE
    structured_constraints: List[str] = Field(default_factory=list)
    evidence_citations: List[EvidenceCitation] = Field(default_factory=list)
    applicability_domain: List[str] = Field(default_factory=list)
    confidence: str = "medium"
    interaction_templates: List[InteractionTemplate] = Field(default_factory=list)

    @property
    def is_trusted(self) -> bool:
        return (
            self.curator_status == CuratorStatus.TRUSTED
            and self.conflict_status == ConflictStatus.NONE
        )

    def trusted_templates(self) -> List[InteractionTemplate]:
        if not self.is_trusted:
            return []
        return list(self.interaction_templates)

    @classmethod
    def load_yaml(cls, path: Path | str) -> "RulePack":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))
