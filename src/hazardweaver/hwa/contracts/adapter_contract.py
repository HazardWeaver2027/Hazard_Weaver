"""First-class geospatial-temporal adapter contracts."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class AdapterType(str, Enum):
    SPATIAL_AGGREGATION = "spatial_aggregation"
    TEMPORAL_AGGREGATION = "temporal_aggregation"
    SCHEMA_MAPPING = "schema_mapping"
    UNIT_CONVERSION = "unit_conversion"
    FEATURE_DERIVATION = "feature_derivation"


class AdapterContract(BaseModel):
    adapter_id: str
    adapter_type: AdapterType
    source_schema: str
    target_schema: str
    source_spatial_support: Optional[str] = None
    target_spatial_support: Optional[str] = None
    source_temporal_support: Optional[str] = None
    target_temporal_support: Optional[str] = None
    units_in: Optional[str] = None
    units_out: Optional[str] = None
    uncertainty_behavior: str = "not_declared"
    known_failure_modes: List[str] = Field(default_factory=list)
    implementation_ref: str
    field_sources: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_adapter(self) -> "AdapterContract":
        if self.source_schema == self.target_schema and self.adapter_type != AdapterType.SCHEMA_MAPPING:
            raise ValueError("non_mapping_adapter_requires_distinct_source_target_schema")
        if not self.implementation_ref:
            raise ValueError("adapter_requires_implementation_ref")
        return self


class AdapterRegistry(BaseModel):
    adapters: List[AdapterContract] = Field(default_factory=list)

    def get(self, adapter_id: str) -> AdapterContract:
        for adapter in self.adapters:
            if adapter.adapter_id == adapter_id:
                return adapter
        raise KeyError(f"Unknown adapter: {adapter_id}")

    def list_ids(self) -> List[str]:
        return sorted(adapter.adapter_id for adapter in self.adapters)

    @classmethod
    def load_yaml(cls, path: Path | str) -> "AdapterRegistry":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if isinstance(raw, list):
            raw = {"adapters": raw}
        return cls.model_validate(raw)
