"""Deterministic workflow validation."""

from __future__ import annotations

from typing import Any, Dict, Optional

from hazardweaver.hwa.contracts import ValidationResult, ValidationStatus, WorkflowSpec
from hazardweaver.hwa.interaction_graph import default_pfdf_ieg


class WorkflowValidator:
    """Validates spatial, temporal, schema, and interaction conformance."""

    POSTFIRE_VALIDITY_YEARS = 5.0

    def __init__(self, interaction_spec_id: str = "pfdf_cascade_v1", watershed_index=None):
        self.interaction_spec_id = interaction_spec_id
        self.ieg = default_pfdf_ieg()
        self.watershed_index = watershed_index

    def validate(
        self,
        workflow: WorkflowSpec,
        context: Dict[str, Any],
        *,
        strict_interaction: bool = True,
    ) -> ValidationResult:
        result = ValidationResult()
        row = context.get("record", {})

        # Temporal window
        tsf = float(row.get("TimeSinceFire_yr", 0))
        if tsf > self.POSTFIRE_VALIDITY_YEARS:
            result.temporal_relation = ValidationStatus.FAIL
            result.messages.append(f"time_since_fire_exceeded:{tsf}")

        # Spatial / pairing checks for invalid-route variants
        if context.get("wrong_fire_pairing"):
            result.spatial_relation = ValidationStatus.FAIL
            result.messages.append("wrong_fire_pairing")
        if context.get("wrong_catchment"):
            result.spatial_relation = ValidationStatus.FAIL
            result.messages.append("wrong_catchment_pairing")
        elif context.get("donor_watershed_id") and self.watershed_index is not None:
            target_ws = str(row.get("WatershedID", ""))
            donor_ws = str(context.get("donor_watershed_id"))
            if target_ws and donor_ws and target_ws != donor_ws:
                result.spatial_relation = ValidationStatus.FAIL
                result.messages.append(f"watershed_mismatch:{donor_ws}!={target_ws}")

        # Schema / unit mismatch
        if context.get("schema_unit_mismatch"):
            result.interface_compatibility = ValidationStatus.FAIL
            result.messages.append("schema_unit_mismatch")

        # Interaction path: rainfall + burn state must reach volume
        if strict_interaction:
            if not self.ieg.admissible_path("rainfall_forcing", "debris_flow_volume"):
                result.interaction_conformance = ValidationStatus.FAIL
                result.messages.append("interaction_path_invalid")
            if not self.ieg.admissible_path("burn_severity", "debris_flow_volume"):
                result.interaction_conformance = ValidationStatus.FAIL
                result.messages.append("burn_to_volume_path_invalid")

        return result

    def should_abstain(self, validation: ValidationResult) -> bool:
        return not validation.is_valid
