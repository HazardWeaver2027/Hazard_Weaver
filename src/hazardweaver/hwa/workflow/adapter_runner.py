"""Runtime execution for registered PFDF adapter contracts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models.burn_state_net.landsat_adapter import modhigh50_proxy_km2
from models.pfdf_volume_adapter.gorr import MISSING, _fval

from hazardweaver.hwa.contracts import AdapterContract, AdapterRegistry

RAINFALL_FIELDS = (
    "i30RainfallAnomaly",
    "i15RainfallAnomaly",
    "i60RainfallAnomaly",
    "TotalRainfall_mm",
)

TERRAIN_FIELDS = (
    "Area_km2",
    "Area50_km2",
    "Ruggedness",
    "Slope_deg",
    "SoilTexture",
)


class AdapterRunResult:
    __slots__ = ("adapter_id", "valid", "patches", "trace", "quality_flags")

    def __init__(
        self,
        adapter_id: str,
        *,
        valid: bool,
        patches: Dict[str, Any],
        trace: Dict[str, Any],
        quality_flags: Optional[List[str]] = None,
    ):
        self.adapter_id = adapter_id
        self.valid = valid
        self.patches = patches
        self.trace = trace
        self.quality_flags = quality_flags or []


class AdapterRunner:
    """Apply adapter contracts to inventory records and burn summaries."""

    def __init__(self, registry: Optional[AdapterRegistry] = None):
        self.registry = registry

    def apply(
        self,
        adapter_id: str,
        record: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
        burn_summary: Optional[Dict[str, Any]] = None,
    ) -> AdapterRunResult:
        context = context or {}
        if self.registry is not None:
            contract = self.registry.get(adapter_id)
        else:
            contract = AdapterContract(
                adapter_id=adapter_id,
                adapter_type="schema_mapping",
                source_schema="unknown",
                target_schema="unknown",
                implementation_ref="inline",
            )

        if adapter_id == "pfdf_record_to_event_rainfall_v1":
            return self._rainfall(record, contract)
        if adapter_id == "pfdf_record_to_terrain_condition_v1":
            return self._terrain(record, contract)
        if adapter_id == "pfdf_burn_summary_to_watershed_v1":
            return self._burn_summary(record, contract, burn_summary=burn_summary, context=context)
        return AdapterRunResult(
            adapter_id,
            valid=False,
            patches={},
            trace={"error": "unknown_adapter"},
            quality_flags=["unknown_adapter"],
        )

    def _rainfall(self, record: Dict[str, Any], contract: AdapterContract) -> AdapterRunResult:
        missing = [f for f in RAINFALL_FIELDS if _fval(record, f) == MISSING]
        i30 = _fval(record, "i30RainfallAnomaly")
        area = _fval(record, "Area_km2")
        valid = i30 > 0 and area > 0
        trace = {
            "adapter_type": contract.adapter_type.value,
            "fields_checked": list(RAINFALL_FIELDS),
            "missing_fields": missing,
            "i30RainfallAnomaly": i30 if i30 != MISSING else None,
            "Area_km2": area if area != MISSING else None,
        }
        flags: List[str] = []
        if not valid:
            flags.append("missing_event_rainfall_fields")
        return AdapterRunResult(
            contract.adapter_id,
            valid=valid,
            patches={},
            trace=trace,
            quality_flags=flags,
        )

    def _terrain(self, record: Dict[str, Any], contract: AdapterContract) -> AdapterRunResult:
        area = _fval(record, "Area_km2")
        rugged = _fval(record, "Ruggedness")
        valid = area > 0
        missing = [f for f in TERRAIN_FIELDS if _fval(record, f) == MISSING]
        trace = {
            "adapter_type": contract.adapter_type.value,
            "fields_checked": list(TERRAIN_FIELDS),
            "missing_fields": missing,
            "Area_km2": area if area != MISSING else None,
            "Ruggedness": rugged if rugged != MISSING else None,
        }
        flags: List[str] = []
        if rugged == MISSING:
            flags.append("terrain_ruggedness_missing")
        if not valid:
            flags.append("missing_slope_or_basin_area_fields")
        return AdapterRunResult(
            contract.adapter_id,
            valid=valid,
            patches={},
            trace=trace,
            quality_flags=flags,
        )

    def _burn_summary(
        self,
        record: Dict[str, Any],
        contract: AdapterContract,
        *,
        burn_summary: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> AdapterRunResult:
        patches: Dict[str, Any] = {}
        trace: Dict[str, Any] = {"adapter_type": contract.adapter_type.value}
        flags: List[str] = []

        if burn_summary is not None:
            mh50 = modhigh50_proxy_km2(record, burn_summary)
            patches["ModHigh50_km2"] = mh50
            trace["modhigh50_km2_proxy"] = mh50
            trace["source"] = "burn_summary_post_state"
            return AdapterRunResult(
                contract.adapter_id,
                valid=True,
                patches=patches,
                trace=trace,
                quality_flags=flags,
            )

        area50 = _fval(record, "Area50_km2")
        if area50 == MISSING:
            area50 = _fval(record, "Area_km2")
        trace["Area50_km2"] = area50 if area50 != MISSING else None
        trace["source"] = "record_pre_burn_alignment"
        if area50 == MISSING or area50 <= 0:
            flags.append("watershed_geometry_incomplete")
            return AdapterRunResult(
                contract.adapter_id,
                valid=False,
                patches={},
                trace=trace,
                quality_flags=flags,
            )
        return AdapterRunResult(
            contract.adapter_id,
            valid=True,
            patches={},
            trace=trace,
            quality_flags=flags,
        )


def is_adapter_node(node, adapter_ids: Optional[set[str]] = None) -> bool:
    if node.params.get("node_type") == "adapter":
        return True
    if adapter_ids and node.capability_id in adapter_ids:
        return True
    return node.capability_id.startswith("pfdf_record_to_") or node.capability_id.endswith(
        "_to_watershed_v1"
    )
