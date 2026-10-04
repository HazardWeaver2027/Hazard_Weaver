"""Propagate benchmark-side mechanism overlays from inventory rows into HWA tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping

_HCG_BASELINE_SUPPORT: Dict[str, Any] = {
    "units": "m",
    "crs": "EPSG:4326",
    "spatial_support": "raster",
    "temporal_frequency": "static",
    "spatial_resolution": "30m",
    "modality": "eo",
    "variable_id": "depth",
    "provenance": "official_cap",
}


def hcg_baseline_support() -> Dict[str, Any]:
    return dict(_HCG_BASELINE_SUPPORT)


def apply_inventory_mechanism_overlays(
    task: MutableMapping[str, Any],
    inventory_row: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Copy paired-intervention fields into task metadata for admissibility probes."""
    meta = task.setdefault("metadata", {})
    for key in (
        "scientific_condition_override",
        "hcg_near_miss_role",
        "hcg_perturb_field",
        "hcg_pair_id",
        "hkc_intervention_role",
        "hkc_intervention_type",
        "hkc_pair_id",
        "src_dynamic_variant",
        "src_dynamic_pair_id",
    ):
        if inventory_row.get(key) is not None:
            meta[key] = inventory_row[key]

    ic = inventory_row.get("input_contract")
    if isinstance(ic, Mapping) and ic:
        meta["hcg_input_contract"] = dict(ic)
        meta["hcg_typed_probe"] = True
        if str(inventory_row.get("hcg_near_miss_role") or "") == "pi_near_miss":
            meta["hcg_available_support"] = hcg_baseline_support()
            # E4 / HWA-loop v3: bind on allowed-edge g6 routes (not hcg_multi_hop graph).
            meta["headline_route_mode"] = "g6_single_hop"

    oc = inventory_row.get("output_contract")
    if isinstance(oc, Mapping) and oc:
        meta["hcg_output_contract"] = dict(oc)

    override = inventory_row.get("scientific_condition_override")
    if isinstance(override, Mapping) and override:
        meta["scientific_condition_override"] = dict(override)

    return task
