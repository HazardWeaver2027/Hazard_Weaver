"""MH-4 TC wind+surge event-level building structural loss evaluator (DL-051 Gate B)."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hcg.carp.scientific.mh4_reference import (
    CURRENCY_YEAR,
    LOSS_VARIABLE,
    LOSS_VARIABLE_LEGACY_ALIAS,
    REFERENCE_LOSS_SOURCE_ID,
)

EVALUATOR_ID = "mh4_tc_building_loss"
STATUS = "GATE_B_REFERENCE_FROZEN"
PI_REQUIRED = "PI_REQUIRED"

EXPOSURE_INVENTORY_ID = "climada_nightlight_proxy_conus_v1"
MAPE_FLOOR_USD = 1e6


def reference_loss_ready() -> bool:
    return REFERENCE_LOSS_SOURCE_ID not in (PI_REQUIRED, "", None)


def evaluator_ready() -> bool:
    """Full evaluator needs exposure pin; reference route (R02) needs reference only."""
    exposure_ok = EXPOSURE_INVENTORY_ID not in (PI_REQUIRED, "", None) and "pending" not in EXPOSURE_INVENTORY_ID
    return reference_loss_ready() and exposure_ok


def _extract_loss_usd(mapping: Mapping[str, Any]) -> float:
    for key in (LOSS_VARIABLE, LOSS_VARIABLE_LEGACY_ALIAS, "ncei_property_usd_2023"):
        if key in mapping and mapping[key] is not None:
            return float(mapping[key])
    if "ncei_property_usd" in mapping:
        return float(mapping["ncei_property_usd"])
    value = mapping.get("value")
    if isinstance(value, Mapping):
        for key in (LOSS_VARIABLE, LOSS_VARIABLE_LEGACY_ALIAS):
            if key in value:
                return float(value[key])
    if "reference_score" in mapping:
        return float(mapping["reference_score"])
    raise ValueError(f"missing {LOSS_VARIABLE} in artifact or reference")


def score_event_loss(pred_usd: float, ref_usd: float, *, mape_floor: float = MAPE_FLOOR_USD) -> Dict[str, Any]:
    """Score one TC event: log_mae primary; RAE/MAPE when ref > floor."""
    pred = float(pred_usd)
    ref = float(ref_usd)
    log_mae = abs(math.log1p(max(pred, 0.0)) - math.log1p(max(ref, 0.0)))
    bias = pred - ref
    rae: Optional[float] = None
    mape: Optional[float] = None
    if ref > mape_floor:
        rae = abs(pred - ref) / abs(ref)
        mape = rae
    return {
        "metric_name": "log_mae",
        "score": float(log_mae),
        "higher_is_better": False,
        "log_mae": float(log_mae),
        "signed_loss_bias_usd": float(bias),
        "relative_absolute_error": rae,
        "loss_mape": mape,
        "prediction_usd": pred,
        "reference_usd": ref,
        "mape_floor_usd": mape_floor,
    }


def score_artifact(
    taskpack: Mapping[str, Any],
    final_artifact: Mapping[str, Any],
    *,
    reference_view: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Score HWB artifact; reference routes allowed when reference contract frozen."""
    if not reference_loss_ready():
        return {
            "status": "BLOCKED",
            "evaluator_id": EVALUATOR_ID,
            "reason": "MH-4 reference loss not frozen (DL-051)",
            "exposure_inventory_id": EXPOSURE_INVENTORY_ID,
            "reference_loss_source_id": REFERENCE_LOSS_SOURCE_ID,
            "currency_year": CURRENCY_YEAR,
            "vulnerability_policy": "route_native_versioned",
        }
    ref_view = reference_view if reference_view is not None else (taskpack.get("reference_view") or {})
    outputs = ref_view.get("outputs") or ref_view
    pred = _extract_loss_usd(final_artifact)
    ref = _extract_loss_usd(outputs)
    result = score_event_loss(pred, ref)
    result["status"] = "OK"
    result["evaluator_id"] = EVALUATOR_ID
    vuln_decl = final_artifact.get("vulnerability_declaration_id")
    if vuln_decl:
        result["vulnerability_declaration_id"] = vuln_decl
    return result


# Backward-compatible alias for legacy evaluator_ref strings.
mh4_tc_building_loss_pending = score_artifact
