"""Shared MH-4 Gate B helpers (eligible events, reference loss extraction)."""

from __future__ import annotations

from typing import Any, Dict, List

from hazardweaver.hcg.carp.scientific.mh4_data import load_split_manifest
from hazardweaver.hcg.carp.scientific.mh4_reference import (
    LOSS_VARIABLE,
    event_abstain_reason,
    load_event_reference,
)


def eligible_event_ids(split: str) -> List[str]:
    manifest = load_split_manifest(split) or {}
    ids: List[str] = []
    for nhc_id in manifest.get("event_ids") or []:
        ref = load_event_reference(str(nhc_id))
        if ref and not event_abstain_reason(ref):
            ids.append(str(nhc_id))
    return ids


def reference_loss_usd_2023(ref: Dict[str, Any]) -> float:
    if LOSS_VARIABLE in ref and ref[LOSS_VARIABLE] is not None:
        return float(ref[LOSS_VARIABLE])
    if ref.get("ncei_property_usd_2023") is not None:
        return float(ref["ncei_property_usd_2023"])
    if ref.get("ncei_property_usd") is not None:
        return float(ref["ncei_property_usd"])
    raise ValueError(f"missing deflated reference for {ref.get('nhc_id')}")


def marginal_wind_usd_2023(ref: Dict[str, Any]) -> float:
    if ref.get("wind_marginal_usd_2023") is not None:
        return float(ref["wind_marginal_usd_2023"])
    if ref.get("wind_marginal_usd") is not None:
        return float(ref["wind_marginal_usd"])
    raise ValueError(f"missing wind marginal for {ref.get('nhc_id')}")


def marginal_surge_usd_2023(ref: Dict[str, Any]) -> float:
    if ref.get("surge_marginal_usd_2023") is not None:
        return float(ref["surge_marginal_usd_2023"])
    if ref.get("surge_marginal_usd") is not None:
        return float(ref["surge_marginal_usd"])
    raise ValueError(f"missing surge marginal for {ref.get('nhc_id')}")


def independent_sum_usd_2023(ref: Dict[str, Any]) -> float:
    if ref.get("independent_sum_usd_2023") is not None:
        return float(ref["independent_sum_usd_2023"])
    if ref.get("independent_sum_usd") is not None:
        return float(ref["independent_sum_usd"])
    return marginal_wind_usd_2023(ref) + marginal_surge_usd_2023(ref)
