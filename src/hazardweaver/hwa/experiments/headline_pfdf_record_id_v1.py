"""Resolve PFDF record_id for headline MH-1 atlas / state-variant cells."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def resolve_headline_pfdf_record_id(
    inventory_row: Optional[Mapping[str, Any]] = None,
    *,
    task: Optional[Mapping[str, Any]] = None,
) -> str:
    """Map headline inventory row → USGS PFDF ``record_id`` for portfolio caps."""
    from hazardweaver.hwa.experiments.headline_scenario_binding_v1 import agent_strict_headline_bind

    if agent_strict_headline_bind():
        raise KeyError("headline_pfdf_record_id_auto_bind_disabled_in_agent_strict")
    inv = dict(inventory_row or {})
    meta = dict((task or {}).get("metadata") or {})
    instance_id = str(inv.get("instance_id") or meta.get("instance_id") or "").strip()
    if instance_id.startswith("atlas:MH-1:"):
        return instance_id.split("atlas:MH-1:", 1)[1].strip()

    scenario_id = str(
        inv.get("scenario_id")
        or meta.get("scenario_id")
        or ((task or {}).get("solver_visible") or {}).get("inputs", {}).get("parameters", {}).get(
            "scenario_id"
        )
        or ""
    ).strip()
    if scenario_id and "_" in scenario_id and not scenario_id.startswith("mh1_"):
        return scenario_id

    if scenario_id:
        from hazardweaver.hwa.experiments.mh1_pilot_crc_v1 import resolve_record_id_for_scenario

        return resolve_record_id_for_scenario(scenario_id)

    raise KeyError("headline_pfdf_record_id_unresolved")
