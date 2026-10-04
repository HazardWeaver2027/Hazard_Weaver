"""Bind headline inventory scenario_id/split into run_capability handles.

Auto-bind is the default for throughput; set HWA_HEADLINE_AGENT_STRICT=1 to require
LLM-provided scenario_id/record_id (no metadata backfill).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def agent_strict_headline_bind() -> bool:
    from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_any_enabled

    return agent_strict_any_enabled()


def resolve_headline_scenario_context(
    inventory_row: Optional[Mapping[str, Any]] = None,
    *,
    task: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Map headline row → ``scenario_id`` + ``split`` for scientific caps."""
    inv = dict(inventory_row or {})
    meta = dict((task or {}).get("metadata") or {})
    solver_params = ((task or {}).get("solver_visible") or {}).get("parameters") or {}

    scenario_id = str(
        inv.get("scenario_id")
        or meta.get("scenario_id")
        or solver_params.get("scenario_id")
        or ""
    ).strip()

    instance_id = str(inv.get("instance_id") or meta.get("instance_id") or "").strip()
    if not scenario_id and instance_id.startswith("atlas:"):
        parts = instance_id.split(":", 2)
        if len(parts) == 3:
            scenario_id = parts[2].strip()

    split = str(
        inv.get("split")
        or meta.get("split")
        or solver_params.get("split")
        or ""
    ).strip()
    if not split:
        track = str(inv.get("track") or meta.get("track") or "")
        if track == "FL-2" and scenario_id.startswith("australia_2022"):
            split = "official_test_high"
        elif track == "FL-2" and scenario_id.startswith("mozambique_2019"):
            split = "hwb_holdout"
        else:
            split = "official_test"

    out: Dict[str, str] = {}
    if scenario_id:
        out["scenario_id"] = scenario_id
    if split:
        out["split"] = split
    return out


def apply_headline_scenario_binding(
    handles: Dict[str, Any],
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
    task: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Backfill scenario_id/split unless Agent-Strict mode is on."""
    if agent_strict_headline_bind():
        return handles
    ctx = resolve_headline_scenario_context(inventory_row, task=task)
    if ctx.get("scenario_id") and not handles.get("scenario_id"):
        handles["scenario_id"] = ctx["scenario_id"]
    if ctx.get("split") and not handles.get("split"):
        handles["split"] = ctx["split"]
    return handles
