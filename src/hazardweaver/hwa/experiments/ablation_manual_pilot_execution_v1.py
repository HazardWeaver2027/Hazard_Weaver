"""Execution-handle binding for manual ablation pilot (split/scenario authority)."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, MutableMapping, Optional

from hazardweaver.hwa.experiments.ablation_manual_pilot_v1 import (
    _fl2_split_for_row,
    ablation_manual_pilot_enabled,
)


def resolve_ablation_execution_split(inventory_row: Mapping[str, Any]) -> str:
    explicit = str(inventory_row.get("split") or "").strip()
    if explicit:
        return explicit
    fl2 = _fl2_split_for_row(inventory_row)
    return str(fl2 or "").strip()


def execution_handles_suffix(
    track: str,
    scenario_id: str,
    *,
    split: str | None = None,
) -> str:
    resolved_split = str(split or "official_test").strip() or "official_test"
    if track == "PFDF":
        return (
            f" Target execution handles (pass verbatim to run_capability after commit): "
            f"record_id={scenario_id}, scenario_id={scenario_id}, split={resolved_split}."
        )
    return (
        f" Target execution handles (pass verbatim to run_capability after commit): "
        f"scenario_id={scenario_id}, split={resolved_split}."
    )


def patch_goal_execution_handles(goal: str, *, scenario_id: str, split: str) -> str:
    """Replace stale split=official_test handles suffix in curated goals."""
    text = str(goal or "")
    if not text.strip():
        return text
    suffix = execution_handles_suffix("", scenario_id, split=split)
    handle_tail_patterns = (
        r"\s*Target execution handles \(pass verbatim to run_capability after commit\):.*$",
        r"\s*Target execution handles:.*$",
    )
    for pattern in handle_tail_patterns:
        if re.search(pattern, text):
            text = re.sub(pattern, "", text).rstrip()
            break
    resolved_split = str(split or "").strip()
    if resolved_split and resolved_split != "official_test":
        text = re.sub(r"\bsplit=official_test\b", f"split={resolved_split}", text)
    return text + suffix


def bind_ablation_inventory_metadata(
    meta: MutableMapping[str, Any],
    inventory_row: Mapping[str, Any],
) -> None:
    scenario_id = str(inventory_row.get("scenario_id") or meta.get("scenario_id") or "").strip()
    split = resolve_ablation_execution_split(inventory_row)
    if scenario_id:
        meta["scenario_id"] = scenario_id
    if split:
        meta["split"] = split
    hidden = dict(meta.get("grader_hidden") or {})
    if scenario_id:
        hidden["scenario_id"] = scenario_id
    if split:
        hidden["split"] = split
    iid = str(inventory_row.get("instance_id") or meta.get("instance_id") or "").strip()
    if iid:
        hidden["instance_id"] = iid
    if hidden:
        meta["grader_hidden"] = hidden


def ablation_authoritative_execution_handles(
    handles: Mapping[str, Any],
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
    task_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Inventory-authoritative scenario_id/split — overrides agent copy-paste errors."""
    meta = dict(task_metadata or {})
    inv = dict(inventory_row or meta.get("grader_hidden") or meta)
    out = dict(handles)
    scenario_id = str(inv.get("scenario_id") or meta.get("scenario_id") or "").strip()
    split = resolve_ablation_execution_split(inv) or str(meta.get("split") or "").strip()
    if scenario_id:
        out["scenario_id"] = scenario_id
    if split:
        out["split"] = split
    return out


def ablation_abstention_allowed(
    task: Mapping[str, Any],
    reason_code: str,
    routes: Any,
) -> Dict[str, Any]:
    """Solve-mandatory ablation cells: no spurious abstain before route probe + execution."""
    if not ablation_manual_pilot_enabled(task):
        return {"allowed": True}
    meta = task.get("metadata") or {}
    if str(meta.get("expected_action") or "solve").strip().lower() != "solve":
        return {"allowed": True}
    from hazardweaver.hwa.runtime.abstention_gate import pi_adm_empty

    route_list = list(routes or [])
    if not route_list:
        return {
            "allowed": False,
            "error": "ablation_solve_mandatory_requires_route_probe",
            "message": (
                "Ablation solve-mandatory: abstention forbidden before route enumeration. "
                "Call controller_enumerate_routes, commit a route, run_capability, then submit_solution."
            ),
            "reason_code": str(reason_code or ""),
        }
    if not pi_adm_empty(route_list):
        return {
            "allowed": False,
            "error": "ablation_solve_mandatory_admissible_routes",
            "message": (
                "Ablation solve-mandatory: abstention is forbidden while admissible routes exist. "
                "Use controller_commit_route, run_capability with the execution handles from the goal, "
                "then submit_solution."
            ),
            "reason_code": str(reason_code or ""),
        }
    return {"allowed": True}
