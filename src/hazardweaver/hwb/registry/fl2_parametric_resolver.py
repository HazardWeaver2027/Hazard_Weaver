"""Resolve parametric FL-2 HWB taskpack reference views by scenario_id."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

REFS_BY_TASKPACK = {
    "hwb_fl2_parametric_v1": Path(__file__).resolve().parent
    / "taskpacks"
    / "hwb_fl2_parametric_v1_refs.json",
    "hwb_fl2_solver_parametric_v1": Path(__file__).resolve().parent
    / "taskpacks"
    / "hwb_fl2_solver_parametric_v1_refs.json",
}


def _refs_path_for_taskpack(taskpack: Mapping[str, Any]) -> Path:
    tid = str(taskpack.get("taskpack_id") or "")
    if tid in REFS_BY_TASKPACK:
        return REFS_BY_TASKPACK[tid]
    sidecar = (taskpack.get("reference_view") or {}).get("refs_sidecar")
    if sidecar:
        return Path(__file__).resolve().parent / "taskpacks" / str(sidecar)
    return REFS_BY_TASKPACK["hwb_fl2_parametric_v1"]


def load_scenario_refs(taskpack_id: Optional[str] = None) -> Dict[str, Any]:
    if taskpack_id and taskpack_id in REFS_BY_TASKPACK:
        path = REFS_BY_TASKPACK[taskpack_id]
    else:
        path = REFS_BY_TASKPACK["hwb_fl2_parametric_v1"]
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_reference_view(
    taskpack: Mapping[str, Any],
    *,
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge per-scenario outputs/tolerance from sidecar into reference_view."""
    ref = deepcopy(dict(taskpack.get("reference_view") or {}))
    sid = scenario_id
    if sid is None:
        params = (taskpack.get("solver_view") or {}).get("parameters") or {}
        sid = params.get("scenario_id")
    if not sid:
        return ref

    refs = load_scenario_refs(str(taskpack.get("taskpack_id") or ""))
    scenario_ref = refs.get("scenarios", {}).get(str(sid))
    if not scenario_ref:
        return ref

    outputs = dict(ref.get("outputs") or {})
    outputs.update(scenario_ref.get("outputs") or {})
    outputs["scenario_id"] = sid
    ref["outputs"] = outputs

    if scenario_ref.get("tolerance"):
        ref["tolerance"] = scenario_ref["tolerance"]
    if scenario_ref.get("secondary_tolerance"):
        ref["secondary_tolerance"] = scenario_ref["secondary_tolerance"]
    if scenario_ref.get("gate_policy"):
        ref["gate_policy"] = scenario_ref["gate_policy"]
    if scenario_ref.get("mozambique_gate_reference"):
        ref["mozambique_gate_reference"] = scenario_ref["mozambique_gate_reference"]
    if scenario_ref.get("mozambique_anomaly"):
        ref["mozambique_anomaly"] = scenario_ref["mozambique_anomaly"]

    prov = dict((ref.get("trajectory_constraints") or {}).get("provenance") or {})
    prov.update(scenario_ref.get("provenance") or {})
    prov["scenario_id"] = sid
    constraints = dict(ref.get("trajectory_constraints") or {})
    constraints["provenance"] = prov
    ref["trajectory_constraints"] = constraints
    return ref
