"""Resolve parametric PFDF / DR-OUT / TC-TRK HWB taskpack reference views by scenario_id."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

def _seven_track_parametric_refs() -> Dict[str, Path]:
    from hazardweaver.hwb.registry.seven_track_parametric_v1 import SEVEN_TRACK_PARAMETRIC_IDS, parametric_refs_path

    out: Dict[str, Path] = {}
    for taskpack_id in SEVEN_TRACK_PARAMETRIC_IDS:
        slug = taskpack_id.replace("hwb_", "").replace("_parametric_v1", "")
        out[taskpack_id] = parametric_refs_path(slug)
    return out


REFS_BY_TASKPACK = {
    "hwb_pfdf_parametric_v1": Path(__file__).resolve().parent
    / "taskpacks"
    / "hwb_pfdf_parametric_v1_refs.json",
    "hwb_drout_parametric_v1": Path(__file__).resolve().parent
    / "taskpacks"
    / "hwb_drout_parametric_v1_refs.json",
    "hwb_tctrk_parametric_v1": Path(__file__).resolve().parent
    / "taskpacks"
    / "hwb_tctrk_parametric_v1_refs.json",
    "hwb_mh1_parametric_v1": Path(__file__).resolve().parent
    / "taskpacks"
    / "hwb_mh1_parametric_v1_refs.json",
    "hwb_mh1_atlas_state_variant_v1": Path(__file__).resolve().parent
    / "taskpacks"
    / "hwb_mh1_atlas_state_variant_v1_refs.json",
    **_seven_track_parametric_refs(),
}


def _refs_path_for_taskpack(taskpack: Mapping[str, Any]) -> Path:
    tid = str(taskpack.get("taskpack_id") or "")
    if tid in REFS_BY_TASKPACK:
        return REFS_BY_TASKPACK[tid]
    sidecar = (taskpack.get("reference_view") or {}).get("refs_sidecar")
    if sidecar:
        return Path(__file__).resolve().parent / "taskpacks" / str(sidecar)
    raise KeyError(f"no parametric refs for taskpack: {tid}")


def load_scenario_refs(taskpack_id: Optional[str] = None) -> Dict[str, Any]:
    if not taskpack_id or taskpack_id not in REFS_BY_TASKPACK:
        return {}
    path = REFS_BY_TASKPACK[taskpack_id]
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _base_scenario_id(scenario_id: str) -> str:
    """Strip difficulty suffix e.g. CAP-MH4-R01__L3 → CAP-MH4-R01."""
    sid = str(scenario_id or "").strip()
    if "__" in sid:
        return sid.split("__", 1)[0]
    return sid


def _canonicalize_reference_outputs(
    outputs: Dict[str, Any],
    tolerance: Optional[Mapping[str, Any]] = None,
) -> None:
    """Mirror primary metric scalar into ``reference_score`` for DCA/VCE auditors."""
    if outputs.get("reference_score") is not None:
        return
    metric = str((tolerance or {}).get("metric") or outputs.get("metric_name") or "").strip()
    if metric and outputs.get(metric) is not None:
        try:
            outputs["reference_score"] = float(outputs[metric])
        except (TypeError, ValueError):
            pass
    elif outputs.get("rmse_depth") is not None:
        outputs["reference_score"] = float(outputs["rmse_depth"])
    elif outputs.get("lead_error_km") is not None:
        outputs["reference_score"] = float(outputs["lead_error_km"])


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

    taskpack_id = str(taskpack.get("taskpack_id") or "")
    try:
        refs_path = _refs_path_for_taskpack(taskpack)
        refs = json.loads(refs_path.read_text(encoding="utf-8"))
    except KeyError:
        refs = load_scenario_refs(taskpack_id)
    lookup = _base_scenario_id(str(sid))
    scenario_ref = (refs.get("scenarios") or {}).get(lookup) or {}
    if not scenario_ref and taskpack_id == "hwb_hw_med_parametric_v1":
        from hazardweaver.hwb.registry.hwmed_agent_replay_reference_v1 import (
            HW_MED_TASKPACK_ID,
            hwmed_agent_replay_eval_authority,
            hwmed_replay_scenario_ref_for_capability,
        )

        if taskpack_id == HW_MED_TASKPACK_ID and hwmed_agent_replay_eval_authority(None):
            try:
                scenario_ref = hwmed_replay_scenario_ref_for_capability(lookup)
            except KeyError:
                scenario_ref = {}
    if not scenario_ref and taskpack_id == "hwb_mh1_atlas_state_variant_v1":
        from hazardweaver.hwb.registry.mh1_atlas_record_gold_v1 import try_mh1_record_scenario_ref

        scenario_ref = try_mh1_record_scenario_ref(lookup) or {}
    if not scenario_ref and taskpack_id == "hwb_tctrk_parametric_v1":
        from hazardweaver.hwb.registry.tctrk_atlas_storm_gold_v1 import try_tctrk_storm_scenario_ref

        scenario_ref = try_tctrk_storm_scenario_ref(lookup) or {}
    if not scenario_ref:
        return ref

    outputs = dict(ref.get("outputs") or {})
    outputs.update(scenario_ref.get("outputs") or {})
    outputs["scenario_id"] = sid
    _canonicalize_reference_outputs(outputs, ref.get("tolerance") or scenario_ref.get("tolerance"))
    ref["outputs"] = outputs

    if scenario_ref.get("tolerance"):
        ref["tolerance"] = scenario_ref["tolerance"]
    if scenario_ref.get("capability_id"):
        ref["capability_id"] = scenario_ref["capability_id"]

    prov = dict((ref.get("trajectory_constraints") or {}).get("provenance") or {})
    prov.update(scenario_ref.get("provenance") or {})
    prov["scenario_id"] = sid
    constraints = dict(ref.get("trajectory_constraints") or {})
    constraints["provenance"] = prov
    ref["trajectory_constraints"] = constraints
    return ref


def _bind_reference_view_to_capability(
    taskpack: Mapping[str, Any],
    cap: str,
) -> Dict[str, Any]:
    """Merge sidecar gold/tolerance for one capability id into reference_view."""
    tp = dict(taskpack)
    cap_id = str(cap or "").strip()
    if not cap_id:
        return tp
    taskpack_id = str(tp.get("taskpack_id") or "")
    refs = load_scenario_refs(taskpack_id)
    lookup = _base_scenario_id(cap_id)
    scenario_ref = (refs.get("scenarios") or {}).get(lookup) or (refs.get("scenarios") or {}).get(cap_id) or {}
    if not scenario_ref and taskpack_id == "hwb_hw_med_parametric_v1":
        from hazardweaver.hwb.registry.hwmed_agent_replay_reference_v1 import (
            hwmed_agent_replay_eval_authority,
            hwmed_replay_scenario_ref_for_capability,
        )

        if hwmed_agent_replay_eval_authority(None):
            try:
                scenario_ref = hwmed_replay_scenario_ref_for_capability(lookup or cap_id)
            except KeyError:
                scenario_ref = {}
    if not scenario_ref:
        return tp
    ref = dict(tp.get("reference_view") or {})
    if scenario_ref.get("tolerance"):
        ref["tolerance"] = dict(scenario_ref["tolerance"])
    if scenario_ref.get("outputs"):
        # Replace (do not merge) — prior scenario resolve may leave foreign metric keys.
        outputs = dict(scenario_ref["outputs"])
        _canonicalize_reference_outputs(outputs, ref.get("tolerance") or scenario_ref.get("tolerance"))
        ref["outputs"] = outputs
    ref["capability_id"] = cap_id
    tp["reference_view"] = ref
    return tp


def align_reference_view_to_bound_capability(
    taskpack: Mapping[str, Any],
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Atlas/parametric rows: bind gold + tolerance to the resolved allowed capability."""
    from hazardweaver.hwb.registry.solver_allowed_edges_v1 import resolve_solver_allowed_edge_ids

    tp = dict(taskpack)
    edges = resolve_solver_allowed_edge_ids(tp, inventory_row=inventory_row)
    if len(edges) != 1:
        return tp
    return _bind_reference_view_to_capability(tp, str(edges[0]).strip())


def align_reference_view_to_executed_capability(
    taskpack: Mapping[str, Any],
    *,
    executed_capability_id: str,
) -> Dict[str, Any]:
    """After route execute: bind gold/tolerance to the terminal capability that ran."""
    cap = str(executed_capability_id or "").strip()
    if not cap:
        return dict(taskpack)
    return _bind_reference_view_to_capability(dict(taskpack), cap)
