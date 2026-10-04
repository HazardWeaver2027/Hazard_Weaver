"""Property-based V_q decomposition (Fusion property-based; witness not pass criterion)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from hazardweaver.hwb.evaluators.causality_firewall import check_v_causal
from hazardweaver.hwb.evaluators.execution_certificate import verify_trajectory_certificates

_FL2_SOLVER_CAPS = {f"CAP-FL2-{i:02d}" for i in range(1, 7)}
_OFFICIAL_EXEC_TOOLS = frozenset({"run_capability", "execute_capability"})


@dataclass
class VqPropertyResult:
    vq: bool
    v_integrity: bool
    v_causal: bool
    v_sci: bool
    v_cap: bool
    v_prov: bool
    errors: List[str] = field(default_factory=list)
    rejection_codes: List[str] = field(default_factory=list)
    matched_witness_id: Optional[str] = None  # diagnostic only, not pass criterion


def _allowed_edges(taskpack: Mapping[str, Any]) -> Set[str]:
    solver = taskpack.get("solver_view") or {}
    return {str(e) for e in (solver.get("allowed_edge_ids") or [])}


def _step_capability_ids(step: Mapping[str, Any]) -> List[str]:
    ids: List[str] = []
    if step.get("capability_id"):
        ids.append(str(step["capability_id"]))
    if step.get("adapter_id"):
        ids.append(str(step["adapter_id"]))
    return ids


def _trajectory_edge_set(steps: List[Mapping[str, Any]]) -> Set[str]:
    edges: Set[str] = set()
    for step in steps:
        if step.get("kind") == "submit":
            continue
        for cid in _step_capability_ids(step):
            edges.add(cid)
    return edges


def _fl2_solver_caps_from_execution_registry(trajectory: Mapping[str, Any]) -> Set[str]:
    """Official ``run_capability`` rows persisted under ``workdir/executions`` (HWA v2 ledger)."""
    meta = trajectory.get("hwa_run_meta") or {}
    wd_raw = str(meta.get("workdir") or "").strip()
    if not wd_raw:
        return set()
    wd = Path(wd_raw)
    if not wd.is_dir():
        return set()
    try:
        from hazardweaver.hwa.agent_runtime.execution_schema import list_registry_executions
    except ImportError:
        return set()

    caps: Set[str] = set()
    for row in list_registry_executions(wd):
        if str(row.get("status") or "") not in {"ok", "success"}:
            continue
        prov = row.get("provenance") if isinstance(row.get("provenance"), Mapping) else {}
        if str(prov.get("tool") or "") not in _OFFICIAL_EXEC_TOOLS:
            continue
        for cid in row.get("executed_capability_ids") or []:
            cap = str(cid or "").strip().split("__", 1)[0]
            if cap in _FL2_SOLVER_CAPS:
                caps.add(cap)
    return caps


def check_v_integrity(
    steps: List[Mapping[str, Any]],
    trajectory: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> List[str]:
    errors: List[str] = []
    forbidden = set(constraints.get("forbidden_shortcuts") or [])
    caps = _trajectory_edge_set(steps)
    route_id = str((trajectory.get("route_summary") or {}).get("route_id") or "")
    if "direct_submit_without_spread" in forbidden:
        spread_edges = {
            "eval_wf_activity_to_spread_v0",
            "hw_wildfire_spread_aspp_bridge_v0",
            "eval_wf_spread_to_mask_v0",
        }
        if not caps.intersection(spread_edges):
            errors.append("hidden_shortcut:direct_submit_without_spread")
    if "direct_submit_without_floodcast_infer" in forbidden:
        registry_caps = _fl2_solver_caps_from_execution_registry(trajectory)
        if not caps.intersection(_FL2_SOLVER_CAPS) and not registry_caps.intersection(_FL2_SOLVER_CAPS):
            errors.append("hidden_shortcut:direct_submit_without_floodcast_infer")
    if "hand_substitute_for_lisflood" in forbidden:
        if "lisflood" in route_id and "CAP-FL2-06" in caps and "CAP-FL2-05" not in caps:
            errors.append("hidden_shortcut:hand_substitute_for_lisflood")
    for step in steps:
        if step.get("kind") == "submit" and not step.get("ok", True):
            errors.append("failed_step_submit")
    if trajectory.get("read_hidden_witness") or trajectory.get("read_evaluator_view"):
        errors.append("integrity:read_hidden_reference")
    return errors


def check_v_cap(steps: List[Mapping[str, Any]], allowed: Set[str]) -> List[str]:
    errors: List[str] = []
    for step in steps:
        if step.get("kind") == "submit":
            continue
        for cid in _step_capability_ids(step):
            if cid and cid not in allowed:
                errors.append(f"invalid_capability:{cid}")
    return errors


def check_v_prov(steps: List[Mapping[str, Any]], trajectory: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if "final_artifact" not in trajectory and trajectory.get("terminal_action") not in ("abstain", "clarify"):
        errors.append("missing_final_artifact")
    exec_ids = {str(s.get("execution_id")) for s in steps if s.get("execution_id")}
    raw_prov = (trajectory.get("final_artifact") or {}).get("provenance")
    prov = raw_prov if isinstance(raw_prov, Mapping) else {}
    ref_exec = prov.get("execution_id")
    if ref_exec and (not exec_ids or str(ref_exec) not in exec_ids):
        errors.append("provenance:execution_id_not_in_steps")
    return errors


def check_v_sci(taskpack: Mapping[str, Any], steps: List[Mapping[str, Any]]) -> List[str]:
    ref = taskpack.get("reference_view") or {}
    forbidden_caps = set((ref.get("trajectory_constraints") or {}).get("forbidden_capabilities") or [])
    errors: List[str] = []
    for step in steps:
        cid = step.get("capability_id")
        if cid and str(cid) in forbidden_caps:
            errors.append(f"forbidden_capability:{cid}")
    return errors


def check_trajectory_vq_properties(
    taskpack: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    scenario: Optional[Mapping[str, Any]] = None,
) -> VqPropertyResult:
    """Property-based V_q = integrity ∧ causal ∧ sci ∧ cap ∧ prov."""
    errors: List[str] = []
    rejection_codes: List[str] = []

    steps = [s for s in (trajectory.get("steps") or []) if isinstance(s, Mapping)]
    ref = taskpack.get("reference_view") or {}
    constraints = ref.get("trajectory_constraints") or {}
    allowed = _allowed_edges(taskpack)
    cutoff = (scenario or {}).get("cutoff_manifest") or taskpack.get("cutoff_manifest") or {}

    e_int = check_v_integrity(steps, trajectory, constraints)
    e_cap = check_v_cap(steps, allowed)
    e_sci = check_v_sci(taskpack, steps)
    e_prov = check_v_prov(steps, trajectory)
    e_causal = check_v_causal(steps, cutoff_manifest=cutoff)
    cert_result = verify_trajectory_certificates(taskpack, trajectory, cutoff_manifest=cutoff)
    e_cert = cert_result.errors

    for bucket in (e_int, e_causal, e_sci, e_cap, e_prov, e_cert):
        errors.extend(bucket)
        for err in bucket:
            code = err.split(":")[0]
            if code not in rejection_codes:
                rejection_codes.append(code)

    v_integrity = len(e_int) == 0
    v_causal = len(e_causal) == 0
    v_sci = len(e_sci) == 0
    v_prov = len(e_prov) == 0 and not any(e.startswith("certificate:") for e in e_cert)
    v_cap = len(e_cap) == 0 and not any("capability_not" in e for e in e_cert)
    vq = v_integrity and v_causal and v_sci and v_cap and v_prov

    return VqPropertyResult(
        vq=vq,
        v_integrity=v_integrity,
        v_causal=v_causal,
        v_sci=v_sci,
        v_cap=v_cap,
        v_prov=v_prov,
        errors=errors,
        rejection_codes=rejection_codes,
        matched_witness_id=None,
    )
