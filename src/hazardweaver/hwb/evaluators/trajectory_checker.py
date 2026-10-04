"""Refactor trajectory_checker to property-based V_q ()."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from hazardweaver.hwb.evaluators.vq_properties import VqPropertyResult, check_trajectory_vq_properties
from hazardweaver.hwb.registry.g6_hard_anchor_meta import taskpack_id_for_g6_task

SCHEMA_VERSION = "HWA_TRAJECTORY_v1"
REQUIRED_ADAPTER_KIND = "adapter"


@dataclass
class TrajectoryVqResult:
    vq: bool
    matched_witness_id: Optional[str]
    errors: List[str] = field(default_factory=list)
    rejection_codes: List[str] = field(default_factory=list)
    properties: Optional[VqPropertyResult] = None


def _allowed_edges(taskpack: Mapping[str, Any]) -> Set[str]:
    solver = taskpack.get("solver_view") or {}
    edges = solver.get("allowed_edge_ids") or []
    return {str(e) for e in edges}


def _validate_schema(trajectory: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    ver = trajectory.get("schema_version")
    if ver not in (SCHEMA_VERSION, "HWA_TRAJECTORY_v2"):
        errors.append("invalid_trajectory_schema")
    if not isinstance(trajectory.get("steps"), list) or not trajectory["steps"]:
        if trajectory.get("terminal_action") not in ("abstain", "clarify"):
            errors.append("invalid_trajectory_schema:empty_steps")
    terminal = trajectory.get("terminal_action")
    if terminal not in ("abstain", "clarify") and "final_artifact" not in trajectory:
        errors.append("invalid_trajectory_schema:missing_final_artifact")
    return errors


def _step_capability_ids(step: Mapping[str, Any]) -> List[str]:
    ids: List[str] = []
    if step.get("capability_id"):
        ids.append(str(step["capability_id"]))
    if step.get("adapter_id"):
        ids.append(str(step["adapter_id"]))
    return ids


def _witness_edge_set(witness: Mapping[str, Any]) -> Set[str]:
    return {str(e) for e in (witness.get("edges") or [])}


def _trajectory_edge_set(steps: List[Mapping[str, Any]]) -> Set[str]:
    edges: Set[str] = set()
    for step in steps:
        if step.get("kind") == "submit":
            continue
        for cid in _step_capability_ids(step):
            edges.add(cid)
    return edges


def _check_dependency_order(steps: List[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    indices = [int(s.get("step_index", -1)) for s in steps]
    if indices != sorted(indices):
        errors.append("dependency_order_violation:non_monotonic_step_index")
    if len(indices) != len(set(indices)):
        errors.append("dependency_order_violation:duplicate_step_index")
    return errors


def _check_provenance(steps: List[Mapping[str, Any]], constraints: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    prov_req = constraints.get("provenance") or {}
    for key in ("crs", "units", "time_window"):
        if prov_req.get(key) and not any((s.get("provenance") or {}).get(key) for s in steps):
            errors.append(f"unit_crs_time_violation:{key}")
    return errors


def _check_required_adapters(steps: List[Mapping[str, Any]], constraints: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    required = {str(a) for a in (constraints.get("required_adapters") or [])}
    seen: Set[str] = set()
    for step in steps:
        aid = step.get("adapter_id")
        if aid:
            seen.add(str(aid))
        cap = step.get("capability_id")
        if cap and str(cap) in (constraints.get("adapter_edges") or {}):
            spec = (constraints.get("adapter_edges") or {})[str(cap)]
            if spec.get("requires_explicit_adapter_id") and not aid:
                errors.append(f"implicit_adapter:{cap}")
    missing = required - seen
    cap_ids = {str(s.get("capability_id") or "") for s in steps}
    for adapter in sorted(missing):
        if adapter not in cap_ids:
            errors.append(f"implicit_adapter:missing_required:{adapter}")
    return errors


def _match_witness_legacy(
    steps: List[Mapping[str, Any]],
    witnesses: List[Mapping[str, Any]],
    allowed: Set[str],
) -> tuple[Optional[str], List[str]]:
    """Legacy diagnostic witness match — NOT used for property-based pass/fail."""
    traj_edges = _trajectory_edge_set(steps)
    errors: List[str] = []
    for step in steps:
        if step.get("kind") == "submit":
            continue
        for cid in _step_capability_ids(step):
            if cid and cid not in allowed:
                errors.append(f"invalid_capability:{cid}")
    matched: Optional[str] = None
    for witness in witnesses:
        w_edges = _witness_edge_set(witness)
        if w_edges <= traj_edges:
            matched = str(witness.get("witness_id"))
            break
    if matched is None and witnesses:
        errors.append("no_matching_witness")
    return matched, errors


def check_trajectory_vq(
    taskpack: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    vq_mode: str = "property",
    scenario: Optional[Mapping[str, Any]] = None,
) -> TrajectoryVqResult:
    """
    V_q checker.

    vq_mode='property' (default): property-based, no witness-edge pass criterion.
    vq_mode='legacy': deprecated witness superset matching.
    """
    errors: List[str] = []
    rejection_codes: List[str] = []

    taskpack_id = str(taskpack.get("taskpack_id") or "")
    tid = str(trajectory.get("task_id") or "")
    source_g6 = str((taskpack.get("source") or {}).get("g6_task_id") or "")
    ref_g6 = str((taskpack.get("reference_view") or {}).get("source_g6_task_id") or "")
    allowed_task_ids = {
        taskpack_id,
        taskpack_id.replace("hwb_", ""),
        source_g6,
        ref_g6,
        tid,
    }
    if tid.startswith("H_A_"):
        allowed_task_ids.add(taskpack_id_for_g6_task(tid))
    if tid and tid not in allowed_task_ids:
        errors.append("task_id_mismatch")
        rejection_codes.append("task_id_mismatch")

    schema_errors = _validate_schema(trajectory)
    errors.extend(schema_errors)
    rejection_codes.extend(schema_errors)
    if schema_errors:
        return TrajectoryVqResult(vq=False, matched_witness_id=None, errors=errors, rejection_codes=rejection_codes)

    steps = [s for s in trajectory["steps"] if isinstance(s, Mapping)]
    ref = taskpack.get("reference_view") or {}
    constraints = ref.get("trajectory_constraints") or {}

    errors.extend(_check_dependency_order(steps))
    errors.extend(_check_provenance(steps, constraints))
    errors.extend(_check_required_adapters(steps, constraints))

    if vq_mode == "legacy":
        allowed = _allowed_edges(taskpack)
        witnesses = ref.get("accepted_witnesses") or []
        matched, witness_errors = _match_witness_legacy(steps, witnesses, allowed)
        errors.extend(witness_errors)
        for err in errors:
            code = err.split(":")[0]
            if code not in rejection_codes:
                rejection_codes.append(code)
        return TrajectoryVqResult(
            vq=len(errors) == 0,
            matched_witness_id=matched,
            errors=errors,
            rejection_codes=rejection_codes,
        )

    props = check_trajectory_vq_properties(taskpack, trajectory, scenario=scenario)
    errors.extend(props.errors)
    rejection_codes.extend(props.rejection_codes)
    for err in errors:
        code = err.split(":")[0]
        if code not in rejection_codes:
            rejection_codes.append(code)

    return TrajectoryVqResult(
        vq=props.vq and len(errors) == 0,
        matched_witness_id=props.matched_witness_id,
        errors=errors,
        rejection_codes=rejection_codes,
        properties=props,
    )


def load_trajectory(path: Path | str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
