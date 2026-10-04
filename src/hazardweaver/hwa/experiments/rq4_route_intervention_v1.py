"""RQ4 route-space interventions (E8–E14) — task perturbation + runtime hooks."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from hazardweaver.hwb.registry.solver_allowed_edges_v1 import resolve_solver_allowed_edge_ids
from hazardweaver.hwb.run.eval_dca_submission_v1 import resolve_taskpack_for_inventory_row

RQ4_CATEGORIES: Sequence[str] = (
    "E8_scientific_condition",
    "E9_missing_required_data",
    "E10_unavailable_capability",
    "E11_invalid_intermediate_artifact",
    "E12_execution_failure",
    "E13_resource_budget_change",
    "E14_mh_coupling_change",
)

STATIC_CONTROL_CATEGORY = "static_control"


def rq4_static_control_enabled() -> bool:
    try:
        from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import route_eligibility_static_frozen

        if route_eligibility_static_frozen():
            return True
    except ImportError:
        pass
    raw = (
        os.environ.get("HWA_RQ4_STATIC_CONTROL")
        or os.environ.get("ICLR_RQ4_STATIC_CONTROL")
        or ""
    ).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def intervention_spec_from_row(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    spec = row.get("rq4_intervention")
    if isinstance(spec, Mapping) and spec.get("category"):
        return dict(spec)
    return None


def _primary_capability(row: Mapping[str, Any]) -> str:
    try:
        tp = resolve_taskpack_for_inventory_row(row)
        edges = resolve_solver_allowed_edge_ids(tp, inventory_row=row)
        caps = [str(e) for e in edges if e and not str(e).startswith("schema_map")]
        if caps:
            return caps[0]
    except Exception:  # noqa: BLE001
        pass
    return ""


def build_intervention_spec(
    row: Mapping[str, Any],
    *,
    category: str,
    static_control: bool = False,
) -> Dict[str, Any]:
    cap = _primary_capability(row)
    track = str(row.get("track") or "")
    spec: Dict[str, Any] = {
        "category": category,
        "static_control": static_control,
        "target_capability_id": cap,
        "track": track,
    }
    if category == "E8_scientific_condition":
        spec["blocked_capability_ids"] = [cap] if cap else []
        spec["note"] = "Primary route capability marked scientifically inapplicable at s0."
    elif category == "E9_missing_required_data":
        spec["drop_input_keys"] = ["scenario_id"]
    elif category == "E10_unavailable_capability":
        spec["unavailable_capability_ids"] = [cap] if cap else []
    elif category == "E11_invalid_intermediate_artifact":
        spec["inject_unit_mismatch"] = True
    elif category == "E12_execution_failure":
        spec["force_capability_failure"] = cap
    elif category == "E13_resource_budget_change":
        spec["max_steps_override"] = 6
        spec["max_wall_s_override"] = 600.0
    elif category == "E14_mh_coupling_change":
        spec["mh_coupling_delta"] = {"operator_id": "scale_fraction_mod_high", "factor": 1.1}
    return spec


def apply_rq4_intervention_to_task(
    task: MutableMapping[str, Any],
    spec: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Mutate solver_visible / metadata for a single intervention cell."""
    category = str(spec.get("category") or "")
    meta = task.setdefault("metadata", {})
    meta["rq4_intervention"] = dict(spec)
    meta["rq4_static_control"] = bool(spec.get("static_control")) or rq4_static_control_enabled()

    sv = task.setdefault("solver_visible", {})
    constraints = sv.setdefault("constraints", {})
    rq4 = constraints.setdefault("rq4_intervention", {})
    rq4.update(dict(spec))

    inputs = sv.setdefault("inputs", {})
    params = inputs.setdefault("parameters", {})

    if category == "E8_scientific_condition":
        blocked = [str(c) for c in (spec.get("blocked_capability_ids") or []) if c]
        if blocked:
            allowed = [str(e) for e in (inputs.get("allowed_edge_ids") or []) if str(e)]
            inputs["allowed_edge_ids"] = [e for e in allowed if e not in set(blocked)]
            rq4["removed_from_allowed_edges"] = blocked

    elif category == "E9_missing_required_data":
        for key in spec.get("drop_input_keys") or []:
            params.pop(str(key), None)
            inputs.pop(str(key), None)

    elif category == "E10_unavailable_capability":
        unavailable = [str(c) for c in (spec.get("unavailable_capability_ids") or []) if c]
        sv["unavailable_capability_ids"] = unavailable

    elif category == "E11_invalid_intermediate_artifact":
        rq4["type_incompatibility"] = {
            "kind": "unit_mismatch",
            "expected_unit": "m",
            "actual_unit": "ft",
        }
        meta["headline_rq4_type_mismatch_probe"] = True

    elif category == "E12_execution_failure":
        forced = str(spec.get("force_capability_failure") or "")
        rq4["force_capability_failure"] = forced
        defer_s0 = bool(spec.get("defer_s0_invalidation"))
        rq4["defer_s0_invalidation"] = defer_s0
        blocked = [str(c) for c in (spec.get("defer_s0_blocked_capabilities") or []) if c]
        if blocked:
            rq4["defer_s0_blocked_capabilities"] = blocked
        narrow = bool(spec.get("defer_s0_narrow_allowed_to_shock")) and defer_s0 and forced
        rq4["defer_s0_narrow_allowed_to_shock"] = narrow
        allowed = [str(e) for e in (inputs.get("allowed_edge_ids") or []) if str(e).strip()]
        existing_full = [
            str(e).strip()
            for e in (meta.get("rq4_full_allowed_edge_ids") or [])
            if str(e).strip()
        ]
        # W3 E12: curator whitelist may already be stamped before s0 narrow (e.g.
        # ablation_manual_pilot narrows solver_visible first). Never shrink full_allowed
        # to the shock-only s0 surface — post-shock refresh needs the full inventory.
        if existing_full:
            meta["rq4_full_allowed_edge_ids"] = list(existing_full)
        elif allowed:
            meta["rq4_full_allowed_edge_ids"] = list(allowed)
        if narrow and forced:
            inputs["allowed_edge_ids"] = [forced]
            rq4["s0_allowed_edge_ids"] = [forced]
            meta["rq4_defer_s0_narrow_active"] = True
        elif blocked and defer_s0:
            blocked_set = set(blocked)
            inputs["allowed_edge_ids"] = [e for e in allowed if e not in blocked_set]
        if forced and not spec.get("static_control") and not defer_s0:
            try:
                from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import src_eligibility_gate_off

                if not src_eligibility_gate_off():
                    meta["rq4_s0_invalidated_capabilities"] = [forced]
            except ImportError:
                pass

    elif category == "E13_resource_budget_change":
        if spec.get("max_steps_override") is not None:
            meta["rq4_max_steps_override"] = int(spec["max_steps_override"])
        if spec.get("max_wall_s_override") is not None:
            meta["rq4_max_wall_s_override"] = float(spec["max_wall_s_override"])

    elif category == "E14_mh_coupling_change":
        delta = dict(spec.get("mh_coupling_delta") or {})
        if delta:
            params["mh_coupling_intervention"] = delta
            inputs["mh1_burn_preconditioning"] = delta

    return task


def apply_rq4_intervention_from_row(
    task: MutableMapping[str, Any],
    inventory_row: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    spec = intervention_spec_from_row(inventory_row)
    if not spec:
        return task
    out = copy.deepcopy(dict(task))
    return apply_rq4_intervention_to_task(out, spec)


def maybe_rq4_execute_block(
    name: str,
    args: Mapping[str, Any],
    task: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Runtime hook: unavailable caps, forced failures (E10/E12)."""
    meta = task.get("metadata") or {}
    sv = task.get("solver_visible") or {}
    rq4 = (sv.get("constraints") or {}).get("rq4_intervention") or meta.get("rq4_intervention") or {}
    if not rq4:
        return None

    if name != "run_capability":
        return None

    cid = str(args.get("capability_id") or "")
    unavailable = {str(c) for c in (sv.get("unavailable_capability_ids") or rq4.get("unavailable_capability_ids") or [])}
    if cid and cid in unavailable:
        return {
            "ok": False,
            "error": "rq4_capability_unavailable",
            "capability_id": cid,
            "category": rq4.get("category"),
        }

    forced = str(rq4.get("force_capability_failure") or "")
    if forced and cid == forced:
        return {
            "ok": False,
            "error": "rq4_forced_execution_failure",
            "capability_id": cid,
            "category": rq4.get("category"),
        }
    return None


def rq4_budget_overrides(task: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    meta = task.get("metadata") or {}
    steps = meta.get("rq4_max_steps_override")
    wall = meta.get("rq4_max_wall_s_override")
    out: Dict[str, Optional[float]] = {"max_steps": None, "max_wall_s": None}
    if steps is not None:
        out["max_steps"] = float(int(steps))
    if wall is not None:
        out["max_wall_s"] = float(wall)
    return out


def rq4_skip_reinstantiate(task: Mapping[str, Any]) -> bool:
    if rq4_static_control_enabled():
        return True
    meta = task.get("metadata") or {}
    return bool(meta.get("rq4_static_control"))
