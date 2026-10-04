"""Taskpack reference_view enrichment for parametric headline evaluation."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional


def _is_ablation_manual_pilot_row(inventory_row: Optional[Mapping[str, Any]]) -> bool:
    return str((inventory_row or {}).get("ablation_manual_pilot") or "") == "v1"


def executed_capability_id_from_trajectory(trajectory: Mapping[str, Any]) -> str:
    """Infer terminal capability from HWB submission trajectory (route-neutral)."""
    fa = trajectory.get("final_artifact") or {}
    value = fa.get("value") or {}
    if isinstance(value, Mapping):
        cap = str(value.get("capability_id") or "").strip()
        if cap:
            return cap
    for step in reversed(list(trajectory.get("steps") or [])):
        if not isinstance(step, Mapping):
            continue
        if str(step.get("kind") or "") == "submit":
            continue
        cap = str(step.get("capability_id") or step.get("adapter_id") or "").strip()
        if cap:
            return cap
    route_id = str((trajectory.get("route_summary") or {}).get("route_id") or "")
    match = re.search(r"CAP-[A-Z0-9-]+", route_id)
    if match:
        return match.group(0)
    return ""


def _apply_ablation_manual_pilot_eval_overrides(
    taskpack: Mapping[str, Any],
    *,
    inventory_row: Mapping[str, Any],
) -> Dict[str, Any]:
    """Ablation pilot: inventory whitelist is authoritative for V_q; gold binds at DCA to executed cap."""
    out = dict(taskpack)
    allowed = [str(edge).strip() for edge in (inventory_row.get("allowed_edge_ids") or []) if str(edge).strip()]
    if allowed:
        sv = dict(out.get("solver_view") or {})
        sv["allowed_edge_ids"] = allowed
        out["solver_view"] = sv
    ref_view = dict(out.get("reference_view") or {})
    tol_override = inventory_row.get("ablation_dca_tolerance")
    if isinstance(tol_override, Mapping):
        tol = dict(ref_view.get("tolerance") or {})
        tol.update(dict(tol_override))
        ref_view["tolerance"] = tol
    gate_override = inventory_row.get("ablation_gate_policy_override")
    if isinstance(gate_override, Mapping):
        ref_view["gate_policy"] = {**(ref_view.get("gate_policy") or {}), **dict(gate_override)}
    out["reference_view"] = ref_view
    return out


def _unified_scenario_gold_allowed_edge(
    inventory_row: Optional[Mapping[str, Any]],
) -> str:
    """Scenario-pinned gold CAP for unified benchmark V_q (overrides pilot whitelist)."""
    inv = inventory_row or {}
    if not inv.get("unified_benchmark_v1"):
        return ""
    try:
        from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import unified_scenario_gold_capability_id

        return str(unified_scenario_gold_capability_id(inv) or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _apply_solver_allowed_edges(
    taskpack: Mapping[str, Any],
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Mirror headline ``prepare_headline_solver_task`` edges into ``solver_view`` for V_q."""
    from hazardweaver.hwb.registry.solver_allowed_edges_v1 import resolve_solver_allowed_edge_ids

    out = dict(taskpack)
    inv = inventory_row or {}
    scenario_gold = _unified_scenario_gold_allowed_edge(inventory_row)
    if scenario_gold:
        if str(inv.get("route_eligibility") or "") == "multi_eligible":
            from hazardweaver.hwa.benchmark.unified_inventory_curator_v1 import unified_allowed_capability_ids

            allowed = unified_allowed_capability_ids(inv)
            if allowed:
                sv = dict(out.get("solver_view") or {})
                sv["allowed_edge_ids"] = allowed
                out["solver_view"] = sv
                return out
        sv = dict(out.get("solver_view") or {})
        sv["allowed_edge_ids"] = [scenario_gold]
        out["solver_view"] = sv
        return out
    if _is_ablation_manual_pilot_row(inventory_row):
        allowed = [
            str(edge).strip()
            for edge in ((inventory_row or {}).get("allowed_edge_ids") or [])
            if str(edge).strip()
        ]
        if allowed:
            sv = dict(out.get("solver_view") or {})
            sv["allowed_edge_ids"] = allowed
            out["solver_view"] = sv
            return out
    edges = resolve_solver_allowed_edge_ids(out, inventory_row=inventory_row)
    if not edges:
        return out
    sv = dict(out.get("solver_view") or {})
    if not (sv.get("allowed_edge_ids") or []):
        sv["allowed_edge_ids"] = edges
        out["solver_view"] = sv
    return out


def enrich_taskpack_for_evaluation(
    taskpack: Mapping[str, Any],
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
    final_artifact: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve scenario-specific reference_view for parametric taskpacks."""
    inv = inventory_row or {}
    if _is_ablation_manual_pilot_row(inv) and not inv.get("unified_benchmark_v1"):
        return _apply_ablation_manual_pilot_eval_overrides(taskpack, inventory_row=inv)
    taskpack_id = str(taskpack.get("taskpack_id") or inv.get("taskpack_id") or "")
    scenario_id = str(inv.get("scenario_id") or "")
    if not scenario_id and isinstance(final_artifact, Mapping):
        value = final_artifact.get("value") or {}
        if isinstance(value, Mapping):
            scenario_id = str(value.get("scenario_id") or "")
    out = dict(taskpack)
    if (
        str((out.get("metadata") or {}).get("hieraplan_eval_view") or "")
        and str((out.get("metadata") or {}).get("native_outcome_eval_v1") or "") != "native_outcome_eval_v1"
    ):
        return _apply_solver_allowed_edges(out, inventory_row=inventory_row)
    if scenario_id and taskpack_id:
        ref_view = taskpack.get("reference_view") or {}
        try:
            if taskpack_id.startswith("hwb_fl2"):
                from hazardweaver.hwb.registry.fl2_parametric_resolver import resolve_reference_view

                ref_view = resolve_reference_view(taskpack, scenario_id=scenario_id)
            elif "parametric" in taskpack_id or str(inv.get("source") or "").startswith(
                ("param:", "variant:")
            ):
                from hazardweaver.hwb.registry.track_parametric_resolver import resolve_reference_view

                ref_view = resolve_reference_view(taskpack, scenario_id=scenario_id)
            out["reference_view"] = ref_view
        except Exception:  # noqa: BLE001
            pass

    tol_override = inv.get("ablation_dca_tolerance")
    if isinstance(tol_override, Mapping):
        ref_view = dict(out.get("reference_view") or {})
        tol = dict(ref_view.get("tolerance") or {})
        tol.update(dict(tol_override))
        ref_view["tolerance"] = tol
        out["reference_view"] = ref_view

    unified_tol = inv.get("unified_dca_tolerance")
    if isinstance(unified_tol, Mapping) and inv.get("unified_benchmark_v1"):
        ref_view = dict(out.get("reference_view") or {})
        tol = dict(ref_view.get("tolerance") or {})
        tol.update(dict(unified_tol))
        ref_view["tolerance"] = tol
        out["reference_view"] = ref_view

    gate_override = inv.get("ablation_gate_policy_override")
    if isinstance(gate_override, Mapping):
        ref_view = dict(out.get("reference_view") or {})
        ref_view["gate_policy"] = {**(ref_view.get("gate_policy") or {}), **dict(gate_override)}
        out["reference_view"] = ref_view

    return _apply_solver_allowed_edges(out, inventory_row=inventory_row)
