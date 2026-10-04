"""MH-1..4 unified @143 HKC bindings — scenario-gold grounding (DL-218).

Fixes Tier-0 infra bugs:
- MH-1 S: ``pfdf_burn_volume_cascade`` must be solver-visible + DCA gold (atlas CRC).
- MH M-tier: ``hkc_task_requires`` / bindings must follow ``scenario_id`` CAP gold, not lazy witness default.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import unified_scenario_gold_capability_id

MH_TRACKS = frozenset({"MH-1", "MH-2", "MH-3", "MH-4"})
MH_MTRACKS = frozenset({"MH-2", "MH-3", "MH-4"})
PFDF_PORTFOLIO_CAPS = frozenset(
    {"pfdf_burn_volume_cascade", "pfdf_volume_gorr_v2", "burn_state_net_prithvi_v1"}
)

# Per-cap HKC stubs (aligned with ablation_manual_pilot curated templates).
MH_CAP_CATALOG: Dict[str, Dict[str, Any]] = {
    "pfdf_burn_volume_cascade": {
        "theory_constraints": ["require:burn_preconditioned_pfdf_volume_cascade"],
        "theory_alignment_refs": ["RF-TABULAR-PORTFOLIO-CASCADE"],
    },
    "pfdf_volume_gorr_v2": {
        "theory_constraints": ["require:tabular_pfdf_volume_gorr_terminal"],
        "theory_alignment_refs": ["RF-TABULAR-PORTFOLIO"],
    },
    "burn_state_net_prithvi_v1": {
        "theory_constraints": ["require:landsat_burn_state_prithvi_encoder"],
        "theory_alignment_refs": ["RF-DEEP-LEARNING-BURN-ENCODER"],
    },
    "CAP-MH1-01": {
        "theory_constraints": ["require:empirical_logistic_postfire_debris_flow_likelihood_at_watershed"],
        "theory_alignment_refs": ["RF-EMPIRICAL-LOGISTIC-PFDF-INIT"],
    },
    "CAP-MH1-02": {
        "theory_constraints": ["require:empirical_potential_debris_flow_volume_regression"],
        "theory_alignment_refs": ["RF-EMPIRICAL-VOLUME-REGRESSION"],
    },
    "CAP-MH2-01": {
        "theory_constraints": ["require:empirical_areal_landslide_coverage_logistic_model"],
        "theory_alignment_refs": ["RF-EMPIRICAL-AREAL-COVERAGE-LOGISTIC"],
    },
    "CAP-MH2-02": {
        "theory_constraints": ["require:empirical_cell_occurrence_logistic_model"],
        "theory_alignment_refs": ["RF-EMPIRICAL-CELL-OCCURRENCE-LO"],
    },
    "CAP-MH3-01": {
        "theory_constraints": ["require:reduced_physics_coupled_inundation_solver_at_scenario"],
        "theory_alignment_refs": ["RF-REDUCED-PHYSICS-COUPLED-INUND"],
    },
    "CAP-MH3-02": {
        "theory_constraints": ["require:two_dimensional_hydraulic_shallow_water_solver"],
        "theory_alignment_refs": ["RF-SHALLOW-WATER-SOLVER"],
    },
    "CAP-MH3-03": {
        "theory_constraints": ["require:inertial_hydrodynamic_raster_flood_solver"],
        "theory_alignment_refs": ["RF-INERTIAL-HYDRODYNAMIC-SOLVER"],
    },
    "CAP-MH4-R02": {
        "theory_constraints": ["require:independent_marginal_wind_surge_damage_sum"],
        "theory_alignment_refs": ["RF-INDEPENDENT-MARGINAL-DAMAGE"],
    },
    "CAP-MH4-R03": {
        "theory_constraints": ["require:dependence_aware_joint_wind_surge_damage_model"],
        "theory_alignment_refs": ["RF-DEPENDENCE-AWARE-JOINT-DAMAGE"],
    },
}


def is_mh_track(track: str) -> bool:
    return str(track or "").upper().strip() in MH_TRACKS


def _taskpack_for_row(
    inventory_row: Mapping[str, Any],
    taskpack: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if taskpack:
        return dict(taskpack)
    try:
        from hazardweaver.hwb.run.eval_dca_submission_v1 import resolve_taskpack_for_inventory_row

        return resolve_taskpack_for_inventory_row(dict(inventory_row))
    except Exception:  # noqa: BLE001
        return {}


def mh1_witness_gold_capability_id(
    inventory_row: Mapping[str, Any],
    *,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> str:
    """MH-1 atlas / PFDF single_route: witness ``edges[0]`` is scientific gold."""
    if str(inventory_row.get("track") or "").upper() != "MH-1":
        return ""
    if str(inventory_row.get("route_eligibility") or "") != "single_route":
        return ""
    tp = _taskpack_for_row(inventory_row, taskpack)
    witnesses = (tp.get("reference_view") or {}).get("accepted_witnesses") or []
    if not witnesses:
        return ""
    edges = witnesses[0].get("edges") or witnesses[0].get("capability_ids") or []
    if edges:
        return str(edges[0]).strip()
    return ""


def mh_unified_gold_capability_id(
    inventory_row: Mapping[str, Any],
    *,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> str:
    """Authoritative scenario gold for MH unified cells."""
    scenario = unified_scenario_gold_capability_id(inventory_row)
    if scenario:
        return scenario
    witness_gold = mh1_witness_gold_capability_id(inventory_row, taskpack=taskpack)
    if witness_gold:
        return witness_gold
    return ""


def _sorted_caps(caps: List[str]) -> List[str]:
    return sorted({str(c).strip() for c in caps if str(c).strip()})


def _allowed_caps(
    inventory_row: Mapping[str, Any],
    *,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    allowed = [str(e) for e in (inventory_row.get("allowed_edge_ids") or []) if str(e).strip()]
    if not allowed:
        tp = _taskpack_for_row(inventory_row, taskpack)
        sv = tp.get("solver_view") or {}
        allowed = [str(e) for e in (sv.get("allowed_edge_ids") or []) if str(e).strip()]
    gold = mh_unified_gold_capability_id(inventory_row, taskpack=taskpack)
    if gold and gold not in allowed:
        allowed.append(gold)
    return _sorted_caps(allowed)


def _catalog_binding(cap: str) -> Dict[str, Any]:
    return dict(MH_CAP_CATALOG.get(str(cap or "").strip(), {}))


def mh_hkc_route_bindings_for_row(
    inventory_row: Mapping[str, Any],
    *,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Scenario-gold HKC bindings; overrides witness-centric inventory stubs."""
    if not is_mh_track(str(inventory_row.get("track") or "")):
        return {}
    if not inventory_row.get("unified_benchmark_v1"):
        return dict(inventory_row.get("hkc_route_bindings") or {})

    gold = mh_unified_gold_capability_id(inventory_row, taskpack=taskpack)
    allowed = _allowed_caps(inventory_row, taskpack=taskpack)
    out: Dict[str, Dict[str, Any]] = {}
    for cap in allowed:
        entry = _catalog_binding(cap)
        if not entry and gold:
            inv_bind = dict((inventory_row.get("hkc_route_bindings") or {}).get(cap) or {})
            inv_bind.pop("scientifically_inapplicable", None)
            inv_bind.pop("inapplicability_reason", None)
            entry = inv_bind
        if cap == gold and entry:
            entry = {k: v for k, v in entry.items() if k not in {"scientifically_inapplicable", "inapplicability_reason"}}
        out[cap] = entry
    return out


def mh_hkc_task_requires_for_row(
    inventory_row: Mapping[str, Any],
    *,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    if not is_mh_track(str(inventory_row.get("track") or "")):
        return []
    mech = str(inventory_row.get("ablation_mechanism") or "").strip().lower()
    if mech not in {"hkc", "dual"}:
        return []
    if str(inventory_row.get("route_eligibility") or "") != "multi_eligible":
        return []
    gold = mh_unified_gold_capability_id(inventory_row, taskpack=taskpack)
    req = _catalog_binding(gold).get("theory_constraints") or []
    if req:
        return list(req)
    legacy = list(inventory_row.get("hkc_task_requires") or [])
    return legacy


def mh_pfdf_shortcut_inapplicable(cap: str, gold: str) -> bool:
    """Volume-only PFDF route is inadmissible when cascade burn preconditioning is gold."""
    return gold == "pfdf_burn_volume_cascade" and cap == "pfdf_volume_gorr_v2"


def curate_mh_unified_inventory_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Tier-0 curator: gold ∈ allowed + scenario-centric HKC on MH tracks."""
    from copy import deepcopy

    out = deepcopy(dict(row))
    if not out.get("unified_benchmark_v1") or not is_mh_track(str(out.get("track") or "")):
        return out
    tp = _taskpack_for_row(out)
    allowed = _allowed_caps(out, taskpack=tp)
    gold = mh_unified_gold_capability_id(out, taskpack=tp)
    bindings = mh_hkc_route_bindings_for_row(out, taskpack=tp)
    if gold and gold in PFDF_PORTFOLIO_CAPS:
        for cap in list(allowed):
            if mh_pfdf_shortcut_inapplicable(cap, gold):
                bind = dict(bindings.get(cap) or _catalog_binding(cap))
                bind["scientifically_inapplicable"] = True
                bind["inapplicability_reason"] = "theory_mismatch"
                bindings[cap] = bind
    out["allowed_edge_ids"] = allowed
    out["hkc_route_bindings"] = bindings
    requires = mh_hkc_task_requires_for_row(out, taskpack=tp)
    if requires:
        out["hkc_task_requires"] = requires
    if gold:
        out["unified_scenario_gold_capability_id"] = gold
    prior = row.get("allowed_edge_ids") or []
    out["mh_unified_inventory_curation_v1"] = {
        "schema": "mh_unified_inventory_curation_v1",
        "allowed_before": list(prior),
        "allowed_after": allowed,
        "gold": gold,
        "gold_in_allowed": bool(gold and gold in allowed),
    }
    return out


def apply_mh_unified_task_overlays(
    task: MutableMapping[str, Any],
    inventory_row: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Runtime overlay for MH unified cells (mirrors E1-E3 + mtrack curator)."""
    if not inventory_row.get("unified_benchmark_v1"):
        return task
    if not is_mh_track(str(inventory_row.get("track") or "")):
        return task

    curated = curate_mh_unified_inventory_row(inventory_row)
    allowed = list(curated.get("allowed_edge_ids") or [])
    bindings = dict(curated.get("hkc_route_bindings") or {})
    requires = list(curated.get("hkc_task_requires") or [])
    gold = str(curated.get("unified_scenario_gold_capability_id") or "").strip()

    meta = task.setdefault("metadata", {})
    if bindings:
        meta["ablation_hkc_route_bindings"] = bindings
        meta["mh_hkc_route_bindings"] = bindings
    if requires:
        meta["ablation_hkc_gate_requires"] = requires
        meta["mh_hkc_gate_requires"] = requires
    mech = str(inventory_row.get("ablation_mechanism") or "").strip().lower()
    if (
        mech in {"hkc", "dual"}
        and str(inventory_row.get("route_eligibility") or "") == "multi_eligible"
        and requires
    ):
        meta["mh_hkc_exact_require_grounding"] = True
    if gold:
        meta["unified_scenario_gold_capability_id"] = gold
        meta["mh_scenario_gold_capability_id"] = gold
    if allowed:
        inp = task.setdefault("solver_visible", {}).setdefault("inputs", {})
        inp["allowed_edge_ids"] = allowed
    return task
