"""Curate unified @143 inventory rows — scenario gold reachable without runtime pin (DL-203)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import (
    unified_scenario_gold_capability_id,
    witness_capability_id,
)

THREE_MTRACKS = frozenset({"HW-MED", "L2", "E1-E3"})

# Per-cap HKC stubs for v100 scenario gold caps beyond ablation witness/decoy pair.
_MTRACK_HKC_STUBS: Dict[str, Dict[str, Any]] = {
    "CAP-HWMED-03": {
        "theory_constraints": ["require:learned_spectral_neural_weather_medium_range"],
        "theory_alignment_refs": ["RF-NEURAL-WEATHER"],
    },
    "CAP-HWMED-04": {
        "theory_constraints": ["require:operational_physics_nwp_medium_range"],
        "theory_alignment_refs": ["RF-PHYSICS-NWP"],
    },
    "CAP-HWMED-05": {
        "theory_constraints": ["require:learned_spectral_neural_weather_medium_range"],
        "theory_alignment_refs": ["RF-NEURAL-WEATHER"],
    },
    "CAP-HWMED-06": {
        "theory_constraints": ["require:operational_physics_nwp_medium_range"],
        "theory_alignment_refs": ["RF-PHYSICS-NWP"],
    },
    "CAP-L2-03": {
        "theory_constraints": ["require:ml_rainfall_threshold_landslide"],
        "theory_alignment_refs": ["RF-ML-RAINFALL"],
    },
    "CAP-L2-04": {
        "theory_constraints": ["require:physics_rainfall_threshold_landslide"],
        "theory_alignment_refs": ["RF-PHYSICS-RAINFALL"],
    },
    "CAP-L2-05": {
        "theory_constraints": ["require:ml_rainfall_threshold_landslide"],
        "theory_alignment_refs": ["RF-ML-RAINFALL"],
    },
}


def _sorted_caps(caps: List[str]) -> List[str]:
    return sorted({str(c).strip() for c in caps if str(c).strip()})


def unified_allowed_capability_ids(row: Mapping[str, Any]) -> List[str]:
    """Solver-visible whitelist: inventory allowed ∪ scenario gold (no runtime pin)."""
    allowed = [str(e) for e in (row.get("allowed_edge_ids") or []) if str(e).strip()]
    gold = unified_scenario_gold_capability_id(row)
    if gold and gold not in allowed:
        allowed.append(gold)
    return _sorted_caps(allowed)


def _hkc_binding_for_cap(row: Mapping[str, Any], cap: str) -> Dict[str, Any]:
    bindings = dict(row.get("hkc_route_bindings") or {})
    existing = dict(bindings.get(cap) or {})
    if existing.get("scientifically_inapplicable"):
        existing = {k: v for k, v in existing.items() if k != "scientifically_inapplicable"}
        existing.pop("inapplicability_reason", None)
    if existing and not existing.get("scientifically_inapplicable"):
        return existing
    stub = _MTRACK_HKC_STUBS.get(cap)
    if stub:
        return dict(stub)
    witness = witness_capability_id(row)
    if witness and witness in bindings:
        wit = dict(bindings[witness])
        wit.pop("scientifically_inapplicable", None)
        wit.pop("inapplicability_reason", None)
        if wit:
            return wit
    return {
        "theory_constraints": list(row.get("hkc_task_requires") or []),
        "theory_alignment_refs": [],
    }


def curate_hkc_route_bindings(row: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Ensure every allowed cap (incl. scenario gold) is HKC-admissible on full HWA."""
    if str(row.get("track") or "") not in THREE_MTRACKS:
        return dict(row.get("hkc_route_bindings") or {})
    out: Dict[str, Dict[str, Any]] = {
        str(k): dict(v) for k, v in (row.get("hkc_route_bindings") or {}).items()
    }
    for cap in unified_allowed_capability_ids(row):
        entry = _hkc_binding_for_cap(row, cap)
        if entry.get("scientifically_inapplicable"):
            entry = {k: v for k, v in entry.items() if k not in {"scientifically_inapplicable", "inapplicability_reason"}}
        out[cap] = entry
    return out


def curate_solver_visible_route_briefs(row: Mapping[str, Any]) -> Dict[str, str]:
    briefs = dict(row.get("solver_visible_route_briefs") or row.get("ablation_route_briefs") or {})
    for cap in unified_allowed_capability_ids(row):
        if cap not in briefs:
            briefs[cap] = f"Scientific capability route {cap} for this scenario."
    return briefs


def curate_unified_inventory_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a curated copy safe for sealed @143 inventory (Tier 0)."""
    out = deepcopy(dict(row))
    if not out.get("unified_benchmark_v1"):
        return out
    track = str(out.get("track") or "")
    if track not in THREE_MTRACKS:
        return out
    allowed = unified_allowed_capability_ids(out)
    out["allowed_edge_ids"] = allowed
    out["hkc_route_bindings"] = curate_hkc_route_bindings(out)
    out["solver_visible_route_briefs"] = curate_solver_visible_route_briefs(out)
    gold = unified_scenario_gold_capability_id(out)
    if gold:
        out["unified_scenario_gold_capability_id"] = gold
    prior = row.get("allowed_edge_ids") or []
    out["unified_inventory_curation_v1"] = {
        "schema": "unified_inventory_curation_v1",
        "allowed_before": list(prior),
        "allowed_after": allowed,
        "gold_in_allowed": bool(gold and gold in allowed),
        "gold_was_missing_from_inventory": bool(gold and gold not in prior),
    }
    return out


def apply_mtrack_unified_task_overlays(
    task: MutableMapping[str, Any],
    inventory_row: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Runtime overlay: HW-MED/L2 allowed ∪ gold + curated HKC (mirrors E1-E3 overlay)."""
    track = str(inventory_row.get("track") or "")
    if track not in {"HW-MED", "L2"}:
        return task
    if not inventory_row.get("unified_benchmark_v1"):
        return task
    curated = curate_unified_inventory_row(inventory_row)
    allowed = list(curated.get("allowed_edge_ids") or [])
    bindings = dict(curated.get("hkc_route_bindings") or {})
    briefs = dict(curated.get("solver_visible_route_briefs") or {})
    meta = task.setdefault("metadata", {})
    if bindings:
        meta["ablation_hkc_route_bindings"] = bindings
        meta["mtrack_hkc_route_bindings"] = bindings
    gold = str(curated.get("unified_scenario_gold_capability_id") or "").strip()
    if gold:
        meta["unified_scenario_gold_capability_id"] = gold
    if allowed:
        inp = task.setdefault("solver_visible", {}).setdefault("inputs", {})
        inp["allowed_edge_ids"] = allowed
    if briefs:
        inp = task.setdefault("solver_visible", {}).setdefault("inputs", {})
        inp["route_briefs"] = briefs
    return task


def audit_row_solvability(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Audit one row for Tier-0 solvability (gold ∈ allowed, HKC gold admissible)."""
    gold = unified_scenario_gold_capability_id(row)
    allowed = [str(e) for e in (row.get("allowed_edge_ids") or []) if str(e).strip()]
    bindings = dict(row.get("hkc_route_bindings") or {})
    gold_bind = bindings.get(gold) or {}
    return {
        "instance_id": row.get("instance_id"),
        "track": row.get("track"),
        "route_eligibility": row.get("route_eligibility"),
        "gold": gold,
        "gold_in_allowed": bool(gold and gold in allowed),
        "n_allowed": len(allowed),
        "gold_hkc_inapplicable": bool(gold_bind.get("scientifically_inapplicable")),
        "structurally_unsolvable": bool(
            gold
            and str(row.get("route_eligibility") or "") == "multi_eligible"
            and gold not in allowed
        ),
    }
