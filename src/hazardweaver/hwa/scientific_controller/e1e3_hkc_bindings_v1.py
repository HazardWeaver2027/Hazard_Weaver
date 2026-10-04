"""E1-E3 HKC frozen bindings — manifest RF families + scenario-gold gate requires."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import (
    _gold_from_single_route_scenario_id,
    e1e3_multi_scenario_capability_id,
)

E1E3_TRACK = "E1-E3"

CAP_TO_ROUTE_FAMILY: Dict[str, str] = {
    "CAP-E1E3-01": "RF-CLASSICAL-SIGNAL-TO-CHAIN",
    "CAP-E1E3-02": "RF-CNN-PHASE-PICKING",
    "CAP-E1E3-03": "RF-ATTENTION-RECURRENT-DETECTIO",
    "CAP-E1E3-04": "RF-WINDOW-CLASSIFIER-PICKER",
    "CAP-E1E3-05": "RF-OCTREE-TRAVEL-TIME-ASSOCIATI",
    "CAP-E1E3-06": "RF-PROBABILISTIC-MIXTURE-ASSOCI",
    "CAP-E1E3-07": "RF-GRID-SEARCH-ASSOCIATION",
    "CAP-E1E3-08": "RF-GMPE--OBSERVATION-FUSION",
}

E1E3_CAP_REQUIRE: Dict[str, str] = {
    "CAP-E1E3-01": "require:classical_sta_lta_travel_time_location_gmpe_chain",
    "CAP-E1E3-02": "require:deep_learning_cnn_phase_picking_pretrained",
    "CAP-E1E3-03": "require:attention_recurrent_joint_detection_phase_picking",
    "CAP-E1E3-04": "require:generative_window_classifier_phase_detection",
    "CAP-E1E3-05": "require:octree_travel_time_phase_association",
    "CAP-E1E3-06": "require:probabilistic_gaussian_mixture_phase_association",
    "CAP-E1E3-07": "require:grid_search_realtime_phase_association",
    "CAP-E1E3-08": "require:gmpe_observation_fusion_shakemap",
}

# Primary literature anchors (capabilities manifest + author repos).
CAP_PAPER_IDS: Dict[str, str] = {
    "CAP-E1E3-01": "doi:10.1785/BSSA.1978.68.5.1521",
    "CAP-E1E3-02": "doi:10.1785/0220190223",
    "CAP-E1E3-03": "doi:10.1038/s41586-019-1852-1",
    "CAP-E1E3-04": "doi:10.1785/0220180250",
    "CAP-E1E3-05": "doi:10.1785/0220210038",
    "CAP-E1E3-06": "doi:10.1785/0220210152",
    "CAP-E1E3-07": "doi:10.1785/0220190052",
    "CAP-E1E3-08": "doi:10.5066/F7D21VHZ",
}

E1E3_RF_FAMILIES = frozenset(CAP_TO_ROUTE_FAMILY.values())

E1E3_FROZEN_CONTRACTS_REL = (
    "runs/eskc_compiler/route_contracts/e1e3_manifest_rf_frozen_bindings_v1.jsonl"
)


def is_e1e3_track(track: str) -> bool:
    return str(track or "").upper().strip() == E1E3_TRACK


def is_e1e3_capability(capability_id: str) -> bool:
    return str(capability_id or "").upper().startswith("CAP-E1E3-")


def cap_to_route_family_id(capability_id: str) -> str:
    return CAP_TO_ROUTE_FAMILY.get(str(capability_id or "").upper().strip(), "")


def cap_to_require(capability_id: str) -> str:
    return E1E3_CAP_REQUIRE.get(str(capability_id or "").upper().strip(), "")


def cap_to_paper_id(capability_id: str) -> str:
    return CAP_PAPER_IDS.get(str(capability_id or "").upper().strip(), "")


def _hcg_scenario_is_decoy(inventory_row: Mapping[str, Any], scenario_cap: str) -> bool:
    """True when scenario gold CAP is also the HCG decoy (structurally inadmissible)."""
    cap = str(scenario_cap or "").upper().strip()
    if not cap:
        return False
    mech = str(inventory_row.get("ablation_mechanism") or "").strip().lower()
    if mech != "hcg":
        return False
    decoy = str(inventory_row.get("ablation_decoy_capability_id") or "").upper().strip()
    if cap == decoy:
        return True
    hcg_decoys = inventory_row.get("hcg_decoy_route_contracts") or {}
    return cap in {str(k).upper().strip() for k in hcg_decoys}


def e1e3_unified_gold_capability_id(inventory_row: Mapping[str, Any]) -> str:
    """Authoritative DCA gold for unified E1-E3 cells (multi + single_route).

    HCG multi: when scenario CAP equals the decoy (inadmissible), gold binds to witness.
    """
    scenario = e1e3_multi_scenario_capability_id(inventory_row)
    if scenario:
        if _hcg_scenario_is_decoy(inventory_row, scenario):
            witness = str(inventory_row.get("ablation_witness_capability_id") or "").strip()
            return witness or scenario
        return scenario
    gold = _gold_from_single_route_scenario_id(inventory_row)
    if gold:
        return gold
    allowed = [str(e) for e in (inventory_row.get("allowed_edge_ids") or []) if str(e).strip()]
    if len(allowed) == 1 and is_e1e3_capability(allowed[0]):
        return allowed[0]
    return ""


def e1e3_gold_capability_id(inventory_row: Mapping[str, Any]) -> str:
    """Scenario CAP gold for unified E1-E3 (multi + single_route)."""
    return e1e3_unified_gold_capability_id(inventory_row)


def _allowed_caps(inventory_row: Mapping[str, Any]) -> List[str]:
    allowed = [str(e) for e in (inventory_row.get("allowed_edge_ids") or []) if str(e).strip()]
    caps = [c for c in allowed if is_e1e3_capability(c)] if allowed else []
    gold = e1e3_gold_capability_id(inventory_row)
    if gold and gold not in caps:
        caps.append(gold)
    if caps:
        return caps
    return [gold] if gold else []


def e1e3_hkc_route_binding_entry(capability_id: str) -> Dict[str, Any]:
    cap = str(capability_id or "").upper().strip()
    req = cap_to_require(cap)
    rf = cap_to_route_family_id(cap)
    paper = cap_to_paper_id(cap)
    refs: List[str] = []
    if rf:
        refs.append(rf)
    if paper:
        refs.append(paper)
    entry: Dict[str, Any] = {
        "route_family_id": rf,
        "theory_constraints": [req] if req else [],
        "theory_alignment_refs": refs,
    }
    if paper:
        entry["paper_id"] = paper
    return entry


def e1e3_hkc_route_bindings_for_row(inventory_row: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not is_e1e3_track(str(inventory_row.get("track") or "")):
        return {}
    bindings: Dict[str, Dict[str, Any]] = {}
    for cap in _allowed_caps(inventory_row):
        bindings[cap] = e1e3_hkc_route_binding_entry(cap)
    return bindings


def e1e3_hkc_task_requires_for_row(inventory_row: Mapping[str, Any]) -> List[str]:
    if not is_e1e3_track(str(inventory_row.get("track") or "")):
        return []
    mech = str(inventory_row.get("ablation_mechanism") or "").strip().lower()
    if mech and mech != "hkc":
        return []
    gold = e1e3_gold_capability_id(inventory_row)
    req = cap_to_require(gold)
    return [req] if req else []


def apply_e1e3_hkc_task_overlays(
    task: MutableMapping[str, Any],
    inventory_row: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Stamp scenario-gold HKC bindings on unified / ablation E1-E3 cells."""
    if not is_e1e3_track(str(inventory_row.get("track") or "")):
        return task
    bindings = e1e3_hkc_route_bindings_for_row(inventory_row)
    requires = e1e3_hkc_task_requires_for_row(inventory_row)
    allowed = list(bindings.keys())
    meta = task.setdefault("metadata", {})
    if bindings:
        meta["ablation_hkc_route_bindings"] = bindings
        meta["e1e3_hkc_route_bindings"] = bindings
    if requires:
        meta["ablation_hkc_gate_requires"] = requires
        meta["e1e3_hkc_gate_requires"] = requires
    mech = str(inventory_row.get("ablation_mechanism") or "").strip().lower()
    if mech == "hkc" and requires:
        meta["e1e3_hkc_exact_require_grounding"] = True
    if allowed:
        inp = task.setdefault("solver_visible", {}).setdefault("inputs", {})
        inp["allowed_edge_ids"] = allowed
    gold = e1e3_gold_capability_id(inventory_row)
    if gold:
        meta["e1e3_scenario_gold_capability_id"] = gold
    return task


def e1e3_route_family_overlay(capability_id: str) -> Dict[str, str]:
    """Per-cap RF family stamp for g6 allowed-edge route synthesis."""
    cap = str(capability_id or "").upper().strip()
    if not is_e1e3_capability(cap):
        return {}
    rf = cap_to_route_family_id(cap)
    if not rf:
        return {}
    paper = cap_to_paper_id(cap)
    out: Dict[str, str] = {"route_family_id": rf, "hkc_family_id": rf}
    if paper:
        out["paper_id"] = paper
    return out
