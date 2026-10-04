"""Headline route mode — Path C tier-aware dispatch (DL-133)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from hazardweaver.hwa.route_controller.drout_pilot_slice import (
    DR_OUT_GRAPH_TARGET,
    DR_OUT_HCG_PACK,
    DR_OUT_PILOT_FAMILY,
)
from hazardweaver.hwa.route_controller.pfdf_pilot_slice import (
    PFDF_GRAPH_SOURCES,
    PFDF_GRAPH_TARGET,
    PFDF_HCG_PACK,
    PFDF_PILOT_FAMILY,
)
from hazardweaver.hwa.route_controller.seven_track_pilot_slice import SEVEN_TRACK_REGISTRY
from hazardweaver.hwa.route_controller.tctrk_pilot_slice import (
    TC_TCTRK_GRAPH_TARGET,
    TC_TCTRK_HCG_PACK,
    TC_TCTRK_PILOT_FAMILY,
)
from hazardweaver.hwa.experiments.headline_route_profile_v1 import is_g1_profile, cross_system_cbr_allowed

RouteMode = str  # "g6_single_hop" | "hcg_multi_hop"

# Tracks with no multi-cap rows in headline218 — still need 11-track paper breadth.
_SINGLE_CAP_BREADTH_TRACKS = frozenset({"HW-MED", "L2", "E1-E3", "MH-2", "MH-4", "WF-3"})

# manifest_rf_v1 bindings use registry_stub_input_v1 for most RF headline routes.
_MANIFEST_RF_GRAPH_SOURCES: Tuple[str, ...] = ("registry_stub_input_v1",)
_EQ_E1E3_GRAPH_SOURCES: Tuple[str, ...] = ("eq_waveform_3c",)

# Extended 11-track graph registry (seven-track + FL-2/MH-1/DR-OUT/TC-TRK)
_HEADLINE_TRACK_GRAPH: Dict[str, Dict[str, Any]] = {}

for _tp, _cfg in SEVEN_TRACK_REGISTRY.items():
    sources: Tuple[str, ...] = _MANIFEST_RF_GRAPH_SOURCES
    if _tp == "E1-E3":
        sources = _EQ_E1E3_GRAPH_SOURCES
    _HEADLINE_TRACK_GRAPH[_tp] = {
        "family_id": str(_cfg["family_id"]),
        "domain": str(_cfg.get("domain") or _tp.lower()),
        "graph_target": str(_cfg["graph_target"]),
        "hcg_pack": str(_cfg["hcg_pack"]),
        "graph_sources": sources,
    }

_HEADLINE_TRACK_GRAPH["MH-1"] = {
    "family_id": PFDF_PILOT_FAMILY,
    "domain": "pfdf",
    "graph_target": PFDF_GRAPH_TARGET,
    "hcg_pack": PFDF_HCG_PACK,
    "graph_sources": tuple(PFDF_GRAPH_SOURCES),
}
_HEADLINE_TRACK_GRAPH["FL-2"] = {
    "family_id": "fl2_v1",
    "domain": "fl2",
    "graph_target": "fl2_depth_grid_v1",
    "hcg_pack": "manifest_rf_v1",
    "graph_sources": _MANIFEST_RF_GRAPH_SOURCES,
}
_HEADLINE_TRACK_GRAPH["DR-OUT"] = {
    "family_id": DR_OUT_PILOT_FAMILY,
    "domain": "drout",
    "graph_target": DR_OUT_GRAPH_TARGET,
    "hcg_pack": DR_OUT_HCG_PACK,
    "graph_sources": _MANIFEST_RF_GRAPH_SOURCES,
}
_HEADLINE_TRACK_GRAPH["TC-TRK"] = {
    "family_id": TC_TCTRK_PILOT_FAMILY,
    "domain": "tctrk",
    "graph_target": TC_TCTRK_GRAPH_TARGET,
    "hcg_pack": TC_TCTRK_HCG_PACK,
    "graph_sources": _MANIFEST_RF_GRAPH_SOURCES,
}

_HCG_READONLY_TOOLS: Tuple[str, ...] = (
    "tool_hcg_find_paths",
    "tool_hcg_explain_edge",
)


def infer_track_graph_config(
    track: str,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    tr = str(track or (taskpack or {}).get("headline_target") or "").upper().strip()
    if tr in _HEADLINE_TRACK_GRAPH:
        return dict(_HEADLINE_TRACK_GRAPH[tr])
    tp_id = str((taskpack or {}).get("taskpack_id") or "").upper()
    for key, cfg in _HEADLINE_TRACK_GRAPH.items():
        if tp_id.startswith(f"HWB_{key.replace('-', '')}") or key.replace("-", "") in tp_id:
            return dict(cfg)
    return None


def _capability_edges(allowed_edges: List[str]) -> List[str]:
    return [e for e in allowed_edges if e and not str(e).startswith("schema_map")]


def resolve_headline_route_mode(
    inventory_row: Mapping[str, Any],
    taskpack: Mapping[str, Any],
    *,
    allowed_edges: Optional[List[str]] = None,
) -> RouteMode:
    """Return execution dispatch mode for headline full_hwa cells."""
    forced = str(inventory_row.get("headline_route_mode") or "").strip()
    if forced in {"hcg_multi_hop", "g6_single_hop"}:
        return forced

    if inventory_row.get("controller_ablation_v2"):
        track = str(inventory_row.get("track") or taskpack.get("headline_target") or "")
        graph = infer_track_graph_config(track, taskpack)
        caps = _capability_edges(allowed_edges or [])
        min_caps = 1 if track in _SINGLE_CAP_BREADTH_TRACKS else 2
        if graph and len(caps) >= min_caps:
            return "hcg_multi_hop"

    tier = str(inventory_row.get("difficulty_tier") or "L1").upper()
    meta = taskpack.get("metadata") or {}
    if meta.get("requires_multi_hop") or meta.get("headline_requires_multi_hop"):
        return "hcg_multi_hop"
    if tier in {"L1", "L2", "L3"}:
        return "g6_single_hop"
    if tier == "L4":
        if is_g1_profile() and cross_system_cbr_allowed():
            return "g6_single_hop"
        return "hcg_multi_hop"
    edges = allowed_edges or []
    caps = _capability_edges(edges)
    if len(caps) > 1:
        return "hcg_multi_hop"
    return "g6_single_hop"


def headline_route_class(route_mode: RouteMode) -> str:
    if route_mode == "hcg_multi_hop":
        return "hcg_multi_hop_dag"
    return "edge_routable"


def apply_graph_config_to_solver_visible(
    solver_visible: Dict[str, Any],
    graph_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Inject goal_artifacts / hcg_packs for multi-hop controller enumeration."""
    out = dict(solver_visible)
    inputs = dict(out.get("inputs") or {})
    goal = str(graph_cfg.get("graph_target") or "")
    if goal:
        inputs["goal_artifacts"] = [goal]
    sources = list(graph_cfg.get("graph_sources") or [])
    if sources:
        inputs["graph_sources"] = sources
    out["inputs"] = inputs
    pack = str(graph_cfg.get("hcg_pack") or "")
    allowed_inv = dict(out.get("allowed_inventory") or {})
    if pack:
        allowed_inv["hcg_packs"] = [pack]
        out["pack_refs"] = [pack]
    out["allowed_inventory"] = allowed_inv
    out["hkc_family_id"] = str(graph_cfg.get("family_id") or "")
    out["route_family_id"] = str(graph_cfg.get("family_id") or "")
    return out


def hcg_readonly_tool_ids() -> List[str]:
    return list(_HCG_READONLY_TOOLS)
