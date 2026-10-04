"""Graph reachability wrapping with session-state filtering."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwa.agent_runtime.hcg_tools import make_default_store, tool_hcg_find_paths
from hazardweaver.hwa.scientific_controller.admissibility import route_from_hcg_path
from hazardweaver.hwa.scientific_controller.state import SessionState


def _path_blocked(path_info: Mapping[str, Any], state: SessionState) -> bool:
    edges = list(path_info.get("edges") or [])
    for eid in edges:
        if eid in state.invalidated_edges or eid in state.invalidated_capabilities:
            return True
    return False


def find_reachable_paths(
    sources: Sequence[str],
    target: str,
    state: SessionState,
    *,
    store: Any = None,
    max_paths: int = 8,
    packs: Optional[Sequence[str]] = None,
    executable_only: bool = True,
    require_all_sources: bool = False,
) -> List[Dict[str, Any]]:
    """Enumerate HCG paths and filter invalidated edges from s_k."""
    graph_store = resolve_graph_store(
        store,
        state,
        packs=packs,
        sources=sources,
    )
    raw = tool_hcg_find_paths(
        sources,
        target,
        store=graph_store,
        max_paths=max_paths,
        packs=None,
        executable_only=executable_only,
        require_all_sources=require_all_sources,
    )
    out: List[Dict[str, Any]] = []
    for pinfo in raw:
        if _path_blocked(pinfo, state):
            continue
        route = route_from_hcg_path(pinfo)
        route["available"] = list(pinfo.get("available") or [])
        route["executable"] = bool(pinfo.get("valid", True))
        out.append(route)
    return out


def init_state_from_task(task: Mapping[str, Any]) -> SessionState:
    solver = task.get("solver_visible") or {}
    inputs = solver.get("inputs") or {}
    refs = inputs.get("sample_refs") or []
    sources: List[str] = []
    for ref in refs:
        if isinstance(ref, Mapping):
            sid = ref.get("sample_id") or ref.get("record_id")
            if sid:
                sources.append(str(sid))
    goal_artifacts = list(inputs.get("goal_artifacts") or [])
    target = str(goal_artifacts[0]) if goal_artifacts else ""
    graph_sources = [str(s) for s in (inputs.get("graph_sources") or []) if str(s).strip()]
    if graph_sources and not sources:
        sources = graph_sources
    packs = list((solver.get("pack_refs") or []))
    if not packs:
        inv = solver.get("allowed_inventory") or {}
        packs = list(inv.get("hcg_packs") or [])

    # HCG compose default: map fixture sample refs to graph artifact ids
    if target == "wf_postfire_risk_index" and not any(
        s.startswith("wf_") and s not in {"fixture_wf_hard"} for s in sources
    ):
        sources = ["wf_firms_hotspots", "wf_landfire_fuel"]

    state = SessionState(
        sources=sources,
        target=target,
        packs=packs,
        theory_arm=str((solver.get("theory_arm") or "verified")),
    )
    state.available_artifacts.update(sources)
    for ga in goal_artifacts:
        state.available_artifacts.add(str(ga))
    meta = task.get("metadata") or {}
    for cid in meta.get("rq4_s0_invalidated_capabilities") or []:
        cap = str(cid or "").strip()
        if cap:
            state.invalidated_capabilities.add(cap)
            state.invalidated_edges.add(cap)
    return state


def _expand_headline_hcg_packs(
    packs: Sequence[str],
    *,
    sources: Optional[Sequence[str]] = None,
) -> List[str]:
    """manifest_rf_v1 route cards need headline + seven-track graph nodes merged."""
    ordered = list(packs)
    if "manifest_rf_v1" in ordered:
        for extra in ("iclr_headline_v1", "hcg_7track_v1"):
            if extra not in ordered:
                ordered.append(extra)
    src = {str(s) for s in (sources or []) if str(s).strip()}
    if "pfdf_v1_held" not in ordered and src.intersection(
        {
            "landsat_pre_post_patch_v1",
            "usgs_pfdf_record_v1",
            "log_volume_v1",
            "burn_severity_summary_v1",
        }
    ):
        ordered.append("pfdf_v1_held")
    if not ordered:
        ordered = ["graph_eval_v0"]
    return ordered


def _store_with_packs(store: Any, packs: Sequence[str]) -> Any:
    """Ensure merged packs are present on the graph store (controller store may be stale)."""
    packs_list = list(packs)
    if store is None:
        return make_default_store(packs=packs_list)
    for name in packs_list:
        try:
            store.merge_pack(str(name))
        except Exception:  # noqa: BLE001
            continue
    return store


def resolve_graph_store(
    store: Any,
    state: SessionState,
    *,
    packs: Optional[Sequence[str]] = None,
    sources: Optional[Sequence[str]] = None,
) -> Any:
    """Merge session packs/sources onto a graph store (enumerate + repair paths)."""
    packs_list = _expand_headline_hcg_packs(
        list(packs or state.packs or []),
        sources=sources or state.sources,
    )
    return _store_with_packs(store, packs_list)


def make_store_for_task(
    task: Mapping[str, Any],
    *,
    include_holdouts: bool = False,
    state: Optional[SessionState] = None,
) -> Any:
    st = state or init_state_from_task(task)
    if state is None:
        from hazardweaver.hwa.runtime.runtime_closure import hydrate_session_graph_from_task

        hydrate_session_graph_from_task(task, st)
    packs = _expand_headline_hcg_packs(st.packs or [], sources=st.sources)
    return make_default_store(include_holdouts=include_holdouts, packs=packs)
