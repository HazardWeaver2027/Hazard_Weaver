"""HWA default-runtime methodology closure helpers (P0 wiring, not experiments)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def g6_committed_allowed_edge_dispatch(host: Any, capability_id: str) -> bool:
    """Controller-committed g6_single_hop routes may execute portfolio/registry caps."""
    from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import (
        _allowed_edges,
        uses_allowed_edge_route_synthesis,
    )

    ctrl = getattr(host, "controller", None)
    if ctrl is None:
        return False
    task = getattr(ctrl, "task", None) or getattr(host, "task", None) or {}
    if not isinstance(task, Mapping):
        return False
    cid = str(capability_id or "").strip()
    if not cid:
        return False
    state = getattr(ctrl, "state", None)
    rid = str(getattr(state, "active_route_id", "") or "").strip() if state is not None else ""
    if rid:
        route = (getattr(ctrl, "_route_cache", {}) or {}).get(rid) or {}
        if route.get("g6_coreexec"):
            caps = [str(c) for c in (route.get("capability_ids") or route.get("edges") or []) if str(c)]
            if cid in caps or cid in set(_allowed_edges(task)):
                return True
    if not uses_allowed_edge_route_synthesis(task):
        return False
    if cid not in set(_allowed_edges(task)):
        return False
    if not rid:
        return False
    route = (getattr(ctrl, "_route_cache", {}) or {}).get(rid) or {}
    return bool(route.get("g6_coreexec"))


def legacy_bypass_allowed(host: Any) -> bool:
    """Non-headline CapabilityLoader bypass is opt-in only (ReAct / ablation arms)."""
    task = getattr(host, "task", None) or {}
    if not isinstance(task, Mapping):
        return True
    sv = task.get("solver_visible") or {}
    if sv.get("allow_legacy_bypass") is True:
        return True
    if sv.get("legacy_mode") is True:
        return True
    if task.get("allow_legacy_bypass") is True:
        return True
    return False


def controller_gate_active() -> bool:
    from hazardweaver.hwa.agent_runtime import unified_tools as ut

    return getattr(ut, "_CONTROLLER_GATE", None) is not None


def methodology_closure_flags(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Read-only summary for audit scripts and seal reports."""
    sv = task.get("solver_visible") or {}
    legacy = bool(sv.get("legacy_mode"))
    return {
        "controller_default": not legacy,
        "legacy_react_arm": legacy,
        "allow_legacy_bypass": legacy_bypass_allowed(type("H", (), {"task": task})()),
        "pfdf_src_pilot": _is_pfdf_src(task),
        "headline_route_mode": (task.get("metadata") or {}).get("headline_route_mode"),
    }


def _is_pfdf_src(task: Mapping[str, Any]) -> bool:
    try:
        from hazardweaver.hwa.route_controller.pfdf_pilot_slice import is_pfdf_pilot_task

        return bool(is_pfdf_pilot_task(task)) and not bool((task.get("solver_visible") or {}).get("legacy_mode"))
    except Exception:  # noqa: BLE001
        return False


def init_pfdf_controller_state(controller: Any) -> None:
    """Align PFDF SRC pilot graph sources/target with bench controller (D-PFDF-1)."""
    hydrate_session_graph_from_task(getattr(controller, "task", None) or {}, controller.state)
    sync_controller_graph_store(controller)


def hydrate_session_graph_from_task(task: Mapping[str, Any], state: Any) -> None:
    """Inject track/PFDF graph sources, target, and HCG packs into session state."""
    from hazardweaver.hwa.experiments.headline_route_mode_v1 import infer_track_graph_config
    from hazardweaver.hwa.route_controller.pfdf_pilot_slice import is_pfdf_pilot_task

    meta = task.get("metadata") or {}
    if str(meta.get("headline_route_mode") or "") == "hcg_multi_hop":
        track = str(meta.get("track") or task.get("headline_target") or "")
        graph_cfg = infer_track_graph_config(track, task)
        if graph_cfg:
            sources = list(graph_cfg.get("graph_sources") or [])
            if not sources and str(meta.get("headline_route_mode") or "") == "hcg_multi_hop":
                from hazardweaver.hwa.experiments.headline_route_mode_v1 import _MANIFEST_RF_GRAPH_SOURCES

                sources = list(_MANIFEST_RF_GRAPH_SOURCES)
            if sources and not list(state.sources or []):
                state.sources = sources
            target = str(graph_cfg.get("graph_target") or "")
            if target and not str(state.target or "").strip():
                state.target = target
            pack = str(graph_cfg.get("hcg_pack") or "")
            packs = list(state.packs or [])
            if pack and pack not in packs:
                state.packs = packs + [pack]
            return
    if is_pfdf_pilot_task(task):
        from hazardweaver.hwa.route_controller.pfdf_pilot_slice import (
            PFDF_GRAPH_SOURCES,
            PFDF_GRAPH_TARGET,
            PFDF_HCG_PACK,
        )

        if not list(state.sources or []):
            state.sources = list(PFDF_GRAPH_SOURCES)
        if not str(state.target or "").strip():
            state.target = PFDF_GRAPH_TARGET
        packs = list(state.packs or [])
        if PFDF_HCG_PACK not in packs:
            state.packs = packs + [PFDF_HCG_PACK]


def sync_controller_graph_store(controller: Any) -> None:
    """Merge session packs onto controller.store after graph-state hydration."""
    from hazardweaver.hwa.scientific_controller.reachability import resolve_graph_store

    state = controller.state
    controller.store = resolve_graph_store(
        getattr(controller, "store", None),
        state,
        packs=state.packs,
        sources=state.sources,
    )


def init_headline_graph_state(controller: Any) -> None:
    """Inject graph sources/target/packs for headline multi-hop (Path C, DL-133)."""
    hydrate_session_graph_from_task(getattr(controller, "task", None) or {}, controller.state)
    sync_controller_graph_store(controller)


def counterfactual_hint_for_route(
    route: Optional[Mapping[str, Any]],
    *,
    controller: Any,
) -> Dict[str, Any]:
    """UNKNOWN / not-admissible → HCG counterfactual reachability (P1-4 wiring)."""
    if route is None:
        return {}
    a_sci = route.get("A_sci") or {}
    verdict = ""
    if isinstance(a_sci, Mapping):
        verdict = str(a_sci.get("verdict") or "")
    from hazardweaver.hwa.scientific_controller.reason_codes import ASciVerdict

    if verdict not in {
        ASciVerdict.UNKNOWN_PENDING_THEORY.value,
        "UNKNOWN",
        "SCI_UNKNOWN_PENDING_THEORY",
    }:
        return {}
    from hazardweaver.hwa.route_controller.adaptation import counterfactual_recovery

    routes = list(getattr(controller, "_route_cache", {}).values())
    return counterfactual_recovery(
        routes,
        controller.state,
        {"route_id": route.get("route_id")},
        task=controller.task,
        graph=getattr(controller, "graph", None),
    )
