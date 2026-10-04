"""Seven-track vertical slice registry — HKC default binding (DL-107)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from hazardweaver.hwa.scientific_controller.hkc_pilot_assets import (
    SEVEN_TRACK_FAMILIES,
    SEVEN_TRACK_TASKPACK_TO_FAMILY,
    resolve_family_slice_paths,
)

TrackConfig = Dict[str, Any]

SEVEN_TRACK_REGISTRY: Dict[str, TrackConfig] = {
    "WF-3": {
        "family_id": "wf3_process_v1",
        "domain": "wf3",
        "default_cap": "CAP-WF3-01",
        "cap_prefix": "CAP-WF3-",
        "hcg_pack": "manifest_rf_v1",
        "graph_target": "burn_state_grid_v1",
        "hcg_route_families": frozenset({"RF-DEEP-LEARNING-SEGMENTATION", "RF-STATISTICAL-SPREAD-MODEL"}),
    },
    "L2": {
        "family_id": "l2_v1",
        "domain": "l2",
        "default_cap": "CAP-L2-01",
        "cap_prefix": "CAP-L2-",
        "hcg_pack": "manifest_rf_v1",
        "graph_target": "landslide_debris_bundle_v1",
        "hcg_route_families": frozenset({"RF-PHYSICS-BASED-RUNOUT", "RF-EMPIRICAL-VOLUME-ESTIMATE"}),
    },
    "E1-E3": {
        "family_id": "eq_e1e3_v1",
        "domain": "eq",
        "default_cap": "CAP-E1E3-01",
        "cap_prefix": "CAP-E1E3-",
        "hcg_pack": "manifest_rf_v1",
        "graph_target": "eq_catalog_source_groundmotion_bundle_v1",
        "hcg_route_families": frozenset({"RF-ATTENTION-RECURRENT-DETECTIO", "RF-PHASE-PICKING-LEARNED"}),
    },
    "HW-MED": {
        "family_id": "hw_med_v1",
        "domain": "hwmed",
        "default_cap": "CAP-HWMED-01",
        "cap_prefix": "CAP-HWMED-",
        "hcg_pack": "manifest_rf_v1",
        "graph_target": "heatwave_medium_bundle_v1",
        "hcg_route_families": frozenset({"RF-NEURAL-FORECAST-MEDIUM", "RF-STATISTICAL-HEATWAVE-BASELINE"}),
    },
    "MH-2": {
        "family_id": "mh2_v1",
        "domain": "mh2",
        "default_cap": "CAP-MH2-01",
        "cap_prefix": "CAP-MH2-",
        "hcg_pack": "manifest_rf_v1",
        "graph_target": "mh_trigger_bundle_v1",
        "hcg_route_families": frozenset({"RF-EARTHQUAKE-TRIGGERING", "RF-COUPLED-HAZARD-CHAIN"}),
    },
    "MH-3": {
        "family_id": "mh3_v1",
        "domain": "mh3",
        "default_cap": "CAP-MH3-01",
        "cap_prefix": "CAP-MH3-",
        "hcg_pack": "manifest_rf_v1",
        "graph_target": "mh_compound_bundle_v1",
        "hcg_route_families": frozenset({"RF-COMPOUND-FLOOD-FIRE", "RF-MULTI-HAZARD-GRAPH"}),
    },
    "MH-4": {
        "family_id": "mh4_v1",
        "domain": "mh4",
        "default_cap": "CAP-MH4-01",
        "cap_prefix": "CAP-MH4-",
        "hcg_pack": "manifest_rf_v1",
        "graph_target": "mh_amplification_bundle_v1",
        "hcg_route_families": frozenset({"RF-TROPICAL-CYCLONE-LOSS", "RF-TC-EXPOSURE-AMPLIFICATION"}),
    },
}

_FAMILY_TO_TASKPACK = {cfg["family_id"]: tp for tp, cfg in SEVEN_TRACK_REGISTRY.items()}


def taskpack_for_family(family_id: str) -> str:
    return _FAMILY_TO_TASKPACK.get(str(family_id or "").strip(), "")


def config_for_taskpack(taskpack: str) -> Optional[TrackConfig]:
    return SEVEN_TRACK_REGISTRY.get(str(taskpack or "").upper().strip())


def config_for_family(family_id: str) -> Optional[TrackConfig]:
    tp = taskpack_for_family(family_id)
    return config_for_taskpack(tp) if tp else None


def is_seven_track_pilot_task(task: Mapping[str, Any]) -> bool:
    domain = str(task.get("domain") or "").lower()
    family = str(task.get("task_family") or "").upper()
    tp = str(task.get("taskpack_id") or "").upper()
    hkc = str(
        task.get("hkc_family_id")
        or (task.get("solver_visible") or {}).get("hkc_family_id")
        or ""
    )
    cap = str(task.get("capability_id") or "").upper()
    if hkc in SEVEN_TRACK_FAMILIES:
        return True
    if tp in SEVEN_TRACK_REGISTRY or family in SEVEN_TRACK_REGISTRY:
        return True
    for cfg in SEVEN_TRACK_REGISTRY.values():
        prefix = str(cfg.get("cap_prefix") or "")
        if prefix and cap.startswith(prefix):
            return True
    if any(x in domain for x in ("wf3", "landslide", "l2", "eq", "hwmed", "mh2", "mh3", "mh4")):
        return True
    return False


def resolve_seven_track_family_id(
    task: Mapping[str, Any],
    route: Optional[Mapping[str, Any]] = None,
) -> str:
    from hazardweaver.hwa.scientific_controller.e1e3_hkc_bindings_v1 import (
        E1E3_RF_FAMILIES,
        cap_to_route_family_id,
        is_e1e3_capability,
    )

    if route is not None:
        hkc = str(route.get("hkc_family_id") or route.get("family_id") or "")
        if hkc in E1E3_RF_FAMILIES:
            return hkc
        if hkc in SEVEN_TRACK_FAMILIES:
            return hkc
        rf = str(route.get("route_family_id") or "")
        if rf in E1E3_RF_FAMILIES:
            return rf
        caps = {str(c).upper() for c in (route.get("capability_ids") or route.get("edges") or [])}
        for cap in caps:
            if is_e1e3_capability(cap):
                fam = cap_to_route_family_id(cap)
                if fam:
                    return fam
        for cfg in SEVEN_TRACK_REGISTRY.values():
            fam = str(cfg["family_id"])
            if rf in {fam, *cfg.get("hcg_route_families", ())}:
                return fam
            prefix = str(cfg.get("cap_prefix") or "")
            if prefix and any(c.startswith(prefix) for c in caps):
                return fam
    if is_seven_track_pilot_task(task):
        tp = str(task.get("taskpack_id") or task.get("task_family") or "").upper()
        cfg = config_for_taskpack(tp)
        if cfg:
            return str(cfg["family_id"])
        hkc = str(
            task.get("hkc_family_id")
            or (task.get("solver_visible") or {}).get("hkc_family_id")
            or ""
        )
        if hkc in SEVEN_TRACK_FAMILIES:
            return hkc
        cap = str(task.get("capability_id") or "").upper()
        for cfg in SEVEN_TRACK_REGISTRY.values():
            prefix = str(cfg.get("cap_prefix") or "")
            if prefix and cap.startswith(prefix):
                return str(cfg["family_id"])
    solver = task.get("solver_visible") or {}
    return str(task.get("hkc_family_id") or solver.get("hkc_family_id") or "")


def build_seven_track_src_pilot_task(
    track: str,
    *,
    task_id: str = "",
    capability_id: str = "",
    scenario_id: str = "pilot_scenario_000",
) -> Dict[str, Any]:
    tp = str(track).upper()
    cfg = config_for_taskpack(tp)
    if cfg is None:
        raise KeyError(f"unknown seven-track: {track}")
    fam = str(cfg["family_id"])
    cap = str(capability_id or cfg.get("default_cap") or "").upper()
    tid = task_id or f"{cfg['domain']}-src-pilot"
    return {
        "task_id": tid,
        "taskpack_id": tp,
        "domain": str(cfg["domain"]),
        "task_family": tp,
        "hkc_family_id": fam,
        "capability_id": cap,
        "user_facing_goal": f"Run {tp} SRC pilot via ScientificRouteController.",
        "solver_visible": {
            "route_family_id": fam,
            "hkc_family_id": fam,
            "pack_refs": [str(cfg["hcg_pack"])],
            "inputs": {"goal_artifacts": [str(cfg["graph_target"])],
            },
            "parameters": {
                "capability_id": cap,
                "scenario_id": scenario_id,
                "split": "official_test",
            },
            "allowed_inventory": {"hcg_packs": [str(cfg["hcg_pack"])],
            },
        },
    }


def load_first_family_card(
    family_id: str,
    *,
    route_cards_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    path = route_cards_path
    if path is None:
        cards_path, _ = resolve_family_slice_paths(family_id)
        if cards_path.is_file():
            path = cards_path
        else:
            from hazardweaver.hwa.scientific_controller.hkc_pilot_assets import resolve_hkc_pilot_paths

            paths = resolve_hkc_pilot_paths()
            if not paths.route_cards_exists():
                return None
            path = paths.route_cards
    if path is None or not Path(path).is_file():
        return None
    target = str(family_id)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        binding = row.get("route_binding") or {}
        if str(binding.get("family_id") or "") == target:
            return dict(row)
    return None


def attach_seven_track_hkc_binding(
    route: Mapping[str, Any],
    card: Mapping[str, Any],
    family_id: str,
) -> Dict[str, Any]:
    out = dict(route)
    out["paper_id"] = str(card.get("paper_id") or out.get("paper_id") or "")
    out["item_id"] = str(card.get("item_id") or out.get("item_id") or "")
    out["hkc_family_id"] = str(family_id)
    return out


def build_seven_track_terminal_src_route(
    card: Mapping[str, Any],
    family_id: str,
    *,
    capability_id: str,
) -> Dict[str, Any]:
    """Single-cap SRC route for seven-track headline execution (login-safe terminal)."""
    cap = str(capability_id or "").strip().upper()
    route = {
        "route_id": f"route:{family_id}:terminal:{cap or 'default'}",
        "capability_ids": [cap] if cap else [],
        "edges": [cap] if cap else [],
        "executable": True,
        "track_src_terminal": True,
        "seven_track_terminal": True,
    }
    return attach_seven_track_hkc_binding(route, card, family_id)


def seven_track_src_binding_status(route: Mapping[str, Any]) -> str:
    """HKC contract binding present → SRC_PILOT (tri-state A_sci preserved on route)."""
    if str(route.get("hkc_binding_source") or "") == "scientific_route_contract":
        return "SRC_PILOT"
    return "SRC_PENDING_BINDING"


def annotate_seven_track_src_routes(
    routes: List[Mapping[str, Any]],
    controller: Any,
    family_id: str,
) -> List[Dict[str, Any]]:
    card = load_first_family_card(family_id)
    annotated: List[Dict[str, Any]] = []
    if card is not None:
        cap = str(controller.task.get("capability_id") or "").upper()
        terminal = build_seven_track_terminal_src_route(card, family_id, capability_id=cap)
        annotated.append(controller.gate.annotate_route(terminal, controller.state))

    seen: set[str] = set()
    for raw in routes:
        route = attach_seven_track_hkc_binding(raw, card, family_id) if card else dict(raw)
        rid = str(route.get("route_id") or "")
        if rid and rid in seen:
            continue
        ann = controller.gate.annotate_route(route, controller.state)
        if rid:
            seen.add(rid)
        annotated.append(ann)

    def _rank(route: Mapping[str, Any]) -> tuple:
        binding = 0 if route.get("hkc_binding_source") == "scientific_route_contract" else 1
        terminal = 0 if route.get("track_src_terminal") or route.get("seven_track_terminal") else 1
        length = len(route.get("edges") or route.get("capability_ids") or [])
        return (binding, terminal, length)

    annotated.sort(key=_rank)
    return annotated


# Per-track convenience aliases for tests
def is_wf3_pilot_task(task: Mapping[str, Any]) -> bool:
    return is_seven_track_pilot_task(task) and resolve_seven_track_family_id(task) == "wf3_process_v1"


def resolve_hkc_family_id(
    task: Mapping[str, Any],
    route: Optional[Mapping[str, Any]] = None,
) -> str:
    return resolve_seven_track_family_id(task, route)


def build_wf3_src_pilot_task(**kwargs: Any) -> Dict[str, Any]:
    return build_seven_track_src_pilot_task("WF-3", **kwargs)


WF3_PILOT_FAMILY = "wf3_process_v1"
