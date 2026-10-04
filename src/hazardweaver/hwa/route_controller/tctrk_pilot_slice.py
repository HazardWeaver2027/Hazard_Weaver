"""TC-TRK vertical slice — official caps + HKC tc_tctrk_v1 binding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from hazardweaver.hwa.route_controller.multi_track_slice_v1 import TC_TRK_OFFICIAL_CAPS
from hazardweaver.hwa.scientific_controller.hkc_pilot_assets import resolve_hkc_pilot_paths

TC_TCTRK_TASKPACK = "TC-TRK"
TC_TCTRK_PILOT_FAMILY = "tc_tctrk_v1"
TC_TCTRK_DEFAULT_CAP = "CAP-TCTRK-01"
TC_TCTRK_GRAPH_TARGET = "tc_track_bundle_v1"
TC_TCTRK_HCG_PACK = "manifest_rf_v1"
TC_TCTRK_HCG_ROUTE_FAMILIES = frozenset(
    {
        "RF-STATISTICAL-WEAK-BASELINE",
        "RF-DETERMINISTIC-NEURAL-WEATHER",
        "RF-PHYSICS-BASED-ENSEMBLE-NWP",
        "RF-PROBABILISTIC-DIFFUSION-WEAT",
        "RF-SPECTRAL-NEURAL-WEATHER-MODE",
        "RF-STATISTICAL-BIAS-CORRECTION",
    }
)


def is_tctrk_pilot_task(task: Mapping[str, Any]) -> bool:
    domain = str(task.get("domain") or "").lower()
    family = str(task.get("task_family") or "").upper()
    tp = str(task.get("taskpack_id") or "").upper()
    cap = str(task.get("capability_id") or "").upper()
    hkc = str(
        task.get("hkc_family_id")
        or (task.get("solver_visible") or {}).get("hkc_family_id")
        or ""
    )
    if any(x in domain for x in ("tctrk", "cyclone", "tc")) or family == "TC-TRK" or tp == "TC-TRK":
        return True
    if hkc == TC_TCTRK_PILOT_FAMILY:
        return True
    if cap in TC_TRK_OFFICIAL_CAPS:
        return True
    return False


def resolve_hkc_family_id(
    task: Mapping[str, Any],
    route: Optional[Mapping[str, Any]] = None,
) -> str:
    """Map TC-TRK tasks / RF-* routes to HKC `tc_tctrk_v1` for Route Card / SRC binding."""
    if route is not None:
        hkc = str(route.get("hkc_family_id") or route.get("family_id") or "")
        if hkc == TC_TCTRK_PILOT_FAMILY:
            return TC_TCTRK_PILOT_FAMILY
        rf = str(route.get("route_family_id") or "")
        if rf in {TC_TCTRK_PILOT_FAMILY, *TC_TCTRK_HCG_ROUTE_FAMILIES}:
            return TC_TCTRK_PILOT_FAMILY
        caps = {str(c).upper() for c in (route.get("capability_ids") or route.get("edges") or [])}
        if caps & set(TC_TRK_OFFICIAL_CAPS):
            return TC_TCTRK_PILOT_FAMILY
    if is_tctrk_pilot_task(task):
        return TC_TCTRK_PILOT_FAMILY
    solver = task.get("solver_visible") or {}
    return str(
        task.get("hkc_family_id")
        or solver.get("hkc_family_id")
        or ""
    )


def build_tctrk_src_pilot_task(
    *,
    task_id: str = "tctrk-src-pilot-01",
    capability_id: str = TC_TCTRK_DEFAULT_CAP,
    scenario_id: str = "pilot_scenario_000",
) -> Dict[str, Any]:
    cap = str(capability_id).strip().upper()
    if cap not in TC_TRK_OFFICIAL_CAPS:
        cap = TC_TCTRK_DEFAULT_CAP
    return {
        "task_id": task_id,
        "taskpack_id": TC_TCTRK_TASKPACK,
        "domain": "tctrk",
        "task_family": TC_TCTRK_TASKPACK,
        "hkc_family_id": TC_TCTRK_PILOT_FAMILY,
        "capability_id": cap,
        "user_facing_goal": f"Run TC-TRK SRC pilot ({cap}) via ScientificRouteController.",
        "solver_visible": {
            "route_family_id": TC_TCTRK_PILOT_FAMILY,
            "hkc_family_id": TC_TCTRK_PILOT_FAMILY,
            "pack_refs": [TC_TCTRK_HCG_PACK],
            "inputs": {"goal_artifacts": [TC_TCTRK_GRAPH_TARGET]},
            "parameters": {
                "capability_id": cap,
                "scenario_id": scenario_id,
                "split": "official_test",
            },
            "allowed_inventory": {"hcg_packs": [TC_TCTRK_HCG_PACK]},
        },
    }


def tctrk_slice_metadata(task: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "pilot_vertical_slice": TC_TCTRK_TASKPACK,
        "hkc_family_id": TC_TCTRK_PILOT_FAMILY,
        "official_caps": sorted(TC_TRK_OFFICIAL_CAPS),
        "is_tctrk_pilot": is_tctrk_pilot_task(task),
    }


def load_first_tctrk_v1_card(
    *,
    route_cards_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    path = route_cards_path
    if path is None:
        paths = resolve_hkc_pilot_paths()
        if not paths.route_cards_exists():
            return None
        path = paths.route_cards
    if path is None or not Path(path).is_file():
        return None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        binding = row.get("route_binding") or {}
        if str(binding.get("family_id") or "") == TC_TCTRK_PILOT_FAMILY:
            return dict(row)
    return None


def attach_tctrk_hkc_binding(
    route: Mapping[str, Any],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    out = dict(route)
    out["paper_id"] = str(card.get("paper_id") or out.get("paper_id") or "")
    out["item_id"] = str(card.get("item_id") or out.get("item_id") or "")
    out["hkc_family_id"] = TC_TCTRK_PILOT_FAMILY
    return out


def build_tctrk_terminal_src_route(
    card: Mapping[str, Any],
    *,
    capability_id: str = TC_TCTRK_DEFAULT_CAP,
) -> Dict[str, Any]:
    cap = str(capability_id).strip().upper()
    if cap not in TC_TRK_OFFICIAL_CAPS:
        cap = TC_TCTRK_DEFAULT_CAP
    route = {
        "route_id": f"route:tctrk:terminal:{cap}",
        "capability_ids": [cap],
        "edges": [cap],
        "executable": True,
        "track_src_terminal": True,
    }
    return attach_tctrk_hkc_binding(route, card)


def tctrk_src_binding_status(route: Mapping[str, Any]) -> str:
    if str(route.get("hkc_binding_source") or "") == "scientific_route_contract":
        return "SRC_PILOT"
    return "SRC_PENDING_BINDING"


def annotate_tctrk_src_routes(
    routes: List[Mapping[str, Any]],
    controller: Any,
) -> List[Dict[str, Any]]:
    card = load_first_tctrk_v1_card()
    annotated: List[Dict[str, Any]] = []
    if card is not None:
        cap = str(
            (controller.task.get("capability_id") or TC_TCTRK_DEFAULT_CAP)
        ).upper()
        terminal = build_tctrk_terminal_src_route(card, capability_id=cap)
        annotated.append(controller.gate.annotate_route(terminal, controller.state))

    seen: set[str] = set()
    for raw in routes:
        route = attach_tctrk_hkc_binding(raw, card) if card is not None else dict(raw)
        rid = str(route.get("route_id") or "")
        if rid and rid in seen:
            continue
        ann = controller.gate.annotate_route(route, controller.state)
        if rid:
            seen.add(rid)
        annotated.append(ann)

    def _rank(route: Mapping[str, Any]) -> tuple:
        binding = 0 if route.get("hkc_binding_source") == "scientific_route_contract" else 1
        terminal = 0 if route.get("track_src_terminal") else 1
        length = len(route.get("edges") or route.get("capability_ids") or [])
        return (binding, terminal, length)

    annotated.sort(key=_rank)
    return annotated
