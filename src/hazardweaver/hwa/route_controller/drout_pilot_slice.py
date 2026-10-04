"""DR-OUT vertical slice — official caps + HKC dr_out_v1 binding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from hazardweaver.hwa.route_controller.multi_track_slice_v1 import DR_OUT_OFFICIAL_CAPS
from hazardweaver.hwa.scientific_controller.hkc_pilot_assets import resolve_hkc_pilot_paths

DR_OUT_TASKPACK = "DR-OUT"
DR_OUT_PILOT_FAMILY = "dr_out_v1"
DR_OUT_DEFAULT_CAP = "CAP-DROUT-01"
DR_OUT_GRAPH_TARGET = "dr_outlook_bundle_v1"
DR_OUT_HCG_PACK = "manifest_rf_v1"
DR_OUT_HCG_ROUTE_FAMILIES = frozenset(
    {
        "RF-EXPERT-OPERATIONAL-SYNTHESIS",
        "RF-OBJECTIVE-STATISTICAL-TENDEN",
        "RF-ORDINAL-STATISTICAL-TRANSITI",
        "RF-PERSISTENCE-BASELINE",
        "RF-SUBSEASONAL-ML-FORECAST",
    }
)


def is_drout_pilot_task(task: Mapping[str, Any]) -> bool:
    domain = str(task.get("domain") or "").lower()
    family = str(task.get("task_family") or "").upper()
    tp = str(task.get("taskpack_id") or "").upper()
    cap = str(task.get("capability_id") or "").upper()
    hkc = str(
        task.get("hkc_family_id")
        or (task.get("solver_visible") or {}).get("hkc_family_id")
        or ""
    )
    if "drout" in domain or "drought" in domain or family == "DR-OUT" or tp == "DR-OUT":
        return True
    if hkc == DR_OUT_PILOT_FAMILY:
        return True
    if cap in DR_OUT_OFFICIAL_CAPS:
        return True
    return False


def resolve_hkc_family_id(
    task: Mapping[str, Any],
    route: Optional[Mapping[str, Any]] = None,
) -> str:
    """Map DR-OUT tasks / RF-* routes to HKC `dr_out_v1` for Route Card / SRC binding."""
    if route is not None:
        hkc = str(route.get("hkc_family_id") or route.get("family_id") or "")
        if hkc == DR_OUT_PILOT_FAMILY:
            return DR_OUT_PILOT_FAMILY
        rf = str(route.get("route_family_id") or "")
        if rf in {DR_OUT_PILOT_FAMILY, *DR_OUT_HCG_ROUTE_FAMILIES}:
            return DR_OUT_PILOT_FAMILY
        caps = {str(c).upper() for c in (route.get("capability_ids") or route.get("edges") or [])}
        if caps & set(DR_OUT_OFFICIAL_CAPS):
            return DR_OUT_PILOT_FAMILY
    if is_drout_pilot_task(task):
        return DR_OUT_PILOT_FAMILY
    solver = task.get("solver_visible") or {}
    return str(
        task.get("hkc_family_id")
        or solver.get("hkc_family_id")
        or ""
    )


def build_drout_src_pilot_task(
    *,
    task_id: str = "drout-src-pilot-01",
    capability_id: str = DR_OUT_DEFAULT_CAP,
    scenario_id: str = "pilot_scenario_000",
) -> Dict[str, Any]:
    cap = str(capability_id).strip().upper()
    if cap not in DR_OUT_OFFICIAL_CAPS:
        cap = DR_OUT_DEFAULT_CAP
    return {
        "task_id": task_id,
        "taskpack_id": DR_OUT_TASKPACK,
        "domain": "drout",
        "task_family": DR_OUT_TASKPACK,
        "hkc_family_id": DR_OUT_PILOT_FAMILY,
        "capability_id": cap,
        "user_facing_goal": f"Run DR-OUT SRC pilot ({cap}) via ScientificRouteController.",
        "solver_visible": {
            "route_family_id": DR_OUT_PILOT_FAMILY,
            "hkc_family_id": DR_OUT_PILOT_FAMILY,
            "pack_refs": [DR_OUT_HCG_PACK],
            "inputs": {"goal_artifacts": [DR_OUT_GRAPH_TARGET]},
            "parameters": {
                "capability_id": cap,
                "scenario_id": scenario_id,
                "split": "official_test",
            },
            "allowed_inventory": {"hcg_packs": [DR_OUT_HCG_PACK]},
        },
    }


def drout_slice_metadata(task: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "pilot_vertical_slice": DR_OUT_TASKPACK,
        "hkc_family_id": DR_OUT_PILOT_FAMILY,
        "official_caps": sorted(DR_OUT_OFFICIAL_CAPS),
        "is_drout_pilot": is_drout_pilot_task(task),
    }


def load_first_drout_v1_card(
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
        if str(binding.get("family_id") or "") == DR_OUT_PILOT_FAMILY:
            return dict(row)
    return None


def attach_drout_hkc_binding(
    route: Mapping[str, Any],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    out = dict(route)
    out["paper_id"] = str(card.get("paper_id") or out.get("paper_id") or "")
    out["item_id"] = str(card.get("item_id") or out.get("item_id") or "")
    out["hkc_family_id"] = DR_OUT_PILOT_FAMILY
    return out


def build_drout_terminal_src_route(
    card: Mapping[str, Any],
    *,
    capability_id: str = DR_OUT_DEFAULT_CAP,
) -> Dict[str, Any]:
    cap = str(capability_id).strip().upper()
    if cap not in DR_OUT_OFFICIAL_CAPS:
        cap = DR_OUT_DEFAULT_CAP
    route = {
        "route_id": f"route:drout:terminal:{cap}",
        "capability_ids": [cap],
        "edges": [cap],
        "executable": True,
        "track_src_terminal": True,
    }
    return attach_drout_hkc_binding(route, card)


def drout_src_binding_status(route: Mapping[str, Any]) -> str:
    if str(route.get("hkc_binding_source") or "") == "scientific_route_contract":
        return "SRC_PILOT"
    return "SRC_PENDING_BINDING"


def annotate_drout_src_routes(
    routes: List[Mapping[str, Any]],
    controller: Any,
) -> List[Dict[str, Any]]:
    card = load_first_drout_v1_card()
    annotated: List[Dict[str, Any]] = []
    if card is not None:
        cap = str(
            (controller.task.get("capability_id") or DR_OUT_DEFAULT_CAP)
        ).upper()
        terminal = build_drout_terminal_src_route(card, capability_id=cap)
        annotated.append(controller.gate.annotate_route(terminal, controller.state))

    seen: set[str] = set()
    for raw in routes:
        route = attach_drout_hkc_binding(raw, card) if card is not None else dict(raw)
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
