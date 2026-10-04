"""PFDF vertical slice — portfolio caps + HKC pfdf_v1 binding (D-PFDF-1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from hazardweaver.hcg.runtime.portfolio_probe_resolver import PFDF_OFFICIAL_CAPS
from hazardweaver.hwa.scientific_controller.hkc_pilot_assets import resolve_hkc_pilot_paths

PFDF_TASKPACK = "PFDF"
PFDF_PILOT_FAMILY = "pfdf_v1"
PFDF_DEFAULT_RECORD_ID = "Thomas_Thomas1_208"
PFDF_GRAPH_SOURCES = ("landsat_pre_post_patch_v1", "usgs_pfdf_record_v1")
PFDF_GRAPH_TARGET = "log_volume_v1"
PFDF_HCG_PACK = "pfdf_v1_held"


def is_pfdf_pilot_task(task: Mapping[str, Any]) -> bool:
    domain = str(task.get("domain") or "").lower()
    family = str(task.get("task_family") or "").upper()
    tp = str(task.get("taskpack_id") or "").upper()
    cap = str(task.get("capability_id") or "")
    hkc = str(
        task.get("hkc_family_id")
        or (task.get("solver_visible") or {}).get("hkc_family_id")
        or ""
    )
    if "pfdf" in domain or family == "PFDF" or tp == "PFDF":
        return True
    if hkc == PFDF_PILOT_FAMILY:
        return True
    if cap in PFDF_OFFICIAL_CAPS:
        return True
    return False


def resolve_hkc_family_id(
    task: Mapping[str, Any],
    route: Optional[Mapping[str, Any]] = None,
) -> str:
    """Map PFDF tasks / portfolio routes to HKC `pfdf_v1` for Route Card / SRC binding."""
    if route is not None:
        hkc = str(route.get("hkc_family_id") or route.get("family_id") or "")
        if hkc == PFDF_PILOT_FAMILY:
            return PFDF_PILOT_FAMILY
        caps = set(route.get("capability_ids") or route.get("edges") or [])
        if caps & set(PFDF_OFFICIAL_CAPS):
            return PFDF_PILOT_FAMILY
    if is_pfdf_pilot_task(task):
        return PFDF_PILOT_FAMILY
    solver = task.get("solver_visible") or {}
    return str(
        task.get("hkc_family_id")
        or solver.get("hkc_family_id")
        or ""
    )


def build_pfdf_src_pilot_task(
    *,
    task_id: str = "pfdf-src-pilot-volume",
    capability_id: str = "pfdf_volume_gorr_v2",
    record_id: str = PFDF_DEFAULT_RECORD_ID,
) -> Dict[str, Any]:
    """PFDF task for ScientificRouteController (no legacy_mode bypass)."""
    cap = str(capability_id).strip()
    if cap not in PFDF_OFFICIAL_CAPS:
        cap = "pfdf_volume_gorr_v2"
    return {
        "task_id": task_id,
        "taskpack_id": PFDF_TASKPACK,
        "domain": "pfdf",
        "task_family": PFDF_TASKPACK,
        "hkc_family_id": PFDF_PILOT_FAMILY,
        "capability_id": cap,
        "user_facing_goal": f"Run PFDF SRC pilot ({cap}) via ScientificRouteController.",
        "solver_visible": {
            "route_family_id": PFDF_PILOT_FAMILY,
            "hkc_family_id": PFDF_PILOT_FAMILY,
            "pack_refs": [PFDF_HCG_PACK],
            "inputs": {
                "goal_artifacts": [PFDF_GRAPH_TARGET],
                "sample_refs": [{"record_id": record_id, "sample_id": record_id}],
            },
            "parameters": {
                "capability_id": cap,
                "record_id": record_id,
            },
            "allowed_inventory": {"hcg_packs": [PFDF_HCG_PACK]},
        },
    }


def pfdf_slice_metadata(task: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "pilot_vertical_slice": PFDF_TASKPACK,
        "hkc_family_id": PFDF_PILOT_FAMILY,
        "official_caps": sorted(PFDF_OFFICIAL_CAPS),
        "d_pfdf_1": "SRC_PILOT",
        "is_pfdf_pilot": is_pfdf_pilot_task(task),
    }


def load_first_pfdf_v1_card(
    *,
    route_cards_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Return first pilot slice Route Card bound to ``pfdf_v1`` (HKC DL-093)."""
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
        if str(binding.get("family_id") or "") == PFDF_PILOT_FAMILY:
            return dict(row)
    return None


def attach_pfdf_hkc_binding(
    route: Mapping[str, Any],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach HKC paper/item ids so AdmissibilityGate resolves scientific_route_contract."""
    out = dict(route)
    out["paper_id"] = str(card.get("paper_id") or out.get("paper_id") or "")
    out["item_id"] = str(card.get("item_id") or out.get("item_id") or "")
    out["hkc_family_id"] = PFDF_PILOT_FAMILY
    return out


def build_pfdf_terminal_src_route(
    card: Mapping[str, Any],
    *,
    capability_id: str = "pfdf_volume_gorr_v2",
) -> Dict[str, Any]:
    """Single-cap SRC route for portfolio PFDF execution (login-safe terminal)."""
    cap = str(capability_id).strip()
    if cap not in PFDF_OFFICIAL_CAPS:
        cap = "pfdf_volume_gorr_v2"
    route = {
        "route_id": f"route:pfdf:terminal:{cap}",
        "capability_ids": [cap],
        "edges": [cap],
        "executable": True,
        "pfdf_portfolio_terminal": True,
    }
    return attach_pfdf_hkc_binding(route, card)


def pfdf_src_binding_status(route: Mapping[str, Any]) -> str:
    """D-PFDF-1 seal: HKC contract binding present → SRC_PILOT (not full admissible gate)."""
    if str(route.get("hkc_binding_source") or "") == "scientific_route_contract":
        return "SRC_PILOT"
    return "SRC_PENDING_BINDING"


def annotate_pfdf_src_routes(
    routes: List[Mapping[str, Any]],
    controller: Any,
) -> List[Dict[str, Any]]:
    """Gate-annotate PFDF SRC candidates; prepend terminal portfolio route when HKC card exists."""
    card = load_first_pfdf_v1_card()
    annotated: List[Dict[str, Any]] = []
    if card is not None:
        terminal = build_pfdf_terminal_src_route(card)
        annotated.append(controller.gate.annotate_route(terminal, controller.state))

    seen: set[str] = set()
    for raw in routes:
        route = attach_pfdf_hkc_binding(raw, card) if card is not None else dict(raw)
        rid = str(route.get("route_id") or "")
        if rid and rid in seen:
            continue
        ann = controller.gate.annotate_route(route, controller.state)
        if rid:
            seen.add(rid)
        annotated.append(ann)

    def _rank(route: Mapping[str, Any]) -> tuple:
        binding = 0 if route.get("hkc_binding_source") == "scientific_route_contract" else 1
        terminal = 0 if route.get("pfdf_portfolio_terminal") else 1
        length = len(route.get("edges") or route.get("capability_ids") or [])
        return (binding, terminal, length)

    annotated.sort(key=_rank)
    return annotated
