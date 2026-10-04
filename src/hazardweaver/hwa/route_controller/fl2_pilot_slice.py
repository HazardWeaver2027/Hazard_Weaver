"""FL-2 vertical slice — CAP-FL2-04~06 official solver inference (FOUR_MODULE §3)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

FL2_TASKPACK = "FL-2"
FL2_PILOT_FAMILY = "fl2_v1"
FL2_SOLVER_OFFICIAL_CAPS = frozenset({"CAP-FL2-04", "CAP-FL2-05", "CAP-FL2-06"})
FL2_PILOT_READY_CAP = "CAP-FL2-06"
FL2_BLOCKED_OFFICIAL_CAPS = frozenset({"CAP-FL2-01", "CAP-FL2-02", "CAP-FL2-03"})
# HCG route-family registry ids (not HKC family_id)
FL2_HCG_ROUTE_FAMILIES = frozenset(
    {
        "RF-TERRAIN-INDEX-BASELINE",
        "RF-REDUCED-PHYSICS-SHALLOW-WATE",
        "RF-HYDRODYNAMIC-SOLVER",
    }
)


def is_fl2_pilot_task(task: Mapping[str, Any]) -> bool:
    domain = str(task.get("domain") or "").lower()
    family = str(task.get("task_family") or "").lower()
    tp = str(task.get("taskpack_id") or task.get("task_layer") or "").upper()
    rf = str((task.get("solver_visible") or {}).get("route_family_id") or "")
    hkc = str(task.get("hkc_family_id") or (task.get("solver_visible") or {}).get("hkc_family_id") or "")
    if "fl2" in domain or "fl-2" in family or tp == "FL-2":
        return True
    if hkc == FL2_PILOT_FAMILY:
        return True
    if rf in {FL2_PILOT_FAMILY, *FL2_HCG_ROUTE_FAMILIES}:
        return True
    return False


def resolve_hkc_family_id(
    task: Mapping[str, Any],
    route: Optional[Mapping[str, Any]] = None,
) -> str:
    """Map FL-2 / HCG RF-* routes to HKC `fl2_v1` for Route Card / SRC binding."""
    if route is not None:
        hkc = str(route.get("hkc_family_id") or route.get("family_id") or "")
        if hkc == FL2_PILOT_FAMILY:
            return FL2_PILOT_FAMILY
        rf = str(route.get("route_family_id") or "")
        if rf == FL2_PILOT_FAMILY:
            return FL2_PILOT_FAMILY
    if is_fl2_pilot_task(task):
        return FL2_PILOT_FAMILY
    solver = task.get("solver_visible") or {}
    return str(
        task.get("hkc_family_id")
        or solver.get("hkc_family_id")
        or solver.get("route_family_id")
        or task.get("task_family")
        or ""
    )


def _cap_from_task(task: Mapping[str, Any]) -> str:
    params = (task.get("solver_visible") or {}).get("parameters") or {}
    cap = str(params.get("capability_id") or task.get("capability_id") or "").strip()
    if cap in FL2_SOLVER_OFFICIAL_CAPS:
        return cap
    return FL2_PILOT_READY_CAP


def pilot_ready_capability(task: Optional[Mapping[str, Any]] = None) -> str:
    if task is not None and not is_fl2_pilot_task(task):
        return ""
    if task is None:
        return FL2_PILOT_READY_CAP
    return _cap_from_task(task)


def fl2_slice_metadata(task: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "pilot_vertical_slice": "FL-2",
        "pilot_ready_cap": pilot_ready_capability(task),
        "solver_official_caps": sorted(FL2_SOLVER_OFFICIAL_CAPS),
        "blocked_official_caps": sorted(FL2_BLOCKED_OFFICIAL_CAPS),
        "is_fl2_pilot": is_fl2_pilot_task(task),
    }
