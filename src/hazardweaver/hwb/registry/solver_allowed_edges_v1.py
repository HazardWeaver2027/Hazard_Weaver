"""Resolve solver-visible allowed_edge_ids without reference_view witness leakage."""

from __future__ import annotations

import hashlib
import re
from typing import Any, List, Mapping, Optional

from hazardweaver.hwb.registry.track_parametric_resolver import load_scenario_refs

_CASE_RE = re.compile(r"^case_(\d+)$", re.IGNORECASE)


def _base_scenario_id(scenario_id: str) -> str:
    """Strip difficulty suffix e.g. CAP-E1E3-01__L1 → CAP-E1E3-01."""
    sid = str(scenario_id or "").strip()
    if "__" in sid:
        return sid.split("__", 1)[0]
    return sid


def _list_ref_capability_ids(taskpack_id: str) -> List[str]:
    refs = load_scenario_refs(taskpack_id)
    caps: List[str] = []
    for sc in (refs.get("scenarios") or {}).values():
        cap = sc.get("capability_id")
        if cap:
            caps.append(str(cap))
    return sorted(set(caps))


def _stable_cap_pick(caps: List[str], key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return caps[int(digest[:8], 16) % len(caps)]


def _atlas_case_capability(caps: List[str], scenario_id: str) -> Optional[str]:
    m = _CASE_RE.match(str(scenario_id or "").strip())
    if not m or not caps:
        return None
    idx = int(m.group(1))
    return caps[(idx - 1) % len(caps)]


_G6_ADAPTER_PREFIXES = ("align_", "schema_map")
_HEADLINE_G6_TABULAR_PREFIXES = ("ridge_", "rf_", "persist_", "heat_rule_", "spi_proxy_", "gorr_")


def _filter_fl2_solver_official_caps(edges: List[str], *, taskpack_id: str) -> List[str]:
    """Drop FL-2 G2 caps (01–03) when taskpack binds solver official path (04–06)."""
    if taskpack_id != "hwb_fl2_solver_parametric_v1":
        return edges
    try:
        from hazardweaver.hwa.route_controller.fl2_pilot_slice import (
            FL2_BLOCKED_OFFICIAL_CAPS,
            FL2_SOLVER_OFFICIAL_CAPS,
        )
    except Exception:  # noqa: BLE001
        return edges
    filtered = [e for e in edges if str(e).strip() not in FL2_BLOCKED_OFFICIAL_CAPS]
    if filtered:
        return filtered
    return sorted(FL2_SOLVER_OFFICIAL_CAPS)


def _filter_headline_controller_edges(
    edges: List[str],
    *,
    taskpack_id: str,
    inventory_row: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Drop non-executable HCG adapter edges from headline controller routes."""
    if str((inventory_row or {}).get("routing_tier") or "") != "headline":
        return edges
    try:
        from hazardweaver.hcg.registry.canonical_ids import is_headline_capability
        from hazardweaver.hwa.route_controller.multi_track_slice_v1 import MH1_OFFICIAL_CAPS
    except Exception:  # noqa: BLE001
        return edges

    filtered: List[str] = []
    for edge in edges:
        e = str(edge).strip()
        if not e or any(e.startswith(p) for p in _G6_ADAPTER_PREFIXES):
            continue
        if is_headline_capability(e) or e in MH1_OFFICIAL_CAPS:
            filtered.append(e)
            continue
        if any(e.startswith(p) for p in _HEADLINE_G6_TABULAR_PREFIXES):
            filtered.append(e)
    if filtered:
        return filtered
    caps = _list_ref_capability_ids(taskpack_id)
    headline_caps = [c for c in caps if c.startswith("CAP-")]
    return headline_caps or edges


def resolve_solver_allowed_edge_ids(
    taskpack: Mapping[str, Any],
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Return controller-routable edge ids for the agent (no gold scores).

    Precedence:
      1. Explicit ``solver_view.allowed_edge_ids`` when non-empty
      2. Parametric refs lookup by scenario_id (incl. variant tier suffix strip)
      3. ``reference_view.capability_id`` after parametric materialization
      4. Atlas ``case_N`` → stable round-robin over track capability anchors
      5. Other atlas scenario ids → deterministic pick from track anchors
      6. ``CAP-PLACEHOLDER`` taskpack rows → all track anchors (multi-route)
    """
    sv = taskpack.get("solver_view") or {}
    edges = [str(e) for e in (sv.get("allowed_edge_ids") or []) if str(e)]
    if edges:
        filtered = _filter_headline_controller_edges(
            edges,
            taskpack_id=str(taskpack.get("taskpack_id") or ""),
            inventory_row=inventory_row,
        )
        return _filter_fl2_solver_official_caps(
            filtered,
            taskpack_id=str(taskpack.get("taskpack_id") or ""),
        )

    ref = taskpack.get("reference_view") or {}
    cap = ref.get("capability_id")
    if cap:
        return [str(cap)]

    params = sv.get("parameters") or {}
    taskpack_id = str(taskpack.get("taskpack_id") or "")
    scenario_id = str(
        params.get("scenario_id")
        or (inventory_row or {}).get("scenario_id")
        or ""
    ).strip()
    if not scenario_id or not ref.get("refs_sidecar"):
        return []

    refs = load_scenario_refs(taskpack_id)
    lookup = _base_scenario_id(scenario_id)
    scenario_ref = (refs.get("scenarios") or {}).get(lookup) or {}
    cap = scenario_ref.get("capability_id")
    if cap:
        return [str(cap)]

    caps = _list_ref_capability_ids(taskpack_id)
    if not caps:
        return []

    if scenario_id == "CAP-PLACEHOLDER":
        return caps

    case_cap = _atlas_case_capability(caps, scenario_id)
    if case_cap:
        return [case_cap]

    source = str((inventory_row or {}).get("source") or "")
    instance_id = str((inventory_row or {}).get("instance_id") or "")
    if source == "atlas_materialized" or instance_id.startswith("atlas:"):
        picked = [_stable_cap_pick(caps, f"{taskpack_id}:{scenario_id}")]
        return _filter_headline_controller_edges(
            picked,
            taskpack_id=taskpack_id,
            inventory_row=inventory_row,
        )

    return []
