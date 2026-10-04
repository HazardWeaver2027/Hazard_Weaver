"""Route-level metric contract — admissibility vs taskpack tolerance (no VERIFY backdoors)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hazardweaver.hwb.bridge.submission_metric_v1 import metrics_compatible, taskpack_tolerance_metric
from hazardweaver.hcg.runtime.portfolio_probe_resolver import PFDF_OFFICIAL_CAPS

# Terminal capability → primary scalar tolerance metric for headline routing.
_CAPABILITY_TOLERANCE_METRIC: Dict[str, str] = {
    "burn_state_net_prithvi_v1": "burn_severity_summary_v1",
    "pfdf_volume_gorr_v2": "log_volume_v1",
    "pfdf_burn_volume_cascade": "log_volume_v1",
    "joint_pfdf_net_prithvi_v1": "log_volume_v1",
    "joint_pfdf_net_v1": "log_volume_v1",
    **{f"CAP-FL2-{i:02d}": "rmse_depth" for i in range(1, 7)},
    **{f"CAP-MH3-{i:02d}": "mae" for i in range(1, 7)},
    **{f"CAP-WF3-{i:02d}": "pick_f1" for i in range(1, 7)},
    **{f"CAP-HWMED-{i:02d}": "metric_value" for i in range(1, 7)},
    **{f"CAP-E1E3-{i:02d}": "gmpe_coverage" for i in range(1, 7)},
    **{f"CAP-EQ-{i:02d}": "replay_parity" for i in range(1, 4)},
    **{f"CAP-MH2-{i:02d}": "log_mae" for i in range(1, 7)},
    "CAP-MH4-R01": "substrate_coverage",
    "CAP-MH4-R02": "log_mae",
    "CAP-MH4-R03": "log_mae",
    "CAP-MH4-R06": "log_mae",
    **{f"CAP-L2-{i:02d}": "depth_rmse" for i in range(1, 6)},
    **{f"CAP-DROUT-{i:02d}": "sdo_skill" for i in range(1, 6)},
    **{f"CAP-TCTRK-{i:02d}": "lead_error_km" for i in range(1, 7)},
    **{f"CAP-MH1-{i:02d}": "log_volume_v1" for i in range(1, 6)},
}
for _cap in PFDF_OFFICIAL_CAPS:
    if _cap != "burn_state_net_prithvi_v1":
        _CAPABILITY_TOLERANCE_METRIC[_cap] = "log_volume_v1"


def _route_capability_ids(route: Mapping[str, Any]) -> List[str]:
    caps = [str(c) for c in (route.get("capability_ids") or []) if str(c)]
    if caps:
        return caps
    edges = [str(e) for e in (route.get("edges") or []) if str(e)]
    return [e for e in edges if not e.startswith("schema_")]


def terminal_capability_id(route: Mapping[str, Any]) -> str:
    caps = _route_capability_ids(route)
    return caps[-1] if caps else ""


def capability_tolerance_metric(capability_id: str) -> str:
    cid = str(capability_id or "").strip()
    if not cid:
        return ""
    if cid in _CAPABILITY_TOLERANCE_METRIC:
        return _CAPABILITY_TOLERANCE_METRIC[cid]
    cid_u = cid.upper()
    if cid_u.startswith("CAP-MH1-") or cid.startswith("mh1_"):
        return "log_volume_v1"
    if cid_u.startswith("CAP-FL2-"):
        return "rmse_depth"
    if cid_u.startswith("CAP-MH3-"):
        return "mae"
    if cid_u.startswith("CAP-WF3-"):
        return "pick_f1"
    if cid_u.startswith("CAP-HWMED-"):
        return "metric_value"
    if cid_u.startswith("CAP-E1E3-") or cid_u.startswith("CAP-EQ-"):
        return "gmpe_coverage"
    if cid_u.startswith("CAP-MH2-"):
        return "log_mae"
    if cid_u.startswith("CAP-MH4-"):
        if cid_u == "CAP-MH4-R01":
            return "substrate_coverage"
        return "log_mae"
    if cid_u.startswith("CAP-L2-"):
        return "depth_rmse"
    if cid_u.startswith("CAP-DROUT-"):
        return "sdo_skill"
    if cid_u.startswith("CAP-TCTRK-"):
        return "lead_error_km"
    return ""


def route_tolerance_metric(route: Mapping[str, Any]) -> str:
    return capability_tolerance_metric(terminal_capability_id(route))


@lru_cache(maxsize=1)
def _mh1_volume_caps() -> frozenset[str]:
    caps = {c for c, m in _CAPABILITY_TOLERANCE_METRIC.items() if m == "log_volume_v1"}
    caps.update(PFDF_OFFICIAL_CAPS)
    caps.update(f"CAP-MH1-{i:02d}" for i in range(1, 6))
    caps.update(
        {
            "pfdf_volume_gorr_v2",
            "pfdf_burn_volume_cascade",
            "joint_pfdf_net_prithvi_v1",
            "joint_pfdf_net_v1",
        }
    )
    caps.update(f"mh1_{suffix}" for suffix in (
        "marginal_volume_v1",
        "usgs_empirical_v1",
        "dependence_volume_v1",
        "mechanistic_volume_v1",
        "d8_runout_v1",
    ))
    return frozenset(caps)


def _fl2_blocked_on_solver(taskpack: Optional[Mapping[str, Any]], cap: str) -> bool:
    if str((taskpack or {}).get("taskpack_id") or "") != "hwb_fl2_solver_parametric_v1":
        return False
    from hazardweaver.hwa.route_controller.fl2_pilot_slice import FL2_BLOCKED_OFFICIAL_CAPS

    return str(cap or "").strip() in FL2_BLOCKED_OFFICIAL_CAPS


def resolve_taskpack_for_route_gate(task: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    meta = task.get("metadata") or {}
    if meta.get("hwb_headline_inventory"):
        inv_row = {
            "instance_id": meta.get("instance_id") or meta.get("internal_task_id"),
            "taskpack_id": meta.get("taskpack_id") or task.get("taskpack_id"),
            "scenario_id": meta.get("scenario_id"),
            "track": meta.get("track") or task.get("headline_target"),
            "difficulty_tier": meta.get("difficulty_tier"),
            "source": meta.get("source"),
            "routing_tier": meta.get("routing_tier") or "headline",
        }
        if inv_row.get("taskpack_id") and inv_row.get("scenario_id"):
            try:
                from hazardweaver.hwb.run.eval_dca_submission_v1 import resolve_taskpack_for_inventory_row

                return resolve_taskpack_for_inventory_row(inv_row)
            except Exception:  # noqa: BLE001
                pass
    tp_id = str(
        task.get("taskpack_id")
        or meta.get("taskpack_id")
        or ""
    ).strip()
    if not tp_id:
        return None
    try:
        from hazardweaver.hwb.registry.expand_parametric_headline_v1 import _load_taskpack

        loaded = _load_taskpack(tp_id)
        return dict(loaded) if isinstance(loaded, Mapping) else None
    except Exception:  # noqa: BLE001
        return None


def _cap_ref_tolerance_metric(
    taskpack: Optional[Mapping[str, Any]],
    capability_id: str,
) -> str:
    if not taskpack or not capability_id:
        return ""
    taskpack_id = str(taskpack.get("taskpack_id") or "")
    if not taskpack_id:
        return ""
    try:
        from hazardweaver.hwb.registry.track_parametric_resolver import load_scenario_refs

        refs = load_scenario_refs(taskpack_id)
        scenario_ref = (refs.get("scenarios") or {}).get(str(capability_id).strip()) or {}
        tol = (scenario_ref.get("tolerance") or {}).get("metric")
        return str(tol or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def evaluate_route_metric_contract(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, str]:
    """Return (ok, reason_code). Empty reason_code when compatible or unknown."""
    tp = taskpack if taskpack is not None else resolve_taskpack_for_route_gate(task)
    tol = taskpack_tolerance_metric(tp)
    cap = terminal_capability_id(route)
    if cap and _fl2_blocked_on_solver(tp, cap):
        return False, "fl2_g2_cap_blocked_on_solver_taskpack"
    route_metric = _cap_ref_tolerance_metric(tp, cap) or route_tolerance_metric(route)
    if cap and _cap_ref_tolerance_metric(tp, cap):
        route_metric = _cap_ref_tolerance_metric(tp, cap)
        tol = route_metric
    if not tol or not route_metric:
        return True, ""
    if metrics_compatible(route_metric, tol):
        return True, ""
    return False, "route_metric_family_mismatch"


def apply_route_metric_contract(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Annotate route; clear admissible when output metric cannot satisfy taskpack."""
    out = dict(route)
    ok, code = evaluate_route_metric_contract(out, task, taskpack=taskpack)
    if ok:
        return out
    out["admissible"] = False
    out["metric_contract_ok"] = False
    out["metric_contract_code"] = code
    a_cap = dict(out.get("A_cap") or {})
    codes = list(a_cap.get("codes") or [])
    if code and code not in codes:
        codes.append(code)
    a_cap["codes"] = codes
    a_cap["metric_contract"] = code
    out["A_cap"] = a_cap
    return out


def filter_routes_by_metric_contract(
    routes: Sequence[Mapping[str, Any]],
    task: Mapping[str, Any],
    *,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    return [apply_route_metric_contract(r, task, taskpack=taskpack) for r in routes]
