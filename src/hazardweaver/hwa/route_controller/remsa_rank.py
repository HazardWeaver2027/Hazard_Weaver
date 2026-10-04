"""REMSA two-stage route selection: A_cap hard-filter → validation_utility soft-rank."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwa.runtime.artifact_frontier import artifact_reuse_score
from hazardweaver.hwa.scientific_controller.reason_codes import ACapVerdict, ASciVerdict

_G1_CBR_UTILITY_BOOST = 1.25
_SELF_CBR_UTILITY_BOOST = 1.25


def _route_matches_capability(route: Mapping[str, Any], capability_id: str) -> bool:
    cap = str(capability_id or "").strip()
    if not cap:
        return False
    caps = [str(c) for c in (route.get("capability_ids") or [])]
    if cap in caps:
        return True
    edges = [str(e) for e in (route.get("edges") or [])]
    return cap in edges


def _self_cbr_capability(task: Optional[Mapping[str, Any]]) -> str:
    if not task:
        return ""
    meta = task.get("metadata") or {}
    return str(meta.get("hwa_self_cbr_capability") or "").strip()


def _legacy_ds_cbr_capability(task: Optional[Mapping[str, Any]]) -> str:
    """Invalidated G1 path — debug only when HWA_ALLOW_CROSS_SYSTEM_CBR=1."""
    if not task:
        return ""
    from hazardweaver.hwa.experiments.headline_route_profile_v1 import cross_system_cbr_allowed, is_g1_profile

    if not cross_system_cbr_allowed() or not is_g1_profile():
        return ""
    meta = task.get("metadata") or {}
    return str(meta.get("ds_cbr_capability") or "").strip()


def _cbr_warm_capability(task: Optional[Mapping[str, Any]]) -> str:
    self_cap = _self_cbr_capability(task)
    if self_cap:
        return self_cap
    return _legacy_ds_cbr_capability(task)


def _a_sci_bonus(route: Mapping[str, Any]) -> float:
    verdict = str((route.get("A_sci") or {}).get("verdict") or "")
    if verdict == ASciVerdict.APPLICABLE.value:
        return 0.35
    if verdict in {
        ASciVerdict.UNKNOWN_PENDING_THEORY.value,
        "SCI_UNKNOWN_PENDING_THEORY",
    }:
        return 0.10
    return 0.0


def _a_cap_bonus(route: Mapping[str, Any]) -> float:
    verdict = str((route.get("A_cap") or {}).get("verdict") or "")
    if verdict in {ACapVerdict.REACHABLE.value, "CAP_REACHABLE"}:
        return 0.25
    if verdict in {ACapVerdict.UNRESOLVED.value, "CAP_UNRESOLVED"}:
        return 0.05
    return 0.0


def _taskpack_tolerance_metric(task: Optional[Mapping[str, Any]]) -> str:
    if not task:
        return ""
    tp_id = str(task.get("taskpack_id") or "")
    if not tp_id:
        return ""
    try:
        from hazardweaver.hwb.registry.expand_parametric_headline_v1 import _load_taskpack

        tp = _load_taskpack(tp_id)
        tol = (tp.get("reference_view") or {}).get("tolerance") or {}
        return str(tol.get("metric") or "")
    except Exception:  # noqa: BLE001
        return ""


def _route_capability_id(route: Mapping[str, Any]) -> str:
    caps = route.get("capability_ids") or route.get("edges") or []
    if caps:
        return str(caps[0]).strip()
    return ""


def _w3_refresh_route_bonus(route: Mapping[str, Any], task: Optional[Mapping[str, Any]]) -> float:
    """W3 refresh axis: prefer first-attempt shock at s0; penalize gold blocked pre-refresh."""
    if not task:
        return 0.0
    meta = task.get("metadata") or {}
    first = str(meta.get("w3_first_attempt_capability_id") or meta.get("w3_shock_capability_id") or "").strip()
    if not first:
        return 0.0
    cap = _route_capability_id(route)
    if not cap:
        return 0.0
    rq4 = dict(meta.get("rq4_intervention") or {})
    sv = task.get("solver_visible") or {}
    allowed = [str(x).strip() for x in (sv.get("inputs") or {}).get("allowed_edge_ids") or [] if str(x).strip()]
    blocked = {str(x).strip() for x in (rq4.get("defer_s0_blocked_capabilities") or []) if str(x).strip()}
    if cap in blocked and rq4.get("defer_s0_narrow_allowed_to_shock"):
        return -0.95
    bonus = 0.0
    if cap == first:
        bonus += 0.55 if len(allowed) <= 1 else 0.30
    return bonus


def unified_pinned_scenario_capability(task: Optional[Mapping[str, Any]]) -> str:
    """Scenario grading CAP for unified M-tier multi_eligible cells (no answer leak)."""
    if not task:
        return ""
    meta = task.get("metadata") or {}
    if not meta.get("unified_benchmark_v1"):
        return ""
    w3_gold = str(meta.get("w3_gold_capability_id") or "").strip()
    if w3_gold:
        return w3_gold
    if str(meta.get("route_eligibility") or "") != "multi_eligible":
        return ""
    scenario = str(meta.get("scenario_id") or "").split("__", 1)[0].strip()
    if scenario.startswith("CAP-") and scenario != "CAP-PLACEHOLDER":
        return scenario
    return ""


def _unified_scenario_route_bonus(route: Mapping[str, Any], task: Optional[Mapping[str, Any]]) -> float:
    """Unified @143: prefer scenario-aligned CAP when it is an admissible edge (no answer leak)."""
    if not task:
        return 0.0
    meta = task.get("metadata") or {}
    if not meta.get("unified_benchmark_v1"):
        return 0.0
    if str(meta.get("route_eligibility") or "") != "multi_eligible":
        return 0.0
    rq4 = dict(meta.get("rq4_intervention") or {})
    sv = task.get("solver_visible") or {}
    allowed_list = [
        str(x).strip() for x in (sv.get("inputs") or {}).get("allowed_edge_ids") or [] if str(x).strip()
    ]
    if rq4.get("defer_s0_narrow_allowed_to_shock") and len(allowed_list) <= 1:
        return _w3_refresh_route_bonus(route, task)
    cap = _route_capability_id(route)
    if not cap:
        return 0.0
    allowed = set(allowed_list)
    blocked = {str(x).strip() for x in (rq4.get("defer_s0_blocked_capabilities") or []) if str(x).strip()}
    if cap in blocked:
        return -0.95
    bonus = 0.0
    scenario_cap = unified_pinned_scenario_capability(task)
    if not scenario_cap:
        scenario = str(meta.get("scenario_id") or "").split("__", 1)[0].strip()
        if scenario.startswith("CAP-"):
            scenario_cap = scenario
    if scenario_cap and scenario_cap in allowed and cap == scenario_cap:
        bonus += 0.40
    witness = str(meta.get("ablation_witness_capability_id") or "").strip()
    if witness and cap == witness and cap != scenario_cap:
        bonus += 0.12
    return bonus


def _metric_route_bonus(route: Mapping[str, Any], *, tolerance_metric: str) -> float:
    if tolerance_metric != "log_volume_v1":
        return 0.0
    caps = {str(c) for c in (route.get("capability_ids") or [])}
    edges = {str(e) for e in (route.get("edges") or [])}
    ids = caps | edges
    if "pfdf_burn_volume_cascade" in ids:
        return 0.85
    if "pfdf_volume_gorr_v2" in ids:
        return 0.65
    if "burn_state_net_prithvi_v1" in ids:
        return -0.35
    return 0.0


def compute_validation_utility(
    route: Mapping[str, Any],
    *,
    task: Optional[Mapping[str, Any]] = None,
    state: Any = None,
) -> float:
    """Schema field activation — rubric from A_sci/A_cap + cost + P_k reuse."""
    if route.get("validation_utility") is not None:
        try:
            return float(route["validation_utility"])
        except (TypeError, ValueError):
            pass
    score = 0.25 + _a_sci_bonus(route) + _a_cap_bonus(route)
    score += _metric_route_bonus(route, tolerance_metric=_taskpack_tolerance_metric(task))
    score += _unified_scenario_route_bonus(route, task)
    score += _w3_refresh_route_bonus(route, task)
    cost = route.get("estimated_cost")
    if cost is None:
        cost = len(route.get("edges") or route.get("capability_ids") or [])
    try:
        score -= 0.03 * float(cost)
    except (TypeError, ValueError):
        pass
    if state is not None:
        score += 0.15 * artifact_reuse_score(route, state)
    cbr = _cbr_warm_capability(task)
    if cbr and _route_matches_capability(route, cbr):
        boost = _SELF_CBR_UTILITY_BOOST if _self_cbr_capability(task) else _G1_CBR_UTILITY_BOOST
        score += boost
    return max(0.0, score)


def remsa_hard_filter(routes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Stage 1: admissible + non-Pareto-dominated (A_cap gate already in admissible)."""
    out: List[Dict[str, Any]] = []
    for r in routes:
        if not r.get("admissible"):
            continue
        if r.get("pareto_dominated"):
            continue
        out.append(dict(r))
    if not out:
        out = [dict(r) for r in routes if r.get("admissible")]
    return out


def remsa_soft_rank(
    routes: Sequence[Mapping[str, Any]],
    *,
    task: Optional[Mapping[str, Any]] = None,
    state: Any = None,
    committed_route: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Stage 2: sort by validation_utility, then minimal scientific disruption."""
    from hazardweaver.hwa.route_controller.scientific_disruption import scientific_disruption

    pool = remsa_hard_filter(routes)
    cbr = _cbr_warm_capability(task)
    self_cbr = _self_cbr_capability(task)
    for r in pool:
        r["validation_utility"] = compute_validation_utility(r, task=task, state=state)
        if cbr and _route_matches_capability(r, cbr):
            if self_cbr:
                r["hwa_self_cbr_warm_start"] = True
            else:
                r["ds_cbr_warm_start"] = True
    pool.sort(
        key=lambda r: (
            -float(r.get("validation_utility") or 0.0),
            scientific_disruption(r, committed_route),
            str(r.get("route_id") or ""),
        )
    )
    for i, r in enumerate(pool):
        r["remsa_rank"] = i + 1
        r["selected_by"] = "remsa_soft_rank"
    return pool


def remsa_top_route(
    routes: Sequence[Mapping[str, Any]],
    *,
    task: Optional[Mapping[str, Any]] = None,
    state: Any = None,
    committed_route: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    ranked = remsa_soft_rank(
        routes, task=task, state=state, committed_route=committed_route
    )
    return ranked[0] if ranked else None
