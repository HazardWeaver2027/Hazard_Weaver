"""A_sci and A_cap admissibility evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwa.agent_runtime.execution_schema import build_route_candidate
from hazardweaver.hwa.scientific_controller.compatibility import (
    check_support_compatibility,
    compatibility_verdict_from_issues,
    evaluate_interface_compatibility,
)
from hazardweaver.hwa.scientific_controller.reason_codes import (
    APPLICABLE_ASCI,
    ACapVerdict,
    ASciVerdict,
    REACHABLE_ACAP,
)
from hazardweaver.hwa.scientific_controller.state import SessionState


def _a_sci_dict(verdict: str, *, codes: Optional[List[str]] = None, refs: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "verdict": verdict,
        "codes": list(codes or []),
        "refs": list(refs or []),
    }


def _a_cap_dict(
    verdict: str,
    *,
    codes: Optional[List[str]] = None,
    missing_artifacts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "verdict": verdict,
        "codes": list(codes or []),
        "missing_artifacts": list(missing_artifacts or []),
    }


def _req_stem(constraint: str) -> str:
    return str(constraint or "").replace("require:", "").replace("_", " ").strip().lower()


def _task_requires_grounded(
    task_constraints: Sequence[str],
    route_constraints: Sequence[str],
    refs: Sequence[str],
    *,
    exact_match: bool = False,
) -> bool:
    """PI-gold RQ2: APPLICABLE only if each task require: is grounded in route constraints or refs."""
    task_reqs = [c for c in task_constraints if str(c).startswith("require:")]
    if not task_reqs:
        return False
    route_reqs = [str(c) for c in route_constraints if str(c).startswith("require:")]
    route_stems = {_req_stem(c) for c in route_reqs}
    ref_blob = " ".join(str(r) for r in refs).lower()
    for tr in task_reqs:
        tr_s = str(tr).strip()
        if exact_match:
            if tr_s not in route_reqs:
                return False
            continue
        stem = _req_stem(tr)
        if stem in route_stems:
            continue
        tokens = [t for t in stem.split() if len(t) > 3]
        if any(any(t in rs for t in tokens) for rs in route_stems):
            continue
        if tokens and any(t in ref_blob for t in tokens):
            continue
        return False
    return True


def evaluate_A_sci(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    theory_arm: str = "verified",
) -> Dict[str, Any]:
    """Scientific applicability A_sci(π, q; K) — DEV stub until G4 freeze."""
    solver = task.get("solver_visible") or {}
    constraints = list((solver.get("inputs") or {}).get("constraints") or [])
    route_constraints = list(route.get("theory_constraints") or [])
    blocked_regions = set(route.get("blocked_regions") or [])
    task_region = str((solver.get("inputs") or {}).get("region") or "").strip()

    if theory_arm == "off":
        return _a_sci_dict(ASciVerdict.UNKNOWN_PENDING_THEORY.value, codes=["theory_off"])

    if route.get("scientifically_inapplicable"):
        reason = str(route.get("inapplicability_reason") or "theory_mismatch")
        if "region" in reason:
            return _a_sci_dict(ASciVerdict.INAPPLICABLE_REGION.value, codes=[reason])
        if "temporal" in reason:
            return _a_sci_dict(ASciVerdict.INAPPLICABLE_TEMPORAL_SCALE.value, codes=[reason])
        if "hazard" in reason:
            return _a_sci_dict(ASciVerdict.INAPPLICABLE_HAZARD_CLASS.value, codes=[reason])
        return _a_sci_dict(ASciVerdict.INAPPLICABLE_THEORY_MISMATCH.value, codes=[reason])

    if task_region and task_region in blocked_regions:
        return _a_sci_dict(ASciVerdict.INAPPLICABLE_REGION.value, codes=[f"blocked_region:{task_region}"])

    refs = list(route.get("theory_alignment_refs") or [])
    if theory_arm in {"verified", "raw", "shuffled"} and route.get("requires_theory") and not refs:
        if theory_arm == "shuffled":
            return _a_sci_dict(ASciVerdict.INAPPLICABLE_THEORY_MISMATCH.value, codes=["shuffled_theory_mismatch"])
        return _a_sci_dict(ASciVerdict.UNKNOWN_PENDING_THEORY.value, codes=["theory_refs_pending"])

    if theory_arm in {"verified", "raw", "src_conformal_hkc"}:
        meta = task.get("metadata") or {}
        exact = bool(
            meta.get("e1e3_hkc_exact_require_grounding")
            or meta.get("mh_hkc_exact_require_grounding")
        )
        if not _task_requires_grounded(
            constraints, route_constraints, refs, exact_match=exact
        ):
            return _a_sci_dict(
                ASciVerdict.INAPPLICABLE_THEORY_MISMATCH.value,
                codes=["task_require_not_grounded_in_route"],
            )

    return _a_sci_dict(ASciVerdict.APPLICABLE.value, refs=refs)


def evaluate_A_cap(
    route: Mapping[str, Any],
    state: SessionState,
    *,
    graph: Any = None,
) -> Dict[str, Any]:
    """Capability reachability A_cap(π, s_k; G)."""
    capability_ids = list(route.get("capability_ids") or [])
    edge_ids = list(route.get("edges") or capability_ids)
    adapter_ids = list(route.get("adapter_ids") or [])

    for eid in edge_ids:
        if eid in state.invalidated_edges or eid in state.invalidated_capabilities:
            return _a_cap_dict(
                ACapVerdict.EDGE_INVALIDATED.value,
                codes=[f"invalidated:{eid}"],
            )

    missing: List[str] = []
    required_inputs = list(route.get("required_artifacts") or route.get("consumes") or [])
    available = set(state.available_artifacts) | set(state.sources)
    for art in required_inputs:
        if art and art not in available and art not in state.produced_handles:
            missing.append(str(art))
    if missing:
        return _a_cap_dict(
            ACapVerdict.UNREACHABLE_MISSING_ARTIFACT.value,
            missing_artifacts=missing,
            codes=["missing_input_artifacts"],
        )

    if route.get("executable") is False:
        return _a_cap_dict(
            ACapVerdict.UNREACHABLE_IMPL_MISSING.value,
            codes=["implementation_ref_missing"],
        )

    try:
        from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import hcg_untyped

        if hcg_untyped():
            return _a_cap_dict(ACapVerdict.REACHABLE.value, codes=["hcg_untyped_skip_semantics"])
    except ImportError:
        pass

    if graph is not None:
        for cid in capability_ids:
            caps = getattr(graph, "capabilities", {}) or {}
            if cid and cid not in caps and cid not in getattr(graph, "adapters", {}):
                if cid not in state.invalidated_capabilities:
                    return _a_cap_dict(
                        ACapVerdict.UNREACHABLE_CAPABILITY_UNAVAILABLE.value,
                        codes=[f"unknown_capability:{cid}"],
                    )
            if cid in caps:
                cap = caps[cid]
                impl = getattr(cap, "implementation_ref", None)
                if impl is None and route.get("require_impl", True):
                    return _a_cap_dict(
                        ACapVerdict.UNREACHABLE_IMPL_MISSING.value,
                        codes=[f"no_impl:{cid}"],
                    )

    support = route.get("support") or {}
    avail_support = route.get("available_support") or support
    ok, issues = check_support_compatibility(support, avail_support)
    if not ok:
        verdict = compatibility_verdict_from_issues(issues)
        return _a_cap_dict(verdict, codes=issues)

    ic = route.get("input_contract") or {}
    oc = route.get("output_contract") or {}
    iface_verdict, iface_issues = evaluate_interface_compatibility(
        input_contract=ic,
        output_contract=oc,
        available_support=avail_support if isinstance(avail_support, dict) else {},
    )
    if iface_issues:
        return _a_cap_dict(iface_verdict, codes=iface_issues)

    for aid in adapter_ids:
        if aid in state.invalidated_edges:
            return _a_cap_dict(ACapVerdict.EDGE_INVALIDATED.value, codes=[f"invalidated_adapter:{aid}"])

    return _a_cap_dict(ACapVerdict.REACHABLE.value)


def is_admissible(a_sci: Mapping[str, Any], a_cap: Mapping[str, Any]) -> bool:
    return (
        str(a_sci.get("verdict")) in APPLICABLE_ASCI
        and str(a_cap.get("verdict")) in REACHABLE_ACAP
    )


def admissibility_reason_code(
    a_sci: Mapping[str, Any],
    a_cap: Mapping[str, Any],
) -> str:
    """Distinguish UNKNOWN/UNRESOLVED vs hard REJECT for G3 API clarity."""
    asci = str(a_sci.get("verdict") or "")
    acap = str(a_cap.get("verdict") or "")
    if ASciVerdict.UNKNOWN_PENDING_THEORY.value in asci or "UNKNOWN" in asci:
        return "unknown_pending_theory"
    if ACapVerdict.UNRESOLVED.value in acap or "UNRESOLVED" in acap:
        return "unresolved_capability"
    if is_admissible(a_sci, a_cap):
        return "admissible"
    return "reject_not_admissible"


def annotate_route(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
    state: SessionState,
    *,
    graph: Any = None,
) -> Dict[str, Any]:
    """Attach A_sci, A_cap, admissible flag to a route candidate."""
    out = dict(route)
    a_sci = evaluate_A_sci(out, task, theory_arm=state.theory_arm)
    a_cap = evaluate_A_cap(out, state, graph=graph)
    out["A_sci"] = a_sci
    out["A_cap"] = a_cap
    out["admissible"] = is_admissible(a_sci, a_cap)
    if not out.get("route_id"):
        edges = out.get("edges") or out.get("capability_ids") or []
        rid = "route:" + ("+".join(edges[:3]) if edges else "unknown")
        out["route_id"] = rid
    return out


def admissible_routes(
    routes: Sequence[Mapping[str, Any]],
    task: Mapping[str, Any],
    state: SessionState,
    *,
    graph: Any = None,
    admissible_only: bool = False,
) -> List[Dict[str, Any]]:
    annotated = [annotate_route(r, task, state, graph=graph) for r in routes]
    if admissible_only:
        return [r for r in annotated if r.get("admissible")]
    return annotated


def route_from_hcg_path(
    path_info: Mapping[str, Any],
    *,
    route_id: Optional[str] = None,
) -> Dict[str, Any]:
    edges = list(path_info.get("edges") or [])
    caps = [e for e in edges if not str(e).startswith("adapter_")]
    adapters = [e for e in edges if str(e).startswith("adapter_")]
    return build_route_candidate(
        route_id=route_id or (f"route:{'+'.join(edges[:3])}" if edges else None),
        capability_ids=caps or edges,
        adapter_ids=adapters,
        validation_utility=path_info.get("validation_utility"),
        estimated_cost=path_info.get("estimated_cost"),
    ) | {
        "edges": edges,
        "executable": path_info.get("valid", True),
        "available": list(path_info.get("available") or []),
    }
