"""G6 CoreExec controller bridge — allowed_edge_ids only (no reference_view witnesses).

Solver-visible route candidates are synthesized from ``solver_visible.inputs.allowed_edge_ids``
only. ``reference_view.accepted_witnesses`` is never read (construction-time / HWB only).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from hazardweaver.hwa.scientific_controller.reason_codes import ACapVerdict, ASciVerdict


def is_same_llm_g6_coreexec_task(task: Mapping[str, Any]) -> bool:
    meta = task.get("metadata") or {}
    return bool(meta.get("same_llm_g6_coreexec") or meta.get("hwb_headline_inventory"))


def uses_allowed_edge_route_synthesis(task: Mapping[str, Any]) -> bool:
    """Tasks whose controller routes come from solver_visible.inputs.allowed_edge_ids only."""
    if not is_same_llm_g6_coreexec_task(task):
        return False
    meta = task.get("metadata") or {}
    if str(meta.get("headline_route_mode") or "") == "hcg_multi_hop":
        return False
    return True


def headline_hcg_hybrid_g6_fallback_eligible(task: Mapping[str, Any]) -> bool:
    """L4 Path C: try HCG multi-hop first; fall back to g6 when enumerate is empty (G0)."""
    from hazardweaver.hwa.experiments.coupled_4m_mode_v1 import hcg_g6_fallback_disabled

    if hcg_g6_fallback_disabled():
        return False
    if not is_same_llm_g6_coreexec_task(task):
        return False
    meta = task.get("metadata") or {}
    if meta.get("hcg_typed_probe") or meta.get("hcg_input_contract"):
        return False
    if str(meta.get("headline_route_mode") or "") != "hcg_multi_hop":
        return False
    return bool(_allowed_edges(task))


def _allowed_edges(task: Mapping[str, Any]) -> List[str]:
    solver = task.get("solver_visible") or {}
    return [str(e) for e in (solver.get("inputs") or {}).get("allowed_edge_ids") or []]


def _g6_route_contract_overlay(task: Mapping[str, Any]) -> Dict[str, Any]:
    meta = task.get("metadata") or {}
    overlay: Dict[str, Any] = {}
    ic = meta.get("hcg_input_contract")
    if isinstance(ic, Mapping) and ic:
        overlay["input_contract"] = dict(ic)
    oc = meta.get("hcg_output_contract")
    if isinstance(oc, Mapping) and oc:
        overlay["output_contract"] = dict(oc)
    avail = meta.get("hcg_available_support")
    if isinstance(avail, Mapping) and avail:
        overlay["available_support"] = dict(avail)
    override = meta.get("scientific_condition_override")
    if isinstance(override, Mapping) and override:
        overlay["scientific_condition_override"] = dict(override)
    return overlay


def g6_requires_typed_acap(task: Mapping[str, Any], route: Mapping[str, Any]) -> bool:
    meta = task.get("metadata") or {}
    if meta.get("hcg_typed_probe"):
        return True
    if route.get("input_contract") or route.get("output_contract"):
        return True
    if meta.get("scientific_condition_override"):
        return True
    return False


def routes_from_allowed_edges(task: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Build controller-visible routes from allowed_edge_ids (Option B — no witness leakage)."""
    allowed = _allowed_edges(task)
    if not allowed:
        return []

    schema_edges = sorted(e for e in allowed if e.startswith("schema_map"))
    schema = schema_edges[0] if schema_edges else ""
    capabilities = sorted(e for e in allowed if not e.startswith("schema_map"))
    contract_overlay = _g6_route_contract_overlay(task)

    routes: List[Dict[str, Any]] = []
    for cap in capabilities:
        edges = [schema, cap] if schema else [cap]
        route_id = f"route:cap:{cap}"
        route: Dict[str, Any] = {
            "route_id": route_id,
            "edges": edges,
            "capability_ids": [cap],
            "admissible": True,
            "executable": True,
            "g6_coreexec": True,
            "route_source": "allowed_edge_ids",
            "A_sci": {
                "verdict": ASciVerdict.APPLICABLE.value,
                "codes": [],
                "refs": [],
                "source": "allowed_edge_ids",
            },
            "A_cap": {
                "verdict": ACapVerdict.REACHABLE.value,
                "codes": [],
                "missing_artifacts": [],
                "source": "allowed_edge_ids",
            },
        }
        from hazardweaver.hwa.scientific_controller.e1e3_hkc_bindings_v1 import e1e3_route_family_overlay

        route.update(e1e3_route_family_overlay(cap))
        route.update(contract_overlay)
        routes.append(route)
    return routes


def _commit_overlay_for_src_terminal(route: Mapping[str, Any]) -> Dict[str, Any]:
    """Allow SRC propose/commit for HKC-bound terminal caps not in default graph store."""
    from hazardweaver.hwa.scientific_controller.admissibility import is_admissible

    out = dict(route)
    a_sci = dict(out.get("A_sci") or {})
    a_cap = dict(out.get("A_cap") or {})
    if not is_admissible(a_sci, a_cap):
        a_sci = {
            **a_sci,
            "verdict": ASciVerdict.APPLICABLE.value,
            "codes": list(a_sci.get("codes") or []),
            "commit_overlay": True,
        }
        a_cap = {
            "verdict": ACapVerdict.REACHABLE.value,
            "codes": [],
            "missing_artifacts": [],
            "source": "src_terminal_cap",
        }
    out["A_sci"] = a_sci
    out["A_cap"] = a_cap
    out["admissible"] = is_admissible(a_sci, a_cap)
    return out


def prepare_route_for_controller(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
    state: Any,
    *,
    graph: Any = None,
) -> Dict[str, Any]:
    """Annotate route for propose/commit; g6 allowed-edge routes skip HCG graph re-check."""
    if route.get("g6_coreexec") and g6_requires_typed_acap(task, route):
        from hazardweaver.hwa.route_controller.admissibility_gate import AdmissibilityGate

        # g6 enumerate never passes graph — tabular caps are not in the session graph.
        return AdmissibilityGate(task).annotate_route(route, state)
    if route.get("g6_coreexec") and not g6_requires_typed_acap(task, route):
        return dict(route)
    if route.get("track_src_terminal") or route.get("pfdf_portfolio_terminal") or route.get(
        "seven_track_terminal"
    ):
        return _commit_overlay_for_src_terminal(route)
    from hazardweaver.hwa.scientific_controller.admissibility import annotate_route

    return annotate_route(route, task, state, graph=graph)


def execution_ids_from_run_capability_result(result: Mapping[str, Any]) -> tuple[str, str]:
    """Extract execution/artifact ids from ``run_capability`` return (mint may sit on outer wrap)."""
    inner = result.get("result") if isinstance(result.get("result"), Mapping) else result
    eid = ""
    aid = ""
    if isinstance(inner, Mapping):
        eid = str(inner.get("execution_id") or "").strip()
        aid = str(inner.get("final_artifact_id") or "").strip()
        er = inner.get("execution_result")
        if isinstance(er, Mapping):
            eid = eid or str(er.get("execution_id") or "").strip()
            aid = aid or str(er.get("final_artifact_id") or "").strip()
    eid = eid or str(result.get("execution_id") or "").strip()
    aid = aid or str(result.get("final_artifact_id") or "").strip()
    outer_er = result.get("execution_result")
    if isinstance(outer_er, Mapping):
        eid = eid or str(outer_er.get("execution_id") or "").strip()
        aid = aid or str(outer_er.get("final_artifact_id") or "").strip()
    return eid, aid


def execute_g6_coreexec_route(
    controller: Any,
    route: Mapping[str, Any],
    *,
    execution_token: Optional[str] = None,
    host: Any = None,
    out_dir: Any = None,
    handles: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute terminal g6 tabular capability via run_capability (lease already bound)."""
    from hazardweaver.hwa.agent_runtime.unified_tools import run_capability

    rid = str(route.get("route_id") or "")
    # Token is validated (and consumed) inside run_capability — do not pre-validate here.

    caps = [str(c) for c in (route.get("capability_ids") or []) if c]
    terminal = caps[-1] if caps else ""
    if not terminal:
        return {"ok": False, "error": "no_terminal_capability", "route_id": rid}

    meta = controller.task.get("metadata") or {}
    from hazardweaver.hwa.experiments.rq4_route_intervention_v1 import maybe_rq4_execute_block
    from hazardweaver.hwa.runtime.execution_failure_class import attach_failure_class

    blocked = maybe_rq4_execute_block(
        "run_capability",
        {"capability_id": terminal},
        controller.task,
    )
    if blocked is not None:
        fail_view = attach_failure_class({**blocked, "ok": False})
        if hasattr(controller, "handle_execution_failure"):
            try:
                controller.handle_execution_failure(
                    route,
                    failed_capability_id=terminal,
                    reason=str(blocked.get("error") or "rq4_forced_execution_failure"),
                )
            except Exception:  # noqa: BLE001
                pass
        return {
            "ok": False,
            "route_id": rid,
            "capability_id": terminal,
            "error": fail_view.get("error") or "rq4_forced_execution_failure",
            "failure_class": fail_view.get("failure_class", "unknown"),
            "result": fail_view,
        }

    sv_in = (controller.task.get("solver_visible") or {}).get("inputs") or {}
    params = sv_in.get("parameters") or {}
    agent_handles = dict(handles or {})
    from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

    if agent_strict_v2_enabled():
        from hazardweaver.hwa.experiments.ablation_manual_pilot_execution_v1 import (
            ablation_authoritative_execution_handles,
        )
        from hazardweaver.hwa.experiments.agent_strict_v2 import (
            apply_strict_run_capability_handle_defaults,
            validate_agent_handles,
        )

        hidden = meta.get("grader_hidden") or meta
        inv_row = hidden if isinstance(hidden, Mapping) else meta
        agent_handles = ablation_authoritative_execution_handles(
            agent_handles,
            inventory_row=inv_row if isinstance(inv_row, Mapping) else meta,
            task_metadata=meta if isinstance(meta, Mapping) else None,
        )
        agent_handles = apply_strict_run_capability_handle_defaults(
            terminal,
            agent_handles,
            inventory_row=inv_row if isinstance(inv_row, Mapping) else None,
            task_metadata=meta if isinstance(meta, Mapping) else None,
        )
        ok_h, errs, _ = validate_agent_handles(terminal, agent_handles, provenance_src="agent")
        if not ok_h:
            return {
                "ok": False,
                "error": "agent_strict_v2_requires_explicit_handles",
                "details": errs,
                "route_id": rid,
                "capability_id": terminal,
            }
    else:
        agent_handles = {}
        for key in ("scenario_id", "split", "anchor_id"):
            val = meta.get(key) or sv_in.get(key) or params.get(key)
            if val is not None and str(val).strip():
                agent_handles[key] = val
        record_id = str(meta.get("pfdf_record_id") or "").strip()
        if not record_id:
            try:
                from hazardweaver.hwa.experiments.headline_pfdf_record_id_v1 import resolve_headline_pfdf_record_id

                inv_row = {
                    "instance_id": meta.get("instance_id"),
                    "scenario_id": meta.get("scenario_id") or params.get("scenario_id"),
                    "source": meta.get("source"),
                }
                record_id = resolve_headline_pfdf_record_id(inv_row, task=controller.task)
            except KeyError:
                record_id = ""
        if record_id:
            agent_handles["record_id"] = record_id
    workdir = out_dir or controller.workdir
    if workdir:
        agent_handles["out_dir"] = str(workdir)
    agent_handles["route_id"] = rid
    result = run_capability(
        terminal,
        host=host,
        out_dir=workdir,
        route_id=rid,
        controller_token=execution_token,
        handles=agent_handles,
        record_id=agent_handles.get("record_id"),
        scenario_id=agent_handles.get("scenario_id"),
        split=agent_handles.get("split"),
    )
    inner = result.get("result") if isinstance(result.get("result"), Mapping) else result
    execution_id, final_artifact_id = execution_ids_from_run_capability_result(result)
    ok = bool(result.get("ok", True))
    from hazardweaver.hwa.runtime.execution_failure_class import attach_failure_class

    if ok:
        controller.state.active_route_id = rid
        controller._log(
            "execute",
            route=dict(route),
            execution_id=execution_id,
            notes="g6_coreexec_tabular",
            extra={"ok": True, "capability_id": terminal},
        )
    else:
        err = str(
            result.get("error")
            or (inner.get("error") if isinstance(inner, Mapping) else None)
            or "execution_failed"
        )
        fail_view = attach_failure_class({**dict(result), "ok": False, "error": err})
        controller._log(
            "execute",
            route=dict(route),
            notes=err,
            extra={
                "ok": False,
                "capability_id": terminal,
                "failure_class": fail_view.get("failure_class"),
            },
        )
        if hasattr(controller, "handle_execution_failure"):
            try:
                controller.handle_execution_failure(
                    route,
                    failed_capability_id=terminal,
                    reason=err,
                )
            except Exception:  # noqa: BLE001
                pass
        result = fail_view
    out = {
        "ok": ok,
        "route_id": rid,
        "capability_id": terminal,
        "execution_id": execution_id or None,
        "final_artifact_id": final_artifact_id or None,
        "result": result,
    }
    if not ok:
        out["failure_class"] = result.get("failure_class", "unknown")
        out["error"] = result.get("error") or "execution_failed"
    return out


# Backward-compat alias (tests / imports); do not use reference_view.
g6_routes_from_task = routes_from_allowed_edges
