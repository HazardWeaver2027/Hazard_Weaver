"""Unified @143 execution overlays — solve-mandatory policy without answer leakage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence


def unified_benchmark_enabled(task: Mapping[str, Any]) -> bool:
    meta = task.get("metadata") or {}
    return bool(meta.get("unified_benchmark_v1"))


def _rq4_intervention_spec(task: Mapping[str, Any]) -> Dict[str, Any]:
    meta = task.get("metadata") or {}
    sv = task.get("solver_visible") or {}
    rq4 = meta.get("rq4_intervention") or (sv.get("constraints") or {}).get("rq4_intervention") or {}
    return dict(rq4) if isinstance(rq4, Mapping) else {}


def unified_e12_shock_capability_id(task: Mapping[str, Any]) -> str:
    rq4 = _rq4_intervention_spec(task)
    meta = task.get("metadata") or {}
    return str(
        rq4.get("force_capability_failure")
        or meta.get("w3_shock_capability_id")
        or meta.get("w3_first_attempt_capability_id")
        or ""
    ).strip()


def unified_e12_defer_s0_narrow_active(task: Mapping[str, Any]) -> bool:
    rq4 = _rq4_intervention_spec(task)
    meta = task.get("metadata") or {}
    if bool(rq4.get("defer_s0_narrow_active")) or bool(meta.get("rq4_defer_s0_narrow_active")):
        return True
    if not rq4.get("defer_s0_narrow_allowed_to_shock"):
        return False
    shock = unified_e12_shock_capability_id(task)
    if not shock:
        return False
    allowed = [
        str(e).strip()
        for e in ((task.get("solver_visible") or {}).get("inputs") or {}).get("allowed_edge_ids") or []
        if str(e).strip()
    ]
    return len(allowed) == 1 and allowed[0] == shock


def unified_e12_shock_s0_operational_a_sci(
    task: Mapping[str, Any],
    route: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """E12 defer+s0: shock/decoy must be admissible at s0 so enumerate ≠ 0.

    HKC frozen registry may mark routing-pool decoys ``SCI_INAPPLICABLE``; the
    W3 E12 contract still requires the agent to commit the shock route, observe
    forced execution failure, then reinstantiate onto gold (Full) or stay invalid
    (static_s0).
    """
    if not unified_benchmark_enabled(task):
        return None
    rq4 = _rq4_intervention_spec(task)
    if str(rq4.get("category") or "") != "E12_execution_failure":
        return None
    if not unified_e12_defer_s0_narrow_active(task):
        return None
    shock = unified_e12_shock_capability_id(task)
    if not shock:
        return None
    cap = str((route.get("capability_ids") or route.get("edges") or [""])[0]).strip()
    if cap != shock:
        return None
    from hazardweaver.hwa.scientific_controller.reason_codes import ASciVerdict

    return {
        "verdict": ASciVerdict.APPLICABLE.value,
        "codes": [],
        "refs": [],
        "source": "unified_e12_shock_s0_operational",
    }


def unified_e12_shock_s0_operational_a_cap(
    task: Mapping[str, Any],
    route: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """E12 defer+s0: HCG-typed shock/decoy must be committable at s0 (A_cap as well as A_sci)."""
    if unified_e12_shock_s0_operational_a_sci(task, route) is None:
        return None
    from hazardweaver.hwa.scientific_controller.reason_codes import ACapVerdict

    return {
        "verdict": ACapVerdict.REACHABLE.value,
        "codes": [],
        "missing_artifacts": [],
        "source": "unified_e12_shock_s0_operational",
    }


def apply_unified_e12_shock_s0_operational_admissibility(
    task: Mapping[str, Any],
    route: MutableMapping[str, Any],
) -> None:
    override = unified_e12_shock_s0_operational_a_sci(task, route)
    if override is None:
        return
    route["A_sci"] = override
    route["hkc_binding_source"] = "unified_e12_shock_s0_operational"
    cap_override = unified_e12_shock_s0_operational_a_cap(task, route)
    if cap_override is not None:
        route["A_cap"] = cap_override


def unified_inventory_allowed_caps(task: Mapping[str, Any]) -> set[str]:
    """Curator whitelist caps for unified g6 cells (no solver pin)."""
    meta = task.get("metadata") or {}
    caps: set[str] = set()
    for key in ("rq4_full_allowed_edge_ids",):
        caps.update(str(c).strip() for c in (meta.get(key) or []) if str(c).strip())
    if not caps:
        rq4 = _rq4_intervention_spec(task)
        forced = str(rq4.get("force_capability_failure") or "").strip()
        gold = str(
            meta.get("unified_scenario_gold_capability_id")
            or meta.get("w3_gold_capability_id")
            or ""
        ).strip()
        blocked = {str(c).strip() for c in (rq4.get("defer_s0_blocked_capabilities") or []) if c}
        if gold:
            caps.add(gold)
        if forced:
            caps.add(forced)
        caps.update(blocked)
    return caps


def _current_allowed_caps(task: Mapping[str, Any]) -> set[str]:
    sv = (task.get("solver_visible") or {}).get("inputs") or {}
    return {str(c).strip() for c in (sv.get("allowed_edge_ids") or []) if str(c).strip()}


def unified_e12_post_shock_allowed_edge_ids(
    task: Mapping[str, Any],
    *,
    full_allowed: Sequence[str],
    invalidated: set[str],
) -> List[str]:
    """After E12 shock failure: Full arm restores curator whitelist minus invalidated caps."""
    return [
        str(e).strip()
        for e in full_allowed
        if str(e).strip() and str(e).strip() not in invalidated
    ]


def unified_g6_operational_a_sci(
    task: Mapping[str, Any],
    route: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Unified W3/E12 g6 routes: curator whitelist must enumerate without frozen-K gaps."""
    if not unified_benchmark_enabled(task) or not route.get("g6_coreexec"):
        return None
    cap = str((route.get("capability_ids") or route.get("edges") or [""])[0]).strip()
    if not cap:
        return None

    shock = unified_e12_shock_s0_operational_a_sci(task, route)
    if shock is not None:
        return shock

    meta = task.get("metadata") or {}
    gold = str(
        meta.get("unified_scenario_gold_capability_id") or meta.get("w3_gold_capability_id") or ""
    ).strip()
    if gold and cap == gold:
        from hazardweaver.hwa.scientific_controller.reason_codes import ASciVerdict

        return {
            "verdict": ASciVerdict.APPLICABLE.value,
            "codes": [],
            "refs": [],
            "source": "unified_scenario_gold_curator_v1",
        }

    allowed_now = _current_allowed_caps(task)
    if cap not in allowed_now:
        return None
    whitelist = unified_inventory_allowed_caps(task) or allowed_now
    if cap not in whitelist:
        return None

    a_sci = route.get("A_sci") or {}
    verdict = str(a_sci.get("verdict") or "")
    codes = {str(c) for c in (a_sci.get("codes") or [])}
    from hazardweaver.hwa.scientific_controller.reason_codes import ASciVerdict

    if verdict == ASciVerdict.APPLICABLE.value:
        return None
    exact_grounding = bool(
        meta.get("e1e3_hkc_exact_require_grounding") or meta.get("mh_hkc_exact_require_grounding")
    )
    if exact_grounding and (
        verdict == ASciVerdict.INAPPLICABLE_THEORY_MISMATCH.value
        or "task_require_not_grounded_in_route" in codes
    ):
        return None
    if verdict in {
        ASciVerdict.UNKNOWN_PENDING_THEORY.value,
        ASciVerdict.INAPPLICABLE_THEORY_MISMATCH.value,
    } or codes.intersection({"missing_frozen_binding", "theory_mismatch", "task_require_not_grounded_in_route"}):
        return {
            "verdict": ASciVerdict.APPLICABLE.value,
            "codes": [],
            "refs": [],
            "source": "unified_g6_whitelist_operational",
        }
    return None


def apply_unified_g6_operational_admissibility(
    task: Mapping[str, Any],
    route: MutableMapping[str, Any],
) -> None:
    override = unified_g6_operational_a_sci(task, route)
    if override is None:
        return
    route["A_sci"] = override
    route["hkc_binding_source"] = str(override.get("source") or "unified_g6_whitelist_operational")


def seal_unified_runtime_allowed_edges(
    task: MutableMapping[str, Any],
    inventory_row: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Apply W3 defer+s0 shock-only narrow when configured; no scenario-gold pin."""
    if not inventory_row.get("unified_benchmark_v1"):
        return task
    from hazardweaver.hwa.experiments.ablation_manual_pilot_v1 import unified_runtime_allowed_edge_ids

    narrow = unified_runtime_allowed_edge_ids(inventory_row)
    if narrow:
        inp = task.setdefault("solver_visible", {}).setdefault("inputs", {})
        inp["allowed_edge_ids"] = list(narrow)
    return task


def solve_mandatory_episode_policy_enabled(task: Mapping[str, Any]) -> bool:
    """Unified @143 and manual ablation pilot share solve-mandatory terminal hygiene."""
    if unified_benchmark_enabled(task):
        return True
    from hazardweaver.hwa.experiments.ablation_manual_pilot_v1 import ablation_manual_pilot_enabled

    return ablation_manual_pilot_enabled(task)


def apply_unified_benchmark_overlays(
    task: MutableMapping[str, Any],
    inventory_row: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Stamp unified benchmark metadata for runtime gates and REMSA (no gold in solver_view)."""
    if not inventory_row.get("unified_benchmark_v1"):
        return task
    meta = task.setdefault("metadata", {})
    meta["unified_benchmark_v1"] = 1
    meta["route_eligibility"] = str(inventory_row.get("route_eligibility") or "")
    for key in (
        "ablation_witness_capability_id",
        "ablation_decoy_capability_id",
        "ablation_mechanism",
        "ablation_decoy_utility_boost",
        "w3_first_attempt_capability_id",
        "w3_shock_capability_id",
        "w3_gold_capability_id",
        "w3_situation_overlay",
    ):
        if inventory_row.get(key) is not None:
            meta[key] = inventory_row[key]
    situation = str(inventory_row.get("w3_situation_overlay") or "").strip()
    if situation:
        goal = str(task.get("user_facing_goal") or task.get("user_goal") or "").strip()
        if situation not in goal:
            task["user_facing_goal"] = f"{situation} {goal}".strip()
    briefs = inventory_row.get("solver_visible_route_briefs") or inventory_row.get("ablation_route_briefs")
    if isinstance(briefs, Mapping) and briefs:
        inp = task.setdefault("solver_visible", {}).setdefault("inputs", {})
        inp["route_briefs"] = dict(briefs)
    goal = str(task.get("user_facing_goal") or "").strip()
    if goal and str(inventory_row.get("expected_action") or "solve").strip().lower() == "solve":
        if "Abstention is forbidden" not in goal:
            task["user_facing_goal"] = (
                f"{goal} Review the symmetric route briefs before commit. "
                "Abstention is forbidden on solve-mandatory cells when admissible routes exist: "
                "commit one route, run_capability, then submit_solution."
            )
    if str(inventory_row.get("expected_action") or "solve").strip().lower() == "solve":
        meta["react_clarify_forbidden"] = True
        meta["ablation_require_run_capability"] = True
    from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import unified_scenario_gold_capability_id

    gold = unified_scenario_gold_capability_id(inventory_row)
    if gold:
        meta["unified_scenario_gold_capability_id"] = gold
    allowed = [
        str(e).strip()
        for e in (inventory_row.get("allowed_edge_ids") or [])
        if str(e).strip()
    ]
    if allowed:
        meta["rq4_full_allowed_edge_ids"] = list(allowed)
    if str(inventory_row.get("route_eligibility") or "") == "multi_eligible":
        meta["unified_require_route_commit_before_submit"] = True
        meta["unified_route_commit_step_nudge"] = int(
            inventory_row.get("unified_route_commit_step_nudge") or 4
        )
    return task


def unified_abstention_allowed(
    task: Mapping[str, Any],
    reason_code: str,
    routes: Any,
) -> Dict[str, Any]:
    """Solve-mandatory unified cells: mirror ablation gate, track-agnostic."""
    if not unified_benchmark_enabled(task):
        return {"allowed": True}
    meta = task.get("metadata") or {}
    if str(meta.get("expected_action") or "solve").strip().lower() != "solve":
        return {"allowed": True}
    from hazardweaver.hwa.runtime.abstention_gate import pi_adm_empty

    route_list = list(routes or [])
    if not route_list:
        return {
            "allowed": False,
            "error": "unified_solve_mandatory_requires_route_probe",
            "message": (
                "Unified solve-mandatory: abstention forbidden before route enumeration. "
                "Call controller_enumerate_routes, commit a route, run_capability, then submit_solution."
            ),
            "reason_code": str(reason_code or ""),
        }
    if not pi_adm_empty(route_list):
        return {
            "allowed": False,
            "error": "unified_solve_mandatory_admissible_routes",
            "message": (
                "Unified solve-mandatory: abstention is forbidden while admissible routes exist. "
                "Use controller_commit_route, run_capability with execution handles from the goal, "
                "then submit_solution."
            ),
            "reason_code": str(reason_code or ""),
        }
    return {"allowed": True}


def unified_inspect_artifact_blocked(
    task: Mapping[str, Any],
    artifact_id: str,
    *,
    prior_inspect_count: int = 0,
) -> Optional[Dict[str, Any]]:
    """Block inspect loops on solve-mandatory unified E1-E3 cells."""
    if not solve_mandatory_episode_policy_enabled(task):
        return None
    meta = task.get("metadata") or {}
    if str(meta.get("expected_action") or "solve").strip().lower() != "solve":
        return None
    aid = str(artifact_id or "").strip().upper()
    if aid.startswith("CAP-E1E3-") or aid.startswith("EDGE:CAP-E1E3-"):
        return {
            "ok": False,
            "error": "inspect_capability_route_forbidden",
            "tool": "inspect_artifact",
            "message": (
                "E1-E3 solve-mandatory: do not inspect capability routes. "
                "Use controller_enumerate_routes, controller_commit_route, "
                "run_capability, then submit_solution."
            ),
        }
    if prior_inspect_count >= 2:
        return {
            "ok": False,
            "error": "inspect_artifact_budget_exhausted",
            "tool": "inspect_artifact",
            "message": (
                "Solve-mandatory inspect budget exhausted (max 2). "
                "Commit a route and run_capability."
            ),
        }
    return None


def gate_unified_route_commit_before_submit(
    task: Mapping[str, Any],
    route_id: str,
) -> Optional[Dict[str, Any]]:
    """Block submit_solution without explicit route_id on unified multi-route cells."""
    if not unified_benchmark_enabled(task):
        return None
    meta = task.get("metadata") or {}
    if not meta.get("unified_require_route_commit_before_submit"):
        return None
    rid = str(route_id or "").strip()
    if rid:
        return None
    return {
        "ok": False,
        "error": "unified_route_commit_required",
        "tool": "submit_solution",
        "message": (
            "Unified multi-route: submit_solution requires route_id from controller_commit_route. "
            "Enumerate routes, commit one capability, run_capability, then submit."
        ),
    }


def gate_submit_requires_run_capability(
    task: Mapping[str, Any],
    workdir: Path,
    route_id: str,
) -> Optional[Dict[str, Any]]:
    """Require a successful registry execution for the submitted route."""
    meta = task.get("metadata") or {}
    if not meta.get("ablation_require_run_capability"):
        return None
    from hazardweaver.hwa.agent_runtime.execution_schema import list_registry_executions

    rid = str(route_id or "").strip()
    success = [
        e
        for e in list_registry_executions(Path(workdir))
        if str(e.get("status") or "") in {"ok", "success"}
        and str(e.get("route_id") or "").strip() == rid
    ]
    if success:
        return None
    return {
        "ok": False,
        "error": "submit_requires_run_capability",
        "tool": "submit_solution",
        "route_id": rid,
        "message": (
            "Solve-mandatory: submit_solution requires a successful run_capability "
            "(or controller_commit_route execution) for this route_id first."
        ),
    }


def outer_wall_s_for_inventory_row(
    inventory_row: Mapping[str, Any],
    *,
    base_s: int | None = None,
) -> int:
    """Per-cell outer wall (seconds). Must be >= inner ``HWA_AGENT_MAX_WALL_S`` + buffer."""
    import os

    from hazardweaver.hwa.experiments.unified_backbone_agent_limits_v1 import (
        OUTER_WALL_BUFFER_S,
        UNIFIED_CELL_OUTER_WALL_S,
    )

    if base_s is not None:
        return int(base_s)
    env_outer = os.environ.get("HWA_CELL_OUTER_WALL_S")
    if env_outer:
        return int(env_outer)
    inner = int(os.environ.get("HWA_AGENT_MAX_WALL_S") or 0)
    if inner > 0:
        return max(UNIFIED_CELL_OUTER_WALL_S, inner + OUTER_WALL_BUFFER_S)
    return UNIFIED_CELL_OUTER_WALL_S
