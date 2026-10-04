"""E12 Full-arm agent loop guards — shock-only s0 commit + post-shock refresh nudge."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import (
    _current_allowed_caps,
    _rq4_intervention_spec,
    unified_benchmark_enabled,
    unified_e12_defer_s0_narrow_active,
    unified_e12_shock_capability_id,
)


def _route_caps(route: Mapping[str, Any]) -> list[str]:
    return [
        str(c).strip()
        for c in (route.get("capability_ids") or route.get("edges") or [])
        if str(c).strip() and not str(c).startswith("schema")
    ]


def unified_e12_s0_shock_only_violation(
    task: Mapping[str, Any],
    route: Mapping[str, Any],
) -> Optional[str]:
    """Reject s0 commits off the forced shock route (E12 defer+narrow contract)."""
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
    caps = _route_caps(route)
    if not caps:
        return "route_has_no_capabilities"
    if any(cap != shock for cap in caps):
        return "e12_s0_shock_only_commit"
    return None


def unified_e12_post_shock_context(
    task: Mapping[str, Any],
    state: Any,
) -> Optional[Dict[str, Any]]:
    """After shock invalidation, Full arm is in post-failure refresh (multi-route admissible)."""
    if not unified_benchmark_enabled(task):
        return None
    rq4 = _rq4_intervention_spec(task)
    if str(rq4.get("category") or "") != "E12_execution_failure":
        return None
    if unified_e12_defer_s0_narrow_active(task):
        return None
    shock = unified_e12_shock_capability_id(task)
    invalidated = set(getattr(state, "invalidated_capabilities", None) or ())
    if not shock or shock not in invalidated:
        return None
    allowed = sorted(_current_allowed_caps(task))
    if not allowed or shock in allowed:
        return None
    meta = task.get("metadata") or {}
    gold = str(
        meta.get("unified_scenario_gold_capability_id")
        or meta.get("w3_gold_capability_id")
        or ""
    ).strip()
    return {
        "gold_capability_id": gold,
        "allowed_edge_ids": allowed,
        "n_allowed": len(allowed),
    }


def apply_unified_e12_post_shock_enumerate_overlay(
    controller: Any,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Mark post-shock refresh; track enumerate-only spin (no route_id leakage)."""
    ctx = unified_e12_post_shock_context(controller.task, controller.state)
    out = dict(result)
    if not ctx:
        controller._e12_post_shock_enum_streak = 0  # noqa: SLF001
        return out
    n_adm = int(out.get("n_routes") or len([r for r in (out.get("routes") or []) if r.get("admissible")]) or 0)
    out["e12_post_shock_refresh"] = True
    out["n_post_shock_admissible"] = n_adm
    out["next_step"] = "controller_propose_route"
    try:
        from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import route_eligibility_static_whitelist_s0

        static_whitelist = route_eligibility_static_whitelist_s0()
    except ImportError:
        static_whitelist = False
    refresh_phrase = (
        "Admissible routes on the frozen s0 whitelist remain after shock invalidation."
        if static_whitelist
        else "Admissible routes were refreshed."
    )
    out["message"] = (
        "E12 post-shock: the committed shock route failed and was invalidated. "
        f"{refresh_phrase} Review symmetric route briefs, "
        "choose one admissible route, commit, then submit_solution. "
        "Do not call controller_enumerate_routes repeatedly."
    )
    streak = int(getattr(controller, "_e12_post_shock_enum_streak", 0) or 0) + 1
    controller._e12_post_shock_enum_streak = streak  # noqa: SLF001
    out["e12_post_shock_enumerate_streak"] = streak
    if streak >= 3:
        out["e12_enumerate_spin_warning"] = True
    return out


def block_unified_e12_post_shock_enumerate_spin(
    controller: Any,
) -> Optional[Dict[str, Any]]:
    """Forbid pure enumerate loops once post-shock refresh is active."""
    ctx = unified_e12_post_shock_context(controller.task, controller.state)
    if not ctx:
        return None
    active = str(getattr(controller.state, "active_route_id", "") or "").strip()
    pending = str(getattr(controller.state, "pending_route_id", "") or "").strip()
    if active or pending:
        return None
    streak = int(getattr(controller, "_e12_post_shock_enum_streak", 0) or 0)
    if streak < 2:
        return None
    return {
        "ok": False,
        "error": "e12_post_shock_enumerate_spin_forbidden",
        "next_step": "controller_propose_route",
        "e12_post_shock_enumerate_streak": streak,
        "n_post_shock_admissible": ctx.get("n_allowed"),
        "message": (
            "Post-shock enumerate spin forbidden: shock already failed and admissible "
            "routes were refreshed. Choose a route from the last enumeration and call "
            "controller_propose_route, then controller_commit_route, then submit_solution."
        ),
    }


def reset_unified_e12_post_shock_enum_streak(controller: Any, route: Mapping[str, Any]) -> None:
    if not unified_e12_post_shock_context(controller.task, controller.state):
        return
    allowed = set(_current_allowed_caps(controller.task))
    caps = _route_caps(route)
    if caps and all(cap in allowed for cap in caps):
        controller._e12_post_shock_enum_streak = 0  # noqa: SLF001


E12_MAX_ROUTE_COMMITS = 2


def count_controller_route_commits(workdir: Any) -> int:
    """Count committed routes in controller_decisions.jsonl."""
    from pathlib import Path

    path = Path(getattr(workdir, "workdir", workdir) or "") / "controller_decisions.jsonl"
    if not path.is_file():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("action") or "") == "commit":
            n += 1
    return n


def maybe_terminalize_e12_over_commit_spin(
    env: Any,
    task: Mapping[str, Any],
    *,
    n_commits: int,
) -> bool:
    """Fast invalid when E12 Full agent exceeds shock+recovery commit budget (>2)."""
    if not unified_benchmark_enabled(task):
        return False
    rq4 = _rq4_intervention_spec(task)
    if str(rq4.get("category") or "") != "E12_execution_failure":
        return False
    if int(n_commits) <= E12_MAX_ROUTE_COMMITS:
        return False
    from hazardweaver.hwa.experiments.ablation_manual_pilot_terminal_v1 import (
        force_ablation_terminal_pre_controller_spin,
        solve_mandatory_episode_policy_enabled,
    )

    if not solve_mandatory_episode_policy_enabled(task):
        return False
    return force_ablation_terminal_pre_controller_spin(
        env,
        task,
        {
            "error": "over_commit_spin",
            "n_commits": int(n_commits),
            "e12_max_route_commits": E12_MAX_ROUTE_COMMITS,
        },
        emit_source="unified_e12_over_commit_spin_v1",
    )


def unified_e12_post_shock_loop_nudge(
    controller: Any,
    refresh: Mapping[str, Any],
    *,
    step_n: int,
) -> Optional[str]:
    """Neutral system hint when agent stalls after post-shock refresh."""
    ctx = unified_e12_post_shock_context(controller.task, controller.state)
    if not ctx:
        return None
    if str(getattr(controller.state, "active_route_id", "") or "").strip():
        return None
    meta = controller.task.get("metadata") or {}
    nudge_after = int(meta.get("unified_route_commit_step_nudge") or 2)
    streak = int(getattr(controller, "_e12_post_shock_enum_streak", 0) or 0)
    if streak < 1 and step_n < nudge_after:
        return None
    n_adm = refresh.get("n_post_shock_admissible") or ctx.get("n_allowed") or "?"
    return (
        f"E12 post-shock refresh (step {step_n}): shock execution failed; "
        f"{n_adm} admissible route(s) remain. Stop re-enumerating. "
        "Review route briefs, call controller_propose_route on an admissible route_id, "
        "controller_commit_route, then submit_solution."
    )
