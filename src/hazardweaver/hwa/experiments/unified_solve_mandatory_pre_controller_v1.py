"""Solve-mandatory pre-controller hygiene — list_inventory budget + enumerate nudge (Tier 0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import solve_mandatory_episode_policy_enabled

LIST_INVENTORY_BUDGET = 3
POST_ENUMERATE_LIST_INVENTORY_BUDGET = 8
PRE_CONTROLLER_ENUMERATE_NUDGE_STEP = 2

_CONTROLLER_WORKFLOW_TOOLS = frozenset(
    {
        "controller_enumerate_routes",
        "controller_propose_route",
        "controller_commit_route",
    }
)


@dataclass
class PreControllerGuardState:
    enumerate_seen: bool = False
    list_inventory_streak: int = 0
    post_enumerate_list_inventory_streak: int = 0
    run_capability_ok_seen: bool = False
    enumerate_nudge_sent: bool = False


def pre_controller_policy_active(task: Mapping[str, Any], *, use_controller: bool) -> bool:
    return bool(use_controller and solve_mandatory_episode_policy_enabled(task))


def record_pre_controller_tool(
    state: PreControllerGuardState,
    tool_name: str,
    *,
    observation_ok: Optional[bool] = None,
) -> None:
    name = str(tool_name or "").strip()
    if name == "run_capability" and observation_ok:
        state.run_capability_ok_seen = True
        state.post_enumerate_list_inventory_streak = 0
        return
    if name in _CONTROLLER_WORKFLOW_TOOLS:
        state.enumerate_seen = True
        state.list_inventory_streak = 0
        return
    if name == "list_inventory":
        if state.enumerate_seen and not state.run_capability_ok_seen:
            state.post_enumerate_list_inventory_streak += 1
        elif not state.enumerate_seen:
            state.list_inventory_streak += 1


def pre_controller_enumerate_nudge(
    task: Mapping[str, Any],
    state: PreControllerGuardState,
    *,
    step_n: int,
    use_controller: bool,
) -> Optional[str]:
    if not pre_controller_policy_active(task, use_controller=use_controller):
        return None
    if state.enumerate_seen or state.enumerate_nudge_sent:
        return None
    if int(step_n) < PRE_CONTROLLER_ENUMERATE_NUDGE_STEP:
        return None
    state.enumerate_nudge_sent = True
    return (
        "Solve-mandatory controller workflow: do not repeat list_inventory. "
        "Call controller_enumerate_routes once, review the route briefs, then "
        "controller_propose_route → controller_commit_route → run_capability → submit_solution."
    )


def should_terminalize_list_inventory_spin(
    task: Mapping[str, Any],
    state: PreControllerGuardState,
    *,
    use_controller: bool,
) -> bool:
    if not pre_controller_policy_active(task, use_controller=use_controller):
        return False
    if state.enumerate_seen:
        return False
    return int(state.list_inventory_streak) >= LIST_INVENTORY_BUDGET


def should_terminalize_post_enumerate_list_inventory_spin(
    task: Mapping[str, Any],
    state: PreControllerGuardState,
    *,
    use_controller: bool,
) -> bool:
    if not pre_controller_policy_active(task, use_controller=use_controller):
        return False
    if not state.enumerate_seen or state.run_capability_ok_seen:
        return False
    return int(state.post_enumerate_list_inventory_streak) >= POST_ENUMERATE_LIST_INVENTORY_BUDGET


def maybe_terminalize_list_inventory_pre_controller_spin(
    env: Any,
    task: Mapping[str, Any],
    state: PreControllerGuardState,
    *,
    use_controller: bool,
    step_n: int,
) -> bool:
    if should_terminalize_list_inventory_spin(task, state, use_controller=use_controller):
        from hazardweaver.hwa.experiments.ablation_manual_pilot_terminal_v1 import (
            force_ablation_terminal_pre_controller_spin,
        )

        return force_ablation_terminal_pre_controller_spin(
            env,
            task,
            {
                "error": "list_inventory_spin",
                "exit_reason": "list_inventory_spin",
                "list_inventory_streak": int(state.list_inventory_streak),
                "step": int(step_n),
            },
            emit_source="unified_pre_controller_list_inventory_spin_v1",
        )
    if should_terminalize_post_enumerate_list_inventory_spin(
        task, state, use_controller=use_controller
    ):
        from hazardweaver.hwa.experiments.ablation_manual_pilot_terminal_v1 import (
            force_ablation_terminal_pre_controller_spin,
        )

        return force_ablation_terminal_pre_controller_spin(
            env,
            task,
            {
                "error": "post_enumerate_list_inventory_spin",
                "exit_reason": "post_enumerate_list_inventory_spin",
                "post_enumerate_list_inventory_streak": int(
                    state.post_enumerate_list_inventory_streak
                ),
                "step": int(step_n),
            },
            emit_source="unified_post_enumerate_list_inventory_spin_v1",
        )
    return False
