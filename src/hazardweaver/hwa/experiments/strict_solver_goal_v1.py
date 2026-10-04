"""Agent-Strict v2: non-gold instance bindings surfaced in user_facing_goal.

Instance identity (which case / record / scenario to execute) is **not** the answer.
Gold scores, witness routes, and grader-only fields stay hidden. Under pure-agent
strict mode the LLM must still route, commit, call run_capability with explicit
handles, and submit — we only remove the guesswork of *which* inventory row applies.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def resolve_strict_solver_execution_hints(
    inventory_row: Mapping[str, Any],
    *,
    taskpack_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Map any headline inventory row → explicit run_capability handle hints."""
    inv = dict(inventory_row)
    instance_id = str(inv.get("instance_id") or "").strip()
    track = str(inv.get("track") or "").strip()

    from hazardweaver.hwa.experiments.headline_scenario_binding_v1 import resolve_headline_scenario_context

    hints = dict(resolve_headline_scenario_context(inventory_row=inv))

    if instance_id.startswith("atlas:"):
        parts = instance_id.split(":", 2)
        if len(parts) == 3 and parts[2]:
            hints.setdefault("scenario_id", parts[2].strip())

    if instance_id.startswith("variant:") or instance_id.startswith("param:"):
        suffix = instance_id.rsplit(":", 1)[-1].strip()
        if suffix:
            hints.setdefault("scenario_id", suffix)

    scenario_id = str(
        hints.get("scenario_id") or inv.get("scenario_id") or ""
    ).strip()
    if (not scenario_id or scenario_id == "CAP-PLACEHOLDER") and taskpack_params:
        fb_sid = str(taskpack_params.get("scenario_id") or "").strip()
        if fb_sid and fb_sid != "CAP-PLACEHOLDER":
            scenario_id = fb_sid
    if (not hints.get("split")) and taskpack_params:
        fb_split = str(taskpack_params.get("split") or "").strip()
        if fb_split:
            hints["split"] = fb_split
    if scenario_id and scenario_id != "CAP-PLACEHOLDER":
        hints["scenario_id"] = scenario_id

    # PFDF headline portfolio (MH-1 atlas + water-fire edges): record_id is the case key.
    if instance_id.startswith("atlas:MH-1:"):
        rec = instance_id.split("atlas:MH-1:", 1)[1].strip()
        if rec:
            hints["record_id"] = rec
            hints.setdefault("scenario_id", rec)
    elif track == "MH-1" and scenario_id:
        hints.setdefault("record_id", scenario_id)

    out: Dict[str, str] = {}
    for key in ("record_id", "scenario_id", "split"):
        val = str(hints.get(key) or "").strip()
        if val and val != "CAP-PLACEHOLDER":
            out[key] = val
    return out


def augment_user_facing_goal_for_strict(
    raw_goal: str,
    hints: Mapping[str, str],
) -> str:
    """Append standardized execution-handle hints (visible to solver, not grader gold)."""
    goal = str(raw_goal or "").strip()
    if not hints:
        return goal
    parts: list[str] = []
    if hints.get("record_id"):
        parts.append(f"record_id={hints['record_id']}")
    sid = str(hints.get("scenario_id") or "").strip()
    if sid and sid != hints.get("record_id"):
        parts.append(f"scenario_id={sid}")
    if hints.get("split"):
        parts.append(f"split={hints['split']}")
    if not parts:
        return goal
    binding = ", ".join(parts)
    suffix = (
        f"Target execution handles (pass verbatim to run_capability after commit): {binding}."
    )
    if suffix in goal:
        return goal
    return f"{goal} {suffix}" if goal else suffix
