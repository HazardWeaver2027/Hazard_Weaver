"""Optional pack-local plan helpers (not Phase-04 ConstrainedCompiler)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwa.agent_runtime.tool_specs import CORE_TOOL_NAMES, OPTIONAL_PLAN_TOOL_NAMES


def suggest_plan(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Propose a non-binding tool DAG from solver_visible fields."""
    if "gold" in task:
        raise ValueError("suggest_plan refused: gold present")
    sv = task.get("solver_visible") or {}
    inputs = sv.get("inputs") or {}
    sample_refs = list(inputs.get("sample_refs") or [])
    allowed = sv.get("allowed_inventory") or {}
    model_ids = list(allowed.get("model_ids") or [])
    dataset_ids = list(allowed.get("dataset_ids") or [])
    pack_refs = list(sv.get("pack_refs") or [])
    constraints = list(inputs.get("constraints") or [])

    steps: List[Dict[str, Any]] = [
        {"tool": "list_inventory", "args": {"kind": "all"}, "why": "discover HWA assets"},
    ]
    for ref in pack_refs[:4]:
        steps.append(
            {
                "tool": "read_card",
                "args": {"card_path": ref},
                "why": "read pack card for schema / pitfalls",
            }
        )
    for ref in sample_refs:
        ds = ref.get("dataset_id")
        sid = ref.get("sample_id")
        split = ref.get("split")
        args: Dict[str, Any] = {"dataset_id": ds, "sample_id": sid}
        if split:
            args["split"] = split
        steps.append({"tool": "load_sample", "args": args, "why": "load HWA sample features"})
        if model_ids and ds:
            pred_args = {
                "model_id": model_ids[0],
                "dataset_id": ds,
                "sample_id": sid,
            }
            if split:
                pred_args["split"] = split
            steps.append(
                {
                    "tool": "run_predictor",
                    "args": pred_args,
                    "why": "run first allowed predictor (agent may choose another)",
                }
            )
    steps.append(
        {
            "tool": "submit_answer",
            "args": {"answer": "<FILL>", "rationale": "<brief>", "model_id_used": model_ids[0] if model_ids else None},
            "why": "submit final answer matching success_criteria",
        }
    )
    return {
        "ok": True,
        "binding": False,
        "note": "Optional suggestion only — LLM remains the controller.",
        "allowed_dataset_ids": dataset_ids,
        "allowed_model_ids": model_ids,
        "constraints": constraints,
        "suggested_steps": steps,
    }


def validate_plan(
    plan: Any,
    task: Mapping[str, Any],
    *,
    known_tools: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Validate a proposed plan against allowed inventory / known tools."""
    if "gold" in task:
        raise ValueError("validate_plan refused: gold present")
    sv = task.get("solver_visible") or {}
    allowed = sv.get("allowed_inventory") or {}
    allowed_models = set(allowed.get("model_ids") or [])
    allowed_datasets = set(allowed.get("dataset_ids") or [])
    allowed_tools = set(allowed.get("tool_ids") or []) | set(OPTIONAL_PLAN_TOOL_NAMES)
    if known_tools:
        known = set(known_tools)
    else:
        known = set(CORE_TOOL_NAMES) | set(OPTIONAL_PLAN_TOOL_NAMES)

    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except json.JSONDecodeError as exc:
            return {"ok": False, "errors": [f"plan is not valid JSON: {exc}"]}

    steps: List[Any]
    if isinstance(plan, dict) and "steps" in plan:
        steps = list(plan["steps"] or [])
    elif isinstance(plan, dict) and "suggested_steps" in plan:
        steps = list(plan["suggested_steps"] or [])
    elif isinstance(plan, list):
        steps = plan
    else:
        return {"ok": False, "errors": ["plan must be a list of steps or {steps: [...]}"]}

    errors: List[str] = []
    warnings: List[str] = []
    saw_submit = False
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"step[{i}] must be an object")
            continue
        tool = step.get("tool") or step.get("name")
        args = step.get("args") or step.get("arguments") or {}
        if not tool:
            errors.append(f"step[{i}] missing tool name")
            continue
        if tool not in known:
            errors.append(f"step[{i}] unknown tool: {tool}")
        if tool in allowed_tools or tool in OPTIONAL_PLAN_TOOL_NAMES:
            pass
        elif allowed_tools and tool not in allowed_tools and tool not in OPTIONAL_PLAN_TOOL_NAMES:
            errors.append(f"step[{i}] tool not in allowed_inventory.tool_ids: {tool}")
        if not isinstance(args, dict):
            errors.append(f"step[{i}] args must be an object")
            continue
        if "model_id" in args and allowed_models and args["model_id"] not in allowed_models:
            errors.append(f"step[{i}] model_id not allowed: {args['model_id']}")
        if "dataset_id" in args and allowed_datasets and args["dataset_id"] not in allowed_datasets:
            errors.append(f"step[{i}] dataset_id not allowed: {args['dataset_id']}")
        if tool == "submit_answer":
            saw_submit = True
            if "answer" not in args:
                warnings.append(f"step[{i}] submit_answer missing answer")
    if not saw_submit:
        errors.append("plan must include submit_answer")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_steps": len(steps),
        "binding": False,
    }
