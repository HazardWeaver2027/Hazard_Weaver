"""Auto submit_solution after strict chained commit+execute (avoid abstain spin)."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional


def _parse_commit_payload(content: Any) -> Dict[str, Any]:
    if isinstance(content, Mapping):
        payload = dict(content)
    else:
        try:
            payload = json.loads(str(content or "{}"))
        except (TypeError, json.JSONDecodeError):
            return {}
    if not isinstance(payload, Mapping):
        return {}
    result = payload.get("result")
    if isinstance(result, Mapping):
        return dict(result)
    return dict(payload)


def _submit_args_from_commit(result: Mapping[str, Any]) -> Dict[str, str]:
    exec_block = result.get("execution") if isinstance(result.get("execution"), Mapping) else {}
    args = result.get("submit_solution_args")
    if not isinstance(args, Mapping) and isinstance(exec_block, Mapping):
        args = exec_block.get("submit_solution_args")
    args = dict(args) if isinstance(args, Mapping) else {}
    rid = str(args.get("route_id") or result.get("route_id") or "").strip()
    eid = str(
        args.get("execution_id")
        or result.get("execution_id")
        or exec_block.get("execution_id")
        or ""
    ).strip()
    aid = str(
        args.get("final_artifact_id")
        or result.get("final_artifact_id")
        or exec_block.get("final_artifact_id")
        or ""
    ).strip()
    return {"route_id": rid, "execution_id": eid, "final_artifact_id": aid}


def try_finish_strict_chained_commit(env: Any, task: Mapping[str, Any], commit_content: Any) -> bool:
    """When chained commit returns execution + submit_solution_args, terminal-submit once."""
    from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_chained_commit_enabled

    if not agent_strict_v2_chained_commit_enabled():
        return False
    if getattr(env, "submitted", False):
        return False
    result = _parse_commit_payload(commit_content)
    if not result.get("ok"):
        return False
    exec_block = result.get("execution") if isinstance(result.get("execution"), Mapping) else {}
    executed_ok = bool(exec_block.get("ok")) or bool(result.get("execution_id"))
    if not executed_ok:
        return False
    if str(result.get("next_step") or "") not in {"submit_solution", ""}:
        return False
    handles = _submit_args_from_commit(result)
    if not all(handles.values()):
        return False
    from pathlib import Path

    from hazardweaver.hwa.experiments.unified_gold_registry_submit_v1 import (
        _inventory_row,
        prefer_unified_gold_registry_handles,
    )

    wd = Path(getattr(env, "workdir", "."))
    inv = _inventory_row(wd, task)
    gold_handles = prefer_unified_gold_registry_handles(wd, inv)
    if gold_handles:
        handles = gold_handles
    submit_result = env.execute(
        {
            "name": "submit_solution",
            "arguments": {
                **handles,
                "rationale": "strict_chained_autosubmit",
            },
        }
    )
    ok = bool(getattr(env, "submitted", False))
    if not ok and isinstance(submit_result, Mapping):
        ok = bool(submit_result.get("ok"))
    if not ok:
        from hazardweaver.hwa.experiments.ablation_manual_pilot_terminal_v1 import maybe_terminalize_failed_submit

        ok = maybe_terminalize_failed_submit(env, task, submit_result or {}, handles)
    if ok:
        env.append_transcript(
            {
                "role": "system",
                "event": "strict_chained_autosubmit",
                "route_id": handles["route_id"],
                "execution_id": handles["execution_id"],
                "final_artifact_id": handles["final_artifact_id"],
            }
        )
    return ok
