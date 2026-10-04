"""Rebuild native HWA_TRAJECTORY_v2 steps from tool_calls.jsonl audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def read_tool_calls(workdir: Path) -> List[Dict[str, Any]]:
    path = Path(workdir) / "tool_calls.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _unwrap_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(result.get("result"), Mapping):
        return dict(result["result"])
    return dict(result)


from hazardweaver.hwa.runtime.terminal_action import normalize_terminal_action, terminal_from_answer_record


def _terminal_action_from_answer(workdir: Path) -> str:
    path = Path(workdir) / "answer.json"
    if not path.is_file():
        return ""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return terminal_from_answer_record(raw)


def steps_from_tool_calls(
    workdir: Path,
    *,
    rows: Optional[List[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Convert tool_calls audit rows into native ledger-compatible v2 steps."""
    tool_rows = list(rows) if rows is not None else read_tool_calls(workdir)
    steps: List[Dict[str, Any]] = []
    for i, row in enumerate(tool_rows):
        name = str(row.get("name") or "")
        ok = bool(row.get("ok", True))
        args = row.get("arguments") or {}
        result = _unwrap_result(row.get("result") or {})
        if name == "run_capability":
            steps.append(
                {
                    "step_index": i,
                    "kind": "tool_execution",
                    "tool": "run_capability",
                    "ok": ok,
                    "capability_id": str(
                        args.get("capability_id") or result.get("capability_id") or ""
                    ),
                    "execution_id": result.get("execution_id"),
                    "lease_id": result.get("lease_id"),
                    "route_id": result.get("route_id"),
                    "execution_certificate": result.get("execution_certificate"),
                    "reachability_certificate": result.get("reachability_certificate"),
                    "execution_event": result.get("execution_event"),
                    "final_artifact_id": result.get("final_artifact_id"),
                    "error": row.get("error"),
                    "source": "tool_calls_fallback",
                }
            )
        elif name in {"submit_solution", "submit", "submit_abstention", "submit_clarification"}:
            action = {
                "submit_solution": "solve",
                "submit_abstention": "abstain",
                "submit_clarification": "clarify",
                "submit": _terminal_action_from_answer(workdir) or "solve",
            }.get(name, "solve")
            steps.append(
                {
                    "step_index": i,
                    "kind": "submit",
                    "ok": ok,
                    "action": action,
                    "route_id": args.get("route_id") or result.get("route_id"),
                    "execution_id": args.get("execution_id") or result.get("execution_id"),
                    "final_artifact_id": args.get("final_artifact_id")
                    or result.get("final_artifact_id"),
                    "source": "tool_calls_fallback",
                }
            )
        elif name:
            cap = str(
                args.get("capability_id")
                or result.get("capability_id")
                or ""
            ).strip()
            if name == "run_capability" and cap:
                steps.append(
                    {
                        "step_index": i,
                        "kind": "tool_execution",
                        "tool": name,
                        "ok": ok,
                        "capability_id": cap,
                        "execution_id": result.get("execution_id"),
                        "lease_id": result.get("lease_id"),
                        "route_id": result.get("route_id"),
                        "execution_certificate": result.get("execution_certificate"),
                        "reachability_certificate": result.get("reachability_certificate"),
                        "execution_event": result.get("execution_event"),
                        "final_artifact_id": result.get("final_artifact_id"),
                        "error": row.get("error"),
                        "source": "tool_calls_fallback",
                    }
                )
            else:
                steps.append(
                    {
                        "step_index": i,
                        "kind": "agent_tool",
                        "tool": name,
                        "ok": ok,
                        "error": row.get("error"),
                        "source": "tool_calls_fallback",
                    }
                )
    return steps


def stamp_final_artifact_scenario(
    final_artifact: Optional[Mapping[str, Any]],
    *,
    scenario_id: str,
) -> Optional[Dict[str, Any]]:
    """Ensure FL-2 parametric submissions carry sealed scenario_id for HWB E_q."""
    if not final_artifact or not scenario_id:
        return dict(final_artifact) if final_artifact else None
    fa = dict(final_artifact)
    value = dict(fa.get("value") or {})
    value.setdefault("scenario_id", scenario_id)
    value.setdefault("metric_name", "rmse_depth")
    fa["value"] = value
    return fa
