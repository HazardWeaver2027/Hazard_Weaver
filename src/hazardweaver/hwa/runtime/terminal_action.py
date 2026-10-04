"""Canonical HWA terminal actions for HWB dual_gate (solve / abstain / clarify)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

VALID_TERMINAL_ACTIONS = frozenset({"solve", "abstain", "clarify"})

_TERMINAL_ALIASES = {
    "submit_abstention": "abstain",
    "submit_solution": "solve",
    "submit_clarification": "clarify",
    "execute_route": "solve",
}


def normalize_terminal_action(raw: Any) -> str:
    """Map tool names / legacy aliases to HWB terminal_action."""
    action = str(raw or "").strip().lower()
    if not action:
        return ""
    mapped = _TERMINAL_ALIASES.get(action, action)
    return mapped if mapped in VALID_TERMINAL_ACTIONS else mapped


def _infer_terminal_from_payload(body: Mapping[str, Any]) -> str:
    """Infer HWB terminal_action from answer body when action is omitted."""
    raw_action = str(body.get("action") or "").strip().lower()
    if raw_action in {"submit_clarification", "partial_failure"} or body.get("status") == "partial_failure":
        return "clarify"
    if raw_action in {"submit_abstention"} or body.get("reason_code"):
        return "abstain"
    log_volume = body.get("log_volume")
    if log_volume is not None:
        try:
            if math.isfinite(float(log_volume)):
                return "solve"
        except (TypeError, ValueError):
            pass
    return ""


def terminal_from_answer_record(answer: Mapping[str, Any]) -> str:
    """Best-effort terminal from answer.json envelope or body."""
    body = answer.get("answer")
    if isinstance(body, Mapping):
        terminal = normalize_terminal_action(body.get("action"))
        if terminal in VALID_TERMINAL_ACTIONS:
            return terminal
        inferred = _infer_terminal_from_payload(body)
        if inferred in VALID_TERMINAL_ACTIONS:
            return inferred
    terminal = normalize_terminal_action(answer.get("action"))
    if terminal in VALID_TERMINAL_ACTIONS:
        return terminal
    if isinstance(body, Mapping):
        inferred = _infer_terminal_from_payload(body)
        if inferred in VALID_TERMINAL_ACTIONS:
            return inferred
    return ""


def coerce_answer_body(answer: Any) -> Dict[str, Any]:
    """Normalize answer dict in-place semantics (returns new dict)."""
    if not isinstance(answer, Mapping):
        return {}
    body = dict(answer)
    raw = body.get("action")
    if raw is not None and str(raw).strip():
        body["action"] = normalize_terminal_action(raw)
    else:
        inferred = _infer_terminal_from_payload(body)
        if inferred:
            body["action"] = inferred
    return body


def audit_submitted_terminal(
    workdir: Path,
    *,
    submitted: bool = False,
) -> Dict[str, Any]:
    """
    Validate terminal_action on a completed run.

    submitted=True with empty/unknown action → invalid_terminal_answer.
    """
    workdir = Path(workdir)
    path = workdir / "answer.json"
    if not path.is_file():
        return {"ok": False, "exit_reason": "missing_answer_json", "terminal_action": ""}
    try:
        answer = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ok": False, "exit_reason": "invalid_answer_json", "terminal_action": ""}

    body = answer.get("answer")
    body_map = dict(body) if isinstance(body, Mapping) else {}
    raw_action = str(body_map.get("action") or answer.get("action") or "")
    terminal = terminal_from_answer_record(answer)

    out: Dict[str, Any] = {
        "ok": True,
        "terminal_action": terminal,
        "raw_action": raw_action,
        "exit_reason": "",
    }
    if not submitted:
        return out
    if not terminal:
        out.update(
            {
                "ok": False,
                "exit_reason": "invalid_terminal_answer_empty",
                "detail": "submitted=True but answer.action missing or unmapped",
            }
        )
        return out
    if terminal not in VALID_TERMINAL_ACTIONS:
        out.update(
            {
                "ok": False,
                "exit_reason": "invalid_terminal_action",
                "detail": f"unmapped action: {raw_action!r}",
            }
        )
    return out


def patch_answer_terminal_aliases(workdir: Path) -> Optional[str]:
    """Rewrite answer.json body.action aliases (offline backfill). Returns terminal or None."""
    workdir = Path(workdir)
    path = workdir / "answer.json"
    if not path.is_file():
        return None
    answer = json.loads(path.read_text(encoding="utf-8"))
    body = answer.get("answer")
    if not isinstance(body, Mapping):
        return None
    terminal = terminal_from_answer_record(answer)
    if not terminal:
        return None
    new_body = dict(body)
    new_body["action"] = terminal
    answer = dict(answer)
    answer["answer"] = new_body
    path.write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")
    return terminal
