"""DCA is only defined on completed terminal episodes (not loop auto-emits)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_INCOMPLETE_EXITS = frozenset({"limit_wall", "limit_steps", "exception", "timeout", "max_wall_s"})
_INTEGRITY_FAIL_EXITS = frozenset({"ablation_integrity_fail"})


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def is_dca_eligible(workdir: Path) -> bool:
    """True only when the agent produced a real terminal episode worth scoring."""
    meta = _read_json(workdir / "run_meta.json") or {}
    ans = _read_json(workdir / "answer.json") or {}
    if not ans:
        return False
    if str(ans.get("emit_source") or "") == "agent_loop_unterminated":
        return False
    exit_reason = str(meta.get("exit_reason") or "")
    if not meta.get("submitted"):
        if exit_reason in _INCOMPLETE_EXITS or exit_reason in _INTEGRITY_FAIL_EXITS:
            return False
        sub_int = meta.get("submission_integrity") or {}
        if sub_int.get("valid") is False:
            return False
        action = str((ans.get("answer") or {}).get("action") or "")
        if action == "clarify" and "unterminated_after_" in str((ans.get("answer") or {}).get("question") or ""):
            return False
    return True


def incomplete_episode_dca_dict(
    *,
    taskpack_id: str = "",
    exit_reason: str = "",
) -> Dict[str, Any]:
    return {
        "taskpack_id": taskpack_id,
        "agent_id": "hazardweaver",
        "dca_score": 0.0,
        "counted": False,
        "contract_violated": False,
        "budget_within_limit": True,
        "metric_name": "incomplete_episode",
        "raw_score": None,
        "outcome": "unknown",
        "reason_code": "incomplete_episode",
        "difficulty_tier": "L1",
        "valid": False,
        "exit_reason": exit_reason,
    }
