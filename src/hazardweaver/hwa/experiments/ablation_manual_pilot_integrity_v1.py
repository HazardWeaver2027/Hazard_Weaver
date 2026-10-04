"""Fail-closed submission integrity for manual ablation pilot cells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from hazardweaver.hwa.experiments.ablation_manual_pilot_v1 import ablation_manual_pilot_enabled


def _expected_action(task: Mapping[str, Any]) -> str:
    meta = task.get("metadata") or {}
    return str(meta.get("expected_action") or task.get("expected_action") or "solve").strip().lower()


def _load_answer_record(workdir: Path) -> Dict[str, Any]:
    path = workdir / "answer.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _ablation_terminal_execution_record(workdir: Path, task: Mapping[str, Any]) -> bool:
    """Terminal invalid-DCA after commit+run_capability is a sealed ablation episode."""
    if not ablation_manual_pilot_enabled(task):
        return False
    ans = _load_answer_record(workdir)
    if str(ans.get("terminal_policy") or "") != "ablation_manual_pilot_invalid_dca_v1":
        return False
    body = ans.get("answer") or {}
    if str(body.get("action") or "") != "solve":
        return False
    if not all(str(body.get(k) or "").strip() for k in ("route_id", "execution_id", "final_artifact_id")):
        return False
    from hazardweaver.hwa.experiments.agent_strict_v2 import validate_strict_tool_chain
    from hazardweaver.hwa.runtime.trajectory_ledger import load_ledger_steps

    transcript: list[Dict[str, Any]] = []
    tpath = workdir / "transcript.jsonl"
    if tpath.is_file():
        transcript = [
            json.loads(line)
            for line in tpath.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    chain_errors = validate_strict_tool_chain(
        transcript,
        ledger_steps=load_ledger_steps(workdir),
        require_submit=False,
    )
    blocking = ("missing_controller_commit_route", "missing_run_capability_ok")
    return not any(err in blocking for err in chain_errors)


def _terminal_action(workdir: Path) -> str:
    path = workdir / "answer.json"
    if not path.is_file():
        return ""
    try:
        from hazardweaver.hwa.runtime.terminal_action import terminal_from_answer_record

        return str(terminal_from_answer_record(json.loads(path.read_text(encoding="utf-8"))) or "")
    except (OSError, json.JSONDecodeError):
        return ""


def validate_ablation_pilot_submission(
    workdir: Path,
    task: Mapping[str, Any],
    *,
    submitted: bool,
) -> Dict[str, Any]:
    """Return integrity envelope; invalid → caller must clear submitted."""
    workdir = Path(workdir)
    if not ablation_manual_pilot_enabled(task):
        return {"valid": True, "status": "SKIPPED", "errors": []}

    errors: list[str] = []
    expected = _expected_action(task)
    terminal = _terminal_action(workdir)

    if submitted and expected == "solve":
        if terminal not in {"solve"}:
            errors.append(f"ablation_solve_required_terminal:{terminal or 'missing'}")
        from hazardweaver.hwa.experiments.agent_strict_v2 import (
            agent_strict_v2_enabled,
            validate_strict_submission_integrity,
        )

        if agent_strict_v2_enabled():
            if _ablation_terminal_execution_record(workdir, task):
                return {
                    "valid": True,
                    "status": "VALID_ABLATION_TERMINAL_EXECUTION",
                    "errors": [],
                    "terminal_policy": "ablation_manual_pilot_invalid_dca_v1",
                }
            from hazardweaver.hwa.runtime.trajectory_ledger import load_ledger_steps

            transcript: list[Dict[str, Any]] = []
            tpath = workdir / "transcript.jsonl"
            if tpath.is_file():
                transcript = [
                    json.loads(line)
                    for line in tpath.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            chain = validate_strict_submission_integrity(
                transcript,
                ledger_steps=load_ledger_steps(workdir),
                submitted=submitted,
                require_submit=True,
            )
            if not chain.get("valid"):
                errors.extend(list(chain.get("errors") or []))

    if errors:
        return {"valid": False, "status": "INVALID_SUBMISSION", "errors": errors}
    return {"valid": True, "status": "VALID_SUBMISSION", "errors": []}


def finalize_ablation_pilot_run(
    workdir: Path,
    task: Mapping[str, Any],
    *,
    submitted: bool,
    exit_reason: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Apply ablation pilot fail-closed rules to loop/runner outcomes."""
    integrity = validate_ablation_pilot_submission(workdir, task, submitted=submitted)
    if submitted and not integrity.get("valid"):
        return False, "ablation_integrity_fail", integrity
    return bool(submitted), str(exit_reason or ""), integrity
