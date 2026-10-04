"""Classify headline capability execution failures for agent observations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def classify_execution_failure(
    error: Optional[str] = None,
    *,
    payload: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return failure_class for run_capability / commit execution blocks."""
    err = str(error or "").strip().lower()
    if payload and payload.get("failure_class"):
        return str(payload["failure_class"])
    if not err and payload:
        err = str(payload.get("error") or "").strip().lower()

    if any(x in err for x in ("blocked", "task_mismatch", "not_headline", "probe")):
        return "infra_blocked"
    if any(
        x in err
        for x in (
            "lease",
            "pre_commit",
            "pi_adm",
            "controller_token",
            "scoped_execution",
        )
    ):
        return "lease_or_gate"
    if any(
        x in err
        for x in (
            "dispatch",
            "eval",
            "metric",
            "contract",
            "checkpoint",
            "scientific",
            "handler",
        )
    ):
        return "scientific_eval_failed"
    if err:
        return "unknown"
    return "unknown"


def attach_failure_class(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload.get("ok", True) and "failure_class" not in payload:
        payload = dict(payload)
        payload["failure_class"] = classify_execution_failure(
            str(payload.get("error") or ""),
            payload=payload,
        )
    return payload
