"""Backfill final_artifact scores from **real execution outputs only** (no native replay)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwa.agent_runtime.execution_schema import (
    _extract_submission_metrics,
    _value_or_uri_has_score,
    load_execution,
)
from hazardweaver.hwa.runtime.trajectory_ledger import load_ledger_steps


def _ledger_produced_artifacts(workdir: Path, execution_id: str) -> Optional[Dict[str, Any]]:
    for step in reversed(load_ledger_steps(workdir)):
        if str(step.get("execution_id") or "") != str(execution_id).strip():
            continue
        ev = step.get("execution_event")
        if isinstance(ev, Mapping):
            nested = ev.get("produced_artifacts")
            if isinstance(nested, Mapping):
                return dict(nested)
        break
    return None


def _execution_blocked(produced: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(produced, Mapping):
        return False
    output = produced.get("output")
    if not isinstance(output, Mapping):
        return False
    if output.get("blocked") is True:
        return True
    if output.get("ok") is False and output.get("reason"):
        return True
    return False


def backfill_final_artifact_from_execution(
    workdir: Path,
    *,
    execution_id: str,
    final_artifact_id: str,
    capability_id: str = "",
    inventory_row: Optional[Mapping[str, Any]] = None,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Patch ``final_artifact.value_or_uri`` only from ledger ``produced_artifacts``."""
    workdir = Path(workdir)
    er = load_execution(workdir, execution_id)
    if er is None:
        return {
            "ok": False,
            "error": "unknown_execution_id",
            "verify_layer": "execution_score_backfill",
            "execution_id": execution_id,
        }

    fa = er.get("final_artifact") or {}
    if _value_or_uri_has_score(fa.get("value_or_uri")):
        return {
            "ok": True,
            "verify_layer": "execution_score_backfill",
            "skipped": True,
            "reason": "already_has_score",
        }

    produced = _ledger_produced_artifacts(workdir, execution_id)
    if _execution_blocked(produced):
        reason = ""
        if isinstance(produced, Mapping):
            out = produced.get("output") or {}
            if isinstance(out, Mapping):
                reason = str(out.get("reason") or "execution_blocked")
        return {
            "ok": False,
            "error": "execution_blocked_no_score",
            "verify_layer": "execution_score_backfill",
            "detail": reason,
            "capability_id": capability_id,
        }

    metrics = None
    if isinstance(produced, Mapping):
        metrics = _extract_submission_metrics({"produced_artifacts": produced})
        if not metrics:
            metrics = _extract_submission_metrics(produced)

    if not metrics:
        return {
            "ok": False,
            "error": "final_artifact_missing_score",
            "verify_layer": "execution_score_backfill",
            "capability_id": capability_id,
            "detail": "no_produced_artifact_metrics",
        }

    cap = capability_id or str((er.get("executed_capability_ids") or [""])[0] or "")
    value = {**dict(metrics), "capability_id": cap}
    from hazardweaver.hwa.runtime.track_native_metrics import _patch_workdir_metrics

    _patch_workdir_metrics(
        workdir,
        execution_id=str(execution_id).strip(),
        final_artifact_id=str(final_artifact_id).strip(),
        capability_id=cap,
        value=value,
        provenance_extra={"dispatch": "execution_produced_artifacts"},
    )

    er2 = load_execution(workdir, execution_id)
    fa2 = (er2 or {}).get("final_artifact") or {}
    if not _value_or_uri_has_score(fa2.get("value_or_uri")):
        return {
            "ok": False,
            "error": "final_artifact_missing_score",
            "verify_layer": "execution_score_backfill",
            "capability_id": cap,
        }
    return {
        "ok": True,
        "verify_layer": "execution_score_backfill",
        "capability_id": cap,
        "metric_name": value.get("metric_name"),
    }
