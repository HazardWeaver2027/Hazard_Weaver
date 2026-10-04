"""Shared helpers for MH-4 native eval writers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir

TASKPACK = "MH-4"


def write_recipe_log(capability_id: str, *, command: str, returncode: int = 0, note: str = "") -> Path:
    cap_dir = cap_scientific_dir(TASKPACK, capability_id)
    cap_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "taskpack_id": TASKPACK,
        "capability_id": capability_id,
        "official_command": command,
        "returncode": returncode,
        "status": "executed" if returncode == 0 else "failed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }
    path = cap_dir / "recipe_run.log"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def publish_scientific_metrics(
    capability_id: str,
    metrics: Dict[str, Any],
    *,
    route_family: str,
    data_source: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_root = SCIENTIFIC_RUNS_ROOT
    metrics = dict(metrics)
    blocked = bool(metrics.get("blocked"))
    if blocked:
        metrics["validation_tier"] = "blocked"
    else:
        metrics.setdefault("validation_tier", "scientific")
    score = metrics.get("score")
    if score is not None and metrics.get("metric_value") is None:
        metrics["metric_value"] = score
    metrics.setdefault("synthetic_only", False)
    write_metrics(TASKPACK, capability_id, metrics, base=out_root)
    write_replay_manifest(
        TASKPACK,
        capability_id,
        family_id=route_family,
        exec_ok=bool(metrics.get("ok")) and not metrics.get("blocked"),
        metric_name=str(metrics.get("metric_name") or ""),
        metric_value=metrics.get("score"),
        protocol="scientific-native",
        notes=str(metrics.get("note") or ""),
        extra=extra or {},
        base=out_root,
    )
    write_recipe_log(
        capability_id,
        command=f"mh4_scientific.{capability_id}",
        note=str(metrics.get("note") or ""),
    )
    return metrics
