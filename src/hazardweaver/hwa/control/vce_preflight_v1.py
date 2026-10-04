"""Build HWB-evaluable solve submission for VCE VERIFY — same as batch eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwb.bridge.hwa_workdir import build_hwb_submission
from hazardweaver.hwb.bridge.submission_metric_v1 import normalize_submission_value


def build_vce_preflight_submission(
    *,
    workdir: Path,
    route_id: str,
    execution_id: str,
    final_artifact_id: str,
    inventory_row: Optional[Mapping[str, Any]] = None,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cap = str((inventory_row or {}).get("capability_id") or "").strip()
    body: Dict[str, Any] = {
        "action": "solve",
        "execution_id": str(execution_id).strip(),
        "final_artifact_id": str(final_artifact_id).strip(),
        "route_id": str(route_id).strip(),
    }
    if cap:
        body["capability_id"] = cap
    return build_hwb_submission(
        workdir,
        answer_body=body,
        taskpack=taskpack,
        route_id=str(route_id).strip(),
    )


_normalize_submission_metric_name = normalize_submission_value

__all__ = ["build_vce_preflight_submission", "_normalize_submission_metric_name"]
