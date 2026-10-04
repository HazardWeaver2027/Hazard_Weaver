"""VCE VERIFY — delegates to ``build_hwb_submission`` (batch eval path)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwa.runtime.submission_prepare_v1 import ensure_execution_submission_ready


def vce_track_metric_preview(
    *,
    workdir: Path,
    route_id: str,
    execution_id: str,
    final_artifact_id: str,
    inventory_row: Optional[Mapping[str, Any]] = None,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return ensure_execution_submission_ready(
        workdir=Path(workdir),
        route_id=str(route_id).strip(),
        execution_id=str(execution_id).strip(),
        final_artifact_id=str(final_artifact_id).strip(),
        inventory_row=inventory_row,
        taskpack=taskpack,
    )
