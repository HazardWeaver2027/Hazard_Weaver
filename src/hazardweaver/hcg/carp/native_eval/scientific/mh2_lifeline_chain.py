"""CAP-MH2-06 lifeline chain eval — scientific path (BLOCKED until segment GT packaged)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import write_blocked
from hazardweaver.hcg.carp.scientific.mh2_data import (
    TASKPACK,
    VBCI_EVENT_ID,
    event_dir,
    lifeline_gt_manifest,
    lifeline_gt_ready,
)
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir

LIFELINE_CAP = "CAP-MH2-06"
DATA_SOURCE = "groundfailure_official_test_v1"
FAMILY_ID = "RF-HAZARD-TO-NETWORK-MODEL-CHAI"


def _lifeline_blocker_reason() -> str:
    manifest = lifeline_gt_manifest() or {}
    status = manifest.get("status", "BLOCKED")
    if status != "READY":
        missing = manifest.get("missing_requirements") or []
        base = manifest.get("blocker") or "segment-level lifeline GT not packaged"
        if missing:
            return f"{base}; missing: {', '.join(missing)}"
        return str(base)
    if not manifest.get("source_license_audited"):
        return "lifeline GT license not audited (PI_REQUIRED)"
    if int(manifest.get("segment_count", 0) or 0) <= 0:
        return "lifeline segment_count is zero"
    return "lifeline GT scaffold incomplete"


def eval_lifeline_chain_scientific(
    capability_id: str = LIFELINE_CAP,
    *,
    out_base: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run lifeline chain eval when GT ready; otherwise honest BLOCKED."""
    if capability_id != LIFELINE_CAP:
        return write_blocked(TASKPACK, capability_id, "not a lifeline cap", out_base=out_base)

    if not lifeline_gt_ready():
        return write_blocked(
            TASKPACK,
            capability_id,
            _lifeline_blocker_reason(),
            out_base=out_base,
        )

    # Future: load frozen network + GF landslide posterior + segment labels → chain metric.
    return write_blocked(
        TASKPACK,
        capability_id,
        "lifeline chain replay not yet implemented (GT packaging incomplete)",
        out_base=out_base,
    )


def write_lifeline_blocked_certificate(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    """Write engineering certificate documenting honest BLOCKED state."""
    out_root = out_base or SCIENTIFIC_RUNS_ROOT
    cap_dir = cap_scientific_dir(TASKPACK, LIFELINE_CAP, runs_root=out_root)
    cap_dir.mkdir(parents=True, exist_ok=True)
    reason = _lifeline_blocker_reason()
    payload = {
        "taskpack_id": TASKPACK,
        "capability_id": LIFELINE_CAP,
        "official_command": "hwb_mh2_lifeline_chain --network frozen_v1",
        "returncode": 1,
        "status": "blocked",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "note": f"DL-050 honest BLOCKED: {reason}",
    }
    (cap_dir / "recipe_run.log").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "metric_name": "lifeline_chain_iou",
        "metric_value": None,
        "synthetic_only": False,
        "data_source": DATA_SOURCE,
        "evaluator": "mh2_lifeline_chain@blocked",
        "reference_present": False,
        "note": reason,
        "blocked": True,
    }
    write_metrics(TASKPACK, LIFELINE_CAP, metrics, base=out_root)
    write_replay_manifest(
        TASKPACK,
        LIFELINE_CAP,
        family_id=FAMILY_ID,
        exec_ok=False,
        metric_name="lifeline_chain_iou",
        metric_value=0.0,
        notes=reason,
        base=out_root,
    )
    return {"ok": False, "blocked": True, "reason": reason, "metrics": metrics}
