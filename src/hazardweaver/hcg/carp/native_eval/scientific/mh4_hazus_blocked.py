"""CAP-MH4-R04 — Hazus honest runtime blocker (DL-052)."""

from __future__ import annotations

from typing import Any, Dict

from hazardweaver.hcg.carp.native_eval.scientific.mh4_eval_common import publish_scientific_metrics
from hazardweaver.hcg.carp.scientific.mh4_data import TASKPACK

CAPABILITY_ID = "CAP-MH4-R04"
BLOCKER = "RUNTIME_BLOCKED_WINDOWS_ARCGIS"
DATA_SOURCE = "hazus_hurricane_official"


def run_cap_mh4_r04() -> Dict[str, Any]:
    metrics = {
        "ok": False,
        "blocked": True,
        "blocker": BLOCKER,
        "taskpack_id": TASKPACK,
        "capability_id": CAPABILITY_ID,
        "route_family_id": "RF-MH4-ENGINEERING-COMBINED",
        "anchor_id": "AN-MH4-HAZUS-HURRICANE",
        "validation_tier": "blocked",
        "data_source": DATA_SOURCE,
        "metric_name": "log_mae",
        "note": "Hazus Hurricane requires Windows+ArcGIS Pro; anchor retained, no local reimplementation",
    }
    return publish_scientific_metrics(
        CAPABILITY_ID,
        metrics,
        route_family="RF-MH4-ENGINEERING-COMBINED",
        data_source=DATA_SOURCE,
    )
