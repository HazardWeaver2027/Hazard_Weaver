"""CAP-MH4-R01 — NHC/IBTrACS hazard substrate integrity (input only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from hazardweaver.hcg.carp.native_eval.scientific.mh4_eval_common import publish_scientific_metrics
from hazardweaver.hcg.carp.scientific.mh4_common import eligible_event_ids
from hazardweaver.hcg.carp.scientific.mh4_data import TASKPACK, nhc_substrate_ready
from hazardweaver.hcg.carp.scientific.paths import PROJECT_ROOT

CAPABILITY_ID = "CAP-MH4-R01"
ROUTE_FAMILY = None
DATA_SOURCE = "nhc_products_v1"
VENDOR_ROOT = PROJECT_ROOT / "data" / "vendor" / "nhc_mh4"


def run_cap_mh4_r01() -> Dict[str, Any]:
    if not nhc_substrate_ready():
        payload = {
            "ok": False,
            "blocked": True,
            "capability_id": CAPABILITY_ID,
            "reason": "run materialize_mh4_nhc_substrate.py",
        }
        return publish_scientific_metrics(
            CAPABILITY_ID,
            payload,
            route_family="substrate",
            data_source=DATA_SOURCE,
        )

    per_event: List[Dict[str, Any]] = []
    missing = []
    for split in ("official_test", "hwb_holdout"):
        for nhc_id in eligible_event_ids(split):
            pin_path = VENDOR_ROOT / nhc_id / "FETCH_PIN.json"
            track_path = VENDOR_ROOT / nhc_id / "ibtracs_track.csv"
            if not pin_path.is_file() or not track_path.is_file():
                missing.append(nhc_id)
                continue
            pin = json.loads(pin_path.read_text(encoding="utf-8"))
            per_event.append(
                {
                    "nhc_id": nhc_id,
                    "split": split,
                    "n_points": pin.get("n_points"),
                    "sha256": pin.get("sha256"),
                }
            )

    ok = len(missing) == 0 and len(per_event) > 0
    metrics = {
        "ok": ok,
        "blocked": not ok,
        "taskpack_id": TASKPACK,
        "capability_id": CAPABILITY_ID,
        "role": "hazard_substrate_input_only",
        "headline_eligible": False,
        "data_source": DATA_SOURCE,
        "metric_name": "substrate_coverage",
        "score": float(len(per_event)),
        "higher_is_better": True,
        "n_events": len(per_event),
        "missing": missing,
        "per_event": per_event,
        "note": "REJ-MH4-01: hazard substrate integrity only, not loss endpoint",
    }
    return publish_scientific_metrics(
        CAPABILITY_ID,
        metrics,
        route_family="substrate",
        data_source=DATA_SOURCE,
    )
