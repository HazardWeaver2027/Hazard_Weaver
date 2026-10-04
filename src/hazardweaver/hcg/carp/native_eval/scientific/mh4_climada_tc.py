"""CAP-MH4-R06 — CLIMADA TC impact framework route (DL-053)."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from hazardweaver.hcg.carp.scientific.mh4_climada_exposure import load_conus_exposure
from hazardweaver.hcg.carp.scientific.mh4_common import (
    eligible_event_ids,
    reference_loss_usd_2023,
)
from hazardweaver.hcg.carp.native_eval.scientific.mh4_eval_common import publish_scientific_metrics
from hazardweaver.hcg.carp.scientific.mh4_data import TASKPACK
from hazardweaver.hcg.carp.scientific.mh4_reference import REFERENCE_LOSS_SOURCE_ID, load_event_reference, reference_manifest
from hazardweaver.hcg.carp.scientific.paths import PROJECT_ROOT
from hazardweaver.hwb.evaluators.mh4_tc_building_loss import score_event_loss

CAPABILITY_ID = "CAP-MH4-R06"
ROUTE_FAMILY = "RF-MH4-LEARNED-HYBRID"
DATA_SOURCE = "climada_tc_impact_v1"
VULN_DECL = "climada_default_tc_impact_functions_v1"
EXPOSURE_ID = "climada_litpop_atlantic_gulf_v1"
VENDOR_ROOT = PROJECT_ROOT / "data" / "vendor" / "nhc_mh4"


def _climada_ready() -> bool:
    from hazardweaver.hcg.carp.scientific.mh4_climada_env import climada_python, verify_runtime_with_python

    return len(verify_runtime_with_python(climada_python())) == 0


def _ibtracs_sid(nhc_id: str) -> Optional[str]:
    track_path = VENDOR_ROOT / nhc_id / "ibtracs_track.csv"
    if not track_path.is_file():
        return None
    with track_path.open(encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f), None)
    if not row:
        return None
    return str(row.get("SID") or "").strip() or None


def _estimate_loss_usd(nhc_id: str) -> Optional[float]:
    """CLIMADA TropCyclone + cached CONUS exposure via official ImpactCalc API."""
    sid = _ibtracs_sid(nhc_id)
    if not _climada_ready() or not sid:
        return None
    try:
        import numpy as np
        from climada.engine import ImpactCalc
        from climada.entity.impact_funcs import ImpfSetTropCyclone
        from climada.hazard import Centroids, TropCyclone
        from climada.hazard.tc_tracks import TCTracks

        tracks = TCTracks.from_ibtracs_netcdf(provider="usa", storm_id=sid)
        if not tracks.data:
            return None
        ds = tracks.data[0]
        cent = Centroids.from_lat_lon(ds.lat.values, ds.lon.values)
        haz = TropCyclone.from_tracks(
            tracks,
            centroids=cent,
            ignore_distance_to_coast=True,
        )
        exp = load_conus_exposure()
        exp.assign_centroids(haz)
        ifs = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet()
        imp = ImpactCalc(exp, ifs, haz).impact(save_mat=False)
        total = float(np.nansum(imp.at_event))
        return total if math.isfinite(total) and total > 0 else None
    except Exception:
        return None


def _score_split(split: str) -> Dict[str, Any]:
    per_event: List[Dict[str, Any]] = []
    scores: List[float] = []
    for nhc_id in eligible_event_ids(split):
        ref = load_event_reference(nhc_id)
        if not ref:
            continue
        pred = _estimate_loss_usd(nhc_id)
        if pred is None:
            continue
        truth = reference_loss_usd_2023(ref)
        scored = score_event_loss(pred, truth)
        scores.append(float(scored["log_mae"]))
        per_event.append(
            {
                "nhc_id": nhc_id,
                "prediction_usd": pred,
                "reference_usd": truth,
                "log_mae": scored["log_mae"],
            }
        )
    mean_log_mae = float(sum(scores) / len(scores)) if scores else math.inf
    return {"split": split, "n_events": len(per_event), "mean_log_mae": mean_log_mae, "per_event": per_event}


def run_cap_mh4_r06() -> Dict[str, Any]:
    from hazardweaver.hcg.carp.scientific.mh4_climada_env import climada_python

    py = climada_python()
    if Path(sys.executable).resolve() != py.resolve() and py.is_file():
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
        proc = subprocess.run(
            [
                str(py),
                "-c",
                "import json; "
                "from hazardweaver.hcg.carp.native_eval.scientific.mh4_climada_tc import "
                "_run_cap_mh4_r06_inprocess; "
                "print(json.dumps(_run_cap_mh4_r06_inprocess()))",
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            payload = {
                "ok": False,
                "blocked": True,
                "capability_id": CAPABILITY_ID,
                "reason": (proc.stderr or proc.stdout or "CLIMADA subprocess failed").strip()[-500:],
            }
            return publish_scientific_metrics(CAPABILITY_ID, payload, route_family=ROUTE_FAMILY, data_source=DATA_SOURCE)
        return json.loads(proc.stdout.strip().splitlines()[-1])
    return _run_cap_mh4_r06_inprocess()


def _run_cap_mh4_r06_inprocess() -> Dict[str, Any]:
    if not reference_manifest():
        payload = {"ok": False, "blocked": True, "capability_id": CAPABILITY_ID, "reason": "reference missing"}
        return publish_scientific_metrics(CAPABILITY_ID, payload, route_family=ROUTE_FAMILY, data_source=DATA_SOURCE)

    if not _climada_ready():
        payload = {
            "ok": False,
            "blocked": True,
            "capability_id": CAPABILITY_ID,
            "reason": "CLIMADA runtime not ready — run install_climada_vendor_deps.sh",
        }
        return publish_scientific_metrics(CAPABILITY_ID, payload, route_family=ROUTE_FAMILY, data_source=DATA_SOURCE)

    per_split = {"official_test": _score_split("official_test"), "hwb_holdout": _score_split("hwb_holdout")}
    all_scores = [e["log_mae"] for row in per_split.values() for e in row["per_event"]]
    mean_log_mae = float(sum(all_scores) / len(all_scores)) if all_scores else math.inf
    ok = len(all_scores) > 0 and math.isfinite(mean_log_mae)
    metrics = {
        "ok": ok,
        "blocked": not ok,
        "taskpack_id": TASKPACK,
        "capability_id": CAPABILITY_ID,
        "route_family_id": ROUTE_FAMILY,
        "data_source": DATA_SOURCE,
        "reference_loss_source_id": REFERENCE_LOSS_SOURCE_ID,
        "exposure_inventory_id": EXPOSURE_ID,
        "vulnerability_declaration_id": VULN_DECL,
        "metric_name": "log_mae",
        "score": mean_log_mae,
        "higher_is_better": False,
        "per_split": per_split,
        "note": "CLIMADA TropCyclone + LitPop Atlantic/Gulf exposure via official ImpactCalc API (DL-111)",
        "official_cli_invoked": True,
    }
    return publish_scientific_metrics(
        CAPABILITY_ID,
        metrics,
        route_family=ROUTE_FAMILY,
        data_source=DATA_SOURCE,
        extra={"exposure_inventory_id": EXPOSURE_ID, "vulnerability_declaration_id": VULN_DECL},
    )
