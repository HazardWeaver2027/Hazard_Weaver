"""DR-OUT g2_train caps (ordinal + subseasonal ML)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.expansion.dev_replay import load_dev_manifest, ordinal_hit_rate
from hazardweaver.hcg.carp.native_eval.blocked import write_blocked

CAP_FAMILIES = {
    "CAP-DROUT-04": "RF-ORDINAL-STATISTICAL-TRANSITI",
    "CAP-DROUT-05": "RF-SUBSEASONAL-ML-FORECAST",
}


def train_cap(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if capability_id not in CAP_FAMILIES:
        return write_blocked("DR-OUT", capability_id, "unknown g2 cap", out_base=out_base)
    try:
        from hazardweaver.hcg.carp.scientific.drout_data import scientific_data_ready
        from hazardweaver.hcg.carp.native_eval.scientific.drout_cpc_sdo import eval_drout_scientific

        if scientific_data_ready():
            return eval_drout_scientific(capability_id, out_base=out_base)
    except ImportError:
        pass
    try:
        sdo, usdm, objective = _load_arrays()
    except FileNotFoundError as exc:
        return write_blocked("DR-OUT", capability_id, str(exc), out_base=out_base)

    if capability_id == "CAP-DROUT-04":
        pred = np.clip(usdm[:-1] + np.sign(objective[1:] - objective[:-1]), 0, 4).astype(np.int8)
        score = ordinal_hit_rate(pred, sdo[1:])
        metric_name = "ordinal_hit_rate"
    else:
        features = np.stack([usdm.ravel(), objective.ravel()], axis=1)
        coef, _, _, _ = np.linalg.lstsq(features, sdo.ravel(), rcond=None)
        pred = np.clip(np.round(features @ coef), 0, 4).astype(np.int8)
        score = ordinal_hit_rate(pred, sdo.ravel())
        metric_name = "ml_mapper_skill"

    metrics = {
        "metric_name": metric_name,
        "metric_value": float(score),
        "synthetic_only": False,
        "data_source": "cpc_sdo_dev_archive_v1",
        "note": "g2_train on PI-approved dev months",
    }
    write_metrics("DR-OUT", capability_id, metrics, base=out_base)
    write_replay_manifest(
        "DR-OUT",
        capability_id,
        family_id=CAP_FAMILIES[capability_id],
        exec_ok=True,
        metric_name=metric_name,
        metric_value=float(score),
        notes="DR-OUT g2_train dev replay",
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}


def _load_arrays():
    base, meta = load_dev_manifest("DR-OUT")
    sdo = np.load(base / meta["files"]["cpc_sdo_category"])
    usdm = np.load(base / meta["files"]["usdm_category"])
    objective = np.load(base / meta["files"]["cpc_objective_tendency"])
    return sdo, usdm, objective


def train_all() -> None:
    for cap in CAP_FAMILIES:
        train_cap(cap)
