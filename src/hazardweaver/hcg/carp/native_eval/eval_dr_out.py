"""DR-OUT native eval — CPC SDO vendor_fetch + persistence/ordinal A2 replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.expansion.dev_replay import load_dev_manifest, ordinal_hit_rate
from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic, write_blocked

G2_CAPS = {"CAP-DROUT-04", "CAP-DROUT-05"}


def _load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base, meta = load_dev_manifest("DR-OUT")
    sdo = np.load(base / meta["files"]["cpc_sdo_category"])
    usdm = np.load(base / meta["files"]["usdm_category"])
    objective = np.load(base / meta["files"]["cpc_objective_tendency"])
    return sdo, usdm, objective


def _emit(
    capability_id: str,
    family_id: str,
    metric_name: str,
    metric_value: float,
    *,
    out_base: Optional[Path],
    note: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = {
        "metric_name": metric_name,
        "metric_value": metric_value,
        "synthetic_only": False,
        "data_source": "cpc_sdo_dev_archive_v1",
        "note": note,
        **(extra or {}),
    }
    write_metrics("DR-OUT", capability_id, metrics, base=out_base)
    write_replay_manifest(
        "DR-OUT",
        capability_id,
        family_id=family_id,
        exec_ok=True,
        metric_name=metric_name,
        metric_value=metric_value,
        notes=note,
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}


def eval_dr_out_cap(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    try:
        from hazardweaver.hcg.carp.scientific.drout_data import scientific_data_ready
        from hazardweaver.hcg.carp.native_eval.scientific.drout_cpc_sdo import eval_drout_scientific

        if scientific_data_ready():
            return eval_drout_scientific(capability_id, out_base=out_base)
    except ImportError:
        pass

    if capability_id in G2_CAPS:
        from hazardweaver.hcg.carp.native_eval.train_dr_out import train_cap

        return train_cap(capability_id, out_base=out_base)
    try:
        sdo, usdm, objective = _load_arrays()
    except FileNotFoundError as exc:
        return write_blocked("DR-OUT", capability_id, str(exc), out_base=out_base)

    if capability_id == "CAP-DROUT-01":
        score = ordinal_hit_rate(sdo[1:], sdo[:-1])
        return _emit(
            capability_id,
            "RF-EXPERT-OPERATIONAL-SYNTHESIS",
            "sdo_skill",
            score,
            out_base=out_base,
            note="CPC SDO archive ingest dev replay",
        )
    if capability_id == "CAP-DROUT-02":
        persistence = usdm[:-1]
        score = ordinal_hit_rate(persistence, sdo[1:])
        return _emit(
            capability_id,
            "RF-PERSISTENCE-BASELINE",
            "persistence_skill",
            score,
            out_base=out_base,
            note="USDM persistence baseline on dev months",
        )
    if capability_id == "CAP-DROUT-03":
        score = ordinal_hit_rate(objective, sdo)
        return _emit(
            capability_id,
            "RF-OBJECTIVE-STATISTICAL-TENDEN",
            "objective_skill",
            score,
            out_base=out_base,
            note="CPC objective tendency GeoTIFF dev replay",
        )
    return eval_blocked_generic("DR-OUT", capability_id, "no native eval wired", out_base=out_base)
