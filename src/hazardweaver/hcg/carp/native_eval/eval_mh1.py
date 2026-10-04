"""MH-1 native eval — ocelote/pfdf A2 replay + chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.expansion.dev_replay import (
    average_precision,
    brier_score,
    load_dev_manifest,
    volume_mae,
)
from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic, write_blocked

G2_CAPS = {"CAP-MH1-04", "CAP-MH1-05"}


def _load_mh1():
    base, meta = load_dev_manifest("MH-1")
    burned = np.load(base / meta["files"]["burned_area_frac"])
    soil = np.load(base / meta["files"]["soil_moisture"])
    slope = np.load(base / meta["files"]["slope_deg"])
    rainfall = np.load(base / meta["files"]["rainfall_mm"])
    likelihood = np.load(base / meta["files"]["pfdf_likelihood"])
    log_volume = np.load(base / meta["files"]["log_volume"])
    occurrence = np.load(base / meta["files"]["occurrence_label"])
    return burned, soil, slope, rainfall, likelihood, log_volume, occurrence


def eval_mh1_cap(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    try:
        from hazardweaver.hcg.carp.scientific.mh1_data import scientific_data_ready
        from hazardweaver.hcg.carp.native_eval.scientific.mh1_ocelote import eval_mh1_scientific

        if scientific_data_ready():
            return eval_mh1_scientific(capability_id, out_base=out_base)
    except ImportError:
        pass

    if capability_id in G2_CAPS:
        from hazardweaver.hcg.carp.native_eval.train_mh1 import train_cap

        return train_cap(capability_id, out_base=out_base)
    try:
        burned, soil, slope, rainfall, likelihood, log_volume, occurrence = _load_mh1()
    except FileNotFoundError as exc:
        return write_blocked("MH-1", capability_id, str(exc), out_base=out_base)

    if capability_id == "CAP-MH1-01":
        score = brier_score(occurrence, likelihood)
        metric_name = "brier"
        family = "RF-EMPIRICAL-LOGISTIC-PFDF-INIT"
        note = "USGS Staley likelihood dev replay"
        invert = True
    elif capability_id == "CAP-MH1-02":
        pred_vol = np.log1p(np.exp(log_volume) * likelihood)
        score = volume_mae(pred_vol, log_volume)
        metric_name = "volume_mae"
        family = "RF-EMPIRICAL-VOLUME-REGRESSION"
        note = "USGS potential-volume dev replay"
        invert = False
    elif capability_id == "CAP-MH1-03":
        threshold = float(np.percentile(rainfall, 75))
        pred = (rainfall >= threshold).astype(np.int8)
        score = float(np.mean(pred == occurrence))
        metric_name = "threshold_accuracy"
        family = "RF-INTENSITY-DURATION-THRESHOLD"
        note = "Rainfall threshold alert dev replay"
        invert = False
    else:
        return eval_blocked_generic("MH-1", capability_id, "no native eval wired", out_base=out_base)

    metric_value = 1.0 - score if invert else score
    metrics = {
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "raw_score": float(score),
        "synthetic_only": False,
        "data_source": "ocelote_pfdf_dev_inventory_v1",
        "note": note,
    }
    write_metrics("MH-1", capability_id, metrics, base=out_base)
    write_replay_manifest(
        "MH-1",
        capability_id,
        family_id=family,
        exec_ok=True,
        metric_name=metric_name,
        metric_value=float(metric_value),
        notes=note,
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}
