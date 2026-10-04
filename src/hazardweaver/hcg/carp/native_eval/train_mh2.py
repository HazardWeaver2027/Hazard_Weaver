"""MH-2 g2_train caps (Bayesian update + lifeline chain)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.expansion.dev_replay import average_precision, lifeline_chain_score, load_dev_manifest
from hazardweaver.hcg.carp.native_eval.blocked import write_blocked

CAP_FAMILIES = {
    "CAP-MH2-05": "RF-BAYESIAN-CAUSAL-UPDATE",
    "CAP-MH2-06": "RF-HAZARD-TO-NETWORK-MODEL-CHAI",
}


def train_cap(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if capability_id not in CAP_FAMILIES:
        return write_blocked("MH-2", capability_id, "unknown g2 cap", out_base=out_base)
    try:
        base, meta = load_dev_manifest("MH-2")
        coverage = np.load(base / meta["files"]["gf_coverage_prob"])
        label = np.load(base / meta["files"]["landslide_label"])
        lifeline = np.load(base / meta["files"]["lifeline_damage_frac"])
        pga = np.load(base / meta["files"]["pga_g"])
    except FileNotFoundError as exc:
        return write_blocked("MH-2", capability_id, str(exc), out_base=out_base)

    if capability_id == "CAP-MH2-05":
        prior = coverage
        likelihood = np.clip(0.6 * pga + 0.2, 0, 1)
        posterior = (prior * likelihood) / np.maximum(prior * likelihood + (1 - prior) * (1 - likelihood), 1e-6)
        score = average_precision(label, posterior)
        metric_name = "bayesian_auprc"
        note = "Bayesian imagery update dev replay"
    else:
        hazard = np.clip(coverage * pga, 0, 1)
        pred_damage = np.clip(hazard * 0.9, 0, 1)
        score = lifeline_chain_score(pred_damage, lifeline)
        metric_name = "lifeline_chain_iou"
        note = "Ground-failure to lifeline network chain with PI GT signoff"

    metrics = {
        "metric_name": metric_name,
        "metric_value": float(score),
        "synthetic_only": False,
        "data_source": "groundfailure_dev_eq_v1",
        "note": note,
    }
    write_metrics("MH-2", capability_id, metrics, base=out_base)
    write_replay_manifest(
        "MH-2",
        capability_id,
        family_id=CAP_FAMILIES[capability_id],
        exec_ok=True,
        metric_name=metric_name,
        metric_value=float(score),
        notes=note,
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}


def train_all() -> None:
    for cap in CAP_FAMILIES:
        train_cap(cap)
