"""MH-2 native eval — groundfailure + Newmark/Bayesian + lifeline chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.expansion.dev_replay import average_precision, lifeline_chain_score, load_dev_manifest
from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic, write_blocked
from hazardweaver.hcg.carp.scientific.mh2_data import scientific_data_ready

G2_CAPS = {"CAP-MH2-05", "CAP-MH2-06"}
GF_CAPS = {
    "CAP-MH2-01": ("RF-EMPIRICAL-AREAL-COVERAGE-LOG", "nowicki_jessee_2018"),
    "CAP-MH2-02": ("RF-EMPIRICAL-CELL-OCCURRENCE-LO", "nowicki_2014"),
    "CAP-MH2-03": ("RF-EMPIRICAL-AREAL-COVERAGE-MOD", "godt_2008"),
}


def _load_mh2():
    base, meta = load_dev_manifest("MH-2")
    pga = np.load(base / meta["files"]["pga_g"])
    slope = np.load(base / meta["files"]["slope_deg"])
    precip = np.load(base / meta["files"]["antecedent_precip_mm"])
    coverage = np.load(base / meta["files"]["gf_coverage_prob"])
    label = np.load(base / meta["files"]["landslide_label"])
    lifeline = np.load(base / meta["files"]["lifeline_damage_frac"])
    return pga, slope, precip, coverage, label, lifeline


def eval_mh2_cap(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if scientific_data_ready():
        from hazardweaver.hcg.carp.native_eval.scientific.mh2_groundfailure import eval_mh2_scientific

        return eval_mh2_scientific(capability_id, out_base=out_base)

    if capability_id in G2_CAPS:
        from hazardweaver.hcg.carp.native_eval.train_mh2 import train_cap

        return train_cap(capability_id, out_base=out_base)
    try:
        pga, slope, precip, coverage, label, lifeline = _load_mh2()
    except FileNotFoundError as exc:
        return write_blocked("MH-2", capability_id, str(exc), out_base=out_base)

    if capability_id in GF_CAPS:
        family_id, model = GF_CAPS[capability_id]
        if model == "nowicki_2014":
            prob = np.clip(0.5 * coverage + 0.1 * (pga > 0.2), 0, 1)
        elif model == "godt_2008":
            prob = np.clip(0.35 * coverage + 0.15 * precip / 100.0, 0, 1)
        else:
            prob = coverage
        score = average_precision(label, prob)
        metric_name = "gf_auprc"
        note = f"groundfailure {model} dev replay"
    elif capability_id == "CAP-MH2-04":
        displacement = np.clip(0.8 * pga**2 / np.maximum(slope, 1.0), 0, 1)
        score = average_precision(label, displacement)
        metric_name = "newmark_auprc"
        family_id = "RF-MECHANISTIC-SLOPE-DISPLACEME"
        note = "Newmark displacement dev replay"
    else:
        return eval_blocked_generic("MH-2", capability_id, "no native eval wired", out_base=out_base)

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
        family_id=family_id,
        exec_ok=True,
        metric_name=metric_name,
        metric_value=float(score),
        notes=note,
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}
