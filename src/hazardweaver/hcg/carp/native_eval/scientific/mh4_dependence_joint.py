"""CAP-MH4-R03 — dependence-aware joint loss from NCEI marginals (DL-052)."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from hazardweaver.hcg.carp.scientific.mh4_common import (
    eligible_event_ids,
    independent_sum_usd_2023,
    marginal_surge_usd_2023,
    marginal_wind_usd_2023,
    reference_loss_usd_2023,
)
from hazardweaver.hcg.carp.native_eval.scientific.mh4_eval_common import publish_scientific_metrics
from hazardweaver.hcg.carp.scientific.mh4_data import TASKPACK
from hazardweaver.hcg.carp.scientific.mh4_reference import (
    REFERENCE_LOSS_SOURCE_ID,
    load_event_reference,
    reference_manifest,
)
from hazardweaver.hwb.evaluators.mh4_tc_building_loss import score_event_loss

CAPABILITY_ID = "CAP-MH4-R03"
ROUTE_FAMILY = "RF-MH4-DEPENDENCE"
DATA_SOURCE = "ncei_marginal_copula_v1"
# Fixed weak dependence amplification (auditable; not fit on test labels).
LAMBDA_DEP = 0.35


def _dependence_prediction(ref: Dict[str, Any]) -> float:
    wind = marginal_wind_usd_2023(ref)
    surge = marginal_surge_usd_2023(ref)
    base = independent_sum_usd_2023(ref)
    if wind <= 0 or surge <= 0:
        return base
    amplification = LAMBDA_DEP * math.sqrt(wind * surge)
    return base + amplification


def _score_split(split: str) -> Dict[str, Any]:
    per_event: List[Dict[str, Any]] = []
    scores: List[float] = []
    for nhc_id in eligible_event_ids(split):
        ref = load_event_reference(nhc_id)
        if not ref:
            continue
        pred = _dependence_prediction(ref)
        truth = reference_loss_usd_2023(ref)
        scored = score_event_loss(pred, truth)
        scores.append(float(scored["log_mae"]))
        per_event.append(
            {
                "nhc_id": nhc_id,
                "prediction_usd": pred,
                "reference_usd": truth,
                "log_mae": scored["log_mae"],
                "lambda_dep": LAMBDA_DEP,
            }
        )
    mean_log_mae = float(sum(scores) / len(scores)) if scores else math.inf
    return {"split": split, "n_events": len(per_event), "mean_log_mae": mean_log_mae, "per_event": per_event}


def run_cap_mh4_r03() -> Dict[str, Any]:
    if not reference_manifest():
        payload = {
            "ok": False,
            "blocked": True,
            "capability_id": CAPABILITY_ID,
            "reason": "reference package missing",
        }
        return publish_scientific_metrics(CAPABILITY_ID, payload, route_family=ROUTE_FAMILY, data_source=DATA_SOURCE)

    per_split = {
        "official_test": _score_split("official_test"),
        "hwb_holdout": _score_split("hwb_holdout"),
    }
    all_scores = []
    for row in per_split.values():
        all_scores.extend(e["log_mae"] for e in row["per_event"])
    mean_log_mae = float(sum(all_scores) / len(all_scores)) if all_scores else math.inf

    metrics = {
        "ok": len(all_scores) > 0,
        "blocked": len(all_scores) == 0,
        "taskpack_id": TASKPACK,
        "capability_id": CAPABILITY_ID,
        "route_family_id": ROUTE_FAMILY,
        "data_source": DATA_SOURCE,
        "reference_loss_source_id": REFERENCE_LOSS_SOURCE_ID,
        "metric_name": "log_mae",
        "score": mean_log_mae,
        "higher_is_better": False,
        "lambda_dep": LAMBDA_DEP,
        "per_split": per_split,
        "note": "Dependence-aware: independent_sum + lambda*sqrt(wind*surge); NCEI marginals only",
    }
    return publish_scientific_metrics(
        CAPABILITY_ID,
        metrics,
        route_family=ROUTE_FAMILY,
        data_source=DATA_SOURCE,
        extra={"lambda_dep": LAMBDA_DEP},
    )
