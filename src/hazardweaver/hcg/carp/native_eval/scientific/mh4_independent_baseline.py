"""CAP-MH4-R02 — independent marginal wind+surge sum weak baseline (DL-052)."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from hazardweaver.hcg.carp.scientific.mh4_common import (
    eligible_event_ids,
    independent_sum_usd_2023,
    reference_loss_usd_2023,
)
from hazardweaver.hcg.carp.native_eval.scientific.mh4_eval_common import publish_scientific_metrics
from hazardweaver.hcg.carp.scientific.mh4_data import TASKPACK
from hazardweaver.hcg.carp.scientific.mh4_reference import REFERENCE_LOSS_SOURCE_ID, load_event_reference, reference_manifest
from hazardweaver.hwb.evaluators.mh4_tc_building_loss import score_event_loss

CAPABILITY_ID = "CAP-MH4-R02"
ROUTE_FAMILY = "RF-MH4-INDEPENDENT"
DATA_SOURCE = "ncei_stormevents_property_v1"


def _score_split(split: str) -> Dict[str, Any]:
    per_event: List[Dict[str, Any]] = []
    scores: List[float] = []
    for nhc_id in eligible_event_ids(split):
        ref = load_event_reference(nhc_id)
        if not ref:
            continue
        pred = independent_sum_usd_2023(ref)
        truth = reference_loss_usd_2023(ref)
        scored = score_event_loss(pred, truth)
        scores.append(float(scored["log_mae"]))
        per_event.append(
            {
                "nhc_id": nhc_id,
                "storm_name": ref.get("storm_name"),
                "prediction_usd": pred,
                "reference_usd": truth,
                "log_mae": scored["log_mae"],
            }
        )
    mean_log_mae = float(sum(scores) / len(scores)) if scores else math.inf
    return {"split": split, "n_events": len(per_event), "mean_log_mae": mean_log_mae, "per_event": per_event}


def run_cap_mh4_r02() -> Dict[str, Any]:
    if not reference_manifest():
        payload = {
            "ok": False,
            "blocked": True,
            "capability_id": CAPABILITY_ID,
            "reason": "reference package missing — run materialize_mh4_scientific.py",
        }
        return publish_scientific_metrics(CAPABILITY_ID, payload, route_family=ROUTE_FAMILY, data_source=DATA_SOURCE)

    per_split = {
        "official_test": _score_split("official_test"),
        "hwb_holdout": _score_split("hwb_holdout"),
    }
    all_scores = [e["log_mae"] for row in per_split.values() for e in row["per_event"]]
    mean_log_mae = float(sum(all_scores) / len(all_scores)) if all_scores else math.inf
    metrics = {
        "ok": len(all_scores) > 0,
        "blocked": len(all_scores) == 0,
        "taskpack_id": TASKPACK,
        "capability_id": CAPABILITY_ID,
        "route_family_id": ROUTE_FAMILY,
        "validation_tier": "scientific",
        "synthetic_only": False,
        "data_source": DATA_SOURCE,
        "reference_loss_source_id": REFERENCE_LOSS_SOURCE_ID,
        "metric_name": "log_mae",
        "score": mean_log_mae,
        "higher_is_better": False,
        "per_split": per_split,
        "note": "Weak baseline: NCEI wind-marginal + surge-marginal sum vs total NCEI property (USD 2023)",
    }
    return publish_scientific_metrics(
        CAPABILITY_ID,
        metrics,
        route_family=ROUTE_FAMILY,
        data_source=DATA_SOURCE,
        extra={"reference_loss_source_id": REFERENCE_LOSS_SOURCE_ID},
    )


def run_all_scientific() -> Dict[str, Any]:
    return {CAPABILITY_ID: run_cap_mh4_r02()}
