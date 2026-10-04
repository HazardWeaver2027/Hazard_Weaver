"""Per-storm TC-TRK atlas gold (persistence lead-error on TCBench-small IBTrACS)."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

_STORM_SID_RE = re.compile(r"^\d{4}\d{3}[NS]\d{5}$")


def is_tctrk_storm_scenario(scenario_id: str) -> bool:
    return bool(_STORM_SID_RE.match(str(scenario_id or "").strip()))


def tctrk_storm_scenario_ref(storm_sid: str) -> Dict[str, Any]:
    from hazardweaver.hcg.carp.native_eval.eval_tc_tctrk import eval_tctrk_storm_persistence

    out = eval_tctrk_storm_persistence(str(storm_sid).strip())
    if not out.get("ok"):
        raise ValueError(f"tctrk_storm_gold_unavailable:{storm_sid}")
    metrics = out.get("metrics") or {}
    score = float(metrics["lead_error_km"])
    sid = str(storm_sid).strip()
    return {
        "storm_sid": sid,
        "scenario_id": sid,
        "outputs": {
            "metric_name": "lead_error_km",
            "reference_score": score,
            "lead_error_km": score,
            "scenario_id": sid,
        },
        "tolerance": {
            "metric": "lead_error_km",
            "max_abs_error": 50.0,
        },
        "provenance": {
            "claim_tier": "tctrk_atlas_headline_storm",
            "gold_source": "tcbench_small_persistence_per_storm",
            "storm_sid": sid,
        },
    }


def try_tctrk_storm_scenario_ref(scenario_id: str) -> Optional[Dict[str, Any]]:
    if not is_tctrk_storm_scenario(scenario_id):
        return None
    try:
        return tctrk_storm_scenario_ref(scenario_id)
    except (KeyError, ValueError, TypeError):
        return None
