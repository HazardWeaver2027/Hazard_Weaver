"""HW-MED agent replay gold — EWB eval_subset canonical MAE (DL-207).

Unified / agent benchmark execution uses ``eval_hwmed_cap(..., force_eval_subset_replay=True)``
(see ``HWA_HWMED_AGENT_REPLAY``). Grader reference_view MUST use the same dispatch
outputs, not CIRA scientific ``native_metrics`` from ``runs/carp/scientific/``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional

HW_MED_TASKPACK_ID = "hwb_hw_med_parametric_v1"
HW_MED_METRIC = "max_mae"
HW_MED_REPLAY_AUTHORITY = "ewb_eval_subset_replay_v1"


def hwmed_agent_replay_eval_authority(row: Optional[Mapping[str, Any]] = None) -> bool:
    """True when HWA/HWB grading must bind to eval_subset replay (not scientific native)."""
    flag = str(os.environ.get("HWA_HWMED_AGENT_REPLAY", "")).strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    inv = dict(row or {})
    if inv.get("unified_benchmark_v1"):
        return True
    try:
        from hazardweaver.hwa.experiments.ablation_manual_pilot_v1 import ablation_hwmed_agent_replay_enabled

        if ablation_hwmed_agent_replay_enabled({"metadata": inv}):
            return True
    except ImportError:
        pass
    return bool(inv.get("hwb_headline_inventory"))


@lru_cache(maxsize=1)
def hwmed_eval_subset_replay_mae_by_capability() -> Dict[str, float]:
    """Canonical replay MAE per CAP-HWMED-* — same code path as agent ``run_capability``."""
    from hazardweaver.hcg.carp.native_eval.eval_hw_med import VENDOR_CAPS, eval_hwmed_cap

    out: Dict[str, float] = {}
    for cap in sorted(VENDOR_CAPS):
        result = eval_hwmed_cap(str(cap), force_eval_subset_replay=True)
        metrics = dict(result.get("metrics") or {})
        out[str(cap)] = float(metrics["metric_value"])
    return out


def hwmed_replay_scenario_ref_for_capability(capability_id: str) -> Dict[str, Any]:
    """Sidecar-shaped scenario ref for one CAP-HWMED-* (replay authority)."""
    cap = str(capability_id or "").strip().split("__", 1)[0]
    table = hwmed_eval_subset_replay_mae_by_capability()
    if cap not in table:
        raise KeyError(f"unknown HW-MED capability for replay ref: {cap}")
    mae = float(table[cap])
    return {
        "capability_id": cap,
        "outputs": {
            "metric_name": HW_MED_METRIC,
            "reference_score": mae,
            "scenario_id": cap,
            HW_MED_METRIC: mae,
        },
        "tolerance": {
            "metric": HW_MED_METRIC,
            "max_abs_error": 1e-6,
        },
        "provenance": {
            "scenario_id": cap,
            "hwmed_eval_authority": HW_MED_REPLAY_AUTHORITY,
            "not_scientific_native_metrics": True,
        },
    }


def replay_mae_matches_execution(capability_id: str, executed_mae: float, *, epsilon: float = 1e-5) -> bool:
    cap = str(capability_id or "").strip().split("__", 1)[0]
    expected = hwmed_eval_subset_replay_mae_by_capability().get(cap)
    if expected is None:
        return False
    return abs(float(executed_mae) - float(expected)) <= float(epsilon)
