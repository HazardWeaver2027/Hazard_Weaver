"""Unified agent-native eval gold — MH-1 / TC-TRK / WF-3 (DL-213).

Agent benchmark execution dispatches ``eval_*_cap`` native replay paths. Grader
``reference_view`` MUST bind to the same dispatch outputs, not stale parametric
catalog / scientific ``native_metrics`` that disagree with agent replay (cf. DL-207 HW-MED).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

MH1_CAPS = tuple(f"CAP-MH1-{i:02d}" for i in range(1, 6))
WF3_CAPS = tuple(f"CAP-WF3-{i:02d}" for i in range(1, 7))
TCTRK_CAPS = (
    "CAP-TCTRK-01",
    "CAP-TCTRK-02",
    "CAP-TCTRK-03",
    "CAP-TCTRK-04",
    "CAP-TCTRK-05",
    "CAP-TCTRK-06",
)

TRACK_NATIVE_CAPS: Dict[str, tuple[str, ...]] = {
    "MH-1": MH1_CAPS,
    "TC-TRK": TCTRK_CAPS,
    "WF-3": WF3_CAPS,
}

NATIVE_EVAL_AUTHORITY = "hcg_native_eval_cap_v1"


def unified_agent_native_eval_authority(row: Optional[Mapping[str, Any]] = None) -> bool:
    """True when unified grading must bind to agent ``eval_*_cap`` replay (DL-213)."""
    flag = str(os.environ.get("HWA_UNIFIED_AGENT_NATIVE_EVAL", "")).strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    inv = dict(row or {})
    track = str(inv.get("track") or "").upper()
    if track in TRACK_NATIVE_CAPS and inv.get("unified_benchmark_v1"):
        return True
    return bool(inv.get("hwb_headline_inventory"))


def _cap_norm(capability_id: str) -> str:
    return str(capability_id or "").strip().split("__", 1)[0]


def grading_metric_name_for_capability(
    track: str,
    capability_id: str,
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Authoritative grader metric name — always from native eval dispatch output."""
    cap = _cap_norm(capability_id)
    track_u = str(track or "").upper()
    allowed = TRACK_NATIVE_CAPS.get(track_u) or ()
    if cap not in allowed:
        return None
    try:
        metrics = _native_eval_metrics(track_u, cap, inventory_row=inventory_row)
    except KeyError:
        return None
    return str(metrics.get("metric_name") or "metric_value")


def expected_metric_name_for_capability(
    track: str,
    capability_id: str,
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Alias for grading metric (DL-214: no hardcoded dispatch labels)."""
    return grading_metric_name_for_capability(
        track,
        capability_id,
        inventory_row=inventory_row,
    )


def _native_eval_metrics(
    track: str,
    capability_id: str,
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    track_u = str(track or "").upper()
    cap = _cap_norm(capability_id)
    if track_u == "MH-1":
        from hazardweaver.hcg.carp.native_eval.eval_mh1 import eval_mh1_cap

        out = eval_mh1_cap(cap)
    elif track_u == "TC-TRK":
        from hazardweaver.hcg.carp.native_eval.eval_tc_tctrk import eval_tctrk_cap

        input_artifacts: Optional[Dict[str, Any]] = None
        if inventory_row:
            sid = str(inventory_row.get("scenario_id") or "").strip()
            if sid:
                input_artifacts = {"scenario_id": sid, "storm_sid": sid}
        out = eval_tctrk_cap(cap, input_artifacts=input_artifacts)
    elif track_u == "WF-3":
        from hazardweaver.hcg.carp.native_eval.eval_wf3_wsts import eval_wf3_cap

        out = eval_wf3_cap(cap)
    else:
        raise KeyError(f"unsupported track for native eval ref: {track_u}")
    if not out.get("ok"):
        raise KeyError(f"native eval failed for {track_u}/{cap}: {out}")
    metrics = dict(out.get("metrics") or {})
    if metrics.get("metric_value") is None:
        raise KeyError(f"native eval missing metric_value for {track_u}/{cap}")
    return metrics


def native_eval_reference_score(
    track: str,
    capability_id: str,
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
) -> float:
    metrics = _native_eval_metrics(
        str(track).upper(),
        capability_id,
        inventory_row=inventory_row,
    )
    return float(metrics["metric_value"])


def native_eval_scenario_ref_for_capability(
    track: str,
    capability_id: str,
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Sidecar-shaped scenario ref for one CAP (agent native eval authority)."""
    track_u = str(track or "").upper()
    cap = _cap_norm(capability_id)
    allowed = TRACK_NATIVE_CAPS.get(track_u) or ()
    if cap not in allowed:
        raise KeyError(f"unknown {track_u} capability for native eval ref: {cap}")
    metrics = _native_eval_metrics(track_u, cap, inventory_row=inventory_row)
    metric_name = str(
        metrics.get("metric_name")
        or grading_metric_name_for_capability(track_u, cap, inventory_row=inventory_row)
        or "metric_value"
    )
    score = float(metrics["metric_value"])
    return {
        "capability_id": cap,
        "outputs": {
            "metric_name": metric_name,
            "reference_score": score,
            "scenario_id": cap,
            metric_name: score,
        },
        "tolerance": {
            "metric": metric_name,
            "max_abs_error": 1e-6,
        },
        "provenance": {
            "scenario_id": cap,
            "unified_native_eval_authority": NATIVE_EVAL_AUTHORITY,
            "not_parametric_catalog_only": True,
        },
    }


def native_score_matches_execution(
    track: str,
    capability_id: str,
    executed_score: float,
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
    epsilon: float = 1e-5,
) -> bool:
    try:
        expected = native_eval_reference_score(
            track,
            capability_id,
            inventory_row=inventory_row,
        )
    except KeyError:
        return False
    return abs(float(executed_score) - float(expected)) <= float(epsilon)
