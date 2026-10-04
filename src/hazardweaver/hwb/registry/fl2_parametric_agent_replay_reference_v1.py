"""FL-2 parametric (G2 neural) agent replay gold — per-scenario rmse_depth (DL-231)."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

FL2_PARAMETRIC_TASKPACK_ID = "hwb_fl2_parametric_v1"
FL2_G2_CAPS = frozenset({"CAP-FL2-01", "CAP-FL2-02", "CAP-FL2-03"})
FL2_REPLAY_AUTHORITY = "fl2_parametric_per_scenario_replay_v1"


def _cap_norm(capability_id: str) -> str:
    return str(capability_id or "").strip().split("__", 1)[0]


def fl2_parametric_agent_replay_eval_authority(row: Optional[Mapping[str, Any]] = None) -> bool:
    flag = str(os.environ.get("HWA_FL2_PARAMETRIC_AGENT_REPLAY", "")).strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    inv = dict(row or {})
    if str(inv.get("taskpack_id") or "") != FL2_PARAMETRIC_TASKPACK_ID:
        return False
    if str(inv.get("track") or "").upper() != "FL-2":
        return False
    return bool(
        inv.get("unified_benchmark_v1")
        or inv.get("hwb_headline_inventory")
        or inv.get("ablation_manual_pilot")
    )


def fl2_parametric_replay_scenario_ref_for_capability(
    scenario_id: str,
    capability_id: str,
    *,
    split: str = "official_test",
    inventory_row: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Sidecar-shaped ref bound to executed G2 cap at inventory scenario."""
    cap = _cap_norm(capability_id)
    sid = str(scenario_id or "").strip()
    if cap not in FL2_G2_CAPS:
        raise KeyError(f"not an FL-2 parametric G2 cap: {cap}")
    if not sid:
        raise KeyError("scenario_id required for FL-2 parametric replay ref")

    from hazardweaver.hcg.carp.native_eval.eval_fl2 import eval_fl2_cap_for_scenario
    from hazardweaver.hwb.registry.unified_benchmark_tolerance_v1 import calibrate_tolerance

    out = eval_fl2_cap_for_scenario(cap, sid, split=str(split or "official_test"))
    if not out.get("ok"):
        raise KeyError(f"FL-2 parametric replay failed for {cap}/{sid}: {out}")
    metrics = dict(out.get("metrics") or {})
    rmse_val = float(metrics.get("metric_value") or metrics.get("rmse_depth") or 0.0)
    csi_val = float(metrics.get("csi_0.01") or 0.0)
    tol = calibrate_tolerance("rmse_depth", rmse_val)
    return {
        "capability_id": cap,
        "outputs": {
            "metric_name": "rmse_depth",
            "reference_score": rmse_val,
            "scenario_id": sid,
            "rmse_depth": rmse_val,
            "csi_0.01": csi_val,
        },
        "tolerance": tol,
        "provenance": {
            "scenario_id": sid,
            "split": split,
            "unified_native_eval_authority": FL2_REPLAY_AUTHORITY,
            "not_dead_parametric_catalog_only": True,
        },
    }
