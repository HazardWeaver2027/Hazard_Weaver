"""FL-2 solver agent replay gold — capability-bound rmse_depth (DL-215).

Agent benchmark execution runs ``CAP-FL2-04/05/06`` official solver replays per
scenario. Grader ``reference_view`` MUST bind to the **executed** capability's
sealed prediction RMSE, not the sidecar's default HAND witness when gold is
LISFLOOD/SFINCS (cf. DL-213 MH-1 / DL-207 HW-MED).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional

FL2_SOLVER_TASKPACK_ID = "hwb_fl2_solver_parametric_v1"
FL2_SOLVER_CAPS = frozenset({"CAP-FL2-04", "CAP-FL2-05", "CAP-FL2-06"})
FL2_REPLAY_AUTHORITY = "fl2_solver_sealed_prediction_replay_v1"


def _cap_norm(capability_id: str) -> str:
    return str(capability_id or "").strip().split("__", 1)[0]


def fl2_solver_agent_replay_eval_authority(row: Optional[Mapping[str, Any]] = None) -> bool:
    """True when grading must bind to executed solver-cap replay refs."""
    flag = str(os.environ.get("HWA_FL2_SOLVER_AGENT_REPLAY", "")).strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    inv = dict(row or {})
    if str(inv.get("taskpack_id") or "") != FL2_SOLVER_TASKPACK_ID:
        return False
    if str(inv.get("track") or "").upper() != "FL-2":
        return False
    return bool(
        inv.get("unified_benchmark_v1")
        or inv.get("dynamic_intervention_w3_47_v1")
        or inv.get("hwb_headline_inventory")
        or inv.get("ablation_manual_pilot")
    )


@lru_cache(maxsize=256)
def fl2_sealed_prediction_rmse(
    capability_id: str,
    scenario_id: str,
    split: str,
) -> float:
    """RMSE from sealed CARP scientific predictions for one scenario."""
    cap = _cap_norm(capability_id)
    sid = str(scenario_id or "").strip()
    if cap not in FL2_SOLVER_CAPS or not sid:
        raise KeyError(f"invalid FL-2 replay key: cap={cap!r} scenario={sid!r}")
    from hazardweaver.hcg.carp.native_eval.fl2_subset import rmse
    from hazardweaver.hcg.carp.scientific.fl2_data import TASKPACK, load_scenario_arrays
    from hazardweaver.hcg.carp.scientific.paths import cap_scientific_dir

    sub = "test" if str(split or "").strip() == "official_test" else "holdout"
    cap_dir = cap_scientific_dir(TASKPACK, cap)
    pred_path = cap_dir / "predictions" / sub / f"{sid}_depth.npy"
    if not pred_path.is_file():
        raise KeyError(f"missing sealed prediction: {pred_path}")
    _dem, _initial, truth = load_scenario_arrays(str(split or "hwb_holdout"), sid)
    import numpy as np

    pred = np.load(pred_path)
    return float(rmse(pred, truth))


def fl2_solver_replay_scenario_ref_for_capability(
    scenario_id: str,
    capability_id: str,
    *,
    split: str = "hwb_holdout",
    inventory_row: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Sidecar-shaped ref bound to executed solver capability."""
    cap = _cap_norm(capability_id)
    sid = str(scenario_id or "").strip()
    if cap not in FL2_SOLVER_CAPS:
        raise KeyError(f"not an FL-2 solver replay cap: {cap}")
    if not sid:
        raise KeyError("scenario_id required for FL-2 solver replay ref")

    from hazardweaver.hwb.registry.fl2_parametric_resolver import load_scenario_refs

    sidecar = load_scenario_refs(FL2_SOLVER_TASKPACK_ID)
    scenario_ref = dict((sidecar.get("scenarios") or {}).get(sid) or {})
    witness = str((scenario_ref.get("provenance") or {}).get("reference_witness") or "").strip()

    if witness == cap and scenario_ref.get("outputs"):
        ref = dict(scenario_ref)
        ref["capability_id"] = cap
        return ref

    rmse_val = fl2_sealed_prediction_rmse(cap, sid, str(split or "hwb_holdout"))
    from hazardweaver.hwb.registry.unified_benchmark_tolerance_v1 import calibrate_tolerance

    tol = calibrate_tolerance("rmse_depth", rmse_val)
    return {
        "capability_id": cap,
        "outputs": {
            "rmse_depth": rmse_val,
            "metric_name": "rmse_depth",
            "reference_score": rmse_val,
            "scenario_id": sid,
        },
        "tolerance": tol,
        "provenance": {
            "scenario_id": sid,
            "split": split,
            "reference_witness": cap,
            "reference_mode": FL2_REPLAY_AUTHORITY,
            "sidecar_witness": witness or None,
        },
    }


def replay_rmse_matches_execution(
    capability_id: str,
    scenario_id: str,
    split: str,
    executed_rmse: float,
    *,
    epsilon: float = 1e-4,
) -> bool:
    cap = _cap_norm(capability_id)
    ref = fl2_sealed_prediction_rmse(cap, str(scenario_id), str(split))
    return abs(float(executed_rmse) - ref) <= epsilon
