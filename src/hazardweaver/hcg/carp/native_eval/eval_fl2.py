"""FL-2 native eval (Phase 3 rollout)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic, write_blocked
from hazardweaver.hcg.carp.native_eval.eval_fl2_baseline import eval_hand_baseline
from hazardweaver.hcg.carp.native_eval.eval_fl2_solver import eval_fl2_solver
from hazardweaver.hcg.carp.native_eval.fl2_subset import (
    csi_at_threshold,
    load_scenario_arrays,
    load_scenarios,
    rmse,
)
from hazardweaver.hcg.carp.native_eval.train_fl2_floodcast import _predict, train_cap

G2_CAPS = {
    "CAP-FL2-01": "RF-SPATIAL-CONVOLUTIONAL-SURROG",
    "CAP-FL2-02": "RF-SPECTRAL-NEURAL-OPERATOR",
    "CAP-FL2-03": "RF-FOUNDATION-PRETRAINED-NEURAL",
}


def _eval_g2_cap(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    family_id = G2_CAPS[capability_id]
    try:
        train_cap(capability_id)
        base, scenarios = load_scenarios()
    except (FileNotFoundError, ValueError) as exc:
        return write_blocked("FL-2", capability_id, str(exc), out_base=out_base)

    rmses: List[float] = []
    csis: List[float] = []
    for row in scenarios:
        dem, initial, truth = load_scenario_arrays(row, base)
        pred_seq = []
        state = initial
        for _ in range(truth.shape[0]):
            state = _predict(capability_id, state, dem)
            pred_seq.append(state)
        pred = np.stack(pred_seq)
        rmses.append(rmse(pred, truth))
        csis.append(csi_at_threshold(pred, truth, 0.01))

    mean_rmse = float(np.mean(rmses))
    metrics = {
        "metric_name": "rmse_depth",
        "metric_value": mean_rmse,
        "csi_0.01": float(np.mean(csis)),
        "n_scenarios": len(rmses),
        "synthetic_only": False,
        "data_source": "floodcast_eval_subset",
        "note": "FloodCast eval_subset light_train engineering eval",
    }
    write_metrics("FL-2", capability_id, metrics, base=out_base)
    write_replay_manifest(
        "FL-2",
        capability_id,
        family_id=family_id,
        exec_ok=True,
        metric_name="rmse_depth",
        metric_value=mean_rmse,
        notes="g2_train eval_subset replay",
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}


def eval_fl2_cap_for_scenario(
    capability_id: str,
    scenario_id: str,
    *,
    split: str = "official_test",
    out_base: Optional[Path] = None,
) -> Dict[str, Any]:
    """Per-scenario G2 replay — aligns agent rescore with unified FL-2 parametric gold."""
    cap = str(capability_id).strip()
    sid = str(scenario_id or "").strip()
    if cap not in G2_CAPS or not sid:
        return eval_fl2_cap(cap, out_base=out_base)
    try:
        from hazardweaver.hcg.carp.native_eval.train_fl2_floodcast import _predict, train_cap
        from hazardweaver.hcg.carp.scientific.fl2_data import load_scenario_arrays, scientific_data_ready

        if not scientific_data_ready():
            raise FileNotFoundError("FL-2 scientific data not ready")
        train_cap(cap)
        dem, initial, truth = load_scenario_arrays(split, sid)
        pred_seq: List[np.ndarray] = []
        state = initial
        for _ in range(truth.shape[0]):
            state = _predict(cap, state, dem)
            pred_seq.append(state)
        pred = np.stack(pred_seq)
        mean_rmse = float(rmse(pred, truth))
        metrics = {
            "metric_name": "rmse_depth",
            "metric_value": mean_rmse,
            "csi_0.01": float(csi_at_threshold(pred, truth, 0.01)),
            "scenario_id": sid,
            "split": split,
            "n_scenarios": 1,
            "synthetic_only": False,
            "data_source": "floodcastbench_official_test_v1",
            "note": f"G2 per-scenario engineering replay ({sid})",
        }
        write_metrics("FL-2", cap, metrics, base=out_base)
        return {"ok": True, "metrics": metrics}
    except (FileNotFoundError, ValueError, OSError) as exc:
        return write_blocked("FL-2", cap, str(exc), out_base=out_base)


def eval_fl2_cap(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    from hazardweaver.hcg.carp.scientific.fl2_data import scientific_data_ready
    from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT

    if scientific_data_ready():
        from hazardweaver.hcg.carp.native_eval.scientific.fl2_floodcast import eval_fl2_scientific

        sci_base = out_base or SCIENTIFIC_RUNS_ROOT
        return eval_fl2_scientific(capability_id, out_base=sci_base)

    if capability_id == "CAP-FL2-06":
        return eval_hand_baseline(out_base=out_base)
    if capability_id in ("CAP-FL2-04", "CAP-FL2-05"):
        return eval_fl2_solver(capability_id, out_base=out_base)
    if capability_id in G2_CAPS:
        return _eval_g2_cap(capability_id, out_base=out_base)
    return eval_blocked_generic("FL-2", capability_id, "no native eval wired", out_base=out_base)
