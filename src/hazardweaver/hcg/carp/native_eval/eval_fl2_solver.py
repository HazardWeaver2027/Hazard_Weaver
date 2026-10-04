"""FL-2 reduced-physics solver stubs (engineering)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import write_blocked
from hazardweaver.hcg.carp.native_eval.fl2_subset import (
    csi_at_threshold,
    load_scenario_arrays,
    load_scenarios,
    rmse,
)

SOLVER_CAPS = {
    "CAP-FL2-04": ("RF-REDUCED-PHYSICS-SHALLOW-WATE", "reduced_physics"),
    "CAP-FL2-05": ("RF-HYDRODYNAMIC-SOLVER", "inertial"),
}


def _reduced_physics_step(state: np.ndarray, dem: np.ndarray, rain: float = 0.05) -> np.ndarray:
    flux = np.zeros_like(state)
    flux[1:, :] += 0.25 * np.maximum(state[:-1, :] - dem[:-1, :], 0.0)
    flux[:, 1:] += 0.25 * np.maximum(state[:, :-1] - dem[:, :-1], 0.0)
    return np.clip(state + rain + 0.9 * flux, 0.0, 3.0)


def _inertial_step(state: np.ndarray, dem: np.ndarray) -> np.ndarray:
    slope = np.gradient(dem)
    drive = 0.1 * (slope[0] + slope[1])
    return np.clip(state * 0.92 + np.maximum(drive, 0.0), 0.0, 3.0)


def eval_fl2_solver(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if capability_id not in SOLVER_CAPS:
        return write_blocked("FL-2", capability_id, "unknown solver cap", out_base=out_base)
    family_id, mode = SOLVER_CAPS[capability_id]
    try:
        base, scenarios = load_scenarios()
    except (FileNotFoundError, ValueError) as exc:
        return write_blocked("FL-2", capability_id, str(exc), out_base=out_base)

    rmses: List[float] = []
    csis: List[float] = []
    step_fn = _reduced_physics_step if mode == "reduced_physics" else _inertial_step
    for row in scenarios:
        dem, initial, truth = load_scenario_arrays(row, base)
        state = initial.copy()
        pred_seq = []
        for _ in range(truth.shape[0]):
            state = step_fn(state, dem)
            pred_seq.append(state.copy())
        pred = np.stack(pred_seq)
        rmses.append(rmse(pred, truth))
        csis.append(csi_at_threshold(pred, truth, 0.01))

    mean_rmse = float(np.mean(rmses))
    metrics = {
        "metric_name": "rmse_depth",
        "metric_value": mean_rmse,
        "csi_0.01": float(np.mean(csis)),
        "solver_mode": mode,
        "n_scenarios": len(rmses),
        "synthetic_only": False,
        "data_source": "floodcast_eval_subset",
        "note": f"engineering {mode} solver stub; not full SFINCS/LISFLOOD",
    }
    write_metrics("FL-2", capability_id, metrics, base=out_base)
    write_replay_manifest(
        "FL-2",
        capability_id,
        family_id=family_id,
        exec_ok=True,
        metric_name="rmse_depth",
        metric_value=mean_rmse,
        notes=f"{mode} engineering solver stub",
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}
