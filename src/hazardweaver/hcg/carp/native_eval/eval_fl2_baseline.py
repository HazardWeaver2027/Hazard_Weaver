"""FL-2 HAND threshold baseline eval."""

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

CAPABILITY_ID = "CAP-FL2-06"
FAMILY_ID = "RF-TERRAIN-INDEX-BASELINE"


def _hand_inundation(dem: np.ndarray, water_level: float) -> np.ndarray:
    return np.clip(water_level - dem, 0.0, 3.0).astype(np.float32)


def eval_hand_baseline(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    try:
        base, scenarios = load_scenarios()
    except (FileNotFoundError, ValueError) as exc:
        return write_blocked("FL-2", CAPABILITY_ID, str(exc), out_base=out_base)

    rmses: List[float] = []
    csis: List[float] = []
    for row in scenarios:
        dem, initial, truth = load_scenario_arrays(row, base)
        wl = float(np.percentile(dem, 25) + initial.max())
        pred = _hand_inundation(dem, wl)
        pred_seq = np.stack([pred] * truth.shape[0])
        rmses.append(rmse(pred_seq, truth))
        csis.append(csi_at_threshold(pred_seq, truth, 0.01))

    mean_rmse = float(np.mean(rmses))
    mean_csi = float(np.mean(csis))
    metrics = {
        "metric_name": "rmse_depth",
        "metric_value": mean_rmse,
        "csi_0.01": mean_csi,
        "n_scenarios": len(rmses),
        "synthetic_only": False,
        "data_source": "floodcast_eval_subset",
        "note": "HAND threshold engineering baseline",
    }
    write_metrics("FL-2", CAPABILITY_ID, metrics, base=out_base)
    write_replay_manifest(
        "FL-2",
        CAPABILITY_ID,
        family_id=FAMILY_ID,
        exec_ok=True,
        metric_name="rmse_depth",
        metric_value=mean_rmse,
        notes="HAND threshold baseline",
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}
