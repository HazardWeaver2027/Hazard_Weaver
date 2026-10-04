"""Per-scenario FL-2 solver metrics (CAP-FL2-04~06, aligned with HCG batch predictions)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hazardweaver.hwa.agent_runtime.execution_schema import artifacts_dir, load_execution
from hazardweaver.hwa.route_controller.fl2_pilot_slice import FL2_SOLVER_OFFICIAL_CAPS


def score_hand_scenario(*, split: str, scenario_id: str) -> Dict[str, Any]:
    """Faithful HAND rmse_depth + CSI@0.01 on one FloodCastBench scenario."""
    from hazardweaver.hcg.carp.native_eval.fl2_subset import csi_at_threshold, rmse
    from hazardweaver.hcg.carp.native_eval.scientific.fl2_hand_faithful import (
        compute_hand_depth_sequence,
    )
    from hazardweaver.hcg.carp.scientific import fl2_data

    if scenario_id not in fl2_data.scenario_ids(split):
        return {"status": "data_not_ready", "split": split, "scenario_id": scenario_id}
    dem, initial, truth = fl2_data.load_scenario_arrays(split, scenario_id)
    pred = compute_hand_depth_sequence(dem, initial, truth.shape[0])
    return {
        "status": "ok",
        "split": split,
        "scenario_id": scenario_id,
        "metric_name": "rmse_depth",
        "rmse_depth": float(rmse(pred, truth)),
        "csi_0.01": float(csi_at_threshold(pred, truth, 0.01)),
    }


def _load_batch_prediction(
    capability_id: str,
    *,
    split: str,
    scenario_id: str,
) -> Optional[np.ndarray]:
    """Load official batch pred_depth if present (login-safe; no re-infer)."""
    from hazardweaver.hcg.carp.scientific import fl2_data
    from hazardweaver.hcg.carp.scientific.paths import cap_scientific_dir

    cap_dir = cap_scientific_dir(fl2_data.TASKPACK, capability_id)
    pred_dir = (
        cap_dir
        / fl2_data.predictions_base_dir(split)
        / fl2_data.prediction_subdir(split)
    )
    pred_path = fl2_data.prediction_path(cap_dir, split, scenario_id)
    if pred_path.is_file():
        data = np.load(pred_path)
        if "pred_depth" in data:
            return np.asarray(data["pred_depth"], dtype=np.float32)
    legacy = pred_dir / f"{scenario_id}_depth.npy"
    if legacy.is_file():
        return np.load(legacy).astype(np.float32)
    blocked = pred_dir / f"{scenario_id}_blocked.json"
    if blocked.is_file():
        return None
    return None


def score_fl2_solver_scenario(
    *,
    capability_id: str,
    split: str,
    scenario_id: str,
) -> Dict[str, Any]:
    """Score one scenario for CAP-FL2-04~06 using batch predictions or HAND faithful."""
    from hazardweaver.hcg.carp.native_eval.fl2_subset import csi_at_threshold, rmse
    from hazardweaver.hcg.carp.scientific import fl2_data

    if capability_id not in FL2_SOLVER_OFFICIAL_CAPS:
        return {
            "status": "unsupported_cap",
            "capability_id": capability_id,
            "split": split,
            "scenario_id": scenario_id,
        }
    if scenario_id not in fl2_data.scenario_ids(split):
        return {"status": "data_not_ready", "split": split, "scenario_id": scenario_id}

    pred = _load_batch_prediction(capability_id, split=split, scenario_id=scenario_id)
    if pred is None and capability_id == "CAP-FL2-06":
        return score_hand_scenario(split=split, scenario_id=scenario_id)
    if pred is None:
        return {
            "status": "blocked",
            "capability_id": capability_id,
            "split": split,
            "scenario_id": scenario_id,
            "metric_name": "rmse_depth",
        }

    _, _, truth = fl2_data.load_scenario_arrays(split, scenario_id)
    if pred.shape != truth.shape:
        from skimage.transform import resize

        aligned = np.zeros_like(truth, dtype=np.float32)
        for t in range(min(pred.shape[0], truth.shape[0])):
            aligned[t] = resize(
                pred[t],
                truth.shape[1:],
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
        pred = aligned

    return {
        "status": "ok",
        "capability_id": capability_id,
        "split": split,
        "scenario_id": scenario_id,
        "metric_name": "rmse_depth",
        "rmse_depth": float(rmse(pred, truth)),
        "csi_0.01": float(csi_at_threshold(pred, truth, 0.01)),
        "source": "hcg_batch_prediction",
    }


def patch_workdir_metrics_for_scenario(
    workdir: Path,
    *,
    execution_id: str,
    final_artifact_id: str,
    scenario_id: str,
    split: str,
    capability_id: str = "CAP-FL2-06",
) -> Dict[str, Any]:
    """Overwrite execution registry artifact with per-scenario solver metrics."""
    workdir = Path(workdir)
    scored = score_fl2_solver_scenario(
        capability_id=capability_id,
        split=split,
        scenario_id=scenario_id,
    )
    if scored.get("status") != "ok":
        raise RuntimeError(f"scenario_score_failed:{scored}")

    value = {
        "scenario_id": scenario_id,
        "metric_name": "rmse_depth",
        "rmse_depth": scored["rmse_depth"],
        "csi_0.01": scored["csi_0.01"],
        "split": split,
        "capability_id": capability_id,
    }
    art_path = artifacts_dir(workdir) / f"{final_artifact_id}.json"
    if art_path.is_file():
        body = json.loads(art_path.read_text(encoding="utf-8"))
    else:
        body = {"artifact_id": final_artifact_id, "schema_id": "hwa.final_artifact/v1"}
    body["value_or_uri"] = value
    body.pop("value", None)
    art_path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")

    er = load_execution(workdir, execution_id)
    if er:
        fa = dict(er.get("final_artifact") or {})
        fa["artifact_id"] = final_artifact_id
        fa["schema_id"] = str(fa.get("schema_id") or "hwa.final_artifact/v1")
        fa["value_or_uri"] = value
        fa.setdefault("units_support", {"units": None, "support": {}})
        er["final_artifact"] = fa
        er["provenance"] = {
            **dict(er.get("provenance") or {}),
            "inference_mode": "scientific_replay",
            "scenario_id": scenario_id,
            "split": split,
            "capability_id": capability_id,
            "dispatch": f"fl2_{capability_id.lower().replace('-', '_')}_per_scenario",
        }
        from hazardweaver.hwa.agent_runtime.execution_schema import write_execution

        write_execution(workdir, er)
    return value
