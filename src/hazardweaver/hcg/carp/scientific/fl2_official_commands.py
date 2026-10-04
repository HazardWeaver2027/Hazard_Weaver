"""Official FloodCastBench / solver command templates per CAP-FL2-* (Phase C)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from hazardweaver.hcg.carp.acquire.handlers.sfincs import DOCKER_IMAGE, OFFICIAL_DOCKER_CMD
from hazardweaver.hcg.carp.scientific.fl2_data import PROJECT_ROOT, VENDOR_ROOT

TASKPACK = "FL-2"

CAP_MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "CAP-FL2-01": {
        "family_id": "RF-SPATIAL-CONVOLUTIONAL-SURROG",
        "mode": "unet_train_infer",
        "metric_name": "rmse_depth",
        "paper_config": "U-Net 3D conv; batch=1; 100 epochs; Adam lr=0.001",
    },
    "CAP-FL2-02": {
        "family_id": "RF-SPECTRAL-NEURAL-OPERATOR",
        "mode": "fno_train_infer",
        "metric_name": "rmse_depth",
        "paper_config": "FNO 4 layers; 12 Fourier modes; latent=20",
    },
    "CAP-FL2-03": {
        "family_id": "RF-FOUNDATION-PRETRAINED-NEURAL",
        "mode": "fno_plus_train_infer",
        "metric_name": "rmse_depth",
        "paper_config": "FNO+ with DEM + rainfall t=1..20",
    },
    "CAP-FL2-04": {
        "family_id": "RF-REDUCED-PHYSICS-SHALLOW-WATE",
        "mode": "sfincs_docker",
        "metric_name": "rmse_depth",
    },
    "CAP-FL2-05": {
        "family_id": "RF-HYDRODYNAMIC-SOLVER",
        "mode": "lisflood_fp_cli",
        "metric_name": "rmse_depth",
    },
    "CAP-FL2-06": {
        "family_id": "RF-TERRAIN-INDEX-BASELINE",
        "mode": "hand_faithful",
        "metric_name": "rmse_depth",
    },
}

G2_CAPS = ["CAP-FL2-01", "CAP-FL2-02", "CAP-FL2-03"]


def cap_family_id(capability_id: str) -> str:
    spec = CAP_MODEL_CONFIG.get(capability_id) or {}
    return str(spec.get("family_id") or "unknown")


def scientific_runner() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "native_eval"
        / "scientific"
        / "fl2_floodcast_worker.py"
    )


def build_g2_train_command(capability_id: str) -> str:
    runner = scientific_runner()
    py = os.environ.get("PYTHON", "python")
    mode = CAP_MODEL_CONFIG[capability_id]["mode"]
    return f"{py} {runner} --capability-id {capability_id} --mode {mode} --split official_test"


from hazardweaver.hcg.carp.scientific.fl2_sfincs_runtime import build_sfincs_run_command
from hazardweaver.hcg.carp.scientific.fl2_lisflood_runtime import build_lisflood_run_command


def build_sfincs_command(scenario_dir: str) -> str:
    return build_sfincs_run_command(scenario_dir)


def build_lisflood_command(model_dir: str) -> str:
    return build_lisflood_run_command(model_dir)


def build_hand_command(capability_id: str) -> str:
    runner = scientific_runner()
    py = os.environ.get("PYTHON", "python")
    return f"{py} {runner} --capability-id {capability_id} --mode hand_faithful --split official_test"


def build_official_command(capability_id: str, *, split: str = "official_test") -> str:
    spec = CAP_MODEL_CONFIG.get(capability_id)
    if spec is None:
        raise ValueError(f"unknown capability {capability_id}")
    mode = spec["mode"]
    if mode in ("unet_train_infer", "fno_train_infer", "fno_plus_train_infer"):
        return build_g2_train_command(capability_id)
    if mode == "sfincs_docker":
        return OFFICIAL_DOCKER_CMD.replace("$MODEL_DIR", f"runs/carp/scientific/FL-2/{capability_id}/driver")
    if mode == "lisflood_fp_cli":
        return build_lisflood_command(f"runs/carp/scientific/FL-2/{capability_id}/driver")
    if mode == "hand_faithful":
        return build_hand_command(capability_id)
    raise ValueError(f"unsupported mode {mode}")
