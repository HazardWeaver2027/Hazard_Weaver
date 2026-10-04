"""Official SeisBench / stage command templates per CAP-E1E3-* (Phase C)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from hazardweaver.hcg.carp.scientific.e1e3_data import VENDOR_ROOT, seisbench_python

TASKPACK = "E1-E3"

CAP_MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "CAP-E1E3-01": {
        "family_id": "RF-CLASSICAL-SIGNAL-TO-CHAIN",
        "mode": "sta_lta_picker",
        "metric_name": "pick_f1",
    },
    "CAP-E1E3-02": {
        "family_id": "RF-CNN-PHASE-PICKING",
        "mode": "seisbench_classify",
        "model_class": "PhaseNet",
        "weight_variant": "original",
        "model_store": "phasenet",
        "metric_name": "pick_f1",
    },
    "CAP-E1E3-03": {
        "family_id": "RF-ATTENTION-RECURRENT-DETECTIO",
        "mode": "seisbench_classify",
        "model_class": "EQTransformer",
        "weight_variant": "original",
        "model_store": "eqtransformer",
        "metric_name": "pick_f1",
    },
    "CAP-E1E3-04": {
        "family_id": "RF-WINDOW-CLASSIFIER-PICKER",
        "mode": "seisbench_classify",
        "model_class": "GPD",
        "weight_variant": "stead",
        "model_store": "gpd",
        "metric_name": "pick_f1",
    },
    "CAP-E1E3-05": {
        "family_id": "RF-OCTREE-TRAVEL-TIME-ASSOCIATI",
        "mode": "pyocto_association",
        "metric_name": "association_score",
    },
    "CAP-E1E3-06": {
        "family_id": "RF-PROBABILISTIC-MIXTURE-ASSOCI",
        "mode": "gamma_association",
        "metric_name": "association_score",
    },
    "CAP-E1E3-07": {
        "family_id": "RF-GRID-SEARCH-ASSOCIATION",
        "mode": "real_association",
        "metric_name": "association_score",
    },
    "CAP-E1E3-08": {
        "family_id": "RF-GMPE--OBSERVATION-FUSION",
        "mode": "shakemap_gmpe",
        "metric_name": "gmpe_coverage",
    },
}

PICKER_CAPS = ["CAP-E1E3-02", "CAP-E1E3-03", "CAP-E1E3-04"]


def cap_family_id(capability_id: str) -> str:
    spec = CAP_MODEL_CONFIG.get(capability_id) or {}
    return str(spec.get("family_id") or "unknown")


def build_sta_lta_command(*, split: str = "official_test") -> str:
    runner = (
        Path(__file__).resolve().parent.parent
        / "native_eval"
        / "scientific"
        / "e1e3_picker_eval_worker.py"
    )
    py = os.environ.get("PYTHON", "python")
    return f"{py} {runner} --mode sta_lta --split {split}"


def build_seisbench_classify_command(capability_id: str, *, split: str = "official_test") -> str:
    spec = CAP_MODEL_CONFIG[capability_id]
    store = spec["model_store"]
    variant = spec["weight_variant"]
    runner = (
        Path(__file__).resolve().parent.parent
        / "native_eval"
        / "scientific"
        / "e1e3_picker_eval_worker.py"
    )
    return (
        f"{seisbench_python()} {runner} --mode seisbench --model-store {store} "
        f"--weight-variant {variant} --split {split}"
    )


def build_pyocto_command() -> str:
    return (
        f"pyocto associate --picks runs/carp/scientific/E1-E3/CAP-E1E3-05/predictions/test/picks.json "
        f"--vendor-root {VENDOR_ROOT / 'pyocto'}"
    )


def build_gamma_command() -> str:
    return f"python -m gamma --config {VENDOR_ROOT / 'gamma' / 'config.yaml'}"


def build_real_command() -> str:
    return f"REAL --config {VENDOR_ROOT / 'real' / 'REAL.config'}"


def build_shakemap_command() -> str:
    return (
        "usgs_shakemap replay --holdout "
        "data/scientific/E1-E3/hwb_holdout/traces --schema usgs_shakemap_v1"
    )


def build_official_command(capability_id: str, *, split: str = "official_test") -> str:
    spec = CAP_MODEL_CONFIG.get(capability_id)
    if spec is None:
        raise ValueError(f"unknown capability {capability_id}")
    mode = spec["mode"]
    if mode == "sta_lta_picker":
        return build_sta_lta_command(split=split)
    if mode == "seisbench_classify":
        return build_seisbench_classify_command(capability_id, split=split)
    if mode == "pyocto_association":
        return build_pyocto_command()
    if mode == "gamma_association":
        return build_gamma_command()
    if mode == "real_association":
        return build_real_command()
    if mode == "shakemap_gmpe":
        return build_shakemap_command()
    raise ValueError(f"unsupported mode {mode}")
