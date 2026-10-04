"""Anchor metadata for g6_hard_v1 → HWB TaskPack / baseline eligibility."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional, Tuple

# g2_recipes/anchors.yaml data keys
ANCHOR_DATASET: Dict[str, str] = {
    "A_PFDF_ASSESS_V1": "usgs_pfdf",
    "A_TCCF_CHARLESTON_HARVEY_V1": "tccf_dual",
    "A_CF_COOPS_USGS_OPS_V1": "coops",
    "A_WF_ALGERIA_V1": "algeria",
    "A_WF_PORTUGAL_V1": "portugal",
    "A_WF_MTBS_EVENT_V1": "mtbs",
    "A_LS_IT_SU_V1": "ls_it",
    "A_LS_INV_V3_HARMONIZE_V1": "ls_inv",
    "A_DH_GHCND_SPI_V1": "ghcn",
    "A_TC_IBTRACS_V1": "ibtracs",
}

HEADLINE_TARGET: Dict[str, str] = {
    "A_PFDF_ASSESS_V1": "MH-1",
    "A_TCCF_CHARLESTON_HARVEY_V1": "MH-3",
    "A_CF_COOPS_USGS_OPS_V1": "MH-3",
    "A_WF_ALGERIA_V1": "WF-3",
    "A_WF_PORTUGAL_V1": "WF-3",
    "A_WF_MTBS_EVENT_V1": "WF-3",
    "A_LS_IT_SU_V1": "L2",
    "A_LS_INV_V3_HARMONIZE_V1": "L2",
    "A_DH_GHCND_SPI_V1": "DR-OUT",
    "A_TC_IBTRACS_V1": "TC-TRK",
}

# Tabular supervised ML — AIDE / MLE-STAR native scope
TABULAR_SUPERVISED_ANCHORS: FrozenSet[str] = frozenset(
    {
        "A_LS_IT_SU_V1",
        "A_WF_PORTUGAL_V1",
        "A_WF_ALGERIA_V1",
        "A_WF_MTBS_EVENT_V1",
        "A_DH_GHCND_SPI_V1",
        "A_PFDF_ASSESS_V1",
    }
)

# Workflow / DAG / time-series orchestration — N/A for AIDE
WORKFLOW_ANCHORS: FrozenSet[str] = frozenset(
    {
        "A_CF_COOPS_USGS_OPS_V1",
        "A_TCCF_CHARLESTON_HARVEY_V1",
        "A_TC_IBTRACS_V1",
        "A_LS_INV_V3_HARMONIZE_V1",
    }
)

AIDE_GOAL: Dict[str, str] = {
    "A_LS_IT_SU_V1": (
        "Build a classifier to predict landslide susceptibility (binary target) "
        "from terrain and soil features in the Italy slope-unit inventory."
    ),
    "A_WF_PORTUGAL_V1": (
        "Build a regressor to predict log1p burned area from weather and fire "
        "index features for the Portugal forest fires dataset."
    ),
    "A_WF_ALGERIA_V1": (
        "Build a classifier to predict wildfire occurrence (fire vs not fire) "
        "from weather and fire danger index features."
    ),
    "A_WF_MTBS_EVENT_V1": (
        "Build a classifier to predict wildfire vs non-wildfire MTBS burn events "
        "from acreage and location features."
    ),
    "A_DH_GHCND_SPI_V1": (
        "Build a regressor to predict standardized precipitation anomaly (SPI-like) "
        "from lagged station precipitation features."
    ),
    "A_PFDF_ASSESS_V1": (
        "Build a regressor to predict log post-fire debris-flow volume from burn, "
        "terrain, and rainfall features (USGS PFDF tabular subset)."
    ),
}

AIDE_EVAL: Dict[str, str] = {
    "A_LS_IT_SU_V1": "accuracy",
    "A_WF_PORTUGAL_V1": "MAE",
    "A_WF_ALGERIA_V1": "accuracy",
    "A_WF_MTBS_EVENT_V1": "accuracy",
    "A_DH_GHCND_SPI_V1": "MAE",
    "A_PFDF_ASSESS_V1": "MAE",
}

# First live AIDE pilots (CoreExec)
AIDE_PILOT_TASK_IDS: Tuple[str, ...] = (
    "H_A_LS_IT_SU_V1_COREEXEC_dag_rf_selected_path",
    "H_A_WF_PORTUGAL_V1_COREEXEC_dag_ridge_selected_path",
    "H_A_DH_GHCND_SPI_V1_COREEXEC_dag_rf_selected_path",
)


def taskpack_id_for_g6_task(g6_task_id: str) -> str:
    return "hwb_g6_" + g6_task_id.lower().replace("-", "_") + "_v1"


def anchor_from_task(task: Dict[str, Any]) -> str:
    return str(task.get("anchor_id") or "")
