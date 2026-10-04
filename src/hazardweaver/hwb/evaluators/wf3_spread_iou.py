"""Native-Route WF-3 spread IoU on raster grids."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np


def grid_iou(pred: np.ndarray, ref: np.ndarray, *, threshold: float = 0.5) -> float:
    p = (np.asarray(pred, dtype=np.float64) >= threshold).astype(np.uint8)
    r = (np.asarray(ref, dtype=np.float64) >= threshold).astype(np.uint8)
    inter = float(np.logical_and(p, r).sum())
    union = float(np.logical_or(p, r).sum())
    if union <= 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def score_spread_pair(pred_grid: np.ndarray, ref_grid: np.ndarray) -> Dict[str, Any]:
    iou = grid_iou(pred_grid, ref_grid)
    return {
        "metric_name": "spread_iou",
        "score": float(iou),
        "higher_is_better": True,
        "threshold": 0.5,
    }


def aggregate_iou(scores: Mapping[str, float]) -> float:
    vals = [float(v) for v in scores.values()]
    return float(np.mean(vals)) if vals else float("nan")
