"""Threat Score and related footprint metrics (Earth's Future 2026 anchor)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def binary_footprint(arr: np.ndarray, *, threshold: float = 0.0) -> np.ndarray:
    """Anchor-native debris-flow susceptibility index threshold (DFSI > 0)."""
    return np.asarray(arr, dtype=np.float64) > threshold


def threat_score(predicted: np.ndarray, observed: np.ndarray, *, threshold: float = 0.0) -> float:
    """Threat Score TS = TP / (TP + FP + FN); equivalent to footprint IoU."""
    pred = binary_footprint(predicted, threshold=threshold)
    obs = binary_footprint(observed, threshold=threshold)
    tp = float(np.sum(pred & obs))
    fp = float(np.sum(pred & ~obs))
    fn = float(np.sum(~pred & obs))
    denom = tp + fp + fn
    if denom <= 0:
        return float("nan")
    return tp / denom


def footprint_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    threshold: float = 0.0,
) -> Dict[str, float]:
    pred = binary_footprint(predicted, threshold=threshold)
    obs = binary_footprint(observed, threshold=threshold)
    tp = float(np.sum(pred & obs))
    fp = float(np.sum(pred & ~obs))
    fn = float(np.sum(~pred & obs))
    tn = float(np.sum(~pred & ~obs))
    hit_rate = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    far = fp / (tp + fp) if (tp + fp) > 0 else float("nan")
    bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else float("nan")
    ts = threat_score(predicted, observed, threshold=threshold)
    return {
        "threat_score": ts,
        "hit_rate": hit_rate,
        "false_alarm_ratio": far,
        "one_minus_far": 1.0 - far if np.isfinite(far) else float("nan"),
        "bias": bias,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def aggregate_threat_score(scores: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Mean threat score across anchor cases with secondary metrics averaged."""
    if not scores:
        return float("nan"), {}
    primary = float(np.nanmean([scores.get("threat_score", float("nan"))]))
    secondary = {
        k: float(np.nanmean([scores.get(k, float("nan"))]))
        for k in ("hit_rate", "false_alarm_ratio", "bias")
    }
    return primary, secondary
