"""Frozen grader — no LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hazardweaver.hwb.synthetic.evaluators import counterfactual as cf_util
from hazardweaver.hwb.synthetic.evaluators import metrics
from hazardweaver.hwb.synthetic.io.bundle_io import load_bundle, load_counterfactual_labels


def _pick_metric(label_type: str) -> str:
    if label_type == "binary":
        return "f1_score"
    if label_type == "raster_mask":
        return "iou"
    return "mae"


def evaluate_bundle(bundle_path: Path, predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
    data = load_bundle(bundle_path)
    labels = data["labels"]
    manifest = data["manifest"]
    label_type = manifest.get("label_type", "regression")

    if predictions is None:
        # oracle-style eval on hidden labels using mean predictor
        predictions = np.full_like(labels, float(np.mean(labels)))

    metric_name = _pick_metric(label_type)
    fn = getattr(metrics, metric_name)
    score = fn(predictions, labels)

    return {
        "metric": metric_name,
        "score": score,
        "n_samples": int(manifest.get("n_samples", labels.shape[0])),
    }


def evaluate_counterfactual_family(bundle_path: Path, predictions_dict: Optional[Dict] = None) -> Dict[str, Any]:
    bundle_path = Path(bundle_path)
    data = load_bundle(bundle_path)
    labels = data["labels"]
    manifest = data["manifest"]
    cf_names = manifest.get("counterfactual_names", [])

    with open(bundle_path / "evaluation" / "grader_config.json", encoding="utf-8") as f:
        grader_cfg = json.load(f)
    dep_scores = grader_cfg.get("dependency_scores", {})

    cf_labels = {name: load_counterfactual_labels(bundle_path, name) for name in cf_names}
    summary = cf_util.summarize_counterfactual_family(labels, cf_labels, dep_scores)
    return {"counterfactual_summary": summary, "dependency_scores": dep_scores}


def summarize_generation_statistics(bundle_path: Path) -> Dict[str, Any]:
    data = load_bundle(bundle_path)
    labels = data["labels"]
    return {
        "label_mean": float(np.mean(labels)),
        "label_std": float(np.std(labels)),
        "label_min": float(np.min(labels)),
        "label_max": float(np.max(labels)),
        "finite": bool(np.isfinite(labels).all()),
    }
