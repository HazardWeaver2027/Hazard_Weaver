"""Shared dev replay helpers for expansion A2 tracks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from hazardweaver.hcg.carp.expansion.paths import DEV_ROOTS, PIN_ROOTS, PROJECT_ROOT

METRIC_FNS = {}


def _checksum(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_dev_manifest(taskpack_id: str) -> Tuple[Path, Dict[str, Any]]:
    root = DEV_ROOTS[taskpack_id]
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing expansion dev manifest {manifest}")
    return root, json.loads(manifest.read_text(encoding="utf-8"))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order].astype(np.int64)
    n_pos = y.sum()
    if n_pos == 0:
        return 0.0
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    return float(np.sum(precision * y) / n_pos)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob.astype(np.float64) - y_true.astype(np.float64)) ** 2))


def ordinal_hit_rate(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(pred.astype(np.int64) == truth.astype(np.int64)))


def volume_mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(pred.astype(np.float64) - truth.astype(np.float64))))


def lifeline_chain_score(pred_damage: np.ndarray, truth_damage: np.ndarray) -> float:
    pred_bin = pred_damage >= 0.5
    truth_bin = truth_damage >= 0.5
    tp = np.logical_and(pred_bin, truth_bin).sum()
    fp = np.logical_and(pred_bin, ~truth_bin).sum()
    fn = np.logical_and(~pred_bin, truth_bin).sum()
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 1.0


def write_pin_manifest(taskpack_id: str, payload: Dict[str, Any]) -> Path:
    root = PIN_ROOTS[taskpack_id]
    root.mkdir(parents=True, exist_ok=True)
    name = payload.pop("pin_name", "CHECKPOINT_PIN.json")
    path = root / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
