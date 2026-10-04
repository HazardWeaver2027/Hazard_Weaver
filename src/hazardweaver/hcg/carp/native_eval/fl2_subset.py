"""Shared FloodCast eval_subset loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_MANIFEST = PROJECT_ROOT / "data" / "vendor" / "floodcastbench" / "eval_subset" / "manifest.json"


def load_scenarios() -> Tuple[Path, List[Dict[str, Any]]]:
    if not EVAL_MANIFEST.is_file():
        raise FileNotFoundError(f"missing {EVAL_MANIFEST}")
    meta = json.loads(EVAL_MANIFEST.read_text(encoding="utf-8"))
    base = EVAL_MANIFEST.parent
    scenarios = meta.get("scenarios") or []
    if not scenarios:
        raise ValueError("empty floodcast eval_subset scenarios")
    return base, scenarios


def load_scenario_arrays(
    row: Dict[str, Any], base: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dem = np.load(base / row["dem"])
    initial = np.load(base / row["initial_depth"])
    truth = np.load(base / row["truth_depth"])
    return dem, initial, truth


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred.astype(np.float64) - truth.astype(np.float64)) ** 2)))


def csi_at_threshold(pred: np.ndarray, truth: np.ndarray, thresh: float) -> float:
    p = pred >= thresh
    t = truth >= thresh
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 1.0
