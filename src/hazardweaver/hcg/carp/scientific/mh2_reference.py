"""Reference raster helpers for MH-2 replay parity (DL-046)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.scientific.mh2_fidelity import PRIMARY_EVENT_ID, REFERENCE_MANIFEST


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference_manifest(ref_dir: Path) -> Optional[Dict[str, Any]]:
    path = ref_dir / REFERENCE_MANIFEST
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_reference_manifest(
    ref_dir: Path,
    *,
    model_key: str,
    event_id: str,
    reference_kind: str,
    prob_path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    ref_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_id": event_id,
        "model_key": model_key,
        "reference_kind": reference_kind,
        "prob_path": str(prob_path),
        "prob_sha256": sha256_file(prob_path),
        "pinned_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    path = ref_dir / REFERENCE_MANIFEST
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        models = existing.get("models") or {}
        if not isinstance(models, dict):
            models = {}
        models[model_key] = payload
        existing["models"] = models
        existing["updated_at"] = payload["pinned_at"]
        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    else:
        path.write_text(
            json.dumps({"event_id": event_id, "models": {model_key: payload}}, indent=2) + "\n",
            encoding="utf-8",
        )
    return path


def load_reference_prob(ref_dir: Optional[Path], model_key: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if ref_dir is None or not ref_dir.is_dir():
        return None, None
    manifest = load_reference_manifest(ref_dir)
    if manifest and manifest.get("models", {}).get(model_key):
        entry = manifest["models"][model_key]
        path = Path(entry["prob_path"])
        if path.is_file():
            return np.load(path).astype(np.float32).ravel(), str(entry.get("reference_kind", "pinned"))
    for name in (f"{model_key}_prob.npy", "reference_prob.npy", "prob.npy"):
        path = ref_dir / name
        if path.is_file():
            return np.load(path).astype(np.float32).ravel(), "legacy_npy"
    return None, None


def align_vectors(pred: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(pred.size, ref.size)
    if n == 0:
        return np.array([]), np.array([])
    return pred.ravel()[:n].astype(np.float64), ref.ravel()[:n].astype(np.float64)


def replay_parity(pred: np.ndarray, ref: Optional[np.ndarray]) -> Tuple[Optional[float], bool]:
    """Pearson r on aligned finite cells; None if reference missing."""
    if ref is None or pred.size == 0 or ref.size == 0:
        return None, False
    a, b = align_vectors(pred, ref)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return 0.0, True
    a = a[mask]
    b = b[mask]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float(1.0 - np.mean(np.abs(a - b))), True
    corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        return 0.0, True
    return float(max(0.0, min(1.0, corr))), True


def coarsen_raster(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    h, w = arr.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    trimmed = arr[:h2, :w2]
    return trimmed.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))
