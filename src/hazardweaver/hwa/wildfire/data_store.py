"""HW-owned wildfire sample store. Reads only local manifests/npz — no external packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "hw_wildfire" / "manifests" / "datasets.json"
PACK_ROOT = PROJECT_ROOT / "configs" / "wildfire_agent_pack"


class WildfireDataStore:
    def __init__(self, manifest_path: Path | None = None):
        self.manifest_path = Path(manifest_path or DEFAULT_MANIFEST)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"HW wildfire dataset manifest missing: {self.manifest_path}. Run scripts/bootstrap/wildfire_hw_assets.py"
            )
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._cache: Dict[str, Dict[str, Any]] = {}

    def list_dataset_ids(self) -> List[str]:
        return sorted(self.manifest["datasets"].keys())

    def get_dataset_meta(self, dataset_id: str) -> Dict[str, Any]:
        if dataset_id not in self.manifest["datasets"]:
            raise KeyError(f"unknown dataset_id: {dataset_id}")
        return dict(self.manifest["datasets"][dataset_id])

    def _load_npz(self, dataset_id: str) -> Dict[str, Any]:
        if dataset_id in self._cache:
            return self._cache[dataset_id]
        meta = self.get_dataset_meta(dataset_id)
        path = PROJECT_ROOT / meta["npz_path"]
        with np.load(path, allow_pickle=False) as z:
            payload = {k: z[k] for k in z.files}
        self._cache[dataset_id] = payload
        return payload

    def resolve_index(self, dataset_id: str, sample_id: str, split: Optional[str] = None) -> Tuple[str, int]:
        meta = self.get_dataset_meta(dataset_id)
        split = split or meta.get("default_split", "test")
        mapping = meta["sample_ids"][split]
        if sample_id not in mapping:
            raise KeyError(f"sample_id {sample_id!r} not in {dataset_id}/{split}")
        return split, int(mapping[sample_id])

    def load_sample(
        self,
        dataset_id: str,
        sample_id: str,
        *,
        split: Optional[str] = None,
        include_label: bool = False,
    ) -> Dict[str, Any]:
        meta = self.get_dataset_meta(dataset_id)
        split, idx = self.resolve_index(dataset_id, sample_id, split=split)
        arrays = self._load_npz(dataset_id)
        x_key = f"{split}_x"
        y_key = f"{split}_y"
        if x_key not in arrays:
            raise KeyError(f"{dataset_id} missing array {x_key}")
        features = np.asarray(arrays[x_key][idx])
        out: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "sample_id": sample_id,
            "split": split,
            "index": idx,
            "features": features,
            "feature_shape": list(features.shape),
            "dtype": str(features.dtype),
            "task_family": meta["task_family"],
            "schema": meta.get("schema", {}),
        }
        if include_label:
            out["label"] = np.asarray(arrays[y_key][idx])
        return out
