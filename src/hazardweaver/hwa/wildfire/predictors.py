"""Load HWA wildfire checkpoints and run inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from hazardweaver.hwa.wildfire.models import build_arch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "hw_wildfire" / "manifests" / "models.json"


class WildfirePredictorRegistry:
    def __init__(self, manifest_path: Path | None = None, *, device: str = "cpu"):
        self.manifest_path = Path(manifest_path or DEFAULT_MANIFEST)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"HW wildfire model manifest missing: {self.manifest_path}. "
                "Run scripts/bootstrap/wildfire_hw_assets.py"
            )
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.device = torch.device(device)
        self._models: Dict[str, torch.nn.Module] = {}

    def list_model_ids(self) -> List[str]:
        return sorted(self.manifest["models"].keys())

    def get_meta(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self.manifest["models"]:
            raise KeyError(f"unknown model_id: {model_id}")
        return dict(self.manifest["models"][model_id])

    def load(self, model_id: str) -> torch.nn.Module:
        if model_id in self._models:
            return self._models[model_id]
        meta = self.get_meta(model_id)
        ckpt_path = PROJECT_ROOT / meta["checkpoint_path"]
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_arch(payload["arch"], **payload.get("arch_kwargs", {}))
        model.load_state_dict(payload["state_dict"])
        model.to(self.device)
        model.eval()
        self._models[model_id] = model
        return model

    def predict(
        self,
        model_id: str,
        features: np.ndarray,
        *,
        dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        meta = self.get_meta(model_id)
        if dataset_id and dataset_id not in meta.get("compatible_datasets", []):
            raise ValueError(
                f"model {model_id} incompatible with dataset {dataset_id}; "
                f"allowed={meta.get('compatible_datasets')}"
            )

        model = self.load(model_id)
        x = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=self.device)
        expected = list(meta.get("example_feature_shape", []))
        if expected:
            if list(x.shape) == expected:
                x = x.unsqueeze(0)
            elif list(x.shape) != [1, *expected] and list(x.shape[1:]) != expected:
                raise ValueError(
                    f"feature shape {list(x.shape)} incompatible with model {model_id} "
                    f"expected {expected} (or batched {[1, *expected]})"
                )

        with torch.no_grad():
            logits = model(x)
        logits_np = logits.detach().cpu().numpy()
        result: Dict[str, Any] = {
            "model_id": model_id,
            "task_family": meta["task_family"],
            "output_kind": meta["output_kind"],
            "logits": logits_np,
            "logits_shape": list(logits_np.shape),
        }
        if meta["output_kind"] == "class_logits":
            flat = logits_np.reshape(logits_np.shape[0], -1)[0]
            result["pred_class_id"] = int(np.argmax(flat))
            result["probs"] = _softmax(flat).tolist()
        elif meta["output_kind"] == "regression_vector":
            result["prediction"] = logits_np.reshape(logits_np.shape[0], -1)[0].tolist()
        elif meta["output_kind"] == "segmentation_logits":
            mask = (logits_np[0, 0] > 0).astype(np.float32)
            result["pred_mask"] = mask
            result["burned_fraction"] = float(mask.mean())
            result["burned_pixels"] = int(mask.sum())
        return result


def _softmax(v: np.ndarray) -> np.ndarray:
    z = v - np.max(v)
    e = np.exp(z)
    return e / e.sum()
