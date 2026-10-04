"""Load HW-registered capabilities for inference-only execution."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Type

from hazardweaver.hwa.capabilities.registry import CapabilityRegistry, default_registry_path
from hazardweaver.hwa.contracts import CapabilityCard, CapabilityKind
from hazardweaver.hwa.contracts.predictor import BasePredictor, CapabilityNotReady


def _import_class(entrypoint: str) -> Type[Any]:
    if ":" not in entrypoint:
        raise ValueError(f"Invalid entrypoint (expected module:Class): {entrypoint}")
    module_name, class_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class CapabilityLoader:
    """Resolve capability_id -> inference-ready predictor."""

    def __init__(
        self,
        registry: Optional[CapabilityRegistry] = None,
        *,
        device: str = "cpu",
        require_checkpoint: bool = True,
    ):
        self.registry = registry or CapabilityRegistry.load_yaml(default_registry_path())
        self.device = device
        self.require_checkpoint = require_checkpoint
        self._cache: Dict[str, BasePredictor] = {}

    def get_card(self, capability_id: str) -> CapabilityCard:
        return self.registry.get(capability_id)

    def load(self, capability_id: str, *, force_reload: bool = False) -> BasePredictor:
        if not force_reload and capability_id in self._cache:
            return self._cache[capability_id]

        card = self.get_card(capability_id)
        cls = _import_class(card.execution.entrypoint)
        ckpt = card.resolved_checkpoint()

        if card.kind == CapabilityKind.DOMAIN_FORMULA:
            predictor = cls()
        elif card.kind == CapabilityKind.PREDICTIVE_MODEL:
            if self.require_checkpoint and ckpt and not Path(ckpt).exists():
                raise CapabilityNotReady(
                    f"Capability {capability_id} missing checkpoint at {ckpt}"
                )
            if hasattr(cls, "from_card"):
                predictor = cls.from_card(card, device=self.device)
            elif ckpt and Path(ckpt).exists():
                predictor = cls.load_checkpoint(ckpt, device=self.device)
            else:
                # Allow uninitialized model for smoke tests when require_checkpoint=False
                if self.require_checkpoint and card.metadata.get("requires_checkpoint", True):
                    raise CapabilityNotReady(
                        f"Capability {capability_id} has no checkpoint_uri"
                    )
                predictor = cls()
        else:
            predictor = cls()

        if not isinstance(predictor, BasePredictor):
            # Wrap legacy adapters at runtime if needed
            predictor = _LegacyAdapterWrapper(card.capability_id, predictor)

        self._cache[capability_id] = predictor
        return predictor


class _LegacyAdapterWrapper(BasePredictor):
    """Thin wrapper for pre-protocol model classes."""

    def __init__(self, capability_id: str, inner: Any):
        self.capability_id = capability_id
        self.inner = inner

    def predict(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if hasattr(self.inner, "predict_summary"):
            oracle = kwargs.get("oracle", False)
            out = self.inner.predict_summary(record, oracle=oracle)
            return {
                "watershed_summary": out.watershed_summary,
                "valid": out.valid,
                "model_id": out.model_id,
            }
        if hasattr(self.inner, "predict_row"):
            out = self.inner.predict_row(record)
            return {
                "log_volume": out.log_volume,
                "valid": out.valid,
                "model_id": out.model_id,
            }
        if hasattr(self.inner, "predict"):
            return self.inner.predict(record, **kwargs)
        raise TypeError(f"Cannot wrap {type(self.inner)} for inference")
