"""Unified HWA capability predictor protocol (inference only)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class CapabilityNotReady(RuntimeError):
    """Raised when a capability requires a checkpoint or asset that is missing."""


class BasePredictor(ABC):
    capability_id: str

    @abstractmethod
    def predict(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Return structured prediction outputs for a single inventory record."""


class BurnStatePredictor(BasePredictor):
    """Burn-state capability output contract."""

    def predict_summary(self, record: Dict[str, Any], *, oracle: bool = False) -> Dict[str, Any]:
        out = self.predict(record, oracle=oracle)
        return out.get("watershed_summary", out)


class VolumePredictor(BasePredictor):
    """Volume capability output contract."""

    def predict_volume(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return self.predict(record)
