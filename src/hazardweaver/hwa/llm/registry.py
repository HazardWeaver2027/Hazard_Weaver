"""Load models/llm/registry.json — roles for HWB/HW vLLM profiles."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = PROJECT_ROOT / "models" / "llm" / "registry.json"


class LLMRegistry:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or DEFAULT_REGISTRY)
        if not self.path.is_file():
            raise FileNotFoundError(f"LLM registry missing: {self.path}")
        self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def models(self) -> List[Dict[str, Any]]:
        return list(self.data.get("models", []))

    def get(self, role: str) -> Dict[str, Any]:
        for m in self.models():
            if m.get("role") == role:
                return dict(m)
        raise KeyError(f"unknown LLM role: {role}")

    def iteration_model(self) -> Dict[str, Any]:
        """Default for everyday testing: L4 7B."""
        return self.get("l4_fast")

    def primary_model(self) -> Dict[str, Any]:
        """Paper / batch: B200 FP8."""
        return self.get("primary_b200")


@lru_cache(maxsize=1)
def default_registry() -> LLMRegistry:
    return LLMRegistry()
