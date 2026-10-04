"""P0-7 arm-level llm_provider normalization (single source of truth).

Maps transport ``vllm`` + profile/model_id → ``vllm_7b`` | ``vllm_fp8`` | ``vllm_72b``.
Claim Guard: correct provider ≠ STRENGTH.
"""

from __future__ import annotations

import os
from typing import Any, Optional

VLLM_PROVIDERS = frozenset({"vllm_7b", "vllm_fp8", "vllm_72b"})
CLOSED_API_PROVIDERS = frozenset({"minimax_m27", "deepseek_v4_flash"})


def normalize_llm_provider(
    raw: Any = None,
    *,
    profile: Optional[str] = None,
    model_id: Optional[str] = None,
) -> str:
    """Normalize to an arm-level provider tag."""
    p = str(raw or "").strip().lower()
    if p in VLLM_PROVIDERS or p in CLOSED_API_PROVIDERS:
        return p

    closed_provider = os.environ.get("HW_CLOSED_API_PROVIDER", "").strip().lower()
    if closed_provider == "minimax":
        return "minimax_m27"
    if closed_provider == "deepseek_v4_flash":
        return "deepseek_v4_flash"
    tag = os.environ.get("HW_CLOSED_API_PROVIDER_TAG", "").strip()
    if tag in CLOSED_API_PROVIDERS:
        return tag

    prof = (
        profile
        or os.environ.get("HW_LLM_PROFILE")
        or ""
    ).strip().lower()
    mid = (model_id or os.environ.get("VLLM_MODEL_ID") or "").strip().lower()

    # Profile wins when explicit
    if "72b" in prof or prof == "w1_72b_compare":
        return "vllm_72b"
    if "fp8" in prof or "b200" in prof or "primary" in prof:
        return "vllm_fp8"
    if "l4" in prof or "7b" in prof or prof in {"l4_fast", "coder_7b"}:
        return "vllm_7b"

    # model_id hints
    if "72b" in mid:
        return "vllm_72b"
    if "fp8" in mid:
        return "vllm_fp8"
    if "7b" in mid or "coder-7b" in mid:
        return "vllm_7b"

    if p in {"vllm", "vllm_openai", ""}:
        # Safe default for unspecified local vLLM = 7B arm
        return "vllm_7b"
    if p.startswith("vllm"):
        return "vllm_7b"
    return p or "vllm_7b"
