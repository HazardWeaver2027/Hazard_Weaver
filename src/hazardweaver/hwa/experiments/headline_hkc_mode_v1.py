"""Headline HKC mode — off (benchmark-curated) vs enhance (Route Cards). DL-143/DL-145."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping

HKC_MODE_OFF = "off"
HKC_MODE_ENHANCE = "enhance"
_DEFAULT_MODE = HKC_MODE_OFF


def headline_hkc_mode() -> str:
    raw = str(
        os.environ.get("HWA_HEADLINE_HKC_MODE")
        or os.environ.get("ICLR_HWA_HKC_MODE")
        or _DEFAULT_MODE
    ).strip().lower()
    if raw in {HKC_MODE_ENHANCE, "on", "1", "true", "yes"}:
        return HKC_MODE_ENHANCE
    return HKC_MODE_OFF


def headline_hkc_off(task: Mapping[str, Any] | None = None) -> bool:
    if headline_hkc_mode() != HKC_MODE_OFF:
        return False
    if task is None:
        return True
    meta = task.get("metadata") or {}
    return bool(meta.get("hwb_headline_inventory") or meta.get("same_llm_g6_coreexec"))


def is_headline_inventory_task(task: Mapping[str, Any]) -> bool:
    meta = task.get("metadata") or {}
    return bool(meta.get("hwb_headline_inventory") or meta.get("same_llm_g6_coreexec"))


def headline_run_provenance(
    task: Mapping[str, Any],
    controller: Any | None = None,
) -> Dict[str, Any]:
    """Audit fields for run_meta (DL-143 historical HKC mode audit)."""
    from hazardweaver.hwa.runtime.runtime_closure import methodology_closure_flags

    out: Dict[str, Any] = {
        "hkc_mode": headline_hkc_mode(),
        "methodology_closure": methodology_closure_flags(task),
    }
    if controller is not None:
        gate = getattr(controller, "gate", None)
        out["route_card_loaded"] = bool(
            gate is not None
            and (
                getattr(gate, "route_card_index", None) is not None
                or getattr(gate, "contract_index", None) is not None
            )
        )
        state = getattr(controller, "state", None)
        if state is not None:
            out["theory_arm"] = str(getattr(state, "theory_arm", "") or "")
    return out
