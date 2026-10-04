"""HWA control mode switch (legacy ReAct loop vs Verified Commit Executor)."""

from __future__ import annotations

import os

_VALID_MODES = frozenset({"legacy", "vce"})


def hwa_control_mode() -> str:
    """Return active control mode from ``HWA_CONTROL_MODE`` (default ``legacy``)."""
    raw = str(os.environ.get("HWA_CONTROL_MODE") or "legacy").strip().lower()
    if raw in {"", "off", "0", "false", "legacy"}:
        return "legacy"
    if raw in {"vce", "on", "1", "true"}:
        return "vce"
    if raw in _VALID_MODES:
        return raw
    return "legacy"


def is_vce_mode() -> bool:
    return hwa_control_mode() == "vce"
