"""Headline route profile — G0 hybrid vs G1 CBR (env-driven, no solver gold)."""

from __future__ import annotations

import os
from pathlib import Path

G1_PROFILE = "g1_v1"
G1_FROZEN_DL142 = True  # cross-system DS import invalid for paper (see DL-142)

_DEFAULT_DS_CBR_REL = "runs/benchmark/iclr_ds_agent_full218_ollama_minimax27_v1"


def cross_system_cbr_allowed() -> bool:
    """Debug-only gate for invalidated G1 arm (DL-142)."""
    return str(os.environ.get("HWA_ALLOW_CROSS_SYSTEM_CBR", "")).lower() in (
        "1",
        "true",
        "yes",
    )


def headline_route_profile() -> str:
    return str(
        os.environ.get("HWA_HEADLINE_ROUTE_PROFILE")
        or os.environ.get("ICLR_HWA_ROUTE_PROFILE")
        or ""
    ).strip()


def is_g1_profile(profile: str | None = None) -> bool:
    return (profile or headline_route_profile()) == G1_PROFILE


def default_ds_cbr_out_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    env = os.environ.get("HWA_DS_CBR_OUT_ROOT") or os.environ.get("ICLR_DS_CBR_OUT_ROOT")
    if env:
        return Path(env)
    return root / _DEFAULT_DS_CBR_REL
