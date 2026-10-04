"""SeisBench vendor roots and cache on /blue project storage (never $HOME)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SEISBENCH_VENDOR_ROOT = PROJECT_ROOT / "data" / "vendor" / "seisbench"
SEISBENCH_CACHE_DIR = SEISBENCH_VENDOR_ROOT / "cache"

# Legacy user-level cache (HPG policy: do not write new data here).
LEGACY_HOME_CACHE = Path.home() / ".seisbench"
LEGACY_BLUE_USER_CACHE = Path("/workspace/hazardweaver")


def resolve_seisbench_cache_dir() -> Path:
    env = os.environ.get("SEISBENCH_CACHE_DIR", "").strip()
    if env:
        return Path(env)
    return SEISBENCH_CACHE_DIR


def apply_seisbench_cache_env() -> Path:
    """Ensure SEISBENCH_CACHE_DIR points at project blue storage."""
    cache = resolve_seisbench_cache_dir()
    os.environ["SEISBENCH_CACHE_DIR"] = str(cache)
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def seisbench_subprocess_env() -> Dict[str, str]:
    cache = apply_seisbench_cache_env()
    return {**os.environ, "SEISBENCH_CACHE_DIR": str(cache)}


def vendor_store_dir(store_name: str) -> Path:
    return SEISBENCH_VENDOR_ROOT / store_name
