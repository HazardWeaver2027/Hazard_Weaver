"""Optional wall-clock guard for run_capability scientific dispatch (login/HWA agent)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from typing import Callable, TypeVar

T = TypeVar("T")


def run_capability_wall_s() -> float:
    raw = str(os.environ.get("HWA_RUN_CAPABILITY_WALL_S", "0") or "0").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def run_with_optional_wall(fn: Callable[[], T], *, label: str = "run_capability") -> T:
    wall = run_capability_wall_s()
    if wall <= 0.0:
        return fn()
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn)
        try:
            return fut.result(timeout=wall)
        except FutTimeout:
            raise TimeoutError(f"{label}_wall_timeout_after_{wall:g}s") from None
