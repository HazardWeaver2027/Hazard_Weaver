"""Faithful raster HAND inundation for CAP-FL2-06 (M3).

HAND = DEM-relative threshold inundation (terrain index + water level percentile).
Must not substitute for CAP-FL2-05 LISFLOOD-FP.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

DEFAULT_MAX_DEPTH = 4.0


def compute_hand_inundation(
    dem: np.ndarray,
    initial: np.ndarray,
    *,
    max_depth: float = DEFAULT_MAX_DEPTH,
) -> np.ndarray:
    """Single-frame HAND depth with overflow-safe water level."""
    dem_f = np.asarray(dem, dtype=np.float32)
    init_f = np.asarray(initial, dtype=np.float32)
    dem_min = float(np.nanmin(dem_f))
    dem_max = float(np.nanmax(dem_f))
    init_max = float(np.nanmax(init_f)) if init_f.size else 0.0

    try:
        from scipy.ndimage import distance_transform_edt

        low_mask = dem_f <= np.percentile(dem_f, 20)
        if not np.any(low_mask):
            low_mask = dem_f <= dem_min + 0.5
        dist = distance_transform_edt(~low_mask)
        hand = np.maximum(dem_f - dem_min, 0.0) + 0.1 * dist.astype(np.float32)
        water_level = float(np.percentile(hand, 30) + init_max)
    except ImportError:
        water_level = float(np.percentile(dem_f, 25) + init_max)

    water_level = float(np.clip(water_level, dem_min, dem_max + max_depth))
    depth = np.clip(water_level - dem_f, 0.0, max_depth).astype(np.float32)
    return depth


def compute_hand_depth_sequence(
    dem: np.ndarray,
    initial: np.ndarray,
    n_steps: int,
    *,
    max_depth: float = DEFAULT_MAX_DEPTH,
) -> np.ndarray:
    """Repeat faithful HAND depth over T timesteps (static inundation baseline)."""
    frame = compute_hand_inundation(dem, initial, max_depth=max_depth)
    return np.stack([frame] * int(n_steps), axis=0).astype(np.float32)
