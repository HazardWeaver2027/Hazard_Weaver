"""TC-TRK linear post-processing train (engineering)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TCBENCH_SMALL = PROJECT_ROOT / "data" / "raw" / "tc_tcbench_small_v1"
CHECKPOINT_ROOT = PROJECT_ROOT / "runs" / "carp" / "batch2" / "TC-TRK" / "CAP-TCTRK-06"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _load_positions(csv_path: Path) -> List[Tuple[float, float]]:
    rows: List[Tuple[float, float]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row.get("LAT") or row.get("lat") or row.get("latitude") or "nan")
                lon = float(row.get("LON") or row.get("lon") or row.get("longitude") or "nan")
            except ValueError:
                continue
            if math.isfinite(lat) and math.isfinite(lon):
                rows.append((lat, lon))
    return rows


def train_linear() -> Dict[str, Any]:
    ibtracs = TCBENCH_SMALL / "2023_IBTrACS.csv"
    if not ibtracs.is_file():
        raise FileNotFoundError(f"missing {ibtracs}")
    positions = _load_positions(ibtracs)
    if len(positions) < 3:
        raise ValueError("insufficient IBTrACS positions")

    features: List[List[float]] = []
    targets: List[List[float]] = []
    for i in range(1, len(positions) - 1):
        prev = positions[i - 1]
        curr = positions[i]
        truth = positions[i + 1]
        vel_lat = curr[0] - prev[0]
        vel_lon = curr[1] - prev[1]
        features.append([curr[0], curr[1], vel_lat, vel_lon, 1.0])
        targets.append([truth[0] - curr[0], truth[1] - curr[1]])

    x = np.array(features, dtype=np.float64)
    y = np.array(targets, dtype=np.float64)
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    meta = {
        "capability_id": "CAP-TCTRK-06",
        "coef_shape": list(coef.shape),
        "n_pairs": len(features),
        "status": "trained",
        "note": "linear bias correction on persistence residuals",
    }
    np.save(CHECKPOINT_ROOT / "linear_coef.npy", coef)
    (CHECKPOINT_ROOT / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    print(json.dumps(train_linear(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
