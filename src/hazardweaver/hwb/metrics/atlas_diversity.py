"""Atlas diversity metrics — defend 3062-scale claims (v3 §7)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ATLAS = ROOT / "hwb/manifests/HWB_ATLAS_MANIFEST_v1.jsonl"


def load_atlas_manifest(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    src = path or DEFAULT_ATLAS
    if not src.is_file():
        return []
    return [
        json.loads(line)
        for line in src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compute_atlas_diversity(
    rows: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    atlas_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Coverage matrix + effective diversity ratio for Atlas pool."""
    data = list(rows) if rows is not None else load_atlas_manifest(atlas_path)
    if not data:
        return {
            "n_atlas": 0,
            "n_tracks": 0,
            "n_unique_scenarios": 0,
            "effective_diversity_ratio": 0.0,
            "coverage_matrix": {},
        }

    by_track: Dict[str, int] = defaultdict(int)
    scenarios: set[str] = set()
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in data:
        track = str(row.get("track") or "unknown")
        scenario = str(row.get("scenario_id") or "")
        split_role = str(row.get("split_role") or "unknown")
        tier = str(row.get("tier") or "atlas")
        by_track[track] += 1
        scenarios.add(f"{track}:{scenario}")
        matrix[track][f"{split_role}:{tier}"] += 1

    n_atlas = len(data)
    n_unique = len(scenarios)
    ratio = n_unique / n_atlas if n_atlas else 0.0

    return {
        "n_atlas": n_atlas,
        "n_tracks": len(by_track),
        "n_unique_scenarios": n_unique,
        "effective_diversity_ratio": round(ratio, 4),
        "per_track_counts": dict(sorted(by_track.items())),
        "coverage_matrix": {t: dict(cells) for t, cells in sorted(matrix.items())},
        "hwb_scoreable_count": sum(1 for r in data if r.get("hwb_scoreable")),
        "materialized_count": sum(1 for r in data if r.get("materialized")),
    }
