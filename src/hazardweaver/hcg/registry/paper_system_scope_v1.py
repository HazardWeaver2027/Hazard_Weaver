"""Paper-system scope — 11 TaskPacks, 66-cap denominator (DL-121).

Supersedes ad-hoc **seven-track / 43-cap** reporting as the primary paper denominator.
Pilot-4 (FL-2, TC-TRK, DR-OUT, MH-1) and completion-7 share one audit surface.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parents[3]
_ASSETS_CSV = _REPO / "docs" / "engineering" / "hcg" / "HAZARDWEAVER_MODEL_ROUTE_ASSETS_v1.csv"

PILOT_FOUR_TRACKS: Tuple[str, ...] = ("FL-2", "TC-TRK", "DR-OUT", "MH-1")
COMPLETION_SEVEN_TRACKS: Tuple[str, ...] = (
    "WF-3",
    "L2",
    "E1-E3",
    "HW-MED",
    "MH-2",
    "MH-3",
    "MH-4",
)
ELEVEN_TRACKS: Tuple[str, ...] = PILOT_FOUR_TRACKS + COMPLETION_SEVEN_TRACKS

# Paper denominator: 65 formal manifest CAP-* + 1 route-registry bridge (HCG_MANIFEST_DRIFT_v1).
PAPER_SYSTEM_CAP_DENOMINATOR = 66
FORMAL_MANIFEST_CAP_COUNT_11TRACK = 65
REGISTRY_BRIDGE_CAPABILITY_IDS: Tuple[str, ...] = ("wf3_learned_eo_spread_v1",)

TRACK_SLUG: Dict[str, str] = {
    "FL-2": "fl2",
    "TC-TRK": "tctrk",
    "DR-OUT": "drout",
    "MH-1": "mh1",
    "WF-3": "wf-3",
    "L2": "l2",
    "E1-E3": "e1-e3",
    "HW-MED": "hw-med",
    "MH-2": "mh-2",
    "MH-3": "mh-3",
    "MH-4": "mh-4",
}

PI_SIGNED_ABSTAIN_CAPS: Tuple[str, ...] = (
    "CAP-MH2-06",
    "CAP-MH3-02",
    "CAP-MH3-03",
    "CAP-MH3-04",
    "CAP-MH4-R04",
    "CAP-MH4-R05",
)


@lru_cache(maxsize=1)
def formal_cap_ids_11track() -> List[str]:
    """CAP-* headline rows across all 11 TaskPacks (expect 65)."""
    ids: List[str] = []
    with _ASSETS_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            track = (row.get("track_task") or "").strip()
            if track not in ELEVEN_TRACKS:
                continue
            if (row.get("headline_non_headline") or "headline").strip().lower() not in ("", "headline"):
                continue
            cap = (row.get("current_capability_id") or row.get("asset_id") or "").strip()
            if cap.startswith("CAP-"):
                ids.append(cap)
    ids.sort()
    return ids


def caps_per_track() -> Dict[str, int]:
    counts: Dict[str, int] = {t: 0 for t in ELEVEN_TRACKS}
    with _ASSETS_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            track = (row.get("track_task") or "").strip()
            if track not in counts:
                continue
            if (row.get("headline_non_headline") or "headline").strip().lower() not in ("", "headline"):
                continue
            cap = (row.get("current_capability_id") or "").strip()
            if cap.startswith("CAP-"):
                counts[track] += 1
    return counts
