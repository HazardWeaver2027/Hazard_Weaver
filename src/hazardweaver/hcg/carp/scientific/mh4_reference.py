"""MH-4 reference loss package (DL-051 / DL-058).

Reference semantics: event-level *reported property-damage reference* from NCEI Storm Events
(`ncei_stormevents_property_v1`). Not ground-truth economic loss — see REFERENCE_SEMANTICS.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hazardweaver.hcg.carp.scientific.paths import taskpack_data_root

TASKPACK = "MH-4"
REFERENCE_LOSS_SOURCE_ID = "ncei_stormevents_property_v1"
REFERENCE_SEMANTICS_LABEL = "event_level_reported_property_damage_reference"
REFERENCE_CROSSCHECK_ID = "nhc_tropical_cyclone_reports"
SHELDUS_SOURCE_ID = "sheldus_optional_robustness_v1"
LOSS_VARIABLE = "event_level_direct_property_damage_usd_2023"
LOSS_VARIABLE_LEGACY_ALIAS = "total_direct_building_structural_loss_usd_2023"
CURRENCY_YEAR = "USD 2023"

# Paper-side frozen splits (Atlantic + Gulf headline storms).
CALIBRATION_YEAR_RANGE = (1996, 2018)
OFFICIAL_TEST_YEAR_RANGE = (2019, 2021)
HOLDOUT_YEAR_RANGE = (2022, 2024)

OFFICIAL_TEST_EVENTS: List[Dict[str, Any]] = [
    {"nhc_id": "AL052019", "name": "DORIAN", "year": 2019},
    {"nhc_id": "AL092019", "name": "HUMBERTO", "year": 2019},
    {"nhc_id": "AL132020", "name": "LAURA", "year": 2020},
    {"nhc_id": "AL192020", "name": "SALLY", "year": 2020},
    {"nhc_id": "AL092021", "name": "IDA", "year": 2021},
    {"nhc_id": "AL082021", "name": "HENRI", "year": 2021},
]

HOLDOUT_EVENTS: List[Dict[str, Any]] = [
    {"nhc_id": "AL092022", "name": "IAN", "year": 2022},
    {"nhc_id": "AL072022", "name": "FIONA", "year": 2022},
    {"nhc_id": "AL172022", "name": "NICOLE", "year": 2022},
    {"nhc_id": "AL032023", "name": "IDALIA", "year": 2023},
    {"nhc_id": "AL052023", "name": "LEE", "year": 2023},
    {"nhc_id": "AL092024", "name": "HELENE", "year": 2024},
    {"nhc_id": "AL142024", "name": "MILTON", "year": 2024},
]

ATLANTIC_GULF_STATES = {
    "ALABAMA", "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA", "LOUISIANA",
    "MAINE", "MARYLAND", "MASSACHUSETTS", "MISSISSIPPI", "NEW HAMPSHIRE",
    "NEW JERSEY", "NEW YORK", "NORTH CAROLINA", "PENNSYLVANIA", "RHODE ISLAND",
    "SOUTH CAROLINA", "TEXAS", "VIRGINIA",
}

_DAMAGE_RE = re.compile(r"^\s*([0-9,.]+)\s*([KkMm])?\s*$")


def reference_root() -> Path:
    return taskpack_data_root(TASKPACK) / "reference"


def parse_damage_property(value: str) -> float:
    """Parse NCEI Storm Events DAMAGE_PROPERTY field to USD."""
    if not value or str(value).strip().upper() in {"", "0", "0.00K", "0.00M"}:
        return 0.0
    text = str(value).strip().replace(",", "")
    m = _DAMAGE_RE.match(text)
    if not m:
        return 0.0
    amount = float(m.group(1))
    suffix = (m.group(2) or "").upper()
    if suffix == "K":
        return amount * 1_000.0
    if suffix == "M":
        return amount * 1_000_000.0
    return amount


def load_event_reference(nhc_id: str) -> Optional[Dict[str, Any]]:
    path = reference_root() / "events" / nhc_id / "reference.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def reference_manifest() -> Optional[Dict[str, Any]]:
    path = reference_root() / "reference_manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def event_abstain_reason(row: Dict[str, Any]) -> Optional[str]:
    if row.get("abstain"):
        return str(row.get("abstain_reason") or "abstain")
    loss = row.get(LOSS_VARIABLE) or row.get(LOSS_VARIABLE_LEGACY_ALIAS)
    if loss is None and row.get("ncei_property_usd_2023") is None:
        if row.get("ncei_property_usd") is None:
            return "missing_ncei_property"
    if row.get("nhc_crosscheck") not in ("MATCHED", "ACCEPTED"):
        return f"nhc_crosscheck={row.get('nhc_crosscheck')}"
    return None


def eligible_events(split: str) -> List[Dict[str, Any]]:
    if split == "official_test":
        return list(OFFICIAL_TEST_EVENTS)
    if split == "hwb_holdout":
        return list(HOLDOUT_EVENTS)
    return []
