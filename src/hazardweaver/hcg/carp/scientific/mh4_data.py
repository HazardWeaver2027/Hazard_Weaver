"""MH-4 Phase C acquisition / scientific data readiness (DL-039 / DL-051 / DL-052)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.scientific.mh4_reference import (
    REFERENCE_LOSS_SOURCE_ID,
    event_abstain_reason,
    load_event_reference,
    reference_manifest,
)
from hazardweaver.hcg.carp.scientific.paths import (
    official_test_manifest,
    hwb_holdout_manifest,
    scientific_signoff_path,
    taskpack_data_root,
)

TASKPACK = "MH-4"
NHC_SUBSTRATE_MANIFEST = "nhc_substrate_manifest.json"


def acquisition_contract_path() -> Path:
    return taskpack_data_root(TASKPACK) / "ACQUISITION_CONTRACT.json"


def load_acquisition_contract() -> Optional[Dict[str, Any]]:
    path = acquisition_contract_path()
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def acquisition_contract_ready() -> bool:
    data = load_acquisition_contract()
    if not data:
        return False
    return data.get("gate_a") == "PAPER_APPROVED_FOR_ACQUISITION"


def reference_loss_ready() -> bool:
    manifest = reference_manifest()
    if not manifest:
        return False
    if manifest.get("reference_loss_source_id") != REFERENCE_LOSS_SOURCE_ID:
        return False
    return int(manifest.get("n_eligible") or 0) > 0


def nhc_substrate_ready() -> bool:
    path = taskpack_data_root(TASKPACK) / NHC_SUBSTRATE_MANIFEST
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data.get("n_events_pinned") or 0) > 0


def _eligible_in_split(split: str) -> int:
    path = official_test_manifest(TASKPACK) if split == "official_test" else hwb_holdout_manifest(TASKPACK)
    if not path.is_file():
        return 0
    manifest = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for nhc_id in manifest.get("event_ids") or []:
        ref = load_event_reference(str(nhc_id))
        if ref and not event_abstain_reason(ref):
            count += 1
    return count


def scientific_data_ready() -> bool:
    """Gate B: reference + splits + signoff + NHC substrate manifest."""
    if not acquisition_contract_ready():
        return False
    if not reference_loss_ready():
        return False
    if not nhc_substrate_ready():
        return False
    off = official_test_manifest(TASKPACK)
    hold = hwb_holdout_manifest(TASKPACK)
    signoff = scientific_signoff_path(TASKPACK)
    if not off.is_file() or not hold.is_file() or not signoff.is_file():
        return False
    off_m = json.loads(off.read_text(encoding="utf-8"))
    hold_m = json.loads(hold.read_text(encoding="utf-8"))
    if off_m.get("fixture_scale") is True or hold_m.get("fixture_scale") is True:
        return False
    if not off_m.get("event_ids") or not hold_m.get("event_ids"):
        return False
    if not off_m.get("checksum") or not hold_m.get("checksum"):
        return False
    sig = json.loads(signoff.read_text(encoding="utf-8"))
    if not sig.get("approved") or sig.get("approval_tier") != "scientific":
        return False
    contract = load_acquisition_contract() or {}
    if contract.get("reference_loss_source_id") in (None, "", "PI_REQUIRED"):
        return False
    if _eligible_in_split("official_test") < 1 or _eligible_in_split("hwb_holdout") < 1:
        return False
    return contract.get("gate_b") == "SCIENTIFIC_READY"


def load_split_manifest(split: str) -> Optional[Dict[str, Any]]:
    path = official_test_manifest(TASKPACK) if split == "official_test" else hwb_holdout_manifest(TASKPACK)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
