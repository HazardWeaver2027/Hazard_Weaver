"""Resolve REPO_PIN paths for scientific tier checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hazardweaver.hcg.carp.expansion.paths import PIN_ROOTS, pin_manifest_path
from hazardweaver.hcg.carp.scientific.paths import PROJECT_ROOT, taskpack_data_root, taskpack_repo_pin

VENDOR_PIN_ROOTS = {
    "WF-3": PROJECT_ROOT / "data" / "vendor" / "wildfirespreadts",
    "E1-E3": PROJECT_ROOT / "data" / "vendor" / "seisbench",
    "FL-2": PROJECT_ROOT / "data" / "vendor" / "floodcastbench",
    "TC-TRK": PROJECT_ROOT / "data" / "vendor" / "tcbench",
    "HW-MED": PROJECT_ROOT / "data" / "vendor" / "ewb",
    "MH-3": PROJECT_ROOT / "data" / "vendor" / "mh3",
    "DR-OUT": PROJECT_ROOT / "data" / "vendor" / "cpc_sdo",
    "MH-1": PROJECT_ROOT / "data" / "vendor" / "ocelote_pfdf",
    "MH-1-URSA": PROJECT_ROOT / "data" / "vendor" / "ursa",
    "MH-4": PROJECT_ROOT / "data" / "vendor" / "climada",
}


def resolve_pin_path(taskpack: str, capability_id: Optional[str] = None) -> Optional[Path]:
    if taskpack == "MH-4" and capability_id:
        scientific_pin = taskpack_repo_pin(taskpack)
        if scientific_pin.is_file():
            data = json.loads(scientific_pin.read_text(encoding="utf-8"))
            if capability_id == "CAP-MH4-R06":
                climada = data.get("climada") or {}
                vendor_pin = VENDOR_PIN_ROOTS["MH-4"] / "REPO_PIN.json"
                if vendor_pin.is_file():
                    return vendor_pin
                if climada.get("repo_commit"):
                    return scientific_pin
            if capability_id in ("CAP-MH4-R02", "CAP-MH4-R03"):
                if data.get("repo_commit"):
                    return scientific_pin
            if capability_id == "CAP-MH4-R01":
                nhc = data.get("nhc_substrate") or {}
                if nhc.get("repo_commit"):
                    return scientific_pin
            if capability_id == "CAP-MH4-R04":
                hazus = data.get("hazus") or {}
                if hazus.get("repo_commit"):
                    return scientific_pin
    if taskpack == "MH-1" and capability_id == "CAP-MH1-05":
        ursa_pin = taskpack_data_root(taskpack) / "URSA_PIN.json"
        if ursa_pin.is_file():
            return ursa_pin
        vendor_ursa = VENDOR_PIN_ROOTS["MH-1-URSA"] / "REPO_PIN.json"
        if vendor_ursa.is_file():
            return vendor_ursa
    scientific_pin = taskpack_repo_pin(taskpack)
    if scientific_pin.is_file():
        return scientific_pin
    if taskpack in PIN_ROOTS:
        vendor_pin = pin_manifest_path(taskpack)
        if vendor_pin.is_file():
            return vendor_pin
    vendor_root = VENDOR_PIN_ROOTS.get(taskpack)
    if vendor_root:
        for name in ("REPO_PIN.json", "CHECKPOINT_PIN.json", "FETCH_PIN.json", "REPLAY_PIN.json"):
            candidate = vendor_root / name
            if candidate.is_file():
                return candidate
    return None


def pin_commit_ok(pin_path: Optional[Path]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    if pin_path is None or not pin_path.is_file():
        return False, "missing REPO_PIN or track-equivalent pin manifest", None
    data = json.loads(pin_path.read_text(encoding="utf-8"))
    commit = (
        data.get("repo_commit")
        or data.get("version_or_commit")
        or data.get("version")
        or data.get("repo_tag")
        or ""
    )
    if not commit or str(commit).upper() == "UNKNOWN":
        return False, "pin commit is UNKNOWN or missing", data
    return True, "ok", data
