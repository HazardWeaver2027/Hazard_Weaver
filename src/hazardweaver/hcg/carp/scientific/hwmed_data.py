"""HW-MED Phase C scientific data layout and readiness checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from hazardweaver.hcg.carp.scientific.paths import (
    hwb_holdout_manifest,
    official_test_manifest,
    scientific_signoff_path,
    taskpack_data_root,
    taskpack_repo_pin,
)

TASKPACK = "HW-MED"
DATA_SOURCE = "ewb_official_scientific_v1"
OFFICIAL_CASE_COUNT = 46
CIRA_ARCHIVE_END = "2025-05-26"


def data_root() -> Path:
    return taskpack_data_root(TASKPACK)


def split_protocol_path() -> Path:
    return data_root() / "SPLIT_PROTOCOL.json"


def vendor_root() -> Path:
    return data_root().parent.parent / "vendor" / "ewb"


def ewb_repo_dir() -> Path:
    return vendor_root() / "repo"


def events_yaml_path() -> Path:
    vend = ewb_repo_dir() / "src" / "extremeweatherbench" / "data" / "events.yaml"
    if vend.is_file():
        return vend
    return data_root() / "events.yaml"


def holdout_cases_yaml_path() -> Path:
    return data_root() / "hwb_holdout" / "holdout_cases.yaml"


def official_test_root() -> Path:
    return data_root() / "official_test"


def hwb_holdout_root() -> Path:
    return data_root() / "hwb_holdout"


def case_dir(split: str, case_key: str) -> Path:
    root = official_test_root() if split == "official_test" else hwb_holdout_root()
    return root / "cases" / case_key


def load_split_manifest(split: str) -> Optional[Dict[str, Any]]:
    path = official_test_manifest(TASKPACK) if split == "official_test" else hwb_holdout_manifest(TASKPACK)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def case_ids(split: str) -> List[str]:
    manifest = load_split_manifest(split) or {}
    return list(
        manifest.get("case_ids")
        or manifest.get("holdout_ids")
        or manifest.get("storm_ids")
        or []
    )


def load_split_protocol() -> Dict[str, Any]:
    path = split_protocol_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def repo_commit() -> str:
    pin = taskpack_repo_pin(TASKPACK)
    if pin.is_file():
        data = json.loads(pin.read_text(encoding="utf-8"))
        return str(data.get("repo_commit") or data.get("version_or_commit") or "")
    return ""


def scientific_data_ready() -> bool:
    off = official_test_manifest(TASKPACK)
    hold = hwb_holdout_manifest(TASKPACK)
    sign = scientific_signoff_path(TASKPACK)
    pin = taskpack_repo_pin(TASKPACK)
    events = events_yaml_path()
    if not all(p.is_file() for p in (off, hold, sign, pin, events)):
        return False
    off_data = json.loads(off.read_text(encoding="utf-8"))
    if off_data.get("fixture_scale"):
        return False
    case_list = off_data.get("case_ids") or []
    if len(case_list) < 1:
        return False
    hold_data = json.loads(hold.read_text(encoding="utf-8"))
    if hold_data.get("fixture_scale"):
        return False
    if not (hold_data.get("holdout_ids") or hold_data.get("case_ids")):
        return False
    pin_data = json.loads(pin.read_text(encoding="utf-8"))
    commit = pin_data.get("repo_commit") or pin_data.get("version_or_commit") or ""
    if not commit or str(commit).upper() == "UNKNOWN":
        return False
    return True
