"""MH-3 Phase C scientific data layout and readiness checks."""

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

TASKPACK = "MH-3"
DATA_SOURCE = "sfincs_official_scientific_v1"


def data_root() -> Path:
    return taskpack_data_root(TASKPACK)


def split_protocol_path() -> Path:
    return data_root() / "SPLIT_PROTOCOL.json"


def train_split_path() -> Path:
    return data_root() / "TRAIN_SPLIT.json"


def vendor_root() -> Path:
    return data_root().parent.parent / "vendor" / "sfincs"


def sfincs_repo_dir() -> Path:
    return vendor_root() / "repo"


def official_test_root() -> Path:
    return data_root() / "official_test"


def hwb_holdout_root() -> Path:
    return data_root() / "hwb_holdout"


def train_root() -> Path:
    return data_root() / "train"


def scenario_dir(split: str, scenario_id: str) -> Path:
    if split == "official_test":
        root = official_test_root()
    elif split == "hwb_holdout":
        root = hwb_holdout_root()
    elif split == "train":
        root = train_root()
    else:
        root = data_root() / split
    return root / "scenarios" / scenario_id


def load_scenario_bundle(split: str, scenario_id: str) -> Optional[Dict[str, Any]]:
    scen_dir = scenario_dir(split, scenario_id)
    meta_path = scen_dir / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    files = meta.get("files") or {}
    arrays = {}
    for key, fname in files.items():
        fp = scen_dir / fname
        if fp.is_file():
            import numpy as np

            arrays[key] = np.load(fp)
    return {"meta": meta, "dir": scen_dir, "arrays": arrays}


def official_scenario_id() -> str:
    ids = scenario_ids("official_test")
    return ids[0] if ids else "charleston_official"


def sfincs_model_dir(scenario_id: Optional[str] = None) -> Path:
    sid = scenario_id or official_scenario_id()
    return scenario_dir("official_test", sid) / "sfincs_model"


def load_split_manifest(split: str) -> Optional[Dict[str, Any]]:
    path = official_test_manifest(TASKPACK) if split == "official_test" else hwb_holdout_manifest(TASKPACK)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def scenario_ids(split: str) -> List[str]:
    manifest = load_split_manifest(split) or {}
    return list(
        manifest.get("scenario_ids")
        or manifest.get("holdout_ids")
        or manifest.get("event_ids")
        or []
    )


def load_train_split() -> Dict[str, Any]:
    path = train_split_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def train_scenario_ids() -> List[str]:
    data = load_train_split()
    return list(data.get("scenario_ids") or [])


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
    train = train_split_path()
    if not all(p.is_file() for p in (off, hold, sign, pin, train)):
        return False
    off_data = json.loads(off.read_text(encoding="utf-8"))
    if off_data.get("fixture_scale"):
        return False
    if not (off_data.get("scenario_ids") or off_data.get("event_ids")):
        return False
    hold_data = json.loads(hold.read_text(encoding="utf-8"))
    if hold_data.get("fixture_scale"):
        return False
    if not (hold_data.get("holdout_ids") or hold_data.get("scenario_ids")):
        return False
    pin_data = json.loads(pin.read_text(encoding="utf-8"))
    commit = pin_data.get("repo_commit") or pin_data.get("version_or_commit") or ""
    if not commit or str(commit).upper() == "UNKNOWN":
        return False
    train_data = json.loads(train.read_text(encoding="utf-8"))
    if not train_data.get("scenario_ids"):
        return False
    off_ids = set(scenario_ids("official_test"))
    train_ids = set(train_scenario_ids())
    if off_ids.intersection(train_ids):
        return False
    return True
