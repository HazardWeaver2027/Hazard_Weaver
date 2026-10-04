"""MH-1 Phase C scientific data layout and readiness checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.scientific.paths import (
    hwb_holdout_manifest,
    official_test_manifest,
    scientific_signoff_path,
    taskpack_data_root,
    taskpack_repo_pin,
)

TASKPACK = "MH-1"
DATA_SOURCE = "ocelote_pfdf_official_inventory_v1"
SPEC_REF = "docs/engineering/hcg/external_expansion/MH-1/HWB_INUNDATION_HOLDOUT_SPEC.md"


def data_root() -> Path:
    return taskpack_data_root(TASKPACK)


def vendor_root() -> Path:
    return data_root().parent.parent / "vendor" / "ocelote_pfdf"


def ocelote_repo_dir() -> Path:
    return vendor_root() / "repo"


def split_protocol_path() -> Path:
    return data_root() / "SPLIT_PROTOCOL.json"


def train_split_path() -> Path:
    return data_root() / "TRAIN_SPLIT.json"


def official_test_root() -> Path:
    return data_root() / "official_test"


def hwb_holdout_root() -> Path:
    return data_root() / "hwb_holdout"


def train_pool_root() -> Path:
    return data_root() / "train_pool"


def inundation_gt_manifest_path() -> Path:
    return hwb_holdout_root() / "inundation_gt" / "manifest.json"


def record_dir(split: str, record_id: str) -> Path:
    if split == "train_pool":
        return train_pool_root() / "records" / record_id
    root = official_test_root() if split == "official_test" else hwb_holdout_root()
    return root / "records" / record_id


def load_split_manifest(split: str) -> Optional[Dict[str, Any]]:
    path = official_test_manifest(TASKPACK) if split == "official_test" else hwb_holdout_manifest(TASKPACK)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def record_ids(split: str) -> List[str]:
    manifest = load_split_manifest(split) or {}
    if split == "train_pool":
        train = json.loads(train_split_path().read_text(encoding="utf-8")) if train_split_path().is_file() else {}
        return list(train.get("record_ids") or [])
    return list(manifest.get("record_ids") or [])


def train_record_ids() -> List[str]:
    return record_ids("train_pool")


def load_record_meta(split: str, record_id: str) -> Dict[str, Any]:
    meta_path = record_dir(split, record_id) / "meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_record_row(split: str, record_id: str) -> Dict[str, Any]:
    inv_path = record_dir(split, record_id) / "inventory_row.json"
    if not inv_path.is_file():
        return {}
    return json.loads(inv_path.read_text(encoding="utf-8"))


def record_ready(split: str, record_id: str) -> bool:
    root = record_dir(split, record_id)
    return (
        (root / "inventory_row.json").is_file()
        and (root / "label" / "occurrence_label.npy").is_file()
        and (root / "meta.json").is_file()
    )


def load_record_arrays(split: str, record_id: str) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    row = load_record_row(split, record_id)
    label_dir = record_dir(split, record_id) / "label"
    occurrence = np.load(label_dir / "occurrence_label.npy")
    log_volume = np.load(label_dir / "log_volume.npy")
    return row, occurrence, log_volume


def load_split_protocol() -> Dict[str, Any]:
    path = split_protocol_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def inundation_gt_ready() -> bool:
    holdout = load_split_manifest("hwb_holdout") or {}
    if holdout.get("inundation_gt_status") == "BLOCKED":
        return False
    path = inundation_gt_manifest_path()
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return bool(data.get("segment_count", 0) > 0 and data.get("source_license_audited"))


def repo_commit() -> str:
    pin = taskpack_repo_pin(TASKPACK)
    if pin.is_file():
        data = json.loads(pin.read_text(encoding="utf-8"))
        return str(data.get("repo_commit") or data.get("version_or_commit") or "")
    return ""


def pin_version() -> str:
    return repo_commit()


def scientific_data_ready() -> bool:
    off = official_test_manifest(TASKPACK)
    hold = hwb_holdout_manifest(TASKPACK)
    sign = scientific_signoff_path(TASKPACK)
    pin = taskpack_repo_pin(TASKPACK)
    if not all(p.is_file() for p in (off, hold, sign, pin)):
        return False
    off_data = json.loads(off.read_text(encoding="utf-8"))
    if off_data.get("fixture_scale"):
        return False
    pin_data = json.loads(pin.read_text(encoding="utf-8"))
    ver = pin_data.get("repo_commit") or pin_data.get("version_or_commit") or ""
    if not ver or str(ver).upper() == "UNKNOWN":
        return False
    test_ids = record_ids("official_test")
    if not test_ids:
        return False
    ready = [rid for rid in test_ids if record_ready("official_test", rid)]
    if not ready:
        return False
    repo = ocelote_repo_dir()
    return repo.is_dir() and any(repo.iterdir())


def concat_split_arrays(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    occurrences: List[np.ndarray] = []
    log_volumes: List[np.ndarray] = []
    for rid in record_ids(split):
        if not record_ready(split, rid):
            continue
        row, occ, logv = load_record_arrays(split, rid)
        rows.append(row)
        occurrences.append(np.atleast_1d(occ))
        log_volumes.append(np.atleast_1d(logv))
    if not rows:
        raise FileNotFoundError(f"no ready records for split={split}")
    return (
        np.concatenate(occurrences),
        np.concatenate(log_volumes),
        np.array([1.0] * len(rows), dtype=np.float32),
        rows,
    )
