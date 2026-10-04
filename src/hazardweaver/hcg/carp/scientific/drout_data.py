"""DR-OUT Phase C scientific data layout and readiness checks."""

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

TASKPACK = "DR-OUT"
DATA_SOURCE = "cpc_sdo_official_archive_v1"
TRAIN_YEARS = (2015, 2016, 2017, 2018)
OFFICIAL_TEST_YEARS = (2019, 2020, 2021)
HOLDOUT_YEARS = (2022, 2023)
CAUSALITY_MAX_YEAR = 2023

SPEC_REF = "docs/engineering/hcg/external_expansion/DR-OUT/HWB_DR_OUT_DROUGHT_HOLDOUT_SPEC.md"
CAUSALITY_AUDIT_PATH_REL = "causality/ISSUE_TIME_CAUSALITY_AUDIT.json"


def data_root() -> Path:
    return taskpack_data_root(TASKPACK)


def vendor_root() -> Path:
    return data_root().parent.parent / "vendor" / "cpc_sdo"


def split_protocol_path() -> Path:
    return data_root() / "SPLIT_PROTOCOL.json"


def train_split_path() -> Path:
    return data_root() / "TRAIN_SPLIT.json"


def causality_audit_path() -> Path:
    return data_root() / CAUSALITY_AUDIT_PATH_REL


def official_test_root() -> Path:
    return data_root() / "official_test"


def hwb_holdout_root() -> Path:
    return data_root() / "hwb_holdout"


def train_pool_root() -> Path:
    return data_root() / "train_pool"


def month_dir(split: str, issue_yyyymm: str) -> Path:
    if split == "train_pool":
        return train_pool_root() / "months" / issue_yyyymm
    root = official_test_root() if split == "official_test" else hwb_holdout_root()
    return root / "months" / issue_yyyymm


def load_split_manifest(split: str) -> Optional[Dict[str, Any]]:
    path = official_test_manifest(TASKPACK) if split == "official_test" else hwb_holdout_manifest(TASKPACK)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def issue_months(split: str) -> List[str]:
    manifest = load_split_manifest(split) or {}
    if split == "official_test":
        return list(manifest.get("issue_months") or [])
    return list(manifest.get("holdout_ids") or manifest.get("issue_months") or [])


def train_issue_months() -> List[str]:
    path = train_split_path()
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("issue_months") or [])
    protocol = load_split_protocol()
    return list(protocol.get("train_issue_months") or [])


def load_split_protocol() -> Dict[str, Any]:
    path = split_protocol_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_month_meta(split: str, issue_yyyymm: str) -> Dict[str, Any]:
    meta_path = month_dir(split, issue_yyyymm) / "meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def month_ingest_ready(split: str, issue_yyyymm: str) -> bool:
    ingest = month_dir(split, issue_yyyymm) / "ingest"
    meta = load_month_meta(split, issue_yyyymm)
    if not meta.get("ingest_ready"):
        return False
    required = ("sdo_category.npy", "usdm_category.npy")
    return all((ingest / name).is_file() for name in required)


def load_causality_audit() -> Dict[str, Any]:
    path = causality_audit_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def month_causality_pass(issue_yyyymm: str) -> bool:
    audit = load_causality_audit()
    for row in audit.get("months") or []:
        if row.get("issue_yyyymm") == issue_yyyymm:
            return row.get("overall") == "pass"
    return False


def pin_version() -> str:
    pin = taskpack_repo_pin(TASKPACK)
    if pin.is_file():
        data = json.loads(pin.read_text(encoding="utf-8"))
        return str(data.get("version_or_commit") or data.get("repo_commit") or "")
    return ""


def scientific_data_ready() -> bool:
    off = official_test_manifest(TASKPACK)
    hold = hwb_holdout_manifest(TASKPACK)
    sign = scientific_signoff_path(TASKPACK)
    pin = taskpack_repo_pin(TASKPACK)
    audit = causality_audit_path()
    if not all(p.is_file() for p in (off, hold, sign, pin, audit)):
        return False
    off_data = json.loads(off.read_text(encoding="utf-8"))
    if off_data.get("fixture_scale"):
        return False
    pin_data = json.loads(pin.read_text(encoding="utf-8"))
    ver = pin_data.get("version_or_commit") or pin_data.get("repo_commit") or ""
    if not ver or str(ver).upper() == "UNKNOWN":
        return False
    test_months = issue_months("official_test")
    if not test_months:
        return False
    ready_months = [m for m in test_months if month_ingest_ready("official_test", m)]
    if not ready_months:
        return False
    passed = [m for m in ready_months if month_causality_pass(m)]
    return len(passed) >= 1


def load_month_arrays(split: str, issue_yyyymm: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ingest = month_dir(split, issue_yyyymm) / "ingest"
    sdo = np.load(ingest / "sdo_category.npy")
    usdm = np.load(ingest / "usdm_category.npy")
    obj_path = ingest / "objective_tendency.npy"
    objective = np.load(obj_path) if obj_path.is_file() else np.zeros_like(sdo)
    return sdo, usdm, objective


def concat_split_arrays(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sdos: List[np.ndarray] = []
    usdms: List[np.ndarray] = []
    objs: List[np.ndarray] = []
    for month in issue_months(split):
        if not month_ingest_ready(split, month):
            continue
        s, u, o = load_month_arrays(split, month)
        sdos.append(s.ravel())
        usdms.append(u.ravel())
        objs.append(o.ravel())
    if not sdos:
        raise FileNotFoundError(f"no ingest-ready months for split={split}")
    return np.concatenate(sdos), np.concatenate(usdms), np.concatenate(objs)


def concat_train_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sdos: List[np.ndarray] = []
    usdms: List[np.ndarray] = []
    objs: List[np.ndarray] = []
    for month in train_issue_months():
        if not month_ingest_ready("train_pool", month):
            continue
        s, u, o = load_month_arrays("train_pool", month)
        sdos.append(s.ravel())
        usdms.append(u.ravel())
        objs.append(o.ravel())
    if not sdos:
        raise FileNotFoundError("no train_pool ingest-ready months")
    return np.concatenate(sdos), np.concatenate(usdms), np.concatenate(objs)


def ordered_months_all_splits() -> List[Tuple[str, str]]:
    """Return (split, issue_yyyymm) sorted chronologically."""
    rows: List[Tuple[str, str]] = []
    for split in ("train_pool", "official_test", "hwb_holdout"):
        months = train_issue_months() if split == "train_pool" else issue_months(split)
        for m in months:
            rows.append((split, m))
    return sorted(rows, key=lambda x: x[1])
