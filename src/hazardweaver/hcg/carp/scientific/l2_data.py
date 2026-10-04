"""L2 Phase C scientific data layout and readiness checks."""

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

TASKPACK = "L2"
DATA_SOURCE = "lhasa_official_test_v1"
TRAIN_YEARS = (2015, 2016, 2017, 2018)
OFFICIAL_TEST_YEARS = (2019, 2020)
HOLDOUT_MIN_YEAR = 2021

SPEC_REF = "docs/engineering/hcg/external_expansion/L2/HWB_L2_LANDSLIDE_HOLDOUT_SPEC.md"


def data_root() -> Path:
    return taskpack_data_root(TASKPACK)


def vendor_root() -> Path:
    return data_root().parent.parent / "vendor" / "lhasa"


def lhasa_repo_dir() -> Path:
    return vendor_root() / "repo"


def static_root() -> Path:
    vend = vendor_root() / "static"
    if vend.is_dir():
        return vend
    return data_root() / "static"


def split_protocol_path() -> Path:
    return data_root() / "SPLIT_PROTOCOL.json"


def train_split_path() -> Path:
    return data_root() / "TRAIN_SPLIT.json"


def official_test_root() -> Path:
    return data_root() / "official_test"


def hwb_holdout_root() -> Path:
    return data_root() / "hwb_holdout"


def event_dir(split: str, event_id: str) -> Path:
    if split == "train_pool":
        return data_root() / "train_pool" / "events" / event_id
    root = official_test_root() if split == "official_test" else hwb_holdout_root()
    return root / "events" / event_id


def load_split_manifest(split: str) -> Optional[Dict[str, Any]]:
    path = official_test_manifest(TASKPACK) if split == "official_test" else hwb_holdout_manifest(TASKPACK)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def event_ids(split: str) -> List[str]:
    manifest = load_split_manifest(split) or {}
    if split == "official_test":
        return list(manifest.get("event_ids") or [])
    return list(manifest.get("holdout_ids") or manifest.get("event_ids") or [])


def load_split_protocol() -> Dict[str, Any]:
    path = split_protocol_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_event_meta(split: str, event_id: str) -> Dict[str, Any]:
    meta_path = event_dir(split, event_id) / "meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def event_features_ready(split: str, event_id: str) -> bool:
    feat = event_dir(split, event_id) / "features"
    label = event_dir(split, event_id) / "label" / "occurrence_label.npy"
    meta = load_event_meta(split, event_id)
    if not meta.get("features_ready"):
        return False
    required = ("rainfall_mm.npy", "antecedent_wetness.npy", "slope_deg.npy")
    return label.is_file() and all((feat / name).is_file() for name in required)


def repo_commit() -> str:
    pin = taskpack_repo_pin(TASKPACK)
    if pin.is_file():
        data = json.loads(pin.read_text(encoding="utf-8"))
        return str(data.get("repo_commit") or data.get("version_or_commit") or "")
    return ""


def earthdata_configured() -> bool:
    netrc = Path.home() / ".netrc"
    if not netrc.is_file():
        return False
    text = netrc.read_text(encoding="utf-8", errors="replace")
    return "urs.earthdata.nasa.gov" in text and "pps.eosdis.nasa.gov" in text


def scientific_data_ready() -> bool:
    off = official_test_manifest(TASKPACK)
    hold = hwb_holdout_manifest(TASKPACK)
    sign = scientific_signoff_path(TASKPACK)
    pin = taskpack_repo_pin(TASKPACK)
    repo = lhasa_repo_dir() / "lhasa.py"
    if not all(p.is_file() for p in (off, hold, sign, pin)):
        return False
    off_data = json.loads(off.read_text(encoding="utf-8"))
    if off_data.get("fixture_scale"):
        return False
    if not repo.is_file():
        return False
    test_ids = event_ids("official_test")
    if not test_ids:
        return False
    return any(event_features_ready("official_test", eid) for eid in test_ids)


def load_event_arrays(split: str, event_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = event_dir(split, event_id)
    rainfall = np.load(root / "features" / "rainfall_mm.npy")
    antecedent = np.load(root / "features" / "antecedent_wetness.npy")
    slope = np.load(root / "features" / "slope_deg.npy")
    label = np.load(root / "label" / "occurrence_label.npy")
    return rainfall, antecedent, slope, label


def load_lhasa_predictions(split: str, event_id: str) -> Optional[np.ndarray]:
    path = event_dir(split, event_id) / "predictions" / "lhasa_prob.npy"
    if not path.is_file():
        return None
    return np.load(path)


def concat_split_arrays(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rains: List[np.ndarray] = []
    ants: List[np.ndarray] = []
    slopes: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for eid in event_ids(split):
        if not event_features_ready(split, eid):
            continue
        r, a, s, l = load_event_arrays(split, eid)
        rains.append(r.ravel())
        ants.append(a.ravel())
        slopes.append(s.ravel())
        labels.append(l.ravel())
    if not rains:
        raise FileNotFoundError(f"no feature-ready events for split={split}")
    return (
        np.concatenate(rains),
        np.concatenate(ants),
        np.concatenate(slopes),
        np.concatenate(labels),
    )


def train_event_ids() -> List[str]:
    path = train_split_path()
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("event_ids") or [])
    protocol = load_split_protocol()
    return list(protocol.get("train_event_ids") or [])
