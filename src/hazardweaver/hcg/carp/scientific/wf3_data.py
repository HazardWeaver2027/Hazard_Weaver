"""WF-3 Phase C scientific data layout and readiness checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hazardweaver.hcg.carp.scientific.paths import (
    PROJECT_ROOT,
    hwb_holdout_manifest,
    official_test_manifest,
    scientific_signoff_path,
    taskpack_data_root,
    taskpack_repo_pin,
)

TASKPACK = "WF-3"
VENDOR_ROOT = PROJECT_ROOT / "data" / "vendor" / "wildfirespreadts"
DEFAULT_HDF5_ROOT = VENDOR_ROOT / "hdf5"
NIFC_MANIFEST = PROJECT_ROOT / "data" / "processed" / "wf3_nifc_progression_v1" / "manifest.json"

# Authoritative 12-fold table from WildfireSpreadTS FireSpreadDataModule.split_fires
FOLD_YEAR_TABLE: List[Tuple[int, int, int, int]] = [
    (2018, 2019, 2020, 2021),
    (2018, 2019, 2021, 2020),
    (2018, 2020, 2019, 2021),
    (2018, 2020, 2021, 2019),
    (2018, 2021, 2019, 2020),
    (2018, 2021, 2020, 2019),
    (2019, 2020, 2018, 2021),
    (2019, 2020, 2021, 2018),
    (2019, 2021, 2018, 2020),
    (2019, 2021, 2020, 2018),
    (2020, 2021, 2018, 2019),
    (2020, 2021, 2019, 2018),
]

DEFAULT_HOLDOUT_FIRES = [
    {"holdout_fire_id": "hwb_wf3_ballard_v1", "nifc_name": "Ballard"},
    {"holdout_fire_id": "hwb_wf3_blue_ridge_v1", "nifc_name": "Blue Ridge"},
    {"holdout_fire_id": "hwb_wf3_cottonwood_v1", "nifc_name": "Cottonwood"},
]

WSTS_ANCHOR_YEARS = {2018, 2019, 2020, 2021}


def fold_year_map(fold_id: int) -> Dict[str, List[int]]:
    if fold_id < 0 or fold_id >= len(FOLD_YEAR_TABLE):
        raise ValueError(f"invalid fold_id {fold_id}; expected 0..{len(FOLD_YEAR_TABLE) - 1}")
    train_a, train_b, val_y, test_y = FOLD_YEAR_TABLE[fold_id]
    return {
        "train_years": [train_a, train_b],
        "val_years": [val_y],
        "test_years": [test_y],
    }


def all_train_years_across_folds() -> set[int]:
    years: set[int] = set()
    for fold_id in range(len(FOLD_YEAR_TABLE)):
        years.update(fold_year_map(fold_id)["train_years"])
    return years


def hdf5_dir() -> Path:
    env = os.environ.get("WSTS_HDF5_DIR", "").strip()
    if env:
        return Path(env)
    manifest = load_split_manifest("official_test") or {}
    root = manifest.get("hdf5_root")
    if root:
        return Path(str(root))
    return DEFAULT_HDF5_ROOT


def official_test_root() -> Path:
    return taskpack_data_root(TASKPACK) / "official_test"


def hwb_holdout_root() -> Path:
    return taskpack_data_root(TASKPACK) / "hwb_holdout"


def load_split_manifest(split: str) -> Optional[Dict[str, Any]]:
    path = official_test_manifest(TASKPACK) if split == "official_test" else hwb_holdout_manifest(TASKPACK)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def holdout_fire_ids() -> List[str]:
    manifest = load_split_manifest("hwb_holdout") or {}
    ids = list(manifest.get("holdout_fire_ids") or manifest.get("holdout_ids") or [])
    if ids:
        return ids
    for row in manifest.get("fires") or []:
        fid = row.get("holdout_fire_id") or row.get("fire_id")
        if fid:
            ids.append(str(fid))
    return sorted(set(ids))


def official_test_fold_ids() -> List[int]:
    manifest = load_split_manifest("official_test") or {}
    fold_ids = manifest.get("fold_ids")
    if isinstance(fold_ids, list) and fold_ids:
        return [int(x) for x in fold_ids]
    return list(range(len(FOLD_YEAR_TABLE)))


def hdf5_ready(hdf5_root: Optional[Path] = None) -> bool:
    root = hdf5_root or hdf5_dir()
    if not root.is_dir():
        return False
    for year in WSTS_ANCHOR_YEARS:
        year_dir = root / str(year)
        if not year_dir.is_dir():
            return False
        if not any(year_dir.glob("*.hdf5")):
            return False
    return True


def cell2fire_vendor_ready() -> bool:
    vendor = PROJECT_ROOT / "data" / "vendor" / "cell2fire"
    legacy = PROJECT_ROOT / "data" / "vendor" / "https___github.com_fire2a_Cell2Fire"
    pin = vendor / "REPO_PIN.json"
    if pin.is_file():
        return True
    if vendor.is_dir() and any(vendor.iterdir()):
        return True
    return legacy.is_dir() and (legacy / "ASSET_PIN.json").is_file()


def scientific_data_ready() -> bool:
    off = official_test_manifest(TASKPACK)
    hold = hwb_holdout_manifest(TASKPACK)
    sign = scientific_signoff_path(TASKPACK)
    if not (off.is_file() and hold.is_file() and sign.is_file()):
        return False
    off_data = json.loads(off.read_text(encoding="utf-8"))
    if off_data.get("fixture_scale"):
        return False
    sign_data = json.loads(sign.read_text(encoding="utf-8"))
    if not sign_data.get("approved") or sign_data.get("approval_tier") != "scientific":
        return False
    if not hdf5_ready():
        return False
    for fid in holdout_fire_ids():
        bundle = hwb_holdout_root() / "fires" / fid
        if not (bundle / "prior.npy").is_file() or not (bundle / "truth.npy").is_file():
            return False
    return True


def repo_commit() -> str:
    scientific_pin = taskpack_repo_pin(TASKPACK)
    if scientific_pin.is_file():
        data = json.loads(scientific_pin.read_text(encoding="utf-8"))
        commit = data.get("repo_commit") or data.get("version_or_commit") or ""
        if commit and str(commit).upper() != "UNKNOWN":
            return str(commit)
    replay_pin = VENDOR_ROOT / "REPLAY_PIN.json"
    if replay_pin.is_file():
        data = json.loads(replay_pin.read_text(encoding="utf-8"))
        return str(data.get("commit_sha") or data.get("repo_commit") or "")
    return ""


def fire_dir(hdf5_root: Path, year: int, fire_name: str) -> Path:
    return hdf5_root / str(year) / f"{fire_name}.hdf5"


def list_wsts_fire_names(hdf5_root: Optional[Path] = None) -> List[str]:
    root = hdf5_root or hdf5_dir()
    names: set[str] = set()
    if not root.is_dir():
        return []
    for year in WSTS_ANCHOR_YEARS:
        year_dir = root / str(year)
        if not year_dir.is_dir():
            continue
        for path in year_dir.glob("*.hdf5"):
            names.add(path.stem)
    return sorted(names)
