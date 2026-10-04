"""TC-TRK Phase C scientific data layout and readiness checks."""

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

TASKPACK = "TC-TRK"
DATA_SOURCE = "tcbench_official_test_v1"
OFFICIAL_TEST_YEAR = 2023
HOLDOUT_YEAR = 2024
TRAIN_YEARS = (2017, 2018, 2019, 2020)


def data_root() -> Path:
    return taskpack_data_root(TASKPACK)


def split_protocol_path() -> Path:
    return data_root() / "SPLIT_PROTOCOL.json"


def ibtracs_dir() -> Path:
    return data_root() / "ibtracs"


def vendor_root() -> Path:
    return data_root().parent.parent / "vendor" / "tcbench"


def matched_tracks_dir() -> Path:
    vend = vendor_root() / "matched_tracks"
    if vend.is_dir():
        return vend
    return data_root() / "matched_tracks"


def tcbench_repo_dir() -> Path:
    return vendor_root() / "repo"


def official_ibtracs_eval_dir(year: int = OFFICIAL_TEST_YEAR) -> Path:
    """Single-file IBTrACS folder required by TCBench toolbox.read_hist_track_file."""
    base = vendor_root() / "ibtracs_eval" / str(year)
    base.mkdir(parents=True, exist_ok=True)
    src = ibtracs_dir() / f"{year}_IBTrACS.csv"
    dest = base / src.name
    if src.is_file():
        if dest.exists() or dest.is_symlink():
            dest.unlink(missing_ok=True)
        try:
            dest.symlink_to(src.resolve())
        except OSError:
            import shutil

            shutil.copy2(src, dest)
    return base


def matched_track_files() -> Dict[str, Path]:
    names = (
        "2023_TIGGE_GEFS.csv",
        "2023_PANGU.csv",
        "2023_fcnet.csv",
        "2023_Gencast(weathernext).csv",
    )
    root = matched_tracks_dir()
    return {name: root / name for name in names}


def official_test_root() -> Path:
    return data_root() / "official_test"


def hwb_holdout_root() -> Path:
    return data_root() / "hwb_holdout"


def storm_dir(split: str, sid: str) -> Path:
    root = official_test_root() if split == "official_test" else hwb_holdout_root()
    return root / "storms" / sid


def load_split_manifest(split: str) -> Optional[Dict[str, Any]]:
    path = official_test_manifest(TASKPACK) if split == "official_test" else hwb_holdout_manifest(TASKPACK)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def storm_ids(split: str) -> List[str]:
    manifest = load_split_manifest(split) or {}
    return list(manifest.get("storm_ids") or manifest.get("holdout_ids") or [])


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
    ibtracs = ibtracs_dir() / f"{OFFICIAL_TEST_YEAR}_IBTrACS.csv"
    if not all(p.is_file() for p in (off, hold, sign, pin, ibtracs)):
        return False
    off_data = json.loads(off.read_text(encoding="utf-8"))
    if off_data.get("fixture_scale"):
        return False
    if not (off_data.get("storm_ids") or off_data.get("holdout_ids")):
        return False
    hold_data = json.loads(hold.read_text(encoding="utf-8"))
    if hold_data.get("fixture_scale"):
        return False
    if not (hold_data.get("holdout_ids") or hold_data.get("storm_ids")):
        return False
    pin_data = json.loads(pin.read_text(encoding="utf-8"))
    commit = pin_data.get("repo_commit") or pin_data.get("version_or_commit") or ""
    if not commit or str(commit).upper() == "UNKNOWN":
        return False
    repo = tcbench_repo_dir()
    if not (repo / "dev" / "evaluate_tracks.py").is_file():
        return False
    ibtracs = ibtracs_dir() / f"{OFFICIAL_TEST_YEAR}_IBTrACS.csv"
    if not ibtracs.is_file():
        return False
    tracks = matched_track_files()
    if not all(p.is_file() for p in tracks.values()):
        return False
    return True
