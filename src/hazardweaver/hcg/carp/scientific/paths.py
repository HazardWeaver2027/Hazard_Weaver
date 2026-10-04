"""Canonical paths for Phase C scientific artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]

SCIENTIFIC_DATA_ROOT = PROJECT_ROOT / "data" / "scientific"
SCIENTIFIC_RUNS_ROOT = PROJECT_ROOT / "runs" / "carp" / "scientific"


def taskpack_data_root(taskpack: str) -> Path:
    return SCIENTIFIC_DATA_ROOT / taskpack


def official_test_manifest(taskpack: str) -> Path:
    return taskpack_data_root(taskpack) / "official_test" / "manifest.json"


def hwb_holdout_manifest(taskpack: str) -> Path:
    return taskpack_data_root(taskpack) / "hwb_holdout" / "manifest.json"


def official_test_high_manifest(taskpack: str) -> Path:
    return taskpack_data_root(taskpack) / "official_test_high" / "manifest.json"


def hwb_holdout_high_manifest(taskpack: str) -> Path:
    return taskpack_data_root(taskpack) / "hwb_holdout_high" / "manifest.json"


def scientific_signoff_path(taskpack: str) -> Path:
    return taskpack_data_root(taskpack) / "hwb_holdout" / "PI_SCIENTIFIC_SIGNOFF.json"


def taskpack_repo_pin(taskpack: str) -> Path:
    return taskpack_data_root(taskpack) / "REPO_PIN.json"


def cap_scientific_dir(
    taskpack: str,
    capability_id: str,
    *,
    runs_root: Optional[Path] = None,
) -> Path:
    root = runs_root or SCIENTIFIC_RUNS_ROOT
    return root / taskpack / capability_id
