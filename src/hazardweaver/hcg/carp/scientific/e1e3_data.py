"""E1-E3 Phase C scientific data layout and readiness checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from hazardweaver.hcg.carp.acquire.env_routing import SEISBENCH_PYTHON
from hazardweaver.hcg.carp.acquire.seisbench_paths import (
    SEISBENCH_VENDOR_ROOT as _SEISBENCH_VENDOR_ROOT,
    resolve_seisbench_cache_dir,
)
from hazardweaver.hcg.carp.scientific.paths import (
    PROJECT_ROOT,
    hwb_holdout_manifest,
    official_test_manifest,
    scientific_signoff_path,
    taskpack_data_root,
    taskpack_repo_pin,
)

TASKPACK = "E1-E3"
VENDOR_ROOT = _SEISBENCH_VENDOR_ROOT
SPEC_REF = "docs/engineering/hcg/external_expansion/E1-E3/HWB_E1E3_HOLDOUT_SPEC.md"
DATA_SOURCE = "seisbench_geofon_official_test_v1"

PICKER_STORES = ("phasenet", "eqtransformer", "gpd")


def seisbench_python() -> Path:
    return SEISBENCH_PYTHON


def official_test_root() -> Path:
    return taskpack_data_root(TASKPACK) / "official_test"


def hwb_holdout_root() -> Path:
    return taskpack_data_root(TASKPACK) / "hwb_holdout"


def traces_dir(split: str) -> Path:
    root = official_test_root() if split == "official_test" else hwb_holdout_root()
    return root / "traces"


def load_split_manifest(split: str) -> Optional[Dict[str, Any]]:
    path = official_test_manifest(TASKPACK) if split == "official_test" else hwb_holdout_manifest(TASKPACK)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def official_test_trace_ids() -> List[str]:
    manifest = load_split_manifest("official_test") or {}
    ids = list(manifest.get("trace_ids") or [])
    if ids:
        return ids
    for row in manifest.get("traces") or []:
        tid = row.get("trace_id")
        if tid:
            ids.append(str(tid))
    return sorted(set(ids))


def holdout_trace_ids() -> List[str]:
    manifest = load_split_manifest("hwb_holdout") or {}
    ids = list(manifest.get("trace_ids") or manifest.get("holdout_trace_ids") or [])
    if ids:
        return ids
    for row in manifest.get("traces") or []:
        tid = row.get("trace_id")
        if tid:
            ids.append(str(tid))
    return sorted(set(ids))


def trace_npz_path(split: str, trace_id: str) -> Path:
    return traces_dir(split) / f"{trace_id}.npz"


def load_trace_meta(split: str, trace_id: str) -> Optional[Dict[str, Any]]:
    manifest = load_split_manifest(split) or {}
    for row in manifest.get("traces") or []:
        if str(row.get("trace_id")) == trace_id:
            return row
    return None


def load_split_traces(split: str) -> List[Dict[str, Any]]:
    manifest = load_split_manifest(split) or {}
    rows = list(manifest.get("traces") or [])
    out: List[Dict[str, Any]] = []
    for row in rows:
        tid = str(row.get("trace_id") or "")
        if not tid:
            continue
        npz = trace_npz_path(split, tid)
        if npz.is_file():
            out.append({**row, "npz_path": str(npz)})
    return out


def waveforms_ready() -> bool:
    off_ids = official_test_trace_ids()
    hold_ids = holdout_trace_ids()
    if not off_ids or not hold_ids:
        return False
    for tid in off_ids:
        if not trace_npz_path("official_test", tid).is_file():
            return False
    for tid in hold_ids:
        if not trace_npz_path("hwb_holdout", tid).is_file():
            return False
    manifest = load_split_manifest("official_test") or {}
    return manifest.get("waveforms_ready") is True


def picker_weights_ready() -> bool:
    if not SEISBENCH_PYTHON.is_file():
        return False
    for store in PICKER_STORES:
        pin = VENDOR_ROOT / store / "MODEL_PIN.json"
        if not pin.is_file():
            return False
    return True


def pyocto_vendor_ready() -> bool:
    pin = VENDOR_ROOT / "pyocto" / "REPO_PIN.json"
    if pin.is_file():
        return True
    try:
        import importlib.util

        spec = importlib.util.find_spec("pyocto")
        return spec is not None
    except (ImportError, ValueError):
        return False


def gamma_vendor_ready() -> bool:
    pin = VENDOR_ROOT / "gamma" / "REPO_PIN.json"
    return pin.is_file() and any((VENDOR_ROOT / "gamma").iterdir())


def real_vendor_ready() -> bool:
    pin = VENDOR_ROOT / "real" / "REPO_PIN.json"
    return pin.is_file() and any((VENDOR_ROOT / "real").iterdir())


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
    if not waveforms_ready():
        return False
    if not picker_weights_ready():
        return False
    return True


def repo_commit() -> str:
    scientific_pin = taskpack_repo_pin(TASKPACK)
    if scientific_pin.is_file():
        data = json.loads(scientific_pin.read_text(encoding="utf-8"))
        commit = data.get("repo_commit") or data.get("version_or_commit") or ""
        if commit and str(commit).upper() != "UNKNOWN":
            return str(commit)
    return ""


def seisbench_cache_dir() -> Path:
    return resolve_seisbench_cache_dir()
