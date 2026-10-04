"""FL-2 Phase C scientific data layout and readiness checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.scientific.paths import (
    PROJECT_ROOT,
    hwb_holdout_high_manifest,
    hwb_holdout_manifest,
    official_test_high_manifest,
    official_test_manifest,
    scientific_signoff_path,
    taskpack_data_root,
    taskpack_repo_pin,
)

TASKPACK = "FL-2"
DATA_SOURCE = "floodcastbench_official_test_v1"
VENDOR_ROOT = PROJECT_ROOT / "data" / "vendor" / "floodcastbench"
SPEC_REF = "docs/engineering/hcg/external_expansion/FL-2/HWB_FL2_HOLDOUT_SPEC.md"
ZENODO_RECORD = "https://zenodo.org/records/14017092"
ZENODO_RECORD_ID = "14017092"
FLOODCASTBENCH_REPO = "https://github.com/HydroPML/FloodCastBench"
FLOODCAST_REPO = "https://github.com/HydroPML/FloodCast"
N_STEPS = 20

SPLIT_ROOTS: Dict[str, str] = {
    "official_test": "official_test",
    "hwb_holdout": "hwb_holdout",
    "official_test_high": "official_test_high",
    "hwb_holdout_high": "hwb_holdout_high",
}

SPLIT_PROTOCOLS: Dict[str, str] = {
    "official_test": "anchor_fcb_official_test",
    "hwb_holdout": "hwb_fcb_cross_regional_holdout",
    "official_test_high": "anchor_fcb_official_test_high",
    "hwb_holdout_high": "hwb_fcb_cross_regional_holdout_high",
}

HIGH_FI_SPLITS = frozenset({"official_test_high", "hwb_holdout_high"})
SLICE1_SPLITS = frozenset({"official_test", "hwb_holdout"})
SLICE2_SPLITS = HIGH_FI_SPLITS

PRED_SUBDIR: Dict[str, str] = {
    "official_test": "test",
    "hwb_holdout": "holdout",
    "official_test_high": "test",
    "hwb_holdout_high": "holdout",
}

_SPLIT_MANIFEST_FN = {
    "official_test": official_test_manifest,
    "hwb_holdout": hwb_holdout_manifest,
    "official_test_high": official_test_high_manifest,
    "hwb_holdout_high": hwb_holdout_high_manifest,
}


def split_root(split: str) -> Path:
    dirname = SPLIT_ROOTS.get(split)
    if dirname is None:
        raise ValueError(f"unknown FL-2 split {split!r}")
    return taskpack_data_root(TASKPACK) / dirname


def official_test_root() -> Path:
    return split_root("official_test")


def hwb_holdout_root() -> Path:
    return split_root("hwb_holdout")


def official_test_high_root() -> Path:
    return split_root("official_test_high")


def hwb_holdout_high_root() -> Path:
    return split_root("hwb_holdout_high")


def scenarios_dir(split: str) -> Path:
    return split_root(split) / "scenarios"


def load_split_manifest(split: str) -> Optional[Dict[str, Any]]:
    manifest_fn = _SPLIT_MANIFEST_FN.get(split)
    if manifest_fn is None:
        return None
    path = manifest_fn(TASKPACK)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def scenario_ids(split: str) -> List[str]:
    manifest = load_split_manifest(split) or {}
    ids = list(manifest.get("scenario_ids") or [])
    if ids:
        return ids
    for row in manifest.get("scenarios") or []:
        sid = row.get("scenario_id")
        if sid:
            ids.append(str(sid))
    return sorted(set(ids))


def scenario_dir(split: str, scenario_id: str) -> Path:
    return scenarios_dir(split) / scenario_id


def load_scenario_meta(split: str, scenario_id: str) -> Optional[Dict[str, Any]]:
    manifest = load_split_manifest(split) or {}
    for row in manifest.get("scenarios") or []:
        if str(row.get("scenario_id")) == scenario_id:
            return row
    return None


def load_split_scenarios(split: str) -> List[Dict[str, Any]]:
    manifest = load_split_manifest(split) or {}
    rows = list(manifest.get("scenarios") or [])
    out: List[Dict[str, Any]] = []
    for row in rows:
        sid = str(row.get("scenario_id") or "")
        if not sid:
            continue
        root = scenario_dir(split, sid)
        if (root / "truth_depth.npy").is_file():
            out.append({**row, "scenario_dir": str(root)})
    return out


def load_scenario_arrays(split: str, scenario_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = scenario_dir(split, scenario_id)
    dem = np.load(root / "dem.npy")
    initial = np.load(root / "initial_depth.npy")
    truth = np.load(root / "truth_depth.npy")
    return dem, initial, truth


def load_scenario_full_inputs(
    split: str, scenario_id: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """DEM, initial depth, truth depth, rainfall (T,H,W), landuse (H,W), boundary_conditions (H,W)."""
    root = scenario_dir(split, scenario_id)
    dem, initial, truth = load_scenario_arrays(split, scenario_id)
    rainfall = np.load(root / "rainfall.npy")
    landuse = np.load(root / "landuse.npy")
    boundary = np.load(root / "boundary_conditions.npy")
    return dem, initial, truth, rainfall, landuse, boundary


def scenario_full_inputs_ready(split: str, scenario_id: str) -> bool:
    from hazardweaver.hcg.carp.scientific.fl2_zenodo_relevant_inputs import (
        scenario_input_files_ready,
    )

    return scenario_input_files_ready(split, scenario_id)


def full_input_schema_ready(
    splits: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    from hazardweaver.hcg.carp.scientific.fl2_zenodo_relevant_inputs import (
        input_schema_ready_for_splits,
    )

    split_list = splits or ["official_test", "hwb_holdout"]
    return input_schema_ready_for_splits(split_list)


def _split_scenarios_ready(test_split: str, hold_split: str) -> bool:
    off_ids = scenario_ids(test_split)
    hold_ids = scenario_ids(hold_split)
    if not off_ids or not hold_ids:
        return False
    for sid in off_ids:
        root = scenario_dir(test_split, sid)
        if not (root / "truth_depth.npy").is_file():
            return False
    for sid in hold_ids:
        root = scenario_dir(hold_split, sid)
        if not (root / "truth_depth.npy").is_file():
            return False
    manifest = load_split_manifest(test_split) or {}
    return manifest.get("scenarios_ready") is True and manifest.get("fixture_scale") is False


def scenarios_ready() -> bool:
    return _split_scenarios_ready("official_test", "hwb_holdout")


def slice2_scenarios_ready() -> bool:
    return _split_scenarios_ready("official_test_high", "hwb_holdout_high")


def splits_for_fidelity(fidelity: str = "low") -> List[str]:
    if fidelity == "high":
        return ["official_test_high", "hwb_holdout_high"]
    return ["official_test", "hwb_holdout"]


def is_high_fi_split(split: str) -> bool:
    return split in HIGH_FI_SPLITS


def predictions_base_dir(split: str) -> str:
    return "predictions_high" if is_high_fi_split(split) else "predictions"


def prediction_subdir(split: str) -> str:
    return PRED_SUBDIR.get(split, split)


def prediction_path(cap_dir: Path, split: str, scenario_id: str) -> Path:
    return cap_dir / predictions_base_dir(split) / prediction_subdir(split) / f"{scenario_id}.npz"


def split_scenarios_ready(split: str) -> bool:
    if split in HIGH_FI_SPLITS:
        return slice2_scenarios_ready()
    if split in SLICE1_SPLITS:
        return scenarios_ready()
    manifest = load_split_manifest(split) or {}
    if not manifest.get("scenario_ids"):
        return False
    for sid in scenario_ids(split):
        if not (scenario_dir(split, sid) / "truth_depth.npy").is_file():
            return False
    return manifest.get("scenarios_ready") is True


def repo_pin_ok() -> bool:
    pin = taskpack_repo_pin(TASKPACK)
    if not pin.is_file():
        return False
    data = json.loads(pin.read_text(encoding="utf-8"))
    commit = str(data.get("repo_commit") or data.get("version_or_commit") or "")
    return commit and commit.upper() != "UNKNOWN"


def scientific_signoff_ok() -> bool:
    path = scientific_signoff_path(TASKPACK)
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if not (
        data.get("approved") is True
        and data.get("approval_tier") == "scientific"
    ):
        return False
    if str(data.get("approval_source") or "") == "engineering_a2_dev":
        return False
    if data.get("approval_scope") != "scientific_data_and_holdout_contract_only":
        return False
    if data.get("capability_scientific_pass") is not False:
        return False
    if data.get("neural_routes_official_forward_status") != "BLOCKED_OFFICIAL":
        return False
    if not data.get("signed_by") or not data.get("signed_at"):
        return False
    return True


def contamination_audit_ok() -> bool:
    from hazardweaver.hcg.carp.scientific.fl2_contamination_audit import contamination_audit_ok as _l3_ok

    ok, _ = _l3_ok()
    return ok


def scientific_data_ready() -> bool:
    return (
        scenarios_ready()
        and repo_pin_ok()
        and contamination_audit_ok()
        and scientific_signoff_ok()
    )


def sfincs_vendor_ready() -> bool:
    pin = PROJECT_ROOT / "data" / "vendor" / "sfincs" / "REPO_PIN.json"
    if not pin.is_file():
        return False
    from hazardweaver.hcg.carp.scientific.fl2_sfincs_verify import verify_sfincs_executable

    ok, _ = verify_sfincs_executable()
    return ok


def container_runtime_available() -> bool:
    from hazardweaver.hcg.carp.scientific.fl2_sfincs_runtime import container_runtime_available as _crt

    return _crt()


def lisflood_fp_vendor_ready() -> bool:
    pin = PROJECT_ROOT / "data" / "vendor" / "lisflood_fp" / "REPO_PIN.json"
    if not pin.is_file():
        return False
    data = json.loads(pin.read_text(encoding="utf-8"))
    status = str(data.get("audit_status") or "")
    if status != "PASS" or not data.get("repo_commit"):
        return False
    if data.get("executable_verified") is False:
        return False
    from hazardweaver.hcg.carp.scientific.fl2_lisflood_verify import verify_lisflood_executable

    ok, _ = verify_lisflood_executable()
    return ok


def zenodo_raw_ready() -> bool:
    zip_path = VENDOR_ROOT / "zenodo" / "raw" / "FloodCastBench.zip"
    if not zip_path.is_file():
        return False
    return zip_path.stat().st_size >= 18 * 1024 * 1024 * 1024


def zenodo_extract_ready() -> bool:
    marker = VENDOR_ROOT / "zenodo" / "extracted" / ".extract_complete"
    return marker.is_file()


def docker_available() -> bool:
    import shutil

    return shutil.which("docker") is not None
