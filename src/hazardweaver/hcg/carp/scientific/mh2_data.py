"""MH-2 groundfailure / VBCI scientific paths and readiness checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from hazardweaver.hcg.carp.scientific.paths import (
    hwb_holdout_manifest,
    official_test_manifest,
    scientific_signoff_path,
    taskpack_data_root,
)

TASKPACK = "MH-2"
VBCI_EVENT_ID = "ci38457511"
GF_REFERENCE_MODELS = ("nowicki_jessee_2018", "nowicki_2014", "godt_2008", "newmark")


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]
OFFICIAL_TEST_EVENTS = ("nc72282711", "ci38457511")
HOLDOUT_EVENTS = ("nn00725272",)  # Monte Cristo 2020 (us6000j7ql was invalid comcat id)


def calibration_exclusion_path() -> Path:
    return taskpack_data_root(TASKPACK) / "CALIBRATION_EXCLUSION.json"


def repo_pin_path() -> Path:
    return taskpack_data_root(TASKPACK) / "REPO_PIN.json"


def vbci_pin_path() -> Path:
    return taskpack_data_root(TASKPACK) / "VBCI_PIN.json"


def official_test_root() -> Path:
    return taskpack_data_root(TASKPACK) / "official_test"


def hwb_holdout_root() -> Path:
    return taskpack_data_root(TASKPACK) / "hwb_holdout"


def groundfailure_vendor_root() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "vendor" / "groundfailure"


def groundfailure_repo_dir() -> Path:
    return groundfailure_vendor_root() / "groundfailure"


def vbci_vendor_root() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "vendor" / "vbci"


def vbci_repo_dir() -> Path:
    return vbci_vendor_root() / "VBCI"


def event_dir(split: str, event_id: str) -> Path:
    root = official_test_root() if split == "official_test" else hwb_holdout_root()
    return root / "events" / event_id


def load_calibration_exclusion() -> Dict[str, Any]:
    path = calibration_exclusion_path()
    if not path.is_file():
        return {"event_ids": [], "events": []}
    return json.loads(path.read_text(encoding="utf-8"))


def calibration_event_ids() -> List[str]:
    data = load_calibration_exclusion()
    ids = list(data.get("event_ids") or [])
    for row in data.get("events") or []:
        eid = row.get("usgs_event_id") or row.get("event_id")
        if eid:
            ids.append(str(eid))
    return sorted(set(ids))


def load_split_manifest(split: str) -> Optional[Dict[str, Any]]:
    path = official_test_manifest(TASKPACK) if split == "official_test" else hwb_holdout_manifest(TASKPACK)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def event_shakegrid_path(split: str, event_id: str) -> Path:
    return event_dir(split, event_id) / "shakemap" / "grid.xml"


def event_has_faithful_inputs(split: str, event_id: str) -> bool:
    ev = event_dir(split, event_id)
    grid = event_shakegrid_path(split, event_id)
    meta = ev / "shakemap" / "pga_summary.json"
    if not grid.is_file() or grid.stat().st_size < 100:
        return False
    if not meta.is_file():
        return False
    try:
        summary = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(summary.get("fetch_ok") or (summary.get("shakemap") or {}).get("fetch_ok"))


def vbci_case_ready(event_id: str = VBCI_EVENT_ID) -> bool:
    root = event_dir("official_test", event_id) / "vbci"
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return False
    data = json.loads(manifest.read_text(encoding="utf-8"))
    svi_case = root / "svi_case" / "vbci_inputs.mat"
    return bool(
        (data.get("vbci_case_ready") or data.get("zenodo_ready"))
        and svi_case.is_file()
    )


def gf_reference_ready(event_id: str = VBCI_EVENT_ID) -> bool:
    from hazardweaver.hcg.carp.scientific.mh2_fidelity import REFERENCE_MANIFEST
    from hazardweaver.hcg.carp.scientific.mh2_reference import load_reference_manifest

    ref_dir = event_dir("official_test", event_id) / "reference"
    manifest = load_reference_manifest(ref_dir)
    if not manifest or not manifest.get("models"):
        return False
    for model_key in GF_REFERENCE_MODELS:
        entry = manifest["models"].get(model_key)
        if not entry:
            return False
        if not Path(str(entry.get("prob_path", ""))).is_file():
            return False
    return (ref_dir / REFERENCE_MANIFEST).is_file()


def vbci_reference_ready(event_id: str = VBCI_EVENT_ID) -> bool:
    from hazardweaver.hcg.carp.scientific.mh2_fidelity import REFERENCE_MANIFEST
    from hazardweaver.hcg.carp.scientific.mh2_reference import load_reference_manifest

    ref_dir = event_dir("official_test", event_id) / "vbci" / "reference_posterior"
    manifest = load_reference_manifest(ref_dir) or {}
    entry = (manifest.get("models") or {}).get("vbci_posterior_ls") or {}
    return (
        (ref_dir / "posterior_ls_coarse.npy").is_file()
        and (ref_dir / REFERENCE_MANIFEST).is_file()
        and entry.get("reference_kind") == "pinned_official_vbci_svi_v1"
    )


def lifeline_gt_ready() -> bool:
    holdout = load_split_manifest("hwb_holdout") or {}
    if holdout.get("lifeline_gt_status") == "BLOCKED":
        manifest = lifeline_gt_manifest()
        if not manifest:
            return False
        if manifest.get("status") != "READY":
            return False
    gt_root = hwb_holdout_root() / "lifeline_gt"
    manifest_path = gt_root / "manifest.json"
    if not manifest_path.is_file():
        return False
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return bool(
        data.get("status") == "READY"
        and int(data.get("segment_count", 0) or 0) > 0
        and data.get("source_license_audited")
    )


def lifeline_gt_manifest() -> Optional[Dict[str, Any]]:
    path = hwb_holdout_root() / "lifeline_gt" / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def scientific_data_ready() -> bool:
    off = official_test_manifest(TASKPACK)
    hold = hwb_holdout_manifest(TASKPACK)
    sign = scientific_signoff_path(TASKPACK)
    if not (off.is_file() and hold.is_file() and sign.is_file()):
        return False
    off_data = json.loads(off.read_text(encoding="utf-8"))
    if off_data.get("fixture_scale"):
        return False
    for eid in off_data.get("event_ids") or []:
        if not event_has_faithful_inputs("official_test", str(eid)):
            return False
    for eid in HOLDOUT_EVENTS:
        if not event_has_faithful_inputs("hwb_holdout", eid):
            return False
    if not repo_pin_path().is_file():
        return False
    if not gf_reference_ready(VBCI_EVENT_ID):
        return False
    if not vbci_reference_ready(VBCI_EVENT_ID):
        return False
    if not vbci_case_ready(VBCI_EVENT_ID):
        return False
    return True
