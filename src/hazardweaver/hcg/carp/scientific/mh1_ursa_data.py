"""Ursa 1.0.0 anchor reproduction data for CAP-MH1-05 (not HWB holdout)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hazardweaver.hcg.carp.scientific.paths import taskpack_data_root

TASKPACK = "MH-1"
DATA_SOURCE = "ursa_anchor_reproduction_v1"
ADDENDUM = "docs/engineering/hcg/W3_SCIENTIFIC_COMPLETION_ADDENDUM.md"

URSA_VERSION = "1.0.0"
URSA_DOI = "10.5066/P15CKL9J"
PWFDF_R_DOI = "10.5066/P18RMZBB"
SUPPORTING_DATA_DOI = "10.5066/P13GYER5"
USGS_PYPI_INDEX = "https://code.usgs.gov/api/v4/groups/859/-/packages/pypi/simple"
URSA_MIN_PYTHON = (3, 11)

# Fires overlapping 227 inventory — CAP-MH1-05 must not use locally retrained upstream on these.
INVENTORY_OVERLAP_FIRES = frozenset(
    {"Old", "Grand Prix", "Thomas", "Apple", "El Dorado", "Pipeline"}
)

ANCHOR_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "old_grand_prix_2003",
        "year": 2003,
        "fires": ["Old", "Grand Prix"],
        "label": "2003 Old/Grand Prix",
    },
    {
        "case_id": "thomas_montecito_2017",
        "year": 2017,
        "fires": ["Thomas", "Montecito"],
        "label": "2017 Thomas/Montecito",
    },
    {
        "case_id": "el_dorado_apple_birch_2020",
        "year": 2020,
        "fires": ["El Dorado", "Apple", "Birch Creek"],
        "label": "2020 El Dorado+Apple/Birch Creek",
    },
    {
        "case_id": "pipeline_copeland_2022",
        "year": 2022,
        "fires": ["Pipeline", "Copeland"],
        "label": "2022 Pipeline/Copeland",
    },
]


def data_root() -> Path:
    return taskpack_data_root(TASKPACK)


def anchor_root() -> Path:
    return data_root() / "anchor_reproduction"


def vendor_root() -> Path:
    return data_root().parent.parent / "vendor" / "ursa"


def ursa_repo_dir() -> Path:
    return vendor_root() / "repo"


def ursa_pin_path() -> Path:
    return data_root() / "URSA_PIN.json"


def pwfdf_r_contract_path() -> Path:
    return data_root() / "PWFDF_R_CONTRACT.json"


def case_dir(case_id: str) -> Path:
    return anchor_root() / "cases" / case_id


def load_anchor_manifest() -> Dict[str, Any]:
    path = anchor_root() / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def case_ids() -> List[str]:
    manifest = load_anchor_manifest()
    return list(manifest.get("case_ids") or [c["case_id"] for c in ANCHOR_CASES])


def observed_case_ready(case_id: str) -> bool:
    root = case_dir(case_id)
    obs = root / "observed" / "runout_footprint.npy"
    meta = root / "meta.json"
    return obs.is_file() and meta.is_file()


def case_ready(case_id: str) -> bool:
    root = case_dir(case_id)
    pred = root / "predictions" / "dfsi_hazard.npy"
    return observed_case_ready(case_id) and pred.is_file()


def ursa_repo_ready() -> bool:
    repo = ursa_repo_dir()
    return repo.is_dir() and (repo / "pyproject.toml").is_file()


def _pin_version_ok(pin_data: Dict[str, Any]) -> bool:
    ver = str(
        pin_data.get("version")
        or pin_data.get("repo_tag")
        or pin_data.get("repo_commit")
        or pin_data.get("version_or_commit")
        or ""
    )
    return bool(ver) and ver.upper() != "UNKNOWN"


def ursa_anchor_inputs_ready() -> bool:
    """Inputs for eval: pin + manifest + observed footprints (predictions produced at eval)."""
    pin = ursa_pin_path()
    manifest = anchor_root() / "manifest.json"
    if not pin.is_file() or not manifest.is_file():
        return False
    pin_data = json.loads(pin.read_text(encoding="utf-8"))
    if not _pin_version_ok(pin_data):
        return False
    man = json.loads(manifest.read_text(encoding="utf-8"))
    if man.get("role") != "anchor_reproduction":
        return False
    if man.get("pwfdf_r_as_gt") is not False:
        return False
    if man.get("fixture_scale", True):
        return False
    if not man.get("data_ready"):
        return False
    ids = case_ids()
    return bool(ids) and all(observed_case_ready(cid) for cid in ids)


def ursa_anchor_ready() -> bool:
    """Full anchor package including official predictions (post-eval or HPG materialize)."""
    return ursa_anchor_inputs_ready() and all(case_ready(cid) for cid in case_ids())


def ursa_python_candidates() -> List[str]:
    import os

    candidates: List[str] = []
    for env_name in ("URSA_PYTHON", "PYTHON311"):
        val = os.environ.get(env_name)
        if val and val not in candidates:
            candidates.append(val)
    for path in (
        "/apps/python/3.11/bin/python",
        "/apps/python/3.12/bin/python",
        shutil.which("python3.11"),
        shutil.which("python3.12"),
    ):
        if path and path not in candidates:
            candidates.append(path)
    if sys.version_info >= URSA_MIN_PYTHON and sys.executable not in candidates:
        candidates.append(sys.executable)
    return candidates


def ursa_cli_path() -> Optional[str]:
    for name in ("URSA_CLI", "ursa"):
        path = shutil.which(name)
        if path:
            return path
    for py in ursa_python_candidates():
        candidate = Path(py).parent / "ursa"
        if candidate.is_file():
            return str(candidate)
    return None


def install_ursa_from_repo(*, python: Optional[str] = None) -> Tuple[bool, str]:
    repo = ursa_repo_dir()
    if not ursa_repo_ready():
        return False, f"missing ursa repo at {repo}"
    py = python
    if py is None:
        for candidate in ursa_python_candidates():
            try:
                proc = subprocess.run(
                    [candidate, "-c", "import sys; raise SystemExit(0 if sys.version_info>=(3,11) else 1)"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if proc.returncode == 0:
                    py = candidate
                    break
            except (OSError, subprocess.TimeoutExpired):
                continue
    if py is None:
        return False, "no Python >=3.11 found for ursa install (load module python/3.11 on HPG)"
    proc = subprocess.run(
        [
            py,
            "-m",
            "pip",
            "install",
            "-e",
            str(repo),
            f"--extra-index-url={USGS_PYPI_INDEX}",
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-800:]
        return False, f"pip install -e ursa failed: {tail}"
    cli = Path(py).parent / "ursa"
    if cli.is_file():
        return True, str(cli)
    found = shutil.which("ursa")
    return (bool(found), found or "ursa installed but CLI not on PATH")


def all_cases_ready() -> bool:
    ids = case_ids()
    return bool(ids) and all(case_ready(cid) for cid in ids)


def load_case_meta(case_id: str) -> Dict[str, Any]:
    path = case_dir(case_id) / "meta.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
