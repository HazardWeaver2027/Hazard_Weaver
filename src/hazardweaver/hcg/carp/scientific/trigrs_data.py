"""TRIGRS v2.1 pin and readiness for CAP-L2-05."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.scientific.l2_data import data_root

TRIGRS_REPO = "https://github.com/usgs/landslides-trigrs.git"
TRIGRS_REPO_MIRROR = "https://code.usgs.gov/usgs/landslides-trigrs"
TRIGRS_TAG = "v2.1.0"
TRIGRS_COMMIT = "969c409f"
TRIGRS_RELEASE_NAME = "2.1.0a"
TRIGRS_DOI = "10.5066/F7M044QS"
TRIGRS_ADDENDUM = "docs/engineering/hcg/W3_SCIENTIFIC_COMPLETION_ADDENDUM.md"


def trigrs_pin_path() -> Path:
    return data_root() / "TRIGRS_PIN.json"


def trigrs_vendor_root() -> Path:
    return data_root().parent.parent / "vendor" / "trigrs"


def trigrs_repo_dir() -> Path:
    return trigrs_vendor_root() / "repo"


def os_access_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def trigrs_binary() -> Optional[Path]:
    repo = trigrs_repo_dir()
    candidates = (
        repo / "src" / "TRIGRS" / "trg",
        repo / "src" / "TRIGRS" / "trigrs",
        repo / "trg",
        repo / "trigrs",
        repo / "bin" / "trigrs",
    )
    for candidate in candidates:
        if os_access_executable(candidate):
            return candidate
    return None


def prepare_trigrs_repo() -> None:
    """Symlink Data/ for case-sensitive Linux paths in official tr_in.txt."""
    repo = trigrs_repo_dir()
    data = repo / "data"
    link = repo / "Data"
    if data.is_dir() and not link.exists():
        link.symlink_to("data")


def compile_trigrs() -> bool:
    repo = trigrs_repo_dir()
    if not repo.is_dir():
        return False
    prepare_trigrs_repo()
    if trigrs_binary() is not None:
        return True
    make_dir = repo / "src" / "TRIGRS"
    if not (make_dir / "Makefile").is_file():
        return False
    env = dict(os.environ)
    gsl_lib = os.environ.get("HPC_GSL_LIB") or "/apps/gcc/14.2.0/gsl/2.8.0/lib"
    if Path(gsl_lib).is_dir():
        env["LIBRARY_PATH"] = f"{gsl_lib}:{env.get('LIBRARY_PATH', '')}"
    proc = subprocess.run(
        ["make", "-C", str(make_dir), "trg", "MPIF90=gfortran", "F90=gfortran", "FC=gfortran"],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        log = (proc.stdout or "") + (proc.stderr or "")
        raise RuntimeError(f"TRIGRS make failed: {log[-2000:]}")
    return trigrs_binary() is not None


def write_trigrs_pin(*, archive_sha256: str = "UNKNOWN") -> Path:
    payload = {
        "artifact": "USGS TRIGRS 2.1",
        "release_name": TRIGRS_RELEASE_NAME,
        "repo_url": TRIGRS_REPO,
        "repo_tag": TRIGRS_TAG,
        "repo_commit": TRIGRS_COMMIT,
        "doi": TRIGRS_DOI,
        "archive_sha256": archive_sha256,
        "native_output": "factor_of_safety_grid",
        "primary_eval": "AUPRC(-min_FoS)",
        "execution": "faithful Fortran CLI",
        "addendum_ref": TRIGRS_ADDENDUM,
    }
    path = trigrs_pin_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    vendor_pin = trigrs_vendor_root() / "REPO_PIN.json"
    vendor_pin.parent.mkdir(parents=True, exist_ok=True)
    vendor_pin.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_trigrs_pin() -> Dict[str, Any]:
    path = trigrs_pin_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def clone_trigrs_repo() -> Path:
    from scripts.bootstrap.vendor_git import clone_pinned_repo

    dest = trigrs_repo_dir()
    if (dest / ".git").is_dir() and any(dest.iterdir()):
        prepare_trigrs_repo()
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    out = clone_pinned_repo(TRIGRS_REPO, dest, TRIGRS_TAG)
    prepare_trigrs_repo()
    return out


def trigrs_data_ready() -> bool:
    pin = trigrs_pin_path()
    if not pin.is_file():
        return False
    data = load_trigrs_pin()
    commit = str(data.get("repo_commit") or "")
    if not commit or commit.upper() == "UNKNOWN":
        return False
    repo = trigrs_repo_dir()
    if not repo.is_dir() or not any(repo.iterdir()):
        return False
    if trigrs_binary() is not None:
        return True
    from hazardweaver.hcg.carp.scientific.l2_data import event_dir, event_ids

    for eid in event_ids("official_test"):
        fos = event_dir("official_test", eid) / "predictions" / "factor_of_safety.npy"
        prov = event_dir("official_test", eid) / "predictions" / "trigrs_provenance.json"
        if fos.is_file() and prov.is_file():
            return True
    return False
