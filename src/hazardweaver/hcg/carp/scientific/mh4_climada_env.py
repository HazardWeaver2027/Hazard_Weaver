"""Ensure CLIMADA runtime in persistent envs/climada (isolated from pyhazards)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from hazardweaver.hcg.carp.scientific.climada_pin import (
    CLIMADA_COMMIT,
    CLIMADA_REPO_URL,
    CLIMADA_TAG,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
VENDOR_ROOT = PROJECT_ROOT / "data" / "vendor" / "climada"
REPO_DIR = VENDOR_ROOT / "repo"
BOOTSTRAP_OK = VENDOR_ROOT / "BOOTSTRAP_OK.json"
REPO_PIN = VENDOR_ROOT / "REPO_PIN.json"
CLIMADA_ENV = PROJECT_ROOT / "envs" / "climada"

CLIMADA_PIP_EXTRAS = (
    "netcdf4",
    "tables",
    "h5py",
    "sparse",
    "xarray",
    "cartopy",
    "geopandas",
    "fiona",
    "shapely",
    "rasterio",
    "pyarrow",
)


def climada_python() -> Path:
    """Interpreter for CLIMADA replay (persistent conda env under envs/climada)."""
    candidate = CLIMADA_ENV / "bin" / "python"
    if candidate.is_file():
        return candidate
    return Path(sys.executable)


def climada_repo_dir() -> Path:
    return REPO_DIR


def _run(cmd: List[str], *, cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )


def _run_pip(py: Path, args: List[str]) -> subprocess.CompletedProcess[str]:
    return _run([str(py), "-m", "pip", "install", *args])


def verify_runtime_with_python(py: Path) -> List[str]:
    proc = _run(
        [
            str(py),
            "-c",
            "import climada; from climada.engine import Impact; "
            "from climada.entity import LitPop; from climada.hazard import TropCyclone; "
            "from climada.hazard.tc_tracks import TCTracks",
        ],
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "climada import failed").strip()
        return [err.splitlines()[-1] if err else "climada import failed"]
    return []


def verify_runtime() -> List[str]:
    return verify_runtime_with_python(climada_python())


def _read_bootstrap_marker() -> Dict[str, Any] | None:
    if not BOOTSTRAP_OK.is_file():
        return None
    try:
        return json.loads(BOOTSTRAP_OK.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _marker_matches(marker: Dict[str, Any] | None, py: Path) -> bool:
    if not marker:
        return False
    return marker.get("repo_tag") == CLIMADA_TAG and marker.get("python_executable") == str(py.resolve())


def _repo_at_expected_ref() -> bool:
    if not (REPO_DIR / "setup.py").is_file():
        return False
    proc = _run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR)
    if proc.returncode != 0:
        return False
    head = proc.stdout.strip()
    return head == CLIMADA_COMMIT or head.startswith(CLIMADA_COMMIT[:12])


def _ensure_conda_env(py: Path) -> None:
    if py.is_file():
        return
    CLIMADA_ENV.parent.mkdir(parents=True, exist_ok=True)
    create = _run(
        ["bash", "-lc", f"module load conda && conda create -p {CLIMADA_ENV} python=3.10 pip -y"],
        timeout=600,
    )
    if create.returncode != 0:
        raise RuntimeError(
            "conda create envs/climada failed\n"
            f"stdout: {create.stdout.strip()}\nstderr: {create.stderr.strip()}"
        )


def _ensure_gdal(py: Path) -> None:
    probe = _run([str(py), "-c", "from osgeo import gdal; print(gdal.VersionInfo())"])
    if probe.returncode == 0:
        return
    install = _run(
        ["bash", "-lc", f"module load conda && conda install -p {CLIMADA_ENV} -y gdal libgdal -c conda-forge"],
        timeout=900,
    )
    if install.returncode != 0:
        raise RuntimeError(
            "conda install gdal into envs/climada failed\n"
            f"stdout: {install.stdout.strip()}\nstderr: {install.stderr.strip()}"
        )
    probe = _run([str(py), "-c", "from osgeo import gdal; print(gdal.VersionInfo())"])
    if probe.returncode != 0:
        raise RuntimeError(f"osgeo still missing after conda gdal install: {probe.stderr.strip()}")


def _write_pins(commit: str, py: Path) -> None:
    VENDOR_ROOT.mkdir(parents=True, exist_ok=True)
    pin = {
        "repo_url": CLIMADA_REPO_URL,
        "repo_tag": CLIMADA_TAG,
        "repo_commit": commit,
        "pinned_at": datetime.now(timezone.utc).isoformat(),
    }
    REPO_PIN.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    BOOTSTRAP_OK.write_text(
        json.dumps(
            {
                **pin,
                "python_executable": str(py.resolve()),
                "install_mode": "editable_envs_climada",
                "env_path": str(CLIMADA_ENV),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def bootstrap(*, force: bool = False) -> None:
    """Create envs/climada, install gdal via conda, pip install -e climada (one-time)."""
    py = climada_python()
    _ensure_conda_env(py)
    py = climada_python()
    marker = _read_bootstrap_marker()

    if not force and _marker_matches(marker, py) and len(verify_runtime_with_python(py)) == 0:
        print(f"CLIMADA already installed in {py} ({CLIMADA_TAG})")
        return

    _ensure_gdal(py)
    VENDOR_ROOT.mkdir(parents=True, exist_ok=True)
    if force or not _repo_at_expected_ref():
        from scripts.bootstrap.vendor_git import clone_pinned_repo

        clone_pinned_repo(CLIMADA_REPO_URL, REPO_DIR, CLIMADA_TAG, marker_relpath="setup.py")

    commit_proc = _run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR)
    if commit_proc.returncode != 0:
        raise RuntimeError(f"git rev-parse failed: {commit_proc.stderr.strip()}")
    commit = commit_proc.stdout.strip()

    editable = _run_pip(py, ["-e", str(REPO_DIR)])
    if editable.returncode != 0:
        raise RuntimeError(
            "pip install -e climada failed\n"
            f"stdout: {editable.stdout.strip()}\nstderr: {editable.stderr.strip()}"
        )

    extras = _run_pip(py, list(CLIMADA_PIP_EXTRAS))
    if extras.returncode != 0:
        raise RuntimeError(
            "pip install climada extras failed\n"
            f"stdout: {extras.stdout.strip()}\nstderr: {extras.stderr.strip()}"
        )

    errors = verify_runtime_with_python(py)
    if errors:
        raise RuntimeError("CLIMADA bootstrap finished but verify failed: " + "; ".join(errors))

    _write_pins(commit, py)
    print(f"CLIMADA {CLIMADA_TAG} ({commit[:12]}) installed into {py}")

    from hazardweaver.hcg.carp.scientific.mh4_climada_exposure import ensure_conus_exposure

    ensure_conus_exposure()
    print("CLIMADA CONUS exposure cached")


def main() -> int:
    ap = argparse.ArgumentParser(description="MH-4 CLIMADA env (persistent envs/climada)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--force", action="store_true", help="reinstall even if BOOTSTRAP_OK matches")
    args = ap.parse_args()
    if args.bootstrap:
        bootstrap(force=args.force)
    errors = verify_runtime()
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    print(f"climada runtime OK ({climada_python()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
