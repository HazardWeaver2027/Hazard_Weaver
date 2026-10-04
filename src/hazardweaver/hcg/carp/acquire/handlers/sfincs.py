"""SFINCS shared vendor_fetch handler (FL-2 + MH-3)."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.acquire.batch1_artifacts import write_acquisition_manifest, write_env_pin
from hazardweaver.hcg.carp.acquire.pin_repo import pin_repository

SFINCS_REPO = "https://github.com/Deltares/SFINCS"
SFINCS_COMMIT = "v2.4.1"
DOCKER_IMAGE = "deltares/sfincs-cpu"
OFFICIAL_DOCKER_CMD = f"docker run --rm -v $MODEL_DIR:/data -w /data {DOCKER_IMAGE}"


def _clone_sfincs(vendor_root: Path) -> Path:
    repo_dir = vendor_root / "repo"
    if (repo_dir / "README.md").is_file():
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    subprocess.run(
        ["git", "clone", "--depth", "1", SFINCS_REPO, str(repo_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "fetch", "origin", SFINCS_COMMIT, "--depth", "1"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "checkout", SFINCS_COMMIT],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return repo_dir


def _resolve_commit(repo_dir: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    return SFINCS_COMMIT


def _probe_docker(vendor_root: Path) -> Dict[str, Any]:
    status = {"image": DOCKER_IMAGE, "pull_ok": False, "version_ok": False}
    try:
        pull = subprocess.run(
            ["docker", "pull", DOCKER_IMAGE],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        status["pull_ok"] = pull.returncode == 0
        status["pull_stderr"] = pull.stderr[-500:]
        ver = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        status["version_ok"] = ver.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        status["error"] = str(exc)
    pin_path = vendor_root / "DOCKER_PIN.json"
    pin_path.parent.mkdir(parents=True, exist_ok=True)
    pin_path.write_text(json.dumps({**status, "pinned_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    return status


def acquire_sfincs_cap(
    taskpack_id: str,
    capability_id: str,
    *,
    recipe_id: str = "",
    mode: str = "login",
    out_dir: Path,
) -> Dict[str, Any]:
    vendor_root = Path(__file__).resolve().parents[5] / "data" / "vendor" / "sfincs"
    commit = SFINCS_COMMIT
    docker_status: Dict[str, Any] = {"status": "login_stub"}

    if mode == "hpg":
        repo_dir = _clone_sfincs(vendor_root)
        commit = _resolve_commit(repo_dir)
        docker_status = _probe_docker(vendor_root)
        scientific_pin = (
            Path(__file__).resolve().parents[5] / "data" / "scientific" / "MH-3" / "REPO_PIN.json"
        )
        if scientific_pin.parent.is_dir():
            payload = {
                "repo_url": SFINCS_REPO,
                "repo_commit": commit,
                "version_or_commit": commit,
                "docker_image": DOCKER_IMAGE,
                "official_command": OFFICIAL_DOCKER_CMD,
            }
            scientific_pin.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    result = pin_repository(
        out_dir,
        capability_id=capability_id,
        taskpack_id=taskpack_id,
        repo_url=SFINCS_REPO,
        backend="vendor_fetch",
        recipe_id=recipe_id,
        notes=f"SFINCS @ {commit}; Docker {DOCKER_IMAGE}",
        env_extra={
            "repo_commit": commit,
            "docker_image": DOCKER_IMAGE,
            "official_command": OFFICIAL_DOCKER_CMD,
        },
    )
    vendor_root.mkdir(parents=True, exist_ok=True)
    (vendor_root / "REPO_PIN.json").write_text(
        json.dumps(
            {
                "repo_url": SFINCS_REPO,
                "repo_commit": commit,
                "version_or_commit": commit,
                "docker_image": DOCKER_IMAGE,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_env_pin(
        out_dir,
        extra={
            "repo_commit": commit,
            "docker_image": DOCKER_IMAGE,
            "vendor_root": str(vendor_root),
            "docker_probe": docker_status,
        },
    )
    manifest = write_acquisition_manifest(
        out_dir,
        capability_id=capability_id,
        taskpack_id=taskpack_id,
        backend="vendor_fetch",
        recipe_id=recipe_id,
        status="fetched" if mode == "hpg" else "pinned",
        checksum=commit,
        artifact_path=str(vendor_root / "repo"),
        notes=f"official_command={OFFICIAL_DOCKER_CMD}",
    )
    result["status"] = "fetched" if mode == "hpg" else "pinned"
    result["manifest"] = str(manifest)
    result["repo_commit"] = commit
    return result
