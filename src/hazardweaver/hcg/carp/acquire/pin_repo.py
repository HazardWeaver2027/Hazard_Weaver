"""Utility module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.acquire.batch1_artifacts import write_acquisition_manifest, write_env_pin


def pin_repository(
    out_dir: Path,
    *,
    capability_id: str,
    taskpack_id: str,
    repo_url: str,
    backend: str = "official_repo_replay",
    recipe_id: str = "",
    commit: str = "UNKNOWN",
    official_command: str = "UNKNOWN—resolve from official README/config",
    notes: str = "",
    env_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env_pin = write_env_pin(out_dir, extra={"repo_url": repo_url, "repo_commit": commit, **(env_extra or {})})
    manifest = write_acquisition_manifest(
        out_dir,
        capability_id=capability_id,
        taskpack_id=taskpack_id,
        backend=backend,
        recipe_id=recipe_id,
        status="pinned_stub",
        checksum="UNKNOWN",
        official_command=official_command,
        artifact_path=repo_url,
        notes=notes or "Repo pin only; HPG must record commit SHA and run official command",
    )
    return {
        "ok": True,
        "status": "pinned_stub",
        "env_pin": str(env_pin),
        "manifest": str(manifest),
        "repo_url": repo_url,
        "repo_commit": commit,
    }
