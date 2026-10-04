"""Standard Batch 1 acquisition artifact layout under runs/carp/batch1/."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BATCH1_ROOT = PROJECT_ROOT / "runs" / "carp" / "batch1"


def cap_out_dir(taskpack_id: str, capability_id: str, *, base: Optional[Path] = None) -> Path:
    root = base or BATCH1_ROOT
    return root / taskpack_id / capability_id


def default_env_pin(*, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    pin: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "checksum_status": "UNKNOWN",
    }
    if extra:
        pin.update(extra)
    return pin


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_env_pin(out_dir: Path, *, extra: Optional[Dict[str, Any]] = None) -> Path:
    return write_json(out_dir / "ENV_PIN.json", default_env_pin(extra=extra))


def write_acquisition_manifest(
    out_dir: Path,
    *,
    capability_id: str,
    taskpack_id: str,
    backend: str,
    recipe_id: str = "",
    status: str = "pinned_stub",
    checksum: str = "UNKNOWN",
    official_command: str = "",
    artifact_path: str = "",
    notes: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    payload: Dict[str, Any] = {
        "capability_id": capability_id,
        "taskpack_id": taskpack_id,
        "backend": backend,
        "recipe_id": recipe_id,
        "status": status,
        "checksum": checksum,
        "official_command": official_command,
        "artifact_path": artifact_path,
        "notes": notes,
    }
    if extra:
        payload.update(extra)
    return write_json(out_dir / "ACQUISITION_MANIFEST.json", payload)


def validation_level_from_result(result: Dict[str, Any]) -> str:
    if result.get("ok") and result.get("status") in ("fetched", "present", "replay_ok"):
        return "L1"
    if result.get("ok"):
        return "L0"
    return "L0"
