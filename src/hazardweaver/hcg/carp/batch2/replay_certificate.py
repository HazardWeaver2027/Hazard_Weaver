"""Replay certificate + metric artifacts for Batch 2 L2."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.batch2 import BATCH2_ROOT, FIXTURES_ROOT


def cap_dir(taskpack_id: str, capability_id: str, *, base: Optional[Path] = None) -> Path:
    root = base or BATCH2_ROOT
    return root / taskpack_id / capability_id


def write_replay_manifest(
    taskpack_id: str,
    capability_id: str,
    *,
    family_id: str,
    exec_ok: bool,
    metric_name: str = "",
    metric_value: Optional[float] = None,
    protocol: str = "native-route",
    notes: str = "",
    extra: Optional[Dict[str, Any]] = None,
    base: Optional[Path] = None,
) -> Path:
    out = cap_dir(taskpack_id, capability_id, base=base)
    out.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "capability_id": capability_id,
        "taskpack_id": taskpack_id,
        "family_id": family_id,
        "exec_ok": exec_ok,
        "protocol": protocol,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "notes": notes,
        "batch": "batch2",
    }
    if extra:
        payload.update(extra)
    path = out / "replay_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_metrics(
    taskpack_id: str,
    capability_id: str,
    metrics: Dict[str, Any],
    *,
    base: Optional[Path] = None,
) -> Path:
    out = cap_dir(taskpack_id, capability_id, base=base)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "native_metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return path


def load_replay_manifest(taskpack_id: str, capability_id: str, *, base: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = cap_dir(taskpack_id, capability_id, base=base) / "replay_manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def publish_canonical(
    taskpack_id: str,
    capability_id: str,
    *,
    job_base: Path,
    canonical_base: Optional[Path] = None,
) -> Optional[Path]:
    """Copy exec_ok batch2 artifacts from HPG job dir to canonical runs/carp/batch2/."""
    replay = load_replay_manifest(taskpack_id, capability_id, base=job_base)
    if not replay or not replay.get("exec_ok"):
        return None
    src = cap_dir(taskpack_id, capability_id, base=job_base)
    dst = cap_dir(taskpack_id, capability_id, base=canonical_base or BATCH2_ROOT)
    if src.resolve() == dst.resolve():
        return dst if dst.is_dir() else None
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst
