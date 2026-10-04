"""SHA-256 content hash for HWA_TRAJECTORY_v2.json (DL-110b trajectory integrity)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

TRAJECTORY_FILENAME = "HWA_TRAJECTORY_v2.json"
CONTENT_HASH_SOURCE = "trajectory_v2_sha256"


def compute_trajectory_content_hash(workdir: Path | str) -> str:
    """Return 64-char SHA-256 hex digest of HWA_TRAJECTORY_v2.json file bytes."""
    traj_path = Path(workdir) / TRAJECTORY_FILENAME
    if not traj_path.is_file():
        raise FileNotFoundError(f"missing trajectory artifact: {traj_path}")
    return hashlib.sha256(traj_path.read_bytes()).hexdigest()


def is_valid_content_hash(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text.lower())


def attach_trajectory_content_hash(
    workdir: Path,
    out: Dict[str, Any],
) -> Optional[str]:
    """Compute hash, write run_meta fields, and set out['content_hash']. Returns hash or None."""
    workdir = Path(workdir)
    try:
        content_hash = compute_trajectory_content_hash(workdir)
    except FileNotFoundError:
        out["content_hash"] = None
        out["content_hash_error"] = "missing_trajectory_v2"
        return None

    out["content_hash"] = content_hash
    out["content_hash_source"] = CONTENT_HASH_SOURCE

    meta_path = workdir / "run_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta["content_hash"] = content_hash
    meta["content_hash_source"] = CONTENT_HASH_SOURCE
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return content_hash
