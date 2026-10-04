"""Quarantine stale episode artifacts before forced headline-cell reruns (DL-206).

Legacy hygiene stripped keys from ``run_meta`` in-place, leaving orphan half-meta
that strict DCA scored as ``pick=None``. Force reruns must quarantine **all**
episode artifacts including ``run_meta.json``, and seal terminal state on failure.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_STALE_EPISODE_ARTIFACTS = (
    "answer.json",
    "controller_decisions.jsonl",
    "dca_result.json",
    "transcript.jsonl",
    "hwa_trajectory_v2_ledger.jsonl",
    "HWA_TRAJECTORY_v2.json",
    "tool_calls.jsonl",
    "trajectory.jsonl",
    "run_meta.json",
)

_INCOMPLETE_EXIT_REASONS = frozenset(
    {
        "limit_steps",
        "limit_wall",
        "parse_loop",
        "failed_clarify",
        "env_null_no_user",
        "exception",
        "timeout",
        "max_wall_s",
        "engineering_blocked",
    }
)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _is_orphan_incomplete_meta(meta: Dict[str, Any]) -> bool:
    if meta.get("rerun_quarantine_utc"):
        return True
    if meta.get("submitted"):
        return False
    exit_reason = str(meta.get("exit_reason") or "")
    if not exit_reason:
        return True
    return exit_reason in _INCOMPLETE_EXIT_REASONS


def _quarantine_file(workdir: Path, name: str, quarantine_dir: Optional[Path], stamp: str) -> tuple[Optional[Path], bool]:
    path = workdir / name
    if not path.is_file():
        return quarantine_dir, False
    if quarantine_dir is None:
        quarantine_dir = workdir / f"_quarantine_rerun_{stamp}"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
    path.rename(quarantine_dir / name)
    return quarantine_dir, True


def quarantine_stale_headline_cell_artifacts(workdir: Path) -> Dict[str, Any]:
    """Move prior episode files aside so a forced rerun cannot rescore stale answers."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    moved: List[str] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_dir: Optional[Path] = None

    for name in _STALE_EPISODE_ARTIFACTS:
        quarantine_dir, did_move = _quarantine_file(workdir, name, quarantine_dir, stamp)
        if did_move:
            moved.append(name)

    meta_path = workdir / "run_meta.json"
    if meta_path.is_file():
        meta = _read_json(meta_path)
        if _is_orphan_incomplete_meta(meta):
            quarantine_dir, did_move = _quarantine_file(
                workdir, "run_meta.json", quarantine_dir, stamp
            )
            if did_move and "run_meta.json" not in moved:
                moved.append("run_meta.json")

    return {
        "workdir": str(workdir),
        "quarantine_dir": str(quarantine_dir) if quarantine_dir else None,
        "moved": moved,
    }


def seal_headline_cell_incomplete(
    workdir: Path,
    *,
    exit_reason: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Write terminal incomplete ``run_meta`` + invalid ``dca_result`` (no orphan half-meta)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    meta = _read_json(workdir / "run_meta.json")
    meta.update(
        {
            "submitted": False,
            "exit_reason": str(exit_reason or "incomplete_episode"),
            "sealed_by": "headline_cell_rerun_hygiene_v1",
            "sealed_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    if error:
        meta["seal_error"] = str(error)[:500]
    (workdir / "run_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    dca_path = workdir / "dca_result.json"
    if not dca_path.is_file():
        inv = _read_json(workdir / "inventory_row.json")
        dca = {
            "valid": False,
            "outcome": "invalid",
            "reason_code": "incomplete_episode",
            "error": str(error or exit_reason or "incomplete_episode"),
            "taskpack_id": inv.get("taskpack_id"),
            "sealed_by": "headline_cell_rerun_hygiene_v1",
        }
        dca_path.write_text(json.dumps(dca, indent=2) + "\n", encoding="utf-8")
    return {
        "workdir": str(workdir),
        "exit_reason": meta.get("exit_reason"),
        "sealed": True,
    }


def prepare_workdir_for_force_rerun(workdir: Path) -> Dict[str, Any]:
    """Entry point for ``ICLR_FORCE_RERUN``: quarantine all stale episode state."""
    return quarantine_stale_headline_cell_artifacts(workdir)
