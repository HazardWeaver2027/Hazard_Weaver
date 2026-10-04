"""Per-trace checkpoint I/O for resumable E1-E3 picker scientific replay."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def checkpoint_dir_for(cap_dir: Path, split: str) -> Path:
    sub = "test" if split == "official_test" else "holdout"
    return cap_dir / "predictions" / sub / "checkpoints"


def progress_path_for(cap_dir: Path, split: str) -> Path:
    sub = "test" if split == "official_test" else "holdout"
    return cap_dir / "predictions" / sub / "progress.json"


def checkpoint_path(checkpoint_dir: Path, trace_id: str) -> Path:
    safe = trace_id.replace("/", "_")
    return checkpoint_dir / f"{safe}.json"


def load_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("ok") else None


def write_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".ckpt_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_progress(progress_path: Path, *, completed: int, total: int, split: str) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": split, "completed": completed, "total": total}
    progress_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def collect_predictions(checkpoint_dir: Path) -> List[Dict[str, Any]]:
    if not checkpoint_dir.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("*.json")):
        row = load_checkpoint(path)
        if not row:
            continue
        rows.append(
            {
                "trace_id": row.get("trace_id"),
                "pred_times": row.get("pred_times", []),
                "truth_times": row.get("truth_times", []),
                "pick_f1": float(row.get("pick_f1", 0.0)),
            }
        )
    return rows


def aggregate_f1(predictions: List[Dict[str, Any]]) -> float:
    if not predictions:
        return 0.0
    f1s = [float(row.get("pick_f1", 0.0)) for row in predictions]
    return float(sum(f1s) / len(f1s))
