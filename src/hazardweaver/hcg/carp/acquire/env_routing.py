"""Batch 1 taskpack → Python environment routing (GPL-isolated SeisBench worker)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hazardweaver.hcg.carp.acquire.seisbench_paths import (
    SEISBENCH_CACHE_DIR,
    SEISBENCH_VENDOR_ROOT,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PYHAZARDS_PYTHON = PROJECT_ROOT / "envs" / "pyhazards" / "bin" / "python"
WSTS_PYTHON = PROJECT_ROOT / "envs" / "wsts" / "bin" / "python"
SEISBENCH_PYTHON = PROJECT_ROOT / "envs" / "seisbench" / "bin" / "python"
SEISBENCH_BOOTSTRAP = PROJECT_ROOT / "scripts" / "bootstrap" / "create_seisbench_env.sh"
SEISBENCH_WORKER = PROJECT_ROOT / "experiments" / "hcg" / "carp" / "acquire" / "seisbench_fetch_worker.py"

# WF-3 train uses isolated envs/wsts; E1-E3 weights use seisbench worker; default orchestrator=pyhazards.
TASKPACK_PYTHON: Dict[str, Path] = {
    "WF-3": WSTS_PYTHON,
}

TASKPACK_HPG_CHECKS: Dict[str, List[Tuple[str, str]]] = {
    "WF-3": [
        ("import torch", "torch"),
        (
            "import jsonargparse, jsonargparse.signatures; "
            "from packaging.version import Version; "
            "assert Version(jsonargparse.__version__) >= Version('4.27.7')",
            "jsonargparse[signatures]>=4.27.7",
        ),
        ("from pytorch_lightning.cli import LightningCLI", "pytorch-lightning"),
        ("import wandb", "wandb"),
        ("import segmentation_models_pytorch", "segmentation_models_pytorch"),
    ],
}


def python_for_taskpack(taskpack_id: str) -> Path:
    return TASKPACK_PYTHON.get(taskpack_id, PYHAZARDS_PYTHON)


def _run_import_check(python: Path, stmt: str) -> Tuple[bool, str]:
    if not python.is_file():
        return False, f"missing interpreter: {python}"
    try:
        proc = subprocess.run(
            [str(python), "-c", stmt],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"import check timed out: {stmt}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, err[-1] if err else f"import failed: {stmt}"
    return True, "ok"


def _preflight_seisbench_worker() -> List[str]:
    issues: List[str] = []
    if not SEISBENCH_PYTHON.is_file():
        issues.append(f"seisbench env missing: {SEISBENCH_PYTHON}. Run: bash {SEISBENCH_BOOTSTRAP}")
        return issues
    if not SEISBENCH_WORKER.is_file():
        issues.append(f"seisbench worker missing: {SEISBENCH_WORKER}")
        return issues
    ok, msg = _run_import_check(SEISBENCH_PYTHON, "import seisbench")
    if not ok:
        issues.append(f"seisbench: {msg}")
    return issues


def assert_hpg_interpreter(taskpack_id: str) -> Optional[str]:
    """Return error string if current process is not the routed HPG interpreter."""
    if taskpack_id not in TASKPACK_PYTHON:
        return None
    expected = python_for_taskpack(taskpack_id)
    if expected.resolve() != Path(sys.executable).resolve():
        return (
            f"wrong interpreter for {taskpack_id}: expected {expected}, got {sys.executable}"
        )
    return None


def preflight_taskpack(taskpack_id: str, *, mode: str = "hpg") -> Dict[str, Any]:
    """or HPG: verify interpreter + imports before acquire."""
    if mode == "login":
        return {"ok": True, "taskpack_id": taskpack_id, "mode": mode, "skipped": True}

    python = python_for_taskpack(taskpack_id)
    issues: List[str] = []

    if not python.is_file():
        issues.append(f"interpreter missing: {python}")

    for stmt, label in TASKPACK_HPG_CHECKS.get(taskpack_id, []):
        ok, msg = _run_import_check(python, stmt)
        if not ok:
            issues.append(f"{label}: {msg}")

    if taskpack_id == "E1-E3":
        issues.extend(_preflight_seisbench_worker())

    return {
        "ok": not issues,
        "taskpack_id": taskpack_id,
        "mode": mode,
        "python": str(python),
        "seisbench_worker_python": str(SEISBENCH_PYTHON),
        "issues": issues,
    }


def preflight_all(taskpack_ids: List[str], *, mode: str = "hpg") -> Dict[str, Any]:
    tracks = {tid: preflight_taskpack(tid, mode=mode) for tid in taskpack_ids}
    ok = all(t.get("ok") for t in tracks.values())
    return {"ok": ok, "mode": mode, "tracks": tracks}
