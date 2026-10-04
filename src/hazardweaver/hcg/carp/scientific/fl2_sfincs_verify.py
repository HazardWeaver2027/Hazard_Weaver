"""SFINCS container executable verification for CAP-FL2-04."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

from hazardweaver.hcg.carp.scientific.fl2_data import PROJECT_ROOT

SFINCS_ROOT = PROJECT_ROOT / "data" / "vendor" / "sfincs"
SFINCS_SIF = SFINCS_ROOT / "sfincs-cpu.sif"
REPO_PIN = SFINCS_ROOT / "REPO_PIN.json"


def sfincs_sif_path() -> Path:
    return SFINCS_SIF


def sfincs_sif_present() -> bool:
    return SFINCS_SIF.is_file() and SFINCS_SIF.stat().st_size > 1_000_000


def verify_sfincs_executable() -> Tuple[bool, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "sif_path": str(SFINCS_SIF),
        "sif_present": sfincs_sif_present(),
    }
    if not sfincs_sif_present():
        report["ok"] = False
        report["reason"] = "missing sfincs-cpu.sif (run run_fl2_vendor_build_verify.sbatch)"
        return False, report

    if REPO_PIN.is_file():
        try:
            report["repo_pin"] = json.loads(REPO_PIN.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report["repo_pin"] = {"parse_error": True}

    cmd = [
        "singularity",
        "exec",
        str(SFINCS_SIF),
        "sfincs",
        "--help",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        report["ok"] = False
        report["reason"] = f"singularity exec failed: {exc}"
        return False, report

    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    report["returncode"] = proc.returncode
    report["stdout_head"] = combined[:400]
    if "SFINCS" not in combined:
        report["ok"] = False
        report["reason"] = "sfincs banner not found in singularity exec output"
        return False, report

    report["ok"] = True
    report["executable_verified"] = True
    report["runtime"] = "singularity"
    return True, report
