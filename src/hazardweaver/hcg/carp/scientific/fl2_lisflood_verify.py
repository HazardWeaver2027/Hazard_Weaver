"""LISFLOOD-FP executable verification for CAP-FL2-05 (M3)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

from hazardweaver.hcg.carp.scientific.fl2_data import PROJECT_ROOT

LISFLOOD_ROOT = PROJECT_ROOT / "data" / "vendor" / "lisflood_fp"
LISFLOOD_BIN = LISFLOOD_ROOT / "bin"
LISFLOOD_CLI = LISFLOOD_BIN / "lisflood"
LISFLOOD_LIB = LISFLOOD_BIN / "liblisflood.so"
PIN_PATH = LISFLOOD_ROOT / "REPO_PIN.json"


def _liblisflood_paths() -> list[Path]:
    paths = [
        LISFLOOD_LIB,
        LISFLOOD_ROOT / "repo" / "lisflood-fp-bmi-v5.9" / "liblisflood.so",
    ]
    repo = LISFLOOD_ROOT / "repo"
    if repo.is_dir():
        paths.extend(sorted(repo.rglob("liblisflood.so")))
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen and path.is_file():
            seen.add(key)
            out.append(path)
    return out


def _liblisflood_present() -> bool:
    return bool(_liblisflood_paths())


def _lisflood_env() -> dict[str, str]:
    env = os.environ.copy()
    lib_dirs = [str(p.parent) for p in _liblisflood_paths()]
    if LISFLOOD_BIN.is_dir():
        lib_dirs.insert(0, str(LISFLOOD_BIN))
    prefix = os.pathsep.join(dict.fromkeys(lib_dirs))
    env["LD_LIBRARY_PATH"] = (
        f"{prefix}{os.pathsep}{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else prefix
    )
    return env


def _is_stub_script(cli: Path) -> bool:
    if not cli.is_file():
        return True
    try:
        head = cli.read_text(encoding="utf-8", errors="replace")[:400]
    except OSError:
        return True
    return "Wrapper stub" in head or "exit 0" in head


def verify_lisflood_executable() -> Tuple[bool, Dict[str, Any]]:
    lib_paths = _liblisflood_paths()
    report: Dict[str, Any] = {
        "cli_path": str(LISFLOOD_CLI),
        "liblisflood_present": bool(lib_paths),
        "liblisflood_paths": [str(p) for p in lib_paths],
    }
    if not LISFLOOD_CLI.is_file():
        report["ok"] = False
        report["reason"] = "missing lisflood CLI"
        return False, report
    if _is_stub_script(LISFLOOD_CLI):
        report["ok"] = False
        report["reason"] = "stub CLI (not built lisflood)"
        report["stub_detected"] = True
        return False, report
    if not lib_paths:
        report["ok"] = False
        report["reason"] = "liblisflood.so not found under vendor/lisflood_fp"
        return False, report

    try:
        proc = subprocess.run(
            [str(LISFLOOD_CLI), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=_lisflood_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["ok"] = False
        report["reason"] = f"CLI execution failed: {exc}"
        return False, report

    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
    report["returncode"] = proc.returncode
    if proc.returncode != 0 and "usage" not in combined and "lisflood" not in combined:
        report["ok"] = False
        report["reason"] = f"CLI --help failed rc={proc.returncode}"
        return False, report

    report["ok"] = True
    report["executable_verified"] = True
    return True, report
