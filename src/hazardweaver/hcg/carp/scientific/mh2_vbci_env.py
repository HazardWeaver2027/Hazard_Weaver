"""VBCI (CAP-MH2-05) vendor paths and runtime checks."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hazardweaver.hcg.carp.scientific.mh2_data import project_root, vbci_pin_path, vbci_repo_dir, vbci_vendor_root

VBCI_REPO_URL = "https://github.com/SusuXu/VBCI.git"
VBCI_REPO_COMMIT = "main"
ZENODO_RECORD = "10.5281/zenodo.7319726"
VBCI_METHOD_FILES = ("updating.m", "SVI.m", "pruning.m")


def vbci_method_dir() -> Path:
    repo = vbci_repo_dir()
    method = repo / "code" / "method"
    if method.is_dir():
        return method
    return repo


def vbci_runner() -> str:
    """Prefer MATLAB — VBCI official code targets MATLAB."""
    return shutil.which("matlab") or shutil.which("octave") or ""


def vbci_runner_kind(runner: Optional[str] = None) -> str:
    path = Path(runner or vbci_runner())
    name = path.name.lower()
    if name == "matlab":
        return "matlab"
    if name == "octave":
        return "octave"
    return ""


def build_vbci_eval_argv(
    *,
    case_dir: Path,
    method_dir: Path,
    script_parent: Path,
    runner: Optional[str] = None,
) -> Tuple[List[str], str]:
    """Return argv + runner kind for official VBCI SVI replay."""
    exe = runner or vbci_runner()
    kind = vbci_runner_kind(exe)
    if not exe or not kind:
        raise RuntimeError("neither matlab nor octave in PATH for VBCI replay")

    method_s = str(method_dir.resolve())
    script_s = str(script_parent.resolve())
    case_s = str(case_dir.resolve())
    body = (
        f"addpath('{method_s}'); "
        f"addpath('{script_s}'); "
        f"cd('{case_s}'); "
        f"ridgecrest_vbci_replay;"
    )
    if kind == "matlab":
        return [exe, "-batch", body], kind
    return [exe, "--no-gui", "--eval", body], kind


def verify_vbci_runtime(*, require_runner: bool = False) -> List[str]:
    errors: List[str] = []
    method = vbci_method_dir()
    for name in VBCI_METHOD_FILES:
        if not (method / name).is_file():
            errors.append(f"missing VBCI {name} under {method}")
    script = project_root() / "scripts" / "vbci" / "ridgecrest_vbci_replay.m"
    if not script.is_file():
        errors.append(f"missing replay script {script}")
    runner = vbci_runner()
    if require_runner and not runner:
        errors.append("neither matlab nor octave in PATH for VBCI replay")
    return errors


def write_vbci_pin() -> Path:
    pin = vbci_pin_path()
    pin.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_url": VBCI_REPO_URL,
        "repo_commit": VBCI_REPO_COMMIT,
        "zenodo_doi": ZENODO_RECORD,
        "official_entrypoint": "updating.m",
        "pinned_at": datetime.now(timezone.utc).isoformat(),
    }
    pin.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return pin


def bootstrap_vbci_repo() -> Dict[str, Any]:
    import sys

    root = Path(__file__).resolve().parents[4]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.bootstrap.vendor_git import clone_pinned_repo

    dest = clone_pinned_repo(
        VBCI_REPO_URL,
        vbci_repo_dir(),
        VBCI_REPO_COMMIT,
        marker_relpath="code/method/updating.m",
    )
    write_vbci_pin()
    errors = verify_vbci_runtime()
    return {"ok": not errors, "repo": str(dest), "errors": errors}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--require-runner", action="store_true", help="fail if matlab/octave missing")
    args = ap.parse_args()
    if args.bootstrap:
        status = bootstrap_vbci_repo()
        if not status.get("ok"):
            for e in status.get("errors") or []:
                print(e)
            return 1
        print("VBCI vendor ready")
        return 0
    errors = verify_vbci_runtime(require_runner=args.require_runner)
    if errors:
        for e in errors:
            print(e)
        return 1
    print(f"VBCI vendor ready (runner={vbci_runner_kind() or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
