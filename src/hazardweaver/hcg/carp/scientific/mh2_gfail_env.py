"""Ensure USGS groundfailure (gfail) runtime is importable for MH-2 scientific replay."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from hazardweaver.hcg.carp.scientific.mh2_data import groundfailure_repo_dir, groundfailure_vendor_root

_GFAIL_MODULES = ("gfail", "libcomcat", "mapio", "rasterio")


def gfail_model_inputs_dir() -> Path:
    return groundfailure_vendor_root() / "model_inputs"


def _bootstrap_model_inputs(repo: Path) -> Path:
    """Stage vendor test fixture inputs (merged Ridgecrest + Loma Prieta coverage)."""
    dest = gfail_model_inputs_dir()
    sources = [
        repo / "tests" / "data" / "ci39462536" / "model_inputs",
        repo / "tests" / "data" / "loma_prieta" / "model_inputs",
    ]
    dest.mkdir(parents=True, exist_ok=True)
    for src in sources:
        if not src.is_dir():
            continue
        for path in src.rglob("*"):
            if path.is_dir():
                continue
            rel = path.relative_to(src)
            target = dest / rel
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return dest


def verify_runtime() -> List[str]:
    errors: List[str] = []
    repo = groundfailure_repo_dir()
    if not repo.is_dir():
        errors.append(f"missing groundfailure repo at {repo}")

    inputs = gfail_model_inputs_dir()
    if not (inputs / "global_grad.tif").is_file():
        errors.append(
            f"missing gfail model_inputs at {inputs}; run install_groundfailure_vendor_deps.sh"
        )

    for name in _GFAIL_MODULES:
        try:
            __import__(name)
        except ImportError as exc:
            errors.append(f"import {name} failed: {exc}")

    script = textwrap.dedent(
        """
        from gfail import Jessee2018Model, Nowicki2014Model
        from gfail.models.godt import godt2008
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        errors.append("gfail model import probe failed")
        if proc.stderr:
            errors.append(proc.stderr[-800:])
    return errors


def _run_pip(args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def pip_install_spec() -> List[str]:
    repo = groundfailure_repo_dir()
    if (repo / "pyproject.toml").is_file():
        return ["-e", str(repo)]
    return ["usgs-libcomcat", "mapio", "configobj", "rasterio", "fiona", "scipy"]


def bootstrap_gfail_deps(*, repair: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {"repo": str(groundfailure_repo_dir())}
    errors = verify_runtime()
    if not errors:
        return {**result, "ok": True, "installed": False}

    repo = groundfailure_repo_dir()
    proc = _run_pip(pip_install_spec())
    result["install_returncode"] = proc.returncode
    if proc.returncode != 0:
        return {
            **result,
            "ok": False,
            "pip_stderr": proc.stderr[-4000:],
            "error": "pip install groundfailure failed",
        }

    try:
        _bootstrap_model_inputs(repo)
    except Exception as exc:
        return {**result, "ok": False, "error": f"model_inputs bootstrap failed: {exc}"}

    if repair:
        _run_pip(["--force-reinstall", "numpy>=1.24,<2.0", "scipy>=1.10"])

    errors = verify_runtime()
    result["errors"] = errors
    result["ok"] = not errors
    result["installed"] = True
    return result


def gfail_version() -> str:
    try:
        import gfail

        return getattr(gfail, "__version__", "1.3.2")
    except Exception:
        pass
    proc = subprocess.run(
        [sys.executable, "-m", "gfail.bin.callgf", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if proc.returncode == 0:
        return "callgf"
    return ""


def write_env_pin() -> Path:
    pin = groundfailure_vendor_root() / "ENV_PIN.json"
    pin.parent.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    payload = {
        "python": sys.version.split()[0],
        "repo_path": str(groundfailure_repo_dir()),
        "model_inputs": str(gfail_model_inputs_dir()),
        "gfail_version": gfail_version() or "unknown",
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    pin.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return pin


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MH-2 groundfailure runtime verify/bootstrap")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--bootstrap", action="store_true")
    args = ap.parse_args(argv)

    if args.bootstrap:
        status = bootstrap_gfail_deps(repair=True)
        if status.get("ok"):
            write_env_pin()
            print("groundfailure runtime ready")
            return 0
        for err in status.get("errors") or [status.get("error")]:
            if err:
                print(err, file=sys.stderr)
        return 1

    errors = verify_runtime()
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    print("groundfailure runtime ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
