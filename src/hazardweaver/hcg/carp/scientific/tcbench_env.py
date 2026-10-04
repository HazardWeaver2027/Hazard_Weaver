"""Ensure TCBench official evaluator dependencies are importable in pyhazards."""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from hazardweaver.hcg.carp.scientific.tc_data import tcbench_repo_dir

_EVAL_REQUIREMENTS = Path(__file__).resolve().with_name("tcbench_eval_requirements.txt")
_UPSTREAM_REQUIREMENTS = tcbench_repo_dir() / "requirements.txt"
MAX_NUMPY = "2.0.0"

_OPTIONAL_MODULES = (
    "pint",
    "dask",
    "distributed",
    "dask_ml",
    "xarray",
    "cartopy",
    "joblib",
    "torch",
    "scipy",
    "zarr",
    "metpy",
    "sklearn",
)


def eval_requirements_path() -> Path:
    if _EVAL_REQUIREMENTS.is_file():
        return _EVAL_REQUIREMENTS
    return _UPSTREAM_REQUIREMENTS


def requirements_path() -> Path:
    """Backward-compatible alias; always prefer curated eval requirements."""
    return eval_requirements_path()


def _run_pip(args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def verify_numpy_pandas_abi() -> List[str]:
    errors: List[str] = []
    try:
        import numpy as np
        from packaging.version import Version

        if Version(np.__version__) >= Version(MAX_NUMPY):
            errors.append(
                f"numpy {np.__version__} >= {MAX_NUMPY} breaks pyhazards ABI; "
                "run scripts/bootstrap/install_tcbench_vendor_deps.sh"
            )
    except Exception as exc:
        errors.append(f"numpy check failed: {exc}")
        return errors

    try:
        import pandas as pd  # noqa: F401
    except Exception as exc:
        errors.append(f"pandas ABI/import failed: {exc}")
    return errors


def verify_runtime() -> List[str]:
    """Return human-readable errors; empty list means runtime is ready."""
    errors = list(verify_numpy_pandas_abi())
    if errors:
        return errors

    probe = verify_tcbench_evaluator_import()
    if probe.get("ok"):
        return []

    errors.append(probe.get("error") or "evaluate_tracks import probe failed")
    stderr = probe.get("stderr")
    if stderr:
        errors.append(str(stderr)[-800:])
    return errors


def missing_modules() -> List[str]:
    missing: List[str] = []
    for name in _OPTIONAL_MODULES:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
        except Exception:
            missing.append(name)
    return missing


def verify_tcbench_evaluator_import() -> Dict[str, Any]:
    repo = tcbench_repo_dir()
    dev = repo / "dev"
    if not (dev / "evaluate_tracks.py").is_file():
        return {"ok": False, "error": f"missing TCBench dev/evaluate_tracks.py under {repo}"}
    script = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {str(dev)!r})
        os.chdir({str(dev)!r})
        import evaluate_tracks  # noqa: F401
        import metrics_test  # noqa: F401
        from utils import toolbox  # noqa: F401
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
        return {
            "ok": False,
            "error": "evaluate_tracks import probe failed",
            "stderr": proc.stderr[-4000:],
            "stdout": proc.stdout[-1000:],
        }
    return {"ok": True}


def pip_install_spec() -> List[str]:
    req = eval_requirements_path()
    if req.is_file():
        return ["-r", str(req)]
    return ["Pint>=0.24,<0.25", "cartopy>=0.22.0", "dask[distributed]>=2024.1.0", "dask-ml>=2024.1.0"]


def pip_repair_spec() -> List[str]:
    """Force-reinstall ABI-sensitive wheels after accidental numpy drift."""
    return [
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "xarray>=2023.1.0",
        "shapely>=2.0.0",
        "cartopy>=0.24.1",
    ]


def bootstrap_tcbench_eval_deps(*, repair: bool = False) -> Dict[str, Any]:
    """Install/repair curated deps into the active interpreter (login bootstrap)."""
    result: Dict[str, Any] = {"requirements": str(eval_requirements_path())}

    errors = verify_runtime()
    if not errors:
        return {**result, "ok": True, "installed": False}

    proc = _run_pip(pip_install_spec())
    result["install_returncode"] = proc.returncode
    if proc.returncode != 0:
        return {
            **result,
            "ok": False,
            "pip_stderr": proc.stderr[-4000:],
            "pip_stdout": proc.stdout[-2000:],
            "error": "pip install failed",
        }

    errors = verify_runtime()
    if errors and repair:
        repair_proc = _run_pip(["--force-reinstall", *pip_repair_spec()])
        result["repair_returncode"] = repair_proc.returncode
        if repair_proc.returncode != 0:
            return {
                **result,
                "ok": False,
                "pip_stderr": repair_proc.stderr[-4000:],
                "error": "pip repair failed",
            }
        errors = verify_runtime()

    result["errors"] = errors
    result["ok"] = not errors
    result["installed"] = True
    return result


def ensure_tcbench_eval_deps(*, install: bool = True) -> Dict[str, Any]:
    """Verify runtime; optionally bootstrap on login-style install=True."""
    req = eval_requirements_path()
    result: Dict[str, Any] = {"requirements": str(req)}

    errors = verify_runtime()
    if not errors:
        return {**result, "ok": True, "installed": False, "missing": []}

    result["errors"] = errors
    result["missing"] = missing_modules()
    if not install:
        return {
            **result,
            "ok": False,
            "installed": False,
            "error": errors[0],
        }

    boot = bootstrap_tcbench_eval_deps(repair=True)
    result.update(boot)
    if boot.get("ok"):
        result["installed"] = True
        result["missing"] = []
    return result


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TCBench evaluator runtime verify/bootstrap")
    ap.add_argument("--verify-only", action="store_true", help="Exit 0 iff runtime ready")
    ap.add_argument("--bootstrap", action="store_true", help="Install/repair deps in active env")
    ap.add_argument("--print-install-args", action="store_true")
    ap.add_argument("--print-repair-args", action="store_true")
    args = ap.parse_args(argv)

    if args.print_install_args:
        spec = pip_install_spec()
        print(" ".join(spec))
        return 0
    if args.print_repair_args:
        print(" ".join(pip_repair_spec()))
        return 0
    if args.bootstrap:
        status = bootstrap_tcbench_eval_deps(repair=True)
        if not status.get("ok"):
            for err in status.get("errors") or [status.get("error")]:
                if err:
                    print(err, file=sys.stderr)
            return 1
        print("TCBench evaluator runtime ready")
        return 0

    errors = verify_runtime()
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    print("TCBench evaluator runtime ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
