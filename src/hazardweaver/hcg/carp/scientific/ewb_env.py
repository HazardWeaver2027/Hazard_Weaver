"""Isolated ExtremeWeatherBench env (Python >=3.11) for HW-MED scientific replay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from hazardweaver.hcg.carp.scientific.paths import PROJECT_ROOT

EWB_REPO = "https://github.com/brightbandtech/ExtremeWeatherBench"
EWB_COMMIT = "2f37abd464eb339d3775e2e1537e2f823d87efad"
EWB_ENV_DIR = PROJECT_ROOT / "envs" / "ewb"
EWB_DATA_ROOT = PROJECT_ROOT / "data" / "scientific" / "HW-MED"
EWB_VENDOR_ROOT = PROJECT_ROOT / "data" / "vendor" / "ewb"

CIRA_MODELS_REQUIRED = (
    "FOUR_v200_GFS",
    "PANG_v100_GFS",
    "GRAP_v100_GFS",
)

_PROBE_MODULES = ("extremeweatherbench", "xarray", "dask")


def ewb_python() -> Path:
    override = os.environ.get("HW_EWB_PYTHON", "").strip()
    if override:
        return Path(override)
    return EWB_ENV_DIR / "bin" / "python"


def create_env_script() -> Path:
    return PROJECT_ROOT / "scripts" / "bootstrap" / "create_ewb_env.sh"


def missing_modules(python: Path | None = None) -> List[str]:
    py = python or ewb_python()
    if not py.is_file():
        return list(_PROBE_MODULES)
    missing: List[str] = []
    for name in _PROBE_MODULES:
        proc = subprocess.run(
            [str(py), "-c", f"import {name}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            missing.append(name)
    return missing


def _ewb_probe_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    creds = os.environ.get("HW_GCP_CREDENTIALS", "").strip()
    if creds:
        env["GOOGLE_APPLICATION_CREDENTIALS"] = creds
    else:
        env.setdefault("GCSFS_TOKEN", "anon")
    return env


def verify_cira_forecast_subset(python: Path | None = None) -> List[str]:
    """One-case CIRA subset probe — catches missing anon GCS before 4h eval."""
    py = python or ewb_python()
    if not py.is_file():
        return ["ewb python missing"]
    events = PROJECT_ROOT / "data" / "vendor" / "ewb" / "repo" / "src" / "extremeweatherbench" / "data" / "events.yaml"
    if not events.is_file():
        return [f"events.yaml missing: {events}"]
    script = f"""
import json
import os
os.environ.setdefault("GCSFS_TOKEN", "anon")
import xarray as xr
from extremeweatherbench import defaults, inputs, cases

_orig = inputs.check_for_missing_data
def _ewb_check(data, case_metadata, source_module=None):
    if isinstance(data, xr.Dataset) and "init_time" in data.coords and "lead_time" in data.coords:
        return True
    return _orig(data, case_metadata, source_module=source_module)
inputs.check_for_missing_data = _ewb_check

fc = defaults.cira_fcnv2_heatwave_forecast
case_list = cases.load_individual_cases_from_yaml({str(events)!r})
heat = [c for c in case_list if c.event_type == "heat_wave"]
if not heat:
    print(json.dumps({{"ok": False, "error": "no heat_wave cases"}}))
else:
    probe_n = 0
    probe_case = None
    for c in heat:
        n = int(fc.subset_data_to_case(fc.ds, c).sizes.get("valid_time", 0))
        if n > 0:
            probe_n = n
            probe_case = c.case_id_number
            break
    print(json.dumps({{"ok": probe_n > 0, "case_id": probe_case, "valid_time": probe_n}}))
"""
    proc = subprocess.run(
        [str(py), "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=_ewb_probe_env(),
        timeout=120,
    )
    if proc.returncode != 0:
        return [f"CIRA subset probe failed: {proc.stderr[-500:]}"]
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return [f"CIRA subset probe bad output: {proc.stdout[-300:]}"]
    if not data.get("ok"):
        return [
            "CIRA forecast subset empty (set GCSFS_TOKEN=anon or HW_GCP_CREDENTIALS); "
            f"probe={data}"
        ]
    return []


def verify_cira_models(python: Path | None = None) -> List[str]:
    """Check EWB exposes required CIRA model names for HW-MED caps 02-04."""
    py = python or ewb_python()
    if not py.is_file():
        return ["ewb python missing"]
    script = """
import json
from extremeweatherbench import inputs
names = getattr(inputs, "CIRA_MODEL_NAMES", None)
if names is None:
    print(json.dumps({"ok": False, "error": "CIRA_MODEL_NAMES missing"}))
else:
    if isinstance(names, dict):
        available = set(names.keys()) | set(names.values())
    else:
        available = set(names)
    missing = [m for m in {models!r} if m not in available]
    print(json.dumps({"ok": not missing, "missing": missing, "sample": sorted(list(available))[:8]}))
""".replace(
        "{models!r}", repr(list(CIRA_MODELS_REQUIRED))
    )
    proc = subprocess.run(
        [str(py), "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=_ewb_probe_env(),
    )
    if proc.returncode != 0:
        return [f"CIRA probe failed: {proc.stderr[-500:]}"]
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return [f"CIRA probe bad output: {proc.stdout[-300:]}"]
    if not data.get("ok"):
        missing = data.get("missing") or [data.get("error")]
        return [f"CIRA models unavailable: {missing}"]
    return []


def verify_runtime(*, check_cira: bool = True) -> List[str]:
    errors: List[str] = []
    py = ewb_python()
    if not py.is_file():
        errors.append(f"EWB env missing: {py}; run bash {create_env_script()}")
        return errors
    missing = missing_modules(py)
    if missing:
        errors.append(f"EWB imports missing: {missing}")
        return errors
    if check_cira:
        errors.extend(verify_cira_models(py))
        errors.extend(verify_cira_forecast_subset(py))
    return errors


def write_env_pin(*, python: Path | None = None) -> Path:
    py = python or ewb_python()
    proc = subprocess.run(
        [str(py), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True,
        text=True,
        check=False,
    )
    py_minor = proc.stdout.strip() if proc.returncode == 0 else "unknown"
    payload = {
        "python_env": str(EWB_ENV_DIR),
        "python_executable": str(py),
        "python_version": py_minor,
        "ewb_repo": EWB_REPO,
        "ewb_commit": EWB_COMMIT,
        "install_command": f"bash {create_env_script()}",
        "note": "HW-MED scientific eval requires Python>=3.11; orchestrator stays on envs/pyhazards",
        "pinned_at": datetime.now(timezone.utc).isoformat(),
    }
    EWB_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    path = EWB_DATA_ROOT / "ENV_PIN.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    EWB_VENDOR_ROOT.mkdir(parents=True, exist_ok=True)
    vendor_path = EWB_VENDOR_ROOT / "ENV_PIN.json"
    vendor_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def ensure_ewb_env(*, install: bool = True) -> Dict[str, Any]:
    """Ensure envs/ewb exists and can import extremeweatherbench."""
    py = ewb_python()
    errors = verify_runtime(check_cira=False)
    if not errors:
        pin = write_env_pin(python=py)
        return {"ok": True, "installed": False, "missing": [], "python": str(py), "env_pin": str(pin)}

    missing = missing_modules(py)
    if not install:
        return {
            "ok": False,
            "installed": False,
            "missing": missing,
            "python": str(py),
            "errors": errors,
            "error": f"EWB env incomplete; run: bash {create_env_script()}",
        }

    script = create_env_script()
    if not script.is_file():
        return {"ok": False, "error": f"missing bootstrap script: {script}"}

    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "installed": False,
            "missing": missing,
            "stderr": proc.stderr[-4000:],
            "stdout": proc.stdout[-2000:],
        }

    py = ewb_python()
    still_errors = verify_runtime(check_cira=False)
    if still_errors:
        return {
            "ok": False,
            "installed": True,
            "python": str(py),
            "errors": still_errors,
            "error": "EWB env created but imports still fail",
        }

    pin = write_env_pin(python=py)
    return {
        "ok": True,
        "installed": True,
        "missing": [],
        "python": str(py),
        "env_pin": str(pin),
    }


def require_ewb_python() -> Path:
    status = ensure_ewb_env(install=False)
    py = Path(status.get("python") or ewb_python())
    if not status.get("ok"):
        raise RuntimeError(status.get("error") or f"EWB env not ready: {status}")
    return py


def install_ewb_package(python: Path | None = None) -> None:
    """Install pinned EWB into the isolated env only when imports are missing."""
    py = python or ewb_python()
    if not py.is_file():
        status = ensure_ewb_env(install=True)
        if not status.get("ok"):
            raise RuntimeError(f"EWB env bootstrap failed: {status}")
        return

    if not missing_modules(py):
        return

    proc = subprocess.run(
        [
            str(py),
            "-m",
            "pip",
            "install",
            "--quiet",
            f"git+{EWB_REPO}.git@{EWB_COMMIT}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pip install extremeweatherbench failed: {proc.stderr[-4000:]}")

    still = missing_modules(py)
    if still:
        raise RuntimeError(f"EWB install incomplete; missing: {still}")


def main(argv: List[str] | None = None) -> int:
    """CLI entrypoint — run with envs/pyhazards/bin/python, not envs/ewb."""
    ap = argparse.ArgumentParser(description="HW-MED EWB env verify/bootstrap")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--verify-cira", action="store_true", help="Also probe CIRA model names")
    ap.add_argument("--bootstrap", action="store_true", help="Run create_ewb_env.sh")
    args = ap.parse_args(argv)

    if args.bootstrap:
        status = ensure_ewb_env(install=True)
        if not status.get("ok"):
            print(status.get("error") or status, file=sys.stderr)
            return 1
        print(f"EWB env ready: {status.get('python')}")
        return 0

    errors = verify_runtime(check_cira=args.verify_cira)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    print("EWB runtime ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
