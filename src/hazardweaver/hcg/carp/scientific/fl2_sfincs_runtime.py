"""SFINCS singularity/docker runtime helpers for FL-2."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Tuple

from hazardweaver.hcg.carp.acquire.handlers.sfincs import DOCKER_IMAGE
from hazardweaver.hcg.carp.scientific.fl2_sfincs_verify import sfincs_sif_path, sfincs_sif_present


def container_runtime_available() -> bool:
    return sfincs_sif_present() or shutil.which("docker") is not None


def build_sfincs_run_command(model_dir: str | Path) -> str:
    model = Path(model_dir).resolve()
    sif = sfincs_sif_path()
    if sfincs_sif_present():
        return (
            f"singularity exec --bind {model}:/data --pwd /data {sif} "
            f"sfincs sfincs.inp"
        )
    return (
        f"docker run --rm -v {model}:/data -w /data {DOCKER_IMAGE} "
        f"sfincs sfincs.inp"
    )


def run_sfincs(model_dir: Path, *, timeout_s: int = 7200) -> Tuple[int, str, str]:
    cmd = build_sfincs_run_command(model_dir)
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""
