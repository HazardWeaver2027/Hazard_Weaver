"""LISFLOOD-FP runtime helpers for CAP-FL2-05."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple

from hazardweaver.hcg.carp.scientific.fl2_lisflood_verify import LISFLOOD_CLI, _lisflood_env


def build_lisflood_run_command(model_dir: str | Path) -> str:
    model = Path(model_dir).resolve()
    par = model / "floodcast.par"
    return f"cd {model} && {LISFLOOD_CLI} {par.name}"


def run_lisflood(model_dir: Path, *, par_name: str = "floodcast.par", timeout_s: int = 7200) -> Tuple[int, str, str]:
    par_path = model_dir / par_name
    proc = subprocess.run(
        [str(LISFLOOD_CLI), par_name],
        cwd=str(model_dir),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        env=_lisflood_env(),
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""
