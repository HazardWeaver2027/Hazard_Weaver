"""LISFLOOD-FP driver materialization for CAP-FL2-05 — FloodCastBench → LISFLOOD ASCII."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.native_eval.scientific.fl2_sfincs_driver import _clean_dem

DEFAULT_DT_SECONDS = 300
DEFAULT_CELL_SIZE_M = 480.0
DEFAULT_MANNING = 0.04
NODATA_VALUE = -9999.0


def _write_arc_ascii(path: Path, grid: np.ndarray, *, cell_size: float) -> None:
    nrows, ncols = grid.shape
    lines = [
        f"ncols         {ncols}",
        f"nrows         {nrows}",
        "xllcorner     0",
        "yllcorner     0",
        f"cellsize      {cell_size}",
        f"NODATA_value  {NODATA_VALUE:.0f}",
    ]
    for row in grid:
        lines.append(" ".join(f"{float(v):.4f}" for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_bci(path: Path, *, nrows: int, ncols: int) -> None:
    # LISFLOOD uses 0-based edge indices in official tests (T031).
    lines = [
        f"N 0 {ncols - 1} FREE",
        f"S 0 {ncols - 1} FREE",
        f"W 0 {nrows - 1} FREE",
        f"E 0 {nrows - 1} FREE",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_par(
    path: Path,
    *,
    n_steps: int,
    dt: int = DEFAULT_DT_SECONDS,
) -> None:
    sim_time = float(max(dt, n_steps * dt))
    text = f"""# FloodCastBench → LISFLOOD driver (CAP-FL2-05)
DEMfile                  floodcast_dem.asc
startfile                floodcast_start.asc
bcifile                  floodcast.bci
resroot                  floodcast
dirroot                  results
sim_time                 {sim_time:.1f}
initial_tstep            1.0
massint                  {float(dt):.1f}
saveint                  {float(dt):.1f}
fpfric                   {DEFAULT_MANNING}
adaptoff
elevoff
"""
    path.write_text(text, encoding="utf-8")


def _parse_arc_ascii(path: Path) -> np.ndarray:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    data_start = 0
    for i, line in enumerate(lines):
        if line.lower().startswith("ncols"):
            continue
        if line.lower().startswith("nrows"):
            continue
        if line.lower().startswith(("xll", "yll", "cellsize", "nodata")):
            continue
        if re.match(r"^[-+0-9.eE]", line.split()[0]):
            data_start = i
            break
    rows = [[float(x) for x in ln.split()] for ln in lines[data_start:]]
    return np.asarray(rows, dtype=np.float32)


def read_lisflood_depth_sequence(
    model_dir: Path,
    *,
    n_steps: int,
) -> Optional[np.ndarray]:
    """Parse LISFLOOD .wd snapshots under model_dir/results; return (T, H, W)."""
    results_dir = model_dir / "results"
    if not results_dir.is_dir():
        results_dir = model_dir
    wd_files = sorted(results_dir.glob("floodcast-*.wd"))
    if not wd_files:
        wd_files = sorted(results_dir.glob("*.wd"))
    if not wd_files:
        wd_files = sorted(model_dir.glob("floodcast-*.wd"))
    if not wd_files:
        return None

    def _save_no(path: Path) -> int:
        match = re.search(r"-(\d+)\.wd$", path.name)
        return int(match.group(1)) if match else 0

    wd_files = sorted(wd_files, key=_save_no)
    frames: List[np.ndarray] = []
    for fp in wd_files:
        try:
            frames.append(_parse_arc_ascii(fp))
        except (OSError, ValueError):
            continue
        if len(frames) >= n_steps:
            break
    if not frames:
        return None
    arr = np.stack(frames[:n_steps]).astype(np.float32)
    return np.maximum(arr, 0.0)


def materialize_lisflood_driver(
    scenario_dir: Path,
    *,
    out_dir: Path,
    max_dim: int = 0,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Materialize LISFLOOD ASCII driver from FloodCast scenario arrays."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dem_path = scenario_dir / "dem.npy"
    init_path = scenario_dir / "initial_depth.npy"
    truth_path = scenario_dir / "truth_depth.npy"
    for req in (dem_path, init_path, truth_path):
        if not req.is_file():
            return None, {
                "ok": False,
                "reason": f"missing {req.name}",
                "status": "NOT_IMPLEMENTED",
            }

    dem = np.load(dem_path)
    initial = np.load(init_path)
    truth = np.load(truth_path)
    n_steps = int(truth.shape[0])

    stride = 1
    if max_dim > 0 and max(dem.shape) > max_dim:
        stride = int(np.ceil(max(dem.shape) / max_dim))
        dem = dem[::stride, ::stride]
        initial = initial[::stride, ::stride]
        truth = truth[:, ::stride, ::stride]

    dem, valid = _clean_dem(dem)
    initial = np.where(valid, initial, 0.0).astype(np.float32)
    start = np.maximum(initial, 0.0).astype(np.float32)

    dem_path_out = out_dir / "floodcast_dem.asc"
    start_path_out = out_dir / "floodcast_start.asc"
    bci_path = out_dir / "floodcast.bci"
    par_path = out_dir / "floodcast.par"
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    nrows, ncols = dem.shape
    _write_arc_ascii(dem_path_out, dem, cell_size=DEFAULT_CELL_SIZE_M)
    _write_arc_ascii(start_path_out, start, cell_size=DEFAULT_CELL_SIZE_M)
    _write_bci(bci_path, nrows=nrows, ncols=ncols)
    _write_par(par_path, n_steps=n_steps)

    meta = {
        "ok": True,
        "status": "MATERIALIZED",
        "scenario_dir": str(scenario_dir),
        "driver_path": str(par_path),
        "grid_shape": [int(nrows), int(ncols)],
        "n_steps": n_steps,
        "stride": stride,
        "cell_size_m": DEFAULT_CELL_SIZE_M,
        "materialization_mode": "floodcastbench_to_lisflood_ascii_v1",
    }
    (out_dir / "driver_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return par_path, meta
