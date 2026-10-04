"""SFINCS driver materialization for CAP-FL2-04 — FloodCastBench → SFINCS ASCII grid."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

NODATA_SENTINEL = -3.4e38
DEFAULT_DT_SECONDS = 300
DEFAULT_MANNING = 0.04
DEFAULT_CELL_SIZE_M = 480.0


def _clean_dem(dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dem = dem.astype(np.float32)
    valid = np.isfinite(dem) & (dem > -1e4) & (dem < 1e5)
    if not np.any(valid):
        raise ValueError("dem has no valid cells")
    fill = float(np.median(dem[valid]))
    out = dem.copy()
    out[~valid] = fill
    return out, valid


def _load_ascii_grid(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    return np.asarray(rows, dtype=np.float32)


def _write_ascii_grid(path: Path, grid: np.ndarray) -> None:
    """Write SFINCS ASCII grid (nmax rows × mmax cols)."""
    nmax, mmax = grid.shape
    lines = []
    for row in range(nmax):
        lines.append(" ".join(f"{grid[row, col]:.4f}" for col in range(mmax)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ascii_mask(path: Path, mask: np.ndarray) -> None:
    """Write SFINCS ASCII mask (integer kcs values; list-directed Fortran read)."""
    nmax, mmax = mask.shape
    lines = []
    for row in range(nmax):
        lines.append(" ".join(str(int(mask[row, col])) for col in range(mmax)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sfincs_inp(
    path: Path,
    *,
    mmax: int,
    nmax: int,
    n_steps: int,
    zsini: float,
    dt: int = DEFAULT_DT_SECONDS,
) -> None:
    t0 = datetime(2022, 8, 1, 0, 0, 0)
    t1 = t0 + timedelta(seconds=max(dt, n_steps * dt))
    text = f"""# FloodCastBench → SFINCS driver (CAP-FL2-04)
x0 = 0
y0 = 0
mmax = {mmax}
nmax = {nmax}
dx = {DEFAULT_CELL_SIZE_M}
dy = {DEFAULT_CELL_SIZE_M}
rotation = 0

tref = {t0.strftime("%Y%m%d %H%M%S")}
tstart = {t0.strftime("%Y%m%d %H%M%S")}
tstop = {t1.strftime("%Y%m%d %H%M%S")}
dt = {dt}
dtout = {dt}

depfile = sfincs.dep
mskfile = sfincs.msk
manning = {DEFAULT_MANNING}
zsini = {zsini:.4f}

advection = 1
alpha = 0.5
huthresh = 0.05
theta = 1.0
qinf = 0.0

inputformat = asc
outputformat = net
"""
    path.write_text(text, encoding="utf-8")


def read_sfincs_depth_sequence(
    model_dir: Path,
    *,
    n_steps: int,
) -> Optional[np.ndarray]:
    """Parse SFINCS net output; return depth (T, H, W)."""
    dem = None
    dep_fp = model_dir / "sfincs.dep"
    if dep_fp.is_file():
        dem = _load_ascii_grid(dep_fp)

    for name in ("sfincs_map.nc", "sfincs_his.nc", "depth_pred.npy"):
        fp = model_dir / name
        if not fp.is_file():
            continue
        if fp.suffix == ".npy":
            arr = np.load(fp)
            if arr.ndim == 3:
                return arr.astype(np.float32)[:n_steps]
            if arr.ndim == 2:
                return arr.astype(np.float32)[np.newaxis, ...]
        try:
            import netCDF4  # type: ignore

            ds = netCDF4.Dataset(str(fp))
            for var in ("zs", "h", "depth", "waterdepth", "zsmax"):
                if var not in ds.variables:
                    continue
                data = np.asarray(ds.variables[var][:])
                ds.close()
                if data.ndim == 3:
                    arr = data.astype(np.float32)
                    if var == "zs" and dem is not None:
                        arr = np.maximum(arr - dem[np.newaxis, :, :], 0.0)
                    return arr[:n_steps]
                if data.ndim == 2:
                    return data.astype(np.float32)[np.newaxis, ...]
            ds.close()
        except Exception:
            continue
    return None


def is_full_driver_ready(driver_inp: Path) -> bool:
    if not driver_inp.is_file():
        return False
    text = driver_inp.read_text(encoding="utf-8", errors="replace")
    if "FloodCastBench → SFINCS driver" not in text:
        return False
    if len(text.strip()) < 64:
        return False
    digest = hashlib.sha256(text.encode()).hexdigest()
    return digest != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def materialize_sfincs_driver(
    scenario_dir: Path,
    *,
    out_dir: Path,
    max_dim: int = 0,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Materialize SFINCS ASCII driver from FloodCast scenario arrays."""
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
    msk = valid.astype(np.int32)
    initial = np.where(valid, initial, 0.0).astype(np.float32)
    wet = (initial > 0.01) & valid
    if np.any(wet):
        zsini = float(np.mean(dem[wet] + initial[wet]))
    else:
        zsini = float(np.mean(dem[valid]) + np.mean(initial[valid]))

    dep_path = out_dir / "sfincs.dep"
    msk_path = out_dir / "sfincs.msk"
    inp_path = out_dir / "sfincs.inp"

    nmax, mmax = dem.shape
    _write_ascii_grid(dep_path, dem)
    _write_ascii_mask(msk_path, msk)
    _write_sfincs_inp(inp_path, mmax=mmax, nmax=nmax, n_steps=n_steps, zsini=zsini)

    meta = {
        "ok": True,
        "status": "MATERIALIZED",
        "scenario_dir": str(scenario_dir),
        "driver_path": str(inp_path),
        "grid_shape": [int(nmax), int(mmax)],
        "n_steps": n_steps,
        "stride": stride,
        "cell_size_m": DEFAULT_CELL_SIZE_M,
        "zsini_m": zsini,
        "materialization_mode": "floodcastbench_to_sfincs_ascii_v1",
    }
    (out_dir / "driver_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return inp_path, meta
