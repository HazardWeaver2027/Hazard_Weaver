"""MH-3 SFINCS ASCII driver materialization (Charleston official_test → SFINCS inp/dep/msk)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from hazardweaver.hcg.carp.native_eval.scientific.fl2_sfincs_driver import (
    _clean_dem,
    _write_ascii_grid,
    _write_ascii_mask,
    _write_sfincs_inp,
    read_sfincs_depth_sequence,
)

__all__ = [
    "is_driver_ready",
    "materialize_mh3_sfincs_driver",
    "read_sfincs_depth_sequence",
]


def is_driver_ready(model_dir: Path) -> bool:
    inp = model_dir / "sfincs.inp"
    dep = model_dir / "sfincs.dep"
    msk = model_dir / "sfincs.msk"
    if not all(p.is_file() for p in (inp, dep, msk)):
        return False
    text = inp.read_text(encoding="utf-8", errors="replace")
    if "stub for Charleston" in text:
        return False
    return "mmax =" in text and "depfile = sfincs.dep" in text and "tref =" in text


def materialize_mh3_sfincs_driver(model_dir: Path, *, force: bool = False) -> Dict[str, Any]:
    model_dir = Path(model_dir)
    if not force and is_driver_ready(model_dir):
        return {"ok": True, "skipped": True, "model_dir": str(model_dir)}

    scen_dir = model_dir.parent
    dem_path = scen_dir / "dem.npy"
    truth_path = scen_dir / "truth_depth.npy"
    if not dem_path.is_file() or not truth_path.is_file():
        return {
            "ok": False,
            "reason": f"missing scenario arrays under {scen_dir}",
            "model_dir": str(model_dir),
        }

    dem = np.load(dem_path)
    truth = np.load(truth_path)
    n_steps = int(truth.shape[0])
    initial = np.zeros_like(dem, dtype=np.float32)

    dem, valid = _clean_dem(dem)
    msk = valid.astype(np.int32)
    initial = np.where(valid, initial, 0.0).astype(np.float32)
    wet = (initial > 0.01) & valid
    if np.any(wet):
        zsini = float(np.mean(dem[wet] + initial[wet]))
    else:
        zsini = float(np.mean(dem[valid]) + np.mean(initial[valid]))

    model_dir.mkdir(parents=True, exist_ok=True)
    nmax, mmax = dem.shape
    dep_path = model_dir / "sfincs.dep"
    msk_path = model_dir / "sfincs.msk"
    inp_path = model_dir / "sfincs.inp"
    _write_ascii_grid(dep_path, dem)
    _write_ascii_mask(msk_path, msk)
    _write_sfincs_inp(inp_path, mmax=mmax, nmax=nmax, n_steps=n_steps, zsini=zsini)

    meta = {
        "ok": True,
        "status": "MATERIALIZED",
        "scenario_dir": str(scen_dir),
        "driver_path": str(inp_path),
        "grid_shape": [int(nmax), int(mmax)],
        "n_steps": n_steps,
        "zsini_m": zsini,
        "materialization_mode": "mh3_charleston_to_sfincs_ascii_v1",
    }
    (model_dir / "driver_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta
