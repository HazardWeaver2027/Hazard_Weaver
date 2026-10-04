"""VBCI official Bayesian update replay for CAP-MH2-05 (Ridgecrest primary)."""

from __future__ import annotations

import subprocess
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.scientific.mh2_data import VBCI_EVENT_ID, event_dir, project_root, vbci_repo_dir
from hazardweaver.hcg.carp.scientific.mh2_fidelity import VBCI_REPLAY_PARITY_MIN, parity_ok
from hazardweaver.hcg.carp.scientific.mh2_reference import (
    coarsen_raster,
    load_reference_manifest,
    replay_parity,
    write_reference_manifest,
)
from hazardweaver.hcg.carp.scientific.mh2_vbci_env import (
    build_vbci_eval_argv,
    vbci_method_dir,
    vbci_runner,
    vbci_runner_kind,
)

RIDGECREST_EVENT = VBCI_EVENT_ID
VBCI_COARSEN_FACTOR = 8
VBCI_REFERENCE_KIND = "pinned_official_vbci_svi_v1"
RASTER_NAMES = {
    "PLS": "2019_ridgecrest_prior_landslide_model",
    "PLF": "2019_ridgecrest_prior_liquefaction_model",
    "DPM": "2019_ridgecrest_building_damage_model",
    "BD": "2019_ridgecrest_building_footprint_rasterized",
}


def _read_geotiff(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff")))
            data = zf.read(name)
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(data) as mem:
            with mem.open() as ds:
                return ds.read(1).astype(np.float32), {
                    "transform": ds.transform,
                    "crs": ds.crs,
                    "width": ds.width,
                    "height": ds.height,
                }
    import rasterio

    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.float32), {
            "transform": ds.transform,
            "crs": ds.crs,
            "width": ds.width,
            "height": ds.height,
        }


def _align_to_grid(
    src: np.ndarray,
    src_meta: Dict[str, Any],
    ref_meta: Dict[str, Any],
    *,
    nearest: bool = False,
) -> np.ndarray:
    from rasterio.warp import Resampling, reproject

    dst = np.zeros((ref_meta["height"], ref_meta["width"]), dtype=np.float32)
    reproject(
        source=src.astype(np.float32),
        destination=dst,
        src_transform=src_meta["transform"],
        src_crs=src_meta["crs"],
        dst_transform=ref_meta["transform"],
        dst_crs=ref_meta["crs"],
        resampling=Resampling.nearest if nearest else Resampling.bilinear,
    )
    return dst


def _load_aligned_vendor_arrays(paths: Dict[str, Path]) -> Dict[str, np.ndarray]:
    """Align all VBCI Ridgecrest rasters to the PLS prior grid (official case extent)."""
    ref_arr, ref_meta = _read_geotiff(paths["PLS"])
    arrays: Dict[str, np.ndarray] = {"PLS": ref_arr}
    for key in ("PLF", "DPM"):
        arr, meta = _read_geotiff(paths[key])
        if arr.shape != ref_arr.shape:
            arr = _align_to_grid(arr, meta, ref_meta, nearest=False)
        arrays[key] = arr.astype(np.float32)
    bd_arr, bd_meta = _read_geotiff(paths["BD"])
    arrays["BD"] = _align_to_grid(bd_arr, bd_meta, ref_meta, nearest=True).astype(np.float32)
    shapes = {k: v.shape for k, v in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"VBCI raster alignment failed: {shapes}")
    return arrays


def _read_tif(path: Path) -> np.ndarray:
    return _read_geotiff(path)[0]


def _vendor_paths(event_id: str = RIDGECREST_EVENT) -> Dict[str, Path]:
    root = vbci_repo_dir() / "data" / "2019_ridgecrest"
    return {
        "PLS": root / "prior_models" / f"{RASTER_NAMES['PLS']}.zip",
        "PLF": root / "prior_models" / f"{RASTER_NAMES['PLF']}.zip",
        "DPM": root / "building_damage_model" / f"{RASTER_NAMES['DPM']}.zip",
        "BD": root / "building_footprint" / f"{RASTER_NAMES['BD']}.zip",
    }


def _export_vbci_inputs_mat(case_dir: Path, coarse: Dict[str, np.ndarray]) -> Path:
    from scipy.io import savemat

    mat_path = case_dir / "vbci_inputs.mat"
    savemat(
        mat_path,
        {
            "Y": coarse["DPM"].astype(np.float64),
            "BD": coarse["BD"].astype(np.float64),
            "LS": coarse["PLS"].astype(np.float64),
            "LF": coarse["PLF"].astype(np.float64),
        },
    )
    return mat_path


def prepare_vbci_ridgecrest_case(event_id: str = RIDGECREST_EVENT) -> Dict[str, Any]:
    """Export coarsened official VBCI Ridgecrest inputs for SVI replay."""
    vbci_root = event_dir("official_test", event_id) / "vbci"
    case_dir = vbci_root / "svi_case"
    ref_dir = vbci_root / "reference_posterior"
    case_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    paths = _vendor_paths(event_id)
    missing = [k for k, p in paths.items() if not p.is_file()]
    if missing:
        return {"ok": False, "error": f"missing VBCI vendor assets: {missing}"}

    try:
        arrays = _load_aligned_vendor_arrays(paths)
    except (ValueError, OSError) as exc:
        return {"ok": False, "error": f"VBCI raster alignment failed: {exc}"}

    coarse = {k: coarsen_raster(v, VBCI_COARSEN_FACTOR) for k, v in arrays.items()}
    coarse_shapes = {k: v.shape for k, v in coarse.items()}
    if len(set(coarse_shapes.values())) != 1:
        return {"ok": False, "error": f"VBCI coarse raster shape mismatch: {coarse_shapes}"}
    _export_vbci_inputs_mat(case_dir, coarse)
    return {"ok": True, "case_dir": str(case_dir), "reference_dir": str(ref_dir)}


def _run_vbci_svi(case_dir: Path) -> Dict[str, Any]:
    runner = vbci_runner()
    kind = vbci_runner_kind(runner)
    if not runner or not kind:
        return {"ok": False, "error": "matlab/octave not in PATH for VBCI SVI replay"}

    method_dir = vbci_method_dir()
    script_path = project_root() / "scripts" / "vbci" / "ridgecrest_vbci_replay.m"
    if not script_path.is_file():
        return {"ok": False, "error": f"missing {script_path}"}

    try:
        argv, kind = build_vbci_eval_argv(
            case_dir=case_dir,
            method_dir=method_dir,
            script_parent=script_path.parent,
            runner=runner,
        )
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}

    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0 or "VBCI_RIDGECREST_SVI_OK" not in stdout:
        return {
            "ok": False,
            "error": f"VBCI SVI {kind} replay failed (rc={proc.returncode})",
            "stderr": stderr[-3000:],
            "stdout": stdout[-1500:],
            "runner": runner,
            "runner_kind": kind,
        }
    return {"ok": True, "runner": runner, "runner_kind": kind}


def _load_posterior(case_dir: Path) -> Dict[str, Any]:
    mat_path = case_dir / "posterior_ls.mat"
    if not mat_path.is_file():
        return {"ok": False, "error": "missing posterior_ls.mat from VBCI SVI"}
    try:
        from scipy.io import loadmat

        data = loadmat(mat_path)
        posterior = np.asarray(data["final_QLS"], dtype=np.float32).ravel()
    except Exception as exc:
        return {"ok": False, "error": f"failed to load VBCI posterior mat: {exc}"}
    return {"ok": True, "posterior": posterior, "mat_path": str(mat_path)}


def pin_vbci_reference_for_event(event_id: str = RIDGECREST_EVENT) -> Dict[str, Any]:
    """Freeze official VBCI SVI output as replay reference (materialize/bootstrap)."""
    prep = prepare_vbci_ridgecrest_case(event_id)
    if not prep.get("ok"):
        return prep

    case_dir = Path(prep["case_dir"])
    ref_dir = Path(prep["reference_dir"])
    ref_npy = ref_dir / "posterior_ls_coarse.npy"
    manifest = load_reference_manifest(ref_dir) or {}
    entry = (manifest.get("models") or {}).get("vbci_posterior_ls") or {}
    if ref_npy.is_file() and entry.get("reference_kind") == VBCI_REFERENCE_KIND:
        return {"ok": True, "skipped": True, "reference_dir": str(ref_dir)}

    svi = _run_vbci_svi(case_dir)
    if not svi.get("ok"):
        return svi

    loaded = _load_posterior(case_dir)
    if not loaded.get("ok"):
        return loaded

    posterior = loaded["posterior"]
    np.save(ref_npy, posterior.astype(np.float32))
    write_reference_manifest(
        ref_dir,
        model_key="vbci_posterior_ls",
        event_id=event_id,
        reference_kind=VBCI_REFERENCE_KIND,
        prob_path=ref_npy,
        extra={"coarsen_factor": VBCI_COARSEN_FACTOR, "rng_seed": 42},
    )
    return {
        "ok": True,
        "reference_dir": str(ref_dir),
        "reference_kind": VBCI_REFERENCE_KIND,
        "n_cells": int(posterior.size),
    }


def run_vbci_ridgecrest(
    *,
    out_dir: Path,
    reference_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    prep = prepare_vbci_ridgecrest_case()
    if not prep.get("ok"):
        return prep

    case_dir = Path(prep["case_dir"])
    ref_dir = Path(reference_dir or prep["reference_dir"])
    ref_path = ref_dir / "posterior_ls_coarse.npy"
    if not ref_path.is_file():
        return {
            "ok": False,
            "error": f"missing pinned VBCI reference at {ref_path}; run materialize pin first",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    svi = _run_vbci_svi(case_dir)
    if not svi.get("ok"):
        return svi

    loaded = _load_posterior(case_dir)
    if not loaded.get("ok"):
        return loaded

    posterior = loaded["posterior"]
    posterior_path = out_dir / "posterior_ls.npy"
    np.save(posterior_path, posterior)

    ref = np.load(ref_path).astype(np.float32).ravel()
    parity, ref_present = replay_parity(posterior, ref)
    runner_kind = svi.get("runner_kind", "matlab")

    if not ref_present:
        return {"ok": False, "error": "missing VBCI reference posterior"}
    if not parity_ok(parity, minimum=VBCI_REPLAY_PARITY_MIN):
        return {
            "ok": False,
            "error": f"bayesian_replay_parity {parity} < {VBCI_REPLAY_PARITY_MIN}",
            "posterior": posterior,
            "bayesian_replay_parity": parity,
            "stderr": svi.get("stderr", ""),
        }

    return {
        "ok": True,
        "posterior": posterior,
        "posterior_path": str(posterior_path),
        "bayesian_replay_parity": parity,
        "replay_parity": parity,
        "reference_present": True,
        "reference_kind": VBCI_REFERENCE_KIND,
        "evaluator": f"vbci@{runner_kind}",
        "event_id": RIDGECREST_EVENT,
        "official_command": f"ridgecrest_vbci_replay.m case_dir={case_dir} rng_seed=42",
    }


def load_vbci_inputs(event_id: str = RIDGECREST_EVENT) -> Dict[str, Any]:
    prep = prepare_vbci_ridgecrest_case(event_id)
    if not prep.get("ok"):
        return prep
    return {"ok": True, "event_id": event_id, "case_dir": prep["case_dir"]}
