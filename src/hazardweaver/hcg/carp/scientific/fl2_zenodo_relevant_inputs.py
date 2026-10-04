"""Zenodo FloodCastBench Relevant data → per-scenario COS-FL2 input fields (DL-124)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile

from hazardweaver.hcg.carp.scientific.fl2_data import (
    N_STEPS,
    SPLIT_ROOTS,
    TASKPACK,
    scenario_dir,
    scenario_ids,
    split_root,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ZENODO_EXTRACT = PROJECT_ROOT / "data" / "vendor" / "floodcastbench" / "zenodo" / "extracted"
RELEVANT_ROOT = ZENODO_EXTRACT / "FloodCastBench" / "Relevant data"
DATA_ROOT = PROJECT_ROOT / "data" / "scientific" / TASKPACK
CACHE_ROOT = DATA_ROOT / "_zenodo_input_cache"

DEPTH_STEP_SECONDS = 300
RAIN_STEP_SECONDS = 1800
DEPTH_FRAMES_PER_RAIN = RAIN_STEP_SECONDS // DEPTH_STEP_SECONDS  # 6
NODATA_F32 = np.float32(-3.402823e38)

INPUT_SCHEMA_VERSION = "cos_fl2_zenodo_v1"
REQUIRED_INPUT_FILES = (
    "dem.npy",
    "initial_depth.npy",
    "truth_depth.npy",
    "rainfall.npy",
    "landuse.npy",
    "boundary_conditions.npy",
)

REGION_SPECS: Dict[str, Dict[str, str]] = {
    "pakistan_2022": {
        "rain_dir": "Rainfall/Pakistan flood",
        "lulc_tif": "Land use and land cover/Pakistan.tif",
        "dem_tif": "DEM/Pakistan_DEM.tif",
    },
    "mozambique_2019": {
        "rain_dir": "Rainfall/Mozambique flood",
        "lulc_tif": "Land use and land cover/Mozambique.tif",
        "dem_tif": "DEM/Mozambique_DEM.tif",
    },
    "australia_2022": {
        "rain_dir": "Rainfall/Australia flood",
        "lulc_tif": "Land use and land cover/Australia.tif",
        "dem_tif": "DEM/Australia_DEM.tif",
    },
    "uk_2015": {
        "rain_dir": "Rainfall/UK flood",
        "lulc_tif": "Land use and land cover/UK.tif",
        "dem_tif": "DEM/UK_DEM.tif",
    },
}


def relevant_data_ready() -> bool:
    marker = ZENODO_EXTRACT / ".extract_complete"
    return marker.is_file() and RELEVANT_ROOT.is_dir()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sanitize_nodata(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    bad = ~np.isfinite(out) | (np.abs(out - NODATA_F32) < 1.0)
    out[bad] = 0.0
    return out


def _resize_field(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return _sanitize_nodata(arr)
    from skimage.transform import resize

    h, w = arr.shape
    th, tw = shape
    step_h = max(1, h // max(th, 1))
    step_w = max(1, w // max(tw, 1))
    coarse = _sanitize_nodata(arr[::step_h, ::step_w])
    return resize(
        coarse,
        shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def _read_tiff(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr


def rain_index_for_depth_frame(global_frame: int) -> int:
    return max(0, global_frame // DEPTH_FRAMES_PER_RAIN)


def rainfall_indices_for_window(window_index: int, *, n_steps: int = N_STEPS) -> List[int]:
    start = window_index * n_steps
    return [rain_index_for_depth_frame(start + t) for t in range(n_steps)]


@dataclass
class RegionInputCache:
    region: str
    grid_shape: Tuple[int, int]
    rainfall_stack: np.ndarray
    landuse: np.ndarray
    n_rain_source_frames: int

    @property
    def cache_tag(self) -> str:
        h, w = self.grid_shape
        return f"{self.region}_{h}x{w}"


def _cache_paths(tag: str) -> Tuple[Path, Path]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return (
        CACHE_ROOT / f"{tag}_rainfall.npy",
        CACHE_ROOT / f"{tag}_landuse.npy",
    )


def _load_rainfall_stack(region: str, grid_shape: Tuple[int, int]) -> Tuple[np.ndarray, int]:
    spec = REGION_SPECS[region]
    rain_dir = RELEVANT_ROOT / spec["rain_dir"]
    files = sorted(rain_dir.glob("*.tif"))
    if len(files) < N_STEPS:
        raise FileNotFoundError(f"insufficient rainfall TIFFs under {rain_dir}")
    stack = np.zeros((len(files), *grid_shape), dtype=np.float32)
    for i, path in enumerate(files):
        stack[i] = _resize_field(_read_tiff(path), grid_shape)
    return stack, len(files)


def _load_landuse(region: str, grid_shape: Tuple[int, int]) -> np.ndarray:
    spec = REGION_SPECS[region]
    path = RELEVANT_ROOT / spec["lulc_tif"]
    if not path.is_file():
        raise FileNotFoundError(f"missing landuse {path}")
    lulc = _read_tiff(path)
    return _resize_field(lulc, grid_shape).astype(np.float32)


def get_region_input_cache(region: str, grid_shape: Tuple[int, int]) -> RegionInputCache:
    if region not in REGION_SPECS:
        raise KeyError(f"unknown FL-2 region {region!r}")
    tag = f"{region}_{grid_shape[0]}x{grid_shape[1]}"
    rain_cache, lulc_cache = _cache_paths(tag)
    if rain_cache.is_file() and lulc_cache.is_file():
        rain = np.load(rain_cache)
        lulc = np.load(lulc_cache)
        meta_path = CACHE_ROOT / f"{tag}_meta.json"
        n_src = int(json.loads(meta_path.read_text()).get("n_rain_source_frames", rain.shape[0]))
        return RegionInputCache(region, grid_shape, rain, lulc, n_src)

    rainfall_stack, n_src = _load_rainfall_stack(region, grid_shape)
    landuse = _load_landuse(region, grid_shape)
    np.save(rain_cache, rainfall_stack)
    np.save(lulc_cache, landuse)
    (CACHE_ROOT / f"{tag}_meta.json").write_text(
        json.dumps(
            {
                "region": region,
                "grid_shape": list(grid_shape),
                "n_rain_source_frames": n_src,
                "depth_step_seconds": DEPTH_STEP_SECONDS,
                "rain_step_seconds": RAIN_STEP_SECONDS,
                "alignment": "rain_index = global_depth_frame // 6",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return RegionInputCache(region, grid_shape, rainfall_stack, landuse, n_src)


def build_rainfall_sequence(
    *,
    region: str,
    window_index: int,
    grid_shape: Tuple[int, int],
    n_steps: int = N_STEPS,
) -> np.ndarray:
    cache = get_region_input_cache(region, grid_shape)
    indices = rainfall_indices_for_window(window_index, n_steps=n_steps)
    max_idx = cache.rainfall_stack.shape[0] - 1
    clipped = [min(i, max_idx) for i in indices]
    return cache.rainfall_stack[clipped].astype(np.float32)


def build_boundary_conditions(grid_shape: Tuple[int, int]) -> np.ndarray:
    """Zenodo does not ship explicit BC rasters; open-domain placeholder (documented in meta)."""
    return np.zeros(grid_shape, dtype=np.float32)


def scenario_input_files_ready(split: str, scenario_id: str) -> bool:
    root = scenario_dir(split, scenario_id)
    return all((root / name).is_file() for name in REQUIRED_INPUT_FILES)


def input_schema_ready_for_splits(
    splits: Sequence[str] = ("official_test", "hwb_holdout"),
) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    for split in splits:
        for sid in scenario_ids(split):
            root = scenario_dir(split, sid)
            for name in REQUIRED_INPUT_FILES:
                if not (root / name).is_file():
                    missing.append(f"{split}/{sid}/{name}")
    return not missing, missing


def enrich_scenario_directory(
    split: str,
    scenario_id: str,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    root = scenario_dir(split, scenario_id)
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        return {"ok": False, "error": f"missing meta.json for {split}/{scenario_id}"}

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    region = str(meta.get("region") or "")
    if region not in REGION_SPECS:
        return {"ok": False, "error": f"unsupported region {region!r}"}

    window_index = int(meta.get("window_index", 0))
    grid_shape = tuple(int(x) for x in meta.get("grid_shape") or [])
    if len(grid_shape) != 2:
        return {"ok": False, "error": "invalid grid_shape in meta.json"}

    rain_path = root / "rainfall.npy"
    lulc_path = root / "landuse.npy"
    bc_path = root / "boundary_conditions.npy"
    if not force and rain_path.is_file() and lulc_path.is_file() and bc_path.is_file():
        return {"ok": True, "skipped": True, "scenario_id": scenario_id, "split": split}

    rainfall = build_rainfall_sequence(
        region=region,
        window_index=window_index,
        grid_shape=grid_shape,
    )
    cache = get_region_input_cache(region, grid_shape)
    landuse = cache.landuse
    bc = build_boundary_conditions(grid_shape)

    np.save(rain_path, rainfall)
    np.save(lulc_path, landuse)
    np.save(bc_path, bc)

    meta.update(
        {
            "input_schema_version": INPUT_SCHEMA_VERSION,
            "rainfall_shape": list(rainfall.shape),
            "landuse_shape": list(landuse.shape),
            "boundary_conditions_shape": list(bc.shape),
            "rainfall_alignment": "global_depth_frame // 6 from Zenodo 30-min rainfall",
            "rainfall_source": str(RELEVANT_ROOT / REGION_SPECS[region]["rain_dir"]),
            "landuse_source": str(RELEVANT_ROOT / REGION_SPECS[region]["lulc_tif"]),
            "boundary_conditions_source": "zenodo_not_published_open_domain_v1",
            "materialization_mode": meta.get("materialization_mode", "zenodo_tiff_v1")
            + "+zenodo_relevant_inputs_v1",
            "checksum_rainfall": _sha256_file(rain_path),
            "checksum_landuse": _sha256_file(lulc_path),
            "checksum_boundary_conditions": _sha256_file(bc_path),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return {
        "ok": True,
        "skipped": False,
        "scenario_id": scenario_id,
        "split": split,
        "region": region,
        "window_index": window_index,
        "rainfall_shape": list(rainfall.shape),
    }


def _refresh_split_manifest(split: str) -> Path:
    manifest_path = split_root(split) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenarios = manifest.get("scenarios") or []
    updated: List[Dict[str, Any]] = []
    for row in scenarios:
        sid = str(row.get("scenario_id") or "")
        meta_path = scenario_dir(split, sid) / "meta.json"
        if meta_path.is_file():
            updated.append(json.loads(meta_path.read_text(encoding="utf-8")))
        else:
            updated.append(row)
    manifest["scenarios"] = updated
    manifest["input_schema_version"] = INPUT_SCHEMA_VERSION
    manifest["input_fields"] = list(REQUIRED_INPUT_FILES)
    manifest["checksums"] = {
        str(row["scenario_id"]): hashlib.sha256(
            json.dumps(row, sort_keys=True).encode()
        ).hexdigest()
        for row in updated
        if row.get("scenario_id")
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def enrich_splits_from_zenodo(
    *,
    splits: Optional[Sequence[str]] = None,
    max_scenarios: int = 0,
    force: bool = False,
) -> Dict[str, Any]:
    if not relevant_data_ready():
        return {"ok": False, "reason": f"Zenodo extract not ready under {ZENODO_EXTRACT}"}

    split_list = list(splits or SPLIT_ROOTS.keys())
    results: List[Dict[str, Any]] = []
    for split in split_list:
        ids = scenario_ids(split)
        if max_scenarios > 0:
            ids = ids[:max_scenarios]
        for sid in ids:
            results.append(enrich_scenario_directory(split, sid, force=force))
        if ids:
            _refresh_split_manifest(split)

    n_ok = sum(1 for r in results if r.get("ok"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    return {
        "ok": n_ok == len(results) and bool(results),
        "n_scenarios": len(results),
        "n_ok": n_ok,
        "n_skipped": n_skip,
        "splits": split_list,
        "results": results,
        "cache_root": str(CACHE_ROOT),
    }
