"""Materialize/cache CLIMADA exposure for MH-4 R06 (Atlantic+Gulf CONUS slice)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from hazardweaver.hcg.carp.scientific.paths import PROJECT_ROOT

EXPOSURE_ID = "climada_litpop_atlantic_gulf_v1"
EXPOSURE_ID_LEGACY_ALIAS = "climada_nightlight_proxy_conus_v1"
EXPOSURE_DIR = PROJECT_ROOT / "data" / "vendor" / "climada" / "exposure"
EXPOSURE_HDF5 = EXPOSURE_DIR / "litpop_conus_v1.hdf5"
EXPOSURE_MANIFEST = EXPOSURE_DIR / "manifest.json"
# Atlantic + Gulf bounding box (CONUS TC headline region)
TC_BBOX = (-100.0, 24.0, -65.0, 45.0)
NL_REFERENCE_YEAR = 2016
NL_VALUE_SCALE_USD = 50_000.0
NL_SUBSAMPLE_STEP = 40


def exposure_ready() -> bool:
    return EXPOSURE_HDF5.is_file() and EXPOSURE_MANIFEST.is_file()


def ensure_conus_exposure(*, force: bool = False) -> Path:
    """Build nightlight-based LitPop exposure when GPW is unavailable; cache under vendor."""
    if not force and exposure_ready():
        return EXPOSURE_HDF5

    import geopandas as gpd
    import numpy as np
    from affine import Affine
    from climada.entity import LitPop
    from climada.entity.exposures.litpop import nightlight as nl_util
    from shapely.geometry import Point, box

    EXPOSURE_DIR.mkdir(parents=True, exist_ok=True)
    nlight, meta = nl_util.load_nasa_nl_shape(box(*TC_BBOX), NL_REFERENCE_YEAR)
    arr = np.asarray(nlight[0] if np.ndim(nlight) == 3 else nlight)
    arr_s = arr[::NL_SUBSAMPLE_STEP, ::NL_SUBSAMPLE_STEP]
    transform = meta["transform"] * Affine.scale(NL_SUBSAMPLE_STEP, NL_SUBSAMPLE_STEP)
    height, width = arr_s.shape
    lons = transform.c + transform.a * (np.arange(width) + 0.5)
    lats = transform.f + transform.e * (np.arange(height) + 0.5)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    values = arr_s.ravel().astype(float) * NL_VALUE_SCALE_USD
    mask = values > 0
    lat_flat = lat_grid.ravel()[mask]
    lon_flat = lon_grid.ravel()[mask]
    val_flat = values[mask]
    gdf = gpd.GeoDataFrame(
        {
            "value": val_flat,
            "latitude": lat_flat,
            "longitude": lon_flat,
            "if_TC": 1,
        },
        geometry=[Point(xy) for xy in zip(lon_flat, lat_flat)],
        crs="EPSG:4326",
    )
    exp = LitPop()
    exp.set_gdf(gdf, crs="EPSG:4326")
    exp.value_unit = "USD"
    exp.write_hdf5(str(EXPOSURE_HDF5))

    manifest: Dict[str, Any] = {
        "exposure_inventory_id": EXPOSURE_ID,
        "path": str(EXPOSURE_HDF5.relative_to(PROJECT_ROOT)),
        "method": "climada_litpop_nightlight_atlantic_gulf_v1",
        "note": (
            "CLIMADA LitPop exposure from NASA nightlights (Atlantic+Gulf bbox); "
            "official CLIMADA API materialization (DL-111)"
        ),
        "bbox": list(TC_BBOX),
        "n_points": int(len(gdf)),
        "reference_year": NL_REFERENCE_YEAR,
    }
    EXPOSURE_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return EXPOSURE_HDF5


def load_conus_exposure():
    from climada.entity import LitPop

    ensure_conus_exposure()
    exp = LitPop.from_hdf5(str(EXPOSURE_HDF5))
    if "if_TC" not in exp.gdf.columns:
        exp.gdf["if_TC"] = 1
    return exp
