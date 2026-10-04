"""PFDF inventory access for water–fire agent tools. Forbidden: pyhazards."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from data_adapters.usgs_pfdf import load_inventory
from data_adapters.usgs_pfdf.inventory import DEFAULT_INVENTORY_PATH, PFDFInventory

assert "pyhazards" not in globals()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACK_ROOT = PROJECT_ROOT / "configs" / "pfdf_waterfire_agent_pack"

VOLUME_GOLD_KEYS = frozenset({"Volume_m3", "log_volume", "Volume", "log1p_volume_m3"})

SOLVER_SAFE_KEYS = [
    "FireName",
    "State",
    "WatershedID",
    "DepositLongitude",
    "DepositLatitude",
    "Area_km2",
    "Area50_km2",
    "i30RainfallAnomaly",
    "FireStartDate",
]


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    try:
        if pd.isna(v):
            return None
    except (ValueError, TypeError):
        pass
    if hasattr(v, "item"):
        try:
            return _jsonable(v.item())
        except Exception:
            pass
    if isinstance(v, (str, int)):
        return v
    return str(v)


class PfdfDataAccess:
    """Thin wrapper over USGS PFDF inventory (HWA data_adapters)."""

    def __init__(self, inventory_path: Path | None = None):
        self.inventory_path = Path(inventory_path or DEFAULT_INVENTORY_PATH)
        self._inv: Optional[PFDFInventory] = None
        self._by_id: Dict[str, Dict[str, Any]] = {}

    @property
    def inventory(self) -> PFDFInventory:
        if self._inv is None:
            self._inv = load_inventory(self.inventory_path)
            self._by_id = {
                str(row["record_id"]): row.to_dict() for _, row in self._inv.df.iterrows()
            }
        return self._inv

    def list_record_ids(self, *, limit: Optional[int] = None) -> List[str]:
        ids = [str(x) for x in self.inventory.df["record_id"].tolist()]
        if limit is not None:
            return ids[: int(limit)]
        return ids

    def get_record(self, record_id: str) -> Dict[str, Any]:
        _ = self.inventory
        if record_id not in self._by_id:
            raise KeyError(f"unknown record_id: {record_id}")
        return dict(self._by_id[record_id])

    def load_sample(
        self,
        record_id: str,
        *,
        include_volume_target: bool = False,
    ) -> Dict[str, Any]:
        rec = self.get_record(record_id)
        fields: Dict[str, Any] = {}
        for k in SOLVER_SAFE_KEYS:
            if k in rec:
                fields[k] = _jsonable(rec[k])
        # Include additional non-gold scalars that predictors may need visibility for
        for k, v in rec.items():
            if k in VOLUME_GOLD_KEYS and not include_volume_target:
                continue
            if k in fields or k == "record_id" or str(k).startswith("Unnamed"):
                continue
            if k in (
                "MeandNBR",
                "FractionModHigh",
                "FractionBurned",
                "ModHigh50_km2",
            ):
                fields[k] = _jsonable(v)
        if include_volume_target:
            for k in VOLUME_GOLD_KEYS:
                if k in rec:
                    fields[k] = _jsonable(rec[k])

        out: Dict[str, Any] = {
            "dataset_id": "usgs_pfdf_inventory_v1",
            "record_id": record_id,
            "FireName": rec.get("FireName"),
            "State": rec.get("State"),
            "fields": fields,
        }
        try:
            from data_adapters.landsat.composites import PFDFLandsatStore, fire_id_from_name

            store = PFDFLandsatStore()
            fire_id = fire_id_from_name(str(rec.get("FireName", "")))
            has = None
            if hasattr(store, "has_fire"):
                has = bool(store.has_fire(fire_id))
            elif hasattr(store, "available_fire_ids"):
                has = fire_id in set(store.available_fire_ids())
            out["landsat"] = {"fire_id": fire_id, "has_fire": has}
        except Exception as exc:  # noqa: BLE001
            out["landsat"] = {"error": f"{type(exc).__name__}: {exc}"}
        return out


def resolve_usgs_pfdf_record(input_artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve full USGS PFDF inventory row for portfolio/HCG dispatch.

    Accepts either a complete ``usgs_pfdf_record_v1`` dict (with ``FireName``) or
    a stub ``{"record_id": ...}`` plus optional top-level ``record_id``.
    """
    record = input_artifacts.get("usgs_pfdf_record_v1")
    if isinstance(record, Mapping) and record.get("FireName"):
        return dict(record)
    record_id = None
    if isinstance(record, Mapping):
        record_id = record.get("record_id")
    record_id = record_id or input_artifacts.get("record_id")
    if not record_id:
        raise KeyError("record_id required for usgs_pfdf_record_v1")
    return PfdfDataAccess().get_record(str(record_id))
