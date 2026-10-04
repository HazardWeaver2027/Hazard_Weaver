"""HW-MED EWB native eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic, write_blocked

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EWB_MANIFEST = PROJECT_ROOT / "data" / "vendor" / "ewb" / "eval_subset" / "manifest.json"

VENDOR_CAPS = {
    "CAP-HWMED-01": ("RF-OPERATIONAL-PHYSICS-NWP", "gfs"),
    "CAP-HWMED-02": ("RF-SPECTRAL-NEURAL-WEATHER-MODE", "fcnv2"),
    "CAP-HWMED-03": ("RF-DETERMINISTIC-NEURAL-WEATHER", "pangu"),
    "CAP-HWMED-04": ("RF-GRAPH-NEURAL-WEATHER-MODEL", "graphcast"),
    "CAP-HWMED-05": ("RF-PROBABILISTIC-DIFFUSION-WEAT", "gencast"),
    "CAP-HWMED-06": ("RF-STATISTICAL-BASELINE", "climatology"),
}


def _load_subset() -> tuple[Path, Dict[str, Any]]:
    if not EWB_MANIFEST.is_file():
        raise FileNotFoundError(f"missing {EWB_MANIFEST}")
    meta = json.loads(EWB_MANIFEST.read_text(encoding="utf-8"))
    return EWB_MANIFEST.parent, meta


def _max_mae(forecast: np.ndarray, obs: np.ndarray) -> float:
    return float(np.max(np.abs(forecast.astype(np.float64) - obs.astype(np.float64))))


def _eval_forecast(
    capability_id: str,
    family_id: str,
    source_key: str,
    *,
    out_base: Optional[Path] = None,
) -> Dict[str, Any]:
    try:
        base, meta = _load_subset()
    except FileNotFoundError as exc:
        return write_blocked("HW-MED", capability_id, str(exc), out_base=out_base)

    obs = np.load(base / meta["obs_tmax"])
    forecast_file = meta["forecasts"][source_key]
    forecast = np.load(base / forecast_file)
    if forecast.ndim == 3:
        mae = float(np.mean([_max_mae(forecast[lead], obs) for lead in range(forecast.shape[0])]))
    else:
        mae = _max_mae(forecast, obs)

    metrics = {
        "metric_name": "max_mae",
        "metric_value": mae,
        "source": source_key,
        "synthetic_only": False,
        "data_source": "ewb_eval_subset",
        "note": "fixture forecast replay; not live CIRA fetch",
    }
    write_metrics("HW-MED", capability_id, metrics, base=out_base)
    write_replay_manifest(
        "HW-MED",
        capability_id,
        family_id=family_id,
        exec_ok=True,
        metric_name="max_mae",
        metric_value=mae,
        notes=f"EWB eval_subset {source_key}",
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}


def eval_hwmed_cap(
    capability_id: str,
    *,
    out_base: Optional[Path] = None,
    force_eval_subset_replay: bool = False,
    input_artifacts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    replay_subset = bool(force_eval_subset_replay)
    if not replay_subset and input_artifacts:
        replay_subset = bool(input_artifacts.get("force_eval_subset_replay"))
    if not replay_subset:
        try:
            from hazardweaver.hcg.carp.scientific.hwmed_data import scientific_data_ready
            from hazardweaver.hcg.carp.native_eval.scientific.hwmed_ewb import eval_hwmed_scientific

            if scientific_data_ready():
                return eval_hwmed_scientific(capability_id, out_base=out_base)
        except ImportError:
            pass

    if capability_id in VENDOR_CAPS:
        family_id, source_key = VENDOR_CAPS[capability_id]
        return _eval_forecast(capability_id, family_id, source_key, out_base=out_base)
    return eval_blocked_generic("HW-MED", capability_id, "no native eval wired", out_base=out_base)
