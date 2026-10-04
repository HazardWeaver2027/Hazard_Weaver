"""HW-MED external forecast acquisition — GFS (CAP-01) and GenCast (CAP-05).

EWB remains the benchmark anchor (evaluator + official_test cases). Forecasts for
caps absent from EWB CIRA icechunk are acquired from authoritative external sources
and evaluated under the same HW-MED task contract (surface_air_temperature,
MaximumMeanAbsoluteError, 46 official_test heat_wave cases).

Status ladder (PI binding 2026-08-27):
  EXTERNAL_ACQUISITION_REQUIRED → (bounded audit + materialize) → scientific eval
  Only after a faithful bounded acquisition attempt may a cap be downgraded to
  RUNTIME_BLOCKED or DATA_BLOCKED — never permanent BLOCKED from CIRA absence alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.batch2.replay_certificate import write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import write_external_acquisition_required
from hazardweaver.hcg.carp.portfolio import load_portfolio
from hazardweaver.hcg.carp.scientific.hwmed_data import TASKPACK, taskpack_data_root

SPEC_REF = (
    "docs/engineering/hcg/external_expansion/HW-MED/HWMED_EXTERNAL_ACQUISITION_v1.md"
)

EXTERNAL_ACQUISITION_CAPS: Dict[str, Dict[str, Any]] = {
    "CAP-HWMED-01": {
        "capability_name": "GFS operational physics NWP",
        "route_family_id": "RF-OPERATIONAL-PHYSICS-NWP",
        "ewb_cira_gap": "GFS operational archive not in EWB CIRA_MODEL_NAMES v1.0.2",
        "authoritative_sources": [
            "NOAA NOMADS GFS 0.25° deterministic (surface temp)",
            "WeatherBench2 HRES zarr (fallback when NOMADS window matches case)",
        ],
        "evaluator_contract": (
            "EWB metrics.MaximumMeanAbsoluteError + ERA5 target; "
            "same 46 heat_wave official_test cases; lossless unit/time/grid adapter"
        ),
        "materialize_script": "scripts/bootstrap/materialize_hwmed_gfs_scientific.py",
        "ready_marker": "data/scientific/HW-MED/external_acquisition/CAP-HWMED-01/manifest.json",
    },
    "CAP-HWMED-05": {
        "capability_name": "GenCast probabilistic diffusion weather",
        "route_family_id": "RF-PROBABILISTIC-DIFFUSION-WEAT",
        "ewb_cira_gap": "GenCast not published in EWB cira-icechunk store",
        "authoritative_sources": [
            "DeepMind google-deepmind/gencast official inference / checkpoint",
            "Official GenCast artifact ingest to local zarr under data/scientific/HW-MED/",
        ],
        "evaluator_contract": (
            "EWB metrics.MaximumMeanAbsoluteError (ensemble mean or official EWB GenCast "
            "protocol when published); same HW-MED task contract"
        ),
        "materialize_script": "scripts/bootstrap/materialize_hwmed_gencast_scientific.py",
        "ready_marker": "data/scientific/HW-MED/external_acquisition/CAP-HWMED-05/manifest.json",
    },
}


def external_acquisition_root() -> Path:
    return taskpack_data_root(TASKPACK) / "external_acquisition"


def external_manifest_path(capability_id: str) -> Path:
    return external_acquisition_root() / capability_id / "manifest.json"


def external_forecast_ready(capability_id: str) -> bool:
    path = external_manifest_path(capability_id)
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return bool(data.get("materialized")) and not data.get("runtime_blocked")


def external_acquisition_status(capability_id: str) -> Dict[str, Any]:
    spec = EXTERNAL_ACQUISITION_CAPS.get(capability_id, {})
    manifest_path = external_manifest_path(capability_id)
    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "capability_id": capability_id,
        "external_acquisition_required": not external_forecast_ready(capability_id),
        "materialized": bool(manifest.get("materialized")),
        "runtime_blocked": bool(manifest.get("runtime_blocked")),
        "audit_completed": bool(manifest.get("audit_completed")),
        "manifest_path": str(manifest_path),
        "spec_ref": SPEC_REF,
        **{k: spec.get(k) for k in ("authoritative_sources", "evaluator_contract", "ewb_cira_gap")},
    }


def _family_id(capability_id: str) -> str:
    port = load_portfolio(TASKPACK)
    cap = port.capability_by_id(capability_id)
    return (cap.family_id if cap else "unknown") or "unknown"


def write_external_acquisition_pending(
    capability_id: str,
    *,
    out_base: Optional[Path] = None,
    extra_reason: str = "",
) -> Dict[str, Any]:
    spec = EXTERNAL_ACQUISITION_CAPS[capability_id]
    reason = spec["ewb_cira_gap"]
    if extra_reason:
        reason = f"{reason}; {extra_reason}"
    plan = {
        "status": "EXTERNAL_ACQUISITION_REQUIRED",
        "authoritative_sources": spec["authoritative_sources"],
        "evaluator_contract": spec["evaluator_contract"],
        "materialize_script": spec["materialize_script"],
        "ready_marker": spec["ready_marker"],
        "spec_ref": SPEC_REF,
        "note": (
            "EWB is benchmark anchor; CIRA absence is not permanent BLOCKED. "
            "Bounded faithful acquisition required before RUNTIME_BLOCKED downgrade."
        ),
    }
    return write_external_acquisition_required(
        TASKPACK,
        capability_id,
        reason,
        acquisition_plan=plan,
        out_base=out_base,
    )


def eval_external_cap(
    capability_id: str,
    *,
    out_base: Optional[Path] = None,
) -> Dict[str, Any]:
    if capability_id not in EXTERNAL_ACQUISITION_CAPS:
        raise ValueError(f"not an external acquisition cap: {capability_id}")

    if not external_forecast_ready(capability_id):
        return write_external_acquisition_pending(capability_id, out_base=out_base)

    manifest = json.loads(external_manifest_path(capability_id).read_text(encoding="utf-8"))
    if manifest.get("runtime_blocked"):
        return write_external_acquisition_required(
            TASKPACK,
            capability_id,
            str(manifest.get("runtime_blocked_reason", "RUNTIME_BLOCKED after acquisition audit")),
            acquisition_plan=manifest.get("acquisition_plan") or {},
            out_base=out_base,
            extra={"runtime_blocked": True, "audit_completed": True},
        )

    # Router hook: materialized external forecasts eval via dedicated modules (next sprint).
    eval_module = manifest.get("eval_router")
    if eval_module == "hwmed_gfs_ewb":
        from hazardweaver.hcg.carp.native_eval.scientific.hwmed_gfs import eval_hwmed_gfs_scientific

        return eval_hwmed_gfs_scientific(out_base=out_base)
    if eval_module == "hwmed_gencast_ewb":
        from hazardweaver.hcg.carp.native_eval.scientific.hwmed_gencast import eval_hwmed_gencast_scientific

        return eval_hwmed_gencast_scientific(out_base=out_base)

    write_replay_manifest(
        TASKPACK,
        capability_id,
        family_id=_family_id(capability_id),
        exec_ok=False,
        notes="EXTERNAL materialized but eval router not wired",
        base=out_base,
    )
    return write_external_acquisition_pending(
        capability_id,
        out_base=out_base,
        extra_reason="manifest materialized=true but eval_router missing",
    )
