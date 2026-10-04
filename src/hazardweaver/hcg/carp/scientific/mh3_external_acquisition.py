"""MH-3 solver external acquisition — HEC-RAS / LISFLOOD / ADCIRC (DL-111)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.batch2.replay_certificate import write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import write_external_acquisition_required
from hazardweaver.hcg.carp.scientific.mh3_data import TASKPACK, taskpack_data_root

SPEC_REF = "docs/engineering/hcg/external_expansion/MH-3/MH3_SOLVER_EXTERNAL_ACQUISITION_v1.md"

EXTERNAL_ACQUISITION_CAPS: Dict[str, Dict[str, Any]] = {
    "CAP-MH3-02": {
        "capability_name": "HEC-RAS 2D shallow water",
        "vendor": "USACE HEC-RAS",
        "materialize_script": "scripts/bootstrap/materialize_mh3_hecras_scientific.py",
    },
    "CAP-MH3-03": {
        "capability_name": "LISFLOOD-FP inertial solver",
        "vendor": "LISFLOOD-FP",
        "materialize_script": "scripts/bootstrap/materialize_mh3_lisflood_scientific.py",
    },
    "CAP-MH3-04": {
        "capability_name": "ADCIRC one-way coastal chain",
        "vendor": "ADCIRC",
        "materialize_script": "scripts/bootstrap/materialize_mh3_adcirc_scientific.py",
    },
}


def external_acquisition_root() -> Path:
    return taskpack_data_root(TASKPACK) / "external_acquisition"


def external_manifest_path(capability_id: str) -> Path:
    return external_acquisition_root() / capability_id / "manifest.json"


def external_solver_ready(capability_id: str) -> bool:
    path = external_manifest_path(capability_id)
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return bool(data.get("materialized")) and not data.get("runtime_blocked")


def eval_external_solver_cap(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if capability_id not in EXTERNAL_ACQUISITION_CAPS:
        raise ValueError(f"not an MH-3 external cap: {capability_id}")

    if not external_solver_ready(capability_id):
        spec = EXTERNAL_ACQUISITION_CAPS[capability_id]
        return write_external_acquisition_required(
            TASKPACK,
            capability_id,
            f"{spec['vendor']} vendor bundle not materialized",
            acquisition_plan={
                "status": "EXTERNAL_ACQUISITION_REQUIRED",
                "materialize_script": spec["materialize_script"],
                "spec_ref": SPEC_REF,
            },
            out_base=out_base,
        )

    manifest = json.loads(external_manifest_path(capability_id).read_text(encoding="utf-8"))
    if manifest.get("runtime_blocked"):
        return write_external_acquisition_required(
            TASKPACK,
            capability_id,
            str(manifest.get("runtime_blocked_reason", "RUNTIME_BLOCKED")),
            acquisition_plan=manifest,
            out_base=out_base,
            extra={"runtime_blocked": True},
        )

    router = manifest.get("eval_router", "")
    if router == "mh3_hecras_replay":
        from hazardweaver.hcg.carp.native_eval.scientific.mh3_hecras import eval_mh3_hecras

        return eval_mh3_hecras(out_base=out_base)
    if router == "mh3_lisflood_replay":
        from hazardweaver.hcg.carp.native_eval.scientific.mh3_lisflood import eval_mh3_lisflood

        return eval_mh3_lisflood(out_base=out_base)
    if router == "mh3_adcirc_replay":
        from hazardweaver.hcg.carp.native_eval.scientific.mh3_adcirc import eval_mh3_adcirc

        return eval_mh3_adcirc(out_base=out_base)

    write_replay_manifest(
        TASKPACK,
        capability_id,
        family_id="unknown",
        exec_ok=False,
        notes="MH-3 external materialized but eval router missing",
        base=out_base,
    )
    return write_external_acquisition_required(
        TASKPACK,
        capability_id,
        "eval router missing after materialize",
        acquisition_plan=manifest,
        out_base=out_base,
    )
