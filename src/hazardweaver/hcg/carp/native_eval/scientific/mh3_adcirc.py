"""CAP-MH3-04 — ADCIRC external solver replay (DL-111)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.native_eval.blocked import write_blocked
from hazardweaver.hcg.carp.scientific.mh3_data import TASKPACK
from hazardweaver.hcg.carp.scientific.mh3_external_acquisition import external_manifest_path


def eval_mh3_adcirc(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    manifest = json.loads(external_manifest_path("CAP-MH3-04").read_text(encoding="utf-8"))
    if manifest.get("runtime_blocked"):
        return write_blocked(
            TASKPACK,
            "CAP-MH3-04",
            str(manifest.get("runtime_blocked_reason", "ADCIRC vendor unavailable")),
            out_base=out_base,
        )
    return write_blocked(TASKPACK, "CAP-MH3-04", "ADCIRC eval router incomplete", out_base=out_base)
