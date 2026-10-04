"""CAP-MH3-03 — LISFLOOD-FP external solver replay (DL-111)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.native_eval.blocked import write_blocked
from hazardweaver.hcg.carp.scientific.mh3_data import TASKPACK
from hazardweaver.hcg.carp.scientific.mh3_external_acquisition import external_manifest_path


def eval_mh3_lisflood(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    manifest = json.loads(external_manifest_path("CAP-MH3-03").read_text(encoding="utf-8"))
    if manifest.get("runtime_blocked"):
        return write_blocked(
            TASKPACK,
            "CAP-MH3-03",
            str(manifest.get("runtime_blocked_reason", "LISFLOOD vendor unavailable")),
            out_base=out_base,
        )
    return write_blocked(TASKPACK, "CAP-MH3-03", "LISFLOOD eval router incomplete", out_base=out_base)
