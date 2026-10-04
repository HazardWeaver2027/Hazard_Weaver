"""MH-3 compound inundation native eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic
from hazardweaver.hcg.carp.native_eval.eval_mh3_stages import eval_mh3_stage

MH3_CAPS = {
    "CAP-MH3-01",
    "CAP-MH3-02",
    "CAP-MH3-03",
    "CAP-MH3-04",
    "CAP-MH3-05",
    "CAP-MH3-06",
}


def eval_mh3_cap(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    try:
        from hazardweaver.hcg.carp.scientific.mh3_data import scientific_data_ready
        from hazardweaver.hcg.carp.native_eval.scientific.mh3_sfincs import eval_mh3_scientific

        if scientific_data_ready():
            return eval_mh3_scientific(capability_id, out_base=out_base)
    except ImportError:
        pass

    if capability_id in MH3_CAPS:
        return eval_mh3_stage(capability_id, out_base=out_base)
    return eval_blocked_generic("MH-3", capability_id, "no native eval wired", out_base=out_base)
