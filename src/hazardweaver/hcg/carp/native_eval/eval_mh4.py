"""MH-4 native eval — thin wrapper for manifest CSV eval_mh_4 entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic
from hazardweaver.hcg.carp.native_eval.scientific.mh4_climada_tc import run_cap_mh4_r06
from hazardweaver.hcg.carp.native_eval.scientific.mh4_dependence_joint import run_cap_mh4_r03
from hazardweaver.hcg.carp.native_eval.scientific.mh4_hazus_blocked import run_cap_mh4_r04
from hazardweaver.hcg.carp.native_eval.scientific.mh4_independent_baseline import run_cap_mh4_r02
from hazardweaver.hcg.carp.native_eval.scientific.mh4_nhc_substrate import run_cap_mh4_r01

_RUNNERS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "CAP-MH4-R01": run_cap_mh4_r01,
    "CAP-MH4-R02": run_cap_mh4_r02,
    "CAP-MH4-R03": run_cap_mh4_r03,
    "CAP-MH4-R04": run_cap_mh4_r04,
    "CAP-MH4-R06": run_cap_mh4_r06,
}


def eval_mh4_cap(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if capability_id == "CAP-MH4-R05":
        return eval_blocked_generic(
            "MH-4",
            capability_id,
            "DEFERRED_P2 coupled model-chain; no eval path in repo",
            out_base=out_base,
        )
    runner = _RUNNERS.get(capability_id)
    if runner is None:
        return eval_blocked_generic("MH-4", capability_id, "no native eval wired", out_base=out_base)
    return runner()
