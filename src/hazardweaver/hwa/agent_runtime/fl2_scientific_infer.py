"""HWA bridge: FL-2 scientific inference (M2 G2 + M3 solver caps).

Routes to official/BLOCKED infer paths. Never uses smoke/proxy rollouts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from hazardweaver.hcg.carp.native_eval.scientific.fl2_g2_official_infer import (
    default_scenario_id as g2_default_scenario_id,
    g2_caps,
    run_official_g2_infer,
)
from hazardweaver.hcg.carp.native_eval.scientific.fl2_g2_official_infer import (
    data_package_ready as fl2_data_package_ready,
)
from hazardweaver.hcg.carp.native_eval.scientific.fl2_solver_official_infer import (
    default_scenario_id as solver_default_scenario_id,
    run_official_solver_infer,
    solver_caps,
)
from hazardweaver.hcg.carp.scientific.paths import cap_scientific_dir

TASKPACK = "FL-2"
G2_CAPABILITY_IDS = g2_caps()
SOLVER_CAPABILITY_IDS = solver_caps()
ALL_FL2_INFERENCE_CAPS = G2_CAPABILITY_IDS + SOLVER_CAPABILITY_IDS


def fl2_scientific_ready() -> bool:
    return fl2_data_package_ready()


def run_fl2_capability_inference(
    capability_id: str,
    *,
    scenario_id: Optional[str] = None,
    split: str = "official_test",
    fidelity: Optional[str] = None,
    out_dir: Optional[Path] = None,
    strict_no_defaults: bool = False,
) -> Dict[str, Any]:
    """Dispatch FL-2 cap to official scientific inference path."""
    from hazardweaver.hcg.carp.scientific.fl2_data import splits_for_fidelity

    if strict_no_defaults:
        if not str(scenario_id or "").strip():
            return {
                "ok": False,
                "error": "scenario_id required in agent_strict_v2",
                "capability_id": capability_id,
            }
        if not str(split or "").strip():
            return {
                "ok": False,
                "error": "split required in agent_strict_v2",
                "capability_id": capability_id,
            }
    if fidelity is not None:
        allowed = splits_for_fidelity(fidelity)
        if split not in allowed:
            split = allowed[0]
    if capability_id not in ALL_FL2_INFERENCE_CAPS:
        return {
            "ok": False,
            "error": (
                f"FL-2 inference scope: expected one of {ALL_FL2_INFERENCE_CAPS}, "
                f"got {capability_id}"
            ),
        }
    if not fl2_scientific_ready():
        return {"ok": False, "error": "FL-2 scenario data package not ready"}

    cap_out = out_dir or cap_scientific_dir(TASKPACK, capability_id)
    if capability_id in G2_CAPABILITY_IDS:
        sid = scenario_id or g2_default_scenario_id(split=split)
        return run_official_g2_infer(
            capability_id,
            scenario_id=sid,
            split=split,
            out_dir=Path(cap_out),
        )
    sid = scenario_id or solver_default_scenario_id(split=split)
    return run_official_solver_infer(
        capability_id,
        scenario_id=sid,
        split=split,
        out_dir=Path(cap_out),
    )


def list_fl2_inference_caps() -> List[str]:
    return list(ALL_FL2_INFERENCE_CAPS)


def list_fl2_g2_caps() -> List[str]:
    return list(G2_CAPABILITY_IDS)


def list_fl2_solver_caps() -> List[str]:
    return list(SOLVER_CAPABILITY_IDS)
