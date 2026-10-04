"""Matched-budget contract for Agent-Strict internal ablations (DL-151 era)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

MATCHED_PROTOCOL_ID = "matched_v1"
MATCHED_MAX_WALL_S = 300.0
MATCHED_MAX_STEPS = 15


def matched_protocol_from_env() -> Dict[str, Any]:
    wall = float(os.environ.get("HWA_AGENT_MAX_WALL_S", MATCHED_MAX_WALL_S))
    steps = int(os.environ.get("HWA_AGENT_MAX_STEPS", MATCHED_MAX_STEPS))
    contract = str(os.environ.get("HWA_ABLATION_PROTOCOL_CONTRACT", MATCHED_PROTOCOL_ID)).strip()
    hints = str(os.environ.get("HWA_STRICT_EXECUTION_HINTS", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    react_ablation = str(os.environ.get("HWA_AGENT_STRICT_REACT_ABLATION", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    chained_commit = str(os.environ.get("HWA_STRICT_CHAINED_COMMIT", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    react_pi_gate = str(os.environ.get("HWA_REACT_ADMISSIBILITY_GATE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return {
        "ablation_protocol_contract": contract,
        "max_wall_s": wall,
        "max_steps": steps,
        "strict_execution_hints": hints,
        "react_ablation": react_ablation,
        "strict_chained_commit": chained_commit,
        "react_admissibility_gate": react_pi_gate,
        "hints_as_middleware": hints,
    }


def read_run_protocol(workdir: Path) -> Optional[Dict[str, Any]]:
    meta_path = workdir / "run_meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    condition = str(meta.get("same_llm_condition") or "")
    controller_mode = meta.get("controller_mode")
    # Stamp can inherit stale env; runtime arm is condition + controller_mode.
    react_ablation = condition == "react" and not bool(controller_mode)
    return {
        "ablation_protocol_contract": meta.get("ablation_protocol_contract"),
        "max_wall_s": float(meta.get("max_wall_s") or 0),
        "max_steps": int(meta.get("max_steps") or 0),
        "strict_execution_hints": meta.get("strict_execution_hints"),
        "react_ablation": react_ablation,
        "strict_chained_commit": meta.get("strict_chained_commit"),
        "react_admissibility_gate": meta.get("react_admissibility_gate"),
        "hints_as_middleware": meta.get("hints_as_middleware"),
        "same_llm_condition": condition,
        "controller_mode": controller_mode,
        "exit_reason": meta.get("exit_reason"),
        "n_steps": int(meta.get("n_steps") or 0),
    }


def protocol_matches_matched_v1(proto: Optional[Mapping[str, Any]]) -> bool:
    if not proto:
        return False
    if str(proto.get("ablation_protocol_contract") or "") == MATCHED_PROTOCOL_ID:
        return True
    wall = float(proto.get("max_wall_s") or 0)
    steps = int(proto.get("max_steps") or 0)
    return abs(wall - MATCHED_MAX_WALL_S) < 1.0 and steps == MATCHED_MAX_STEPS


def protocol_mismatch_reason(
    proto: Optional[Mapping[str, Any]],
    *,
    expect_hints: Optional[bool] = None,
    expect_react_ablation: Optional[bool] = None,
    expect_strict_chained_commit: Optional[bool] = None,
    expect_react_admissibility_gate: Optional[bool] = None,
) -> Optional[str]:
    if not proto:
        return "missing_run_meta"
    if not protocol_matches_matched_v1(proto):
        return (
            f"budget_mismatch wall={proto.get('max_wall_s')} steps={proto.get('max_steps')} "
            f"contract={proto.get('ablation_protocol_contract')}"
        )
    if expect_hints is not None:
        got = proto.get("strict_execution_hints")
        if got is not None and bool(got) != expect_hints:
            return f"hints_mismatch expected={expect_hints} got={got}"
    if expect_react_ablation is not None:
        got = proto.get("react_ablation")
        if got is not None and bool(got) != expect_react_ablation:
            return f"react_ablation_mismatch expected={expect_react_ablation} got={got}"
    if expect_strict_chained_commit is not None:
        got = proto.get("strict_chained_commit")
        if got is None or bool(got) != expect_strict_chained_commit:
            return f"strict_chained_commit_mismatch expected={expect_strict_chained_commit} got={got}"
    if expect_react_admissibility_gate is not None:
        got = proto.get("react_admissibility_gate")
        if got is None or bool(got) != expect_react_admissibility_gate:
            return f"react_admissibility_gate_mismatch expected={expect_react_admissibility_gate} got={got}"
    return None


def run_meta_protocol_patch(condition: str = "") -> Dict[str, Any]:
    """Fields to stamp into run_meta after each headline cell."""
    p = matched_protocol_from_env()
    cond = str(condition or "").strip()
    react_ablation = cond == "react"
    return {
        "ablation_protocol_contract": p["ablation_protocol_contract"],
        "strict_execution_hints": p["strict_execution_hints"],
        "react_ablation": react_ablation,
        "strict_chained_commit": p.get("strict_chained_commit"),
        "react_admissibility_gate": p.get("react_admissibility_gate"),
        "hints_as_middleware": p.get("hints_as_middleware"),
    }
