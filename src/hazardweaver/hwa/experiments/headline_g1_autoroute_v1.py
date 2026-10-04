"""G1 deterministic route commit — DS CBR warm-start + controller execute (no answer copy)."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwa.experiments.headline_route_profile_v1 import is_g1_profile


def g1_autocommit_enabled() -> bool:
    if not is_g1_profile():
        return False
    return str(os.environ.get("HWA_HEADLINE_G1_AUTOCOMMIT", "1")).lower() in (
        "1",
        "true",
        "yes",
    )


def _cbr_route_id(task: Mapping[str, Any]) -> str:
    meta = task.get("metadata") or {}
    return str(meta.get("ds_cbr_route_id") or "").strip()


def _pick_cbr_route(
    routes: list[Mapping[str, Any]],
    route_id: str,
) -> Optional[Dict[str, Any]]:
    for route in routes:
        if not route.get("admissible"):
            continue
        if str(route.get("route_id") or "") == route_id:
            return dict(route)
    return None


def _admissible_routes(enum_result: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [dict(r) for r in (enum_result.get("routes") or []) if r.get("admissible")]


def resolve_g1_autocommit_route_id(
    enum_result: Mapping[str, Any],
    task: Mapping[str, Any],
) -> Optional[str]:
    """Pick route for G1 autocommit: DS CBR if admissible, else REMSA rank-1."""
    routes = _admissible_routes(enum_result)
    if not routes:
        return None
    cbr_rid = _cbr_route_id(task)
    if cbr_rid and _pick_cbr_route(routes, cbr_rid):
        return cbr_rid
    # Single-route L4 cells (e.g. HW-MED): CBR may point off-allowed cap — still autocommit.
    return str(routes[0].get("route_id") or "").strip() or None


def apply_g1_autocommit_if_eligible(
    controller: Any,
    enum_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """After enumerate: propose+commit+execute G1 route once (no LLM parse)."""
    out = dict(enum_result)
    if not g1_autocommit_enabled():
        return out
    if getattr(controller, "_g1_autocommit_done", False):
        return out
    state = getattr(controller, "state", None)
    if state is not None and str(getattr(state, "active_route_id", "") or "").strip():
        return out

    task = getattr(controller, "task", {}) or {}
    rid = resolve_g1_autocommit_route_id(enum_result, task)
    if not rid:
        return out

    controller._g1_autocommit_done = True  # noqa: SLF001
    proposed = controller.propose_route(rid, rationale="g1_autocommit")
    if not proposed.get("ok"):
        out["g1_autocommit"] = {"ok": False, "stage": "propose", **proposed}
        return out

    committed = controller.commit_route(route_id=rid)
    exec_result: Dict[str, Any] = {}
    if committed.get("ok"):
        exec_result = controller.execute_route(
            route_id=committed.get("route_id") or rid,
            execution_token=committed.get("execution_token"),
            host=getattr(controller, "env", None),
            out_dir=getattr(controller, "workdir", None),
        )
        committed = {**committed, "execution": exec_result}
        if exec_result.get("ok"):
            committed["execution_id"] = exec_result.get("execution_id")
            committed["final_artifact_id"] = exec_result.get("final_artifact_id")
            committed["next_step"] = "submit_solution"
            committed["message"] = (
                "G1 autocommit executed CBR route. Call submit_solution with "
                "route_id, execution_id, and final_artifact_id from g1_autocommit."
            )
        else:
            committed["ok"] = False
            committed["error"] = exec_result.get("error") or "execution_failed"
            committed["failure_class"] = exec_result.get("failure_class", "unknown")
            committed["next_step"] = "submit_clarification"

    out["g1_autocommit"] = committed
    out["g1_autocommit_applied"] = bool(exec_result.get("ok"))
    return out


def g1_autocommit_submit_hint(refresh: Mapping[str, Any]) -> Optional[str]:
    """System hint for the agent loop after successful G1 autocommit."""
    ac = refresh.get("g1_autocommit") or {}
    if not refresh.get("g1_autocommit_applied"):
        return None
    exec_block = ac.get("execution") or {}
    return (
        "G1 autocommit succeeded. Call submit_solution now with "
        f"route_id={ac.get('route_id')!r}, "
        f"execution_id={ac.get('execution_id') or exec_block.get('execution_id')!r}, "
        f"final_artifact_id={ac.get('final_artifact_id') or exec_block.get('final_artifact_id')!r}."
    )


def try_finish_g1_autocommit_loop(env: Any, refresh: Mapping[str, Any]) -> bool:
    """After autocommit+execute, submit_solution without another LLM turn (avoid limit_wall)."""
    if not refresh.get("g1_autocommit_applied"):
        return False
    ac = refresh.get("g1_autocommit") or {}
    exec_block = ac.get("execution") or {}
    rid = str(ac.get("route_id") or "").strip()
    eid = str(ac.get("execution_id") or exec_block.get("execution_id") or "").strip()
    aid = str(ac.get("final_artifact_id") or exec_block.get("final_artifact_id") or "").strip()
    if not (rid and eid and aid):
        return False
    result = env.execute(
        {
            "name": "submit_solution",
            "arguments": {
                "route_id": rid,
                "execution_id": eid,
                "final_artifact_id": aid,
                "rationale": "g1_autocommit_terminal_submit",
            },
        }
    )
    if getattr(env, "submitted", False) or (isinstance(result, dict) and result.get("ok")):
        env.append_transcript(
            {
                "role": "system",
                "event": "g1_autocommit_terminal_submit",
                "route_id": rid,
                "execution_id": eid,
                "final_artifact_id": aid,
            }
        )
        return True
    return False
