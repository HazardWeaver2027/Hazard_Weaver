"""Π_adm abstention gate — Fusion Task 7 / """

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from hazardweaver.hwa.scientific_controller.reason_codes import ASciVerdict


def pi_adm_empty(routes: Sequence[Mapping[str, Any]]) -> bool:
    return not any(r.get("admissible") for r in routes)


def abstention_allowed(
    routes: Sequence[Mapping[str, Any]],
    *,
    counterfactual_recoverable: bool = False,
) -> Dict[str, Any]:
    """submit_abstention only when Π_adm empty and no user-recoverable Δs."""
    if not pi_adm_empty(routes):
        return {
            "allowed": False,
            "error": "pi_adm_nonempty",
            "message": "Cannot abstain while admissible routes exist.",
        }
    if counterfactual_recoverable:
        return {
            "allowed": False,
            "error": "counterfactual_recoverable",
            "message": "User clarification may restore admissibility.",
        }
    return {"allowed": True}


def pre_commit_before_run_capability(
    routes: Sequence[Mapping[str, Any]],
    *,
    has_lease: bool = False,
    committed: bool = False,
) -> Dict[str, Any]:
    """Block irreversible headline execute when Π_adm empty (abstain/clarify first)."""
    # Post-commit path: lease + active route already validated admissibility at commit.
    if committed and has_lease:
        return {"ok": True}
    if pi_adm_empty(routes):
        return {
            "ok": False,
            "error": "pi_adm_empty_pre_commit",
            "message": (
                "Cannot run_capability when no admissible route; "
                "use submit_clarification or submit_abstention first."
            ),
        }
    if not has_lease:
        return {
            "ok": False,
            "error": "scoped_execution_lease_required",
            "message": "Headline execution requires controller commit + lease.",
        }
    return {"ok": True}


def pre_commit_abstain_check(
    routes: Sequence[Mapping[str, Any]],
    *,
    committed: bool = False,
) -> Dict[str, Any]:
    """Block post-hoc abstain after irreversible commit."""
    if committed and not pi_adm_empty(routes):
        return {
            "ok": False,
            "error": "post_hoc_abstain_blocked",
            "message": "Abstention after commit requires empty Π_adm.",
        }
    return {"ok": True}


def unknown_triggers_clarify(a_sci: Mapping[str, Any]) -> bool:
    verdict = str(a_sci.get("verdict") or "")
    return verdict == ASciVerdict.UNKNOWN_PENDING_THEORY.value


def is_committed_execution_state(host: Any) -> bool:
    """True when a route was committed (lease or active_route_id on controller)."""
    try:
        from hazardweaver.hwa.agent_runtime.tool_host import resolve_agent_host

        resolved = resolve_agent_host(host)
        if getattr(resolved, "_react_headline_solve_ready", None):
            return True
    except Exception:  # noqa: BLE001
        resolved = host
    try:
        from hazardweaver.hwa.agent_runtime.tool_host import resolve_agent_host
        from hazardweaver.hwa.runtime.lease_manager import require_lease

        resolved = resolve_agent_host(host)
        if require_lease(resolved) is not None:
            return True
    except Exception:  # noqa: BLE001
        pass
    ctrl = getattr(host, "controller", None)
    if ctrl is None:
        try:
            from hazardweaver.hwa.agent_runtime.tool_host import resolve_agent_host

            resolved = resolve_agent_host(host)
            ctrl = getattr(resolved, "controller", None)
        except Exception:  # noqa: BLE001
            ctrl = None
    if ctrl is not None:
        state = getattr(ctrl, "state", None)
        if state is not None and str(getattr(state, "active_route_id", "") or "").strip():
            return True
    return False


def _react_allowed_capability_ids(task: Mapping[str, Any]) -> list[str]:
    sv = task.get("solver_visible") or {}
    inputs = sv.get("inputs") or {}
    edges = inputs.get("allowed_edge_ids") or []
    return [str(e) for e in edges if e and str(e).startswith("CAP-")]


def _react_attempted_capability_ids(host: Any) -> set[str]:
    attempted: set[str] = set()
    transcript = getattr(host, "transcript", None) or []
    for row in transcript:
        if str(row.get("role") or "") != "tool":
            continue
        if str(row.get("name") or "") != "run_capability":
            continue
        obs = row.get("observation") or row.get("content") or {}
        if isinstance(obs, str):
            continue
        if isinstance(obs, Mapping) and isinstance(obs.get("result"), Mapping):
            obs = obs["result"]
        cid = str((obs or {}).get("capability_id") or "").strip()
        if cid:
            attempted.add(cid)
    ledger = getattr(host, "ledger", None)
    if ledger is not None:
        for step in getattr(ledger, "steps", ()) or ():
            if str(step.get("tool") or "") != "run_capability":
                continue
            cid = str(step.get("capability_id") or "").strip()
            if cid:
                attempted.add(cid)
    return attempted


def coupled_spurious_no_legal_route_block(
    host: Any,
    reason_code: str,
) -> Optional[Dict[str, Any]]:
    """Coupled-4M solve-mandatory: forbid NO_LEGAL_ROUTE before exhausting allowed CAP-*."""
    code = str(reason_code).strip()
    if code != "NO_LEGAL_ROUTE":
        return None
    try:
        from hazardweaver.hwa.agent_runtime.tool_host import resolve_agent_host
        from hazardweaver.hwa.experiments.coupled_4m_mode_v1 import coupled_solve_mandatory_enabled

        resolved = resolve_agent_host(host)
        task = getattr(resolved, "task", None) or {}
        if not coupled_solve_mandatory_enabled(task):
            return None
    except Exception:  # noqa: BLE001
        return None

    allowed = _react_allowed_capability_ids(task)
    if not allowed:
        return None
    attempted = _react_attempted_capability_ids(resolved)
    remaining = [c for c in allowed if c not in attempted]
    if remaining:
        return {
            "allowed": False,
            "error": "coupled_must_try_allowed_caps",
            "message": (
                "Coupled-4M solve-mandatory: NO_LEGAL_ROUTE forbidden while allowed CAPs remain. "
                f"Call run_capability on: {remaining[:6]}"
            ),
            "remaining_capability_ids": remaining,
        }
    return None


def react_spurious_no_legal_route_block(
    host: Any,
    reason_code: str,
) -> Optional[Dict[str, Any]]:
    """ReAct ablation: forbid NO_LEGAL_ROUTE before exhausting allowed CAP-* attempts."""
    code = str(reason_code).strip()
    if code != "NO_LEGAL_ROUTE":
        return None
    try:
        from hazardweaver.hwa.agent_runtime.tool_host import resolve_agent_host
        from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_react_ablation_enabled
        from hazardweaver.hwa.runtime.react_headline_execute import is_react_headline_task

        resolved = resolve_agent_host(host)
        task = getattr(resolved, "task", None) or {}
        if not agent_strict_v2_react_ablation_enabled(task) or not is_react_headline_task(task):
            return None
    except Exception:  # noqa: BLE001
        return None

    allowed = _react_allowed_capability_ids(task)
    if not allowed:
        return None
    attempted = _react_attempted_capability_ids(resolved)
    remaining = [c for c in allowed if c not in attempted]
    if remaining:
        return {
            "allowed": False,
            "error": "react_must_try_allowed_caps",
            "message": (
                "ReAct ablation: NO_LEGAL_ROUTE forbidden while allowed_edge_ids remain untried. "
                f"Call run_capability on each allowed cap before abstaining. Remaining: {remaining[:6]}"
            ),
            "remaining_capability_ids": remaining,
        }
    return None


def submit_abstention_allowed(
    host: Any,
    reason_code: str,
    routes: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Full abstention gate: Π_adm, post-commit policy, and reason-code semantics."""
    code = str(reason_code).strip()
    try:
        from hazardweaver.hwa.agent_runtime.tool_host import resolve_agent_host
        from hazardweaver.hwa.experiments.ablation_manual_pilot_execution_v1 import ablation_abstention_allowed

        resolved = resolve_agent_host(host)
        task = getattr(resolved, "task", None) or {}
        ablation_gate = ablation_abstention_allowed(task, code, routes)
        if not ablation_gate.get("allowed"):
            return ablation_gate
        from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import unified_abstention_allowed

        unified_gate = unified_abstention_allowed(task, code, routes)
        if not unified_gate.get("allowed"):
            return unified_gate
    except Exception:  # noqa: BLE001
        pass
    if code in {"NO_ADMISSIBLE_ROUTE", "VERIFY_FAILED"}:
        return {"allowed": True, "vce_selective_abstain": True}

    react_block = react_spurious_no_legal_route_block(host, code)
    if react_block is not None:
        return react_block

    coupled_block = coupled_spurious_no_legal_route_block(host, code)
    if coupled_block is not None:
        return coupled_block

    gate = abstention_allowed(routes)
    if not gate.get("allowed"):
        return gate

    committed = is_committed_execution_state(host)
    if committed:
        post = pre_commit_abstain_check(routes, committed=True)
        if not post.get("ok"):
            return {
                "allowed": False,
                "error": post.get("error", "post_hoc_abstain_blocked"),
                "message": post.get("message"),
            }
        if str(reason_code).strip() == "NO_LEGAL_ROUTE":
            return {
                "allowed": False,
                "error": "post_commit_no_legal_route_forbidden",
                "message": (
                    "NO_LEGAL_ROUTE is invalid after controller commit. "
                    "A route was already committed; use submit_clarification or a "
                    "failure-specific abstain reason (e.g. MISSING_REQUIRED_ARTIFACT)."
                ),
            }
    return {"allowed": True}
