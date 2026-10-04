"""Terminal episode policy for manual ablation pilot — no abstention spin on DCA failures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwa.experiments.ablation_manual_pilot_v1 import ablation_manual_pilot_enabled
from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import solve_mandatory_episode_policy_enabled

TERMINAL_DCA_FAILURE_ERRORS = frozenset(
    {
        "task_metric_mismatch",
        "dca_preflight_failed",
        "final_artifact_missing_score",
        "final_artifact_mismatch",
        "contract_violation",
        "limit_steps_unsubmitted_commit",
        "enumerate_spin_limit_steps",
        "clarify_spin_limit_steps",
        "no_admissible_routes_spin",
        "list_inventory_spin",
        "post_enumerate_list_inventory_spin",
        "over_commit_spin",
    }
)

_SPIN_EXIT_REASONS = frozenset(
    {
        "limit_steps",
        "limit_wall",
        "parse_loop",
        "failed_clarify",
        "env_null_no_user",
    }
)

_BLOCKED_ABSTENTION_ERRORS = frozenset(
    {
        "pi_adm_nonempty",
        "post_hoc_abstain_blocked",
        "post_commit_no_legal_route_forbidden",
        "abstention_not_allowed",
        "ablation_solve_mandatory_admissible_routes",
        "ablation_solve_mandatory_requires_route_probe",
        "unified_solve_mandatory_admissible_routes",
        "unified_solve_mandatory_requires_route_probe",
        "no_admissible_routes_spin",
    }
)


def _task_meta(task: Mapping[str, Any]) -> Dict[str, Any]:
    meta = task.get("metadata") or {}
    return dict(meta) if isinstance(meta, Mapping) else {}


def is_terminal_dca_failure(error: str) -> bool:
    return str(error or "").strip() in TERMINAL_DCA_FAILURE_ERRORS


def _parse_tool_payload(content: Any) -> Dict[str, Any]:
    if isinstance(content, Mapping):
        payload = dict(content)
    else:
        try:
            payload = json.loads(str(content or "{}"))
        except (TypeError, json.JSONDecodeError):
            return {}
    if not isinstance(payload, Mapping):
        return {}
    result = payload.get("result")
    if isinstance(result, Mapping):
        return dict(result)
    return dict(payload)


def extract_submit_error(submit_result: Mapping[str, Any]) -> str:
    err = str(submit_result.get("error") or "")
    inner = _parse_tool_payload(submit_result.get("content"))
    if inner:
        err = str(inner.get("error") or (inner.get("result") or {}).get("error") or err)
    return err.strip()


def _committed_route_id(workdir: Path) -> str:
    decisions = workdir / "controller_decisions.jsonl"
    if not decisions.is_file():
        return ""
    for line in reversed(decisions.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("action") or "") != "commit":
            continue
        rid = str(row.get("route_id") or "").strip()
        if rid:
            return rid
        args = row.get("submit_solution_args")
        if isinstance(args, Mapping):
            rid = str(args.get("route_id") or "").strip()
            if rid:
                return rid
    return ""


def _registry_has_successful_execution(workdir: Path, route_id: str = "") -> bool:
    from hazardweaver.hwa.agent_runtime.execution_schema import list_registry_executions

    rid = str(route_id or "").strip()
    for er in list_registry_executions(workdir):
        if str(er.get("status") or "") not in {"ok", "success"}:
            continue
        if rid and str(er.get("route_id") or "").strip() != rid:
            continue
        if str(er.get("execution_id") or "").strip():
            return True
    return False


def recover_submit_handles_from_workdir(workdir: Path, task: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    """Best-effort route/execution/artifact ids from registry + controller commit."""
    workdir = Path(workdir)
    handles = {"route_id": "", "execution_id": "", "final_artifact_id": ""}

    from hazardweaver.hwa.experiments.unified_gold_registry_submit_v1 import (
        _inventory_row,
        prefer_unified_gold_registry_handles,
    )

    inv = _inventory_row(workdir, task)
    gold_handles = prefer_unified_gold_registry_handles(workdir, inv)
    if gold_handles:
        return dict(gold_handles)

    from hazardweaver.hwa.agent_runtime.execution_schema import list_registry_executions

    ok_rows = [
        er
        for er in list_registry_executions(workdir)
        if str(er.get("status") or "") in {"ok", "success"}
    ]
    if ok_rows:
        er = ok_rows[-1]
        fa = er.get("final_artifact") if isinstance(er.get("final_artifact"), Mapping) else {}
        handles["route_id"] = str(er.get("route_id") or "").strip()
        handles["execution_id"] = str(er.get("execution_id") or "").strip()
        handles["final_artifact_id"] = str(fa.get("artifact_id") or "").strip()
        if all(handles.values()):
            return handles

    decisions = workdir / "controller_decisions.jsonl"
    if decisions.is_file():
        for line in reversed(decisions.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("action") or "") != "commit":
                continue
            exec_block = row.get("execution") if isinstance(row.get("execution"), Mapping) else {}
            args = row.get("submit_solution_args")
            if not isinstance(args, Mapping) and isinstance(exec_block, Mapping):
                args = exec_block.get("submit_solution_args")
            args = dict(args) if isinstance(args, Mapping) else {}
            route_id = str(
                args.get("route_id") or row.get("route_id") or exec_block.get("route_id") or ""
            ).strip()
            if route_id and _registry_has_successful_execution(workdir, route_id):
                handles["route_id"] = route_id
                handles["execution_id"] = str(
                    args.get("execution_id")
                    or row.get("execution_id")
                    or exec_block.get("execution_id")
                    or ""
                ).strip()
                handles["final_artifact_id"] = str(
                    args.get("final_artifact_id")
                    or row.get("final_artifact_id")
                    or exec_block.get("final_artifact_id")
                    or ""
                ).strip()
                if handles["route_id"] and handles["execution_id"]:
                    return handles
    return handles


def _extract_dca_dict(failure: Mapping[str, Any]) -> Dict[str, Any]:
    dca = failure.get("dca")
    if isinstance(dca, Mapping):
        if isinstance(dca.get("dca"), Mapping):
            return dict(dca["dca"])
        return dict(dca)
    nested = failure.get("result")
    if isinstance(nested, Mapping) and isinstance(nested.get("dca"), Mapping):
        inner = nested["dca"]
        if isinstance(inner.get("dca"), Mapping):
            return dict(inner["dca"])
        return dict(inner)
    return {}


def build_invalid_terminal_dca(
    task: Mapping[str, Any],
    *,
    failure: Mapping[str, Any],
    handles: Mapping[str, str],
) -> Dict[str, Any]:
    """DCA envelope for ablation pilot terminal invalid solve."""
    meta = _task_meta(task)
    err = str(failure.get("error") or extract_submit_error(failure) or "dca_preflight_failed")
    dca = _extract_dca_dict(failure)
    if dca:
        out = dict(dca)
        out["valid"] = False
        out.setdefault("outcome", "solve")
        out.setdefault("reason_code", err)
        out.setdefault("contract_violated", True)
        out.setdefault("counted", False)
        return out
    return {
        "taskpack_id": str(meta.get("taskpack_id") or task.get("taskpack_id") or ""),
        "agent_id": "hazardweaver",
        "dca_score": 0.0,
        "counted": False,
        "contract_violated": True,
        "budget_within_limit": True,
        "metric_name": err,
        "raw_score": None,
        "outcome": "solve",
        "reason_code": err,
        "difficulty_tier": str(meta.get("difficulty_tier") or "L1"),
        "valid": False,
        "route_id": str(handles.get("route_id") or ""),
        "execution_id": str(handles.get("execution_id") or ""),
        "final_artifact_id": str(handles.get("final_artifact_id") or ""),
        "ablation_terminal_policy": "invalid_dca_terminal_v1",
    }


def _cap_from_route_id(route_id: str) -> str:
    import re

    match = re.search(r"CAP-[A-Z0-9-]+", str(route_id or ""))
    return match.group(0) if match else ""


def try_honest_registry_terminal_dca(
    env: Any,
    task: Mapping[str, Any],
    handles: Mapping[str, str],
    *,
    workdir: Optional[Path] = None,
) -> bool:
    """Grade via official DCA when registry holds ok execution on curator gold route."""
    wd = Path(workdir or getattr(env, "workdir", None) or "")
    if not wd.is_dir():
        return False
    inv_path = wd / "inventory_row.json"
    if not inv_path.is_file():
        return False
    try:
        inv = json.loads(inv_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    route_id = str(handles.get("route_id") or "").strip()
    if not route_id or not _registry_has_successful_execution(wd, route_id):
        return False

    from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import gold_capability_id

    gold = gold_capability_id(inv, wd)
    pick = _cap_from_route_id(route_id)
    if not gold or not pick or pick != gold:
        return False

    from hazardweaver.hwa.runtime.unified_final_artifact_align_v1 import align_unified_final_artifact_if_needed
    from hazardweaver.hwb.bridge.hwa_workdir import load_hwa_submission, load_taskpack_for_submission
    from hazardweaver.hwb.evaluators.dca_scorer import evaluate_dca_submission

    align_unified_final_artifact_if_needed(wd, inv)
    submission = load_hwa_submission(wd)
    taskpack = load_taskpack_for_submission(submission)
    tier = str(inv.get("difficulty_tier") or "L1")
    dca = evaluate_dca_submission(
        taskpack,
        submission,
        agent_id="hazardweaver",
        difficulty_tier=tier,
        inventory_row=inv,
        workdir=wd,
    )
    from hazardweaver.hwa.benchmark.dca_mechanism_unify_v1 import apply_mechanism_veto_to_dca
    from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import (
        apply_unified_route_gate_to_dca,
        infer_dca_route_condition,
    )

    dca_dict = dca.to_dict()
    if inv.get("instance_id"):
        dca_dict = apply_mechanism_veto_to_dca(dca_dict, wd)
    dca_dict = apply_unified_route_gate_to_dca(
        dca_dict,
        wd,
        inventory_row=inv,
        condition=infer_dca_route_condition(wd),
    )
    if not dca_dict.get("valid"):
        return False

    payload = {
        "action": "solve",
        "route_id": route_id,
        "execution_id": str(handles.get("execution_id") or "").strip(),
        "final_artifact_id": str(handles.get("final_artifact_id") or "").strip(),
    }
    answer = {
        "task_id": getattr(env, "task_id", None),
        "answer": payload,
        "rationale": "ablation_terminal_honest_registry_dca_v1",
        "emit_source": "ablation_manual_pilot_honest_registry_terminal_v1",
        "terminal_policy": "honest_registry_dca_v1",
    }
    (wd / "answer.json").write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")
    (wd / "dca_result.json").write_text(json.dumps(dca_dict, indent=2) + "\n", encoding="utf-8")
    env.submitted = True
    if hasattr(env, "append_transcript"):
        env.append_transcript(
            {
                "role": "system",
                "event": "ablation_manual_pilot_honest_registry_terminal",
                "route_id": route_id,
                "execution_id": payload["execution_id"],
                "final_artifact_id": payload["final_artifact_id"],
            }
        )
    return True


def force_ablation_terminal_invalid_submit(
    env: Any,
    task: Mapping[str, Any],
    handles: Mapping[str, str],
    failure: Mapping[str, Any],
    *,
    emit_source: str = "ablation_manual_pilot_terminal_v1",
    allow_blocked_abstention: bool = False,
) -> bool:
    """Write solve terminal + valid=false DCA; mark episode submitted."""
    if not solve_mandatory_episode_policy_enabled(task):
        return False
    if not all(str(handles.get(k) or "").strip() for k in ("route_id", "execution_id", "final_artifact_id")):
        return False

    workdir = Path(getattr(env, "workdir", None) or "")
    if not workdir.is_dir():
        return False

    route_id = str(handles.get("route_id") or "").strip()
    if not _registry_has_successful_execution(workdir, route_id):
        return False

    if try_honest_registry_terminal_dca(env, task, handles, workdir=workdir):
        return True

    err = str(failure.get("error") or extract_submit_error(failure) or "dca_preflight_failed")
    allowed = is_terminal_dca_failure(err) or (
        allow_blocked_abstention and err in _BLOCKED_ABSTENTION_ERRORS
    )
    if not allowed:
        return False

    payload = {
        "action": "solve",
        "route_id": str(handles["route_id"]).strip(),
        "execution_id": str(handles["execution_id"]).strip(),
        "final_artifact_id": str(handles["final_artifact_id"]).strip(),
    }
    answer = {
        "task_id": getattr(env, "task_id", None),
        "answer": payload,
        "rationale": f"ablation_terminal_invalid:{err}",
        "emit_source": emit_source,
        "terminal_policy": "ablation_manual_pilot_invalid_dca_v1",
        "terminal_failure": err,
    }
    answer_path = workdir / "answer.json"
    answer_path.write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")

    dca_dict = build_invalid_terminal_dca(task, failure=failure, handles=handles)
    (workdir / "dca_result.json").write_text(json.dumps(dca_dict, indent=2) + "\n", encoding="utf-8")

    env.submitted = True
    if hasattr(env, "append_transcript"):
        env.append_transcript(
            {
                "role": "system",
                "event": "ablation_manual_pilot_terminal_invalid",
                "error": err,
                "route_id": handles["route_id"],
                "execution_id": handles["execution_id"],
                "final_artifact_id": handles["final_artifact_id"],
            }
        )
    return True


def maybe_terminalize_failed_submit(
    env: Any,
    task: Mapping[str, Any],
    submit_result: Mapping[str, Any],
    handles: Optional[Mapping[str, str]] = None,
) -> bool:
    """On ablation pilot DCA/metric failure, end episode with valid=false instead of spinning."""
    if not solve_mandatory_episode_policy_enabled(task):
        return False
    if getattr(env, "submitted", False):
        return True
    err = extract_submit_error(submit_result)
    if not is_terminal_dca_failure(err):
        return False
    h = dict(handles or {})
    if not all(h.get(k) for k in ("route_id", "execution_id", "final_artifact_id")):
        h = recover_submit_handles_from_workdir(Path(getattr(env, "workdir", ".")))
    return force_ablation_terminal_invalid_submit(env, task, h, submit_result)


def _answer_action(workdir: Path) -> str:
    path = workdir / "answer.json"
    if not path.is_file():
        return ""
    try:
        ans = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    body = ans.get("answer") or {}
    return str(body.get("action") or "").strip()


def force_ablation_terminal_routing_record(
    env: Any,
    task: Mapping[str, Any],
    route_id: str,
    failure: Mapping[str, Any],
    *,
    emit_source: str = "ablation_manual_pilot_routing_terminal_v1",
) -> bool:
    """Terminalize committed route choice when execution/submit chain is incomplete."""
    if not solve_mandatory_episode_policy_enabled(task):
        return False
    rid = str(route_id or "").strip()
    if not rid:
        return False
    workdir = Path(getattr(env, "workdir", None) or "")
    if not workdir.is_dir():
        return False
    err = str(failure.get("error") or failure.get("exit_reason") or "limit_steps_unsubmitted_commit")
    answer = {
        "task_id": getattr(env, "task_id", None),
        "answer": {"action": "solve", "route_id": rid},
        "rationale": f"ablation_terminal_routing:{err}",
        "emit_source": emit_source,
        "terminal_policy": "ablation_manual_pilot_invalid_dca_v1",
        "terminal_failure": err,
    }
    (workdir / "answer.json").write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")
    handles = {"route_id": rid, "execution_id": "", "final_artifact_id": ""}
    dca_dict = build_invalid_terminal_dca(task, failure=failure, handles=handles)
    dca_dict["route_id"] = rid
    (workdir / "dca_result.json").write_text(json.dumps(dca_dict, indent=2) + "\n", encoding="utf-8")
    # Routing-only terminal records mechanism choice but is NOT a valid solve submission.
    env.submitted = False
    if hasattr(env, "append_transcript"):
        env.append_transcript(
            {
                "role": "system",
                "event": "ablation_manual_pilot_terminal_routing",
                "error": err,
                "route_id": rid,
            }
        )
    return True


def force_ablation_terminal_pre_controller_spin(
    env: Any,
    task: Mapping[str, Any],
    failure: Mapping[str, Any],
    *,
    emit_source: str = "unified_pre_controller_list_inventory_spin_v1",
) -> bool:
    """Terminal invalid solve when agent never reached controller_enumerate_routes."""
    if not solve_mandatory_episode_policy_enabled(task):
        return False
    if getattr(env, "submitted", False):
        return True
    workdir = Path(getattr(env, "workdir", None) or "")
    if not workdir.is_dir():
        return False
    err = str(failure.get("error") or "list_inventory_spin").strip()
    if err not in TERMINAL_DCA_FAILURE_ERRORS:
        return False
    answer = {
        "task_id": getattr(env, "task_id", None),
        "answer": {"action": "solve", "route_id": ""},
        "rationale": f"ablation_terminal_pre_controller:{err}",
        "emit_source": emit_source,
        "terminal_policy": "ablation_manual_pilot_invalid_dca_v1",
        "terminal_failure": err,
    }
    (workdir / "answer.json").write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")
    handles = {"route_id": "", "execution_id": "", "final_artifact_id": ""}
    dca_dict = build_invalid_terminal_dca(task, failure=failure, handles=handles)
    (workdir / "dca_result.json").write_text(json.dumps(dca_dict, indent=2) + "\n", encoding="utf-8")
    env.submitted = True
    if hasattr(env, "append_transcript"):
        env.append_transcript(
            {
                "role": "system",
                "event": "ablation_manual_pilot_terminal_pre_controller_spin",
                "error": err,
                "list_inventory_streak": failure.get("list_inventory_streak"),
                "step": failure.get("step"),
            }
        )
    return True


def maybe_terminalize_unsubmitted_spin(
    env: Any,
    task: Mapping[str, Any],
    exit_reason: str,
) -> bool:
    """After enumerate/clarify spin, end episode with invalid solve instead of bare limit_steps."""
    if not solve_mandatory_episode_policy_enabled(task):
        return False
    if getattr(env, "submitted", False):
        return True
    reason = str(exit_reason or "").strip()
    workdir = Path(getattr(env, "workdir", None) or "")
    clarify = _answer_action(workdir) == "clarify"
    if reason not in _SPIN_EXIT_REASONS and not clarify:
        return False
    handles = recover_submit_handles_from_workdir(workdir)
    err = "clarify_spin_limit_steps" if clarify else "limit_steps_unsubmitted_commit"
    failure = {"error": err, "exit_reason": reason}
    if all(str(handles.get(k) or "").strip() for k in ("route_id", "execution_id", "final_artifact_id")):
        return force_ablation_terminal_invalid_submit(
            env,
            task,
            handles,
            failure,
            emit_source="ablation_manual_pilot_spin_terminal_v1",
        )
    route_id = str(handles.get("route_id") or _committed_route_id(workdir)).strip()
    if route_id:
        return force_ablation_terminal_routing_record(
            env,
            task,
            route_id,
            failure,
            emit_source="ablation_manual_pilot_spin_terminal_v1",
        )
    return False


def _controller_admissible_route_count(env: Any) -> Optional[int]:
    ctrl = getattr(env, "controller", None)
    if ctrl is None or not hasattr(ctrl, "enumerate_routes"):
        return None
    try:
        enum = ctrl.enumerate_routes(admissible_only=True)
        return int(enum.get("n_routes") or 0)
    except Exception:  # noqa: BLE001
        return None


def maybe_terminalize_no_admissible_routes_spin(
    env: Any,
    task: Mapping[str, Any],
    abstention_result: Mapping[str, Any],
) -> bool:
    """Stop solve-mandatory NO_LEGAL_ROUTE / blocked-abstention spin when Π_adm is empty."""
    if not solve_mandatory_episode_policy_enabled(task):
        return False
    if getattr(env, "submitted", False):
        return True
    err = extract_submit_error(abstention_result)
    reason = str(
        abstention_result.get("reason_code")
        or (abstention_result.get("arguments") or {}).get("reason_code")
        or ""
    ).strip()
    if reason != "NO_LEGAL_ROUTE" and err not in _BLOCKED_ABSTENTION_ERRORS:
        return False
    n_adm = _controller_admissible_route_count(env)
    if n_adm is None or n_adm > 0:
        return False
    workdir = Path(getattr(env, "workdir", None) or "")
    handles = recover_submit_handles_from_workdir(workdir)
    failure = {"error": "no_admissible_routes_spin", "reason_code": reason or "NO_LEGAL_ROUTE"}
    if all(str(handles.get(k) or "").strip() for k in ("route_id", "execution_id", "final_artifact_id")):
        return force_ablation_terminal_invalid_submit(
            env,
            task,
            handles,
            failure,
            emit_source="unified_no_admissible_routes_spin_v1",
            allow_blocked_abstention=True,
        )
    route_id = str(handles.get("route_id") or _committed_route_id(workdir)).strip()
    if route_id:
        return force_ablation_terminal_routing_record(
            env,
            task,
            route_id,
            failure,
            emit_source="unified_no_admissible_routes_spin_v1",
        )
    return False


def maybe_terminalize_blocked_abstention(
    env: Any,
    task: Mapping[str, Any],
    abstention_result: Mapping[str, Any],
) -> bool:
    """After commit, blocked abstention → terminal invalid solve (no limit_steps spin)."""
    if not solve_mandatory_episode_policy_enabled(task):
        return False
    if getattr(env, "submitted", False):
        return True
    err = extract_submit_error(abstention_result)
    if err not in _BLOCKED_ABSTENTION_ERRORS:
        return False
    from hazardweaver.hwa.runtime.abstention_gate import is_committed_execution_state

    if not is_committed_execution_state(env):
        return False
    handles = recover_submit_handles_from_workdir(Path(getattr(env, "workdir", ".")))
    if not all(handles.values()):
        return False
    failure = {
        "error": err or "post_commit_abstain_blocked",
        "verify_layer": "abstention_gate",
        "tool": "submit_abstention",
    }
    return force_ablation_terminal_invalid_submit(
        env,
        task,
        handles,
        failure,
        emit_source="ablation_manual_pilot_blocked_abstention_v1",
        allow_blocked_abstention=True,
    )
