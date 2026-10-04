"""G1 unified HWA tool semantics (IMPL_SEAL §12.1 / plan A + P0-1/P0-3).

Core tools:
  inspect_artifact / run_capability / ask_user
  submit_solution / submit_clarification / submit_abstention  (typed terminals)
  submit / submit_answer  (legacy aliases → normalize into action payload)

Host AgentTools (wildfire / pfdf) register names and inject workdir side effects.
Legacy ``run_predictor`` / ``run_*_predictor`` / ``submit_answer`` remain aliases.
``submit`` ↔ ``submit_answer`` (bidirectional).

``run_capability`` dispatch order (inference-only; never trains):
  1. HWA CapabilityLoader registry id
  2. WildfirePredictorRegistry model_id
  2.5. FL-2 scientific infer (CAP-FL2-01 M1)
  3. G2 train_runs/<anchor>/<cap>/ artifact replay (metrics; no compile_one)

P0-3: all solver-visible tool returns pass planner/executor projection (no test_*).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hazardweaver.hwa.agent_runtime.observation_projection import (
    project_executor_view,
    project_planner_view,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_RUNS_ROOT = PROJECT_ROOT / "docs" / "engineering" / "hcg" / "train_runs"
CERT_DIR = PROJECT_ROOT / "docs" / "engineering" / "hcg" / "anchor_certificates"
RUNTIME_PROFILES_PATH = (
    PROJECT_ROOT / "docs" / "engineering" / "hcg" / "runtime_profiles_v1.yaml"
)
DEFAULT_RUNTIME_PROFILE_ID = "hw_tabular_cpu"

TOOL_NAME_ALIASES = {
    "submit": "submit_answer",
    "clarify_slot": "ask_user",
}

ABSTAIN_REASON_CODES = frozenset(
    {
        "NO_LEGAL_ROUTE",
        "NO_ADMISSIBLE_ROUTE",
        "VERIFY_FAILED",
        "UNSUPPORTED_REGION",
        "UNSUPPORTED_TEMPORAL_SCALE",
        "MISSING_REQUIRED_ARTIFACT",
        "INCOMPATIBLE_SCHEMA",
        "INCOMPATIBLE_LABEL",
        "BUDGET_OR_RUNTIME_EXCEEDED",
    }
)

TYPED_TERMINAL_TOOLS = frozenset(
    {"submit_solution", "submit_clarification", "submit_abstention"}
)

_CONTROLLER_GATE: Any = None


def set_controller_gate(controller: Any) -> None:
    """Bind active ScientificController for run_capability token enforcement."""
    global _CONTROLLER_GATE
    _CONTROLLER_GATE = controller


def clear_controller_gate() -> None:
    global _CONTROLLER_GATE
    _CONTROLLER_GATE = None


def normalize_tool_name(name: str) -> str:
    """Map G1 / legacy aliases onto registered handler names."""
    n = str(name or "").strip()
    return TOOL_NAME_ALIASES.get(n, n)


@lru_cache(maxsize=1)
def _load_runtime_profiles() -> Dict[str, Any]:
    if not RUNTIME_PROFILES_PATH.is_file():
        return {
            "schema_version": "1.0.0",
            "default_runtime_profile_id": DEFAULT_RUNTIME_PROFILE_ID,
            "profiles": {
                DEFAULT_RUNTIME_PROFILE_ID: {
                    "runtime_profile_id": DEFAULT_RUNTIME_PROFILE_ID,
                    "allow_gpu": False,
                }
            },
            "capability_bindings": {},
            "hw_registry_bindings": {},
        }
    try:
        import yaml  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"PyYAML required to load RuntimeProfile: {exc}") from exc
    doc = yaml.safe_load(RUNTIME_PROFILES_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise RuntimeError("runtime_profiles_v1.yaml must be a mapping")
    return doc


def resolve_runtime_profile(
    capability_id: str,
    *,
    runtime_profile_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Resolve RuntimeProfile for a capability.

    Raises ValueError if an explicit unknown profile id is requested.
    Unbound capability_id falls back to default ``hw_tabular_cpu``.
    """
    doc = _load_runtime_profiles()
    profiles = doc.get("profiles") or {}
    default_id = str(doc.get("default_runtime_profile_id") or DEFAULT_RUNTIME_PROFILE_ID)
    bindings = dict(doc.get("capability_bindings") or {})
    bindings.update(doc.get("hw_registry_bindings") or {})

    if runtime_profile_id is not None and str(runtime_profile_id).strip():
        pid = str(runtime_profile_id).strip()
        if pid not in profiles:
            known = sorted(profiles.keys())
            raise ValueError(
                f"unknown runtime_profile_id={pid!r}; known={known}"
            )
        return pid, dict(profiles[pid] or {})

    cid = str(capability_id or "").strip()
    pid = str(bindings.get(cid) or default_id)
    if pid not in profiles:
        raise ValueError(
            f"runtime_profile_id={pid!r} bound for capability_id={cid!r} "
            f"is missing from profiles"
        )
    return pid, dict(profiles[pid] or {})


def ask_user(
    question: str,
    slots: Sequence[str],
    *,
    host: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Passthrough to host.ask_user (clarify_dir / reply-file semantics unchanged)."""
    return host.ask_user(question, list(slots), **kwargs)


def _missing_fields_error(tool: str, missing: Sequence[str]) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": "missing_required_field",
        "missing_fields": list(missing),
        "tool": tool,
        "allowed_terminal_tools": sorted(TYPED_TERMINAL_TOOLS),
    }


def normalize_legacy_answer(answer: Any) -> Tuple[Any, int]:
    """Coerce stringified JSON object/array once (P0-2). Returns (value, repair_count)."""
    repairs = 0
    if isinstance(answer, str):
        s = answer.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return answer, 0
            if isinstance(parsed, (dict, list)):
                return parsed, 1
    return answer, repairs


def submit(
    answer: Any,
    *,
    host: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Canonical ``submit`` → host.submit_answer (legacy; prefer typed terminals)."""
    coerced, repairs = normalize_legacy_answer(answer)
    if isinstance(coerced, Mapping):
        from hazardweaver.hwa.runtime.terminal_action import coerce_answer_body

        coerced = coerce_answer_body(coerced)
    result = host.submit_answer(coerced, **kwargs)
    if repairs and isinstance(result, dict):
        result = dict(result)
        result["schema_repair_count"] = int(result.get("schema_repair_count") or 0) + repairs
    return result


def submit_answer(
    answer: Any,
    *,
    host: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias of ``submit`` (legacy name)."""
    return submit(answer, host=host, **kwargs)


def submit_solution(
    *,
    route_id: Any = None,
    execution_id: Any = None,
    final_artifact_id: Any = None,
    host: Any,
    rationale: Optional[str] = None,
    model_id_used: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """P0-1/P0-6 typed terminal: solve requires registry-backed execution_id."""
    from hazardweaver.hwa.agent_runtime.execution_schema import (
        resolve_workdir,
        verify_submit_solution_ids,
    )

    missing = [
        name
        for name, val in (
            ("route_id", route_id),
            ("execution_id", execution_id),
            ("final_artifact_id", final_artifact_id),
        )
        if val is None or str(val).strip() == ""
    ]
    if missing:
        return _missing_fields_error("submit_solution", missing)

    workdir = resolve_workdir(host, out_dir=kwargs.get("out_dir"))
    if workdir is None:
        return {
            "ok": False,
            "error": "missing_workdir_for_execution_registry",
            "tool": "submit_solution",
        }
    from hazardweaver.hwa.agent_runtime.tool_host import resolve_agent_host
    from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import (
        gate_submit_requires_run_capability,
        gate_unified_route_commit_before_submit,
    )

    task = getattr(resolve_agent_host(host), "task", None) or {}
    route_gate = gate_unified_route_commit_before_submit(task, str(route_id).strip())
    if route_gate is not None:
        return route_gate
    from hazardweaver.hwa.experiments.unified_gold_registry_submit_v1 import (
        _inventory_row,
        maybe_redirect_unified_submit_to_gold_registry,
    )

    inv = _inventory_row(Path(workdir), task)
    route_id, execution_id, final_artifact_id, _redirected = maybe_redirect_unified_submit_to_gold_registry(
        Path(workdir),
        inv,
        str(route_id).strip(),
        str(execution_id).strip(),
        str(final_artifact_id).strip(),
    )
    run_cap_gate = gate_submit_requires_run_capability(
        task, Path(workdir), str(route_id).strip()
    )
    if run_cap_gate is not None:
        return run_cap_gate
    answer_path = Path(workdir) / "answer.json"
    if answer_path.is_file():
        try:
            existing = json.loads(answer_path.read_text(encoding="utf-8"))
            ans = existing.get("answer") or {}
            if str(ans.get("action") or "") == "solve":
                return {
                    "ok": True,
                    "already_submitted": True,
                    "answer_path": str(answer_path),
                    "task_id": kwargs.get("task_id") or getattr(host, "task_id", None),
                    "answer": ans,
                    "tool": "submit_solution",
                    "message": "Task already submitted; do not call submit_solution again.",
                }
        except (OSError, json.JSONDecodeError):
            pass
    check = verify_submit_solution_ids(
        workdir=workdir,
        route_id=str(route_id).strip(),
        execution_id=str(execution_id).strip(),
        final_artifact_id=str(final_artifact_id).strip(),
    )
    if not check.get("ok"):
        return check
    resolved_artifact_id = str(
        check.get("final_artifact_id") or final_artifact_id
    ).strip()

    from hazardweaver.hwa.experiments.agent_strict_v2 import strict_headline_dca_verify

    dca = strict_headline_dca_verify(
        host,
        route_id=str(route_id).strip(),
        execution_id=str(execution_id).strip(),
        final_artifact_id=resolved_artifact_id,
        out_dir=kwargs.get("out_dir"),
    )
    if not dca.get("ok"):
        return {
            "ok": False,
            "error": str(dca.get("error") or "dca_preflight_failed"),
            "verify_layer": dca.get("verify_layer") or "dca_preflight",
            "tool": "submit_solution",
            "dca": dca,
        }

    payload = {
        "action": "solve",
        "route_id": str(route_id).strip(),
        "execution_id": str(execution_id).strip(),
        "final_artifact_id": resolved_artifact_id,
    }
    return host.submit_answer(
        payload,
        rationale=rationale,
        model_id_used=model_id_used,
        **kwargs,
    )


def submit_clarification(
    *,
    slot_id: Any = None,
    question: Any = None,
    host: Any,
    rationale: Optional[str] = None,
    model_id_used: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """P0-1 typed terminal: clarify decision → answer.action=clarify."""
    from hazardweaver.hwa.agent_runtime.tool_host import resolve_agent_host
    from hazardweaver.hwa.experiments.agent_strict_v2 import react_solve_clarify_forbidden

    task = getattr(resolve_agent_host(host), "task", None) or {}
    if react_solve_clarify_forbidden(task):
        return {
            "ok": False,
            "error": "react_clarify_forbidden",
            "message": (
                "ReAct solve ablation: submit_clarification is blocked; "
                "run_capability on allowed_edge_ids then submit_solution."
            ),
            "tool": "submit_clarification",
        }
    missing = [
        name
        for name, val in (("slot_id", slot_id), ("question", question))
        if val is None or str(val).strip() == ""
    ]
    if missing:
        return _missing_fields_error("submit_clarification", missing)
    payload = {
        "action": "clarify",
        "slot_id": str(slot_id).strip(),
        "question": str(question).strip(),
        "clarify_slot": str(slot_id).strip(),
    }
    return host.submit_answer(
        payload,
        rationale=rationale,
        model_id_used=model_id_used,
        **kwargs,
    )


def submit_abstention(
    *,
    reason_code: Any = None,
    failed_contract_ids: Any = None,
    checked_route_ids: Any = None,
    host: Any,
    rationale: Optional[str] = None,
    model_id_used: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """P0-1 typed terminal: abstain → answer.action=abstain."""
    missing: List[str] = []
    if reason_code is None or str(reason_code).strip() == "":
        missing.append("reason_code")
    if failed_contract_ids is None:
        missing.append("failed_contract_ids")
    if checked_route_ids is None:
        missing.append("checked_route_ids")
    if missing:
        return _missing_fields_error("submit_abstention", missing)
    code = str(reason_code).strip()
    if code not in ABSTAIN_REASON_CODES:
        return {
            "ok": False,
            "error": "invalid_reason_code",
            "reason_code": code,
            "allowed_reason_codes": sorted(ABSTAIN_REASON_CODES),
            "tool": "submit_abstention",
        }
    if isinstance(failed_contract_ids, str):
        failed_contract_ids, _ = normalize_legacy_answer(failed_contract_ids)
    if isinstance(checked_route_ids, str):
        checked_route_ids, _ = normalize_legacy_answer(checked_route_ids)
    if not isinstance(failed_contract_ids, list) or not isinstance(checked_route_ids, list):
        return {
            "ok": False,
            "error": "invalid_field_type",
            "detail": "failed_contract_ids and checked_route_ids must be arrays",
            "tool": "submit_abstention",
        }
    from hazardweaver.hwa.agent_runtime.tool_host import resolve_agent_host
    from hazardweaver.hwa.runtime.abstention_gate import submit_abstention_allowed

    host = resolve_agent_host(host)
    routes: list = []
    ctrl = getattr(host, "controller", None)
    if ctrl is not None:
        try:
            enum = ctrl.enumerate_routes(admissible_only=False)
            routes = list(enum.get("routes") or [])
        except Exception:  # noqa: BLE001
            routes = []
    if not routes:
        task = getattr(host, "task", None) or {}
        try:
            from hazardweaver.hwa.runtime.react_headline_execute import is_react_headline_task

            if is_react_headline_task(task):
                from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import routes_from_allowed_edges

                routes = list(routes_from_allowed_edges(task))
        except Exception:  # noqa: BLE001
            routes = []
    gate = submit_abstention_allowed(host, code, routes)
    if not gate.get("allowed"):
        return {
            "ok": False,
            "error": gate.get("error", "abstention_not_allowed"),
            "message": gate.get("message"),
            "tool": "submit_abstention",
        }
    payload = {
        "action": "abstain",
        "reason_code": code,
        "failed_contract_ids": [str(x) for x in failed_contract_ids],
        "checked_route_ids": [str(x) for x in checked_route_ids],
        "abstain": True,
    }
    return host.submit_answer(
        payload,
        rationale=rationale,
        model_id_used=model_id_used,
        **kwargs,
    )


def inspect_artifact(
    artifact_id: str,
    *,
    host: Any,
    anchor_id: Optional[str] = None,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """Unified inspect: pack card / sample schema / HCG edge / train_run metrics."""
    aid = str(artifact_id or "").strip()
    if not aid:
        return {"ok": False, "error": "artifact_id required", "tool": "inspect_artifact"}

    from hazardweaver.hwa.agent_runtime.tool_host import resolve_agent_host
    from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import unified_inspect_artifact_blocked

    resolved_host = resolve_agent_host(host)
    task = getattr(resolved_host, "task", None) or {}
    prior_inspects = 0
    audit_path = getattr(resolved_host, "_tool_audit", None)
    if audit_path is not None:
        try:
            from pathlib import Path

            p = Path(audit_path)
            if p.is_file():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(row.get("tool") or "") == "inspect_artifact":
                        prior_inspects += 1
        except OSError:
            prior_inspects = 0
    blocked = unified_inspect_artifact_blocked(
        task, aid, prior_inspect_count=prior_inspects
    )
    if blocked is not None:
        return blocked

    sections: List[Dict[str, Any]] = []
    errors: List[str] = []

    # 1) Pack card path (relative)
    card = _try_read_card(host, aid)
    if card is not None:
        sections.append({"kind": "card", **card})

    # 2) Common card locations by id
    if not sections:
        for rel in (
            f"cards/models/{aid}.md",
            f"cards/datasets/{aid}.md",
            f"cards/tasks/{aid}.md",
            f"cards/{aid}.md",
        ):
            card = _try_read_card(host, rel)
            if card is not None:
                sections.append({"kind": "card", **card})
                break

    # 3) G2 / HCG train_run metrics summary
    metrics = _resolve_train_run_metrics(aid, anchor_id=anchor_id)
    if metrics is not None:
        sections.append({"kind": "metrics_summary", **metrics})

    # 4) Sample schema (dataset_id::sample_id or dataset_id/sample_id)
    sample = _try_sample_schema(host, aid, split=split)
    if sample is not None:
        sections.append({"kind": "sample_schema", **sample})

    # 5) HCG edge explanation
    if hasattr(host, "tool_hcg_explain_edge") and (
        aid.startswith("edge:") or "_to_" in aid or aid.endswith("_v0") or aid.endswith("_v1")
    ):
        edge_id = aid[5:] if aid.startswith("edge:") else aid
        try:
            detail = host.tool_hcg_explain_edge(edge_id)
            if detail.get("ok") is not False:
                sections.append({"kind": "hcg_edge", "edge_id": edge_id, "detail": detail})
        except Exception as exc:  # noqa: BLE001 — inspect is best-effort
            errors.append(f"hcg_edge: {type(exc).__name__}: {exc}")

    if not sections:
        return project_planner_view(
            {
                "ok": False,
                "artifact_id": aid,
                "error": "unresolved artifact_id",
                "hints": [
                    "card relative path under pack (e.g. cards/models/<id>.md)",
                    "capability_id with metrics under train_runs/<anchor>/<cap>/",
                    "dataset_id::sample_id for sample schema",
                    "edge:<edge_id> for HCG explain",
                ],
                "errors": errors,
                "tool": "inspect_artifact",
            }
        )

    return project_planner_view(
        {
            "ok": True,
            "artifact_id": aid,
            "n_sections": len(sections),
            "sections": sections,
            "errors": errors or None,
            "tool": "inspect_artifact",
        }
    )

def run_capability(
    capability_id: str,
    *,
    host: Any,
    record: Optional[Mapping[str, Any]] = None,
    record_id: Optional[str] = None,
    handles: Optional[Mapping[str, Any]] = None,
    artifact_refs: Optional[Sequence[str]] = None,
    dataset_id: Optional[str] = None,
    sample_id: Optional[str] = None,
    split: Optional[str] = None,
    anchor_id: Optional[str] = None,
    oracle: bool = False,
    runtime_profile: Optional[str] = None,
    runtime_profile_id: Optional[str] = None,
    out_dir: Any = None,
    route_id: Optional[str] = None,
    controller_token: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run one capability — inference / artifact replay only (never compile_one).

    P0-6: successful runs mint ExecutionResult into the task workdir registry.
    When a ScientificController gate is active, ``controller_token`` is required.
    """
    from hazardweaver.hwa.agent_runtime.execution_schema import mint_and_register_execution
    from hazardweaver.hwa.agent_runtime.tool_host import resolve_agent_host

    host = resolve_agent_host(host)
    cid = str(capability_id or "").strip()
    if not cid:
        return {"ok": False, "error": "capability_id required", "tool": "run_capability"}

    task = getattr(host, "task", None) or {}
    if isinstance(task, Mapping):
        from hazardweaver.hwa.experiments.rq4_route_intervention_v1 import maybe_rq4_execute_block

        blocked = maybe_rq4_execute_block("run_capability", {"capability_id": cid}, task)
        if blocked is not None:
            return project_executor_view(
                {
                    **blocked,
                    "tool": "run_capability",
                    "capability_id": cid,
                }
            )

    handles = dict(handles or {})
    from hazardweaver.hwa.contracts.route_execution import normalize_controller_token, resolve_route_id
    from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled
    from hazardweaver.hwa.agent_runtime.tool_args_sanitize_v1 import coerce_optional_str

    strict_v2_early = agent_strict_v2_enabled()
    if strict_v2_early:
        record_id = coerce_optional_str(record_id)
        sample_id = coerce_optional_str(sample_id)
        route_id = coerce_optional_str(route_id)
        controller_token = coerce_optional_str(controller_token)
        split = coerce_optional_str(split) if split is not None else None
        scenario_id = coerce_optional_str(
            handles.get("scenario_id") or kwargs.get("scenario_id")
        )
        meta = getattr(host, "task", {}) or {}
        meta = meta.get("metadata") if isinstance(meta.get("metadata"), Mapping) else {}
        hints = dict((meta or {}).get("strict_execution_hints") or {})
        if hints.get("record_id") and not record_id and not handles.get("record_id"):
            record_id = str(hints["record_id"])
        if hints.get("scenario_id") and not scenario_id and not handles.get("scenario_id"):
            scenario_id = str(hints["scenario_id"])
        if hints.get("split") and not split and not handles.get("split"):
            split = str(hints["split"])
        if record_id:
            handles.setdefault("record_id", record_id)
        if scenario_id:
            handles.setdefault("scenario_id", scenario_id)
        if split:
            handles.setdefault("split", split)

    controller_token = normalize_controller_token(
        controller_token=controller_token,
        handles=handles,
        kwargs=kwargs,
    )
    active_route_id: Optional[str] = None
    if _CONTROLLER_GATE is not None:
        state = getattr(_CONTROLLER_GATE, "state", None)
        if state is not None:
            active_route_id = str(getattr(state, "active_route_id", "") or "").strip() or None
    if not active_route_id and host is not None:
        host_ctrl = getattr(host, "controller", None)
        host_state = getattr(host_ctrl, "state", None) if host_ctrl is not None else None
        if host_state is not None:
            active_route_id = (
                str(getattr(host_state, "active_route_id", "") or "").strip() or None
            )
    rid, rid_err = resolve_route_id(
        route_id=route_id,
        handles=handles,
        capability_id=cid,
        active_route_id=active_route_id,
        strict_v2=strict_v2_early,
    )
    if rid_err:
        return project_executor_view({
            "ok": False,
            "error": rid_err,
            "capability_id": cid,
            "route_id": route_id,
            "tool": "run_capability",
            "message": (
                "Agent-Strict v2 requires route_id matching controller_commit_route "
                "(e.g. route:cap:<capability_id>)."
            ),
        })
    route_id = rid

    if strict_v2_early and not controller_token and _CONTROLLER_GATE is not None:
        pending_fn = getattr(_CONTROLLER_GATE, "pending_unused_token", None)
        if callable(pending_fn):
            pending = pending_fn(str(route_id))
            if pending:
                controller_token = pending

    if _CONTROLLER_GATE is not None:
        reject_reason = None
        if hasattr(_CONTROLLER_GATE, "execution_token_rejection_reason"):
            reject_reason = _CONTROLLER_GATE.execution_token_rejection_reason(
                controller_token, str(route_id)
            )
        if reject_reason in {
            "controller_token_unknown",
            "controller_token_route_mismatch",
            "controller_token_expired",
        }:
            pending_fn = getattr(_CONTROLLER_GATE, "pending_unused_token", None)
            if callable(pending_fn):
                pending = pending_fn(str(route_id))
                if pending:
                    controller_token = pending
                    reject_reason = _CONTROLLER_GATE.execution_token_rejection_reason(
                        controller_token, str(route_id)
                    )
        if reject_reason or not _CONTROLLER_GATE.validate_execution_token(
            controller_token, str(route_id)
        ):
            err = reject_reason or "controller_token_required"
            msg = (
                "Direct capability execution forbidden without controller commit token. "
                "Pass controller_token (same value as execution_token from commit)."
            )
            if err == "controller_token_already_consumed":
                msg = (
                    "controller_token already used for a prior run_capability call. "
                    "Call controller_commit_route again to mint a fresh execution_token."
                )
            return project_executor_view({
                "ok": False,
                "error": err,
                "capability_id": cid,
                "route_id": route_id,
                "tool": "run_capability",
                "message": msg,
            })

    artifact_refs = list(artifact_refs or [])
    # Common LLM aliases
    if record_id is None and handles.get("record_id") is not None:
        record_id = str(handles.get("record_id"))
    if sample_id is None and handles.get("sample_id") is not None:
        sample_id = str(handles.get("sample_id"))
    if dataset_id is None and handles.get("dataset_id") is not None:
        dataset_id = str(handles.get("dataset_id"))
    if anchor_id is None and handles.get("anchor_id") is not None:
        anchor_id = str(handles.get("anchor_id"))
    if record is None and isinstance(handles.get("record"), Mapping):
        record = handles.get("record")  # type: ignore[assignment]
    if runtime_profile_id is None and runtime_profile is not None:
        runtime_profile_id = runtime_profile
    if runtime_profile_id is None and handles.get("runtime_profile_id") is not None:
        runtime_profile_id = str(handles.get("runtime_profile_id"))
    if route_id is None and handles.get("route_id") is not None:
        route_id = str(handles.get("route_id"))
    if controller_token is None:
        controller_token = normalize_controller_token(handles=handles, kwargs=kwargs)

    try:
        profile_id, profile_doc = resolve_runtime_profile(
            cid, runtime_profile_id=runtime_profile_id
        )
    except ValueError as exc:
        return project_executor_view({
            "ok": False,
            "capability_id": cid,
            "error": str(exc),
            "tool": "run_capability",
            "trained_in_tool": False,
            "runtime_profile": None,
        })

    def _wrap(payload: Dict[str, Any], *, dispatch: str) -> Dict[str, Any]:
        ok = bool(payload.get("ok", True))
        if ok:
            raw_for_mint = payload.get("result") if isinstance(payload.get("result"), Mapping) else payload
            if isinstance(raw_for_mint, Mapping):
                extra: Dict[str, Any] = {}
                if "capability_id" not in raw_for_mint:
                    extra["capability_id"] = cid
                sid = handles.get("scenario_id") or sample_id or record_id
                if sid and "scenario_id" not in raw_for_mint:
                    extra["scenario_id"] = str(sid)
                sp = split or handles.get("split")
                if sp and "split" not in raw_for_mint:
                    extra["split"] = str(sp)
                raw_for_mint = {**dict(raw_for_mint), **extra}
            from hazardweaver.hwa.agent_runtime.execution_schema import mint_raw_result_for_registry

            minted = mint_and_register_execution(
                host=host,
                out_dir=out_dir or kwargs.get("out_dir"),
                raw_result=mint_raw_result_for_registry(raw_for_mint),
                capability_ids=[cid],
                route_id=route_id or f"cap:{cid}",
                ok=True,
                provenance={
                    "tool": "run_capability",
                    "dispatch": dispatch,
                    "capability_id": cid,
                    "runtime_profile": profile_id,
                },
            )
            payload = {**payload, **minted}
            registry_rid = str(minted.get("route_id") or route_id or f"cap:{cid}")
            registry_eid = str(minted.get("execution_id") or "")
            registry_aid = str(minted.get("final_artifact_id") or "")
            if registry_eid and registry_aid:
                payload["submit_solution_args"] = {
                    "route_id": registry_rid,
                    "execution_id": registry_eid,
                    "final_artifact_id": registry_aid,
                    "hint": (
                        "Pass these registry ids to submit_solution exactly once. "
                        "Ignore nested execution_event.execution_id UUID values."
                    ),
                }
            from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

            if agent_strict_v2_enabled():
                for drop_key in (
                    "execution_event",
                    "execution_certificate",
                    "reachability_certificate",
                ):
                    payload.pop(drop_key, None)
                inner = payload.get("result")
                if isinstance(inner, dict):
                    inner = dict(inner)
                    inner.pop("execution_event", None)
                    payload["result"] = inner
            if dispatch != "hcg_execute_capability":
                try:
                    from hazardweaver.hwa.agent_runtime.execution_schema import resolve_workdir
                    from hazardweaver.hwa.runtime.legacy_execution_certificate import (
                        record_legacy_run_capability_ledger,
                    )

                    wd = resolve_workdir(
                        host,
                        out_dir=out_dir or kwargs.get("out_dir"),
                    )
                    if wd is not None:
                        lease_id = ""
                        lease = getattr(host, "active_lease", None)
                        if isinstance(lease, Mapping):
                            lease_id = str(lease.get("lease_id") or "")
                        record_legacy_run_capability_ledger(
                            wd,
                            capability_id=cid,
                            dispatch=dispatch,
                            payload=payload,
                            ok=True,
                            lease_id=lease_id,
                            route_id=str(route_id or payload.get("route_id") or f"cap:{cid}"),
                        )
                except Exception:  # noqa: BLE001
                    pass
        return project_executor_view(payload)

    def _bump_legacy_bypass() -> None:
        st = getattr(host, "session_state", None) or getattr(host, "state", None)
        if st is not None and hasattr(st, "legacy_bypass_count"):
            st.legacy_bypass_count += 1

    def _run_cap_fail(payload: Dict[str, Any]) -> Dict[str, Any]:
        from hazardweaver.hwa.runtime.execution_failure_class import attach_failure_class

        body = {**payload, "ok": False, "tool": payload.get("tool") or "run_capability"}
        return project_executor_view(attach_failure_class(body))

    from hazardweaver.hcg.registry.canonical_ids import is_headline_capability
    from hazardweaver.hcg.runtime.portfolio_probe_resolver import is_portfolio_pfdf_capability

    if is_headline_capability(cid):
        from hazardweaver.hcg.runtime.execute_capability import ExecutionLeaseError
        from hazardweaver.hwa.experiments.agent_strict_v2 import (
            agent_strict_v2_enabled,
            apply_strict_run_capability_handle_defaults,
            sanitize_strict_execution_handles,
            validate_agent_handles,
        )
        from hazardweaver.hwa.runtime.abstention_gate import pre_commit_before_run_capability
        from hazardweaver.hwa.runtime.hcg_execute_bridge import event_to_run_capability_result, run_headline_capability
        from hazardweaver.hwa.runtime.lease_manager import require_lease
        from hazardweaver.hwa.runtime.react_headline_execute import prepare_react_headline_execute
        from hazardweaver.hwa.route_controller.fl2_pilot_slice import FL2_SOLVER_OFFICIAL_CAPS

        strict_v2 = agent_strict_v2_enabled()

        if strict_v2:
            handles = sanitize_strict_execution_handles(handles)
            task = getattr(host, "task", None) or {}
            meta = task.get("metadata") or {}
            inv_row = meta.get("grader_hidden") or meta
            handles = apply_strict_run_capability_handle_defaults(
                cid,
                handles,
                inventory_row=inv_row if isinstance(inv_row, Mapping) else meta,
                task_metadata=meta if isinstance(meta, Mapping) else None,
            )
            from hazardweaver.hwa.experiments.ablation_manual_pilot_execution_v1 import (
                ablation_authoritative_execution_handles,
            )

            handles = ablation_authoritative_execution_handles(
                handles,
                inventory_row=inv_row if isinstance(inv_row, Mapping) else meta,
                task_metadata=meta if isinstance(meta, Mapping) else None,
            )
            oracle = False

        if is_portfolio_pfdf_capability(cid):
            rid = handles.get("record_id") or record_id or sample_id
            if not rid and not strict_v2:
                try:
                    from hazardweaver.hwa.experiments.headline_pfdf_record_id_v1 import resolve_headline_pfdf_record_id

                    task = getattr(host, "task", None) or {}
                    meta = task.get("metadata") or {}
                    rid = resolve_headline_pfdf_record_id(meta, task=task)
                except KeyError:
                    rid = None
            if rid:
                handles["record_id"] = str(rid)
                stub = handles.get("usgs_pfdf_record_v1")
                if not isinstance(stub, Mapping) or not stub.get("FireName"):
                    handles["usgs_pfdf_record_v1"] = {"record_id": str(rid)}
            if kwargs.get("burn_summary"):
                handles["burn_summary"] = dict(kwargs["burn_summary"])
            if not strict_v2:
                handles.setdefault("oracle", bool(oracle))

        if cid in FL2_SOLVER_OFFICIAL_CAPS or str(cid).startswith("CAP-TCTRK-") or str(cid).startswith("CAP-FL2-"):
            if not strict_v2:
                task = getattr(host, "task", None) or {}
                meta = task.get("metadata") or {}
                from hazardweaver.hwa.experiments.headline_scenario_binding_v1 import apply_headline_scenario_binding

                apply_headline_scenario_binding(handles, inventory_row=meta, task=task)

        if strict_v2:
            ok_handles, handle_errors, _prov = validate_agent_handles(cid, handles, provenance_src="agent")
            if not ok_handles:
                return _run_cap_fail({
                    "error": "agent_strict_v2_missing_handles",
                    "capability_id": cid,
                    "message": ";".join(handle_errors),
                    "parameter_provenance": _prov,
                })

        lease = require_lease(host)
        ctrl = getattr(host, "controller", None)
        routes: list = []
        committed = False
        if ctrl is not None:
            try:
                enum = ctrl.enumerate_routes(admissible_only=False)
                routes = list(enum.get("routes") or [])
            except Exception:  # noqa: BLE001
                routes = []
            state = getattr(ctrl, "state", None)
            if state is not None and str(getattr(state, "active_route_id", "") or "").strip():
                committed = True
        elif lease is not None and str(getattr(lease, "route_id", "") or "").strip():
            committed = True
        elif not strict_v2:
            react_lease, react_routes, react_err = prepare_react_headline_execute(host, cid)
            if react_err:
                return _run_cap_fail(react_err)
            if react_lease is not None:
                lease = react_lease
                routes = react_routes
        elif strict_v2:
            from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_react_ablation_enabled
            from hazardweaver.hwa.runtime.react_headline_execute import is_react_headline_task

            task = getattr(host, "task", None) or {}
            if agent_strict_v2_react_ablation_enabled(task) and is_react_headline_task(task):
                react_lease, react_routes, react_err = prepare_react_headline_execute(host, cid)
                if react_err:
                    return _run_cap_fail(react_err)
                if react_lease is not None:
                    lease = react_lease
                    routes = react_routes
            else:
                return _run_cap_fail({
                    "error": "scoped_execution_lease_required",
                    "capability_id": cid,
                    "message": "Agent-Strict v2 requires controller commit lease (no react ephemeral lease).",
                })
        pre = pre_commit_before_run_capability(
            routes, has_lease=lease is not None, committed=committed
        )
        if not pre.get("ok"):
            return _run_cap_fail({
                "error": pre.get("error", "pre_commit_blocked"),
                "capability_id": cid,
                "message": pre.get("message"),
            })
        if lease is None:
            return _run_cap_fail({
                "error": "scoped_execution_lease_required",
                "capability_id": cid,
                "message": "Headline capabilities require a valid Scoped Execution Lease from controller commit.",
            })
        try:
            smoke = bool(handles.get("smoke_mode") or kwargs.get("smoke_mode"))
            if cid in FL2_SOLVER_OFFICIAL_CAPS and not kwargs.get("force_smoke"):
                smoke = False
            full_certs = (not smoke and not bool(kwargs.get("acap_dict_only"))) or cid in FL2_SOLVER_OFFICIAL_CAPS
            from hazardweaver.hwa.runtime.run_capability_wall_v1 import run_with_optional_wall

            if strict_v2:
                from hazardweaver.hwa.runtime.hwa_headline_infer_v2 import (
                    hwa_strict_result_to_run_capability,
                    run_hwa_strict_headline_infer,
                )

                bundle_or_event = run_with_optional_wall(
                    lambda: run_hwa_strict_headline_infer(
                        cid,
                        handles,
                        lease,
                        host=host,
                        execution_id=str(kwargs.get("execution_id") or ""),
                        smoke_mode=smoke,
                        oracle=bool(oracle),
                        burn_summary=kwargs.get("burn_summary"),
                        record_overrides=kwargs.get("record_overrides"),
                        out_dir=out_dir or kwargs.get("out_dir"),
                    ),
                    label="run_hwa_strict_headline_infer",
                )
            else:
                bundle_or_event = run_with_optional_wall(
                    lambda: run_headline_capability(
                        cid,
                        handles,
                        lease,
                        execution_id=str(kwargs.get("execution_id") or ""),
                        smoke_mode=smoke,
                        full_certificates=full_certs,
                    ),
                    label="run_headline_capability",
                )
        except TimeoutError as exc:
            return _run_cap_fail({
                "error": "run_capability_wall_timeout",
                "capability_id": cid,
                "message": str(exc),
            })
        except ExecutionLeaseError as exc:
            return _run_cap_fail({
                "error": str(exc),
                "capability_id": cid,
                "dispatch": "hcg_execute_capability" if not strict_v2 else "hwa_official_eval",
            })
        except ValueError as exc:
            return _run_cap_fail({
                "error": "agent_strict_v2_validation_failed",
                "capability_id": cid,
                "message": str(exc),
                "dispatch": "hwa_official_eval",
            })
        from hazardweaver.hcg.runtime.execute_capability import ExecuteCapabilityBundle

        if strict_v2 and isinstance(bundle_or_event, ExecuteCapabilityBundle):
            payload = hwa_strict_result_to_run_capability(bundle_or_event, capability_id=cid)
        elif isinstance(bundle_or_event, ExecuteCapabilityBundle):
            payload = event_to_run_capability_result(
                bundle_or_event.event,
                capability_id=cid,
                bundle=bundle_or_event,
            )
        else:
            payload = event_to_run_capability_result(bundle_or_event, capability_id=cid)
        dispatch_name = str(payload.get("dispatch") or ("hwa_official_eval" if strict_v2 else "hcg_execute_capability"))
        wrapped = _wrap(payload, dispatch=dispatch_name)
        if not wrapped.get("ok", True):
            from hazardweaver.hwa.runtime.execution_failure_class import attach_failure_class

            wrapped = attach_failure_class(dict(wrapped))
        try:
            from hazardweaver.hwa.agent_runtime.execution_schema import resolve_workdir
            from hazardweaver.hwa.runtime.trajectory_ledger import record_tool_execution

            wd = resolve_workdir(host, out_dir=out_dir or kwargs.get("out_dir"))
            if wd is not None:
                record_tool_execution(
                    wd,
                    tool="run_capability",
                    capability_id=cid,
                    ok=bool(wrapped.get("ok", True)),
                    execution_id=str(wrapped.get("execution_id") or payload.get("execution_id") or ""),
                    lease_id=str(wrapped.get("lease_id") or payload.get("lease_id") or ""),
                    route_id=str(lease.route_id or ""),
                    execution_certificate=wrapped.get("execution_certificate") or payload.get("execution_certificate"),
                    reachability_certificate=wrapped.get("reachability_certificate") or payload.get("reachability_certificate"),
                    execution_event=wrapped.get("execution_event") or payload.get("execution_event"),
                    error=str(wrapped.get("error") or ""),
                    parameter_provenance=wrapped.get("parameter_provenance") or payload.get("parameter_provenance"),
                    inference_backend=str(wrapped.get("dispatch") or payload.get("dispatch") or ""),
                    hcg_role="reachability_only" if strict_v2 else None,
                )
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning(
                "headline_run_capability_ledger_failed capability_id=%s error=%s",
                cid,
                exc,
            )
        if wrapped.get("ok", True):
            from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_react_ablation_enabled
            from hazardweaver.hwa.runtime.react_headline_execute import is_react_headline_task

            task = getattr(host, "task", None) or {}
            if agent_strict_v2_react_ablation_enabled(task) and is_react_headline_task(task):
                from hazardweaver.hwa.agent_runtime.tool_host import resolve_agent_host

                resolved = resolve_agent_host(host)
                resolved._react_headline_solve_ready = {
                    "capability_id": cid,
                    "route_id": str(getattr(lease, "route_id", "") or wrapped.get("route_id") or ""),
                    "execution_id": str(wrapped.get("execution_id") or ""),
                    "final_artifact_id": str(wrapped.get("final_artifact_id") or ""),
                }
        return wrapped

    from hazardweaver.hwa.runtime.runtime_closure import g6_committed_allowed_edge_dispatch, legacy_bypass_allowed

    g6_registry_dispatch = (
        _CONTROLLER_GATE is not None and g6_committed_allowed_edge_dispatch(host, cid)
    )
    if g6_registry_dispatch and not (record_id or sample_id):
        from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

        if not agent_strict_v2_enabled():
            try:
                from hazardweaver.hwa.experiments.headline_pfdf_record_id_v1 import resolve_headline_pfdf_record_id

                task = getattr(host, "task", None) or {}
                meta = task.get("metadata") or {}
                record_id = resolve_headline_pfdf_record_id(meta, task=task)
            except KeyError:
                pass
    if g6_registry_dispatch:
        hw = _try_hw_registry_predict(
            host,
            cid,
            record=record,
            record_id=record_id or sample_id,
            oracle=oracle,
            burn_summary=kwargs.get("burn_summary"),
            record_overrides=kwargs.get("record_overrides"),
        )
        if hw is not None:
            return _wrap(
                {
                    "ok": bool(hw.get("ok", True)),
                    "capability_id": cid,
                    "dispatch": "capability_loader",
                    "artifact_refs": artifact_refs,
                    "result": hw,
                    "tool": "run_capability",
                    "trained_in_tool": False,
                    "runtime_profile": profile_id,
                    "runtime_profile_doc": {
                        "allow_gpu": bool(profile_doc.get("allow_gpu", False)),
                        "allow_compile_one": bool(profile_doc.get("allow_compile_one", False)),
                    },
                },
                dispatch="capability_loader",
            )
        wf = _try_wildfire_predict(
            host,
            cid,
            dataset_id=dataset_id,
            sample_id=sample_id or record_id,
            split=split,
        )
        if wf is not None:
            return _wrap(
                {
                    "ok": bool(wf.get("ok", True)),
                    "capability_id": cid,
                    "dispatch": "wildfire_predictor_registry",
                    "artifact_refs": artifact_refs,
                    "result": wf,
                    "tool": "run_capability",
                    "trained_in_tool": False,
                    "runtime_profile": profile_id,
                },
                dispatch="wildfire_predictor_registry",
            )

    if _CONTROLLER_GATE is not None and not legacy_bypass_allowed(host) and not g6_registry_dispatch:
        return project_executor_view({
            "ok": False,
            "error": "legacy_bypass_forbidden_in_controller_mode",
            "capability_id": cid,
            "tool": "run_capability",
            "message": (
                "Non-headline capabilities cannot bypass the controller in the default "
                "ScientificRouteController path. Use controller_enumerate_routes → "
                "controller_propose_route → controller_commit_route, or set "
                "solver_visible.allow_legacy_bypass=true for explicit ReAct ablations."
            ),
        })

    _bump_legacy_bypass()

    # 1) HWA CapabilityLoader registry
    hw = _try_hw_registry_predict(
        host,
        cid,
        record=record,
        record_id=record_id or sample_id,
        oracle=oracle,
        **{k: v for k, v in kwargs.items() if k in {"burn_summary", "record_overrides"}},
    )
    if hw is not None:
        return _wrap(
            {
                "ok": bool(hw.get("ok", True)),
                "capability_id": cid,
                "dispatch": "capability_loader",
                "artifact_refs": artifact_refs,
                "result": hw,
                "tool": "run_capability",
                "trained_in_tool": False,
                "runtime_profile": profile_id,
                "runtime_profile_doc": {
                    "allow_gpu": bool(profile_doc.get("allow_gpu", False)),
                    "allow_compile_one": bool(profile_doc.get("allow_compile_one", False)),
                },
            },
            dispatch="capability_loader",
        )

    # 2) WildfirePredictorRegistry model_id
    wf = _try_wildfire_predict(
        host,
        cid,
        dataset_id=dataset_id,
        sample_id=sample_id or record_id,
        split=split,
    )
    if wf is not None:
        return _wrap(
            {
                "ok": True,
                "capability_id": cid,
                "dispatch": "wildfire_predictor_registry",
                "artifact_refs": artifact_refs,
                "result": wf,
                "tool": "run_capability",
                "trained_in_tool": False,
                "runtime_profile": profile_id,
                "runtime_profile_doc": {
                    "allow_gpu": bool(profile_doc.get("allow_gpu", False)),
                    "allow_compile_one": bool(profile_doc.get("allow_compile_one", False)),
                },
            },
            dispatch="wildfire_predictor_registry",
        )

    # 2.5) FL-2 scientific infer (M1: CAP-FL2-01 only; before train_runs metrics replay)
    fl2 = _try_fl2_scientific_infer(
        cid,
        handles=handles,
        out_dir=out_dir or kwargs.get("out_dir"),
        split=split,
    )
    if fl2 is not None:
        return _wrap(
            {
                "ok": bool(fl2.get("ok", False)),
                "capability_id": cid,
                "dispatch": "fl2_scientific_infer",
                "artifact_refs": artifact_refs,
                "result": fl2,
                "tool": "run_capability",
                "trained_in_tool": False,
                "runtime_profile": profile_id,
                "runtime_profile_doc": {
                    "allow_gpu": bool(profile_doc.get("allow_gpu", False)),
                    "allow_compile_one": bool(profile_doc.get("allow_compile_one", False)),
                },
            },
            dispatch="fl2_scientific_infer",
        )

    # 3) G2 train_runs inference-only artifact replay (NO compile_one / training)
    g2 = _try_g2_train_run_replay(cid, anchor_id=anchor_id, record=record, handles=handles)
    if g2 is not None:
        return _wrap(
            {
                "ok": bool(g2.get("ok", True)),
                "capability_id": cid,
                "dispatch": "g2_train_run",
                "artifact_refs": artifact_refs,
                "result": g2,
                "tool": "run_capability",
                "trained_in_tool": False,
                "runtime_profile": profile_id,
                "runtime_profile_doc": {
                    "allow_gpu": bool(profile_doc.get("allow_gpu", False)),
                    "allow_compile_one": bool(profile_doc.get("allow_compile_one", False)),
                },
                "note": "inference-only replay of frozen train_runs artifacts; compile_one forbidden",
            },
            dispatch="g2_train_run",
        )

    return project_executor_view({
        "ok": False,
        "capability_id": cid,
        "error": (
            f"unresolved capability_id={cid!r}: not in HWA registry, "
            "wildfire predictors, FL-2 scientific infer, or train_runs metrics"
        ),
        "tool": "run_capability",
        "trained_in_tool": False,
        "runtime_profile": profile_id,
    })


_INVALID_EXECUTION_ID_PLACEHOLDERS = frozenset(
    {"null", "none", "undefined", "nil", "n/a", "na"}
)


def get_execution_result(
    execution_id: str,
    *,
    host: Any = None,
    out_dir: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """P0-6: read a previously minted ExecutionResult from the workdir registry."""
    from hazardweaver.hwa.agent_runtime.execution_schema import load_execution, resolve_workdir

    eid = str(execution_id or "").strip()
    if not eid:
        return {"ok": False, "error": "execution_id required", "tool": "get_execution_result"}
    if eid.lower() in _INVALID_EXECUTION_ID_PLACEHOLDERS:
        return {
            "ok": False,
            "error": "invalid_execution_id",
            "execution_id": eid,
            "tool": "get_execution_result",
            "message": (
                "Do not pass placeholder execution_id values. Read execution_id from "
                "controller_commit_route.result.execution or result.execution_id."
            ),
        }
    workdir = resolve_workdir(host, out_dir=out_dir or kwargs.get("out_dir"))
    if workdir is None:
        return {
            "ok": False,
            "error": "missing_workdir_for_execution_registry",
            "tool": "get_execution_result",
        }
    er = load_execution(workdir, eid)
    if er is None:
        return {
            "ok": False,
            "error": "unknown_execution_id",
            "execution_id": eid,
            "tool": "get_execution_result",
        }
    return project_planner_view({"ok": True, "tool": "get_execution_result", "execution": er})


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _try_read_card(host: Any, card_path: str) -> Optional[Dict[str, Any]]:
    if not hasattr(host, "read_card"):
        return None
    try:
        out = host.read_card(card_path)
    except (ValueError, FileNotFoundError, OSError):
        return None
    text = out.get("text") or ""
    return {
        "card_path": out.get("card_path", card_path),
        "n_chars": len(text),
        "text_preview": text[:1200],
        "text": text if len(text) <= 4000 else text[:4000] + "\n…[truncated]",
    }


def _try_sample_schema(
    host: Any,
    artifact_id: str,
    *,
    split: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    ds: Optional[str] = None
    sid: Optional[str] = None
    if "::" in artifact_id:
        ds, sid = artifact_id.split("::", 1)
    elif artifact_id.count("/") == 1 and not artifact_id.startswith("cards/"):
        ds, sid = artifact_id.split("/", 1)
    if not ds or not sid:
        return None

    # Wildfire load_sample(dataset_id, sample_id)
    if hasattr(host, "load_sample") and hasattr(host, "predictors"):
        try:
            sample = host.load_sample(ds, sid, split=split)
        except Exception:  # noqa: BLE001
            return None
        return {
            "dataset_id": sample.get("dataset_id", ds),
            "sample_id": sample.get("sample_id", sid),
            "split": sample.get("split"),
            "feature_shape": sample.get("feature_shape"),
            "dtype": sample.get("dtype"),
            "task_family": sample.get("task_family"),
            "schema": sample.get("schema"),
        }

    # PFDF load_sample(record_id=...)
    if hasattr(host, "load_sample") and hasattr(host, "data"):
        try:
            sample = host.load_sample(sid if ds in {"pfdf", "usgs_pfdf"} else artifact_id)
        except TypeError:
            try:
                sample = host.load_sample(record_id=sid)
            except Exception:  # noqa: BLE001
                return None
        except Exception:  # noqa: BLE001
            return None
        schema_keys = sorted(
            k for k in sample.keys() if k not in {"features", "Volume_m3", "log_volume"}
        )
        return {
            "record_id": sample.get("record_id") or sid,
            "schema_keys": schema_keys[:80],
            "n_keys": len(schema_keys),
        }
    return None


def _resolve_train_run_metrics(
    capability_id: str,
    *,
    anchor_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    paths = _find_train_run_dirs(capability_id, anchor_id=anchor_id)
    if not paths:
        return None
    root, metrics_path = paths[0]
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    summary = {
        k: metrics.get(k)
        for k in (
            "anchor_id",
            "capability_id",
            "family",
            "task",
            "metric_name",
            "val_metric",
            "backend",
            "n_train",
            "n_val",
            "select_split",
        )
        if k in metrics
    }
    return {
        "train_run_dir": str(root.relative_to(PROJECT_ROOT)),
        "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
        "summary": summary,
        "feature_keys_n": len(metrics.get("feature_keys") or []),
    }


def _find_train_run_dirs(
    capability_id: str,
    *,
    anchor_id: Optional[str] = None,
) -> List[tuple[Path, Path]]:
    if not TRAIN_RUNS_ROOT.is_dir():
        return []
    found: List[tuple[Path, Path]] = []
    if anchor_id:
        root = TRAIN_RUNS_ROOT / anchor_id / capability_id
        mp = root / "metrics.json"
        if mp.is_file():
            found.append((root, mp))
            return found
    # Exact capability folder under any anchor
    for anchor_dir in sorted(TRAIN_RUNS_ROOT.iterdir()):
        if not anchor_dir.is_dir() or not anchor_dir.name.startswith("A_"):
            continue
        root = anchor_dir / capability_id
        mp = root / "metrics.json"
        if mp.is_file():
            found.append((root, mp))
    if found:
        return found
    # capability_id may itself be "anchor/cap"
    if "/" in capability_id:
        root = TRAIN_RUNS_ROOT / capability_id
        mp = root / "metrics.json"
        if mp.is_file():
            return [(root, mp)]
    return []


def _try_hw_registry_predict(
    host: Any,
    capability_id: str,
    *,
    record: Optional[Mapping[str, Any]],
    record_id: Optional[str],
    oracle: bool,
    burn_summary: Optional[Mapping[str, Any]] = None,
    record_overrides: Optional[Mapping[str, Any]] = None,
    allow_portfolio: bool = False,
) -> Optional[Dict[str, Any]]:
    try:
        from hazardweaver.hwa.capabilities.loader import CapabilityLoader
        from hazardweaver.hwa.capabilities.registry import CapabilityRegistry, default_registry_path
        from hazardweaver.hcg.runtime.portfolio_probe_resolver import is_portfolio_pfdf_capability
    except Exception:  # noqa: BLE001
        return None

    if is_portfolio_pfdf_capability(capability_id) and not allow_portfolio:
        return None

    try:
        reg = CapabilityRegistry.load_yaml(default_registry_path())
        if capability_id not in reg.list_ids():
            return None
    except Exception:  # noqa: BLE001
        return None

    # Prefer host's loader (device / require_checkpoint) when present
    loader = None
    if hasattr(host, "_capability_loader"):
        try:
            loader = host._capability_loader()
        except Exception:  # noqa: BLE001
            loader = None
    if loader is None:
        require_ckpt = getattr(host, "require_checkpoint", False)
        device = getattr(host, "device", "cpu")
        loader = CapabilityLoader(require_checkpoint=bool(require_ckpt), device=str(device))

    # Domain shortcuts already on PFDF host (oracle-safe)
    if capability_id == "burn_state_net_prithvi_v1" and hasattr(host, "run_burn_predictor"):
        rid = record_id
        if not rid:
            return {
                "ok": False,
                "error": "record_id required for burn_state_net_prithvi_v1",
                "capability_id": capability_id,
            }
        return host.run_burn_predictor(str(rid), oracle=bool(oracle))
    if capability_id == "pfdf_volume_gorr_v2" and hasattr(host, "run_volume_predictor"):
        rid = record_id
        if not rid:
            return {
                "ok": False,
                "error": "record_id required for pfdf_volume_gorr_v2",
                "capability_id": capability_id,
            }
        return host.run_volume_predictor(
            str(rid),
            burn_summary=dict(burn_summary) if burn_summary else None,
            record_overrides=dict(record_overrides) if record_overrides else None,
        )

    row: Optional[Dict[str, Any]] = dict(record) if record else None
    if row is None and record_id:
        if hasattr(host, "data"):
            try:
                row = dict(host.data.get_record(str(record_id)))
            except Exception as exc:  # noqa: BLE001
                return {
                    "ok": False,
                    "error": f"record load failed: {type(exc).__name__}: {exc}",
                    "capability_id": capability_id,
                }
        else:
            try:
                from hazardweaver.hwa.pfdf_agent.data_access import PfdfDataAccess

                row = dict(PfdfDataAccess().get_record(str(record_id)))
            except Exception as exc:  # noqa: BLE001
                return {
                    "ok": False,
                    "error": f"record load failed: {type(exc).__name__}: {exc}",
                    "capability_id": capability_id,
                }
    if row is None:
        return {
            "ok": False,
            "error": "record or record_id required for HWA CapabilityLoader predict",
            "capability_id": capability_id,
            "dispatch_candidate": "capability_loader",
        }
    if record_overrides:
        row.update(dict(record_overrides))
    if burn_summary:
        row["MeandNBR"] = burn_summary.get("mean_dnbr", row.get("MeandNBR"))
        row["FractionModHigh"] = burn_summary.get(
            "fraction_mod_high", row.get("FractionModHigh")
        )
        row["FractionBurned"] = burn_summary.get("fraction_burned", row.get("FractionBurned"))

    try:
        pred = loader.load(capability_id)
        out = pred.predict(row, oracle=bool(oracle)) if oracle else pred.predict(row)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "capability_id": capability_id,
        }
    return {
        "ok": bool(out.get("valid", True)) if isinstance(out, dict) else True,
        "capability_id": capability_id,
        "record_id": record_id,
        "raw": _compact_predict_out(out),
    }


def _try_wildfire_predict(
    host: Any,
    model_id: str,
    *,
    dataset_id: Optional[str],
    sample_id: Optional[str],
    split: Optional[str],
) -> Optional[Dict[str, Any]]:
    predictors = getattr(host, "predictors", None)
    if predictors is None:
        return None
    try:
        known = set(predictors.list_model_ids())
    except Exception:  # noqa: BLE001
        return None
    if model_id not in known:
        return None
    if not dataset_id or not sample_id:
        return {
            "ok": False,
            "error": "dataset_id and sample_id required for wildfire run_capability",
            "capability_id": model_id,
            "dispatch_candidate": "wildfire_predictor_registry",
        }
    if hasattr(host, "run_predictor"):
        return host.run_predictor(model_id, dataset_id, sample_id, split=split)
    # Fallback via registry directly
    sample = host.load_sample(dataset_id, sample_id, split=split)
    return predictors.predict(model_id, sample["features"], dataset_id=dataset_id)


FL2_SCIENTIFIC_CAPS = frozenset(
    {"CAP-FL2-01", "CAP-FL2-02", "CAP-FL2-03", "CAP-FL2-04", "CAP-FL2-05", "CAP-FL2-06"}
)


def _try_fl2_scientific_infer(
    capability_id: str,
    *,
    handles: Mapping[str, Any],
    out_dir: Any = None,
    split: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Route CAP-FL2-01~06 to official FloodCastBench / solver scientific infer."""
    cid = str(capability_id or "").strip()
    if cid not in FL2_SCIENTIFIC_CAPS:
        return None
    try:
        from hazardweaver.hcg.registry.canonical_ids import is_headline_capability
        from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

        if is_headline_capability(cid) and not agent_strict_v2_enabled():
            return None
    except Exception:  # noqa: BLE001
        pass
    from hazardweaver.hwa.agent_runtime.fl2_scientific_infer import run_fl2_capability_inference
    from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

    scenario_id = handles.get("scenario_id") or handles.get("sample_id")
    split_name = split or handles.get("split") or "official_test"
    return run_fl2_capability_inference(
        cid,
        scenario_id=str(scenario_id) if scenario_id else None,
        split=str(split_name),
        out_dir=Path(out_dir) if out_dir else None,
        strict_no_defaults=agent_strict_v2_enabled(),
    )


def _try_g2_train_run_replay(
    capability_id: str,
    *,
    anchor_id: Optional[str],
    record: Optional[Mapping[str, Any]],
    handles: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Replay frozen train_runs artifacts. Never calls compile_one / fit."""
    _ = record, handles  # reserved for future model.pkl / prediction_vector replay
    metrics_wrap = _resolve_train_run_metrics(capability_id, anchor_id=anchor_id)
    if metrics_wrap is None:
        # Try resolving anchor from certificates when capability appears in routes
        if anchor_id is None:
            anchor_id = _anchor_for_capability(capability_id)
            if anchor_id:
                metrics_wrap = _resolve_train_run_metrics(
                    capability_id, anchor_id=anchor_id
                )
    if metrics_wrap is None:
        return None

    summary = metrics_wrap.get("summary") or {}
    return {
        "ok": True,
        "mode": "frozen_metrics_replay",
        "anchor_id": summary.get("anchor_id") or anchor_id,
        "capability_id": summary.get("capability_id") or capability_id,
        "metrics": summary,
        "metrics_path": metrics_wrap.get("metrics_path"),
        "prediction": {
            "metric_name": summary.get("metric_name"),
            "reported_val_metric": summary.get("val_metric"),
            "note": (
                "G2 train_runs currently seal metrics.json only; "
                "this tool returns frozen validation metrics for callable replay — "
                "does not retrain (compile_one forbidden in-tool). "
                "Hidden test metrics are evaluator-only (P0-3)."
            ),
        },
    }


def _anchor_for_capability(capability_id: str) -> Optional[str]:
    if not CERT_DIR.is_dir():
        return None
    for path in sorted(CERT_DIR.glob("A_*.json")):
        try:
            cert = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for route in cert.get("routes") or []:
            if capability_id in (route.get("capability_ids") or []):
                return str(cert.get("anchor_id") or path.stem)
    return None


def _compact_predict_out(out: Any) -> Any:
    if not isinstance(out, Mapping):
        return out
    compact: Dict[str, Any] = {}
    for k, v in out.items():
        if k in {"logits", "pred_mask", "features"}:
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            compact[k] = v
        elif isinstance(v, Mapping):
            compact[k] = {
                sk: sv
                for sk, sv in v.items()
                if isinstance(sv, (str, int, float, bool)) or sv is None
            }
        elif isinstance(v, (list, tuple)) and len(v) <= 32:
            compact[k] = list(v)
    return compact


# Names for pack / CORE_TOOL_NAMES registration
UNIFIED_CORE_TOOL_NAMES = frozenset(
    {
        "inspect_artifact",
        "run_capability",
        "ask_user",
        "submit",
        "submit_answer",
        "submit_solution",
        "submit_clarification",
        "submit_abstention",
    }
)
