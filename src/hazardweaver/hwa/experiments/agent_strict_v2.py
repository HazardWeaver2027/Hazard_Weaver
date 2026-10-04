"""Agent-Strict v2 contract — no auto-bind, HWA inference, HCG graph-only."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

ProvenanceSource = str  # "agent" | "missing" | "auto"

PFDF_PORTFOLIO_CAPS = frozenset({"burn_state_net_prithvi_v1", "pfdf_volume_gorr_v2"})

# Required handle keys per capability family (agent must supply explicitly).
REQUIRED_HANDLES: Dict[str, Tuple[str, ...]] = {
    "burn_state_net_prithvi_v1": ("record_id",),
    "pfdf_volume_gorr_v2": ("record_id",),
}
for _fl2 in (
    "CAP-FL2-01",
    "CAP-FL2-02",
    "CAP-FL2-03",
    "CAP-FL2-04",
    "CAP-FL2-05",
    "CAP-FL2-06",
):
    REQUIRED_HANDLES[_fl2] = ("scenario_id", "split")
for _tctrk in (
    "CAP-TCTRK-01",
    "CAP-TCTRK-02",
    "CAP-TCTRK-03",
    "CAP-TCTRK-04",
    "CAP-TCTRK-05",
    "CAP-TCTRK-06",
):
    REQUIRED_HANDLES[_tctrk] = ("scenario_id",)

_DEFAULT_CAP_REQUIRED = ("scenario_id",)


def _strict_env() -> str:
    return str(os.environ.get("HWA_HEADLINE_AGENT_STRICT", "")).strip().lower()


def agent_strict_v2_enabled() -> bool:
    return _strict_env() in {"2", "v2", "agent_strict_v2"}


def agent_strict_v2_react_ablation_enabled(
    task: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Allow ReAct ephemeral lease under strict v2 for controller-off ablation only."""
    if not agent_strict_v2_enabled():
        return False
    if task is not None:
        from hazardweaver.hwa.runtime.react_headline_execute import is_react_headline_task

        if is_react_headline_task(task):
            return True
    return str(os.environ.get("HWA_AGENT_STRICT_REACT_ABLATION", "")).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def react_solve_clarify_forbidden(task: Mapping[str, Any]) -> bool:
    """ReAct solve ablation: clarify is not a valid terminal (must execute + submit)."""
    from hazardweaver.hwa.runtime.react_headline_execute import is_react_headline_task

    if not is_react_headline_task(task):
        return False
    meta = task.get("metadata") or {}
    return str(meta.get("expected_action") or "solve").strip().lower() != "clarify"


def strict_execution_hints_enabled() -> bool:
    """Instance-binding hints in user_facing_goal (protocol-level, not gold)."""
    if not agent_strict_v2_enabled():
        return True
    return str(os.environ.get("HWA_STRICT_EXECUTION_HINTS", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def agent_strict_v2_chained_commit_enabled() -> bool:
    """Single-step commit+execute when agent passes explicit handles on commit (SCC-v1)."""
    if not agent_strict_v2_enabled():
        return False
    return str(os.environ.get("HWA_STRICT_CHAINED_COMMIT", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def react_admissibility_gate_enabled() -> bool:
    """ReAct ablation: enforce Π_adm before ephemeral lease (ReAct-Π, R2)."""
    if not agent_strict_v2_react_ablation_enabled():
        return False
    return str(os.environ.get("HWA_REACT_ADMISSIBILITY_GATE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def extract_chained_commit_handles(args: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Handles for SCC: agent must supply record_id / scenario_id / split explicitly."""
    from hazardweaver.hwa.agent_runtime.tool_args_sanitize_v1 import is_placeholder_value

    handles = {
        k: v
        for k, v in dict(args.get("handles") or {}).items()
        if not is_placeholder_value(v)
    }
    for key in ("record_id", "scenario_id", "split", "sample_id", "anchor_id"):
        val = args.get(key)
        if not is_placeholder_value(val) and key not in handles:
            handles[key] = val
    return handles if handles else None


def agent_strict_v2_use_agent_loop() -> bool:
    """Strict v2: one LLM session drives controller + run_capability tools (no VCE bind sub-loop)."""
    return agent_strict_v2_enabled()


def headline_control_mode_label() -> str:
    """Run-meta control_mode string (strict v2 uses vce_agent_loop, not bare vce)."""
    from hazardweaver.hwa.control.mode import is_vce_mode

    if is_vce_mode() and agent_strict_v2_use_agent_loop():
        return "vce_agent_loop"
    if is_vce_mode():
        return "vce"
    return "legacy"


def agent_strict_v2_allows_run_capability() -> bool:
    """Agent may call run_capability directly after commit (no g6 auto-execute)."""
    return agent_strict_v2_enabled()


def agent_strict_v1_enabled() -> bool:
    return _strict_env() in {"1", "true", "yes", "strict"}


def agent_strict_any_enabled() -> bool:
    return agent_strict_v1_enabled() or agent_strict_v2_enabled()


def required_handle_keys(capability_id: str) -> Tuple[str, ...]:
    cid = str(capability_id or "").strip()
    if cid in REQUIRED_HANDLES:
        return REQUIRED_HANDLES[cid]
    if cid.startswith("CAP-FL2-"):
        return ("scenario_id", "split")
    if cid.startswith("CAP-TCTRK-"):
        return ("scenario_id",)
    if cid in PFDF_PORTFOLIO_CAPS:
        return ("record_id",)
    if cid.startswith("CAP-"):
        return _DEFAULT_CAP_REQUIRED
    return ()


def _handle_value(handles: Mapping[str, Any], key: str) -> str:
    from hazardweaver.hwa.agent_runtime.tool_args_sanitize_v1 import is_placeholder_value

    val = handles.get(key)
    if val is None and key == "scenario_id":
        val = handles.get("storm_sid") or handles.get("sample_id")
    if val is None and key == "record_id":
        stub = handles.get("usgs_pfdf_record_v1")
        if isinstance(stub, Mapping) and stub.get("record_id"):
            val = stub.get("record_id")
    if is_placeholder_value(val):
        return ""
    return str(val or "").strip()


def build_parameter_provenance(
    capability_id: str,
    handles: Mapping[str, Any],
    *,
    source: ProvenanceSource = "agent",
) -> Dict[str, ProvenanceSource]:
    out: Dict[str, ProvenanceSource] = {}
    for key in required_handle_keys(capability_id):
        present = bool(_handle_value(handles, key))
        if source == "agent":
            out[key] = "agent" if present else "missing"
        else:
            out[key] = source if present else "missing"
    return out


def validate_agent_handles(
    capability_id: str,
    handles: Mapping[str, Any],
    *,
    provenance_src: ProvenanceSource = "agent",
) -> Tuple[bool, List[str], Dict[str, ProvenanceSource]]:
    """Return (ok, errors, parameter_provenance)."""
    cid = str(capability_id or "").strip()
    required = required_handle_keys(cid)
    errors: List[str] = []
    prov = build_parameter_provenance(cid, handles, source=provenance_src)
    for key in required:
        if not _handle_value(handles, key):
            errors.append(f"missing_required_handle:{key}")
            prov[key] = "missing"
    if any(v == "auto" for v in prov.values()):
        errors.append("auto_bind_forbidden_in_agent_strict_v2")
    ok = not errors
    return ok, errors, prov


def strip_auto_bind_sources() -> bool:
    """True when runtime must not read metadata/solver_visible for param backfill."""
    return agent_strict_v2_enabled()


def apply_strict_solver_visible(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove scenario/split/record leaks from solver-visible surfaces (v2 only)."""
    if not agent_strict_v2_enabled():
        return dict(task)
    out = dict(task)
    sv = dict(out.get("solver_visible") or {})
    inputs = dict(sv.get("inputs") or {})
    params = dict(inputs.get("parameters") or {})
    params.pop("scenario_id", None)
    params.pop("split", None)
    params.pop("record_id", None)
    inputs["parameters"] = params
    inputs.pop("scenario_id", None)
    inputs.pop("split", None)
    sv["inputs"] = inputs
    out["solver_visible"] = sv

    meta = dict(out.get("metadata") or {})
    grader_hidden = dict(meta.get("grader_hidden") or {})
    for key in ("instance_id", "scenario_id", "pfdf_record_id", "internal_task_id"):
        if key in meta:
            grader_hidden[key] = meta.pop(key)
    if grader_hidden:
        meta["grader_hidden"] = grader_hidden
    out["metadata"] = meta
    return out


def apply_strict_vce_tools(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Expose run_capability to VCE solve pool under Agent-Strict v2."""
    if not agent_strict_v2_enabled():
        return dict(task)
    out = dict(task)
    sv = dict(out.get("solver_visible") or {})
    inv = dict((sv.get("allowed_inventory") or {}))
    tools = list(inv.get("tool_ids") or [])
    if "run_capability" not in tools:
        tools.append("run_capability")
    inv["tool_ids"] = tools
    sv["allowed_inventory"] = inv
    out["solver_visible"] = sv
    return out


def strict_instance_instructions(task: Mapping[str, Any]) -> str:
    if not agent_strict_v2_enabled():
        return ""
    track = str((task.get("metadata") or {}).get("track") or "")
    return (
        "Agent-Strict v2: gold scores and grader metadata are withheld. "
        "Target execution handles are stated in user_facing_goal — pass them verbatim "
        "to run_capability after controller_commit_route (with controller_token and route_id). "
        f"Track={track or 'unknown'}."
    )


def strict_inventory_row_from_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Grader inventory row for VERIFY/DCA (includes grader_hidden, not solver leaks)."""
    meta = dict(task.get("metadata") or {})
    hidden = dict(meta.get("grader_hidden") or {})
    return {
        "instance_id": str(hidden.get("instance_id") or meta.get("instance_id") or ""),
        "taskpack_id": str(meta.get("taskpack_id") or ""),
        "scenario_id": str(hidden.get("scenario_id") or meta.get("scenario_id") or ""),
        "track": str(meta.get("track") or meta.get("headline_target") or ""),
        "difficulty_tier": str(meta.get("difficulty_tier") or "L1"),
        "source": str(meta.get("source") or ""),
    }


def sanitize_strict_execution_handles(handles: Mapping[str, Any]) -> Dict[str, Any]:
    """Strict v2: forbid oracle shortcuts — real model inference only."""
    if not agent_strict_v2_enabled():
        return dict(handles)
    out = dict(handles)
    for key in ("oracle", "oracle_burn", "smoke_mode", "smoke_stub"):
        out.pop(key, None)
    return out


def apply_strict_run_capability_handle_defaults(
    capability_id: str,
    handles: Mapping[str, Any],
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
    task_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Bind missing scenario_id/split from instance/cap identity (oracle-strict policy).

    Does not read solver_visible parameters or gold fields. Taskpack template rows use
    ``CAP-PLACEHOLDER``; once the agent commits a parametric cap, ``scenario_id`` defaults
    to that ``capability_id``. ``param:`` / ``variant:`` rows default from the instance suffix.
    """
    if not agent_strict_v2_enabled():
        return dict(handles)
    from hazardweaver.hwa.agent_runtime.tool_args_sanitize_v1 import is_placeholder_value

    out = {
        k: v
        for k, v in dict(handles).items()
        if not is_placeholder_value(v)
    }
    inv = dict(inventory_row or {})
    meta = dict(task_metadata or {})
    hidden = dict(meta.get("grader_hidden") or {})
    instance_id = str(hidden.get("instance_id") or inv.get("instance_id") or "").strip()
    cid = str(capability_id or "").strip()

    if not _handle_value(out, "scenario_id"):
        scenario_id = str(hidden.get("scenario_id") or inv.get("scenario_id") or "").strip()
        if scenario_id and scenario_id != "CAP-PLACEHOLDER":
            out["scenario_id"] = scenario_id
        elif instance_id.startswith(("variant:", "param:")):
            suffix = instance_id.rsplit(":", 1)[-1].strip()
            if suffix:
                out["scenario_id"] = suffix
        elif instance_id.startswith("taskpack:") and cid.startswith("CAP-"):
            out["scenario_id"] = cid

    if not _handle_value(out, "split"):
        split = str(hidden.get("split") or inv.get("split") or "").strip()
        if split:
            out["split"] = split
        else:
            from hazardweaver.hwa.experiments.strict_solver_goal_v1 import resolve_strict_solver_execution_hints

            hints = resolve_strict_solver_execution_hints(inv)
            if hints.get("split"):
                out["split"] = str(hints["split"])

    return out


def strict_headline_dca_verify(
    host: Any,
    *,
    route_id: str,
    execution_id: str,
    final_artifact_id: str,
    out_dir: Any = None,
) -> Dict[str, Any]:
    """VCE-equivalent DCA preflight before submit_solution (strict headline only)."""
    if not agent_strict_v2_enabled():
        return {"ok": True, "verify_layer": "skipped"}
    task = getattr(host, "task", {}) or {}
    meta = task.get("metadata") or {}
    if not meta.get("hwb_headline_inventory"):
        return {"ok": True, "verify_layer": "skipped_non_headline"}
    from pathlib import Path

    from hazardweaver.hwa.agent_runtime.execution_schema import resolve_workdir
    from hazardweaver.hwa.control.vce_verify import vce_verify_execution

    workdir = resolve_workdir(host, out_dir=out_dir)
    if workdir is None:
        return {
            "ok": False,
            "error": "missing_workdir_for_dca_verify",
            "verify_layer": "dca_preflight",
            "tool": "submit_solution",
        }
    inv_row = strict_inventory_row_from_task(task)
    taskpack = None
    try:
        from hazardweaver.hwb.run.eval_dca_submission_v1 import resolve_taskpack_for_inventory_row

        taskpack = resolve_taskpack_for_inventory_row(inv_row)
    except Exception:  # noqa: BLE001
        taskpack = None
    return vce_verify_execution(
        workdir=Path(workdir),
        route_id=str(route_id).strip(),
        execution_id=str(execution_id).strip(),
        final_artifact_id=str(final_artifact_id).strip(),
        taskpack=taskpack,
        inventory_row=inv_row,
        difficulty_tier=str(inv_row.get("difficulty_tier") or "L1"),
    )


STRICT_FORBIDDEN_TERMINALS = frozenset({"submit", "submit_answer"})

ENGINEERING_ERROR_CODES = frozenset(
    {
        "controller_token_required",
        "route_id_required",
        "route_id_format_mismatch",
        "strict_terminal_required",
        "scoped_execution_lease_required",
        "direct_execution_forbidden",
    }
)


def strict_solver_tool_ids(tool_ids: Sequence[str]) -> List[str]:
    """Remove legacy submit terminals from strict v2 solver surface."""
    return [t for t in tool_ids if t not in STRICT_FORBIDDEN_TERMINALS]


def is_strict_forbidden_terminal(tool_name: str) -> bool:
    return str(tool_name or "").strip() in STRICT_FORBIDDEN_TERMINALS


def strict_terminal_block_payload(tool_name: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": "strict_terminal_required",
        "tool": tool_name,
        "message": (
            "Agent-Strict v2: use submit_solution or submit_abstention only "
            "(legacy submit_answer/submit are blocked)."
        ),
    }


def _parse_tool_payload(observation: Any) -> Dict[str, Any]:
    import json

    if isinstance(observation, Mapping):
        if "result" in observation and isinstance(observation.get("result"), Mapping):
            return dict(observation["result"])
        if observation.get("ok") is not None or observation.get("error"):
            return dict(observation)
        content = observation.get("content")
        if isinstance(content, str) and content.strip():
            try:
                parsed = json.loads(content)
                if isinstance(parsed, Mapping):
                    inner = parsed.get("result")
                    if isinstance(inner, Mapping):
                        return dict(inner)
                    return dict(parsed)
            except json.JSONDecodeError:
                pass
    if isinstance(observation, str) and observation.strip():
        try:
            parsed = json.loads(observation)
            if isinstance(parsed, Mapping):
                inner = parsed.get("result")
                if isinstance(inner, Mapping):
                    return dict(inner)
                return dict(parsed)
        except json.JSONDecodeError:
            pass
    return {}


def validate_strict_tool_chain(
    transcript: Sequence[Mapping[str, Any]],
    *,
    ledger_steps: Optional[Sequence[Mapping[str, Any]]] = None,
    require_submit: bool = True,
) -> List[str]:
    """Validate commit → run_capability(ok) → submit_solution chain for strict v2."""
    errors: List[str] = []
    saw_commit = False
    commit_route_id: Optional[str] = None
    saw_runcap_ok = False
    runcap_route_id: Optional[str] = None
    saw_submit = False

    for row in transcript:
        role = str(row.get("role") or "")
        if role != "tool":
            continue
        tool_name = str(row.get("name") or "")
        payload = _parse_tool_payload(row.get("observation") or row.get("content") or row)
        if tool_name == "controller_commit_route" and payload.get("ok"):
            saw_commit = True
            commit_route_id = str(payload.get("route_id") or "").strip() or commit_route_id
        if tool_name == "run_capability":
            if payload.get("ok"):
                saw_runcap_ok = True
                runcap_route_id = str(payload.get("route_id") or "").strip() or runcap_route_id
            elif str(payload.get("error") or "") in ENGINEERING_ERROR_CODES:
                errors.append(f"run_capability_engineering:{payload.get('error')}")
        if tool_name == "submit_solution":
            if payload.get("ok") is True and not payload.get("error"):
                saw_submit = True
        if tool_name in STRICT_FORBIDDEN_TERMINALS:
            errors.append(f"forbidden_terminal:{tool_name}")

    if ledger_steps:
        for step in ledger_steps:
            if step.get("kind") == "controller_decision":
                action = str(step.get("action") or "")
                if action in {"commit_route", "commit"}:
                    extra = step.get("extra") or {}
                    if extra.get("ok", True):
                        saw_commit = True
                        commit_route_id = (
                            str(step.get("route_id") or "").strip() or commit_route_id
                        )
            if step.get("kind") != "tool_execution":
                continue
            if step.get("tool") != "run_capability":
                continue
            if step.get("ok"):
                saw_runcap_ok = True
                runcap_route_id = str(step.get("route_id") or "").strip() or runcap_route_id
            elif str(step.get("error") or "") in ENGINEERING_ERROR_CODES:
                errors.append(f"ledger_run_capability_engineering:{step.get('error')}")

    if not saw_commit:
        errors.append("missing_controller_commit_route")
    if not saw_runcap_ok:
        errors.append("missing_run_capability_ok")
    if require_submit and not saw_submit:
        errors.append("missing_submit_solution_ok")
    if commit_route_id and runcap_route_id and commit_route_id != runcap_route_id:
        errors.append(f"route_id_mismatch:{commit_route_id}!={runcap_route_id}")
    return errors


def validate_strict_submission_integrity(
    transcript: Sequence[Mapping[str, Any]],
    *,
    ledger_steps: Optional[Sequence[Mapping[str, Any]]] = None,
    submitted: bool = False,
    require_submit: bool = True,
) -> Dict[str, Any]:
    """Trajectory-level fail-closed submission validator (Ultimate Method §3.3)."""
    chain_errors = validate_strict_tool_chain(
        transcript,
        ledger_steps=ledger_steps,
        require_submit=require_submit and submitted,
    )
    if chain_errors:
        return {
            "valid": False,
            "status": "INVALID_SUBMISSION",
            "errors": chain_errors,
        }
    if require_submit and not submitted:
        return {
            "valid": False,
            "status": "INCOMPLETE",
            "errors": ["missing_terminal_submission"],
        }
    return {"valid": True, "status": "VALID_SUBMISSION", "errors": []}
