"""Wildfire tool environment: dispatch + workdir + gold leakage guard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwa.agent_runtime.optional_tools import suggest_plan, validate_plan
from hazardweaver.hwa.agent_runtime.prompts import truncate_obs
from hazardweaver.hwa.agent_runtime.tool_specs import OPTIONAL_PLAN_TOOL_NAMES
from hazardweaver.hwa.scientific_controller.tool_policy import CONTROLLER_EXEC_TOOLS
from hazardweaver.hwa.wildfire.data_store import PACK_ROOT
from hazardweaver.hwa.wildfire.tools import WildfireAgentTools


class GoldLeakageError(ValueError):
    """Raised when a solver task still contains gold."""


class WildfireToolEnv:
    """Execute typed HWA tools into a per-task workdir."""

    def __init__(
        self,
        task: Mapping[str, Any],
        *,
        workdir: Path,
        pack_root: Path | None = None,
        tools: WildfireAgentTools | None = None,
        max_obs_chars: int = 8000,
    ):
        if "gold" in task:
            raise GoldLeakageError(
                f"refusing to start agent: task {task.get('task_id')!r} contains gold"
            )
        self.task = dict(task)
        self.task_id = str(task.get("task_id") or "unknown_task")
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.pack_root = Path(pack_root or PACK_ROOT)
        self.max_obs_chars = max_obs_chars
        self.tools = tools or WildfireAgentTools(
            pack_root=self.pack_root,
            default_submit_dir=self.workdir,
            clarify_dir=self.workdir,
        )
        self.tools._agent_env = self  # type: ignore[attr-defined]
        self.submitted = False
        self.n_tool_calls = 0
        self.pending_clarify = False
        self.clarify_mode = str(task.get("clarify_mode") or "") or None
        self.scripted_user = dict(task.get("scripted_user") or {}) or None
        self.controller = None
        self._transcript = self.workdir / "transcript.jsonl"
        self._tool_audit = self.workdir / "tool_calls.jsonl"
        # Fresh files for this run
        for path in (self._transcript, self._tool_audit):
            if path.exists():
                path.unlink()
            path.touch()
        su_path = self.workdir / "scripted_user.json"
        if self.scripted_user:
            su_path.write_text(json.dumps(self.scripted_user, indent=2) + "\n", encoding="utf-8")
        elif su_path.exists():
            su_path.unlink()

    def submit_answer(self, answer: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.tools.submit_answer(answer, **kwargs)

    def append_transcript(self, record: Mapping[str, Any]) -> None:
        row = dict(record)
        row.setdefault("ts_utc", datetime.now(timezone.utc).isoformat())
        with self._transcript.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _audit(
        self,
        name: str,
        *,
        ok: bool,
        error: Optional[str] = None,
        arguments: Optional[Mapping[str, Any]] = None,
        result: Optional[Mapping[str, Any]] = None,
    ) -> None:
        row: Dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "ok": ok,
            "error": error,
        }
        if arguments is not None:
            row["arguments"] = dict(arguments)
        if result is not None and name.startswith("tool_hcg_"):
            # Structured HCG log: keep edges/outputs/sources compact
            row["hcg"] = {
                k: result.get(k)
                for k in (
                    "sources",
                    "target",
                    "packs",
                    "edges",
                    "outputs",
                    "n_paths",
                    "paths",
                    "abstain_reason",
                    "executable_only",
                )
                if k in result
            }
        with self._tool_audit.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def execute(self, action: Mapping[str, Any]) -> Dict[str, Any]:
        name = str(action.get("name") or "")
        args = dict(action.get("arguments") or {})
        call_id = action.get("id")
        self.n_tool_calls += 1
        try:
            result = self._dispatch(name, args)
            ok = True
            err = None
        except Exception as exc:  # noqa: BLE001 — surface to LLM as observation
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            ok = False
            err = str(exc)
        self._audit(
            name,
            ok=ok,
            error=err,
            arguments=args,
            result=result if isinstance(result, dict) else None,
        )
        obs = {
            "tool_call_id": call_id,
            "name": name,
            "ok": ok,
            "result": result,
        }
        content = truncate_obs(obs, self.max_obs_chars)
        self.append_transcript(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "arguments": args,
                "observation": content,
            }
        )
        return {
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        }

    def _dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_allows_run_capability
        from hazardweaver.hwa.experiments.rq4_route_intervention_v1 import maybe_rq4_execute_block

        blocked = maybe_rq4_execute_block(name, args, self.task)
        if blocked is not None:
            return blocked

        if self.controller is not None and name in CONTROLLER_EXEC_TOOLS:
            if not (name == "run_capability" and agent_strict_v2_allows_run_capability()):
                return {
                    "ok": False,
                    "error": "direct_execution_forbidden",
                    "tool": name,
                    "message": (
                        "Capability execution must go through Deterministic Scientific Controller "
                        "(controller_enumerate_routes → controller_propose_route → controller_commit_route)."
                    ),
                }

        if name in OPTIONAL_PLAN_TOOL_NAMES:
            if name == "suggest_plan":
                return suggest_plan(self.task)
            if name == "validate_plan":
                plan = args.get("plan")
                if plan is None:
                    plan = args
                return validate_plan(plan, self.task)
            raise KeyError(f"unknown optional tool: {name}")

        if name in {"get_execution_result"}:
            from hazardweaver.hwa.agent_runtime.unified_tools import get_execution_result as _ger

            return _ger(
                str(args.get("execution_id") or ""),
                host=self,
                out_dir=self.workdir,
            )

        if name in {"submit_solution", "submit_clarification", "submit_abstention"}:
            from hazardweaver.hwa.agent_runtime import unified_tools as ut

            common = {
                "host": self,
                "rationale": args.get("rationale"),
                "model_id_used": args.get("model_id_used"),
                "out_dir": self.workdir,
                "task_id": self.task_id,
            }
            if name == "submit_solution":
                result = ut.submit_solution(
                    route_id=args.get("route_id"),
                    execution_id=args.get("execution_id"),
                    final_artifact_id=args.get("final_artifact_id"),
                    **common,
                )
            elif name == "submit_clarification":
                result = ut.submit_clarification(
                    slot_id=args.get("slot_id"),
                    question=args.get("question"),
                    **common,
                )
            else:
                result = ut.submit_abstention(
                    reason_code=args.get("reason_code"),
                    failed_contract_ids=args.get("failed_contract_ids"),
                    checked_route_ids=args.get("checked_route_ids"),
                    **common,
                )
            if result.get("ok") is True:
                self.submitted = True
            elif name == "submit_solution" and result.get("ok") is not True:
                from hazardweaver.hwa.experiments.ablation_manual_pilot_terminal_v1 import (
                    maybe_terminalize_failed_submit,
                )

                handles = {
                    "route_id": str(args.get("route_id") or ""),
                    "execution_id": str(args.get("execution_id") or ""),
                    "final_artifact_id": str(args.get("final_artifact_id") or ""),
                }
                if maybe_terminalize_failed_submit(self, self.task, result, handles):
                    result = dict(result)
                    result["ok"] = True
                    result["terminal_policy"] = "ablation_manual_pilot_invalid_dca_v1"
                    result["valid"] = False
            elif name == "submit_abstention" and result.get("ok") is not True:
                from hazardweaver.hwa.experiments.ablation_manual_pilot_terminal_v1 import (
                    maybe_terminalize_blocked_abstention,
                    maybe_terminalize_no_admissible_routes_spin,
                )

                if maybe_terminalize_no_admissible_routes_spin(self, self.task, result):
                    result = dict(result)
                    result["ok"] = True
                    result["terminal_policy"] = "unified_no_admissible_routes_spin_v1"
                    result["valid"] = False
                elif maybe_terminalize_blocked_abstention(self, self.task, result):
                    result = dict(result)
                    result["ok"] = True
                    result["terminal_policy"] = "ablation_manual_pilot_blocked_abstention_v1"
                    result["valid"] = False
            return result

        if name in {"submit_answer", "submit"}:
            from hazardweaver.hwa.experiments.agent_strict_v2 import (
                agent_strict_v2_enabled,
                is_strict_forbidden_terminal,
                strict_terminal_block_payload,
            )

            if agent_strict_v2_enabled() and is_strict_forbidden_terminal(name):
                return strict_terminal_block_payload(name)
            # Force write into this task workdir; inject task_id.
            # G1: submit ↔ submit_answer alias (adversarial schema uses submit).
            from hazardweaver.hwa.agent_runtime.unified_tools import normalize_legacy_answer
            from hazardweaver.hwa.runtime.terminal_action import (
                coerce_answer_body,
                normalize_terminal_action,
                terminal_from_answer_record,
            )

            answer, repairs = normalize_legacy_answer(args.get("answer"))
            if isinstance(answer, Mapping):
                answer = coerce_answer_body(answer)
                terminal = normalize_terminal_action(answer.get("action"))
                if terminal == "abstain":
                    from hazardweaver.hwa.agent_runtime import unified_tools as ut

                    abstain_out = ut.submit_abstention(
                        reason_code=answer.get("reason_code"),
                        failed_contract_ids=answer.get("failed_contract_ids") or [],
                        checked_route_ids=answer.get("checked_route_ids") or [],
                        host=self,
                        rationale=args.get("rationale"),
                        model_id_used=args.get("model_id_used"),
                        out_dir=self.workdir,
                        task_id=self.task_id,
                    )
                    if abstain_out.get("ok") is True:
                        self.submitted = True
                    return abstain_out
            if self.controller is not None and isinstance(answer, Mapping):
                terminal = normalize_terminal_action(answer.get("action"))
                if terminal == "solve":
                    rid = str(answer.get("route_id") or "").strip()
                    eid = str(answer.get("execution_id") or "").strip()
                    aid = str(answer.get("final_artifact_id") or "").strip()
                    if not (rid and eid and aid):
                        return {
                            "ok": False,
                            "error": "controller_mode_requires_submit_solution",
                            "tool": name,
                            "message": (
                                "In controller mode, solve via submit_solution using "
                                "execution_id and final_artifact_id from controller_commit_route."
                            ),
                        }
                    from hazardweaver.hwa.agent_runtime.execution_schema import verify_submit_solution_ids

                    check = verify_submit_solution_ids(
                        workdir=self.workdir,
                        route_id=rid,
                        execution_id=eid,
                        final_artifact_id=aid,
                    )
                    if not check.get("ok"):
                        return {**check, "tool": name}
            result = self.tools.submit_answer(
                answer,
                rationale=args.get("rationale"),
                model_id_used=args.get("model_id_used"),
                out_dir=self.workdir,
                task_id=self.task_id,
            )
            if repairs and isinstance(result, dict):
                result = dict(result)
                result["schema_repair_count"] = int(result.get("schema_repair_count") or 0) + repairs
            if result.get("ok") is True:
                terminal = (
                    terminal_from_answer_record({"answer": answer})
                    if isinstance(answer, Mapping)
                    else ""
                )
                if terminal:
                    self.submitted = True
                else:
                    result = dict(result)
                    result["ok"] = False
                    result["error"] = "invalid_terminal_answer_empty"
            return result

        if name in {"ask_user", "clarify_slot"}:
            if self.scripted_user:
                from hazardweaver.hwa.agent_runtime.scripted_user import respond as _su_respond

                su = _su_respond(
                    asked_slots=list(args.get("slots") or []),
                    scripted_user=self.scripted_user,
                )
                if su.get("status") == "answered":
                    reply_path = self.workdir / "clarifications_reply.json"
                    reply_path.write_text(
                        json.dumps({"status": "answered", "answers": su.get("answers") or {}}, indent=2)
                        + "\n",
                        encoding="utf-8",
                    )
                elif su.get("status") in {"refused", "irrelevant"}:
                    # Still call ask_user to log, but return non-pending status
                    result = {
                        "ok": True,
                        "question": str(args.get("question") or ""),
                        "slots": list(args.get("slots") or []),
                        "status": su.get("status"),
                        "note": su.get("note"),
                        "answers": su.get("answers") or {},
                        "scripted_user": True,
                    }
                    self.pending_clarify = False
                    return result

            result = self.tools.ask_user(
                str(args.get("question") or ""),
                list(args.get("slots") or []),
                out_dir=self.workdir,
            )
            if result.get("status") == "pending_user":
                self.pending_clarify = True
            elif result.get("status") == "answered":
                self.pending_clarify = False
            return result

        # Strip agent-runtime-only keys if model hallucinates them
        clean = {k: v for k, v in args.items() if k not in {"out_dir", "task_id"}}
        return self.tools.call(name, **clean)

    def has_pending_clarifications(self) -> bool:
        path = self.workdir / "clarifications.jsonl"
        if not path.is_file():
            return bool(self.pending_clarify)
        pending = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "pending_user":
                pending = True
            if row.get("status") == "answered":
                pending = False
        return pending
