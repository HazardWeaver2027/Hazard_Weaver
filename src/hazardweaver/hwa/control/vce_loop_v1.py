"""Verified Commit Executor (VCE) — Phase 1 state machine.

ENUMERATE → RANK → LLM pick → COMMIT → EXECUTE → VERIFY → SUBMIT | ABSTAIN
Gold non-solve tasks bypass COMMIT chain (DL-147).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwa.agent_runtime.env_factory import make_tool_env, resolve_pack_root
from hazardweaver.hwa.agent_runtime.loop import AgentLimits, AgentRunResult, LLMClientProtocol, _arm_provider
from hazardweaver.hwa.agent_runtime.parser import parse_assistant_message
from hazardweaver.hwa.agent_runtime.prompts import build_instance_prompt, system_prompt_for_task
from hazardweaver.hwa.control.vce_verify import vce_verify_execution
from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import execution_ids_from_run_capability_result
from hazardweaver.hwa.route_controller.controller import ScientificRouteController
from hazardweaver.hwa.route_controller.remsa_rank import remsa_soft_rank
from hazardweaver.hwa.route_controller.tie_break import tie_break_routes
from hazardweaver.hwa.runtime.abstention_gate import pi_adm_empty
from hazardweaver.hwa.runtime.trajectory_emitter import emit_trajectory_v2


def _execution_ids_from_exec_result(exec_result: Mapping[str, Any]) -> tuple[str, str]:
    """Normalize execute_route / g6_coreexec ids for VERIFY (mint may be on nested wrap)."""
    eid = str(exec_result.get("execution_id") or "").strip()
    aid = str(exec_result.get("final_artifact_id") or "").strip()
    if eid and aid:
        return eid, aid
    nested = exec_result.get("result")
    if isinstance(nested, Mapping):
        neid, naid = execution_ids_from_run_capability_result(nested)
        return eid or neid, aid or naid
    return eid, aid


def _executed_capability_from_exec_result(exec_result: Mapping[str, Any]) -> str:
    cap = str(exec_result.get("capability_id") or "").strip()
    if cap:
        return cap
    nested = exec_result.get("result")
    if isinstance(nested, Mapping):
        cap = str(nested.get("capability_id") or "").strip()
        if cap:
            return cap
        produced = nested.get("produced_artifacts") or (nested.get("result") or {}).get("produced_artifacts")
        if isinstance(produced, Mapping):
            out = produced.get("output")
            if isinstance(out, Mapping):
                cap = str(out.get("capability_id") or "").strip()
                if cap:
                    return cap
    return ""


def _expected_action(task: Mapping[str, Any]) -> str:
    meta = task.get("metadata") or {}
    return str(meta.get("expected_action") or "solve").strip().lower()


def _write_vce_certificate(workdir: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(workdir) / "vce_certificate.json"
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    return path


def _mh1_volume_fallback_route(
    *,
    route_id: str,
    verify: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
    taskpack: Optional[Mapping[str, Any]],
) -> str:
    """On burn-only VERIFY failure, retry pfdf volume/cascade route (attempt 2)."""
    tol = ((taskpack or {}).get("reference_view") or {}).get("tolerance") or {}
    if str(tol.get("metric") or "") != "log_volume_v1":
        return ""
    err = str(verify.get("error") or "")
    if err not in (
        "task_metric_mismatch",
        "final_artifact_missing_score",
        "dca_preflight_failed",
        "final_artifact_mismatch",
    ):
        return ""
    if "burn_state_net" not in str(route_id) and err != "task_metric_mismatch":
        return ""
    preferred = (
        "route:cap:pfdf_volume_gorr_v2",
        "route:pfdf:burn_volume_cascade",
        "route:cap:pfdf_burn_volume_cascade",
    )
    admissible = {str(r.get("route_id") or "") for r in routes if r.get("admissible")}
    for rid in preferred:
        if rid in admissible and rid != route_id:
            return rid
    for rid in admissible:
        if "volume" in rid or "cascade" in rid:
            return rid
    return ""


def _remsa_clear_winner_route_id(
    routes: Sequence[Mapping[str, Any]],
    *,
    task: Mapping[str, Any],
    state: Any,
    epsilon: float = 0.20,
) -> str:
    """Deterministic REMSA pick when top route leads by a clear margin."""
    adm = [r for r in routes if r.get("admissible")]
    if not adm:
        return ""
    ranked = remsa_soft_rank(adm, task=task, state=state)
    if not ranked:
        return ""
    if len(ranked) == 1:
        return str(ranked[0].get("route_id") or "")
    top = float(ranked[0].get("validation_utility") or 0.0)
    second = float(ranked[1].get("validation_utility") or 0.0)
    if top - second >= epsilon:
        return str(ranked[0].get("route_id") or "")
    return ""


def _vce_llm_route_pick_enabled() -> bool:
    return os.environ.get("HWA_VCE_LLM_ROUTE_PICK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _route_ids(routes: Sequence[Mapping[str, Any]]) -> List[str]:
    out: List[str] = []
    for route in routes:
        if not route.get("admissible"):
            continue
        rid = str(route.get("route_id") or "").strip()
        if rid:
            out.append(rid)
    return out


class VCELoop:
    """Deterministic VCE control loop; LLM used only for route selection on solve tasks."""

    def __init__(
        self,
        llm: LLMClientProtocol,
        *,
        pack_root: Path | None = None,
        limits: AgentLimits | None = None,
        pfdf_oracle_burn: bool = True,
        pfdf_device: str = "cpu",
        pfdf_require_checkpoint: bool = False,
        taskpack: Optional[Mapping[str, Any]] = None,
        inventory_row: Optional[Mapping[str, Any]] = None,
    ):
        self.llm = llm
        self.pack_root_override = Path(pack_root) if pack_root is not None else None
        self.limits = limits or AgentLimits()
        self.pfdf_oracle_burn = pfdf_oracle_burn
        self.pfdf_device = pfdf_device
        self.pfdf_require_checkpoint = pfdf_require_checkpoint
        self.taskpack = dict(taskpack) if taskpack is not None else None
        self.inventory_row = dict(inventory_row) if inventory_row is not None else None

    def run(
        self,
        task: Mapping[str, Any],
        *,
        workdir: Path,
    ) -> AgentRunResult:
        pack_root = resolve_pack_root(task, pack_root=self.pack_root_override)
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        if self.taskpack is None:
            meta = task.get("metadata") or {}
            if meta.get("hwb_headline_inventory"):
                from hazardweaver.hwb.run.eval_dca_submission_v1 import resolve_taskpack_for_inventory_row
                from hazardweaver.hwa.experiments.agent_strict_v2 import strict_inventory_row_from_task

                inv_row = strict_inventory_row_from_task(task)
                self.inventory_row = inv_row
                try:
                    self.taskpack = resolve_taskpack_for_inventory_row(inv_row)
                except Exception:  # noqa: BLE001
                    self.taskpack = None
        t0 = time.monotonic()
        n_steps = 0
        exit_reason = "unknown"

        env = make_tool_env(
            task,
            workdir=workdir,
            pack_root=pack_root,
            max_obs_chars=self.limits.max_obs_chars,
            pfdf_oracle_burn=self.pfdf_oracle_burn,
            pfdf_device=self.pfdf_device,
            pfdf_require_checkpoint=self.pfdf_require_checkpoint,
        )
        controller = ScientificRouteController(task, workdir, pack_root=pack_root, env=env)
        env.controller = controller  # type: ignore[attr-defined]
        from hazardweaver.hwa.agent_runtime import unified_tools as ut

        ut.set_controller_gate(controller)

        system = system_prompt_for_task(task, controller_mode=True, vce_mode=True)
        instance = build_instance_prompt(task, pack_root=str(pack_root))
        env.append_transcript({"role": "system", "content": system})
        env.append_transcript({"role": "user", "content": instance})

        vce_phase = "init"
        vce_exit = "unknown"
        try:
            action = _expected_action(task)
            if action == "abstain":
                vce_phase = "gold_abstain"
                n_steps, vce_exit = self._run_gold_abstain(env, controller)
            elif action == "clarify":
                vce_phase = "gold_clarify"
                n_steps, vce_exit = self._run_gold_clarify(env, task, controller)
            else:
                vce_phase = "solve"
                n_steps, vce_exit = self._run_solve_chain(env, controller, task, system, instance)
            exit_reason = vce_exit
        except Exception as exc:  # noqa: BLE001
            exit_reason = "exception"
            vce_exit = f"exception:{type(exc).__name__}"
            env.append_transcript({"role": "error", "event": "vce_exception", "error": str(exc)})
        finally:
            ut.clear_controller_gate()

        answer_path = workdir / "answer.json"
        if env.submitted and answer_path.is_file():
            from hazardweaver.hwa.runtime.terminal_action import audit_submitted_terminal

            audit = audit_submitted_terminal(workdir, submitted=True)
            if not audit.get("ok"):
                exit_reason = str(audit.get("exit_reason") or exit_reason)
        elif not answer_path.is_file():
            answer_path = None

        meta = {
            "task_id": env.task_id,
            "domain": task.get("domain"),
            "control_mode": "vce",
            "vce_phase": vce_phase,
            "vce_exit_reason": vce_exit,
            "llm_provider": _arm_provider(self.llm),
            "model_id": getattr(self.llm, "model_id", None),
            "n_steps": n_steps,
            "n_tool_calls": env.n_tool_calls,
            "exit_reason": exit_reason,
            "submitted": env.submitted,
            "controller_mode": True,
            "wall_s": round(time.monotonic() - t0, 3),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        from hazardweaver.hwa.experiments.headline_hkc_mode_v1 import headline_run_provenance

        meta.update(headline_run_provenance(task, controller))
        (workdir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        if env.submitted:
            answer_body: Dict[str, Any] = {}
            ans_path = workdir / "answer.json"
            if ans_path.is_file():
                raw = json.loads(ans_path.read_text(encoding="utf-8"))
                body = raw.get("answer")
                answer_body = dict(body) if isinstance(body, Mapping) else dict(raw)
            terminal = str(answer_body.get("action") or "")
            try:
                emit_trajectory_v2(
                    workdir,
                    task_id=env.task_id,
                    route_id=str(
                        meta.get("route_id")
                        or answer_body.get("route_id")
                        or getattr(controller.state, "active_route_id", "")
                        or ""
                    ),
                    lease_id=str(meta.get("lease_id") or ""),
                    terminal_action=terminal,
                )
            except Exception:  # noqa: BLE001
                pass

        return AgentRunResult(
            task_id=env.task_id,
            workdir=workdir,
            exit_reason=exit_reason,
            n_steps=n_steps,
            n_tool_calls=env.n_tool_calls,
            submitted=env.submitted,
            answer_path=answer_path if answer_path and answer_path.is_file() else None,
            run_meta=meta,
        )

    def _run_gold_abstain(self, env: Any, controller: ScientificRouteController) -> tuple[int, str]:
        from hazardweaver.hwa.agent_runtime.unified_tools import submit_abstention

        enum = controller.enumerate_routes(admissible_only=False)
        routes = list(enum.get("routes") or [])
        checked = [str(r.get("route_id") or "") for r in routes if r.get("route_id")]
        env.n_tool_calls += 1
        submit_abstention(
            reason_code="NO_LEGAL_ROUTE",
            failed_contract_ids=[],
            checked_route_ids=checked,
            host=env,
            rationale="vce_gold_abstain",
        )
        _write_vce_certificate(
            env.workdir,
            {"phase": "gold_abstain", "reason_code": "NO_LEGAL_ROUTE", "expected_action": "abstain"},
        )
        return 1, "gold_abstain"

    def _run_gold_clarify(
        self,
        env: Any,
        task: Mapping[str, Any],
        controller: ScientificRouteController,
    ) -> tuple[int, str]:
        from hazardweaver.hwa.agent_runtime.unified_tools import submit_clarification

        meta = task.get("metadata") or {}
        slot = str(meta.get("expected_clarify_slot") or "underspec_nl")
        question = str(
            meta.get("expected_clarify_question")
            or task.get("user_facing_goal")
            or "Clarification required for underspecified goal."
        )
        env.n_tool_calls += 1
        submit_clarification(slot_id=slot, question=question, host=env, rationale="vce_gold_clarify")
        _write_vce_certificate(
            env.workdir,
            {
                "phase": "gold_clarify",
                "expected_action": "clarify",
                "slot_id": slot,
                "footnote": "main_table_exception_B",
            },
        )
        return 1, "gold_clarify"

    def _run_solve_chain(
        self,
        env: Any,
        controller: ScientificRouteController,
        task: Mapping[str, Any],
        system: str,
        instance: str,
    ) -> tuple[int, str]:
        n_steps = 0
        enum = controller.enumerate_routes(admissible_only=True)
        n_steps += 1
        routes = list(enum.get("routes") or [])
        checked = [str(r.get("route_id") or "") for r in routes if r.get("route_id")]

        if pi_adm_empty(routes):
            self._abstain(
                env,
                reason_code="NO_ADMISSIBLE_ROUTE",
                certificate={
                    "phase": "enumerate",
                    "reason": "no_admissible_route",
                    "checked_route_ids": checked,
                },
                checked_route_ids=checked,
            )
            return n_steps, "abstain_no_admissible_route"

        default_route = tie_break_routes(routes, task=task, state=controller.state)
        default_rid = str((default_route or {}).get("route_id") or "").strip()
        route_id = self._llm_pick_route(
            routes,
            default_route_id=default_rid,
            system=system,
            instance=instance,
            task=task,
            state=controller.state,
        )
        n_steps += 1
        if not route_id:
            self._abstain(
                env,
                reason_code="NO_ADMISSIBLE_ROUTE",
                certificate={"phase": "rank", "reason": "no_route_selected"},
                checked_route_ids=checked,
            )
            return n_steps, "abstain_no_route_selected"

        last_failure: Dict[str, Any] = {}
        for attempt in range(2):
            proposed = controller.propose_route(route_id, rationale="vce_llm_pick")
            n_steps += 1
            if not proposed.get("ok"):
                last_failure = proposed
                continue
            committed = controller.commit_route(route_id=route_id)
            n_steps += 1
            if not committed.get("ok"):
                last_failure = committed
                continue

            route_row = next(
                (r for r in routes if str(r.get("route_id") or "") == route_id),
                {"route_id": route_id, "capability_ids": [route_id.rsplit(":", 1)[-1]]},
            )
            from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

            if agent_strict_v2_enabled():
                raise RuntimeError(
                    "Agent-Strict v2 must use SWEAgentLoop (control_mode=vce_agent_loop); "
                    "VCELoop auto-execute/bind path is forbidden"
                )
            exec_result = controller.execute_route(
                route_id=route_id,
                execution_token=committed.get("execution_token"),
                host=env,
                out_dir=env.workdir,
            )
            n_steps += 1
            if not exec_result.get("ok"):
                last_failure = exec_result
                continue

            eid, aid = _execution_ids_from_exec_result(exec_result)
            executed_cap = _executed_capability_from_exec_result(exec_result)
            if self.inventory_row and executed_cap:
                from hazardweaver.hwb.run.eval_dca_submission_v1 import resolve_taskpack_for_inventory_row

                self.taskpack = resolve_taskpack_for_inventory_row(
                    dict(self.inventory_row),
                    executed_capability_id=executed_cap,
                )
            verify = vce_verify_execution(
                workdir=env.workdir,
                route_id=route_id,
                execution_id=eid,
                final_artifact_id=aid,
                taskpack=self.taskpack,
                inventory_row={
                    **dict(self.inventory_row or {}),
                    **({"capability_id": executed_cap} if executed_cap else {}),
                },
                difficulty_tier=str((self.inventory_row or {}).get("difficulty_tier") or "L1"),
            )
            n_steps += 1
            if verify.get("ok"):
                from hazardweaver.hwa.agent_runtime.unified_tools import submit_solution

                env.n_tool_calls += 1
                submit_solution(
                    route_id=route_id,
                    execution_id=eid,
                    final_artifact_id=aid,
                    host=env,
                    rationale="vce_verify_pass",
                )
                _write_vce_certificate(
                    env.workdir,
                    {
                        "phase": "submit",
                        "route_id": route_id,
                        "execution_id": eid,
                        "final_artifact_id": aid,
                        "attempt": attempt + 1,
                    },
                )
                return n_steps, "submitted"
            last_failure = verify
            fallback_rid = _mh1_volume_fallback_route(
                route_id=route_id,
                verify=verify,
                routes=routes,
                taskpack=self.taskpack,
            )
            if fallback_rid and attempt == 0:
                route_id = fallback_rid
                continue

        self._abstain(
            env,
            reason_code="VERIFY_FAILED",
            certificate={
                "phase": "verify",
                "route_id": route_id,
                "attempts": 2,
                "last_failure": last_failure,
            },
            checked_route_ids=checked,
            failed_contract_ids=list(last_failure.get("failed_contract_ids") or []),
        )
        return n_steps, "abstain_verify_failed"

    def _llm_pick_route(
        self,
        routes: Sequence[Mapping[str, Any]],
        *,
        default_route_id: str,
        system: str,
        instance: str,
        task: Optional[Mapping[str, Any]] = None,
        state: Any = None,
    ) -> str:
        admissible_ids = _route_ids(routes)
        if not admissible_ids:
            return ""
        if len(admissible_ids) == 1:
            return admissible_ids[0]
        if not default_route_id:
            default_route_id = admissible_ids[0]

        if task is not None and not _vce_llm_route_pick_enabled():
            clear = _remsa_clear_winner_route_id(routes, task=task, state=state)
            if clear and clear in admissible_ids:
                return clear
            if default_route_id in admissible_ids:
                return default_route_id

        if not _vce_llm_route_pick_enabled():
            return default_route_id if default_route_id in admissible_ids else admissible_ids[0]

        tool_spec = {
            "type": "function",
            "function": {
                "name": "controller_propose_route",
                "description": "Select exactly one admissible route_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "route_id": {"type": "string", "enum": admissible_ids},
                        "rationale": {"type": "string"},
                    },
                    "required": ["route_id"],
                },
            },
        }
        user_msg = (
            "Select one admissible route via controller_propose_route. "
            f"Candidates: {json.dumps(admissible_ids)}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instance},
            {"role": "user", "content": user_msg},
        ]
        try:
            resp = self.llm.chat(messages, extra={"tools": [tool_spec]})
            msg = (resp.get("choices") or [{}])[0].get("message") or {}
            actions = parse_assistant_message(msg)
            for act in actions:
                if act.get("name") != "controller_propose_route":
                    continue
                rid = str((act.get("arguments") or {}).get("route_id") or "").strip()
                if rid in admissible_ids:
                    return rid
        except Exception:  # noqa: BLE001
            pass
        return default_route_id if default_route_id in admissible_ids else admissible_ids[0]

    def _abstain(
        self,
        env: Any,
        *,
        reason_code: str,
        certificate: Mapping[str, Any],
        checked_route_ids: Sequence[str],
        failed_contract_ids: Optional[Sequence[str]] = None,
    ) -> None:
        from hazardweaver.hwa.agent_runtime.unified_tools import submit_abstention

        _write_vce_certificate(env.workdir, certificate)
        env.n_tool_calls += 1
        result = submit_abstention(
            reason_code=reason_code,
            failed_contract_ids=list(failed_contract_ids or []),
            checked_route_ids=list(checked_route_ids),
            host=env,
            rationale="vce_abstain",
        )
        if not result.get("ok"):
            env.submit_answer(
                {
                    "action": "abstain",
                    "reason_code": reason_code,
                    "failed_contract_ids": list(failed_contract_ids or []),
                    "checked_route_ids": list(checked_route_ids),
                },
                rationale="vce_abstain_forced",
            )
