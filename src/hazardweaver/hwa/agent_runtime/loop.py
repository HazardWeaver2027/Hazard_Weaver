"""SWE-style agent loop: LLM controls tools until submit_answer or limits."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from hazardweaver.hwa.agent_runtime.env_factory import make_tool_env, resolve_pack_root
from hazardweaver.hwa.agent_runtime.parser import parse_assistant_message
from hazardweaver.hwa.agent_runtime.prompts import build_instance_prompt, system_prompt_for_task
from hazardweaver.hwa.agent_runtime.tool_specs import openai_tool_specs
from hazardweaver.hwa.llm.context_budget import fit_chat_payload_to_budget
from hazardweaver.hwa.route_controller.controller import ScientificRouteController
from hazardweaver.hwa.route_controller.tool_surface_policy import tool_specs_for_controller
from hazardweaver.hwa.runtime.trajectory_emitter import emit_trajectory_v2
from hazardweaver.hwa.route_controller.fl2_pilot_slice import is_fl2_pilot_task
from hazardweaver.hwa.scientific_controller.controller import (
    ScientificController,
    controller_mode_from_task,
)
from hazardweaver.hwa.wildfire.data_store import PACK_ROOT


def _arm_provider(llm: Any) -> str:
    """P0-7: arm-level provider tag (never bare transport ``vllm``)."""
    raw = getattr(llm, "provider", None)
    if raw and str(raw) not in {"vllm", "unknown", ""}:
        return str(raw)
    from hazardweaver.hwa.llm.provider_normalize import normalize_llm_provider

    cfg = getattr(llm, "config", None)
    return normalize_llm_provider(
        raw or "vllm",
        profile=getattr(cfg, "profile", None) or getattr(llm, "profile", None),
        model_id=getattr(llm, "model_id", None),
    )


class LLMClientProtocol(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]: ...


@dataclass
class AgentLimits:
    max_steps: int = 30
    max_wall_s: float = 600.0
    max_obs_chars: int = 8000
    temperature: float = 0.2
    max_tokens: int = 1024
    # After ask_user returns pending_user, allow this many further steps before fail.
    clarify_grace_steps: int = 2


@dataclass
class AgentRunResult:
    task_id: str
    workdir: Path
    exit_reason: str
    n_steps: int
    n_tool_calls: int
    submitted: bool
    answer_path: Optional[Path]
    run_meta: Dict[str, Any] = field(default_factory=dict)


class SWEAgentLoop:
    """mini-SWE-inspired linear loop over WildfireToolEnv or PfdfToolEnv."""

    def __init__(
        self,
        llm: LLMClientProtocol,
        *,
        pack_root: Path | None = None,
        limits: AgentLimits | None = None,
        include_optional_tools: bool = True,
        pfdf_oracle_burn: bool = True,
        pfdf_device: str = "cpu",
        pfdf_require_checkpoint: bool = False,
        controller_mode: Optional[bool] = None,
    ):
        self.llm = llm
        self.pack_root_override = Path(pack_root) if pack_root is not None else None
        self.limits = limits or AgentLimits()
        self.include_optional_tools = include_optional_tools
        self.pfdf_oracle_burn = pfdf_oracle_burn
        self.pfdf_device = pfdf_device
        self.pfdf_require_checkpoint = pfdf_require_checkpoint
        self.controller_mode_override = controller_mode

    def run(
        self,
        task: Mapping[str, Any],
        *,
        workdir: Path,
    ) -> AgentRunResult:
        from hazardweaver.hwa.control.mode import is_vce_mode
        from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_use_agent_loop

        if is_vce_mode() and not agent_strict_v2_use_agent_loop():
            from hazardweaver.hwa.control.vce_loop_v1 import VCELoop

            return VCELoop(
                self.llm,
                pack_root=self.pack_root_override,
                limits=self.limits,
                pfdf_oracle_burn=self.pfdf_oracle_burn,
                pfdf_device=self.pfdf_device,
                pfdf_require_checkpoint=self.pfdf_require_checkpoint,
            ).run(task, workdir=workdir)

        pack_root = resolve_pack_root(task, pack_root=self.pack_root_override)
        from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

        pfdf_oracle_burn = self.pfdf_oracle_burn
        if agent_strict_v2_enabled():
            pfdf_oracle_burn = False
        use_controller = (
            self.controller_mode_override
            if self.controller_mode_override is not None
            else True
        )
        if self.controller_mode_override is None and controller_mode_from_task(task) is False:
            sv = (task.get("solver_visible") or {})
            if sv.get("legacy_mode") is True:
                use_controller = False
        env = make_tool_env(
            task,
            workdir=workdir,
            pack_root=pack_root,
            max_obs_chars=self.limits.max_obs_chars,
            pfdf_oracle_burn=pfdf_oracle_burn,
            pfdf_device=self.pfdf_device,
            pfdf_require_checkpoint=self.pfdf_require_checkpoint,
        )
        controller: Optional[ScientificController] = None
        if use_controller:
            controller = ScientificRouteController(
                task,
                workdir,
                pack_root=pack_root,
                env=env,
            )
            env.controller = controller  # type: ignore[attr-defined]
            from hazardweaver.hwa.agent_runtime import unified_tools as ut

            ut.set_controller_gate(controller)
        else:
            from hazardweaver.hwa.agent_runtime import unified_tools as ut

            ut.clear_controller_gate()

        allowed = None
        inv = (task.get("solver_visible") or {}).get("allowed_inventory") or {}
        if inv.get("tool_ids"):
            allowed = list(inv["tool_ids"])
        tools = tool_specs_for_controller(
            pack_root,
            include_optional=self.include_optional_tools,
            allowed_tool_ids=allowed,
        ) if use_controller else openai_tool_specs(
            pack_root,
            include_optional=self.include_optional_tools,
            allowed_tool_ids=allowed,
            controller_mode=False,
        )

        from hazardweaver.hwa.control.mode import is_vce_mode

        system = system_prompt_for_task(
            task,
            controller_mode=bool(use_controller),
            vce_mode=is_vce_mode(),
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": build_instance_prompt(task, pack_root=str(pack_root)),
            },
        ]
        env.append_transcript({"role": "system", "content": system})
        env.append_transcript({"role": "user", "content": messages[1]["content"]})

        t0 = time.monotonic()
        os.environ["HW_LLM_DEADLINE_MONOTONIC"] = str(t0 + self.limits.max_wall_s)
        exit_reason = "unknown"
        n_steps = 0
        clarify_pending_since: Optional[int] = None
        crash_error: Optional[str] = None

        try:
            return self._run_loop_body(
                task=task,
                env=env,
                pack_root=pack_root,
                tools=tools,
                messages=messages,
                allowed=allowed,
                t0=t0,
                controller=controller,
                use_controller=bool(use_controller),
                pfdf_oracle_burn=pfdf_oracle_burn,
            )
        except Exception as exc:  # noqa: BLE001 — must always persist run_meta
            crash_error = f"{type(exc).__name__}: {exc}"
            exit_reason = "exception"
            env.append_transcript(
                {"role": "error", "event": "unhandled_exception", "error": crash_error}
            )
            answer_path = env.workdir / "answer.json"
            meta = {
                "task_id": env.task_id,
                "domain": task.get("domain"),
                "task_layer": task.get("task_layer"),
                "task_family": task.get("task_family"),
                "llm_provider": _arm_provider(self.llm),
                "transport": getattr(self.llm, "transport", None),
                "model_id": getattr(self.llm, "model_id", None),
                "base_url": getattr(getattr(self.llm, "config", None), "base_url", None),
                "profile": getattr(getattr(self.llm, "config", None), "profile", None),
                "n_steps": n_steps,
                "n_tool_calls": env.n_tool_calls,
                "exit_reason": exit_reason,
                "submitted": env.submitted,
                "pending_clarify": env.has_pending_clarifications(),
                "include_optional_tools": self.include_optional_tools,
                "wall_s": round(time.monotonic() - t0, 3),
                "max_steps": self.limits.max_steps,
                "max_wall_s": self.limits.max_wall_s,
                "clarify_grace_steps": self.limits.clarify_grace_steps,
                "pack_root": str(pack_root),
                "error": crash_error,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
            from hazardweaver.hwa.experiments.headline_hkc_mode_v1 import headline_run_provenance

            meta.update(headline_run_provenance(task))
            (env.workdir / "run_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            return AgentRunResult(
                task_id=env.task_id,
                workdir=env.workdir,
                exit_reason=exit_reason,
                n_steps=n_steps,
                n_tool_calls=env.n_tool_calls,
                submitted=env.submitted,
                answer_path=answer_path if answer_path.is_file() else None,
                run_meta=meta,
            )
        finally:
            os.environ.pop("HW_LLM_DEADLINE_MONOTONIC", None)
            from hazardweaver.hwa.agent_runtime import unified_tools as ut

            ut.clear_controller_gate()

    def _run_loop_body(
        self,
        *,
        task: Mapping[str, Any],
        env: Any,
        pack_root: Path,
        tools: Any,
        messages: List[Dict[str, Any]],
        allowed: Optional[List[str]],
        t0: float,
        controller: Optional[ScientificController] = None,
        use_controller: bool = False,
        pfdf_oracle_burn: bool = True,
    ) -> AgentRunResult:
        exit_reason = "unknown"
        n_steps = 0
        clarify_pending_since: Optional[int] = None
        irrelevant_ask_streak = 0
        parse_error_streak = 0
        last_ask_slots: List[str] = []
        last_required_slots: List[str] = []
        resolution_answered = False
        last_pi_refresh_checkpoint: Optional[int] = None
        from hazardweaver.hwa.experiments.unified_solve_mandatory_pre_controller_v1 import PreControllerGuardState

        pre_controller_guard = PreControllerGuardState() if use_controller else None
        pre_controller_terminal = False

        while True:
            if n_steps >= self.limits.max_steps:
                exit_reason = "limit_steps"
                break
            if (time.monotonic() - t0) >= self.limits.max_wall_s:
                exit_reason = "limit_wall"
                break
            if (
                clarify_pending_since is not None
                and env.has_pending_clarifications()
                and (n_steps - clarify_pending_since) >= self.limits.clarify_grace_steps
            ):
                mode = str(getattr(env, "clarify_mode", None) or task.get("clarify_mode") or "")
                has_scripted = bool(getattr(env, "scripted_user", None) or task.get("scripted_user"))
                if mode == "decision" or (mode != "resolution" and not has_scripted):
                    exit_reason = "env_null_no_user"
                    env.append_transcript(
                        {
                            "role": "system",
                            "event": "env_null_no_user",
                            "note": (
                                "ask_user still pending after grace; "
                                "Clarify-Decision / no scripted user — not counted as model fail"
                            ),
                            "step": n_steps,
                        }
                    )
                else:
                    exit_reason = "failed_clarify"
                    env.append_transcript(
                        {
                            "role": "system",
                            "event": "failed_clarify",
                            "note": "ask_user still pending_user after grace steps; honest fail",
                            "step": n_steps,
                        }
                    )
                break

            n_steps += 1
            if controller is not None and use_controller:
                cp = int(getattr(controller.state, "checkpoint", 0) or 0)
                need_refresh = (
                    last_pi_refresh_checkpoint is None
                    or cp != last_pi_refresh_checkpoint
                    or not getattr(controller, "_route_cache", None)
                )
                if need_refresh:
                    refresh = (
                        controller.refresh_admissible_surface()
                        if hasattr(controller, "refresh_admissible_surface")
                        else controller.enumerate_routes(admissible_only=False)
                    )
                    last_pi_refresh_checkpoint = int(refresh.get("checkpoint") or cp)
                else:
                    remaining = (
                        controller.remaining_admissible_route_ids()
                        if hasattr(controller, "remaining_admissible_route_ids")
                        else []
                    )
                    refresh = {
                        "n_routes": len(remaining),
                        "checkpoint": cp,
                        "cached_only": True,
                    }
                env.append_transcript(
                    {
                        "role": "system",
                        "event": "pi_adm_refresh",
                        "step": n_steps,
                        "n_routes": refresh.get("n_routes"),
                        "checkpoint": refresh.get("checkpoint"),
                        "cached_only": bool(refresh.get("cached_only")),
                    }
                )
                if hasattr(controller, "all_candidates_exhausted") and controller.all_candidates_exhausted():
                    recovery = (
                        controller.candidates_exhausted_recovery()
                        if hasattr(controller, "candidates_exhausted_recovery")
                        else {}
                    )
                    hint = str(recovery.get("next_step") or recovery.get("error") or "all_candidates_exhausted")
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "All enumerated routes at this checkpoint were rejected. "
                                f"{hint}"
                            ),
                        }
                    )
                    env.append_transcript(
                        {
                            "role": "system",
                            "event": "pi_candidates_exhausted",
                            "step": n_steps,
                            "rejected_route_ids": recovery.get("rejected_route_ids"),
                        }
                    )
                from hazardweaver.hwa.experiments.headline_g1_autoroute_v1 import (
                    g1_autocommit_submit_hint,
                    try_finish_g1_autocommit_loop,
                )
                from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

                if not agent_strict_v2_enabled():
                    if try_finish_g1_autocommit_loop(env, refresh):
                        exit_reason = "submitted"
                        env.append_transcript(
                            {
                                "role": "system",
                                "event": "g1_autocommit_loop_exit",
                                "step": n_steps,
                            }
                        )
                        break

                    hint = g1_autocommit_submit_hint(refresh)
                    if hint:
                        messages.append({"role": "system", "content": hint})
                        env.append_transcript(
                            {
                                "role": "system",
                                "event": "g1_autocommit_hint",
                                "content": hint,
                                "step": n_steps,
                            }
                        )

                from hazardweaver.hwa.experiments.unified_e12_agent_nudge_v1 import unified_e12_post_shock_loop_nudge

                e12_hint = unified_e12_post_shock_loop_nudge(controller, refresh, step_n=n_steps)
                if e12_hint:
                    messages.append({"role": "system", "content": e12_hint})
                    env.append_transcript(
                        {
                            "role": "system",
                            "event": "e12_post_shock_commit_nudge",
                            "content": e12_hint,
                            "step": n_steps,
                        }
                    )

            if pre_controller_guard is not None:
                from hazardweaver.hwa.experiments.unified_solve_mandatory_pre_controller_v1 import (
                    pre_controller_enumerate_nudge,
                )

                pc_nudge = pre_controller_enumerate_nudge(
                    task,
                    pre_controller_guard,
                    step_n=n_steps,
                    use_controller=use_controller,
                )
                if pc_nudge:
                    messages.append({"role": "system", "content": pc_nudge})
                    env.append_transcript(
                        {
                            "role": "system",
                            "event": "pre_controller_enumerate_nudge",
                            "content": pc_nudge,
                            "step": n_steps,
                        }
                    )

            message: Optional[Dict[str, Any]] = None
            chat_messages, chat_tools = fit_chat_payload_to_budget(messages, tools)
            for empty_retry in range(2):
                resp = self.llm.chat(
                    chat_messages,
                    temperature=self.limits.temperature,
                    max_tokens=self.limits.max_tokens,
                    extra={"tools": chat_tools, "tool_choice": "auto"},
                )
                try:
                    message = resp["choices"][0]["message"]
                except (KeyError, IndexError, TypeError) as exc:
                    exit_reason = "bad_llm_response"
                    env.append_transcript(
                        {"role": "error", "error": f"bad_llm_response: {exc}", "raw": resp}
                    )
                    break
                if not isinstance(message, dict):
                    message = dict(message)
                has_tools = bool(message.get("tool_calls"))
                has_text = bool(str(message.get("content") or "").strip())
                if has_tools or has_text or empty_retry == 1:
                    break
                env.append_transcript(
                    {
                        "role": "system",
                        "event": "empty_llm_retry",
                        "step": n_steps,
                        "retry": empty_retry + 1,
                    }
                )
            if message is None:
                break

            assistant_row: Dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content"),
            }
            if message.get("tool_calls"):
                assistant_row["tool_calls"] = message["tool_calls"]
            messages.append(assistant_row)
            env.append_transcript({"role": "assistant", "message": assistant_row, "step": n_steps})

            actions, parse_err = parse_assistant_message(message)
            if not actions:
                parse_error_streak += 1
                if (
                    messages
                    and str(messages[-1].get("role")) == "assistant"
                    and not messages[-1].get("tool_calls")
                ):
                    content = str(messages[-1].get("content") or "")
                    if len(content) > 400:
                        messages[-1]["content"] = (
                            content[:400] + "\n…[truncated parse-fail prose for context budget]"
                        )
                if parse_error_streak >= 3:
                    exit_reason = "parse_loop"
                    env.append_transcript(
                        {
                            "role": "system",
                            "event": "parse_loop_exit",
                            "parse_error_streak": parse_error_streak,
                            "last_error": parse_err,
                            "step": n_steps,
                        }
                    )
                    break
                note = {
                    "role": "user",
                    "content": (
                        f"parse_error={parse_err}. "
                        "Respond with a tool call (function calling) or a JSON object "
                        'like {"name":"list_inventory","arguments":{"kind":"all"}}.'
                    ),
                }
                messages.append(note)
                env.append_transcript({"role": "parse_error", "error": parse_err, "step": n_steps})
                continue
            parse_error_streak = 0

            for action in actions:
                name = str(action.get("name") or "")
                from hazardweaver.hwa.agent_runtime.tool_args_sanitize_v1 import sanitize_tool_arguments

                args = sanitize_tool_arguments(name, dict(action.get("arguments") or {}))
                # Enforce ablation inventory: refuse tools not advertised in specs.
                if allowed is not None and name and name not in allowed:
                    semantic_obs = False
                    obs = {
                        "ok": False,
                        "name": name,
                        "tool_call_id": action.get("id"),
                        "content": json.dumps(
                            {
                                "ok": False,
                                "error": "tool_not_in_allowed_inventory",
                                "tool": name,
                                "allowed_tool_ids": list(allowed),
                            }
                        ),
                    }
                    env.append_transcript(
                        {
                            "role": "tool_rejected",
                            "tool": name,
                            "step": n_steps,
                            "reason": "tool_not_in_allowed_inventory",
                        }
                    )
                else:
                    from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_allows_run_capability

                    block_exec = (
                        controller
                        and controller.is_exec_tool(name)
                        and not (name == "run_capability" and agent_strict_v2_allows_run_capability())
                    )
                    semantic_obs = False
                    if block_exec:
                        obs = controller.reject_direct_execution(name, action)
                        obs = {
                            "tool_call_id": action.get("id"),
                            "name": name,
                            "content": obs.get("content")
                            if "content" in obs
                            else json.dumps(obs, default=str),
                        }
                        semantic_obs = True
                    elif controller and controller.is_semantic_tool(name):
                        obs = controller.handle_semantic_tool(
                            name,
                            args,
                            action_id=action.get("id"),
                        )
                        semantic_obs = True
                    else:
                        obs = env.execute(action)
                        if controller and obs.get("ok") and isinstance(obs.get("result"), dict):
                            controller.update_state_from_observation(name, obs["result"])
                        elif (
                            controller
                            and not obs.get("ok")
                            and name == "run_capability"
                            and hasattr(controller, "handle_tool_failure")
                        ):
                            try:
                                inner = json.loads(obs.get("content") or "{}")
                                if isinstance(inner.get("result"), dict):
                                    inner = inner["result"]
                            except (TypeError, json.JSONDecodeError):
                                inner = {}
                            cid = str(args.get("capability_id") or inner.get("capability_id") or "")
                            err = str(inner.get("error") or obs.get("error") or "execution_failed")
                            controller.handle_tool_failure(cid, err)
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": obs.get("tool_call_id") or action.get("id"),
                    "name": obs.get("name") or action.get("name"),
                    "content": obs["content"],
                }
                messages.append(tool_msg)
                if semantic_obs:
                    env.append_transcript(
                        {
                            "role": "tool",
                            "tool_call_id": tool_msg["tool_call_id"],
                            "name": tool_msg["name"],
                            "arguments": args,
                            "observation": tool_msg["content"],
                            "step": n_steps,
                        }
                    )
                    if name == "controller_commit_route":
                        from hazardweaver.hwa.experiments.headline_strict_chained_autosubmit_v1 import (
                            try_finish_strict_chained_commit,
                        )

                        if try_finish_strict_chained_commit(env, task, tool_msg["content"]):
                            exit_reason = "submitted"
                            env.append_transcript(
                                {
                                    "role": "system",
                                    "event": "strict_chained_autosubmit_exit",
                                    "step": n_steps,
                                }
                            )
                            break

                    if name == "controller_commit_route":
                        try:
                            commit_payload = json.loads(tool_msg.get("content") or "{}")
                            if isinstance(commit_payload.get("result"), dict):
                                commit_payload = commit_payload["result"]
                        except (TypeError, json.JSONDecodeError):
                            commit_payload = {}
                        if commit_payload.get("ok"):
                            from hazardweaver.hwa.experiments.unified_e12_agent_nudge_v1 import (
                                count_controller_route_commits,
                                maybe_terminalize_e12_over_commit_spin,
                            )

                            n_e12_commits = count_controller_route_commits(env.workdir)
                            if maybe_terminalize_e12_over_commit_spin(
                                env, task, n_commits=n_e12_commits
                            ):
                                exit_reason = "submitted"
                                pre_controller_terminal = True
                                env.append_transcript(
                                    {
                                        "role": "system",
                                        "event": "e12_over_commit_spin_terminal",
                                        "n_commits": n_e12_commits,
                                        "step": n_steps,
                                    }
                                )
                                break

                if pre_controller_guard is not None:
                    from hazardweaver.hwa.experiments.unified_solve_mandatory_pre_controller_v1 import (
                        maybe_terminalize_list_inventory_pre_controller_spin,
                        record_pre_controller_tool,
                    )

                    obs_ok: Optional[bool] = None
                    if name == "run_capability":
                        try:
                            obs_payload = json.loads(obs.get("content") or "{}")
                            if isinstance(obs_payload.get("result"), dict):
                                obs_payload = obs_payload["result"]
                            obs_ok = bool(obs_payload.get("ok"))
                        except (TypeError, json.JSONDecodeError):
                            obs_ok = False
                    record_pre_controller_tool(
                        pre_controller_guard, name, observation_ok=obs_ok
                    )
                    if maybe_terminalize_list_inventory_pre_controller_spin(
                        env,
                        task,
                        pre_controller_guard,
                        use_controller=use_controller,
                        step_n=n_steps,
                    ):
                        exit_reason = "submitted"
                        pre_controller_terminal = True
                        break

                if name in {"ask_user", "clarify_slot"}:
                    try:
                        payload = json.loads(obs.get("content") or "{}")
                        if isinstance(payload.get("result"), dict):
                            payload = payload["result"]
                    except (TypeError, json.JSONDecodeError):
                        payload = {}
                    status = str(payload.get("status") or "")
                    if status == "irrelevant":
                        irrelevant_ask_streak += 1
                        last_ask_slots = [str(s) for s in (payload.get("asked_slots") or payload.get("slots") or [])]
                        last_required_slots = [str(s) for s in (payload.get("required_slots") or [])]
                        if irrelevant_ask_streak >= 2 and last_required_slots:
                            nudge = {
                                "role": "user",
                                "content": (
                                    "ask_user returned irrelevant. Call ask_user once more with "
                                    f"slots={last_required_slots} exactly (copy required_slots), "
                                    "then call submit_clarification with that slot_id. "
                                    "Do not repeat the previous wrong slot."
                                ),
                            }
                            messages.append(nudge)
                            env.append_transcript(
                                {
                                    "role": "system",
                                    "event": "clarify_irrelevant_nudge",
                                    "required_slots": last_required_slots,
                                    "step": n_steps,
                                }
                            )
                            irrelevant_ask_streak = 0
                    elif status == "answered":
                        irrelevant_ask_streak = 0
                        resolution_answered = True
                        ans = payload.get("answers") or {}
                        slot_hint = (
                            (payload.get("matched_required_slots") or [None])[0]
                            or (payload.get("matched_slots") or [None])[0]
                            or (list(ans.keys())[0] if ans else None)
                        )
                        if slot_hint:
                            last_required_slots = [str(slot_hint)]
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "User answered. Now call submit_clarification exactly once "
                                    f"with slot_id='{slot_hint}' and a short question. "
                                    "Do not call ask_user again."
                                ),
                            }
                        )
                        env.append_transcript(
                            {
                                "role": "system",
                                "event": "clarify_answered_nudge_submit",
                                "slot_id": slot_hint,
                                "step": n_steps,
                            }
                        )
                    else:
                        irrelevant_ask_streak = 0

                if env.has_pending_clarifications():
                    if clarify_pending_since is None:
                        clarify_pending_since = n_steps
                else:
                    clarify_pending_since = None
                if env.submitted:
                    exit_reason = "submitted"
                    break
            if pre_controller_terminal or env.submitted:
                break

        if (
            exit_reason in {"limit_steps", "limit_wall", "unknown"}
            and (not env.submitted)
            and env.has_pending_clarifications()
        ):
            mode = str(getattr(env, "clarify_mode", None) or task.get("clarify_mode") or "")
            has_scripted = bool(getattr(env, "scripted_user", None) or task.get("scripted_user"))
            if mode == "decision" or (mode != "resolution" and not has_scripted):
                exit_reason = "env_null_no_user"
            else:
                exit_reason = "failed_clarify"

        # Emit a minimal answer.json on unclean exit so audit/scorer can classify IF/wrong
        # rather than hard-fail the whole SLURM job (missing_answer_json → afterok break).
        answer_path = env.workdir / "answer.json"
        if (not env.submitted) and (not answer_path.is_file()):
            slot = (last_required_slots or last_ask_slots or ["unspecified"])[0]
            emit = {
                "answer": {
                    "action": "clarify",
                    "slot_id": slot,
                    "clarify_slot": slot,
                    "question": f"unterminated_after_{exit_reason}",
                },
                "rationale": f"loop_auto_emit:{exit_reason}",
                "model_id_used": getattr(self.llm, "model_id", None),
                "emit_source": "agent_loop_unterminated",
                "resolution_answered": resolution_answered,
            }
            answer_path.write_text(json.dumps(emit, indent=2) + "\n", encoding="utf-8")
            env.append_transcript(
                {
                    "role": "system",
                    "event": "auto_emit_answer_json",
                    "exit_reason": exit_reason,
                    "slot_id": slot,
                }
            )

        if not answer_path.is_file():
            answer_path = None

        if (not env.submitted) and str(exit_reason or "") in {
            "limit_steps",
            "limit_wall",
            "parse_loop",
            "failed_clarify",
            "env_null_no_user",
        }:
            from hazardweaver.hwa.experiments.ablation_manual_pilot_terminal_v1 import (
                maybe_terminalize_unsubmitted_spin,
            )

            if maybe_terminalize_unsubmitted_spin(env, task, exit_reason):
                exit_reason = "submitted"
                answer_path = env.workdir / "answer.json"

        meta = {
            "task_id": env.task_id,
            "domain": task.get("domain"),
            "task_layer": task.get("task_layer"),
            "task_family": task.get("task_family"),
            "llm_provider": _arm_provider(self.llm),
            "transport": getattr(self.llm, "transport", None) or "vllm",
            "model_id": getattr(self.llm, "model_id", None),
            "base_url": getattr(getattr(self.llm, "config", None), "base_url", None),
            "profile": getattr(getattr(self.llm, "config", None), "profile", None),
            "n_steps": n_steps,
            "n_tool_calls": env.n_tool_calls,
            "exit_reason": exit_reason,
            "submitted": env.submitted,
            "pending_clarify": env.has_pending_clarifications(),
            "include_optional_tools": self.include_optional_tools,
            "wall_s": round(time.monotonic() - t0, 3),
            "max_steps": self.limits.max_steps,
            "max_wall_s": self.limits.max_wall_s,
            "clarify_grace_steps": self.limits.clarify_grace_steps,
            "pack_root": str(pack_root),
            "pfdf_oracle_burn": pfdf_oracle_burn
            if str(task.get("domain") or "").lower() == "pfdf"
            else None,
            "controller_mode": use_controller,
            "controller_checkpoints": controller.state.checkpoint if controller else 0,
            "n_controller_decisions": controller.n_decisions() if controller else 0,
            "route_id": getattr(controller.state, "active_route_id", None) if controller else None,
            "lease_id": (
                (getattr(controller.state, "active_lease", None) or {}).get("lease_id")
                if controller and isinstance(getattr(controller.state, "active_lease", None), dict)
                else None
            ),
            "segment_id": getattr(controller.state, "segment_id", None) if controller else None,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        from hazardweaver.hwa.experiments.headline_hkc_mode_v1 import headline_run_provenance

        meta.update(headline_run_provenance(task, controller))
        try:
            from hazardweaver.hwa.control.mode import is_vce_mode
            from hazardweaver.hwa.experiments.agent_strict_v2 import headline_control_mode_label

            if is_vce_mode():
                meta["control_mode"] = headline_control_mode_label()
        except ImportError:
            pass
        try:
            from hazardweaver.hwa.route_controller.fl2_pilot_slice import fl2_slice_metadata

            if is_fl2_pilot_task(task):
                meta["fl2_pilot_slice"] = fl2_slice_metadata(task)
        except Exception:  # noqa: BLE001
            pass
        if env.submitted and answer_path and answer_path.is_file():
            from hazardweaver.hwa.runtime.terminal_action import audit_submitted_terminal

            audit = audit_submitted_terminal(env.workdir, submitted=True)
            meta["terminal_audit"] = audit
            if not audit.get("ok"):
                exit_reason = str(audit.get("exit_reason") or exit_reason)
                meta["exit_reason"] = exit_reason
        try:
            from hazardweaver.hwa.experiments.ablation_manual_pilot_integrity_v1 import finalize_ablation_pilot_run

            env.submitted, exit_reason, ablation_integrity = finalize_ablation_pilot_run(
                env.workdir,
                task,
                submitted=bool(env.submitted),
                exit_reason=str(exit_reason or ""),
            )
            if ablation_integrity:
                meta["ablation_submission_integrity"] = ablation_integrity
            meta["submitted"] = bool(env.submitted)
            meta["exit_reason"] = exit_reason
        except ImportError:
            pass
        (env.workdir / "run_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        answer_body: Dict[str, Any] = {}
        if answer_path and answer_path.is_file():
            raw_ans = json.loads(answer_path.read_text(encoding="utf-8"))
            body = raw_ans.get("answer")
            answer_body = dict(body) if isinstance(body, Mapping) else dict(raw_ans)
        terminal = str(answer_body.get("action") or "")
        if terminal == "solve":
            from hazardweaver.hwa.runtime.track_native_metrics import patch_headline_workdir_metrics_if_needed

            patch_headline_workdir_metrics_if_needed(env.workdir, answer_body)
        emit_trajectory_v2(
            env.workdir,
            task_id=env.task_id,
            route_id=str(meta.get("route_id") or answer_body.get("route_id") or ""),
            lease_id=str(meta.get("lease_id") or ""),
            terminal_action=terminal,
        )
        return AgentRunResult(
            task_id=env.task_id,
            workdir=env.workdir,
            exit_reason=exit_reason,
            n_steps=n_steps,
            n_tool_calls=env.n_tool_calls,
            submitted=env.submitted,
            answer_path=answer_path,
            run_meta=meta,
        )
