"""PFDF water–fire tool environment (DS6 production: HCG + ask_user)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwa.agent_runtime.env import GoldLeakageError
from hazardweaver.hwa.agent_runtime.optional_tools import suggest_plan, validate_plan
from hazardweaver.hwa.agent_runtime.prompts import truncate_obs
from hazardweaver.hwa.agent_runtime.tool_specs import OPTIONAL_PLAN_TOOL_NAMES
from hazardweaver.hwa.scientific_controller.tool_policy import CONTROLLER_EXEC_TOOLS
from hazardweaver.hwa.pfdf_agent.data_access import PACK_ROOT
from hazardweaver.hwa.pfdf_agent.tools import PfdfAgentTools


class PfdfToolEnv:
    """Execute PFDF pack tools into a per-task workdir (burn→volume + HCG)."""

    def __init__(
        self,
        task: Mapping[str, Any],
        *,
        workdir: Path,
        pack_root: Path | None = None,
        tools: PfdfAgentTools | None = None,
        max_obs_chars: int = 8000,
        default_oracle_burn: bool = True,
        device: str = "cpu",
        require_checkpoint: bool = False,
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
        self.default_oracle_burn = bool(default_oracle_burn)
        self.tools = tools or PfdfAgentTools(
            pack_root=self.pack_root,
            default_submit_dir=self.workdir,
            clarify_dir=self.workdir,
            device=device,
            require_checkpoint=require_checkpoint,
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
        except Exception as exc:  # noqa: BLE001
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

    def _normalize_args(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Map AgentBench sample_id → PFDF record_id; inject default oracle flags."""
        clean = {
            k: v
            for k, v in args.items()
            if k not in {"out_dir", "task_id", "dataset_id", "split"}
        }
        if "record_id" not in clean and "sample_id" in clean:
            clean["record_id"] = clean.pop("sample_id")
        elif "sample_id" in clean and "record_id" in clean:
            clean.pop("sample_id", None)

        if name == "run_burn_predictor" and "oracle" not in clean:
            clean["oracle"] = self.default_oracle_burn
        if name == "compose_burn_to_volume" and "oracle_burn" not in clean:
            clean["oracle_burn"] = self.default_oracle_burn
        return clean

    def _dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_allows_run_capability

        if self.controller is not None and name in CONTROLLER_EXEC_TOOLS:
            if not (name == "run_capability" and agent_strict_v2_allows_run_capability()):
                return {
                    "ok": False,
                    "error": "direct_execution_forbidden",
                    "tool": name,
                    "message": (
                        "Capability execution must go through Deterministic Scientific Controller."
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
            return result

        if name in {"submit_answer", "submit"}:
            from hazardweaver.hwa.experiments.agent_strict_v2 import (
                agent_strict_v2_enabled,
                is_strict_forbidden_terminal,
                strict_terminal_block_payload,
            )

            if agent_strict_v2_enabled() and is_strict_forbidden_terminal(name):
                return strict_terminal_block_payload(name)
            # G1: submit ↔ submit_answer alias (adversarial schema uses submit).
            from hazardweaver.hwa.agent_runtime.unified_tools import normalize_legacy_answer
            from hazardweaver.hwa.runtime.terminal_action import (
                coerce_answer_body,
                terminal_from_answer_record,
            )

            answer, repairs = normalize_legacy_answer(args.get("answer"))
            if isinstance(answer, Mapping):
                answer = coerce_answer_body(answer)
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
                        json.dumps(
                            {"status": "answered", "answers": su.get("answers") or {}},
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                elif su.get("status") in {"refused", "irrelevant"}:
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

        clean = self._normalize_args(name, args)
        result = self.tools.call(name, **clean)
        if name == "run_burn_predictor" and isinstance(result, dict) and result.get("ok"):
            rid = str(clean.get("record_id") or "")
            result = self._apply_mh1_burn_preconditioning(result, record_id=rid)
        return result

    def _mh1_preconditioning_factor(self) -> Optional[float]:
        inputs = (self.task.get("solver_visible") or {}).get("inputs") or {}
        pre = inputs.get("mh1_burn_preconditioning") or {}
        if not isinstance(pre, Mapping) or not pre:
            return None
        if pre.get("operator_id") != "scale_fraction_mod_high":
            return None
        fac = pre.get("factor")
        return float(fac) if fac is not None else None

    def _apply_mh1_burn_preconditioning(
        self,
        burn_result: Mapping[str, Any],
        *,
        record_id: str,
    ) -> Dict[str, Any]:
        factor = self._mh1_preconditioning_factor()
        if factor is None or abs(factor - 1.0) < 1e-9:
            return dict(burn_result)
        from copy import deepcopy

        from models.burn_state_net.landsat_adapter import modhigh50_proxy_km2

        out = dict(burn_result)
        ws = deepcopy(dict(out.get("watershed_summary") or {}))
        old = float(ws.get("fraction_mod_high") or 0.0)
        ws["fraction_mod_high"] = max(0.0, min(1.0, old * factor))
        ws["modhigh50_km2_proxy"] = modhigh50_proxy_km2(
            self.tools.data.get_record(record_id), ws
        )
        out["watershed_summary"] = ws
        out["mh1_preconditioning_applied"] = {
            "operator_id": "scale_fraction_mod_high",
            "factor": factor,
        }
        return out

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
