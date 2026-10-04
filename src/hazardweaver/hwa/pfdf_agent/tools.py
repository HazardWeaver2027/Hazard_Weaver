"""HW PFDF water–fire agent tools — ACI surface. Forbidden: pyhazards imports.

DS6: HCG + ask_user aligned with wildfire pack. Label protocol:
mechanism_derived_oracle (inventory burn) + formula_oracle (Gorr volume).
Never publish as \"synthetic dataset\".
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hazardweaver.hwa.pfdf_agent.data_access import PACK_ROOT, PROJECT_ROOT, PfdfDataAccess
from models.burn_state_net.landsat_adapter import modhigh50_proxy_km2

assert "pyhazards" not in globals()

BURN_CAPABILITY = "burn_state_net_prithvi_v1"
VOLUME_CAPABILITY = "pfdf_volume_gorr_v2"


def _hcg_bridge():
    """Lazy import avoids circular: tools ↔ agent_runtime.env via package __init__."""
    from hazardweaver.hwa.agent_runtime import hcg_tools as hcg_bridge

    return hcg_bridge


class PfdfAgentTools:
    """Concrete implementations matching configs/pfdf_waterfire_agent_pack/tools.json."""

    def __init__(
        self,
        *,
        pack_root: Path | None = None,
        data: PfdfDataAccess | None = None,
        default_submit_dir: Path | None = None,
        device: str | None = None,
        require_checkpoint: bool = True,
        clarify_dir: Path | None = None,
    ):
        self.pack_root = Path(pack_root or PACK_ROOT)
        self.inventory = json.loads((self.pack_root / "inventory.json").read_text(encoding="utf-8"))
        self.tools_spec = json.loads((self.pack_root / "tools.json").read_text(encoding="utf-8"))
        self.data = data or PfdfDataAccess()
        self.default_submit_dir = Path(
            default_submit_dir
            or (PROJECT_ROOT / "runs" / "hw" / "agent" / "_pfdf_tool_smoke")
        )
        self.clarify_dir = Path(clarify_dir) if clarify_dir else self.default_submit_dir
        self.device = device or os.environ.get("HW_PFDF_DEVICE", "cpu")
        self.require_checkpoint = require_checkpoint
        self._loader = None

    def _capability_loader(self):
        if self._loader is None:
            from hazardweaver.hwa.capabilities.loader import CapabilityLoader

            self._loader = CapabilityLoader(
                require_checkpoint=self.require_checkpoint,
                device=self.device,
            )
        return self._loader

    def list_inventory(self, kind: str = "all") -> Dict[str, Any]:
        kind = (kind or "all").strip().lower()
        aliases = {
            "capabilities": "models",
            "capability": "models",
            "cap": "models",
        }
        kind = aliases.get(kind, kind)
        if kind not in {"datasets", "models", "tasks", "tools", "all"}:
            raise ValueError(f"invalid kind: {kind}")
        out: Dict[str, Any] = {"kind": kind, "pack_id": self.inventory.get("pack_id")}
        if kind in {"datasets", "all"}:
            out["datasets"] = [
                {
                    "dataset_id": d["dataset_id"],
                    "task_families": d["task_families"],
                    "status": d["status"],
                    "n_records": d.get("n_records"),
                }
                for d in self.inventory["datasets"]
            ]
        if kind in {"models", "all"}:
            out["models"] = [
                {
                    "model_id": m["model_id"],
                    "task_families": m["task_families"],
                    "inference_only": m.get("inference_only", True),
                    "status": m["status"],
                    "kind": m.get("kind", "predictive_model"),
                }
                for m in self.inventory["models"]
            ]
        if kind in {"tasks", "all"}:
            out["tasks"] = [
                {"task_id": t["task_id"], "status": t["status"]} for t in self.inventory["tasks"]
            ]
        if kind in {"tools", "all"}:
            out["tools"] = [t["name"] for t in self.tools_spec.get("tools", [])]
        if kind == "all":
            out["cascade"] = self.inventory.get("cascade")
            out["label_protocol"] = self.tools_spec.get("label_protocol")
            pse = self.pack_root / "pseudo_kg" / "edges.json"
            if pse.is_file():
                out["pseudo_kg_edges"] = json.loads(pse.read_text(encoding="utf-8"))
        return out

    def read_card(self, card_path: str) -> Dict[str, Any]:
        rel = Path(card_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("card_path must be a relative path under the PFDF agent pack")
        path = (self.pack_root / rel).resolve()
        pack_resolved = self.pack_root.resolve()
        if pack_resolved not in path.parents and path != pack_resolved:
            raise ValueError("card_path escapes pack root")
        if not path.is_file():
            raise FileNotFoundError(f"card not found: {card_path}")
        return {"card_path": card_path, "text": path.read_text(encoding="utf-8")}

    def load_sample(
        self,
        record_id: str,
        *,
        include_volume_target: bool = False,
    ) -> Dict[str, Any]:
        return self.data.load_sample(
            record_id, include_volume_target=bool(include_volume_target)
        )

    def run_burn_predictor(
        self,
        record_id: str,
        *,
        oracle: bool = False,
    ) -> Dict[str, Any]:
        record = self.data.get_record(record_id)
        # Oracle path must NOT instantiate Prithvi (timm/HF download). Inventory fields only.
        if oracle:
            ws = {
                "mean_dnbr": float(record.get("MeandNBR") or 0.0),
                "fraction_mod_high": float(record.get("FractionModHigh") or 0.0),
                "fraction_burned": float(record.get("FractionBurned") or 0.0),
            }
            ws["modhigh50_km2_proxy"] = modhigh50_proxy_km2(record, ws)
            return {
                "ok": True,
                "record_id": record_id,
                "model_id": BURN_CAPABILITY,
                "oracle": True,
                "label_protocol": "mechanism_derived_oracle",
                "watershed_summary": ws,
                "modhigh50_km2_proxy": ws.get("modhigh50_km2_proxy"),
            }
        pred = self._capability_loader().load(BURN_CAPABILITY)
        out = pred.predict(record, oracle=False)
        ws = out.get("watershed_summary", {})
        return {
            "ok": bool(out.get("valid", True)),
            "record_id": record_id,
            "model_id": BURN_CAPABILITY,
            "oracle": False,
            "watershed_summary": ws,
            "modhigh50_km2_proxy": ws.get("modhigh50_km2_proxy")
            if ws
            else None,
        }

    def run_volume_predictor(
        self,
        record_id: str,
        *,
        burn_summary: Optional[Dict[str, Any]] = None,
        record_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = dict(self.data.get_record(record_id))
        if record_overrides:
            record.update(record_overrides)
        if burn_summary:
            record["MeandNBR"] = burn_summary.get("mean_dnbr", record.get("MeandNBR"))
            record["FractionModHigh"] = burn_summary.get(
                "fraction_mod_high", record.get("FractionModHigh")
            )
            record["FractionBurned"] = burn_summary.get(
                "fraction_burned", record.get("FractionBurned")
            )
            if burn_summary.get("modhigh50_km2_proxy") is not None:
                record["ModHigh50_km2"] = burn_summary["modhigh50_km2_proxy"]
            else:
                record["ModHigh50_km2"] = modhigh50_proxy_km2(record, burn_summary)
        pred = self._capability_loader().load(VOLUME_CAPABILITY)
        out = pred.predict(record)
        return {
            "ok": bool(out.get("valid", True)),
            "record_id": record_id,
            "model_id": VOLUME_CAPABILITY,
            "label_protocol": "formula_oracle",
            "log_volume": out.get("log_volume"),
            "ModHigh50_km2_used": record.get("ModHigh50_km2"),
            "raw": {k: out[k] for k in out if k not in {"logits"} and _is_scalarish(out[k])},
        }

    def compose_burn_to_volume(
        self,
        record_id: str,
        *,
        oracle_burn: bool = False,
    ) -> Dict[str, Any]:
        # Oracle burn: inventory → Gorr only (no Prithvi / WorkflowExecutor burn load).
        if oracle_burn:
            burn = self.run_burn_predictor(record_id, oracle=True)
            vol = self.run_volume_predictor(
                record_id, burn_summary=burn.get("watershed_summary")
            )
            return {
                "ok": bool(vol.get("ok")) and vol.get("log_volume") is not None,
                "record_id": record_id,
                "burn_capability_id": BURN_CAPABILITY,
                "volume_capability_id": VOLUME_CAPABILITY,
                "oracle_burn": True,
                "label_protocol": {
                    "burn": "mechanism_derived_oracle",
                    "volume": "formula_oracle",
                },
                "log_volume": vol.get("log_volume"),
                "abstained": False,
                "burn_summary": burn.get("watershed_summary"),
                "workflow_id": "compose_mechanism_derived_burn_formula_volume",
                "timings_sec": {},
            }

        from hazardweaver.hwa.workflow.executor import WorkflowCompiler, WorkflowExecutor

        record = self.data.get_record(record_id)
        ex = WorkflowExecutor(
            require_checkpoint=self.require_checkpoint,
            device=self.device,
        )
        wf = WorkflowCompiler().compile_pfdf_volume_workflow(
            str(record_id),
            use_oracle_burn=False,
            burn_capability_id=BURN_CAPABILITY,
            volume_capability_id=VOLUME_CAPABILITY,
        )
        wf = ex.execute(wf, record)
        burn_sum = None
        if wf.trace and wf.trace.node_outputs:
            burn_sum = wf.trace.node_outputs.get("burn_state")
        return {
            "ok": not bool(wf.abstained) and wf.prediction is not None,
            "record_id": record_id,
            "burn_capability_id": BURN_CAPABILITY,
            "volume_capability_id": VOLUME_CAPABILITY,
            "oracle_burn": False,
            "label_protocol": {
                "burn": "live_predictor",
                "volume": "formula_oracle",
            },
            "log_volume": wf.prediction,
            "abstained": bool(wf.abstained),
            "burn_summary": burn_sum,
            "workflow_id": wf.workflow_id,
            "timings_sec": dict(wf.trace.timings_sec) if wf.trace else {},
        }

    def tool_hcg_find_paths(
        self,
        sources: Sequence[str],
        target: str,
        *,
        packs: Optional[Sequence[str]] = None,
        max_paths: int = 5,
        executable_only: bool = True,
        require_all_sources: bool = True,
    ) -> Dict[str, Any]:
        packs_list = list(packs) if packs else ["pfdf_v1_held"]
        paths = _hcg_bridge().tool_hcg_find_paths(
            list(sources),
            target,
            packs=packs_list,
            max_paths=int(max_paths),
            executable_only=bool(executable_only),
            require_all_sources=bool(require_all_sources),
        )
        return {
            "ok": True,
            "sources": list(sources),
            "target": target,
            "packs": packs_list,
            "executable_only": bool(executable_only),
            "require_all_sources": bool(require_all_sources),
            "n_paths": len(paths),
            "paths": paths,
        }

    def tool_hcg_run_path(
        self,
        sources: Sequence[str],
        target: str,
        *,
        packs: Optional[Sequence[str]] = None,
        require_all_sources: bool = True,
        fixture: str = "wf_hard",
        edge_ids: Optional[Sequence[str]] = None,
        out_dir: Path | None = None,
    ) -> Dict[str, Any]:
        packs_list = list(packs) if packs else ["pfdf_v1_held"]
        _ = edge_ids
        result = _hcg_bridge().tool_hcg_run_path(
            list(sources),
            target,
            packs=packs_list,
            require_all_sources=bool(require_all_sources),
            fixture=fixture or "wf_hard",
            host=self,
            out_dir=out_dir or self.default_submit_dir,
        )
        out = {
            "ok": bool(result.get("ok")),
            "sources": list(sources),
            "target": target,
            "packs": packs_list,
            "fixture": fixture or "wf_hard",
            "edges": result.get("edges") or [],
            "outputs": result.get("answer") or {},
            "abstain_reason": result.get("abstain_reason"),
        }
        for k in (
            "execution_id",
            "final_artifact_id",
            "route_id",
            "route_candidate",
            "execution_result",
            "execution_persisted",
        ):
            if k in result:
                out[k] = result[k]
        return out

    def get_execution_result(self, execution_id: str, **kwargs: Any) -> Dict[str, Any]:
        from hazardweaver.hwa.agent_runtime.unified_tools import get_execution_result as _fn

        return _fn(
            execution_id,
            host=self,
            out_dir=kwargs.get("out_dir") or self.default_submit_dir,
        )

    def tool_hcg_explain_edge(self, edge_id: str) -> Dict[str, Any]:
        detail = _hcg_bridge().tool_hcg_explain_edge(edge_id)
        return {"ok": True, "edge_id": edge_id, "detail": detail}

    def ask_user(
        self,
        question: str,
        slots: Sequence[str],
        *,
        out_dir: Path | None = None,
    ) -> Dict[str, Any]:
        """Clarify slots. Optional reply file: clarifications_reply.json in out_dir."""
        out_dir = Path(out_dir or self.clarify_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "clarifications.jsonl"
        reply_path = out_dir / "clarifications_reply.json"
        reply: Dict[str, Any] | None = None
        if reply_path.is_file():
            try:
                reply = json.loads(reply_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                reply = None
        if reply and reply.get("status") in {"answered", "ok"}:
            row = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "question": question,
                "slots": list(slots),
                "status": "answered",
                "reply": reply.get("answers") or reply.get("reply") or reply,
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            return {
                "ok": True,
                "question": question,
                "slots": list(slots),
                "status": "answered",
                "answers": row["reply"],
                "clarifications_path": str(path),
            }
        row = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "slots": list(slots),
            "status": "pending_user",
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {
            "ok": True,
            "question": question,
            "slots": list(slots),
            "status": "pending_user",
            "clarifications_path": str(path),
            "note": (
                "No clarifications_reply.json yet; agent must wait, use limited "
                "assumptions, or fail honestly — do not invent slot values."
            ),
        }

    def submit_answer(
        self,
        answer: Any,
        *,
        rationale: Optional[str] = None,
        model_id_used: Optional[str] = None,
        out_dir: Path | None = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        out_dir = Path(out_dir or self.default_submit_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": task_id,
            "answer": answer,
            "rationale": rationale,
            "model_id_used": model_id_used,
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
        }
        path = out_dir / "answer.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return {"ok": True, "answer_path": str(path), **payload}

    def run_capability(
        self,
        capability_id: str,
        record: Any = None,
        handles: Any = None,
        artifact_refs: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from hazardweaver.hwa.agent_runtime.tool_host import resolve_agent_host
        from hazardweaver.hwa.agent_runtime.unified_tools import run_capability as _run_cap

        return _run_cap(
            capability_id,
            host=resolve_agent_host(self),
            record=record,
            handles=handles,
            artifact_refs=artifact_refs,
            **kwargs,
        )

    def inspect_artifact(self, artifact_id: str, **kwargs: Any) -> Dict[str, Any]:
        from hazardweaver.hwa.agent_runtime.unified_tools import inspect_artifact as _inspect

        return _inspect(artifact_id, host=self, **kwargs)

    def _submit_solution(self, **kwargs: Any) -> Dict[str, Any]:
        from hazardweaver.hwa.agent_runtime.unified_tools import submit_solution as _fn

        return _fn(host=self, **kwargs)

    def _submit_clarification(self, **kwargs: Any) -> Dict[str, Any]:
        from hazardweaver.hwa.agent_runtime.unified_tools import submit_clarification as _fn

        return _fn(host=self, **kwargs)

    def _submit_abstention(self, **kwargs: Any) -> Dict[str, Any]:
        from hazardweaver.hwa.agent_runtime.unified_tools import submit_abstention as _fn

        return _fn(host=self, **kwargs)

    def call(self, name: str, **kwargs) -> Dict[str, Any]:
        from hazardweaver.hwa.agent_runtime.unified_tools import normalize_tool_name

        name = normalize_tool_name(name)
        handlers = {
            "list_inventory": self.list_inventory,
            "read_card": self.read_card,
            "load_sample": self.load_sample,
            "run_burn_predictor": self.run_burn_predictor,
            "run_volume_predictor": self.run_volume_predictor,
            "compose_burn_to_volume": self.compose_burn_to_volume,
            "run_capability": self.run_capability,
            "inspect_artifact": self.inspect_artifact,
            "tool_hcg_find_paths": self.tool_hcg_find_paths,
            "tool_hcg_run_path": self.tool_hcg_run_path,
            "tool_hcg_explain_edge": self.tool_hcg_explain_edge,
            "ask_user": self.ask_user,
            "submit_answer": self.submit_answer,
            "submit_solution": self._submit_solution,
            "submit_clarification": self._submit_clarification,
            "submit_abstention": self._submit_abstention,
            "get_execution_result": self.get_execution_result,
        }
        if name not in handlers:
            raise KeyError(f"unknown tool: {name}")
        args, warnings = _normalize_pfdf_tool_kwargs(name, dict(kwargs))
        needs_record = name in {
            "load_sample",
            "run_burn_predictor",
            "run_volume_predictor",
            "compose_burn_to_volume",
        }
        if needs_record and not args.get("record_id"):
            return {
                "ok": False,
                "error": "missing required argument: record_id (alias: sample_id)",
                "tool": name,
                "warnings": warnings,
            }
        try:
            out = handlers[name](**args)
        except TypeError as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "tool": name,
                "warnings": warnings,
            }
        if warnings and isinstance(out, dict):
            prev = list(out.get("warnings") or [])
            out = {**out, "warnings": prev + warnings}
        return out


def _normalize_pfdf_tool_kwargs(
    name: str, kwargs: Dict[str, Any]
) -> tuple[Dict[str, Any], List[str]]:
    """Map common LLM aliases; drop unknown kwargs that would TypeError."""
    warnings: List[str] = []
    args = dict(kwargs)
    if name in {"run_capability", "inspect_artifact"}:
        # Unified tools keep sample_id / dataset_id / model_id for dispatch.
        return args, warnings
    if "record_id" not in args and args.get("sample_id") is not None:
        args["record_id"] = args.pop("sample_id")
        warnings.append("aliased sample_id→record_id")
    elif "sample_id" in args:
        args.pop("sample_id", None)
    if name == "run_volume_predictor":
        if "burn_summary" not in args and args.get("burn_output") is not None:
            args["burn_summary"] = args.pop("burn_output")
            warnings.append("aliased burn_output→burn_summary")
        elif "burn_output" in args:
            args.pop("burn_output", None)
    for junk in ("model_id", "dataset_id"):
        if junk in args:
            args.pop(junk)
            warnings.append(f"ignored unexpected kwarg {junk}")
    return args, warnings


def _is_scalarish(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))
