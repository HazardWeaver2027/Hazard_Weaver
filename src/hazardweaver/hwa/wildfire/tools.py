"""HW wildfire agent tools — ACI surface. Forbidden: pyhazards imports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hazardweaver.hwa.wildfire.data_store import PACK_ROOT, PROJECT_ROOT, WildfireDataStore
from hazardweaver.hwa.wildfire.predictors import WildfirePredictorRegistry

# Hard guard: this module must never import pyhazards.
assert "pyhazards" not in globals()


def _hcg_bridge():
    """Lazy import avoids circular: tools ↔ agent_runtime.env via package __init__."""
    from hazardweaver.hwa.agent_runtime import hcg_tools as hcg_bridge

    return hcg_bridge


class WildfireAgentTools:
    """Concrete implementations matching configs/wildfire_agent_pack/tools.json."""

    def __init__(
        self,
        *,
        pack_root: Path | None = None,
        data_store: WildfireDataStore | None = None,
        predictors: WildfirePredictorRegistry | None = None,
        default_submit_dir: Path | None = None,
        clarify_dir: Path | None = None,
    ):
        self.pack_root = Path(pack_root or PACK_ROOT)
        self.inventory = json.loads((self.pack_root / "inventory.json").read_text(encoding="utf-8"))
        self.tools_spec = json.loads((self.pack_root / "tools.json").read_text(encoding="utf-8"))
        self.data_store = data_store or WildfireDataStore()
        self.predictors = predictors or WildfirePredictorRegistry()
        self.default_submit_dir = Path(
            default_submit_dir or (PROJECT_ROOT / "runs" / "hw" / "agent" / "_tool_smoke")
        )
        self.clarify_dir = Path(clarify_dir) if clarify_dir else self.default_submit_dir

    def list_inventory(self, kind: str = "all") -> Dict[str, Any]:
        kind = kind or "all"
        if kind not in {"datasets", "models", "tasks", "tools", "all"}:
            raise ValueError(f"invalid kind: {kind}")
        out: Dict[str, Any] = {"kind": kind}
        if kind in {"datasets", "all"}:
            out["datasets"] = [
                {
                    "dataset_id": d["dataset_id"],
                    "task_families": d["task_families"],
                    "status": d["status"],
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
                }
                for m in self.inventory["models"]
            ]
        if kind in {"tasks", "all"}:
            out["tasks"] = [
                {"task_id": t["task_id"], "status": t["status"]} for t in self.inventory["tasks"]
            ]
        if kind in {"tools", "all"}:
            out["tools"] = [t["name"] for t in self.tools_spec.get("tools", [])]
        return out

    def read_card(self, card_path: str) -> Dict[str, Any]:
        rel = Path(card_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("card_path must be a relative path under the wildfire agent pack")
        path = (self.pack_root / rel).resolve()
        pack_resolved = self.pack_root.resolve()
        if pack_resolved not in path.parents and path != pack_resolved:
            raise ValueError("card_path escapes pack root")
        if not path.is_file():
            raise FileNotFoundError(f"card not found: {card_path}")
        return {"card_path": card_path, "text": path.read_text(encoding="utf-8")}

    def load_sample(
        self,
        dataset_id: str,
        sample_id: str,
        split: Optional[str] = None,
    ) -> Dict[str, Any]:
        sample = self.data_store.load_sample(
            dataset_id, sample_id, split=split, include_label=False
        )
        features = sample["features"]
        return {
            "dataset_id": sample["dataset_id"],
            "sample_id": sample["sample_id"],
            "split": sample["split"],
            "index": sample["index"],
            "feature_shape": sample["feature_shape"],
            "dtype": sample["dtype"],
            "task_family": sample["task_family"],
            "schema": sample["schema"],
            "features": features.tolist(),
        }

    def run_predictor(
        self,
        model_id: str,
        dataset_id: str,
        sample_id: str,
        split: Optional[str] = None,
    ) -> Dict[str, Any]:
        sample = self.data_store.load_sample(
            dataset_id, sample_id, split=split, include_label=False
        )
        pred = self.predictors.predict(
            model_id,
            sample["features"],
            dataset_id=dataset_id,
        )
        out: Dict[str, Any] = {
            "model_id": pred["model_id"],
            "dataset_id": dataset_id,
            "sample_id": sample_id,
            "split": sample["split"],
            "task_family": pred["task_family"],
            "output_kind": pred["output_kind"],
            "logits_shape": pred["logits_shape"],
        }
        if "pred_class_id" in pred:
            out["pred_class_id"] = pred["pred_class_id"]
            out["probs"] = pred["probs"]
        if "prediction" in pred:
            out["prediction"] = pred["prediction"]
        if "burned_fraction" in pred:
            out["burned_fraction"] = pred["burned_fraction"]
            out["burned_pixels"] = pred["burned_pixels"]
            out["pred_mask_shape"] = list(pred["pred_mask"].shape)
        out["logits"] = pred["logits"].tolist()
        return out

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
        packs_list = list(packs) if packs else ["graph_eval_v0"]
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
        packs_list = list(packs) if packs else ["graph_eval_v0"]
        _ = edge_ids  # reserved for explicit-path selection in later phases
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
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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

    def call(self, name: str, **kwargs) -> Dict[str, Any]:
        from hazardweaver.hwa.agent_runtime.unified_tools import normalize_tool_name

        name = normalize_tool_name(name)
        handlers = {
            "list_inventory": self.list_inventory,
            "read_card": self.read_card,
            "load_sample": self.load_sample,
            "run_predictor": self.run_predictor,
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
        return handlers[name](**kwargs)

    def _submit_solution(self, **kwargs: Any) -> Dict[str, Any]:
        from hazardweaver.hwa.agent_runtime.unified_tools import submit_solution as _fn

        return _fn(host=self, **kwargs)

    def _submit_clarification(self, **kwargs: Any) -> Dict[str, Any]:
        from hazardweaver.hwa.agent_runtime.unified_tools import submit_clarification as _fn

        return _fn(host=self, **kwargs)

    def _submit_abstention(self, **kwargs: Any) -> Dict[str, Any]:
        from hazardweaver.hwa.agent_runtime.unified_tools import submit_abstention as _fn

        return _fn(host=self, **kwargs)
