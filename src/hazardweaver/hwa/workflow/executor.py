"""Deterministic workflow compiler and executor."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Set

from hazardweaver.hwa.capabilities.loader import CapabilityLoader
from hazardweaver.hwa.contracts import (
    AdapterRegistry,
    ValidationResult,
    WorkflowEdge,
    WorkflowNode,
    WorkflowSpec,
    WorkflowTrace,
)
from hazardweaver.hwa.validators import WorkflowValidator
from hazardweaver.hwa.workflow.adapter_runner import AdapterRunner, is_adapter_node
from hazardweaver.hwa.workflow.execution_order import node_by_id, topological_node_order

from models.burn_state_net.landsat_adapter import modhigh50_proxy_km2

DEFAULT_BURN_CAPABILITY = "burn_state_net_prithvi_v1"
DEFAULT_VOLUME_CAPABILITY = "pfdf_volume_gorr_v2"
DEFAULT_ADAPTER_PATH = Path("configs/adapters/pfdf_adapters.yaml")


def resolve_default_burn_capability() -> str:
    """Prefer primary Prithvi burn when checkpoint exists."""
    if Path("checkpoints/burn_state_net_prithvi_v1/best.pt").exists():
        return "burn_state_net_prithvi_v1"
    if Path("checkpoints/burn_state_net_aspp_v1/best.pt").exists():
        return "burn_state_net_aspp_v1"
    return "burn_state_net_tabular_v1"


class WorkflowCompiler:
    """Build WorkflowSpec from request + capability IDs."""

    def compile_pfdf_volume_workflow(
        self,
        request_id: str,
        *,
        use_oracle_burn: bool = False,
        burn_capability_id: Optional[str] = None,
        volume_capability_id: str = DEFAULT_VOLUME_CAPABILITY,
    ) -> WorkflowSpec:
        burn_capability_id = burn_capability_id or resolve_default_burn_capability()
        return WorkflowSpec(
            workflow_id=f"wf_{uuid.uuid4().hex[:8]}",
            request_id=request_id,
            interaction_spec_id="pfdf_cascade_v1",
            nodes=[
                WorkflowNode(
                    id="burn_state",
                    capability_id=burn_capability_id,
                ),
                WorkflowNode(
                    id="pfdf_volume",
                    capability_id=volume_capability_id,
                    params={"oracle_burn": use_oracle_burn},
                ),
            ],
            edges=[
                WorkflowEdge(
                    **{"from": "burn_state", "to": "pfdf_volume", "mapping": "basin_severity_summary"}
                ),
            ],
            provenance={"compiler": "deterministic_pfdf_v2"},
        )


class WorkflowExecutor:
    """Execute validated workflows via HWA CapabilityLoader (inference only)."""

    def __init__(
        self,
        loader: Optional[CapabilityLoader] = None,
        validator: Optional[WorkflowValidator] = None,
        *,
        require_checkpoint: bool = False,
        device: str = "cpu",
    ):
        self.loader = loader or CapabilityLoader(require_checkpoint=require_checkpoint, device=device)
        self.validator = validator or WorkflowValidator()

    def execute(
        self,
        workflow: WorkflowSpec,
        record: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowSpec:
        context = dict(context or {})
        context["record"] = record
        validation = self.validator.validate(workflow, context)
        workflow.validation = validation

        trace = WorkflowTrace()
        if self.validator.should_abstain(validation):
            workflow.abstained = True
            workflow.prediction = None
            workflow.trace = trace
            return workflow

        burn_node = next(n for n in workflow.nodes if n.id == "burn_state")
        vol_node = next(n for n in workflow.nodes if n.id == "pfdf_volume")

        t0 = time.perf_counter()
        oof_summary = context.get("oof_burn_summary")
        if oof_summary is not None:
            burn_out = {"watershed_summary": oof_summary, "valid": True, "model_id": "oof_burn_summary"}
        else:
            burn_pred = self.loader.load(burn_node.capability_id)
            burn_out = burn_pred.predict(record, oracle=vol_node.params.get("oracle_burn", False))
        trace.node_outputs["burn_state"] = burn_out.get("watershed_summary", burn_out)
        trace.timings_sec["burn_state"] = time.perf_counter() - t0

        merged = dict(record)
        ws = burn_out.get("watershed_summary", {})
        merged["MeandNBR"] = ws.get("mean_dnbr", merged.get("MeandNBR"))
        merged["FractionModHigh"] = ws.get("fraction_mod_high", merged.get("FractionModHigh"))
        merged["FractionBurned"] = ws.get("fraction_burned", merged.get("FractionBurned"))
        if ws.get("modhigh50_km2_proxy") is not None:
            merged["ModHigh50_km2"] = ws["modhigh50_km2_proxy"]
        elif vol_node.capability_id == "pfdf_volume_gorr_v2":
            merged["ModHigh50_km2"] = modhigh50_proxy_km2(merged, ws)

        t1 = time.perf_counter()
        vol_pred = self.loader.load(vol_node.capability_id)
        vol_out = vol_pred.predict(merged)
        trace.node_outputs["pfdf_volume"] = {"log_volume": vol_out.get("log_volume")}
        trace.timings_sec["pfdf_volume"] = time.perf_counter() - t1

        valid = burn_out.get("valid", True) and vol_out.get("valid", True)
        workflow.prediction = vol_out.get("log_volume") if valid else None
        workflow.abstained = not valid
        workflow.trace = trace
        return workflow

    def execute_compiled(
        self,
        workflow: WorkflowSpec,
        record: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
        adapter_registry: Optional[AdapterRegistry] = None,
    ) -> WorkflowSpec:
        """Execute a Phase 02 compiled workflow with explicit adapter nodes."""

        context = dict(context or {})
        context["record"] = record
        adapter_registry = adapter_registry or AdapterRegistry.load_yaml(DEFAULT_ADAPTER_PATH)
        adapter_ids: Set[str] = set(adapter_registry.list_ids())
        adapter_runner = AdapterRunner(adapter_registry)

        validation = self.validator.validate(workflow, context)
        workflow.validation = validation

        trace = WorkflowTrace()
        if self.validator.should_abstain(validation):
            workflow.abstained = True
            workflow.prediction = None
            workflow.trace = trace
            return workflow

        merged = dict(record)
        order = topological_node_order(workflow)
        nodes = node_by_id(workflow)
        burn_out: Optional[Dict[str, Any]] = None
        adapter_traces: Dict[str, Any] = {}

        for node_id in order:
            node = nodes[node_id]
            t0 = time.perf_counter()

            if is_adapter_node(node, adapter_ids):
                adapter_id = node.capability_id
                burn_summary = None
                if adapter_id == "pfdf_burn_summary_to_watershed_v1" and burn_out is not None:
                    burn_summary = burn_out.get("watershed_summary", burn_out)
                result = adapter_runner.apply(
                    adapter_id,
                    merged,
                    context=context,
                    burn_summary=burn_summary,
                )
                adapter_traces[node_id] = result.trace
                merged.update(result.patches)
                trace.node_outputs[node_id] = {
                    "adapter_id": adapter_id,
                    "valid": result.valid,
                    "patches": result.patches,
                    "quality_flags": result.quality_flags,
                }
                trace.timings_sec[node_id] = time.perf_counter() - t0
                if not result.valid and adapter_id != "pfdf_burn_summary_to_watershed_v1":
                    workflow.abstained = True
                    workflow.prediction = None
                    workflow.trace = trace
                    return workflow
                continue

            if node_id == "burn_state" or "burn_state" in node.capability_id:
                oof_summary = context.get("oof_burn_summary")
                if oof_summary is not None:
                    burn_out = {
                        "watershed_summary": oof_summary,
                        "valid": True,
                        "model_id": "oof_burn_summary",
                    }
                else:
                    burn_pred = self.loader.load(node.capability_id)
                    oracle = node.params.get("oracle_burn", False)
                    burn_out = burn_pred.predict(merged, oracle=oracle)
                ws = burn_out.get("watershed_summary", burn_out)
                trace.node_outputs[node_id] = ws
                trace.timings_sec[node_id] = time.perf_counter() - t0

                merged["MeandNBR"] = ws.get("mean_dnbr", merged.get("MeandNBR"))
                merged["FractionModHigh"] = ws.get("fraction_mod_high", merged.get("FractionModHigh"))
                merged["FractionBurned"] = ws.get("fraction_burned", merged.get("FractionBurned"))
                if ws.get("modhigh50_km2_proxy") is not None:
                    merged["ModHigh50_km2"] = ws["modhigh50_km2_proxy"]
                else:
                    merged["ModHigh50_km2"] = modhigh50_proxy_km2(merged, ws)
                continue

            if node_id == "pfdf_volume" or node.capability_id.startswith("pfdf_volume"):
                vol_pred = self.loader.load(node.capability_id)
                vol_out = vol_pred.predict(merged)
                trace.node_outputs[node_id] = {"log_volume": vol_out.get("log_volume")}
                trace.timings_sec[node_id] = time.perf_counter() - t0

                valid = (burn_out or {}).get("valid", True) and vol_out.get("valid", True)
                workflow.prediction = vol_out.get("log_volume") if valid else None
                workflow.abstained = not valid
                workflow.trace = trace
                workflow.provenance = {
                    **workflow.provenance,
                    "adapter_traces": adapter_traces,
                    "execution_mode": "compiled_dag_v1",
                }
                return workflow

        workflow.abstained = True
        workflow.prediction = None
        workflow.trace = trace
        return workflow
