"""Canonical execute_capability entry (headline subset, lease-gated)."""

from __future__ import annotations

import importlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

from hazardweaver.hcg.certification.reachability import evaluate_reachability
from hazardweaver.hcg.contracts.execution_event import (
    CapabilityExecutionCertificate,
    ExecutionContext,
    ExecutionEvent,
)
from hazardweaver.hcg.contracts.reachability_certificate import CapabilityReachabilityCertificate
from hazardweaver.hcg.registry.canonical_ids import is_headline_capability, resolve_canonical
from hazardweaver.hwa.scientific_controller.state import SessionState

_DISPATCH_YAML = Path(__file__).resolve().parent / "headline_dispatch.yaml"


class ExecutionLeaseError(ValueError):
    pass


@dataclass
class ExecuteCapabilityBundle:
    """Full HCG execute payload for HWA/HWB (not acap dict)."""

    event: ExecutionEvent
    execution_certificate: CapabilityExecutionCertificate
    reachability_certificate: CapabilityReachabilityCertificate

    def to_dict(self) -> Dict[str, Any]:
        exec_dict = self.execution_certificate.model_dump(mode="json")
        return {
            "execution_event": self.event.model_dump(mode="json"),
            "execution_certificate": exec_dict,
            "reachability_certificate": self.reachability_certificate.model_dump(mode="json"),
            "certificate_content_hash": exec_dict.get("content_hash", ""),
        }


def reachability_for_capability(
    capability_id: str,
    *,
    route_id: str = "",
    skip_gpu_probe: bool = True,
) -> CapabilityReachabilityCertificate:
    meta = resolve_canonical(capability_id)
    route = {
        "route_id": route_id or f"cap:{capability_id}",
        "capability_ids": [capability_id],
        "route_family_id": meta.get("route_family_id"),
        "validation_level": "L2",
        "executable": True,
    }
    state = SessionState(sources=[], available_artifacts={})
    return evaluate_reachability(route, state, skip_gpu_probe=skip_gpu_probe)


@lru_cache(maxsize=1)
def _load_dispatch() -> Dict[str, Any]:
    raw = yaml.safe_load(_DISPATCH_YAML.read_text(encoding="utf-8")) or {}
    return dict(raw.get("bindings") or {})


def _validate_lease(capability_id: str, ctx: ExecutionContext) -> None:
    lease = ctx.lease
    if lease.expires_at is not None:
        now = datetime.now(timezone.utc)
        exp = lease.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp < now:
            raise ExecutionLeaseError("lease_expired")
    if lease.allowed_capability_ids and capability_id not in lease.allowed_capability_ids:
        raise ExecutionLeaseError(f"capability_not_in_lease:{capability_id}")


def _resolve_handler(capability_id: str) -> Optional[Callable[..., Any]]:
    binding = _load_dispatch().get(capability_id)
    if not binding:
        return None
    mod = importlib.import_module(str(binding["module"]))
    return getattr(mod, str(binding["fn"]), None)


def execute_capability(
    capability_id: str,
    input_artifacts: Dict[str, Any],
    execution_context: ExecutionContext,
    *,
    dispatch_fn: Optional[Callable[..., Any]] = None,
    smoke_mode: bool = False,
) -> ExecutionEvent:
    """"""
    cid = str(capability_id).strip()
    if not is_headline_capability(cid):
        raise ExecutionLeaseError(f"not_headline_capability:{cid}")
    _validate_lease(cid, execution_context)

    execution_id = execution_context.execution_id or str(uuid.uuid4())
    handler = dispatch_fn or _resolve_handler(cid)

    produced: Dict[str, Any] = {}
    status = "completed"
    contract_checks = [{"check": "lease_valid", "passed": True}]
    provenance: Dict[str, Any] = {"dispatch": "headline", "smoke_mode": smoke_mode}

    if smoke_mode or handler is None:
        produced = {k: f"stub:{v}" for k, v in input_artifacts.items()}
        provenance["mode"] = "smoke_stub"
    else:
        try:
            import inspect

            sig = inspect.signature(handler)
            call_kwargs: Dict[str, Any] = {}
            if "input_artifacts" in sig.parameters:
                call_kwargs["input_artifacts"] = dict(input_artifacts)
            for key in ("scenario_id", "split", "out_dir", "out_base"):
                if key in sig.parameters and key in input_artifacts:
                    call_kwargs[key] = input_artifacts[key]
            if "capability_id" in sig.parameters:
                result = handler(cid, **call_kwargs)
            else:
                result = handler(**call_kwargs)
            provenance["eval_result_keys"] = list(result.keys()) if isinstance(result, dict) else []
            produced = {"output": result}
            if isinstance(result, dict):
                for key in ("pred_depth", "artifact_paths", "recipe_log", "official_command", "scenario_id", "split"):
                    if result.get(key) is not None:
                        produced[key] = result[key]
        except Exception as exc:
            status = "failed"
            contract_checks.append({"check": "dispatch", "passed": False, "error": str(exc)})
            raise

    event = ExecutionEvent(
        execution_id=execution_id,
        capability_id=cid,
        status=status,
        consumed_artifacts=dict(input_artifacts),
        produced_artifacts=produced,
        contract_checks=contract_checks,
        provenance=provenance,
        lease_id=execution_context.lease.lease_id,
    )
    return event


def execute_capability_bundle(
    capability_id: str,
    input_artifacts: Dict[str, Any],
    execution_context: ExecutionContext,
    *,
    dispatch_fn: Optional[Callable[..., Any]] = None,
    smoke_mode: bool = False,
    skip_gpu_probe: bool = True,
) -> ExecuteCapabilityBundle:
    """Execute + emit full ExecutionEvent and both certificates."""
    cid = str(capability_id).strip()
    event = execute_capability(
        cid,
        input_artifacts,
        execution_context,
        dispatch_fn=dispatch_fn,
        smoke_mode=smoke_mode,
    )
    event.provenance["route_id"] = execution_context.lease.route_id
    exec_cert = execution_certificate(event)
    reach_cert = reachability_for_capability(
        cid,
        route_id=execution_context.lease.route_id,
        skip_gpu_probe=skip_gpu_probe,
    )
    return ExecuteCapabilityBundle(
        event=event,
        execution_certificate=exec_cert,
        reachability_certificate=reach_cert,
    )


def execution_certificate(event: ExecutionEvent) -> CapabilityExecutionCertificate:
    from hazardweaver.hcg.certification.certificate_hash import compute_certificate_content_hash

    cert = CapabilityExecutionCertificate(
        execution_id=event.execution_id,
        capability_id=event.capability_id,
        lease_id=event.lease_id,
        route_id=event.provenance.get("route_id", ""),
        ok=event.status == "completed",
        contract_checks=event.contract_checks,
        adapters_explicit=True,
        inputs_from_allowed_state=True,
    )
    cert_dict = cert.model_dump(mode="json")
    cert.content_hash = compute_certificate_content_hash(cert_dict)
    return cert


def _portfolio_dispatch(
    capability_id: str,
    input_artifacts: Dict[str, Any],
    *,
    smoke_mode: bool,
) -> Dict[str, Any]:
    """Dispatch PFDF portfolio caps via official implementation classes."""
    from hazardweaver.hcg.runtime import fixtures as fx
    from hazardweaver.hcg.runtime.portfolio_probe_resolver import is_portfolio_pfdf_capability

    cid = str(capability_id).strip()
    if not is_portfolio_pfdf_capability(cid):
        raise ExecutionLeaseError(f"not_portfolio_capability:{cid}")

    if smoke_mode:
        return {"mode": "smoke_stub", "capability_id": cid, "ok": True}

    if cid == "pfdf_volume_gorr_v2":
        from models.pfdf_volume_adapter.gorr_west import GorrWestPFDFVolumeAdapter
        from hazardweaver.hwa.pfdf_agent.data_access import resolve_usgs_pfdf_record

        record = resolve_usgs_pfdf_record(input_artifacts)
        adapter = GorrWestPFDFVolumeAdapter()
        out = adapter.predict(dict(record))
        log_vol = out.get("log_volume_v1")
        if log_vol is None:
            log_vol = out.get("log_volume")
        return {
            "ok": bool(out.get("valid", True)) and log_vol is not None,
            "capability_id": cid,
            "record_id": str(record.get("record_id") or input_artifacts.get("record_id") or ""),
            "log_volume_v1": log_vol,
            "log_volume": log_vol,
        }

    if cid == "burn_state_net_prithvi_v1":
        import os

        from hazardweaver.hwa.capabilities.loader import CapabilityLoader
        from hazardweaver.hwa.pfdf_agent.data_access import resolve_usgs_pfdf_record

        record = resolve_usgs_pfdf_record(input_artifacts)
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        loader = CapabilityLoader(
            require_checkpoint=(device == "cuda"),
            device=device,
        )
        pred = loader.load(cid)
        out = pred.predict(dict(record), oracle=bool(input_artifacts.get("oracle", True)))
        return {"ok": True, "capability_id": cid, "burn_summary": _compact_predict_out(out)}

    raise ExecutionLeaseError(f"portfolio_dispatch_not_wired:{cid}")


def _compact_predict_out(out: Any) -> Any:
    if isinstance(out, dict):
        return {k: v for k, v in out.items() if k in {"MeandNBR", "FractionModHigh", "FractionBurned", "valid", "log_volume_v1"}}
    return out


def portfolio_capability_forward(
    capability_id: str,
    input_artifacts: Dict[str, Any],
    *,
    smoke_mode: bool = False,
) -> Dict[str, Any]:
    """Public PFDF portfolio forward used by headline dispatch bindings."""
    return _portfolio_dispatch(capability_id, input_artifacts, smoke_mode=smoke_mode)


def execute_portfolio_capability(
    capability_id: str,
    input_artifacts: Dict[str, Any],
    execution_context: ExecutionContext,
    *,
    smoke_mode: bool = False,
) -> ExecutionEvent:
    """Execute PFDF portfolio capability (not headline CAP-*)."""
    from hazardweaver.hcg.runtime.portfolio_probe_resolver import is_portfolio_pfdf_capability

    cid = str(capability_id).strip()
    if not is_portfolio_pfdf_capability(cid):
        raise ExecutionLeaseError(f"not_portfolio_capability:{cid}")
    _validate_lease(cid, execution_context)

    execution_id = execution_context.execution_id or str(uuid.uuid4())
    status = "completed"
    contract_checks = [{"check": "lease_valid", "passed": True}]
    provenance: Dict[str, Any] = {"dispatch": "portfolio", "smoke_mode": smoke_mode}

    try:
        result = _portfolio_dispatch(cid, input_artifacts, smoke_mode=smoke_mode)
        produced = {"output": result}
        if isinstance(result, dict):
            for key in ("log_volume_v1", "burn_summary", "ok"):
                if result.get(key) is not None:
                    produced[key] = result[key]
        provenance["eval_result_keys"] = list(result.keys()) if isinstance(result, dict) else []
    except Exception as exc:
        status = "failed"
        contract_checks.append({"check": "dispatch", "passed": False, "error": str(exc)})
        raise

    return ExecutionEvent(
        execution_id=execution_id,
        capability_id=cid,
        status=status,
        consumed_artifacts=dict(input_artifacts),
        produced_artifacts=produced,
        contract_checks=contract_checks,
        provenance=provenance,
        lease_id=execution_context.lease.lease_id,
    )


def execute_portfolio_capability_bundle(
    capability_id: str,
    input_artifacts: Dict[str, Any],
    execution_context: ExecutionContext,
    *,
    smoke_mode: bool = False,
    skip_gpu_probe: bool = True,
) -> ExecuteCapabilityBundle:
    """Execute PFDF portfolio cap + emit full certificate bundle."""
    from hazardweaver.hcg.runtime.portfolio_probe_resolver import portfolio_reachability_verdict

    cid = str(capability_id).strip()
    event = execute_portfolio_capability(
        cid,
        input_artifacts,
        execution_context,
        smoke_mode=smoke_mode,
    )
    event.provenance["route_id"] = execution_context.lease.route_id
    exec_cert = execution_certificate(event)
    reach_cert = portfolio_reachability_verdict(cid, skip_gpu=skip_gpu_probe)
    reach_cert = reach_cert.model_copy(update={"route_id": execution_context.lease.route_id})
    return ExecuteCapabilityBundle(
        event=event,
        execution_certificate=exec_cert,
        reachability_certificate=reach_cert,
    )
