"""HWA official inference dispatcher for Agent-Strict v2 (no HCG execute_capability)."""

from __future__ import annotations

import inspect
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hcg.contracts.execution_event import (
    CapabilityExecutionCertificate,
    ExecutionContext,
    ExecutionEvent,
    ScopedExecutionLease,
)
from hazardweaver.hcg.contracts.reachability_certificate import CapabilityReachabilityCertificate
from hazardweaver.hcg.runtime.execute_capability import (
    ExecuteCapabilityBundle,
    _resolve_handler,
    execution_certificate,
    reachability_for_capability,
)
from hazardweaver.hcg.runtime.portfolio_probe_resolver import is_portfolio_pfdf_capability
from hazardweaver.hwa.experiments.agent_strict_v2 import (
    PFDF_PORTFOLIO_CAPS,
    build_parameter_provenance,
    sanitize_strict_execution_handles,
    validate_agent_handles,
)
from hazardweaver.hwa.route_controller.fl2_pilot_slice import FL2_SOLVER_OFFICIAL_CAPS


def _invoke_official_handler(
    capability_id: str,
    input_artifacts: Dict[str, Any],
    *,
    smoke_mode: bool = False,
) -> Dict[str, Any]:
    if smoke_mode:
        return {"ok": True, "mode": "smoke_stub", "stub": True}
    handler = _resolve_handler(capability_id)
    if handler is None:
        return {"ok": False, "error": f"no_official_handler:{capability_id}"}
    sig = inspect.signature(handler)
    call_kwargs: Dict[str, Any] = {}
    if "input_artifacts" in sig.parameters:
        call_kwargs["input_artifacts"] = dict(input_artifacts)
    for key in ("scenario_id", "split", "out_dir", "out_base", "force_eval_subset_replay"):
        if key in sig.parameters and key in input_artifacts:
            call_kwargs[key] = input_artifacts[key]
    if "input_artifacts" in sig.parameters:
        call_kwargs["input_artifacts"] = dict(input_artifacts)
    if "capability_id" in sig.parameters:
        return handler(capability_id, **call_kwargs)
    return handler(**call_kwargs)


def _try_portfolio_hwa_infer(
    host: Any,
    capability_id: str,
    handles: Mapping[str, Any],
    *,
    oracle: bool = False,
    burn_summary: Optional[Mapping[str, Any]] = None,
    record_overrides: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not is_portfolio_pfdf_capability(capability_id):
        return None
    from hazardweaver.hwa.agent_runtime.unified_tools import _try_hw_registry_predict

    record_id = str(handles.get("record_id") or "").strip()
    if not record_id:
        return {
            "ok": False,
            "error": "record_id required for portfolio PFDF capability",
            "capability_id": capability_id,
        }
    return _try_hw_registry_predict(
        host,
        capability_id,
        record=None,
        record_id=record_id,
        oracle=oracle,
        burn_summary=burn_summary,
        record_overrides=record_overrides,
        allow_portfolio=True,
    )


def _try_fl2_hwa_infer(
    capability_id: str,
    handles: Mapping[str, Any],
    *,
    out_dir: Any = None,
) -> Optional[Dict[str, Any]]:
    cid = str(capability_id or "").strip()
    if cid not in FL2_SOLVER_OFFICIAL_CAPS and not cid.startswith("CAP-FL2-"):
        return None
    from hazardweaver.hwa.agent_runtime.fl2_scientific_infer import run_fl2_capability_inference

    scenario_id = str(handles.get("scenario_id") or handles.get("sample_id") or "").strip()
    split = str(handles.get("split") or "").strip()
    if not scenario_id or not split:
        return {
            "ok": False,
            "error": "scenario_id and split required for FL-2 inference",
            "capability_id": cid,
        }
    return run_fl2_capability_inference(
        cid,
        scenario_id=scenario_id,
        split=split,
        out_dir=Path(out_dir) if out_dir else None,
        strict_no_defaults=True,
    )


def run_hwa_strict_headline_infer(
    capability_id: str,
    handles: Mapping[str, Any],
    lease: ScopedExecutionLease,
    *,
    host: Any = None,
    execution_id: str = "",
    smoke_mode: bool = False,
    oracle: bool = False,
    burn_summary: Optional[Mapping[str, Any]] = None,
    record_overrides: Optional[Mapping[str, Any]] = None,
    out_dir: Any = None,
) -> ExecuteCapabilityBundle:
    """Faithful HWA inference + HCG reachability only (no execute_capability forward)."""
    cid = str(capability_id or "").strip()
    ok, errors, param_prov = validate_agent_handles(cid, handles, provenance_src="agent")
    if not ok:
        raise ValueError(";".join(errors))

    handles = sanitize_strict_execution_handles(handles)
    if oracle:
        raise ValueError("oracle_inference_forbidden_in_agent_strict_v2")
    if smoke_mode:
        raise ValueError("smoke_mode_forbidden_in_agent_strict_v2")

    input_artifacts = dict(handles)
    dispatch = "hwa_official_eval"
    result: Dict[str, Any]

    if cid in PFDF_PORTFOLIO_CAPS or is_portfolio_pfdf_capability(cid):
        hw = _try_portfolio_hwa_infer(
            host,
            cid,
            handles,
            oracle=oracle,
            burn_summary=burn_summary,
            record_overrides=record_overrides,
        )
        if hw is None:
            result = {"ok": False, "error": f"portfolio_infer_unavailable:{cid}"}
        else:
            dispatch = "hwa_capability_loader"
            result = hw
    elif cid.startswith("CAP-FL2-") or cid in FL2_SOLVER_OFFICIAL_CAPS:
        fl2 = _try_fl2_hwa_infer(cid, handles, out_dir=out_dir)
        result = fl2 or {"ok": False, "error": f"fl2_infer_unavailable:{cid}"}
    else:
        task = getattr(host, "task", None) or {}
        if cid.startswith("CAP-WF3-"):
            from hazardweaver.hwa.experiments.ablation_manual_pilot_v1 import (
                ablation_wf3_agent_replay_enabled,
                wf3_pinned_scientific_dir,
            )

            if ablation_wf3_agent_replay_enabled(task if isinstance(task, Mapping) else None):
                pinned = wf3_pinned_scientific_dir(cid)
                if pinned is not None:
                    input_artifacts["out_base"] = pinned
                    input_artifacts["out_dir"] = str(pinned)
                    input_artifacts["wf3_inference_mode"] = "pinned_replay"
        if cid.startswith("CAP-HWMED-"):
            from hazardweaver.hwa.experiments.ablation_manual_pilot_v1 import ablation_hwmed_agent_replay_enabled

            if ablation_hwmed_agent_replay_enabled(task if isinstance(task, Mapping) else None):
                input_artifacts["force_eval_subset_replay"] = True
                input_artifacts["hwmed_eval_mode"] = "ewb_eval_subset"
        if out_dir and "out_base" not in input_artifacts:
            input_artifacts.setdefault("out_dir", str(out_dir))
            input_artifacts.setdefault("out_base", Path(out_dir))
        result = _invoke_official_handler(cid, input_artifacts, smoke_mode=smoke_mode)

    exec_id = execution_id or str(uuid.uuid4())
    blocked = (
        isinstance(result, Mapping)
        and (result.get("blocked") is True or (result.get("ok") is False and result.get("reason")))
    )
    status = "completed" if (result.get("ok", True) and not blocked) else "failed"
    produced: Dict[str, Any] = {"output": result}
    if isinstance(result, Mapping):
        for key in ("pred_depth", "artifact_paths", "recipe_log", "official_command", "scenario_id", "split"):
            if result.get(key) is not None:
                produced[key] = result[key]

    provenance: Dict[str, Any] = {
        "dispatch": dispatch,
        "inference_backend": dispatch,
        "hcg_role": "reachability_only",
        "parameter_provenance": param_prov,
        "smoke_mode": smoke_mode,
        "agent_strict_v2": True,
    }
    event = ExecutionEvent(
        execution_id=exec_id,
        capability_id=cid,
        status=status,
        consumed_artifacts=dict(handles),
        produced_artifacts=produced,
        contract_checks=[{"check": "agent_strict_v2_handles", "passed": ok}],
        provenance=provenance,
        lease_id=lease.lease_id,
    )
    ctx = ExecutionContext(lease=lease, execution_id=exec_id)
    event.provenance["route_id"] = lease.route_id
    exec_cert = execution_certificate(event)
    reach_cert = reachability_for_capability(
        cid,
        route_id=lease.route_id or "",
        skip_gpu_probe=True,
    )
    return ExecuteCapabilityBundle(
        event=event,
        execution_certificate=exec_cert,
        reachability_certificate=reach_cert,
    )


def hwa_strict_result_to_run_capability(
    bundle: ExecuteCapabilityBundle,
    *,
    capability_id: str,
) -> Dict[str, Any]:
    from hazardweaver.hwa.runtime.hcg_execute_bridge import event_to_run_capability_result

    payload = event_to_run_capability_result(bundle.event, capability_id=capability_id, bundle=bundle)
    dispatch = str((bundle.event.provenance or {}).get("dispatch") or "hwa_official_eval")
    payload["dispatch"] = dispatch
    payload["inference_backend"] = dispatch
    payload["hcg_role"] = "reachability_only"
    payload["parameter_provenance"] = (bundle.event.provenance or {}).get("parameter_provenance")
    return payload
