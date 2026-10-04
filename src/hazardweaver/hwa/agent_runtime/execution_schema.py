"""P0-6 RouteCandidate / ExecutionResult contracts + workdir registry.

Claim Guard: schema mint ≠ STRENGTH; registry-backed ids required for CoreExec success.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

ROUTE_CANDIDATE_KEYS = (
    "route_id",
    "capability_ids",
    "adapter_ids",
    "input_contract",
    "output_contract",
    "support",
    "validation_utility",
    "estimated_cost",
    "theory_alignment_refs",
)

FINAL_ARTIFACT_KEYS = (
    "artifact_id",
    "schema_id",
    "value_or_uri",
    "units_support",
)


def mint_execution_id(*, seed: str = "", counter: Optional[int] = None) -> str:
    """Mint opaque execution id ``E_`` + 12 hex chars."""
    raw = f"{seed}|{counter if counter is not None else time.time_ns()}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"E_{digest}"


def mint_artifact_id(*, execution_id: str, suffix: str = "final") -> str:
    digest = hashlib.sha1(f"{execution_id}|{suffix}".encode("utf-8")).hexdigest()[:10]
    return f"A_{digest}"


def build_route_candidate(
    *,
    route_id: Optional[str] = None,
    capability_ids: Optional[Sequence[str]] = None,
    adapter_ids: Optional[Sequence[str]] = None,
    input_contract: Optional[Mapping[str, Any]] = None,
    output_contract: Optional[Mapping[str, Any]] = None,
    support: Optional[Mapping[str, Any]] = None,
    validation_utility: Optional[float] = None,
    estimated_cost: Optional[float] = None,
    theory_alignment_refs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "route_id": str(route_id) if route_id else None,
        "capability_ids": [str(c) for c in (capability_ids or [])],
        "adapter_ids": [str(a) for a in (adapter_ids or [])],
        "input_contract": dict(input_contract or {}),
        "output_contract": dict(output_contract or {}),
        "support": dict(support or {}),
        "validation_utility": validation_utility,
        "estimated_cost": estimated_cost,
        "theory_alignment_refs": [str(t) for t in (theory_alignment_refs or [])],
    }


def _pick_value(raw: Mapping[str, Any]) -> Any:
    for key in (
        "predicted_value",
        "predicted_label",
        "prediction",
        "forecast",
        "value",
        "label",
        "answer",
    ):
        if key not in raw:
            continue
        val = raw[key]
        if isinstance(val, dict) and len(val) == 1:
            return next(iter(val.values()))
        if isinstance(val, (dict, list)) and key == "answer" and val:
            # HCG answer map target→scalar
            if isinstance(val, dict) and all(
                isinstance(v, (int, float, str, bool)) or v is None for v in val.values()
            ):
                return val
            continue
        if isinstance(val, (dict, list)):
            continue
        return val
    # Nested under result
    nested = raw.get("result")
    if isinstance(nested, Mapping):
        return _pick_value(nested)
    outputs = raw.get("outputs")
    if isinstance(outputs, Mapping):
        for k in ("predicted_value", "predicted_label", "reference_label"):
            if k in outputs and not isinstance(outputs[k], (dict, list)):
                return outputs[k]
    return None


def _drill_produced_artifact_metrics(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract flat HWB submission keys from HCG headline ``produced_artifacts`` shape."""
    from hazardweaver.hwa.runtime.trajectory_emitter import _metrics_to_submission_value

    pa = raw.get("produced_artifacts")
    if not isinstance(pa, Mapping):
        return None
    output = pa.get("output")
    if not isinstance(output, Mapping):
        return None
    metrics = output.get("metrics")
    if isinstance(metrics, Mapping):
        mapped = _metrics_to_submission_value(metrics)
        if mapped:
            return mapped
    if any(k in output for k in ("metric_value", "reference_score", "rmse_depth", "metric_name")):
        mapped = _metrics_to_submission_value(output)
        if mapped:
            return mapped
    inner_raw = output.get("raw")
    if isinstance(inner_raw, Mapping):
        cap = str(
            output.get("capability_id")
            or raw.get("capability_id")
            or inner_raw.get("model_id")
            or ""
        ).strip()
        merged: Dict[str, Any] = {
            **dict(inner_raw),
            "capability_id": cap,
            "record_id": str(output.get("record_id") or raw.get("record_id") or ""),
        }
        pfdf = _extract_pfdf_dispatch_metrics(merged)
        if pfdf:
            return pfdf
    return None


def _extract_pfdf_dispatch_metrics(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Map real PFDF tool / portfolio predict payloads to HWB submission scalars."""
    cap = str(raw.get("capability_id") or raw.get("model_id") or "").strip()
    if cap not in ("burn_state_net_prithvi_v1", "pfdf_volume_gorr_v2"):
        return None
    from hazardweaver.hwa.runtime.track_native_metrics import score_pfdf_capability

    scored = score_pfdf_capability(
        capability_id=cap,
        record_id=str(raw.get("record_id") or ""),
        produced_artifacts={"output": dict(raw)},
    )
    if scored.get("status") != "ok":
        return None
    if scored.get("metric_name") == "burn_severity_summary_v1":
        mv = float(scored["metric_value"])
        return {
            "metric_name": "burn_severity_summary_v1",
            "metric_value": mv,
            "reference_score": mv,
        }
    log_vol = float(scored["log_volume_v1"])
    return {
        "metric_name": "log_volume_v1",
        "log_volume_v1": log_vol,
        "reference_score": log_vol,
    }


def _extract_fl2_solver_metrics(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Score CAP-FL2-04~06 solver runs from real batch preds / HAND replay."""
    cap = str(raw.get("capability_id") or "").strip().upper()
    if cap not in {f"CAP-FL2-{i:02d}" for i in range(4, 7)}:
        return None
    scenario_id = str(raw.get("scenario_id") or "").strip()
    split = str(raw.get("split") or "official_test").strip()
    if not scenario_id:
        return None
    from hazardweaver.hwa.runtime.fl2_scenario_metrics import score_fl2_solver_scenario

    scored = score_fl2_solver_scenario(
        capability_id=cap,
        split=split,
        scenario_id=scenario_id,
    )
    if scored.get("status") != "ok":
        return None
    rmse = float(scored["rmse_depth"])
    out: Dict[str, Any] = {
        "metric_name": "rmse_depth",
        "rmse_depth": rmse,
        "reference_score": rmse,
        "scenario_id": scenario_id,
        "split": split,
    }
    if scored.get("csi_0.01") is not None:
        out["csi_0.01"] = float(scored["csi_0.01"])
    return out


def _extract_submission_metrics(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """HWB E_q scalar keys from g2 / capability replay / headline HCG payloads."""
    from hazardweaver.hwa.runtime.trajectory_emitter import _metrics_to_submission_value

    drilled = _drill_produced_artifact_metrics(raw)
    if drilled:
        return drilled
    pfdf = _extract_pfdf_dispatch_metrics(raw)
    if pfdf:
        return pfdf
    fl2 = _extract_fl2_solver_metrics(raw)
    if fl2:
        return fl2
    for key in ("metrics", "prediction"):
        val = raw.get(key)
        if isinstance(val, Mapping):
            mapped = _metrics_to_submission_value(val)
            if mapped:
                return mapped
    nested = raw.get("result")
    if isinstance(nested, Mapping):
        return _extract_submission_metrics(nested)
    compact = raw.get("raw")
    if isinstance(compact, Mapping):
        merged = {**dict(raw), **dict(compact)}
        pfdf = _extract_pfdf_dispatch_metrics(merged)
        if pfdf:
            return pfdf
        fl2 = _extract_fl2_solver_metrics(merged)
        if fl2:
            return fl2
    execution_event = raw.get("execution_event")
    if isinstance(execution_event, Mapping):
        return _extract_submission_metrics(execution_event)
    return None


def mint_raw_result_for_registry(raw: Any) -> Any:
    """Normalize capability output before ``mint_and_register_execution``."""
    if not isinstance(raw, Mapping):
        return raw
    metrics = _extract_submission_metrics(raw)
    if not metrics:
        return raw
    merged = dict(raw)
    merged["metrics"] = metrics
    if metrics.get("reference_score") is not None and "reference_score" not in merged:
        merged["reference_score"] = metrics["reference_score"]
    return merged


def _value_or_uri_has_score(value_or_uri: Any) -> bool:
    if not isinstance(value_or_uri, Mapping):
        return False
    if str(value_or_uri.get("status") or "") == "ok_no_scalar":
        return False
    for key in ("reference_score", "metric_value", "rmse_depth", "log_volume_v1", "csi_0.01"):
        val = value_or_uri.get(key)
        if val is not None:
            try:
                float(val)
                return True
            except (TypeError, ValueError):
                continue
    return False


def normalize_final_artifact(
    raw_result: Any,
    *,
    artifact_id: str,
    schema_id: str = "hwa.final_artifact/v1",
) -> Dict[str, Any]:
    raw: Mapping[str, Any]
    if isinstance(raw_result, Mapping):
        raw = raw_result
    else:
        raw = {"value": raw_result}
    value = _pick_value(dict(raw))
    metrics_value = _extract_submission_metrics(raw)
    if metrics_value:
        if isinstance(value, Mapping):
            value = {**dict(value), **metrics_value}
        else:
            value = metrics_value
    if value is None and "ok" in raw:
        # Fall back to a compact non-metrics summary
        value = {
            k: raw[k]
            for k in ("capability_id", "dispatch", "answer")
            if k in raw and not str(k).startswith("test")
        } or {"status": "ok_no_scalar"}
    units = None
    if isinstance(raw.get("units"), str):
        units = raw["units"]
    support = raw.get("support") if isinstance(raw.get("support"), Mapping) else {}
    return {
        "artifact_id": artifact_id,
        "schema_id": schema_id,
        "value_or_uri": value,
        "units_support": {"units": units, "support": support},
    }


def build_execution_result(
    *,
    execution_id: str,
    route_id: Optional[str],
    status: str,
    executed_capability_ids: Sequence[str],
    final_artifact: Mapping[str, Any],
    validation_utility: Optional[float] = None,
    runtime_cost: Optional[float] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "execution_result/v1",
        "execution_id": execution_id,
        "route_id": str(route_id) if route_id else None,
        "status": str(status),
        "executed_capability_ids": [str(c) for c in executed_capability_ids],
        "final_artifact": {
            k: final_artifact.get(k) for k in FINAL_ARTIFACT_KEYS if k in final_artifact
        }
        or dict(final_artifact),
        "validation_utility": validation_utility,
        "runtime_cost": runtime_cost,
        "provenance": dict(provenance or {}),
    }


def executions_dir(workdir: Path) -> Path:
    return Path(workdir) / "executions"


def artifacts_dir(workdir: Path) -> Path:
    return Path(workdir) / "artifacts"


def write_execution(workdir: Path, er: Mapping[str, Any]) -> Path:
    """Persist ExecutionResult + artifact body under workdir registry."""
    workdir = Path(workdir)
    eid = str(er.get("execution_id") or "")
    if not eid:
        raise ValueError("execution_id required")
    executions_dir(workdir).mkdir(parents=True, exist_ok=True)
    artifacts_dir(workdir).mkdir(parents=True, exist_ok=True)
    fa = dict(er.get("final_artifact") or {})
    aid = str(fa.get("artifact_id") or mint_artifact_id(execution_id=eid))
    fa = {**fa, "artifact_id": aid}
    art_path = artifacts_dir(workdir) / f"{aid}.json"
    art_path.write_text(json.dumps(fa, indent=2, default=str) + "\n", encoding="utf-8")
    out = dict(er)
    out["final_artifact"] = {
        "artifact_id": aid,
        "schema_id": fa.get("schema_id"),
        "value_or_uri": fa.get("value_or_uri"),
        "units_support": fa.get("units_support"),
    }
    path = executions_dir(workdir) / f"{eid}.json"
    path.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def load_execution(workdir: Path, execution_id: str) -> Optional[Dict[str, Any]]:
    path = executions_dir(Path(workdir)) / f"{str(execution_id).strip()}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_registry_executions(workdir: Path) -> List[Dict[str, Any]]:
    """Load all persisted ExecutionResult records under ``workdir/executions``."""
    root = executions_dir(Path(workdir))
    if not root.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def resolve_submit_execution_id(
    workdir: Path,
    execution_id: str,
    route_id: str = "",
) -> tuple[str, bool]:
    """Map mistaken UUID/HCG ids to the registry ``E_*`` id when unambiguous."""
    eid = str(execution_id or "").strip()
    if load_execution(workdir, eid) is not None:
        return eid, False
    rid = str(route_id or "").strip()
    candidates: List[str] = []
    for er in list_registry_executions(workdir):
        if str(er.get("status") or "") not in {"ok", "success"}:
            continue
        reg_eid = str(er.get("execution_id") or "").strip()
        if not reg_eid.startswith("E_"):
            continue
        if rid and str(er.get("route_id") or "").strip() != rid:
            continue
        candidates.append(reg_eid)
    if len(candidates) == 1:
        return candidates[0], True
    return eid, False


def resolve_workdir(host: Any = None, *, out_dir: Any = None) -> Optional[Path]:
    if out_dir is not None and str(out_dir).strip():
        return Path(out_dir)
    if host is None:
        return None
    for attr in ("default_submit_dir", "clarify_dir", "workdir"):
        val = getattr(host, attr, None)
        if val is not None:
            return Path(val)
    return None


def mint_and_register_execution(
    *,
    host: Any = None,
    out_dir: Any = None,
    raw_result: Any,
    capability_ids: Sequence[str],
    route_id: Optional[str] = None,
    ok: bool = True,
    provenance: Optional[Mapping[str, Any]] = None,
    route_candidate: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Mint ExecutionResult, optionally persist, return solver-facing wrap fields."""
    workdir = resolve_workdir(host, out_dir=out_dir)
    seed = f"{','.join(capability_ids)}|{route_id}|{workdir}"
    eid = mint_execution_id(seed=seed)
    aid = mint_artifact_id(execution_id=eid)
    status = "ok" if ok else "failed"
    mint_input = mint_raw_result_for_registry(raw_result)
    fa = normalize_final_artifact(mint_input, artifact_id=aid)
    er = build_execution_result(
        execution_id=eid,
        route_id=route_id,
        status=status,
        executed_capability_ids=list(capability_ids),
        final_artifact=fa,
        provenance=provenance,
    )
    if workdir is not None:
        write_execution(workdir, er)
    rc = dict(route_candidate) if route_candidate else build_route_candidate(
        route_id=route_id,
        capability_ids=capability_ids,
    )
    return {
        "execution_id": eid,
        "final_artifact_id": aid,
        "route_id": route_id,
        "route_candidate": rc,
        "execution_result": {
            "execution_id": eid,
            "route_id": route_id,
            "status": status,
            "final_artifact_id": aid,
            "executed_capability_ids": list(capability_ids),
        },
        "execution_persisted": workdir is not None,
    }


def normalize_final_artifact_id_for_submit(
    workdir: Path,
    execution_id: str,
    final_artifact_id: str,
) -> tuple[str, bool]:
    """Map mistaken paths/URIs to the registry ``A_*`` id for ``execution_id``.

    Models often copy a filesystem path from run_capability output instead of the
    minted ``final_artifact_id``. The authoritative id always lives on the
    execution record in the workdir registry.
    """
    er = load_execution(workdir, str(execution_id).strip())
    if er is None:
        return str(final_artifact_id or "").strip(), False
    expected = str((er.get("final_artifact") or {}).get("artifact_id") or "").strip()
    got = str(final_artifact_id or "").strip()
    if not expected or got == expected:
        return got, False
    if got.startswith("A_"):
        return got, False
    return expected, True


def verify_submit_solution_ids(
    *,
    workdir: Path,
    route_id: str,
    execution_id: str,
    final_artifact_id: str,
) -> Dict[str, Any]:
    """Return ok=True or structured error for submit_solution hard checks."""
    workdir = Path(workdir)
    execution_id = str(execution_id).strip()
    route_id = str(route_id).strip()
    execution_id, execution_coerced = resolve_submit_execution_id(
        workdir,
        execution_id,
        route_id=route_id,
    )
    original_got = str(final_artifact_id).strip()
    final_artifact_id, coerced = normalize_final_artifact_id_for_submit(
        workdir, execution_id, original_got
    )
    er = load_execution(workdir, execution_id)
    if er is None:
        registry_ids = [
            str(er_row.get("execution_id") or "")
            for er_row in list_registry_executions(workdir)
            if str(er_row.get("execution_id") or "").startswith("E_")
        ]
        return {
            "ok": False,
            "error": "unknown_execution_id",
            "execution_id": execution_id,
            "tool": "submit_solution",
            "hint": (
                "Use run_capability.submit_solution_args.execution_id (E_* prefix). "
                "Do not pass execution_event.execution_id UUID values."
            ),
            "registry_execution_ids": registry_ids[:5],
        }
    fa = er.get("final_artifact") or {}
    if not _value_or_uri_has_score(fa.get("value_or_uri")):
        cap = str((er.get("executed_capability_ids") or [""])[0] or "")
        from hazardweaver.hwa.runtime.execution_score_backfill import backfill_final_artifact_from_execution

        backfill_final_artifact_from_execution(
            workdir,
            execution_id=str(execution_id).strip(),
            final_artifact_id=str(final_artifact_id).strip(),
            capability_id=cap,
        )
        er = load_execution(workdir, execution_id) or er
    if str(er.get("status") or "") not in {"ok", "success"}:
        return {
            "ok": False,
            "error": "execution_not_successful",
            "execution_id": execution_id,
            "status": er.get("status"),
            "tool": "submit_solution",
        }
    fa = er.get("final_artifact") or {}
    aid = str(fa.get("artifact_id") or "")
    if aid != final_artifact_id:
        return {
            "ok": False,
            "error": "final_artifact_mismatch",
            "expected_final_artifact_id": aid,
            "got": original_got,
            "coerced_to": final_artifact_id if coerced else None,
            "tool": "submit_solution",
            "hint": (
                "Use final_artifact_id from run_capability (A_*), not a filesystem path."
            ),
        }
    er_route = er.get("route_id")
    if er_route and str(er_route) != str(route_id).strip():
        return {
            "ok": False,
            "error": "route_execution_mismatch",
            "execution_route_id": er_route,
            "submitted_route_id": route_id,
            "tool": "submit_solution",
        }
    vou = fa.get("value_or_uri")
    if not _value_or_uri_has_score(vou):
        return {
            "ok": False,
            "error": "final_artifact_missing_score",
            "execution_id": execution_id,
            "final_artifact_id": final_artifact_id,
            "value_or_uri": vou,
            "tool": "submit_solution",
            "message": (
                "solve requires a scalar metric in final_artifact; "
                "use clarify or re-run capability with eval output."
            ),
        }
    return {
        "ok": True,
        "execution": er,
        "final_artifact_id": final_artifact_id,
        "final_artifact_id_coerced": coerced,
        "execution_id_coerced": execution_coerced,
    }
