"""Native HWA_TRAJECTORY_v2 emitter (Fusion Task 5)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from hazardweaver.hwa.agent_runtime.execution_schema import artifacts_dir, executions_dir, load_execution
from hazardweaver.hwa.runtime.trajectory_ledger import load_ledger_steps
from hazardweaver.hwa.runtime.terminal_action import normalize_terminal_action, terminal_from_answer_record
from hazardweaver.hwa.runtime.trajectory_tool_calls import steps_from_tool_calls, stamp_final_artifact_scenario


def _scalar_score_from_mapping(metrics: Mapping[str, Any]) -> tuple[Optional[float], str]:
    metric_name = str(metrics.get("metric_name") or "mae")
    for key in (
        "reference_score",
        "val_metric",
        "validation_metric",
        "validation_mae",
        "reported_val_metric",
        "metric_value",
    ):
        if metrics.get(key) is not None:
            return float(metrics[key]), metric_name
    return None, metric_name


def _metrics_to_submission_value(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Map HCG capability metrics dict to HWB scalar submission keys."""
    if not metrics:
        return {}
    rmse = metrics.get("rmse_depth")
    metric_name = str(metrics.get("metric_name") or "")
    if rmse is None and metric_name == "rmse_depth":
        rmse = metrics.get("metric_value")
    out: Dict[str, Any] = {}
    if rmse is not None:
        out["rmse_depth"] = float(rmse)
    csi = metrics.get("csi_0.01")
    if csi is not None:
        out["csi_0.01"] = float(csi)
    log_vol = metrics.get("log_volume_v1")
    if log_vol is not None:
        out["log_volume_v1"] = float(log_vol)
    score, metric_name = _scalar_score_from_mapping(metrics)
    if score is not None and "reference_score" not in out:
        out["metric_name"] = metric_name
        out["reference_score"] = score
    return out


def _value_from_artifact_body(body: Mapping[str, Any]) -> Dict[str, Any]:
    vou = body.get("value_or_uri")
    if isinstance(vou, Mapping):
        mapped = _metrics_to_submission_value(vou)
        return mapped or dict(vou)
    val = body.get("value")
    if isinstance(val, Mapping):
        return dict(val)
    return {}


def _provenance_from_event(prov: Mapping[str, Any], *, capability_id: str = "") -> Dict[str, Any]:
    smoke = bool(prov.get("smoke_mode"))
    out: Dict[str, Any] = {
        "smoke_mode": smoke,
        "capability_id": capability_id or prov.get("capability_id") or "",
        "dispatch": prov.get("dispatch"),
    }
    if smoke or prov.get("mode") == "smoke_stub":
        out["inference_mode"] = "smoke_stub"
    else:
        out["inference_mode"] = "hcg_native_eval"
    return out


def _final_artifact_from_answer_body(body: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Build canonical final_artifact for execute_route / legacy solve answers."""
    if not isinstance(body, Mapping):
        return None
    action = str(body.get("action") or "")
    if action not in ("execute_route", "solve"):
        return None
    score, metric_name = _scalar_score_from_mapping(body)
    if score is None:
        return None
    return {
        "schema_id": "hwa.final_artifact/v1",
        "value": {
            "metric_name": metric_name,
            "reference_score": float(score),
        },
        "provenance": {
            "source": "execute_route_answer",
            "route_id": body.get("route_id"),
            "capability_id": body.get("capability_id"),
            "inference_mode": "hcg_native_eval",
        },
    }


def _final_artifact_from_steps(steps: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for step in reversed(steps):
        if step.get("kind") != "tool_execution":
            continue
        ev = step.get("execution_event") or {}
        if not isinstance(ev, Mapping):
            continue
        prov = ev.get("provenance") or {}
        output = ((ev.get("produced_artifacts") or {}).get("output") or {})
        if not isinstance(output, Mapping):
            continue
        metrics = output.get("metrics") if isinstance(output.get("metrics"), Mapping) else output
        value = _metrics_to_submission_value(metrics if isinstance(metrics, Mapping) else {})
        if not value:
            continue
        return {
            "artifact_id": step.get("final_artifact_id"),
            "execution_id": step.get("execution_id"),
            "value": value,
            "provenance": {
                **_provenance_from_event(prov, capability_id=str(step.get("capability_id") or "")),
                "source": "native_ledger_execution_event",
            },
        }
    return None


def _final_artifact_from_registry(
    workdir: Path,
    *,
    execution_id: str = "",
    artifact_id: str = "",
) -> Optional[Dict[str, Any]]:
    workdir = Path(workdir)
    eid = str(execution_id or "").strip()
    aid = str(artifact_id or "").strip()
    if eid:
        er = load_execution(workdir, eid)
        if er:
            fa = er.get("final_artifact") or {}
            aid = aid or str(fa.get("artifact_id") or "")
            art_path = artifacts_dir(workdir) / f"{aid}.json" if aid else None
            body: Dict[str, Any] = {}
            if art_path is not None and art_path.is_file():
                body = json.loads(art_path.read_text(encoding="utf-8"))
            value = _value_from_artifact_body(body)
            if value:
                prov = dict(er.get("provenance") or {})
                return {
                    "artifact_id": aid or body.get("artifact_id"),
                    "execution_id": eid,
                    "value": value,
                    "provenance": {
                        **prov,
                        "source": "execution_registry",
                    },
                }
    exec_dir = executions_dir(workdir)
    if not exec_dir.is_dir():
        return None
    exec_paths = sorted(exec_dir.glob("E_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in exec_paths:
        try:
            er = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fa = er.get("final_artifact") or {}
        reg_aid = str(fa.get("artifact_id") or "")
        if aid and reg_aid and reg_aid != aid:
            continue
        art_path = artifacts_dir(workdir) / f"{reg_aid}.json" if reg_aid else None
        body = {}
        if art_path is not None and art_path.is_file():
            body = json.loads(art_path.read_text(encoding="utf-8"))
        value = _value_from_artifact_body(body)
        if value:
            return {
                "artifact_id": reg_aid or body.get("artifact_id"),
                "execution_id": str(er.get("execution_id") or ""),
                "value": value,
                "provenance": {
                    **dict(er.get("provenance") or {}),
                    "source": "execution_registry",
                },
            }
    return None


def _artifact_value_has_score(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("reference_score") is not None:
        return True
    metric_name = str(value.get("metric_name") or "")
    if metric_name and value.get(metric_name) is not None:
        return True
    if value.get("metric_value") is not None:
        return True
    if value.get("rmse_depth") is not None:
        return True
    if value.get("csi_0.01") is not None:
        return True
    if value.get("log_volume_v1") is not None:
        return True
    if value.get("volume_mae") is not None:
        return True
    if value.get("average_precision") is not None:
        return True
    if value.get("max_mae") is not None:
        return True
    return False


def resolve_final_artifact(
    workdir: Path,
    *,
    steps: List[Dict[str, Any]],
    answer: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Best-effort final_artifact for HWB native metric scoring."""
    workdir = Path(workdir)
    exec_id = str(answer.get("execution_id") or "")
    art_id = str(answer.get("final_artifact_id") or "")

    step_fa = _final_artifact_from_steps(steps)

    if exec_id or art_id:
        reg = _final_artifact_from_registry(workdir, execution_id=exec_id, artifact_id=art_id)
        if reg and _artifact_value_has_score(reg.get("value") or {}):
            value = dict(reg.get("value") or {})
            if not value.get("capability_id"):
                prov_cap = str((reg.get("provenance") or {}).get("capability_id") or "")
                if prov_cap:
                    value["capability_id"] = prov_cap
                    reg = {**reg, "value": value}
            return reg
    if step_fa:
        return step_fa
    reg = _final_artifact_from_registry(workdir)
    if reg and _artifact_value_has_score(reg.get("value") or {}):
        return reg
    body = answer.get("answer")
    answer_body = dict(body) if isinstance(body, Mapping) else dict(answer)
    return _final_artifact_from_answer_body(answer_body)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _steps_from_ledger(workdir: Path) -> List[Dict[str, Any]]:
    ledger = load_ledger_steps(workdir)
    if ledger:
        return ledger
    tool_steps = steps_from_tool_calls(workdir)
    if tool_steps:
        return tool_steps
    decisions = _read_jsonl(workdir / "controller_decisions.jsonl")
    steps = []
    for i, d in enumerate(decisions):
        steps.append({
            "step_index": i,
            "kind": "controller_decision",
            "action": d.get("action"),
            "route_id": d.get("route_id"),
            "lease_id": d.get("extra", {}).get("lease_id")
            if isinstance(d.get("extra"), dict)
            else None,
            "A_sci": d.get("a_sci") or d.get("A_sci"),
            "A_cap": d.get("a_cap") or d.get("A_cap"),
            "reachability_certificate": d.get("reachability_certificate"),
            "checkpoint": d.get("checkpoint"),
            "execution_id": d.get("execution_id"),
            "source": "controller_decisions_fallback",
        })
    return steps


def emit_trajectory_v2(
    workdir: Path,
    *,
    task_id: str = "",
    route_id: str = "",
    lease_id: str = "",
    terminal_action: str = "",
    scenario_id: str = "",
) -> Path:
    """Finalize trajectory from native ledger (primary) with decision-log fallback."""
    workdir = Path(workdir)
    answer: Dict[str, Any] = {}
    ans_path = workdir / "answer.json"
    if ans_path.is_file():
        answer = json.loads(ans_path.read_text(encoding="utf-8"))
    body = answer.get("answer")
    answer_body = dict(body) if isinstance(body, Mapping) else dict(answer)
    run_meta: Dict[str, Any] = {}
    meta_path = workdir / "run_meta.json"
    if meta_path.is_file():
        run_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    steps = _steps_from_ledger(workdir)
    native = bool(load_ledger_steps(workdir))

    final_artifact = resolve_final_artifact(workdir, steps=steps, answer=answer_body)
    if not final_artifact:
        final_artifact = _final_artifact_from_answer_body(answer_body)
    sid = (
        scenario_id
        or str((run_meta.get("fl2_parametric") or {}).get("scenario_id") or "")
        or str(((run_meta.get("fl2_pilot_slice") or {}).get("parameters") or {}).get("scenario_id") or "")
    )
    if sid:
        final_artifact = stamp_final_artifact_scenario(final_artifact, scenario_id=sid)

    terminal = normalize_terminal_action(terminal_action) or terminal_from_answer_record(answer)
    if not terminal:
        body = answer.get("answer")
        if isinstance(body, Mapping):
            terminal = normalize_terminal_action(body.get("action"))

    ledger_source = "native_ledger" if native else (
        "tool_calls" if steps and steps[0].get("source") != "controller_decisions_fallback" else (
            "controller_decisions_fallback" if steps else "empty"
        )
    )

    traj = {
        "schema_version": "HWA_TRAJECTORY_v2",
        "task_id": task_id or run_meta.get("task_id") or "",
        "route_id": route_id or run_meta.get("route_id") or "",
        "lease_id": lease_id or run_meta.get("lease_id") or "",
        "terminal_action": terminal,
        "steps": steps,
        "n_steps": len(steps),
        "native_ledger": native,
        "ledger_source": ledger_source,
        "methodology_note": (
            "controller_decisions_fallback is dev-only; default AgentRuntime records "
            "hwa_trajectory_v2_ledger.jsonl at event time."
            if ledger_source == "controller_decisions_fallback"
            else None
        ),
        "run_meta": {
            "controller_mode": run_meta.get("controller_mode"),
            "exit_reason": run_meta.get("exit_reason"),
            "n_tool_calls": run_meta.get("n_tool_calls"),
            "fl2_pilot_slice": run_meta.get("fl2_pilot_slice"),
            "fl2_parametric": run_meta.get("fl2_parametric"),
        },
        "emitted_utc": datetime.now(timezone.utc).isoformat(),
    }
    if final_artifact:
        if not final_artifact.get("schema_id"):
            final_artifact = {
                "schema_id": "hwa.final_artifact/v1",
                "value": final_artifact.get("value") or {},
                "artifact_id": final_artifact.get("artifact_id"),
                "execution_id": final_artifact.get("execution_id"),
                "provenance": final_artifact.get("provenance") or {},
            }
        traj["final_artifact"] = final_artifact
    certs: List[Dict[str, Any]] = []
    seen_cert: set[str] = set()
    for step in steps:
        cert = step.get("execution_certificate")
        if isinstance(cert, Mapping) and cert.get("execution_id"):
            eid = str(cert["execution_id"])
            if eid not in seen_cert:
                certs.append(dict(cert))
                seen_cert.add(eid)
    if certs:
        traj["execution_certificates"] = certs
    out = workdir / "HWA_TRAJECTORY_v2.json"
    out.write_text(json.dumps(traj, indent=2) + "\n", encoding="utf-8")
    return out
