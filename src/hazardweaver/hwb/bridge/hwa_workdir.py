"""Bridge HWA agent workdirs → HWB dual_gate submissions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from hazardweaver.hwa.runtime.trajectory_emitter import _final_artifact_from_answer_body, resolve_final_artifact
from hazardweaver.hwa.runtime.terminal_action import normalize_terminal_action, terminal_from_answer_record
from hazardweaver.hwa.runtime.trajectory_ledger import load_ledger_steps
from hazardweaver.hwb.bridge.submission_metric_v1 import normalize_submission_value
from hazardweaver.hwb.registry.g6_hard_anchor_meta import taskpack_id_for_g6_task

ROOT = Path(__file__).resolve().parents[3]
TASKPACK_DIR = ROOT / "hwb" / "registry" / "taskpacks"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _answer_body(answer: Mapping[str, Any]) -> Dict[str, Any]:
    body = answer.get("answer")
    return dict(body) if isinstance(body, Mapping) else dict(answer)


def _infer_unknown_metadata(
    *,
    terminal: str,
    run_meta: Mapping[str, Any],
    answer: Mapping[str, Any],
) -> Dict[str, Any]:
    exit_reason = str(run_meta.get("exit_reason") or "")
    emit_source = str(answer.get("emit_source") or "")
    if terminal == "clarify" or exit_reason == "env_null_no_user" or "env_null" in emit_source:
        return {
            "reason_code": "UNKNOWN",
            "asci_verdict": "SCI_UNKNOWN_PENDING_THEORY",
            "exit_reason": exit_reason,
            "emit_source": emit_source,
        }
    return {}


def _resolve_internal_task_id(
    workdir: Path,
    traj: Mapping[str, Any],
    answer: Mapping[str, Any],
    run_meta: Mapping[str, Any],
) -> str:
    for src in (run_meta, traj.get("run_meta") or {}, answer):
        if not isinstance(src, Mapping):
            continue
        internal = str(src.get("internal_task_id") or "").strip()
        if internal:
            return internal
    parent = workdir.parent.parent
    omap_path = parent / "opaque_id_map.json"
    if not omap_path.is_file():
        omap_path = workdir.parent.parent.parent / "opaque_id_map.json"
    tid = str(traj.get("task_id") or answer.get("task_id") or run_meta.get("task_id") or "")
    if omap_path.is_file() and tid.startswith("T_"):
        omap = json.loads(omap_path.read_text(encoding="utf-8"))
        rev = omap.get("reverse") or {}
        if tid in rev:
            return str(rev[tid])
    return tid


def _normalize_step_certificate(step: Dict[str, Any]) -> Dict[str, Any]:
    """Align registry ``E_*`` execution ids with legacy HCG certificate ids for V_q."""
    cert = step.get("execution_certificate")
    if not isinstance(cert, Mapping):
        return step
    step_eid = str(step.get("execution_id") or "").strip()
    cert_eid = str(cert.get("execution_id") or "").strip()
    if not step_eid or not cert_eid or step_eid == cert_eid:
        return step
    if step_eid.startswith("E_"):
        out = dict(step)
        out_cert = dict(cert)
        out_cert["execution_id"] = step_eid
        out_cert.pop("content_hash", None)
        out["execution_certificate"] = out_cert
        return out
    return step


def _enrich_hwb_step_certificate(workdir: Path, step: Dict[str, Any]) -> Dict[str, Any]:
    """Attach execution_certificate from execution registry when trajectory step omitted it."""
    if step.get("execution_certificate"):
        return step
    eid = str(step.get("execution_id") or "").strip()
    if not eid:
        return step
    from hazardweaver.hwa.agent_runtime.execution_schema import load_execution

    er = load_execution(workdir, eid)
    if not er:
        return step
    out = dict(step)
    cert = er.get("execution_certificate")
    if isinstance(cert, Mapping) and cert.get("execution_id"):
        out["execution_certificate"] = dict(cert)
    ev = er.get("execution_event")
    if isinstance(ev, Mapping) and not out.get("provenance"):
        prov = ev.get("provenance")
        if isinstance(prov, Mapping):
            out["provenance"] = dict(prov)
    return out


def _hwa_steps_to_hwb(steps: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in steps:
        if not isinstance(raw, Mapping):
            continue
        kind = str(raw.get("kind") or "")
        if kind == "tool_execution":
            out.append(
                {
                    "step_index": int(raw.get("step_index", len(out))),
                    "kind": "capability",
                    "capability_id": raw.get("capability_id"),
                    "ok": raw.get("ok", True),
                    "execution_id": raw.get("execution_id"),
                    "lease_id": raw.get("lease_id"),
                    "execution_certificate": raw.get("execution_certificate"),
                    "provenance": (raw.get("execution_event") or {}).get("provenance")
                    if isinstance(raw.get("execution_event"), Mapping)
                    else {},
                }
            )
        elif kind == "controller_decision":
            out.append(
                {
                    "step_index": int(raw.get("step_index", len(out))),
                    "kind": "controller",
                    "action": raw.get("action"),
                    "route_id": raw.get("route_id"),
                    "ok": True,
                    "execution_id": raw.get("execution_id"),
                    "lease_id": raw.get("lease_id"),
                }
            )
        elif kind == "submit":
            out.append(
                {
                    "step_index": int(raw.get("step_index", len(out))),
                    "kind": "submit",
                    "ok": raw.get("ok", True),
                }
            )
    return out


def build_hwb_submission(
    workdir: Path,
    *,
    answer_body: Optional[Mapping[str, Any]] = None,
    taskpack: Optional[Mapping[str, Any]] = None,
    route_id: str = "",
) -> Dict[str, Any]:
    """Single submission builder for batch eval and VCE VERIFY (pre-submit)."""
    workdir = Path(workdir)
    traj = _load_json(workdir / "HWA_TRAJECTORY_v2.json")
    answer = _load_json(workdir / "answer.json")
    run_meta = _load_json(workdir / "run_meta.json")
    body = dict(answer_body) if answer_body is not None else _answer_body(answer)
    if route_id:
        body.setdefault("route_id", route_id)

    action = str(body.get("action") or "").strip().lower()
    if action in {"solve", "submit_solution"}:
        from hazardweaver.hwa.runtime.track_native_metrics import patch_headline_workdir_metrics_if_needed

        patch_headline_workdir_metrics_if_needed(workdir, body)
        inv_row = _load_json(workdir / "inventory_row.json")
        if inv_row:
            from hazardweaver.hwa.runtime.unified_final_artifact_align_v1 import align_unified_final_artifact_if_needed

            align_unified_final_artifact_if_needed(workdir, inv_row)

    internal_task_id = _resolve_internal_task_id(workdir, traj, answer, run_meta)
    solver_task_id = str(traj.get("task_id") or answer.get("task_id") or run_meta.get("task_id") or "")

    terminal = terminal_from_answer_record({**answer, "answer": body}) or normalize_terminal_action(
        traj.get("terminal_action") or body.get("action") or answer.get("action")
    )
    raw_steps = list(traj.get("steps") or [])
    if not raw_steps:
        raw_steps = load_ledger_steps(workdir)
    steps = [
        _normalize_step_certificate(_enrich_hwb_step_certificate(workdir, s))
        for s in _hwa_steps_to_hwb(raw_steps)
    ]

    submission: Dict[str, Any] = {
        "schema_version": str(traj.get("schema_version") or "HWA_TRAJECTORY_v2"),
        "task_id": internal_task_id or solver_task_id,
        "internal_task_id": internal_task_id or None,
        "solver_task_id": solver_task_id or None,
        "terminal_action": terminal or ("solve" if action in {"solve", "submit_solution"} else terminal),
        "steps": steps,
        "route_summary": {"route_id": traj.get("route_id") or body.get("route_id") or route_id or ""},
    }

    steps_for_resolve = list(traj.get("steps") or [])
    if not steps_for_resolve:
        steps_for_resolve = load_ledger_steps(workdir)
    if action in {"solve", "submit_solution"} and body.get("execution_id"):
        final_artifact = resolve_final_artifact(workdir, steps=steps_for_resolve, answer=body)
    else:
        final_artifact = traj.get("final_artifact")
        if not final_artifact or not (final_artifact.get("value") or {}).get("reference_score"):
            final_artifact = resolve_final_artifact(workdir, steps=steps_for_resolve, answer=body)

    if final_artifact:
        if not final_artifact.get("schema_id"):
            final_artifact = {
                "schema_id": "hwa.final_artifact/v1",
                "value": final_artifact.get("value") or {},
                "artifact_id": final_artifact.get("artifact_id"),
                "execution_id": final_artifact.get("execution_id"),
                "provenance": final_artifact.get("provenance") or {},
            }
        tp = taskpack
        if tp is None:
            try:
                tp = load_taskpack_for_submission(
                    {
                        "internal_task_id": internal_task_id,
                        "task_id": internal_task_id or solver_task_id,
                        "hwa_run_meta": {"workdir": str(workdir)},
                    }
                )
            except (FileNotFoundError, OSError, ValueError, KeyError):
                tp = None
        value = dict(final_artifact.get("value") or {})
        final_artifact = {
            **final_artifact,
            "value": normalize_submission_value(value, taskpack=tp),
        }
        submission["final_artifact"] = final_artifact
    elif body.get("final_artifact_id") or body.get("execution_id"):
        eid = str(body.get("execution_id") or "")
        if not eid:
            raise ValueError("unknown_execution_id")
        from hazardweaver.hwa.agent_runtime.execution_schema import load_execution

        er = load_execution(workdir, eid)
        if er is None:
            raise ValueError(f"unknown_execution_id:{eid}")
        vou = (er.get("final_artifact") or {}).get("value_or_uri")
        value = dict(vou) if isinstance(vou, Mapping) else {"reference_score": vou}
        submission["final_artifact"] = {
            "schema_id": "hwa.final_artifact/v1",
            "value": normalize_submission_value(value, taskpack=taskpack),
            "execution_id": eid,
            "artifact_id": str(body.get("final_artifact_id") or ""),
        }
    else:
        fa = _final_artifact_from_answer_body(body)
        if fa:
            submission["final_artifact"] = fa

    meta = _infer_unknown_metadata(terminal=str(submission.get("terminal_action") or ""), run_meta=run_meta, answer=answer)
    if meta:
        submission["abstention_metadata"] = meta

    if traj.get("lease_id"):
        submission["execution_lease"] = {
            "lease_id": traj.get("lease_id"),
            "route_id": traj.get("route_id"),
            "allowed_capability_ids": [
                s.get("capability_id")
                for s in steps
                if s.get("capability_id")
            ],
        }

    certs: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in list(traj.get("execution_certificates") or []):
        if isinstance(raw, Mapping) and raw.get("execution_id"):
            eid = str(raw["execution_id"])
            if eid not in seen:
                certs.append(dict(raw))
                seen.add(eid)
    for step in steps:
        cert = step.get("execution_certificate")
        if isinstance(cert, Mapping) and cert.get("execution_id"):
            eid = str(cert["execution_id"])
            if eid not in seen:
                certs.append(dict(cert))
                seen.add(eid)
    if certs:
        submission["execution_certificates"] = certs
        traj_out = dict(traj)
        traj_out["execution_certificates"] = certs
        submission["trajectory"] = traj_out

    submission["hwa_run_meta"] = {
        "workdir": str(workdir),
        "exit_reason": run_meta.get("exit_reason"),
        "model_id": run_meta.get("model_id") or answer.get("model_id_used"),
        "llm_provider": run_meta.get("llm_provider"),
        "native_ledger": traj.get("native_ledger"),
        "n_steps": traj.get("n_steps"),
        "internal_task_id": internal_task_id,
        "opaque_task_id": run_meta.get("opaque_task_id") or solver_task_id,
    }
    inv_row = _load_json(workdir / "inventory_row.json")
    if inv_row:
        from hazardweaver.hwa.runtime.unified_final_artifact_align_v1 import repair_unified_submission_for_native_eval

        submission = repair_unified_submission_for_native_eval(
            submission,
            workdir=workdir,
            inventory_row=inv_row,
        )
    return submission


def load_hwa_submission(workdir: Path) -> Dict[str, Any]:
    """Load HWA workdir artifacts into an HWB-evaluable submission dict."""
    return build_hwb_submission(Path(workdir))


def resolve_taskpack_id(submission: Mapping[str, Any]) -> str:
    meta = submission.get("hwa_run_meta") or {}
    tp_from_meta = str(meta.get("taskpack_id") or "").strip()
    if tp_from_meta.startswith("hwb_"):
        return tp_from_meta
    internal = str(submission.get("internal_task_id") or "").strip()
    if internal.startswith("H_A_"):
        return taskpack_id_for_g6_task(internal)
    tid = str(submission.get("task_id") or "")
    if tid.startswith("H_A_"):
        return taskpack_id_for_g6_task(tid)
    if tid.startswith("hwb_"):
        return tid
    if tid.startswith("fl2-pilot"):
        return "hwb_fl2_solver_parametric_v1"
    internal = str(meta.get("internal_task_id") or "").strip()
    if internal.startswith("H_A_"):
        return taskpack_id_for_g6_task(internal)
    return tp_from_meta or tid


def load_taskpack_for_submission(submission: Mapping[str, Any]) -> Dict[str, Any]:
    meta = submission.get("hwa_run_meta") or {}
    workdir = Path(str(meta.get("workdir") or ""))
    resolved = workdir / "resolved_taskpack.json"
    if resolved.is_file():
        return json.loads(resolved.read_text(encoding="utf-8"))
    inv_row = workdir / "inventory_row.json"
    if inv_row.is_file():
        from hazardweaver.hwb.run.eval_dca_submission_v1 import resolve_taskpack_for_inventory_row

        return resolve_taskpack_for_inventory_row(json.loads(inv_row.read_text(encoding="utf-8")))
    tp_id = resolve_taskpack_id(submission)
    path = TASKPACK_DIR / f"{tp_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"taskpack not found for {tp_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    man = json.loads(path.read_text(encoding="utf-8"))
    tasks = man.get("tasks") or man.get("scenarios") or []
    return list(tasks)
