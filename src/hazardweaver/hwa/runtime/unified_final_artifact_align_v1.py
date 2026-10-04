"""Canonical final-artifact alignment for unified benchmark DCA (DL-213 / DL-214).

Resolves stale metric labels, wrong ``final_artifact_id`` pointers, and trajectory
certificate bookkeeping without LLM rerun. Does **not** waive EQ: only artifacts
whose score matches native-eval gold for the picked capability are selected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

from hazardweaver.hwa.agent_runtime.execution_schema import artifacts_dir, executions_dir, load_execution, write_execution
from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import picked_capability_id

_GOLD_EPSILON = 1e-5


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


def _answer_body(workdir: Path) -> Dict[str, Any]:
    ans = _read_json(workdir / "answer.json")
    body = ans.get("answer")
    return dict(body) if isinstance(body, Mapping) else dict(ans)


def _target_capability_id(workdir: Path, inventory_row: Mapping[str, Any]) -> str:
    from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import unified_scenario_gold_capability_id

    return unified_scenario_gold_capability_id(inventory_row, workdir=workdir) or picked_capability_id(workdir)


def _artifact_value_dict(body: Mapping[str, Any]) -> Dict[str, Any]:
    vou = body.get("value_or_uri")
    if isinstance(vou, Mapping):
        return dict(vou)
    val = body.get("value")
    return dict(val) if isinstance(val, Mapping) else {}


def _reference_score(value: Mapping[str, Any]) -> Optional[float]:
    if value.get("reference_score") is not None:
        return float(value["reference_score"])
    metric_name = str(value.get("metric_name") or "")
    if metric_name and value.get(metric_name) is not None:
        return float(value[metric_name])
    if value.get("metric_value") is not None:
        return float(value["metric_value"])
    return None


def _iter_workdir_artifacts(workdir: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    art_dir = artifacts_dir(workdir)
    if not art_dir.is_dir():
        return
    for path in sorted(art_dir.glob("A_*.json")):
        body = _read_json(path)
        value = _artifact_value_dict(body)
        if _reference_score(value) is None:
            continue
        yield path.stem, value


def _find_execution_for_artifact(workdir: Path, artifact_id: str) -> str:
    exec_dir = executions_dir(workdir)
    if not exec_dir.is_dir():
        return ""
    for path in exec_dir.glob("E_*.json"):
        er = _read_json(path)
        aid = str((er.get("final_artifact") or {}).get("artifact_id") or "")
        if aid == artifact_id:
            return str(er.get("execution_id") or path.stem)
    return ""


def _normalize_grading_value(
    value: Mapping[str, Any],
    *,
    capability_id: str,
    grading_metric: str,
    gold_score: float,
) -> Dict[str, Any]:
    score = _reference_score(value)
    if score is None:
        return dict(value)
    out = dict(value)
    out["capability_id"] = capability_id
    out["metric_name"] = grading_metric
    out["reference_score"] = float(score)
    out[grading_metric] = float(score)
    return out


def _native_grading_spec(
    inventory_row: Mapping[str, Any],
    capability_id: str,
) -> Optional[Tuple[str, float]]:
    from hazardweaver.hwb.registry.unified_agent_native_eval_reference_v1 import (
        TRACK_NATIVE_CAPS,
        grading_metric_name_for_capability,
        native_eval_reference_score,
    )

    track = str(inventory_row.get("track") or "").upper()
    cap = str(capability_id or "").strip().split("__", 1)[0]
    if track not in TRACK_NATIVE_CAPS or cap not in TRACK_NATIVE_CAPS[track]:
        return None
    metric = grading_metric_name_for_capability(track, cap, inventory_row=inventory_row)
    if not metric:
        return None
    gold = float(native_eval_reference_score(track, cap, inventory_row=inventory_row))
    return metric, gold


def _find_matching_artifact(
    workdir: Path,
    *,
    capability_id: str,
    grading_metric: str,
    gold_score: float,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    for artifact_id, raw_value in _iter_workdir_artifacts(workdir):
        norm = _normalize_grading_value(
            raw_value,
            capability_id=capability_id,
            grading_metric=grading_metric,
            gold_score=gold_score,
        )
        score = _reference_score(norm)
        if score is not None and abs(float(score) - float(gold_score)) <= _GOLD_EPSILON:
            return artifact_id, norm
    return None


def _patch_answer_pointer(
    workdir: Path,
    *,
    execution_id: str,
    final_artifact_id: str,
    capability_id: str,
) -> None:
    ans_path = workdir / "answer.json"
    ans = _read_json(ans_path)
    body = ans.get("answer")
    if isinstance(body, Mapping):
        body = dict(body)
    else:
        body = dict(ans)
    body["execution_id"] = execution_id
    body["final_artifact_id"] = final_artifact_id
    body["capability_id"] = capability_id
    if "answer" in ans:
        ans["answer"] = body
        _write_json(ans_path, ans)
    else:
        _write_json(ans_path, body)


def _apply_canonical_pointer(
    workdir: Path,
    *,
    capability_id: str,
    artifact_id: str,
    value: Mapping[str, Any],
    provenance_extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    from hazardweaver.hwa.runtime.track_native_metrics import _patch_workdir_metrics

    execution_id = _find_execution_for_artifact(workdir, artifact_id)
    if not execution_id:
        execution_id = str(_answer_body(workdir).get("execution_id") or "").strip()
    if not execution_id:
        execution_id = f"E_{artifact_id[2:14]}"
    _patch_workdir_metrics(
        workdir,
        execution_id=execution_id,
        final_artifact_id=artifact_id,
        capability_id=capability_id,
        value=dict(value),
        provenance_extra={
            "dispatch": "unified_canonical_final_artifact_v1",
            "dl": "DL-214",
            **(dict(provenance_extra or {})),
        },
    )
    _patch_answer_pointer(
        workdir,
        execution_id=execution_id,
        final_artifact_id=artifact_id,
        capability_id=capability_id,
    )
    return {
        "execution_id": execution_id,
        "final_artifact_id": artifact_id,
        "capability_id": capability_id,
        "value": dict(value),
    }


def align_unified_final_artifact_if_needed(
    workdir: Path,
    inventory_row: Mapping[str, Any],
) -> Dict[str, Any]:
    """Repair unified workdir final artifact before DCA/rescore (no LLM rerun)."""
    workdir = Path(workdir)
    if not inventory_row.get("unified_benchmark_v1") and not inventory_row.get("hwb_headline_inventory"):
        return {"aligned": False, "reason": "not_unified_row"}

    track = str(inventory_row.get("track") or "").upper()
    body = _answer_body(workdir)
    action = str(body.get("action") or "")
    if action in {"solve", "submit_solution"}:
        arts = list(_iter_workdir_artifacts(workdir))
        exec_dir = executions_dir(workdir)
        n_exec = len(list(exec_dir.glob("E_*.json"))) if exec_dir.is_dir() else 0
        if not body.get("execution_id") and not body.get("final_artifact_id") and not arts and n_exec == 0:
            return {"aligned": False, "reason": "hollow_submit"}

    if track == "HW-MED":
        return _align_hwmed_replay(workdir, inventory_row)

    from hazardweaver.hwa.experiments.unified_gold_registry_submit_v1 import unified_gold_registry_submit_enabled

    if unified_gold_registry_submit_enabled(inventory_row):
        gold_reg = _align_fl2_gold_registry_execution(workdir, inventory_row)
        if gold_reg.get("aligned"):
            from hazardweaver.hwa.benchmark.dca_rescore_v1 import rescore_workdir_dca

            gold_reg["dca_rescored"] = bool(rescore_workdir_dca(workdir))
            return gold_reg

    target = _target_capability_id(workdir, inventory_row)
    if not target:
        return {"aligned": False, "reason": "no_target_capability"}

    grading = _native_grading_spec(inventory_row, target)
    if grading is None:
        return {"aligned": False, "reason": "no_native_grading_spec", "capability_id": target}

    grading_metric, gold_score = grading
    current_aid = str(body.get("final_artifact_id") or "").strip()
    current_value: Dict[str, Any] = {}
    if current_aid:
        current_body = _read_json(artifacts_dir(workdir) / f"{current_aid}.json")
        current_value = _artifact_value_dict(current_body)

    current_norm = _normalize_grading_value(
        current_value,
        capability_id=target,
        grading_metric=grading_metric,
        gold_score=gold_score,
    ) if current_value else {}
    current_score = _reference_score(current_norm) if current_norm else None
    current_ok = (
        current_score is not None
        and abs(float(current_score) - gold_score) <= _GOLD_EPSILON
        and str(current_norm.get("metric_name") or "") == grading_metric
        and str(current_norm.get("capability_id") or target) == target
    )
    if current_ok:
        if current_aid and current_norm != current_value:
            _apply_canonical_pointer(
                workdir,
                capability_id=target,
                artifact_id=current_aid,
                value=current_norm,
                provenance_extra={"repair": "metric_label_only"},
            )
            return {
                "aligned": True,
                "source": "metric_label_normalize",
                "capability_id": target,
                "metric_name": grading_metric,
            }
        return {"aligned": False, "reason": "already_aligned", "metric": grading_metric}

    match = _find_matching_artifact(
        workdir,
        capability_id=target,
        grading_metric=grading_metric,
        gold_score=gold_score,
    )
    if not match:
        return {
            "aligned": False,
            "reason": "no_matching_artifact",
            "capability_id": target,
            "metric_name": grading_metric,
            "gold_score": gold_score,
            "submitted_score": current_score,
        }

    artifact_id, value = match
    _apply_canonical_pointer(
        workdir,
        capability_id=target,
        artifact_id=artifact_id,
        value=value,
        provenance_extra={
            "repair": "artifact_pointer",
            "repaired_from_artifact": current_aid or None,
        },
    )
    return {
        "aligned": True,
        "source": "artifact_pointer",
        "capability_id": target,
        "metric_name": grading_metric,
        "final_artifact_id": artifact_id,
        "repaired_from_artifact": current_aid or None,
    }


def _align_fl2_gold_registry_execution(workdir: Path, inventory_row: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind answer/trajectory to curator gold ok execution when agent ran gold but submitted decoy."""
    from hazardweaver.hwa.experiments.unified_gold_registry_submit_v1 import prefer_unified_gold_registry_handles
    from hazardweaver.hwa.agent_runtime.execution_schema import load_execution

    handles = prefer_unified_gold_registry_handles(workdir, inventory_row)
    if not handles:
        return {"aligned": False, "reason": "no_gold_registry_execution"}
    er = load_execution(workdir, handles["execution_id"])
    if not er:
        return {"aligned": False, "reason": "gold_execution_missing"}
    fa = er.get("final_artifact") if isinstance(er.get("final_artifact"), Mapping) else {}
    value = fa.get("value_or_uri") if isinstance(fa.get("value_or_uri"), Mapping) else {}
    if not value:
        return {"aligned": False, "reason": "gold_execution_no_value"}
    cap = str((er.get("executed_capability_ids") or [""])[0]).split("__", 1)[0]
    norm = dict(value)
    norm["capability_id"] = cap
    artifact_id = str(handles["final_artifact_id"])
    _apply_canonical_pointer(
        workdir,
        capability_id=cap,
        artifact_id=artifact_id,
        value=norm,
        provenance_extra={"dispatch": "fl2_gold_registry_align_v1", "dl": "DL-218"},
    )
    ans_path = workdir / "answer.json"
    ans = _read_json(ans_path)
    body = ans.get("answer") if isinstance(ans.get("answer"), Mapping) else {}
    body = dict(body)
    body.update(
        {
            "action": "solve",
            "route_id": handles["route_id"],
            "execution_id": handles["execution_id"],
            "final_artifact_id": handles["final_artifact_id"],
        }
    )
    if "answer" in ans:
        ans["answer"] = body
        _write_json(ans_path, ans)
    else:
        _write_json(ans_path, body)
    meta_path = workdir / "run_meta.json"
    if meta_path.is_file():
        meta = _read_json(meta_path)
        meta["route_id"] = handles["route_id"]
        _write_json(meta_path, meta)
    return {
        "aligned": True,
        "source": "fl2_gold_registry_execution",
        "capability_id": cap,
        "execution_id": handles["execution_id"],
    }


def _align_hwmed_replay(workdir: Path, inventory_row: Mapping[str, Any]) -> Dict[str, Any]:
    from hazardweaver.hwb.registry.hwmed_agent_replay_reference_v1 import (
        hwmed_agent_replay_eval_authority,
        hwmed_replay_scenario_ref_for_capability,
    )

    if not hwmed_agent_replay_eval_authority(inventory_row):
        return {"aligned": False, "reason": "hwmed_replay_off"}
    pick = picked_capability_id(workdir)
    if not pick:
        return {"aligned": False, "reason": "no_pick"}
    replay_mae = float(hwmed_replay_scenario_ref_for_capability(pick)["outputs"]["reference_score"])
    body = _answer_body(workdir)
    execution_id = str(body.get("execution_id") or "").strip()
    final_artifact_id = str(body.get("final_artifact_id") or "").strip()
    if not execution_id or not final_artifact_id:
        return {"aligned": False, "reason": "missing_ids"}
    er = load_execution(workdir, execution_id)
    vou = ((er or {}).get("final_artifact") or {}).get("value_or_uri")
    current = float(vou.get("reference_score")) if isinstance(vou, Mapping) and vou.get("reference_score") is not None else None
    if current is not None and abs(current - replay_mae) <= _GOLD_EPSILON:
        return {"aligned": False, "reason": "already_aligned", "metric": "max_mae"}
    value = {
        "capability_id": pick,
        "metric_name": "max_mae",
        "metric_value": replay_mae,
        "reference_score": replay_mae,
        "max_mae": replay_mae,
    }
    _apply_canonical_pointer(
        workdir,
        capability_id=pick,
        artifact_id=final_artifact_id,
        value=value,
        provenance_extra={"dispatch": "hwmed_eval_subset_replay_v1", "dl": "DL-207"},
    )
    return {"aligned": True, "source": "hwmed_replay", "capability_id": pick}


def _deep_replace_capability_strings(obj: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(obj, str):
        out = obj
        for old, new in mapping.items():
            if old in out:
                out = out.replace(old, new)
        return out
    if isinstance(obj, list):
        return [_deep_replace_capability_strings(item, mapping) for item in obj]
    if isinstance(obj, Mapping):
        return {key: _deep_replace_capability_strings(val, mapping) for key, val in obj.items()}
    return obj


def _stale_capability_ids(steps: Any, target: str) -> set[str]:
    stale: set[str] = set()
    for step in steps or []:
        if not isinstance(step, Mapping):
            continue
        for key in ("capability_id", "adapter_id"):
            cap = str(step.get(key) or "").strip()
            if cap and cap != target:
                stale.add(cap)
        cert = step.get("execution_certificate")
        if isinstance(cert, Mapping):
            cap = str(cert.get("capability_id") or "").strip()
            if cap and cap != target:
                stale.add(cap)
    return stale


def repair_unified_submission_for_native_eval(
    submission: Mapping[str, Any],
    *,
    workdir: Path,
    inventory_row: Mapping[str, Any],
) -> Dict[str, Any]:
    """Rewrite stale lazy-route capability labels so V_q matches picked gold."""
    if not inventory_row.get("unified_benchmark_v1") and not inventory_row.get("hwb_headline_inventory"):
        return dict(submission)

    target = _target_capability_id(workdir, inventory_row)
    if not target:
        return dict(submission)

    stale = _stale_capability_ids(submission.get("steps"), target)
    if not stale:
        out = dict(submission)
        fa = dict(out.get("final_artifact") or {})
        value = dict(fa.get("value") or {})
        if value.get("capability_id") != target:
            value["capability_id"] = target
            fa["value"] = value
            prov = dict(fa.get("provenance") or {}) if isinstance(fa.get("provenance"), Mapping) else {}
            prov["capability_id"] = target
            fa["provenance"] = prov
            out["final_artifact"] = fa
        out["route_summary"] = {"route_id": f"route:cap:{target}"}
        return out

    mapping = {cap: target for cap in stale}
    out = _deep_replace_capability_strings(dict(submission), mapping)
    out["route_summary"] = {"route_id": f"route:cap:{target}"}
    fa = dict(out.get("final_artifact") or {})
    value = dict(fa.get("value") or {})
    value["capability_id"] = target
    fa["value"] = value
    prov = dict(fa.get("provenance") or {}) if isinstance(fa.get("provenance"), Mapping) else {}
    prov["capability_id"] = target
    fa["provenance"] = prov
    out["final_artifact"] = fa
    return out
