"""HWB native eval for HWA Pilot-4 Fusion B manifest (PFDF / DR-OUT / TC-TRK)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from hazardweaver.hwa.runtime.track_native_metrics import (
    DR_OUT_PROXY_CAPS,
    score_dr_out_capability,
    score_pfdf_capability,
    score_tc_trk_capability,
)

ROOT = Path(__file__).resolve().parents[3]

TC_TRK_ARTIFACT_REPLAY_CAPS = frozenset({"CAP-TCTRK-03", "CAP-TCTRK-04", "CAP-TCTRK-05"})

REQUIRED_WORKDIR_FILES = (
    "HWA_TRAJECTORY_v2.json",
    "run_meta.json",
    "hwa_trajectory_translated.json",
)


def load_pilot4_manifest(path: Path) -> Dict[str, Any]:
    man = json.loads(Path(path).read_text(encoding="utf-8"))
    if man.get("schema") != "hwa_pilot4_fusion_b_manifest_v1":
        raise ValueError(f"unexpected manifest schema: {man.get('schema')}")
    return man


def _metric_close(a: float, b: float, *, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= max(atol, rtol * max(abs(float(a)), abs(float(b)), 1.0))


def _extract_manifest_metric(native_metrics: Mapping[str, Any]) -> tuple[str, float]:
    metric_name = str(native_metrics.get("metric_name") or "")
    if metric_name == "log_volume_v1" or native_metrics.get("log_volume_v1") is not None:
        return "log_volume_v1", float(native_metrics.get("log_volume_v1") or native_metrics.get("metric_value"))
    metric_value = native_metrics.get("metric_value")
    if metric_value is None:
        metric_value = native_metrics.get("reference_score")
    if metric_value is None:
        raise ValueError("manifest native_metrics missing metric value")
    return metric_name or "metric_value", float(metric_value)


def _native_rescore(track: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    cap = str(row.get("capability_id") or "")
    if track == "PFDF":
        return score_pfdf_capability(
            capability_id=cap,
            record_id=str(row.get("record_id") or ""),
        )
    if track == "DR-OUT":
        return score_dr_out_capability(capability_id=cap)
    if track == "TC-TRK":
        return score_tc_trk_capability(capability_id=cap)
    return {"status": "score_failed", "error": f"unknown_track:{track}"}


def _rescore_metric(scored: Mapping[str, Any]) -> tuple[str, float]:
    if scored.get("status") != "ok":
        raise ValueError(f"native_rescore_failed:{scored}")
    if scored.get("metric_name") == "log_volume_v1" or scored.get("log_volume_v1") is not None:
        return "log_volume_v1", float(scored["log_volume_v1"])
    metric_name = str(scored.get("metric_name") or "metric_value")
    metric_value = scored.get("metric_value")
    if metric_value is None:
        raise ValueError(f"native_rescore_missing_value:{scored}")
    return metric_name, float(metric_value)


def _workdir_checks(workdir: Path) -> Dict[str, Any]:
    missing = [name for name in REQUIRED_WORKDIR_FILES if not (workdir / name).is_file()]
    run_meta: Dict[str, Any] = {}
    traj: Dict[str, Any] = {}
    if not missing:
        run_meta = json.loads((workdir / "run_meta.json").read_text(encoding="utf-8"))
        traj = json.loads((workdir / "HWA_TRAJECTORY_v2.json").read_text(encoding="utf-8"))
    return {
        "missing_files": missing,
        "submit_solution": bool(run_meta.get("submit_solution")),
        "terminal_action": str(traj.get("terminal_action") or ""),
        "native_ledger": bool(traj.get("native_ledger")),
    }


def _track_annotations(track: str, capability_id: str) -> Dict[str, Any]:
    notes: Dict[str, Any] = {
        "claim_tier": "pilot4_fusion_b_cap_only",
        "not_headline_pass": True,
    }
    if track == "PFDF":
        notes["d_pfdf_1"] = "SRC_PENDING"
        notes["mh1_crc_substitute"] = False
    if capability_id in DR_OUT_PROXY_CAPS:
        notes["proxy_replay"] = True
        notes["metric_honesty"] = "proxy tolerance — not final native metric"
    if capability_id in TC_TRK_ARTIFACT_REPLAY_CAPS:
        notes["artifact_replay"] = True
        notes["metric_honesty"] = "artifact replay — not live NWP"
    if track == "TC-TRK" and capability_id == "CAP-TCTRK-01":
        notes["persistence_fallback"] = True
    return notes


def evaluate_track_row(row: Mapping[str, Any], *, root: Path = ROOT) -> Dict[str, Any]:
    track = str(row.get("track") or "")
    cap = str(row.get("capability_id") or "")
    workdir = root / str(row.get("workdir") or "")
    hwa_ok = bool(row.get("ok"))
    native_metrics = dict(row.get("native_metrics") or {})

    out: Dict[str, Any] = {
        "track": track,
        "capability_id": cap,
        "workdir": str(workdir),
        "hwa_manifest_ok": hwa_ok,
        "record_id": row.get("record_id"),
        "scenario_id": native_metrics.get("scenario_id"),
        "execution_id": row.get("execution_id"),
        "final_artifact_id": row.get("final_artifact_id"),
        "annotations": _track_annotations(track, cap),
    }

    wd_checks = _workdir_checks(workdir)
    out["workdir_checks"] = wd_checks

    errors: List[str] = []
    if not hwa_ok:
        errors.append("hwa_manifest_not_ok")
    if wd_checks["missing_files"]:
        errors.append(f"missing_workdir_files:{','.join(wd_checks['missing_files'])}")
    if not wd_checks.get("submit_solution"):
        errors.append("submit_solution_false")
    if wd_checks.get("terminal_action") != "solve":
        errors.append(f"terminal_action:{wd_checks.get('terminal_action')}")

    rescored = _native_rescore(track, row)
    out["native_eval"] = rescored
    if rescored.get("status") != "ok":
        errors.append("native_eval_failed")
    else:
        try:
            manifest_name, manifest_value = _extract_manifest_metric(native_metrics)
            rescore_name, rescore_value = _rescore_metric(rescored)
            out["metric_name"] = rescore_name
            out["metric_value"] = rescore_value
            out["reference_score"] = float(native_metrics.get("reference_score") or rescore_value)
            out["manifest_metric_name"] = manifest_name
            out["manifest_metric_value"] = manifest_value
            if manifest_name != rescore_name:
                errors.append(f"metric_name_mismatch:{manifest_name}!={rescore_name}")
            if not _metric_close(manifest_value, rescore_value):
                errors.append(
                    f"metric_value_mismatch:{manifest_value}!={rescore_value}"
                )
        except (TypeError, ValueError) as exc:
            errors.append(f"metric_parse_error:{exc}")

    out["valid"] = hwa_ok and not errors
    out["errors"] = errors
    return out


def evaluate_pilot4_manifest(
    manifest_path: Path,
    *,
    root: Path = ROOT,
) -> Dict[str, Any]:
    man = load_pilot4_manifest(manifest_path)
    records = [evaluate_track_row(row, root=root) for row in man.get("tracks") or []]
    by_track = {str(r["track"]): r for r in records}
    return {
        "schema_version": "HWB_PILOT4_FUSION_B_EVAL_v1",
        "manifest_path": str(manifest_path),
        "decision": man.get("decision"),
        "d_pfdf_1_status": man.get("d_pfdf_1_status"),
        "fusion_b_ok": man.get("fusion_b_ok"),
        "n_valid": sum(1 for r in records if r.get("valid")),
        "n_total": len(records),
        "records": records,
        "by_track": by_track,
    }
