"""HWB evaluation helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from hazardweaver.hwb.evaluators.pilot4_fusion_b_eval import _metric_close
from hazardweaver.hwa.runtime.hcg_a_audit_authority_v1 import (
    abstain_fields_from_audit,
    hcg_a_completion_tier,
    load_hcg_a_audit_record,
)
from hazardweaver.hwa.runtime.seven_track_finish_line_c_common import (
    DEFAULT_HCG_PILOT7_MANIFEST,
    DEFAULT_HWB_HANDOFF_MANIFEST,
    SEVEN_TRACKS,
    load_hcg_pilot7_manifest,
    track_caps_from_manifest,
)
from hazardweaver.hwa.runtime.trajectory_content_hash_v1 import is_valid_content_hash

ROOT = Path(__file__).resolve().parents[3]

SEVEN_TRACK_FINISH_LINE_C_SCHEMA = "SEVEN_TRACK_FINISH_LINE_C_EVAL_v1"

HWB_ELIGIBLE_TIERS = frozenset(
    {"L2_scientific_replay", "L2_src_and_scientific", "gpu_deferred_replay"}
)

ABSTAIN_CAPS = frozenset(
    {
        "CAP-MH2-06",
        "CAP-MH3-02",
        "CAP-MH3-03",
        "CAP-MH3-04",
        "CAP-MH4-R04",
        "CAP-MH4-R05",
    }
)


@dataclass(frozen=True)
class TrackSealConfig:
    track: str
    slug: str
    expected_n_valid: int


TRACK_SEAL_CONFIG: Dict[str, TrackSealConfig] = {
    "WF-3": TrackSealConfig("WF-3", "wf3", 6),
    "L2": TrackSealConfig("L2", "l2", 5),
    "E1-E3": TrackSealConfig("E1-E3", "e1e3", 8),
    "HW-MED": TrackSealConfig("HW-MED", "hw_med", 6),
    "MH-2": TrackSealConfig("MH-2", "mh2", 5),
    "MH-3": TrackSealConfig("MH-3", "mh3", 3),
    "MH-4": TrackSealConfig("MH-4", "mh4", 4),
}

EXPECTED_N_VALID_TOTAL = sum(cfg.expected_n_valid for cfg in TRACK_SEAL_CONFIG.values())


def _load_json(path: Optional[str | Path]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_hwb_handoff(path: Path | str = DEFAULT_HWB_HANDOFF_MANIFEST) -> Dict[str, Any]:
    man_path = Path(path)
    if not man_path.is_file():
        raise FileNotFoundError(f"missing HWA handoff: {man_path}")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man.get("schema") != "hwa_7track_hwb_handoff_v1":
        raise ValueError(f"unexpected handoff schema: {man.get('schema')}")
    return man


def handoff_by_capability(handoff: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("capability_id") or "").upper(): dict(row)
        for row in handoff.get("capabilities") or []
    }


def _extract_metric(native_metrics: Mapping[str, Any]) -> tuple[str, float]:
    metric_name = str(native_metrics.get("metric_name") or "")
    metric_value = native_metrics.get("metric_value")
    if metric_name == "log_volume_v1" or native_metrics.get("log_volume_v1") is not None:
        return "log_volume_v1", float(native_metrics.get("log_volume_v1") or metric_value)
    if metric_value is None:
        metric_value = native_metrics.get("reference_score")
    if metric_value is None:
        raise ValueError("native_metrics missing metric value")
    return metric_name or "metric_value", float(metric_value)


def _native_rescore(track: str, capability_id: str) -> Dict[str, Any]:
    try:
        if track == "WF-3":
            from hazardweaver.hcg.carp.native_eval.eval_wf3_wsts import eval_wf3_cap

            result = eval_wf3_cap(capability_id)
        elif track == "L2":
            from hazardweaver.hcg.carp.native_eval.eval_l2 import eval_l2_cap

            result = eval_l2_cap(capability_id)
        elif track == "E1-E3":
            from hazardweaver.hcg.carp.native_eval.eval_e1e3_anchor import eval_e1e3_cap

            result = eval_e1e3_cap(capability_id)
        elif track == "HW-MED":
            from hazardweaver.hcg.carp.native_eval.eval_hw_med import eval_hwmed_cap

            result = eval_hwmed_cap(capability_id)
        elif track == "MH-2":
            from hazardweaver.hcg.carp.native_eval.eval_mh2 import eval_mh2_cap

            result = eval_mh2_cap(capability_id)
        elif track == "MH-3":
            from hazardweaver.hcg.carp.native_eval.eval_mh3 import eval_mh3_cap

            result = eval_mh3_cap(capability_id)
        elif track == "MH-4":
            from hazardweaver.hcg.carp.native_eval.eval_mh4 import eval_mh4_cap

            result = eval_mh4_cap(capability_id)
        else:
            return {"status": "score_failed", "error": f"unknown_track:{track}"}

        if not isinstance(result, dict):
            return {"status": "score_failed", "error": "invalid_rescore_payload"}
        if result.get("ok"):
            metrics = result.get("metrics") or {}
            metric_name = metrics.get("metric_name")
            metric_value = metrics.get("metric_value")
            if metric_name == "log_volume_v1" or metrics.get("log_volume_v1") is not None:
                return {
                    "status": "ok",
                    "metric_name": "log_volume_v1",
                    "metric_value": metrics.get("log_volume_v1") or metric_value,
                    "source": f"eval_{track}",
                }
            if metric_name and metric_value is not None:
                return {
                    "status": "ok",
                    "metric_name": str(metric_name),
                    "metric_value": float(metric_value),
                    "source": f"eval_{track}",
                }
        blocked = result.get("blocked_reason") or result.get("error") or "eval_failed"
        return {"status": "score_failed", "error": str(blocked), "detail": result}
    except Exception as exc:  # noqa: BLE001 — login rescore is best-effort
        return {"status": "score_failed", "error": str(exc)}


def _cap_annotations(
    *,
    track: str,
    capability_id: str,
    handoff_row: Mapping[str, Any],
    audit: Optional[Mapping[str, Any]],
    abstain: bool,
) -> Dict[str, Any]:
    notes: Dict[str, Any] = {
        "claim_tier": "seven_track_finish_line_c_scientific",
        "not_headline_pass": True,
        "hcg_scientific_authority": not abstain,
        "hwa_trajectory_authority": bool(handoff_row.get("workdir")),
    }
    if abstain:
        notes["claim_tier"] = "hcg_a_abstain"
        notes["hcg_a_abstain"] = True
        notes["hcg_scientific_authority"] = False
        if audit:
            notes["hcg_a_verdict"] = audit.get("verdict")
    tier = str(handoff_row.get("trajectory_tier") or "")
    if tier == "L2_src_and_scientific":
        notes["l2_src_executed"] = bool(handoff_row.get("ok"))
    if tier == "src_binding_only":
        notes["src_binding_only"] = True
        notes["metric_honesty"] = "SRC binding only — not executed"
    return notes


def discover_cap_row(
    capability_id: str,
    *,
    track: str,
    fusion_row: Mapping[str, Any],
    handoff_row: Optional[Mapping[str, Any]],
    root: Path = ROOT,
) -> Dict[str, Any]:
    sci_rel = str(
        (handoff_row or {}).get("scientific_dir")
        or fusion_row.get("scientific_dir")
        or ""
    )
    sci_dir = root / sci_rel if sci_rel else None
    return {
        "track": track,
        "capability_id": capability_id,
        "scientific_dir": str(sci_dir) if sci_dir else sci_rel,
        "native_metrics_path": str(sci_dir / "native_metrics.json") if sci_dir else None,
        "replay_manifest_path": str(sci_dir / "replay_manifest.json") if sci_dir else None,
        "graph_path_id": fusion_row.get("graph_path_id"),
        "completion_tier": fusion_row.get("completion_tier"),
        "handoff": dict(handoff_row or {}),
    }


def evaluate_finish_line_c_cap(cap_row: Mapping[str, Any], *, root: Path = ROOT) -> Dict[str, Any]:
    track = str(cap_row.get("track") or "")
    cap = str(cap_row.get("capability_id") or "").upper()
    handoff_row = dict(cap_row.get("handoff") or {})
    audit = load_hcg_a_audit_record(track, cap)
    audit_tier = hcg_a_completion_tier(track, cap, cap_row)
    abstain = cap in ABSTAIN_CAPS or (
        audit is not None and str(audit.get("completion_tier") or "").upper() != "L2"
    )

    native_path = cap_row.get("native_metrics_path")
    replay_path = cap_row.get("replay_manifest_path")
    file_metrics = _load_json(str(native_path) if native_path else None)
    replay = _load_json(str(replay_path) if replay_path else None)

    workdir_raw = handoff_row.get("workdir")
    workdir = Path(str(workdir_raw)) if workdir_raw else None
    content_hash = handoff_row.get("content_hash")
    trajectory_tier = str(handoff_row.get("trajectory_tier") or "")

    out: Dict[str, Any] = {
        "track": track,
        "capability_id": cap,
        "scientific_dir": cap_row.get("scientific_dir"),
        "native_metrics_path": native_path,
        "replay_manifest_path": replay_path,
        "workdir": str(workdir) if workdir else None,
        "content_hash": content_hash,
        "trajectory_tier": trajectory_tier,
        "hcg_a_completion_tier": audit_tier,
        "hcg_a_abstain": abstain,
        "record_id": None,
        "scenario_id": file_metrics.get("scenario_id") if file_metrics else None,
        "annotations": _cap_annotations(
            track=track,
            capability_id=cap,
            handoff_row=handoff_row,
            audit=audit,
            abstain=abstain,
        ),
    }

    if workdir and (workdir / "HWA_TRAJECTORY_v2.json").is_file():
        traj = _load_json(workdir / "HWA_TRAJECTORY_v2.json")
        out["record_id"] = traj.get("record_id") or traj.get("lease_id")

    errors: List[str] = []
    metric_name = ""
    metric_value: Optional[float] = None

    if abstain:
        if audit:
            abstain_info = abstain_fields_from_audit(audit)
            out["trajectory_tier"] = abstain_info.get("trajectory_tier", trajectory_tier)
        errors.append("hcg_a_abstain")

    if not is_valid_content_hash(content_hash):
        errors.append("invalid_or_missing_content_hash")
        out["annotations"]["invalid_trajectory"] = True

    if not workdir or not (workdir / "HWA_TRAJECTORY_v2.json").is_file():
        errors.append("missing_hwa_trajectory")
        out["annotations"]["invalid_trajectory"] = True

    if trajectory_tier and trajectory_tier not in HWB_ELIGIBLE_TIERS and not abstain:
        errors.append(f"trajectory_tier_ineligible:{trajectory_tier}")

    if audit_tier != "L2":
        errors.append(f"hcg_audit_not_l2:{audit_tier}")

    if not file_metrics:
        errors.append("native_metrics_missing")
    else:
        try:
            metric_name, metric_value = _extract_metric(file_metrics)
            out["file_metric_name"] = metric_name
            out["file_metric_value"] = metric_value
        except (TypeError, ValueError) as exc:
            errors.append(f"file_metric_parse_error:{exc}")

    if not replay:
        errors.append("replay_manifest_missing")
    elif replay.get("exec_ok") is False:
        errors.append("replay_exec_not_ok")

    rescored: Dict[str, Any]
    if os.environ.get("HWB_SEVEN_TRACK_RESCORE", "").lower() in ("1", "true", "yes"):
        rescored = _native_rescore(track, cap)
    else:
        rescored = {"status": "skipped", "error": "login_rescore_skipped"}
    out["native_eval"] = rescored
    rescore_ok = rescored.get("status") == "ok"
    scientific_authority = (
        not abstain
        and audit_tier == "L2"
        and is_valid_content_hash(content_hash)
        and workdir is not None
        and (workdir / "HWA_TRAJECTORY_v2.json").is_file()
        and trajectory_tier in HWB_ELIGIBLE_TIERS
        and bool(file_metrics)
        and replay.get("exec_ok") is not False
        and not errors
    )

    if rescore_ok:
        rescore_name = str(rescored.get("metric_name") or "")
        rescore_value = rescored.get("metric_value")
        if rescore_value is not None:
            out["rescore_metric_name"] = rescore_name
            out["rescore_metric_value"] = float(rescore_value)
            if metric_name and rescore_name and rescore_name != metric_name:
                out["annotations"]["rescore_name_mismatch"] = f"{metric_name}!={rescore_name}"
            if metric_value is not None and not _metric_close(float(metric_value), float(rescore_value)):
                out["annotations"]["rescore_value_drift"] = f"{metric_value}!={rescore_value}"
    elif scientific_authority:
        out["annotations"]["login_rescore_blocked"] = True
        out["annotations"]["rescore_block_reason"] = str(
            rescored.get("error") or "login_rescore_unavailable"
        )

    if metric_name and metric_value is not None:
        out["metric_name"] = metric_name
        out["metric_value"] = float(metric_value)
        out["reference_score"] = float(metric_value)

    out["valid"] = scientific_authority
    out["errors"] = errors
    out["scientific_authority"] = scientific_authority
    return out


def evaluate_finish_line_c_track(
    track: str,
    *,
    root: Path = ROOT,
    handoff_path: Path | str = DEFAULT_HWB_HANDOFF_MANIFEST,
    fusion_path: Path | str = DEFAULT_HCG_PILOT7_MANIFEST,
    caps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    track_u = str(track).upper()
    if track_u not in TRACK_SEAL_CONFIG:
        raise KeyError(f"unknown seven-track: {track_u}")

    fusion = load_hcg_pilot7_manifest(fusion_path)
    handoff = load_hwb_handoff(handoff_path)
    handoff_map = handoff_by_capability(handoff)
    fusion_caps = track_caps_from_manifest(fusion, track_u)
    cap_list = list(caps or [str(r["capability_id"]) for r in fusion_caps])

    records: List[Dict[str, Any]] = []
    fusion_by_cap = {str(r["capability_id"]).upper(): r for r in fusion_caps}
    for cap_id in cap_list:
        cap_u = str(cap_id).upper()
        fusion_row = fusion_by_cap.get(cap_u, {"capability_id": cap_u})
        row = discover_cap_row(
            cap_u,
            track=track_u,
            fusion_row=fusion_row,
            handoff_row=handoff_map.get(cap_u),
            root=root,
        )
        records.append(evaluate_finish_line_c_cap(row, root=root))

    cfg = TRACK_SEAL_CONFIG[track_u]
    return {
        "schema_version": SEVEN_TRACK_FINISH_LINE_C_SCHEMA,
        "track": track_u,
        "slug": cfg.slug,
        "claim_tier": "seven_track_finish_line_c_scientific",
        "not_headline_pass": True,
        "authority": "hcg_scientific_authority+hwa_trajectory_authority",
        "handoff_path": str(handoff_path),
        "fusion_manifest_path": str(fusion_path),
        "expected_n_valid": cfg.expected_n_valid,
        "n_valid": sum(1 for r in records if r.get("valid")),
        "n_total": len(records),
        "caps": cap_list,
        "records": records,
        "by_capability": {str(r["capability_id"]): r for r in records},
    }


def all_track_slugs() -> List[str]:
    return [TRACK_SEAL_CONFIG[t].slug for t in SEVEN_TRACKS if t in TRACK_SEAL_CONFIG]


def track_from_slug(slug: str) -> str:
    for track, cfg in TRACK_SEAL_CONFIG.items():
        if cfg.slug == slug:
            return track
    raise KeyError(f"unknown slug: {slug}")
