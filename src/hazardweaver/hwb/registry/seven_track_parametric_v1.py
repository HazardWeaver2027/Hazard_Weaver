"""Seven-track parametric TaskPack templates and refs sync."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from hazardweaver.hwb.evaluators.seven_track_finish_line_c_common import TRACK_SEAL_CONFIG

ROOT = Path(__file__).resolve().parents[3]
TASKPACK_DIR = Path(__file__).resolve().parent / "taskpacks"

SEVEN_TRACK_PARAMETRIC_IDS = tuple(
    f"hwb_{TRACK_SEAL_CONFIG[t].slug}_parametric_v1" for t in TRACK_SEAL_CONFIG
)


def taskpack_path(slug: str) -> Path:
    return TASKPACK_DIR / f"hwb_{slug}_parametric_v1.json"


def parametric_refs_path(slug: str) -> Path:
    return TASKPACK_DIR / f"hwb_{slug}_parametric_v1_refs.json"


def _tolerance_for_metric(metric_name: str) -> Dict[str, Any]:
    if metric_name in ("brier", "auprc", "average_precision", "threshold_accuracy", "sdo_skill", "iou"):
        return {"metric": metric_name, "min_score": 0.0}
    if metric_name in ("volume_mae", "lead_error_km", "log_mae", "mae"):
        return {"metric": metric_name, "max_abs_error": 1.0}
    if metric_name == "log_volume_v1":
        return {"metric": metric_name, "max_abs_error": 0.5}
    return {"metric": metric_name or "metric_value", "max_abs_error": 1.0}


def build_parametric_taskpack(track: str) -> Dict[str, Any]:
    track_u = str(track).upper()
    cfg = TRACK_SEAL_CONFIG[track_u]
    return {
        "schema_version": "HWB_TASKPACK_v1",
        "taskpack_id": f"hwb_{cfg.slug}_parametric_v1",
        "headline_target": track_u,
        "coverage_tier": "W-covered",
        "hib_is_grader": False,
        "solver_view": {
            "user_goal": (
                f"Given a {track_u} scientific capability anchor, produce the native metric "
                "via HCG scientific replay authority with HWA trajectory witness."
            ),
            "parameters": {"scenario_id": "CAP-PLACEHOLDER", "split": "official_test"},
            "allowed_tools": ["inspect_artifact", "run_capability", "submit"],
            "allowed_edge_ids": [],
            "forbidden_substitutions": ["direct_submit_without_capability"],
        },
        "reference_view": {
            "expected_action": "solve",
            "refs_sidecar": f"hwb_{cfg.slug}_parametric_v1_refs.json",
            "accepted_witnesses": [],
            "outputs": {"metric_name": "metric_value", "reference_score": 0.0},
            "tolerance": {"metric": "metric_value", "max_abs_error": 1.0},
            "trajectory_constraints": {
                "required_adapters": [],
                "required_capability_edges": [],
                "forbidden_shortcuts": ["direct_submit_without_capability"],
                "provenance": {
                    "claim_tier": "seven_track_finish_line_c_scientific",
                    "not_headline_pass": True,
                    "hcg_scientific_authority": True,
                    "hwa_trajectory_authority": True,
                },
            },
        },
        "metadata": {
            "seven_track_finish_line_c": True,
            "not_headline_pass": True,
            "hcg_scientific_authority": True,
            "hwa_trajectory_authority": True,
        },
    }


def ensure_parametric_taskpack_files(track: str) -> None:
    track_u = str(track).upper()
    cfg = TRACK_SEAL_CONFIG[track_u]
    tp_path = taskpack_path(cfg.slug)
    refs_path = parametric_refs_path(cfg.slug)
    if not tp_path.is_file():
        tp_path.write_text(json.dumps(build_parametric_taskpack(track_u), indent=2) + "\n", encoding="utf-8")
    if not refs_path.is_file():
        refs_path.write_text(
            json.dumps(
                {
                    "schema_version": f"HWB_{track_u.replace('-', '_')}_PARAMETRIC_REFS_v1",
                    "taskpack_id": f"hwb_{cfg.slug}_parametric_v1",
                    "scenarios": {},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def sync_refs_from_eval_report(
    report: Mapping[str, Any],
    *,
    taskpack_id: str,
    refs_path: Path,
) -> None:
    refs_path.parent.mkdir(parents=True, exist_ok=True)
    if refs_path.is_file():
        refs = json.loads(refs_path.read_text(encoding="utf-8"))
    else:
        refs = {
            "schema_version": "HWB_SEVEN_TRACK_PARAMETRIC_REFS_v1",
            "taskpack_id": taskpack_id,
            "scenarios": {},
        }
    scenarios: Dict[str, Any] = dict(refs.get("scenarios") or {})
    for rec in report.get("records") or []:
        cap = str(rec.get("capability_id") or "")
        if not cap:
            continue
        metric_name = str(rec.get("metric_name") or "")
        metric_value = rec.get("metric_value")
        if not metric_name or metric_value is None:
            continue
        outputs: Dict[str, Any] = {
            "metric_name": metric_name,
            "reference_score": float(metric_value),
            "scenario_id": cap,
            metric_name: float(metric_value),
        }
        scenarios[cap] = {
            "capability_id": cap,
            "outputs": outputs,
            "tolerance": _tolerance_for_metric(metric_name),
            "provenance": {
                "scenario_id": rec.get("scenario_id") or cap,
                "hcg_scientific_dir": rec.get("scientific_dir"),
                "claim_tier": "seven_track_finish_line_c_scientific",
                "not_headline_pass": True,
                "hcg_scientific_authority": bool(rec.get("valid")),
                "hwa_trajectory_authority": bool(rec.get("workdir")),
            },
        }
    refs["scenarios"] = scenarios
    refs["taskpack_id"] = taskpack_id
    refs_path.write_text(json.dumps(refs, indent=2) + "\n", encoding="utf-8")


def build_abstain_taskpack(
    *,
    capability_id: str,
    track: str,
    verdict: str,
) -> Dict[str, Any]:
    cap = str(capability_id).upper()
    track_u = str(track).upper()
    slug = cap.lower().replace("cap-", "").replace("-", "_")
    return {
        "schema_version": "HWB_TASKPACK_v1",
        "taskpack_id": f"hwb_{slug}_abstain_v1",
        "headline_target": track_u,
        "coverage_tier": "abstention",
        "hib_is_grader": False,
        "solver_view": {
            "user_goal": (
                f"Capability {cap} is PI-signed blocked or external acquisition blocked; "
                "abstain rather than fabricate native replay."
            ),
            "allowed_tools": ["inspect_artifact", "ask_user", "submit"],
            "allowed_edge_ids": [cap],
        },
        "reference_view": {
            "expected_action": "abstain",
            "no_path_certificate": {
                "reason": "hcg_a_abstain",
                "capability_id": cap,
                "hcg_a_verdict": verdict,
                "dual_witness_required": False,
            },
            "checker": {
                "structure": {"checker_id": "seven_track_abstain_v1"},
                "execution": {
                    "mode": "abstain",
                    "forbid_reference_score_fallback": True,
                },
            },
        },
        "metadata": {
            "hcg_a_abstain": True,
            "pi_signed": True,
            "not_headline_pass": True,
            "capability_id": cap,
        },
    }


def ensure_abstain_taskpacks() -> None:
    from hazardweaver.hwb.evaluators.seven_track_finish_line_c_common import ABSTAIN_CAPS
    from hazardweaver.hwa.runtime.hcg_a_audit_authority_v1 import load_hcg_a_audit_record

    track_by_cap = {
        "CAP-MH2-06": "MH-2",
        "CAP-MH3-02": "MH-3",
        "CAP-MH3-03": "MH-3",
        "CAP-MH3-04": "MH-3",
        "CAP-MH4-R04": "MH-4",
        "CAP-MH4-R05": "MH-4",
    }
    for cap in sorted(ABSTAIN_CAPS):
        track = track_by_cap[cap]
        audit = load_hcg_a_audit_record(track, cap) or {}
        verdict = str(audit.get("verdict") or "PI_SIGNED")
        payload = build_abstain_taskpack(capability_id=cap, track=track, verdict=verdict)
        slug = cap.lower().replace("cap-", "").replace("-", "_")
        out = TASKPACK_DIR / f"hwb_{slug}_abstain_v1.json"
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ensure_all_parametric_taskpacks() -> None:
    for track in TRACK_SEAL_CONFIG:
        ensure_parametric_taskpack_files(track)
    ensure_abstain_taskpacks()
