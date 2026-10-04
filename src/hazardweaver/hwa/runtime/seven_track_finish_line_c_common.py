"""Shared helpers for HWA seven-track Fusion B (finish line C trajectories)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

DEFAULT_HCG_PILOT7_MANIFEST = Path("runs/carp/hcg_pilot7_finish_line_manifest_v1.json")
DEFAULT_HWA_7TRACK_AGGREGATE_MANIFEST = Path(
    "runs/hwa/hwa_7track_finish_line_manifest_v1.json"
)

SEVEN_TRACKS: tuple[str, ...] = (
    "WF-3",
    "L2",
    "E1-E3",
    "HW-MED",
    "MH-2",
    "MH-3",
    "MH-4",
)

TRACK_SLUG: Dict[str, str] = {
    "WF-3": "wf-3",
    "L2": "l2",
    "E1-E3": "e1-e3",
    "HW-MED": "hw-med",
    "MH-2": "mh-2",
    "MH-3": "mh-3",
    "MH-4": "mh-4",
}

SEVEN_TRACK_ROUTE_BY_NAME: Dict[str, str] = {
    "WF-3": "wf3_official",
    "L2": "l2_official",
    "E1-E3": "eq_official",
    "HW-MED": "hwmed_official",
    "MH-2": "mh2_official",
    "MH-3": "mh3_official",
    "MH-4": "mh4_official",
}

FINISH_LINE_C_CLAIM_TIER = "seven_track_finish_line_c_scientific"
SRC_REQUIRED_TRACKS = frozenset({"L2"})
OPTIONAL_SRC_TRACKS = frozenset({"WF-3"})
L2_CAPS = tuple(f"CAP-L2-{i:02d}" for i in range(1, 6))
HWB_HEADLINE_ELIGIBLE_TIERS = frozenset(
    {"L2_scientific_replay", "L2_src_and_scientific", "gpu_deferred_replay"}
)
DEFAULT_HWB_HANDOFF_MANIFEST = Path("runs/hwa/hwa_7track_hwb_handoff_v1.json")
DEFAULT_HCG_A_BLOCKERS_MANIFEST = Path("runs/hwa/hwa_7track_hcg_a_blockers_v1.json")
DEFAULT_HCG_A_READINESS_MANIFEST = Path("runs/carp/hcg_7track_scientific_readiness_v1.json")
DEFAULT_HWA_FUSION_B_SEAL_MANIFEST = Path("runs/hwa/hwa_7track_fusion_b_seal_v1.json")
DEFAULT_HWB_SEAL_SUMMARY = Path("runs/hwb/hwb_7track_seal_v1/7track_main_table_summary.json")
HCGA_EXPECTED_L2_TOTAL = 37


from hazardweaver.hwa.runtime.hcg_a_audit_authority_v1 import (
    abstain_fields_from_audit,
    hcg_a_completion_tier,
    hcg_a_is_l2_headline_cap,
    load_hcg_a_audit_record,
)


def track_slug(track: str) -> str:
    track_u = str(track).upper()
    if track_u not in TRACK_SLUG:
        raise KeyError(f"unknown seven-track: {track}")
    return TRACK_SLUG[track_u]


def default_out_root(track: str) -> Path:
    return Path("runs/hwa") / f"{track_slug(track)}_finish_line_c_v1"


def default_scientific_root(track: str) -> str:
    return f"runs/carp/scientific/{str(track).upper()}"


def load_hcg_pilot7_manifest(path: Path | str = DEFAULT_HCG_PILOT7_MANIFEST) -> Dict[str, Any]:
    man_path = Path(path)
    if not man_path.is_file():
        raise FileNotFoundError(f"missing fusion A manifest: {man_path}")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man.get("schema_version") != "HCG_PILOT7_FINISH_LINE_MANIFEST_v1":
        raise ValueError(f"unexpected manifest schema: {man.get('schema_version')}")
    return man


@lru_cache(maxsize=1)
def cached_pilot7_manifest() -> Dict[str, Any]:
    return load_hcg_pilot7_manifest()


def manifest_cap_row(
    manifest: Mapping[str, Any],
    *,
    track: str,
    capability_id: str,
) -> Optional[Dict[str, Any]]:
    track_u = str(track).upper()
    cap = str(capability_id).upper()
    block = (manifest.get("tracks") or {}).get(track_u) or {}
    for row in block.get("capabilities") or []:
        if str(row.get("capability_id") or "").upper() == cap:
            return dict(row)
    return None


def track_caps_from_manifest(
    manifest: Mapping[str, Any],
    track: str,
) -> List[Dict[str, Any]]:
    track_u = str(track).upper()
    block = (manifest.get("tracks") or {}).get(track_u) or {}
    return [dict(row) for row in block.get("capabilities") or []]


def official_caps_for_track(track: str, manifest: Optional[Mapping[str, Any]] = None) -> List[str]:
    man = manifest or cached_pilot7_manifest()
    return [str(r["capability_id"]) for r in track_caps_from_manifest(man, track)]


def is_seven_track_finish_line_c_cap(
    capability_id: str,
    *,
    manifest: Optional[Mapping[str, Any]] = None,
) -> bool:
    cap = str(capability_id).upper()
    man = manifest or cached_pilot7_manifest()
    for track in SEVEN_TRACKS:
        for row in track_caps_from_manifest(man, track):
            if str(row.get("capability_id") or "").upper() == cap:
                return True
    return False


def track_for_capability(
    capability_id: str,
    *,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    cap = str(capability_id).upper()
    man = manifest or cached_pilot7_manifest()
    for track in SEVEN_TRACKS:
        for row in track_caps_from_manifest(man, track):
            if str(row.get("capability_id") or "").upper() == cap:
                return track
    return None


def finish_line_c_ok_field(track: str) -> str:
    slug = track_slug(track).replace("-", "_")
    return f"{slug}_finish_line_c_ok"


def hwb_headline_eligible_for_tier(trajectory_tier: str) -> bool:
    return str(trajectory_tier) in HWB_HEADLINE_ELIGIBLE_TIERS


def _replay_exec_ok_from_row(
    manifest_row: Mapping[str, Any],
    *,
    scientific_dir: str = "",
) -> Optional[bool]:
    if manifest_row.get("replay_manifest_exec_ok") is not None:
        return bool(manifest_row["replay_manifest_exec_ok"])
    replay = manifest_row.get("replay_manifest")
    if isinstance(replay, Mapping) and replay.get("exec_ok") is not None:
        return bool(replay["exec_ok"])
    sci_dir = scientific_dir or str(manifest_row.get("scientific_dir") or "")
    if sci_dir:
        replay_path = Path(sci_dir) / "replay_manifest.json"
        if replay_path.is_file():
            data = json.loads(replay_path.read_text(encoding="utf-8"))
            if data.get("exec_ok") is not None:
                return bool(data["exec_ok"])
    return None


def _native_metrics_blocked(scientific_dir: str) -> bool:
    if not scientific_dir:
        return False
    metrics_path = Path(scientific_dir) / "native_metrics.json"
    if not metrics_path.is_file():
        return False
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return bool(metrics.get("blocked"))


def classify_trajectory_tier(
    manifest_row: Mapping[str, Any],
    *,
    src_out: Optional[Mapping[str, Any]] = None,
    sci_out: Optional[Mapping[str, Any]] = None,
    meta: Optional[Mapping[str, Any]] = None,
    scientific_dir: Optional[str] = None,
    l1_placeholder: bool = False,
    track: str = "",
    capability_id: str = "",
) -> str:
    """Honest trajectory tier for DL-110b — HCG-A audit is eligibility authority."""
    row = dict(manifest_row or {})
    meta_d = dict(meta or {})
    track_u = str(track or row.get("track") or meta_d.get("track") or "").upper()
    cap_id = str(capability_id or row.get("capability_id") or meta_d.get("capability_id") or "").upper()
    sci_dir = str(scientific_dir or row.get("scientific_dir") or "")

    audit = load_hcg_a_audit_record(track_u, cap_id) if track_u and cap_id else None
    if audit and str(audit.get("completion_tier") or "").upper() != "L2":
        return abstain_fields_from_audit(audit)["trajectory_tier"]

    completion_tier = hcg_a_completion_tier(track_u, cap_id, row) if cap_id else str(row.get("completion_tier") or "").upper()
    row["completion_tier"] = completion_tier
    metrics_path = Path(sci_dir) / "native_metrics.json" if sci_dir else None
    has_native_metrics = bool(metrics_path and metrics_path.is_file())

    if l1_placeholder or meta_d.get("l1_placeholder"):
        return "L1_placeholder"

    replay_exec_ok = _replay_exec_ok_from_row(row, scientific_dir=sci_dir)
    if replay_exec_ok is False:
        return "exec_ok_false_replay"

    if _native_metrics_blocked(sci_dir):
        return "L1_blocked_replay"

    if row.get("pi_boundary") and not has_native_metrics:
        return "L1_placeholder"

    if completion_tier != "L2" and not has_native_metrics:
        return "L1_placeholder"

    src_binding_only = bool(src_out and src_out.get("src_pilot_binding_only"))
    src_executed = bool(
        src_out
        and src_out.get("ok")
        and src_out.get("src_pilot")
        and not src_binding_only
    )
    scientific_replay = bool(sci_out and sci_out.get("scientific_replay"))

    if track_u == "L2":
        if src_binding_only:
            return "src_binding_only"
        if src_executed and scientific_replay:
            return "L2_src_and_scientific"

    if row.get("gpu_deferred") and scientific_replay:
        return "gpu_deferred_replay"

    if completion_tier == "L2" and scientific_replay and replay_exec_ok is not False:
        return "L2_scientific_replay"

    if scientific_replay and completion_tier == "L2":
        return "L2_scientific_replay"

    return "L1_placeholder"


def trajectory_tier_fields(
    manifest_row: Mapping[str, Any],
    *,
    src_out: Optional[Mapping[str, Any]] = None,
    sci_out: Optional[Mapping[str, Any]] = None,
    meta: Optional[Mapping[str, Any]] = None,
    scientific_dir: Optional[str] = None,
    l1_placeholder: bool = False,
    track: str = "",
    capability_id: str = "",
) -> Dict[str, Any]:
    track_u = str(track or manifest_row.get("track") or "").upper()
    cap_id = str(capability_id or manifest_row.get("capability_id") or "").upper()
    audit = load_hcg_a_audit_record(track_u, cap_id) if track_u and cap_id else None
    if audit and str(audit.get("completion_tier") or "").upper() != "L2":
        fields = abstain_fields_from_audit(audit)
        fields["src_executed"] = False
        fields["src_binding_only"] = False
        fields["scientific_replay"] = bool(sci_out and sci_out.get("scientific_replay"))
        return fields

    tier = classify_trajectory_tier(
        manifest_row,
        src_out=src_out,
        sci_out=sci_out,
        meta=meta,
        scientific_dir=scientific_dir,
        l1_placeholder=l1_placeholder,
        track=track_u,
        capability_id=cap_id,
    )
    src_binding_only = bool(src_out and src_out.get("src_pilot_binding_only"))
    src_executed = bool(
        src_out
        and src_out.get("ok")
        and src_out.get("src_pilot")
        and not src_binding_only
    )
    if l1_placeholder:
        scientific_replay = False
    elif sci_out is not None:
        scientific_replay = bool(sci_out.get("scientific_replay"))
    else:
        scientific_replay = bool(meta and meta.get("scientific_replay"))

    completion_tier = hcg_a_completion_tier(track_u, cap_id, manifest_row)
    headline_eligible = hcg_a_is_l2_headline_cap(track_u, cap_id, manifest_row) and (
        hwb_headline_eligible_for_tier(tier) or tier == "L2_scientific_replay"
    )

    return {
        "trajectory_tier": tier,
        "hwb_headline_eligible": headline_eligible,
        "src_executed": src_executed,
        "src_binding_only": src_binding_only,
        "scientific_replay": scientific_replay,
        "completion_tier": completion_tier,
        "hcg_a_abstain": False,
    }
