"""Multi-track vertical slice registry — FL-2 sealed; MH-1 / DR-OUT / TC-TRK fusion B."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Mapping, Optional

from hazardweaver.hcg.runtime.portfolio_probe_resolver import PFDF_OFFICIAL_CAPS

from hazardweaver.hwa.runtime.seven_track_finish_line_c_common import (
    SEVEN_TRACK_ROUTE_BY_NAME,
    SEVEN_TRACKS,
    TRACK_SLUG,
    official_caps_for_track,
    track_slug,
)

MH1_PORTFOLIO_CAPS: FrozenSet[str] = frozenset(PFDF_OFFICIAL_CAPS) | frozenset(
    {"pfdf_burn_volume_cascade"}
)
MH1_HEADLINE_CAPS: FrozenSet[str] = frozenset(
    f"CAP-MH1-{i:02d}" for i in range(1, 6)
)
MH1_OFFICIAL_CAPS: FrozenSet[str] = MH1_PORTFOLIO_CAPS | MH1_HEADLINE_CAPS
MH1_FINISH_LINE_C_CAPS: FrozenSet[str] = MH1_HEADLINE_CAPS
DEFAULT_MH1_SCIENTIFIC_ROOT = "runs/carp/scientific/MH-1"
MH1_ROUTE_ID = "route:pfdf"
MH1_CRC_EXPERIMENT = "mh1_pilot_crc_v1"
DR_OUT_OFFICIAL_CAPS: FrozenSet[str] = frozenset(
    {
        "CAP-DROUT-01",
        "CAP-DROUT-02",
        "CAP-DROUT-03",
        "CAP-DROUT-04",
        "CAP-DROUT-05",
    }
)
TC_TRK_OFFICIAL_CAPS: FrozenSet[str] = frozenset(
    {
        "CAP-TCTRK-01",
        "CAP-TCTRK-02",
        "CAP-TCTRK-03",
        "CAP-TCTRK-04",
        "CAP-TCTRK-05",
        "CAP-TCTRK-06",
    }
)

PFDF_DEFAULT_RECORD_ID = "Thomas_Thomas1_208"
PFDF_PRITHVI_FINISH_LINE_B_RECORD = "Apple_Apple1_0"
DEFAULT_HCG_FINISH_LINE_B_MANIFEST = (
    "runs/carp/hcg_pilot4_finish_line_b_manifest_v1.json"
)

FINISH_LINE_B_EXPAND_CAPS: Dict[str, FrozenSet[str]] = {
    "DR-OUT": frozenset(
        {"CAP-DROUT-02", "CAP-DROUT-03", "CAP-DROUT-04", "CAP-DROUT-05"}
    ),
    "TC-TRK": frozenset(
        {"CAP-TCTRK-03", "CAP-TCTRK-04", "CAP-TCTRK-05"}
    ),
}
FINISH_LINE_B_OPTIONAL_CAPS: Dict[str, FrozenSet[str]] = {
    "TC-TRK": frozenset({"CAP-TCTRK-06"}),
}
TC_TRK_SCIENTIFIC_REPLAY_CAPS: FrozenSet[str] = frozenset(
    {"CAP-TCTRK-03", "CAP-TCTRK-04", "CAP-TCTRK-05"}
)

TRACK_SLICE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "FL-2": {
        "status": "SEALED",
        "official_caps": sorted(
            {"CAP-FL2-04", "CAP-FL2-05", "CAP-FL2-06"}
        ),
        "evidence": "runs/hwa/fl2_pilot_submit_20260829_verify/",
        "hwb_table": "runs/hwb/fl2_main_table_v1/fl2_main_table_v1.json",
    },
    "PFDF": {
        "status": "ALIAS_MH1",
        "superseded_by": "MH-1",
        "official_caps": sorted(PFDF_OFFICIAL_CAPS),
        "evidence": "runs/carp/hcg_pfdf_track_readiness_v1.json",
        "finish_line_b_template": "hwa/runtime/pfdf_prithvi_pilot_e2e.py",
        "prithvi_record_id": PFDF_PRITHVI_FINISH_LINE_B_RECORD,
        "d_pfdf_1": "SRC_PILOT",
    },
    "MH-1": {
        "status": "FINISH_LINE_C_READY",
        "official_caps": sorted(MH1_OFFICIAL_CAPS),
        "headline_caps": sorted(MH1_HEADLINE_CAPS),
        "portfolio_caps": sorted(MH1_PORTFOLIO_CAPS),
        "evidence": "runs/carp/hcg_pfdf_track_readiness_v1.json",
        "hkc_binding": "pfdf_v1 · burn→volume preconditioning",
        "src_template": "hwa/runtime/pfdf_src_pilot_e2e.py",
        "prithvi_template": "hwa/runtime/pfdf_prithvi_pilot_e2e.py",
        "crc_template": "hwa/experiments/mh1_pilot_crc_v1.py",
        "finish_line_c_template": "hwa/runtime/mh1_finish_line_c_e2e.py",
        "finish_line_c_expand": sorted(MH1_FINISH_LINE_C_CAPS),
        "prithvi_record_id": PFDF_PRITHVI_FINISH_LINE_B_RECORD,
        "portfolio_route_id": MH1_ROUTE_ID,
        "scientific_root": DEFAULT_MH1_SCIENTIFIC_ROOT,
    },
    "DR-OUT": {
        "status": "FINISH_LINE_B_READY",
        "official_caps": sorted(DR_OUT_OFFICIAL_CAPS),
        "evidence": "runs/carp/hcg_drout_track_readiness_v1.json",
        "hkc_binding": "dr_out_v1 · 77 SRC cards",
        "src_template": "hwa/runtime/track_src_pilot_e2e.py",
        "finish_line_b_template": "hwa/runtime/track_finish_line_b_e2e.py",
        "finish_line_b_expand": sorted(FINISH_LINE_B_EXPAND_CAPS["DR-OUT"]),
    },
    "TC-TRK": {
        "status": "FINISH_LINE_B_READY",
        "official_caps": sorted(TC_TRK_OFFICIAL_CAPS),
        "evidence": "runs/carp/hcg_tctrk_track_readiness_v1.json",
        "hkc_binding": "tc_tctrk_v1 · 1552 SRC cards",
        "src_template": "hwa/runtime/track_src_pilot_e2e.py",
        "finish_line_b_template": "hwa/runtime/track_finish_line_b_e2e.py",
        "finish_line_b_expand": sorted(FINISH_LINE_B_EXPAND_CAPS["TC-TRK"]),
    },
}

_FINISH_LINE_C_TEMPLATE = "hwa/runtime/seven_track_finish_line_c_e2e.py"


def _seven_track_registry_entry(track: str) -> Dict[str, Any]:
    track_u = str(track).upper()
    official_caps: List[str] = []
    try:
        official_caps = sorted(official_caps_for_track(track_u))
    except FileNotFoundError:
        official_caps = []
    return {
        "status": "FINISH_LINE_C_READY",
        "official_caps": official_caps,
        "official_caps_source": "runs/carp/hcg_pilot7_finish_line_manifest_v1.json",
        "scientific_root": f"runs/carp/scientific/{track_u}",
        "finish_line_c_template": _FINISH_LINE_C_TEMPLATE,
        "hcg_fusion_a_manifest": "runs/carp/hcg_pilot7_finish_line_manifest_v1.json",
        "evidence": f"runs/carp/hcg_{TRACK_SLUG[track_u]}_readiness_v1.json",
    }


for _track in SEVEN_TRACKS:
    TRACK_SLICE_REGISTRY[_track] = _seven_track_registry_entry(_track)

TRACK_ROUTE_BY_NAME: Dict[str, str] = {
    "PFDF": MH1_ROUTE_ID,
    "MH-1": MH1_ROUTE_ID,
    "DR-OUT": "drout_official",
    "TC-TRK": "tctrk_official",
    **SEVEN_TRACK_ROUTE_BY_NAME,
}

FUSION_B_PILOT_CAPS: Dict[str, str] = {
    "PFDF": "pfdf_volume_gorr_v2",
    "MH-1": "pfdf_volume_gorr_v2",
    "DR-OUT": "CAP-DROUT-01",
    "TC-TRK": "CAP-TCTRK-01",
}


def _normalize_capability_id(track_u: str, capability_id: str) -> str:
    if track_u in {"PFDF", "MH-1"}:
        cid = str(capability_id).strip()
        if cid.upper().startswith("CAP-MH1-"):
            return cid.upper()
        return cid
    return str(capability_id).upper()


_WF3_LEGACY_PREFIXES = (
    "eval_wf_",
    "hw_wildfire_",
    "wf_firms_",
)


def track_from_capability(capability_id: str) -> Optional[str]:
    cid = str(capability_id or "").strip()
    cid_u = cid.upper()
    if cid_u.startswith("CAP-MH1-"):
        return "MH-1"
    if cid in MH1_PORTFOLIO_CAPS:
        return "MH-1"
    if any(cid.startswith(prefix) for prefix in _WF3_LEGACY_PREFIXES):
        return "WF-3"
    if cid_u.startswith("CAP-FL2-"):
        return "FL-2"
    if cid_u.startswith("CAP-DROUT-"):
        return "DR-OUT"
    if cid_u.startswith("CAP-TCTRK-"):
        return "TC-TRK"
    if cid_u.startswith("CAP-WF3-"):
        return "WF-3"
    if cid_u.startswith("CAP-L2-"):
        return "L2"
    if cid_u.startswith("CAP-E1E3-") or cid_u.startswith("CAP-EQ-"):
        return "E1-E3"
    if cid_u.startswith("CAP-HWMED-"):
        return "HW-MED"
    if cid_u.startswith("CAP-MH2-"):
        return "MH-2"
    if cid_u.startswith("CAP-MH3-"):
        return "MH-3"
    if cid_u.startswith("CAP-MH4-"):
        return "MH-4"
    return None


def is_multi_track_pilot_task(task: Mapping[str, Any]) -> bool:
    from hazardweaver.hwa.route_controller.seven_track_pilot_slice import is_seven_track_pilot_task

    if is_seven_track_pilot_task(task):
        return True
    domain = str(task.get("domain") or "").lower()
    family = str(task.get("task_family") or "").upper()
    tp = str(task.get("taskpack_id") or "").upper()
    cap = str(task.get("capability_id") or "")
    if any(x in domain for x in ("fl2", "drout", "drought", "tc", "cyclone", "pfdf", "mh1", "mh-1")):
        return True
    if family in {"FL-2", "PFDF", "MH-1", "DR-OUT", "TC-TRK"} or tp in {
        "FL-2",
        "PFDF",
        "MH-1",
        "DR-OUT",
        "TC-TRK",
    }:
        return True
    return track_from_capability(cap) is not None


def build_track_pilot_task(
    *,
    track: str,
    capability_id: str,
    task_id: str = "",
    scenario_id: str = "pilot_scenario_000",
    record_id: str = "",
) -> Dict[str, Any]:
    track_u = str(track).upper()
    cap = _normalize_capability_id(track_u, capability_id)
    meta = TRACK_SLICE_REGISTRY.get(track_u)
    if meta is None:
        raise KeyError(f"unknown track: {track}")
    if track_u in SEVEN_TRACKS:
        from hazardweaver.hwa.route_controller.seven_track_pilot_slice import build_seven_track_src_pilot_task

        official = set(official_caps_for_track(track_u))
        if cap not in official:
            raise ValueError(f"{cap} not in {track_u} official caps")
        return build_seven_track_src_pilot_task(
            track_u,
            task_id=task_id or f"{track_slug(track_u)}-flc-{cap.lower()}",
            capability_id=cap,
            scenario_id=scenario_id,
        )
    official = set(meta.get("official_caps") or [])
    if cap not in official:
        raise ValueError(f"{cap} not in {track_u} official caps")

    parameters: Dict[str, Any] = {
        "capability_id": cap,
        "scenario_id": scenario_id,
        "split": "official_test",
    }
    domain = track_u.lower().replace("-", "")
    hkc_family_id = None
    if track_u in {"PFDF", "MH-1"}:
        parameters["record_id"] = record_id or PFDF_DEFAULT_RECORD_ID
        domain = "mh1" if track_u == "MH-1" else "pfdf"
    elif track_u == "DR-OUT":
        from hazardweaver.hwa.route_controller.drout_pilot_slice import DR_OUT_PILOT_FAMILY

        hkc_family_id = DR_OUT_PILOT_FAMILY
        domain = "drout"
    elif track_u == "TC-TRK":
        from hazardweaver.hwa.route_controller.tctrk_pilot_slice import TC_TCTRK_PILOT_FAMILY

        hkc_family_id = TC_TCTRK_PILOT_FAMILY
        domain = "tctrk"

    solver_visible: Dict[str, Any] = {
        "route_family_id": f"{track_u.lower()}_official",
        "inputs": {},
        "parameters": parameters,
    }
    if hkc_family_id:
        solver_visible["hkc_family_id"] = hkc_family_id
        solver_visible["route_family_id"] = hkc_family_id

    return {
        "task_id": task_id or f"{track_u.lower()}-pilot-{cap.lower()}",
        "taskpack_id": track_u,
        "domain": domain,
        "task_family": track_u,
        "hkc_family_id": hkc_family_id or "",
        "capability_id": cap,
        "user_facing_goal": f"Run official {track_u} solver infer ({cap}) and submit solution.",
        "solver_visible": solver_visible,
    }


def multi_track_readiness() -> Dict[str, Any]:
    return {
        "schema": "hwa_multi_track_slice_readiness_v1",
        "tracks": TRACK_SLICE_REGISTRY,
        "replication_template": "hwa/runtime/fl2_pilot_e2e.py",
        "fusion_b_template": "hwa/runtime/track_pilot_e2e.py",
        "pfdf_template": "hwa/runtime/pfdf_src_pilot_e2e.py",
        "mh1_template": "hwa/runtime/pfdf_src_pilot_e2e.py",
        "mh1_prithvi_template": "hwa/runtime/pfdf_prithvi_pilot_e2e.py",
        "mh1_crc_template": "hwa/experiments/mh1_pilot_crc_v1.py",
        "drout_src_template": "hwa/runtime/track_src_pilot_e2e.py",
        "tctrk_src_template": "hwa/runtime/track_src_pilot_e2e.py",
        "finish_line_b_batch": "hwa/runtime/pilot4_finish_line_b_batch.py",
        "finish_line_b_manifest": DEFAULT_HCG_FINISH_LINE_B_MANIFEST,
    }
