"""Expand parametric + Atlas rows into headline inventory candidates."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from hazardweaver.hwb.admission.decision_layer_gate import evaluate_decision_layer_gate
from hazardweaver.hwb.admission.gates import evaluate_gates
from hazardweaver.hwb.metrics.atlas_diversity import load_atlas_manifest
from hazardweaver.hwb.registry.atlas_taskpack_selector import load_distribution_config, load_taskpack_index
from hazardweaver.hwb.registry.track_parametric_resolver import REFS_BY_TASKPACK, resolve_reference_view

ROOT = Path(__file__).resolve().parents[3]
TASKPACKS_DIR = ROOT / "hwb/registry/taskpacks"

# Headline pool exclusions — engineering/runtime blocked (not model-policy failures).
# CAP-MH2-05: dynamic VBCI vendor check (see ``_vbci_engineering_blocked``).
HEADLINE_ENGINEERING_BLOCKED: Dict[str, str] = {}

# Prefix quarantine — FL-2 australia cohort: exception@step0 / limit_steps (not policy signal).
HEADLINE_ENGINEERING_BLOCKED_PREFIXES: Tuple[str, ...] = (
    "australia_2022_test_",
)


def _blocked_by_prefix(value: str) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    for prefix in HEADLINE_ENGINEERING_BLOCKED_PREFIXES:
        if text.startswith(prefix.lower()):
            return f"prefix_blocked:{prefix}"
    return None


def _vbci_engineering_blocked() -> bool:
    """True when VBCI vendor assets are missing ."""
    try:
        from hazardweaver.hcg.carp.scientific.mh2_vbci_env import verify_vbci_runtime

        return bool(verify_vbci_runtime(require_runner=False))
    except Exception:  # noqa: BLE001
        return True


def _is_vbci_cap(*, scenario_id: str, capability_id: str, instance_id: str) -> bool:
    return "CAP-MH2-05" in {scenario_id, capability_id} or "CAP-MH2-05" in instance_id


def is_headline_engineering_blocked(
    scenario_id: str,
    *,
    capability_id: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> bool:
    sid = str(scenario_id or "").strip()
    cid = str(capability_id or "").strip()
    iid = str(instance_id or "").strip()
    if sid in HEADLINE_ENGINEERING_BLOCKED:
        return True
    if cid in HEADLINE_ENGINEERING_BLOCKED:
        return True
    if _is_vbci_cap(scenario_id=sid, capability_id=cid, instance_id=iid) and _vbci_engineering_blocked():
        return True
    if _blocked_by_prefix(sid) or _blocked_by_prefix(iid):
        return True
    return False


def headline_block_reason(scenario_id: str, *, capability_id: Optional[str] = None, instance_id: Optional[str] = None) -> Optional[str]:
    sid = str(scenario_id or "").strip()
    cid = str(capability_id or "").strip()
    iid = str(instance_id or "").strip()
    if sid in HEADLINE_ENGINEERING_BLOCKED:
        return HEADLINE_ENGINEERING_BLOCKED[sid]
    if cid in HEADLINE_ENGINEERING_BLOCKED:
        return HEADLINE_ENGINEERING_BLOCKED[cid]
    if _is_vbci_cap(scenario_id=sid, capability_id=cid, instance_id=iid) and _vbci_engineering_blocked():
        return "VBCI vendor assets missing (ENGINEERING_BLOCKED)"
    pref = _blocked_by_prefix(sid) or _blocked_by_prefix(iid)
    if pref:
        return f"FL-2 australia engineering quarantine ({pref})"
    return None


TRACK_DEFAULT_TASKPACK: Dict[str, str] = {
    "MH-1": "hwb_mh1_atlas_state_variant_v1",
    "MH-2": "hwb_mh2_parametric_v1",
    "MH-3": "hwb_mh3_parametric_v1",
    "MH-4": "hwb_mh4_parametric_v1",
    "FL-2": "hwb_fl2_parametric_v1",
    "WF-3": "hwb_wf3_parametric_v1",
    "L2": "hwb_l2_parametric_v1",
    "E1-E3": "hwb_e1e3_parametric_v1",
    "TC-TRK": "hwb_tctrk_parametric_v1",
    "DR-OUT": "hwb_drout_parametric_v1",
    "HW-MED": "hwb_hw_med_parametric_v1",
}

TIER_VARIANTS = ("L1", "L2", "L3", "L4")


@dataclass
class HeadlineCandidate:
    instance_id: str
    taskpack_id: str
    scenario_id: str
    track: str
    difficulty_tier: str
    source: str
    routing_tier: str
    expected_action: str = "solve"
    gate13: Dict[str, Any] = field(default_factory=dict)
    admission12: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "taskpack_id": self.taskpack_id,
            "scenario_id": self.scenario_id,
            "track": self.track,
            "difficulty_tier": self.difficulty_tier,
            "source": self.source,
            "routing_tier": self.routing_tier,
            "expected_action": self.expected_action,
            "gate13": self.gate13,
            "admission12": self.admission12,
        }


def _load_taskpack(taskpack_id: str) -> Dict[str, Any]:
    path = TASKPACKS_DIR / f"{taskpack_id}.json"
    if not path.is_file():
        raise FileNotFoundError(taskpack_id)
    return json.loads(path.read_text(encoding="utf-8"))


def _virtual_taskpack(
    base: Mapping[str, Any],
    *,
    scenario_id: str,
    difficulty_tier: str,
    track: str,
    split_role: str = "official_test",
) -> Dict[str, Any]:
    tp = deepcopy(dict(base))
    meta = dict(tp.get("metadata") or {})
    meta["difficulty_tier"] = difficulty_tier
    meta["generation_mode"] = meta.get("generation_mode") or "parametric"
    if track.startswith("MH-"):
        meta["mh_coupling"] = True
    tp["metadata"] = meta
    tp["headline_target"] = track
    solver = dict(tp.get("solver_view") or {})
    params = dict(solver.get("parameters") or {})
    params["scenario_id"] = scenario_id
    resolved_split = split_role
    if track == "FL-2":
        try:
            from hazardweaver.hcg.carp.scientific import fl2_data

            for sp in (
                "official_test_high",
                "hwb_holdout_high",
                "official_test",
                "hwb_holdout",
            ):
                if scenario_id in fl2_data.scenario_ids(sp):
                    resolved_split = sp
                    break
        except Exception:  # noqa: BLE001
            pass
    params["split"] = resolved_split
    solver["parameters"] = params
    tp["solver_view"] = solver
    tp["reference_view"] = resolve_reference_view(tp, scenario_id=scenario_id)
    cutoff = dict(tp.get("cutoff_manifest") or meta.get("cutoff_manifest") or {})
    if not cutoff.get("decision_time"):
        cutoff["decision_time"] = f"{scenario_id}T00:00:00Z"
        cutoff["issue_yyyymm"] = scenario_id[:7] if len(scenario_id) >= 7 else scenario_id
    tp["cutoff_manifest"] = cutoff
    return tp


def _evaluate_candidate(
    tp: Mapping[str, Any],
    *,
    instance_id: str,
    scenario_id: str,
    track: str,
    difficulty_tier: str,
    source: str,
) -> Optional[HeadlineCandidate]:
    cap_id = str(
        (tp.get("solver_view") or {}).get("parameters", {}).get("capability_id")
        or scenario_id
        or ""
    ).strip()
    if is_headline_engineering_blocked(scenario_id, capability_id=cap_id, instance_id=instance_id):
        return None
    gate13 = evaluate_decision_layer_gate(tp)
    if gate13.capability_fitting_only:
        return None
    ref = tp.get("reference_view") or {}
    expected = str(ref.get("expected_action") or "solve")
    candidate_row = {
        "instance_id": instance_id,
        "scenario_id": scenario_id,
        "decision_time": (tp.get("cutoff_manifest") or {}).get("decision_time"),
        "cutoff_manifest": tp.get("cutoff_manifest") or {},
        "evaluator_bundle_ref": ref.get("refs_sidecar") or f"hwb/registry/taskpacks/{tp.get('taskpack_id')}_refs.json",
        "materialized": True,
        "split_role": (tp.get("solver_view") or {}).get("parameters", {}).get("split", "official_test"),
    }
    adm12 = evaluate_gates(candidate_row, taskpack=tp)
    return HeadlineCandidate(
        instance_id=instance_id,
        taskpack_id=str(tp.get("taskpack_id") or ""),
        scenario_id=scenario_id,
        track=track,
        difficulty_tier=difficulty_tier,
        source=source,
        routing_tier=gate13.routing_tier,
        expected_action=expected,
        gate13=gate13.to_dict(),
        admission12=adm12.to_dict(),
    )


def _iter_parametric_refs() -> Iterator[Tuple[str, str, str]]:
    for taskpack_id, refs_path in REFS_BY_TASKPACK.items():
        if not refs_path.is_file():
            continue
        refs = json.loads(refs_path.read_text(encoding="utf-8"))
        track = str(refs.get("track") or "")
        if not track:
            base = _load_taskpack(taskpack_id)
            track = str(base.get("headline_target") or "")
        for scenario_id in (refs.get("scenarios") or {}):
            yield taskpack_id, str(scenario_id), track or "unknown"


def _tier_for_track(track: str, config: Mapping[str, Any], variant_idx: int) -> str:
    flagship = set(config.get("flagship_tracks") or [])
    secondary = set(config.get("secondary_tracks") or [])
    if track in flagship:
        mix = config.get("difficulty_mix", {}).get("flagship", {})
        order = ["L3", "L3", "L4", "L4", "L1", "L2"]
    elif track in secondary:
        mix = config.get("difficulty_mix", {}).get("secondary", {})
        order = ["L3", "L3", "L4", "L1", "L2", "L2"]
    else:
        mix = config.get("difficulty_mix", {}).get("breadth", {})
        order = ["L1", "L1", "L2", "L2", "L4", "L1"]
    _ = mix
    return order[variant_idx % len(order)]


def enumerate_headline_candidates(
    *,
    distribution_config: Optional[Mapping[str, Any]] = None,
    atlas_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    taskpack_index: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> List[HeadlineCandidate]:
    """Enumerate all headline-eligible virtual instances before quota sampling."""
    config = dict(distribution_config or load_distribution_config())
    atlas = list(atlas_rows) if atlas_rows is not None else load_atlas_manifest()
    index = dict(taskpack_index or load_taskpack_index())
    out: List[HeadlineCandidate] = []
    seen: set[str] = set()

    def add(cand: Optional[HeadlineCandidate]) -> None:
        if cand is None or cand.instance_id in seen:
            return
        if cand.routing_tier == "atlas":
            return
        seen.add(cand.instance_id)
        out.append(cand)

    for taskpack_id, scenario_id, track in _iter_parametric_refs():
        if not track or track == "unknown":
            try:
                base = _load_taskpack(taskpack_id)
                track = str(base.get("headline_target") or "")
            except FileNotFoundError:
                continue
        try:
            base = _load_taskpack(taskpack_id)
        except FileNotFoundError:
            continue
        tier = _tier_for_track(track, config, len(out))
        tp = _virtual_taskpack(base, scenario_id=scenario_id, difficulty_tier=tier, track=track)
        iid = f"param:{taskpack_id}:{scenario_id}"
        add(_evaluate_candidate(tp, instance_id=iid, scenario_id=scenario_id, track=track, difficulty_tier=tier, source="parametric_refs"))

    for row in atlas:
        if not row.get("materialized"):
            continue
        track = str(row.get("track") or "")
        scenario_id = str(row.get("scenario_id") or "")
        taskpack_id = TRACK_DEFAULT_TASKPACK.get(track)
        if not taskpack_id:
            continue
        try:
            base = _load_taskpack(taskpack_id)
        except FileNotFoundError:
            continue
        tier = _tier_for_track(track, config, hash(scenario_id) % 6)
        split_role = str(row.get("split_role") or "official_test")
        tp = _virtual_taskpack(base, scenario_id=scenario_id, difficulty_tier=tier, track=track, split_role=split_role)
        iid = f"atlas:{track}:{scenario_id}"
        add(_evaluate_candidate(tp, instance_id=iid, scenario_id=scenario_id, track=track, difficulty_tier=tier, source="atlas_materialized"))

    for tid, tp_base in index.items():
        track = str(tp_base.get("headline_target") or "")
        scenario_id = str((tp_base.get("solver_view") or {}).get("parameters", {}).get("scenario_id") or tid)
        # Template taskpack rows use CAP-PLACEHOLDER; real instances are param:/atlas:/variant:.
        if scenario_id == "CAP-PLACEHOLDER":
            continue
        tier = str((tp_base.get("metadata") or {}).get("difficulty_tier") or _tier_for_track(track, config, 0))
        tp = deepcopy(tp_base)
        iid = f"taskpack:{tid}"
        add(_evaluate_candidate(tp, instance_id=iid, scenario_id=scenario_id, track=track, difficulty_tier=tier, source="existing_taskpack"))

    # Tier-shell variants for tracks below quota (flagship + thin single-hazard pools).
    targets_map: Dict[str, int] = dict(config.get("per_track_targets") or {})
    variant_tracks = set(config.get("flagship_tracks") or [])
    variant_tracks.update(config.get("secondary_tracks") or [])
    variant_tracks.update(config.get("single_hazard_tracks") or [])
    for track in sorted(variant_tracks):
        taskpack_id = TRACK_DEFAULT_TASKPACK.get(str(track))
        if not taskpack_id:
            continue
        track_count = sum(1 for c in out if c.track == track)
        quota = int(targets_map.get(track, 0))
        if track_count >= quota:
            continue
        try:
            base = _load_taskpack(taskpack_id)
            refs_path = REFS_BY_TASKPACK.get(taskpack_id)
            if not refs_path or not refs_path.is_file():
                continue
            refs = json.loads(refs_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        for scenario_id in refs.get("scenarios") or {}:
            coupling_seeds = ["base", "tc_coupling", "fl_coupling"] if track == "MH-2" else ["base"]
            for seed in coupling_seeds:
                for tier in TIER_VARIANTS:
                    sid = f"{scenario_id}__{tier}" if seed == "base" else f"{scenario_id}__{tier}__{seed}"
                    tp = _virtual_taskpack(base, scenario_id=sid, difficulty_tier=tier, track=track)
                    if seed != "base":
                        meta = dict(tp.get("metadata") or {})
                        meta["mh_coupling"] = True
                        meta["coupling_seed"] = seed
                        tp["metadata"] = meta
                    iid = f"variant:{taskpack_id}:{sid}"
                    add(
                        _evaluate_candidate(
                            tp,
                            instance_id=iid,
                            scenario_id=sid,
                            track=track,
                            difficulty_tier=tier,
                            source="tier_variant",
                        )
                    )

    return out


def _is_abstain(c: HeadlineCandidate) -> bool:
    return c.expected_action in {"abstain", "clarify"}


def _trim_abstention(selected: List[HeadlineCandidate], cap: float) -> List[HeadlineCandidate]:
    max_abstain = max(1, int(len(selected) * cap)) if selected else 1
    abstain = [c for c in selected if _is_abstain(c)]
    if len(abstain) <= max_abstain:
        return selected
    drop_ids = {c.instance_id for c in abstain[max_abstain:]}
    return [c for c in selected if c.instance_id not in drop_ids]


def _backfill_track_quotas(
    selected: List[HeadlineCandidate],
    candidates: Sequence[HeadlineCandidate],
    targets: Mapping[str, int],
) -> List[HeadlineCandidate]:
    selected_ids = {c.instance_id for c in selected}
    by_track_sel: Dict[str, int] = {}
    for c in selected:
        by_track_sel[c.track] = by_track_sel.get(c.track, 0) + 1

    by_track_pool: Dict[str, List[HeadlineCandidate]] = {}
    for c in candidates:
        by_track_pool.setdefault(c.track, []).append(c)
    for track in by_track_pool:
        by_track_pool[track].sort(
            key=lambda x: (_is_abstain(x), x.routing_tier != "headline", x.instance_id)
        )

    out = list(selected)
    for track, quota in sorted(targets.items()):
        need = quota - by_track_sel.get(track, 0)
        if need <= 0:
            continue
        for cand in by_track_pool.get(track, []):
            if cand.instance_id in selected_ids:
                continue
            out.append(cand)
            selected_ids.add(cand.instance_id)
            by_track_sel[track] = by_track_sel.get(track, 0) + 1
            need -= 1
            if need <= 0:
                break
    return out


def enforce_distribution(
    candidates: Sequence[HeadlineCandidate],
    *,
    distribution_config: Optional[Mapping[str, Any]] = None,
) -> List[HeadlineCandidate]:
    """Sample candidates to per-track quotas and global N with abstention cap."""
    config = dict(distribution_config or load_distribution_config())
    targets: Dict[str, int] = dict(config.get("per_track_targets") or {})
    n_min = int((config.get("target_headline_n") or {}).get("min", 150))
    n_max = int((config.get("target_headline_n") or {}).get("max", 250))
    abstain_cap = float(config.get("abstention_floor_max_fraction", 0.05))

    by_track: Dict[str, List[HeadlineCandidate]] = {}
    for c in candidates:
        by_track.setdefault(c.track, []).append(c)
    for track in by_track:
        by_track[track].sort(key=lambda x: (x.routing_tier != "headline", x.instance_id))

    selected: List[HeadlineCandidate] = []
    for track, quota in sorted(targets.items()):
        pool = by_track.get(track, [])
        selected.extend(pool[:quota])

    if len(selected) < n_min:
        remaining = [c for c in candidates if c not in selected]
        remaining.sort(key=lambda x: x.instance_id)
        selected.extend(remaining[: n_min - len(selected)])

    selected = selected[:n_max]

    selected = _trim_abstention(selected, abstain_cap)
    selected = _backfill_track_quotas(selected, candidates, targets)
    selected = selected[:n_max]

    flagship = set(config.get("flagship_tracks") or [])
    if sum(1 for c in selected if c.track in flagship) < int(config.get("flagship_combined_min", 80)):
        for track in flagship:
            extra = [c for c in candidates if c.track == track and c.instance_id not in {x.instance_id for x in selected}]
            for c in extra:
                selected.append(c)
                if sum(1 for x in selected if x.track in flagship) >= int(config.get("flagship_combined_min", 80)):
                    break
        selected = selected[:n_max]

    selected = _trim_abstention(selected, abstain_cap)
    selected = _backfill_track_quotas(selected, candidates, targets)
    return selected[:n_max]


def build_headline_inventory(
    *,
    distribution_config: Optional[Mapping[str, Any]] = None,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    config = dict(distribution_config or load_distribution_config())
    all_cands = enumerate_headline_candidates(distribution_config=config)
    selected = enforce_distribution(all_cands, distribution_config=config)
    per_track: Dict[str, int] = {}
    per_tier: Dict[str, int] = {}
    abstain_n = 0
    for c in selected:
        per_track[c.track] = per_track.get(c.track, 0) + 1
        per_tier[c.difficulty_tier] = per_tier.get(c.difficulty_tier, 0) + 1
        if c.expected_action in {"abstain", "clarify"}:
            abstain_n += 1

    report = {
        "schema_version": "HWB_HEADLINE_INVENTORY_v1",
        "n_total": len(selected),
        "n_candidates_enumerated": len(all_cands),
        "per_track": dict(sorted(per_track.items())),
        "per_tier": dict(sorted(per_tier.items())),
        "abstention_fraction": abstain_n / len(selected) if selected else 0.0,
        "flagship_tracks": config.get("flagship_tracks"),
        "flagship_combined_min": config.get("flagship_combined_min"),
        "distribution_profile": config.get("distribution_profile"),
        "records": [c.to_dict() for c in selected],
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in report["records"]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        meta_path = out_path.with_suffix(".meta.json")
        meta = {k: v for k, v in report.items() if k != "records"}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return report
