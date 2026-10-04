"""Select Atlas rows for TaskPack upgrade per ICLR v3 distribution targets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwb.admission.decision_layer_gate import evaluate_decision_layer_gate
from hazardweaver.hwb.metrics.atlas_diversity import load_atlas_manifest

ROOT = Path(__file__).resolve().parents[3]
TASKPACKS_DIR = ROOT / "hwb/registry/taskpacks"
DISTRIBUTION_CONFIG = (
    ROOT / "docs/engineering/benchmark/HWB_ICLR_HEADLINE_DISTRIBUTION_v1.json"
)
EXPERT_DATA = ROOT / "docs/engineering/benchmark/HWB_PAPER_EXPERT_DATA_v1.jsonl"


@dataclass
class UpgradeCandidate:
    scenario_id: str
    track: str
    taskpack_id: str
    routing_tier: str
    priority_score: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "track": self.track,
            "taskpack_id": self.taskpack_id,
            "routing_tier": self.routing_tier,
            "priority_score": self.priority_score,
            "reasons": self.reasons,
        }


def load_distribution_config(path: Optional[Path] = None) -> Dict[str, Any]:
    src = path or DISTRIBUTION_CONFIG
    if not src.is_file():
        return {
            "schema_version": "HWB_ICLR_HEADLINE_DISTRIBUTION_v1",
            "target_headline_n": {"min": 150, "max": 250},
            "flagship_tracks": ["MH-1", "FL-2"],
            "per_track_targets": {},
        }
    return json.loads(src.read_text(encoding="utf-8"))


def load_taskpack_index(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Index existing taskpacks by taskpack_id from expert data JSONL."""
    src = path or EXPERT_DATA
    index: Dict[str, Dict[str, Any]] = {}
    if not src.is_file():
        for fp in TASKPACKS_DIR.glob("*.json"):
            if fp.name.endswith("_refs.json"):
                continue
            row = json.loads(fp.read_text(encoding="utf-8"))
            tid = str(row.get("taskpack_id") or fp.stem)
            index[tid] = row
        return index

    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record_type") != "taskpack":
            continue
        tid = str(row.get("taskpack_id") or "")
        tp_path = ROOT / str(row.get("path") or "")
        if tp_path.is_file():
            index[tid] = json.loads(tp_path.read_text(encoding="utf-8"))
        else:
            index[tid] = row
    return index


def _track_priority(track: str, config: Mapping[str, Any]) -> float:
    flagship = set(config.get("flagship_tracks") or [])
    secondary = set(config.get("secondary_tracks") or [])
    if track in flagship:
        return 3.0
    if track in secondary:
        return 2.0
    return 1.0


def select_upgrade_candidates(
    *,
    atlas_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    taskpack_index: Optional[Mapping[str, Dict[str, Any]]] = None,
    distribution_config: Optional[Mapping[str, Any]] = None,
    max_candidates: int = 500,
) -> List[UpgradeCandidate]:
    """Rank existing TaskPacks + materialized Atlas rows for headline upgrade."""
    config = dict(distribution_config or load_distribution_config())
    index = dict(taskpack_index or load_taskpack_index())
    atlas = list(atlas_rows) if atlas_rows is not None else load_atlas_manifest()

    candidates: List[UpgradeCandidate] = []

    for tid, tp in index.items():
        gate = evaluate_decision_layer_gate(tp)
        if gate.capability_fitting_only:
            continue
        track = str(tp.get("headline_target") or "")
        score = _track_priority(track, config)
        if gate.routing_tier == "headline":
            score += 2.0
        if gate.difficulty_tier in {"L3", "L4"}:
            score += 1.5
        candidates.append(
            UpgradeCandidate(
                scenario_id=str((tp.get("solver_view") or {}).get("parameters", {}).get("scenario_id") or tid),
                track=track,
                taskpack_id=tid,
                routing_tier=gate.routing_tier,
                priority_score=score,
                reasons=[c.name for c in gate.criteria if c.passed],
            )
        )

    materialized_by_track: Dict[str, List[Mapping[str, Any]]] = {}
    for row in atlas:
        if not row.get("materialized"):
            continue
        track = str(row.get("track") or "")
        materialized_by_track.setdefault(track, []).append(row)

    for track, rows in materialized_by_track.items():
        quota = (config.get("per_track_targets") or {}).get(track, 0)
        if quota <= 0:
            continue
        for row in rows[:quota]:
            sid = str(row.get("scenario_id") or "")
            candidates.append(
                UpgradeCandidate(
                    scenario_id=sid,
                    track=track,
                    taskpack_id=f"atlas_upgrade:{track}:{sid}",
                    routing_tier="headline_candidate",
                    priority_score=_track_priority(track, config) + 0.5,
                    reasons=["materialized_atlas_seed"],
                )
            )

    candidates.sort(key=lambda c: (-c.priority_score, c.track, c.scenario_id))
    return candidates[:max_candidates]


def build_upgrade_report(
    *,
    max_candidates: int = 500,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build JSON report for Atlas→TaskPack upgrade planning."""
    config = load_distribution_config()
    cands = select_upgrade_candidates(max_candidates=max_candidates)
    headline = [c for c in cands if c.routing_tier in {"headline", "abstention_floor", "headline_candidate"}]
    report = {
        "schema_version": "HWB_ATLAS_UPGRADE_REPORT_v1",
        "distribution_config": config.get("schema_version"),
        "n_candidates": len(cands),
        "n_headline_eligible": len(headline),
        "per_track": {},
        "candidates": [c.to_dict() for c in cands],
    }
    per_track: Dict[str, int] = {}
    for c in headline:
        per_track[c.track] = per_track.get(c.track, 0) + 1
    report["per_track"] = dict(sorted(per_track.items()))

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def enforce_distribution(
    candidates: Sequence[UpgradeCandidate],
    *,
    distribution_config: Optional[Mapping[str, Any]] = None,
    max_total: int = 250,
) -> List[UpgradeCandidate]:
    """Apply per-track quotas from distribution config."""
    config = dict(distribution_config or load_distribution_config())
    targets: Dict[str, int] = dict(config.get("per_track_targets") or {})
    by_track: Dict[str, List[UpgradeCandidate]] = {}
    for c in candidates:
        by_track.setdefault(c.track, []).append(c)
    for track in by_track:
        by_track[track].sort(key=lambda x: -x.priority_score)

    selected: List[UpgradeCandidate] = []
    for track, quota in sorted(targets.items()):
        pool = by_track.get(track, [])
        selected.extend(pool[:quota])
    if len(selected) < int((config.get("target_headline_n") or {}).get("min", 150)):
        rest = [c for c in candidates if c not in selected]
        rest.sort(key=lambda x: -x.priority_score)
        selected.extend(rest[: max(0, int((config.get("target_headline_n") or {}).get("min", 150)) - len(selected))])
    return selected[:max_total]
