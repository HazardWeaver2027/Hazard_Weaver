"""Table A v2 — shared loaders, admission filters, and selection helpers."""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import yaml

from hazardweaver.hcg.registry.paper_system_scope_v1 import ELEVEN_TRACKS, REGISTRY_BRIDGE_CAPABILITY_IDS
from hazardweaver.hwb.registry.g6_hard_anchor_meta import TABULAR_SUPERVISED_ANCHORS

ROOT = Path(__file__).resolve().parents[3]
TABLE_A_DIR = ROOT / "docs/engineering/benchmark/table_a_v2"
CAPABILITIES_PATH = ROOT / "docs/final_four/HCG/manifests/capabilities.jsonl"
ANCHORS_PATH = ROOT / "docs/final_four/HCG/manifests/benchmark_anchors.jsonl"
MATRIX_PATH = ROOT / "docs/engineering/hcg/HCG_CAPABILITY_SYSTEM_MATRIX_v1.csv"
BRIDGE_PATH = Path(__file__).resolve().parent / "table_a_legacy_cap_bridge_v1.yaml"
HEADLINE_INVENTORY = ROOT / "runs/hwb/iclr_benchmark/headline_inventory_v1.jsonl"
SAMPLING_SEED = 20260726

SPLIT_PRIORITY = {"official_test": 0, "hwb_holdout": 1, "atlas": 2}

CHAIN_CAPABILITY_IDS: Set[str] = {
    "CAP-MH1-05",
    "CAP-MH2-06",
    "CAP-MH3-04",
    "CAP-E1E3-01",
    "CAP-MH4-R01",
    "CAP-MH4-R05",
}

SUBSTRATE_ONLY_IDS: Set[str] = {"CAP-MH4-R01"}

BLOCKER_DENY: Set[str] = {
    "DEFERRED_P2",
    "RUNTIME_BLOCKED",
    "RUNTIME_BLOCKED_WINDOWS_ARCGIS",
    "PENDING_EXPOSURE_PIN",
}

CHAIN_ROUTE_FAMILIES: Set[str] = {
    "RF-HAZARD-TO-NETWORK-MODEL-CHAI",
    "RF-EMPIRICAL-INITIATION--PROCES",
    "RF-ONE-WAY-COUPLED-COASTAL-INLA",
    "RF-CLASSICAL-SIGNAL-TO-CHAIN",
    "RF-MH4-COUPLED-IMPACT",
    "RF-MH4-SUBSTRATE",
}

METRIC_ALIASES = {
    "test average precision": "AP",
    "average precision": "AP",
    "auprc": "AUPRC",
    "pick f1 and timing residual": "F1",
    "event precision/recall/f1": "F1",
    "station log residual/rmse": "RMSE",
    "official evaluate_tracks.py lead-time errors": "RMSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "maximum mae": "MAE",
    "accuracy": "accuracy",
    "pixel-hit score": "CSI",
    "brier/auprc/calibration": "AUPRC",
    "mh4_tc_building_loss": "log_MAE",
    "log_mae": "log_MAE",
}


@dataclass
class CapMatrixRow:
    taskpack_id: str
    capability_id: str
    passed: bool
    native_inference_ready: bool
    blocker_code: str
    validation_tier: str
    headline_eligible: bool


@dataclass
class TableACandidate:
    instance_id: str
    track: str
    taskpack_id: str
    capability_id: str
    anchor_id: str
    scenario_id: str
    split: str
    metric: str
    admission_tags: List[str] = field(default_factory=list)
    legacy_g6_task_id: Optional[str] = None
    aide_native_eligible: bool = False
    ds_agent_native_eligible: bool = True
    matrix_passed: bool = False
    blocker_code: str = ""
    priority_score: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "track": self.track,
            "taskpack_id": self.taskpack_id,
            "capability_id": self.capability_id,
            "anchor_id": self.anchor_id,
            "scenario_id": self.scenario_id,
            "split": self.split,
            "metric": self.metric,
            "admission_tags": list(self.admission_tags),
            "legacy_g6_task_id": self.legacy_g6_task_id,
            "aide_native_eligible": self.aide_native_eligible,
            "ds_agent_native_eligible": self.ds_agent_native_eligible,
            "matrix_passed": self.matrix_passed,
            "blocker_code": self.blocker_code,
            "priority_score": self.priority_score,
        }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_capabilities() -> List[Dict[str, Any]]:
    caps = _load_jsonl(CAPABILITIES_PATH)
    for bridge_id in REGISTRY_BRIDGE_CAPABILITY_IDS:
        caps.append(
            {
                "capability_id": bridge_id,
                "taskpack_id": "WF-3",
                "track_id": "TR-WF3",
                "anchor_id": "AN-WF-WSTS",
                "route_family_id": "RF-LEARNED-EO-SPREAD",
                "name": "Learned EO spread bridge",
                "paper_status": "BRIDGE",
            }
        )
    return caps


def load_benchmark_anchors() -> Dict[str, Dict[str, Any]]:
    return {row["anchor_id"]: row for row in _load_jsonl(ANCHORS_PATH)}


def load_matrix() -> Dict[str, CapMatrixRow]:
    out: Dict[str, CapMatrixRow] = {}
    if not MATRIX_PATH.is_file():
        return out
    with MATRIX_PATH.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            cap = (row.get("capability_id") or "").strip()
            if not cap:
                continue
            out[cap] = CapMatrixRow(
                taskpack_id=(row.get("taskpack_id") or "").strip(),
                capability_id=cap,
                passed=(row.get("passed") or "").strip().lower() == "true",
                native_inference_ready=(row.get("native_inference_ready") or "").strip().lower() == "true",
                blocker_code=(row.get("blocker_code") or "").strip(),
                validation_tier=(row.get("validation_tier") or "").strip(),
                headline_eligible=(row.get("headline_eligible") or "").strip().lower() == "true",
            )
    return out


def load_legacy_bridge() -> Dict[str, Dict[str, Any]]:
    if not BRIDGE_PATH.is_file():
        return {}
    data = yaml.safe_load(BRIDGE_PATH.read_text(encoding="utf-8")) or {}
    return dict((data.get("anchors") or {}))


def normalize_metric(raw: str) -> str:
    key = (raw or "").strip().lower()
    if not key:
        return "unknown"
    for pattern, alias in METRIC_ALIASES.items():
        if pattern in key:
            return alias
    token = re.split(r"[,;/]", key)[0].strip()
    return token.upper() if token.isalpha() and len(token) <= 6 else token


def metric_for_cap(cap: Mapping[str, Any], anchors: Mapping[str, Dict[str, Any]]) -> str:
    anchor = anchors.get(str(cap.get("anchor_id") or ""), {})
    return normalize_metric(str(anchor.get("official_metric_evaluator") or "unknown"))


def cap_admission_denied(
    cap: Mapping[str, Any],
    matrix: Mapping[str, CapMatrixRow],
) -> Optional[str]:
    cap_id = str(cap.get("capability_id") or "")
    if cap_id in CHAIN_CAPABILITY_IDS or cap_id in SUBSTRATE_ONLY_IDS:
        return "chain_or_substrate_only"
    rf = str(cap.get("route_family_id") or "")
    if rf in CHAIN_ROUTE_FAMILIES:
        return "chain_route_family"
    if "chain" in str(cap.get("name") or "").lower() and "solver chain" in str(cap.get("name") or "").lower():
        return "chain_in_name"
    paper_status = str(cap.get("paper_status") or "").upper()
    if paper_status in {"DEFERRED_P2", "PENDING_EXPOSURE_PIN"}:
        return f"paper_status_{paper_status}"
    mrow = matrix.get(cap_id)
    if mrow is None:
        return None
    if mrow.blocker_code in BLOCKER_DENY:
        return f"blocker_{mrow.blocker_code}"
    soft_blockers = {"", "VALIDATION_REPLAY", "VR_PROMOTE_PENDING", "ACTIVE_JOB"}
    if mrow.blocker_code not in soft_blockers and not mrow.passed and not mrow.native_inference_ready:
        return "matrix_not_ready"
    return None


def cap_priority(cap_id: str, matrix: Mapping[str, CapMatrixRow], split: str) -> int:
    mrow = matrix.get(cap_id)
    score = SPLIT_PRIORITY.get(split, 3)
    if mrow and mrow.passed:
        score -= 10
    if mrow and mrow.native_inference_ready:
        score -= 5
    if mrow and mrow.blocker_code == "VALIDATION_REPLAY":
        score += 3
    return score


def _scenario_splits(cap_index: int, n_scenarios: int = 3) -> List[str]:
    splits = ["official_test", "hwb_holdout", "atlas"]
    out: List[str] = []
    for i in range(n_scenarios):
        out.append(splits[i % len(splits)])
    return out


def build_instance_id(taskpack_id: str, capability_id: str, scenario_id: str) -> str:
    slug = taskpack_id.replace("-", "").upper()
    cap_slug = capability_id.replace("-", "_")
    scen_slug = scenario_id.replace("-", "_").replace("/", "_")
    return f"TA_{slug}_{cap_slug}_{scen_slug}"


G6_TABULAR_CAP_MAP = {
    "CAP-MH1-02": "A_PFDF_ASSESS_V1",
    "CAP-DROUT-04": "A_DH_GHCND_SPI_V1",
    "CAP-L2-04": "A_LS_IT_SU_V1",
    "CAP-WF3-02": "A_WF_PORTUGAL_V1",
    "CAP-WF3-03": "A_WF_ALGERIA_V1",
    "CAP-WF3-01": "A_WF_MTBS_EVENT_V1",
}


def legacy_anchor_for_cap(cap_id: str) -> Optional[str]:
    return G6_TABULAR_CAP_MAP.get(cap_id)


def enumerate_cap_scenarios(
    cap: Mapping[str, Any],
    anchors: Mapping[str, Dict[str, Any]],
    matrix: Mapping[str, CapMatrixRow],
    *,
    scenarios_per_cap: int = 3,
    cap_ordinal: int = 0,
) -> List[TableACandidate]:
    cap_id = str(cap.get("capability_id") or "")
    track = str(cap.get("taskpack_id") or "")
    anchor_id = str(cap.get("anchor_id") or "")
    metric = metric_for_cap(cap, anchors)
    mrow = matrix.get(cap_id)
    splits = _scenario_splits(cap_ordinal, scenarios_per_cap)
    candidates: List[TableACandidate] = []
    for idx, split in enumerate(splits):
        scenario_id = f"scen_{cap_id.lower().replace('-', '_')}_{split}_{idx:02d}"
        inst = TableACandidate(
            instance_id=build_instance_id(track, cap_id, scenario_id),
            track=track,
            taskpack_id=track,
            capability_id=cap_id,
            anchor_id=anchor_id,
            scenario_id=scenario_id,
            split=split,
            metric=metric,
            admission_tags=[
                "single_cap",
                "fixed_native_metric",
                "no_multi_route",
                "no_coupling",
            ],
            aide_native_eligible=legacy_anchor_for_cap(cap_id) in TABULAR_SUPERVISED_ANCHORS,
            ds_agent_native_eligible=True,
            matrix_passed=bool(mrow.passed) if mrow else False,
            blocker_code=mrow.blocker_code if mrow else "",
            priority_score=cap_priority(cap_id, matrix, split),
        )
        candidates.append(inst)
    return candidates


def build_candidate_pool(
    *,
    scenarios_per_cap: int = 3,
) -> Tuple[List[TableACandidate], Dict[str, Any]]:
    caps = load_capabilities()
    anchors = load_benchmark_anchors()
    matrix = load_matrix()
    pool: List[TableACandidate] = []
    denied: Dict[str, int] = {}
    by_track: Dict[str, int] = {}
    by_cap: Dict[str, int] = {}

    cap_by_track: Dict[str, List[Dict[str, Any]]] = {t: [] for t in ELEVEN_TRACKS}
    for cap in caps:
        track = str(cap.get("taskpack_id") or "")
        if track in cap_by_track:
            cap_by_track[track].append(cap)

    for track in ELEVEN_TRACKS:
        for ord_idx, cap in enumerate(sorted(cap_by_track[track], key=lambda c: c.get("capability_id", ""))):
            cap_id = str(cap.get("capability_id") or "")
            reason = cap_admission_denied(cap, matrix)
            if reason:
                denied[reason] = denied.get(reason, 0) + 1
                continue
            rows = enumerate_cap_scenarios(
                cap,
                anchors,
                matrix,
                scenarios_per_cap=scenarios_per_cap,
                cap_ordinal=ord_idx,
            )
            pool.extend(rows)
            by_track[track] = by_track.get(track, 0) + len(rows)
            by_cap[cap_id] = by_cap.get(cap_id, 0) + len(rows)

    stats = {
        "n_candidates": len(pool),
        "by_track": by_track,
        "unique_caps": len(by_cap),
        "denied_reasons": denied,
        "tracks": list(ELEVEN_TRACKS),
    }
    return pool, stats


def load_headline_instance_ids() -> Set[str]:
    ids: Set[str] = set()
    if not HEADLINE_INVENTORY.is_file():
        return ids
    for row in _load_jsonl(HEADLINE_INVENTORY):
        iid = str(row.get("instance_id") or "")
        if iid:
            ids.add(iid)
    return ids


def select_inventory(
    pool: Sequence[TableACandidate],
    *,
    target_per_track: int = 6,
    max_per_track: int = 7,
    total_targets: Sequence[int] = (66, 72, 77),
    seed: int = SAMPLING_SEED,
    headline_ids: Optional[Set[str]] = None,
    prefer_legacy: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return tiered inventory selections keyed by total target N."""
    headline_ids = headline_ids if headline_ids is not None else load_headline_instance_ids()
    rng = random.Random(seed)
    by_track: Dict[str, List[TableACandidate]] = {t: [] for t in ELEVEN_TRACKS}
    for row in pool:
        if row.instance_id in headline_ids:
            continue
        if row.track in by_track:
            by_track[row.track].append(row)

    legacy_keep_caps: Set[str] = set()
    if prefer_legacy:
        for item in prefer_legacy:
            if str(item.get("verdict") or "").upper() == "KEEP":
                for cap in item.get("capability_ids") or []:
                    legacy_keep_caps.add(str(cap))

    def pick_for_track(track: str, quota: int) -> List[TableACandidate]:
        rows = list(by_track.get(track) or [])
        rows.sort(key=lambda r: (r.priority_score, r.capability_id, r.scenario_id))
        chosen: List[TableACandidate] = []
        used_caps: Set[str] = set()

        def sort_key(r: TableACandidate) -> Tuple[int, int, str, str]:
            legacy_boost = 0 if r.capability_id in legacy_keep_caps else 1
            cap_dup = 1 if r.capability_id in used_caps else 0
            return (legacy_boost, cap_dup, r.priority_score, r.capability_id, r.scenario_id)

        pool_rows = sorted(rows, key=sort_key)
        for row in pool_rows:
            if len(chosen) >= quota:
                break
            if row.instance_id in {c.instance_id for c in chosen}:
                continue
            chosen.append(row)
            used_caps.add(row.capability_id)
        if len(chosen) < quota:
            for row in pool_rows:
                if len(chosen) >= quota:
                    break
                if row in chosen:
                    continue
                chosen.append(row)
        return chosen[:quota]

    tiers: Dict[str, Any] = {}
    for total_target in total_targets:
        per_track = {t: target_per_track for t in ELEVEN_TRACKS}
        selected: List[TableACandidate] = []
        track_counts: Dict[str, int] = {}
        gaps: Dict[str, int] = {}

        for track in ELEVEN_TRACKS:
            picked = pick_for_track(track, per_track[track])
            selected.extend(picked)
            track_counts[track] = len(picked)
            if len(picked) < per_track[track]:
                gaps[track] = per_track[track] - len(picked)

        deficit = total_target - len(selected)
        surplus_tracks = sorted(
            ELEVEN_TRACKS,
            key=lambda t: len(by_track.get(t) or []),
            reverse=True,
        )
        while deficit > 0:
            progressed = False
            for track in surplus_tracks:
                if track_counts.get(track, 0) >= max_per_track:
                    continue
                extra = pick_for_track(track, track_counts.get(track, 0) + 1)
                if len(extra) > track_counts.get(track, 0):
                    new_row = extra[-1]
                    if new_row not in selected:
                        selected.append(new_row)
                        track_counts[track] = track_counts.get(track, 0) + 1
                        deficit -= 1
                        progressed = True
                        if deficit <= 0:
                            break
            if not progressed:
                break

        anchor_splits: Dict[str, int] = {}
        for row in selected:
            anchor_splits[row.split] = anchor_splits.get(row.split, 0) + 1

        tiers[str(total_target)] = {
            "n_selected": len(selected),
            "track_counts": track_counts,
            "track_gaps": gaps,
            "unique_caps": len({r.capability_id for r in selected}),
            "anchor_split_counts": anchor_splits,
            "rows": [r.to_dict() for r in selected],
        }

    return {
        "seed": seed,
        "target_per_track": target_per_track,
        "max_per_track": max_per_track,
        "tiers": tiers,
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def table_a_paths() -> Dict[str, Path]:
    return {
        "dir": TABLE_A_DIR,
        "candidate_pool": TABLE_A_DIR / "candidate_pool.jsonl",
        "reuse_audit": TABLE_A_DIR / "reuse_audit_v1.json",
        "inventory": TABLE_A_DIR / "inventory_v1.jsonl",
        "manifest": TABLE_A_DIR / "manifest_v1.json",
        "report": TABLE_A_DIR / "TABLE_A_SELECTION_REPORT_v1.md",
        "runtime_inventory": ROOT / "runs/hwb/iclr_benchmark/table_a_inventory_v1.jsonl",
    }
