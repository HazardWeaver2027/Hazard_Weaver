"""HWA self-CBR — route warm-start from HWA's own verified execution history only.

Never reads competing baseline outputs (DL-142). Retrieval proposes candidates;
Π_adm + REMSA still gate commit.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BANK_REL = "runs/hwb/iclr_benchmark/hwa_self_cbr_bank_v1.jsonl"
_ELIGIBLE_TIERS = frozenset({"L1", "L2", "L3"})


@dataclass(frozen=True)
class HwaRouteCase:
    instance_id: str
    track: str
    tier: str
    capability_id: str
    route_id: str
    allowed_edge_signature: str
    pred_at_1x: bool
    source_out_root: str
    family_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def allowed_edge_signature(allowed: Sequence[str]) -> str:
    caps = sorted({str(e) for e in allowed if e and not str(e).startswith("schema_")})
    return "|".join(caps)


def self_cbr_enabled() -> bool:
    return str(os.environ.get("HWA_HEADLINE_SELF_CBR", "0")).lower() in ("1", "true", "yes")


def default_self_cbr_bank_path() -> Path:
    env = os.environ.get("HWA_SELF_CBR_BANK_PATH") or os.environ.get("ICLR_HWA_SELF_CBR_BANK")
    if env:
        return Path(env)
    return ROOT / DEFAULT_BANK_REL


def _opaque_id_for_instance(instance_id: str) -> str:
    from hazardweaver.hwa.experiments.headline_inventory_runner_v1 import opaque_id_for_instance

    return opaque_id_for_instance(instance_id)


def _workdir(out_root: Path, *, seed: int, condition: str, instance_id: str) -> Path:
    return out_root / f"seed_{seed}" / condition / _opaque_id_for_instance(instance_id)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _capability_from_cell(wd: Path, row: Mapping[str, Any]) -> Optional[str]:
    traj = _load_json(wd / "trajectory.json") or {}
    route_summary = traj.get("route_summary") or {}
    cap = route_summary.get("selected_edge_id")
    if cap:
        return str(cap).strip()
    answer = _load_json(wd / "answer.json") or {}
    if str(answer.get("action") or "").lower() != "solve":
        return None
    ctrl = _load_json(wd / "controller_state.json") or {}
    rid = str(answer.get("route_id") or ctrl.get("active_route_id") or "").strip()
    if rid.startswith("route:cap:"):
        return rid.split("route:cap:", 1)[-1].strip()
    dec_path = wd / "controller_decisions.jsonl"
    if dec_path.is_file():
        for line in dec_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(ev.get("action") or "") == "execute" and ev.get("ok"):
                extra = ev.get("extra") or {}
                cap2 = extra.get("capability_id")
                if cap2:
                    return str(cap2).strip()
    caps = [c for c in (row.get("allowed_edge_ids") or []) if c and not str(c).startswith("schema_")]
    return str(caps[0]) if len(caps) == 1 else None


def harvest_verified_cases(
    out_root: Path,
    *,
    seed: int = 11,
    condition: str = "full_hwa",
    inventory_path: Optional[Path] = None,
    tiers: Optional[Set[str]] = None,
) -> List[HwaRouteCase]:
    """Collect admissible successful routes from HWA OUT_ROOT (no baseline imports)."""
    from hazardweaver.hwb.metrics.headline_decomposed_v1 import analyze_cell

    from hazardweaver.hwa.experiments.headline_inventory_runner_v1 import load_headline_inventory

    inv = load_headline_inventory(inventory_path or ROOT / "runs/hwb/iclr_benchmark/headline_inventory_v1.jsonl")
    tier_ok = tiers or _ELIGIBLE_TIERS
    cases: List[HwaRouteCase] = []
    for row in inv:
        tier = str(row.get("difficulty_tier") or "").upper()
        if tier not in tier_ok:
            continue
        iid = str(row["instance_id"])
        wd = _workdir(out_root, seed=seed, condition=condition, instance_id=iid)
        if not wd.is_dir():
            continue
        cell = analyze_cell(wd)
        if not cell.get("pred_at_1x") and str(cell.get("action") or "").lower() != "solve":
            continue
        cap = _capability_from_cell(wd, row)
        if not cap:
            continue
        allowed = [str(e) for e in (row.get("allowed_edge_ids") or []) if e]
        if cap not in allowed:
            continue
        cases.append(
            HwaRouteCase(
                instance_id=iid,
                track=str(row.get("track") or ""),
                tier=tier,
                capability_id=cap,
                route_id=f"route:cap:{cap}",
                allowed_edge_signature=allowed_edge_signature(allowed),
                pred_at_1x=bool(cell.get("pred_at_1x")),
                source_out_root=str(out_root),
                family_id=str(row.get("family_id") or row.get("track") or ""),
            )
        )
    return cases


def write_case_bank(cases: Iterable[HwaRouteCase], bank_path: Path) -> int:
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(c.to_dict(), sort_keys=True) for c in cases]
    bank_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def load_case_bank(bank_path: Path | None = None) -> List[HwaRouteCase]:
    path = Path(bank_path or default_self_cbr_bank_path())
    if not path.is_file():
        return []
    out: List[HwaRouteCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            out.append(HwaRouteCase(**row))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def retrieve_self_cbr_cases(
    inventory_row: Mapping[str, Any],
    bank: Sequence[HwaRouteCase],
    *,
    top_k: int = 3,
) -> List[HwaRouteCase]:
    """Structure-similar HWA history cases (track + allowed-cap overlap)."""
    track = str(inventory_row.get("track") or "")
    allowed = {
        str(e)
        for e in (inventory_row.get("allowed_edge_ids") or [])
        if e and not str(e).startswith("schema_")
    }
    scored: List[tuple[float, HwaRouteCase]] = []
    for case in bank:
        if case.instance_id == str(inventory_row.get("instance_id")):
            continue
        score = 0.0
        if track and case.track == track:
            score += 1.0
        case_caps = set(case.allowed_edge_signature.split("|")) if case.allowed_edge_signature else set()
        score += 2.0 * _jaccard(allowed, case_caps)
        if case.pred_at_1x:
            score += 0.5
        if score > 0:
            scored.append((score, case))
    scored.sort(key=lambda x: (-x[0], x[1].instance_id))
    return [c for _, c in scored[:top_k]]


def attach_hwa_self_cbr_metadata(
    task: Dict[str, Any],
    inventory_row: Mapping[str, Any],
    *,
    bank_path: Path | None = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    if not self_cbr_enabled():
        return task
    bank = load_case_bank(bank_path)
    if not bank:
        return task
    retrieved = retrieve_self_cbr_cases(inventory_row, bank, top_k=top_k)
    if not retrieved:
        return task
    meta = dict(task.get("metadata") or {})
    allowed = [
        str(e)
        for e in ((task.get("solver_visible") or {}).get("inputs") or {}).get("allowed_edge_ids") or []
        if e and not str(e).startswith("schema_")
    ]
    pick = retrieved[0]
    cap = pick.capability_id
    if allowed and cap not in allowed:
        for alt in retrieved[1:]:
            if alt.capability_id in allowed:
                pick = alt
                cap = alt.capability_id
                break
        else:
            return task
    meta["hwa_self_cbr_capability"] = cap
    meta["hwa_self_cbr_route_id"] = f"route:cap:{cap}"
    meta["hwa_self_cbr_source"] = "hwa_verified_execution_history_v1"
    meta["hwa_self_cbr_provenance_instance"] = pick.instance_id
    meta["hwa_self_cbr_retrieved_k"] = len(retrieved)
    task["metadata"] = meta
    return task


def should_attach_hwa_self_cbr(condition: str) -> bool:
    return condition == "full_hwa" and self_cbr_enabled()


def _cli_build(args: argparse.Namespace) -> int:
    cases = harvest_verified_cases(
        Path(args.out_root),
        seed=int(args.seed),
        condition=str(args.condition),
        inventory_path=Path(args.inventory) if args.inventory else None,
    )
    n = write_case_bank(cases, Path(args.bank_path))
    print(json.dumps({"ok": True, "n_cases": n, "bank_path": str(args.bank_path)}, indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HWA self-CBR case bank (no baseline leakage)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Harvest verified routes from HWA OUT_ROOT")
    build.add_argument("--out-root", required=True)
    build.add_argument("--bank-path", default=str(default_self_cbr_bank_path()))
    build.add_argument("--seed", type=int, default=11)
    build.add_argument("--condition", default="full_hwa")
    build.add_argument("--inventory", default="")
    build.set_defaults(func=_cli_build)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
