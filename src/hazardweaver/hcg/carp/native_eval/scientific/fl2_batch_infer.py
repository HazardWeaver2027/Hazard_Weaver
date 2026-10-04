"""Batch per-scenario official G2 inference for CAP-FL2-01~03 (M2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from hazardweaver.hcg.carp.native_eval.scientific.fl2_g2_official_infer import (
    audit_official_readiness,
    g2_caps,
    run_official_g2_infer,
    write_cap_blocked_json,
)
from hazardweaver.hcg.carp.scientific.fl2_data import HIGH_FI_SPLITS, TASKPACK, scenario_ids
from hazardweaver.hcg.carp.scientific.paths import cap_scientific_dir

DEFAULT_SPLITS = ["official_test", "hwb_holdout"]


def _collect_scenario_jobs(
    *,
    splits: List[str],
    scenario_ids_filter: Optional[List[str]] = None,
    max_scenarios: Optional[int] = None,
) -> List[Dict[str, str]]:
    jobs: List[Dict[str, str]] = []
    for split in splits:
        for sid in scenario_ids(split):
            if scenario_ids_filter and sid not in scenario_ids_filter:
                continue
            jobs.append({"split": split, "scenario_id": sid})
    if max_scenarios is not None:
        jobs = jobs[: max(0, int(max_scenarios))]
    return jobs


def run_batch_infer(
    capability_id: str,
    *,
    splits: Optional[List[str]] = None,
    scenario_ids_filter: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
    write_npz: bool = True,
    max_scenarios: Optional[int] = None,
) -> Dict[str, Any]:
    if capability_id not in g2_caps():
        return {"ok": False, "error": f"unsupported capability {capability_id}"}

    split_list = list(splits or DEFAULT_SPLITS)
    cap_dir = Path(out_dir) if out_dir else cap_scientific_dir(TASKPACK, capability_id)
    jobs = _collect_scenario_jobs(
        splits=split_list,
        scenario_ids_filter=scenario_ids_filter,
        max_scenarios=max_scenarios,
    )
    if not jobs:
        return {"ok": False, "error": "no scenarios to infer"}

    audit = audit_official_readiness(capability_id)
    results: List[Dict[str, Any]] = []
    n_ok = 0
    n_blocked = 0

    for job in jobs:
        out = run_official_g2_infer(
            capability_id,
            scenario_id=job["scenario_id"],
            split=job["split"],
            out_dir=cap_dir,
            write_npz=write_npz,
        )
        row = {
            "scenario_id": job["scenario_id"],
            "split": job["split"],
            "ok": bool(out.get("ok")),
            "blocked": bool(out.get("blocked")),
            "official_status": out.get("official_status"),
        }
        results.append(row)
        if out.get("ok"):
            n_ok += 1
        else:
            n_blocked += 1

    if audit["conclusion"] == "BLOCKED":
        write_cap_blocked_json(capability_id, cap_dir=cap_dir)

    manifest = {
        "capability_id": capability_id,
        "audit_conclusion": audit["conclusion"],
        "blockers": audit.get("blockers") or [],
        "n_jobs": len(jobs),
        "n_ok": n_ok,
        "n_blocked": n_blocked,
        "splits": split_list,
        "per_split": {
            split: {
                "n_jobs": sum(1 for j in jobs if j["split"] == split),
                "n_ok": sum(1 for r in results if r["split"] == split and r["ok"]),
                "n_blocked": sum(1 for r in results if r["split"] == split and not r["ok"]),
            }
            for split in split_list
        },
        "results": results,
    }
    cap_dir.mkdir(parents=True, exist_ok=True)
    manifest_name = (
        "batch_manifest_high.json"
        if any(s in HIGH_FI_SPLITS for s in split_list)
        else "batch_manifest.json"
    )
    manifest_path = cap_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "ok": n_ok > 0 or audit["conclusion"] == "BLOCKED",
        "capability_id": capability_id,
        "batch_manifest": str(manifest_path),
        "n_jobs": len(jobs),
        "n_ok": n_ok,
        "n_blocked": n_blocked,
        "audit": audit,
        "blocked_json": str(cap_dir / "BLOCKED.json") if audit["conclusion"] == "BLOCKED" else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="FL-2 batch G2 infer (M2)")
    ap.add_argument("--capability-id", required=True, choices=g2_caps())
    ap.add_argument("--split", action="append", default=None)
    ap.add_argument("--scenario-id", action="append", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-scenarios", type=int, default=None)
    ap.add_argument("--no-npz", action="store_true")
    args = ap.parse_args()
    out = run_batch_infer(
        args.capability_id,
        splits=args.split,
        scenario_ids_filter=args.scenario_id,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        write_npz=not args.no_npz,
        max_scenarios=args.max_scenarios,
    )
    print(json.dumps(out, indent=2))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
