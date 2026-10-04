#!/usr/bin/env python3
"""CPU smoke: import package, load manifest, run leakage fixture test path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hazardweaver.paths import BENCHMARK_PUBLIC  # noqa: E402


def main() -> int:
    manifest = BENCHMARK_PUBLIC / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    fixture = next((r for r in rows if "fixture" in str(r.get("taskpack_id", "")).lower()), None)
    if fixture is None:
        fixture = rows[0]
    from hazardweaver.hwb.evaluators.leakage_audit import audit_taskpack_leakage, planner_view  # noqa: E402

    tp_path = BENCHMARK_PUBLIC / "taskpacks" / f"{fixture['taskpack_id']}.json"
    if not tp_path.is_file():
        print(f"smoke: taskpack missing {tp_path}", file=sys.stderr)
        return 1
    tp = json.loads(tp_path.read_text(encoding="utf-8"))
    report = audit_taskpack_leakage(tp)
    pv = planner_view(tp)
    assert "reference_view" not in pv
    if not report.passed:
        print(f"smoke: leakage audit failed critical={report.critical_count}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "OK", "instance_id": fixture.get("instance_id"), "taskpack": fixture.get("taskpack_id")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
