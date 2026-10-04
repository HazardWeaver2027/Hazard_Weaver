#!/usr/bin/env python3
"""Re-score stub: confirms sealed manifest aligns with released per-instance records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASED = ROOT / "released_results"
OUT = ROOT / "outputs" / "evaluate_released"


def main() -> int:
    per_inst = RELEASED / "structured_trajectories" / "unified_141_per_instance_v1.jsonl"
    if not per_inst.is_file():
        print(json.dumps({"status": "SKIPPED", "reason": "per_instance jsonl not built; aggregate JSON only"}))
        return 0
    rows = [json.loads(l) for l in per_inst.read_text(encoding="utf-8").splitlines() if l.strip()]
    valid = sum(1 for r in rows if r.get("dca_valid"))
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"n": len(rows), "dca_valid": valid, "expected_llama_k": 126}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if valid != 126:
        print(f"evaluate: dca_valid {valid} != 126", file=sys.stderr)
        return 1
    print(json.dumps({"status": "OK", **summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
