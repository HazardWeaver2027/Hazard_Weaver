#!/usr/bin/env python3
"""Regenerate paper tables/figures checks against paper_manifest.yaml."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper_artifacts" / "paper_manifest.yaml"


def _resolve_path(data: dict, path: str):
    cur: object = data
    for part in path.split("."):
        if isinstance(cur, list):
            if part.startswith("arm_id="):
                arm = part.split("=", 1)[1]
                cur = next((x for x in cur if isinstance(x, dict) and x.get("arm_id") == arm), None)
            else:
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def main() -> int:
    if not MANIFEST.is_file():
        print("reproduce_paper: missing paper_manifest.yaml", file=sys.stderr)
        return 1
    if yaml is None:
        print("reproduce_paper: PyYAML required", file=sys.stderr)
        return 1
    spec = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for item in spec.get("artifacts", []):
        src = ROOT / item["input"]
        if not src.is_file():
            failures.append(f"missing input {item['input']}")
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        for check in item.get("checks", []):
            path = check["json_path"]
            expected = check["expected"]
            if path.startswith("rows.arm_id="):
                rest = path.split("=", 1)[1]
                arm, _, subpath = rest.partition(".")
                row = next((r for r in data.get("rows", []) if r.get("arm_id") == arm), {})
                cur = row
                for part in subpath.split("."):
                    if not part:
                        continue
                    cur = cur.get(part, {}) if isinstance(cur, dict) else None
            else:
                cur = _resolve_path(data, path)
            if cur != expected:
                failures.append(f"{item['id']}: {path}={cur!r} expected {expected!r}")
    if failures:
        for f in failures:
            print(f"reproduce_paper: {f}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "OK", "artifacts_checked": len(spec.get("artifacts", []))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
