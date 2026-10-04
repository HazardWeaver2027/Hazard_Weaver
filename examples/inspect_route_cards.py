#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / "benchmark/public/manifest.jsonl"
rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
print(rows[0]["instance_id"], rows[0].get("taskpack_id"))
