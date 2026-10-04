"""Contract tests: 141 inventory and evaluator bindings."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_manifest_141():
    rows = [
        json.loads(l)
        for l in (ROOT / "benchmark/public/manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(rows) == 141


def test_sealed_contracts_141():
    n = len(list((ROOT / "benchmark/sealed_eval/evaluator_contracts").glob("*.json")))
    assert n == 141
