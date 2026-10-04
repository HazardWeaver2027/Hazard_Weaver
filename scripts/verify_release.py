#!/usr/bin/env python3
"""Check benchmark bundle, released stats, and public-path hygiene."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hazardweaver.paths import BENCHMARK_PUBLIC, BENCHMARK_SEALED, REPO_ROOT, RELEASED_RESULTS  # noqa: E402

IDENTITY_PATTERNS = [
    re.compile(r"/blue/yd24f"),
    re.compile(r"wz26b\.fsu"),
    re.compile(r"yd24f\.fsu"),
    re.compile(r"slurmInfo"),
]


def _fail(msg: str) -> None:
    print(f"verify failed: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_manifest_rows() -> list[dict]:
    path = BENCHMARK_PUBLIC / "manifest.jsonl"
    if not path.is_file():
        _fail(f"missing {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_counts(rows: list[dict]) -> None:
    if len(rows) != 141:
        _fail(f"benchmark count {len(rows)} != 141")
    sh = sum(1 for r in rows if r.get("hazard_scope") == "single_hazard")
    mh = sum(1 for r in rows if r.get("hazard_scope") == "multi_hazard")
    if sh != 90 or mh != 51:
        _fail(f"hazard split {sh}/{mh} != 90/51")
    one = sum(1 for r in rows if r.get("route_eligibility") == "single_route")
    multi = sum(1 for r in rows if r.get("route_eligibility") == "multi_eligible")
    if one != 45 or multi != 96:
        _fail(f"route split {one}/{multi} != 45/96")


def check_sealed(rows: list[dict]) -> None:
    sm = BENCHMARK_SEALED / "scoring_manifest.jsonl"
    if not sm.is_file():
        _fail("missing scoring_manifest.jsonl")
    sealed_lines = [json.loads(x) for x in sm.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(sealed_lines) != 141:
        _fail(f"scoring_manifest lines {len(sealed_lines)} != 141")
    contracts = list((BENCHMARK_SEALED / "evaluator_contracts").glob("*.json"))
    if len(contracts) != 141:
        _fail(f"evaluator_contracts {len(contracts)} != 141")


def check_released_stats() -> None:
    ps = RELEASED_RESULTS / "rq1_baselines/paper_stats_v1.json"
    if not ps.is_file():
        _fail(f"missing {ps}")
    data = json.loads(ps.read_text(encoding="utf-8"))
    llama = data.get("arms", {}).get("hwa_llama70b", {})
    k = (llama.get("overall") or {}).get("k")
    if k is not None and int(k) != 126:
        _fail(f"HWA Llama DCA k {k} != 126")


def check_checksums() -> None:
    cs = REPO_ROOT / "benchmark" / "checksums.sha256"
    if not cs.is_file():
        _fail("missing benchmark/checksums.sha256")
    lines = [ln for ln in cs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 10:
        _fail("checksums.sha256 too short")
    for line in lines[:50]:
        _digest, rel = line.split(None, 1)
        p = REPO_ROOT / "benchmark" / rel.strip()
        if not p.is_file():
            _fail(f"checksum references missing file {rel}")


def scan_identity() -> None:
    skip_parts = {".git", "__pycache__", ".pytest_cache", "scripts"}
    for p in REPO_ROOT.rglob("*"):
        if p.is_dir():
            continue
        if "src" in p.parts and "hazardweaver" in p.parts:
            continue
        if any(s in p.parts for s in skip_parts):
            continue
        if p.suffix.lower() in {".png", ".svg", ".pdf", ".zip"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat in IDENTITY_PATTERNS:
            if pat.search(text):
                _fail(f"identity pattern {pat.pattern} in {p.relative_to(REPO_ROOT)}")


def check_imports() -> None:
    import importlib

    importlib.import_module("hazardweaver")
    importlib.import_module("hazardweaver.paths")


def main() -> int:
    rows = _load_manifest_rows()
    check_counts(rows)
    check_sealed(rows)
    check_released_stats()
    check_checksums()
    scan_identity()
    check_imports()
    print(json.dumps({"status": "OK", "headline_n": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
