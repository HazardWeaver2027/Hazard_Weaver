#!/usr/bin/env python3
"""Fail if release tree contains cluster account or home paths."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    re.compile(r"/blue/yd24f"),
    re.compile(r"wz26b\.fsu"),
    re.compile(r"yd24f\.fsu"),
]
SKIP_FILES = {"verify_release.py", "check_public_paths.py"}
SKIP_SUFFIX = {".png", ".svg", ".pdf", ".zip", ".pyc"}


def main() -> int:
    hits: list[str] = []
    for p in ROOT.rglob("*"):
        if p.is_dir() or ".git" in p.parts:
            continue
        if p.name in SKIP_FILES:
            continue
        if p.suffix.lower() in SKIP_SUFFIX:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(pat.search(text) for pat in PATTERNS):
            hits.append(str(p.relative_to(ROOT)))
    out = {"hit_count": len(hits), "hits": hits[:50]}
    print(json.dumps(out, indent=2))
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
