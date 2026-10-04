#!/usr/bin/env python3
"""Document external scientific data acquisition (no bulk download on login)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "DATA_PROVENANCE.md"


def main() -> int:
    print(json.dumps({"status": "OK", "see": str(DOC.relative_to(ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
