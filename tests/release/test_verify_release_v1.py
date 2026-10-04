"""Release gate smoke via subprocess."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_verify_release_exit_zero():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/verify_release.py")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
