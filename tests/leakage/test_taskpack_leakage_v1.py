"""Leakage tests on public taskpack fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hazardweaver.hwb.evaluators.leakage_audit import audit_taskpack_leakage, planner_view  # noqa: E402


def test_wf3_fixture_leakage_critical_zero():
    tp = json.loads(
        (ROOT / "benchmark/public/taskpacks/hwb_wf3_spread_fixture_v1.json").read_text(encoding="utf-8")
    )
    report = audit_taskpack_leakage(tp)
    assert report.critical_count == 0
    assert report.passed is True
    assert "reference_view" not in planner_view(tp)
