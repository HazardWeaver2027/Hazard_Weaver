"""Static check: solver modules must not reference sealed_eval paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOLVER_ROOTS = [
    ROOT / "src/hazardweaver/hwa",
    ROOT / "src/hazardweaver/hcg",
    ROOT / "src/hazardweaver/hkc",
    ROOT / "src/hazardweaver/baselines",
]
FORBIDDEN = "benchmark/sealed_eval"


def test_solver_code_no_sealed_eval_string():
    hits = []
    for base in SOLVER_ROOTS:
        if not base.is_dir():
            continue
        for py in base.rglob("*.py"):
            text = py.read_text(encoding="utf-8", errors="ignore")
            if FORBIDDEN in text:
                hits.append(str(py.relative_to(ROOT)))
    assert not hits, hits
