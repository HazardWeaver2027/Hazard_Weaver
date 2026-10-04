"""Repository root resolution for the public release tree."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("HAZARDWEAVER_ROOT", Path(__file__).resolve().parents[2]))
BENCHMARK_PUBLIC = REPO_ROOT / "benchmark" / "public"
BENCHMARK_SEALED = REPO_ROOT / "benchmark" / "sealed_eval"
RELEASED_RESULTS = REPO_ROOT / "released_results"
PAPER_ARTIFACTS = REPO_ROOT / "paper_artifacts"
HKC_DATA_ROOT = Path(os.environ.get("HKC_DATA_ROOT", BENCHMARK_PUBLIC / "hkc_data"))
