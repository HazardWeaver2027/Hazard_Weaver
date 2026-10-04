"""Batch 2 native-route L2 evaluation (replay certificate + metrics)."""

from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[4]
BATCH2_ROOT = _PROJECT / "runs" / "carp" / "batch2"
FIXTURES_ROOT = _PROJECT / "data" / "fixtures" / "hcg"
