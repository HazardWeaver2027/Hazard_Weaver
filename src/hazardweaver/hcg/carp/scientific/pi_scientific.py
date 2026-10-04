"""PI scientific sign-off (separate from engineering A2 dev signoff)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.scientific.paths import scientific_signoff_path


def load_scientific_signoff(taskpack: str) -> Optional[Dict[str, Any]]:
    path = scientific_signoff_path(taskpack)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def scientific_signoff_ok(taskpack: str) -> bool:
    data = load_scientific_signoff(taskpack)
    if not data or not data.get("approved"):
        return False
    return data.get("approval_tier") == "scientific"


def write_scientific_signoff_stub(
    taskpack: str,
    *,
    spec_ref: str = "PHASE_C_PLATFORM.md",
    chain_eval_approved: bool = False,
    root: Optional[Path] = None,
) -> Path:
    """Test-only helper; production signoffs are PI-authored."""
    base = root or scientific_signoff_path(taskpack).parent
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "taskpack_id": taskpack,
        "approved": True,
        "approval_tier": "scientific",
        "spec_ref": spec_ref,
        "chain_eval_approved": chain_eval_approved,
        "note": "Scientific holdout packaging; not engineering_a2_dev",
    }
    path = base / "PI_SCIENTIFIC_SIGNOFF.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
