"""Validate Phase C scientific data manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _has_checksum(manifest: Dict[str, Any]) -> bool:
    if manifest.get("checksum"):
        return True
    checksums = manifest.get("checksums")
    return isinstance(checksums, dict) and len(checksums) > 0


def official_test_ok(manifest: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    if manifest is None:
        return False, "missing official_test manifest"
    if manifest.get("fixture_scale") is True:
        return False, "fixture_scale official_test manifest"
    if not manifest.get("taskpack_id"):
        return False, "official_test manifest missing taskpack_id"
    if not _has_checksum(manifest):
        return False, "official_test manifest missing checksum(s)"
    return True, "ok"


def holdout_ok(
    manifest: Optional[Dict[str, Any]],
    signoff: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    if manifest is None:
        return False, "missing hwb_holdout manifest"
    if manifest.get("fixture_scale") is True:
        return False, "fixture_scale hwb_holdout manifest"
    if not manifest.get("taskpack_id"):
        return False, "hwb_holdout manifest missing taskpack_id"
    if not _has_checksum(manifest):
        return False, "hwb_holdout manifest missing checksum(s)"
    if signoff is None:
        return False, "missing PI_SCIENTIFIC_SIGNOFF"
    if not signoff.get("approved"):
        return False, "PI_SCIENTIFIC_SIGNOFF not approved"
    if signoff.get("approval_tier") != "scientific":
        return False, "PI_SCIENTIFIC_SIGNOFF approval_tier must be scientific"
    return True, "ok"
