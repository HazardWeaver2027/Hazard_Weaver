"""Canonical capability ID mapping for headline subset (manifest ↔ portfolio ↔ registry)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO = Path(__file__).resolve().parents[3]
_HEADLINE_YAML = _REPO / "docs/engineering/hcg/HCG_HEADLINE_CAPABILITY_SUBSET_v1.yaml"
_BRIDGE_YAML = _REPO / "hcg/registry/manifest_rf_bridge_v1.yaml"


@lru_cache(maxsize=1)
def _load_headline_subset() -> List[Dict[str, Any]]:
    if not _HEADLINE_YAML.is_file():
        return []
    raw = yaml.safe_load(_HEADLINE_YAML.read_text(encoding="utf-8")) or {}
    return list(raw.get("capabilities") or [])


@lru_cache(maxsize=1)
def _load_bridge() -> Dict[str, Dict[str, Any]]:
    if not _BRIDGE_YAML.is_file():
        return {}
    raw = yaml.safe_load(_BRIDGE_YAML.read_text(encoding="utf-8")) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for _task, families in (raw.get("entries_by_task") or {}).items():
        for rf_id, entry in (families or {}).items():
            for cap_id in entry.get("capability_ids") or []:
                out[str(cap_id)] = {
                    "manifest_rf_id": rf_id,
                    "task_id": entry.get("task_id"),
                    "legacy_family_id": entry.get("legacy_family_id"),
                    "canonical_output": entry.get("canonical_output"),
                    "route_file": entry.get("route_file"),
                }
    return out


def headline_capability_ids() -> List[str]:
    return [str(c["capability_id"]) for c in _load_headline_subset()]


def is_headline_capability(capability_id: str) -> bool:
    return capability_id in set(headline_capability_ids())


def resolve_canonical(capability_id: str) -> Dict[str, Any]:
    """Return manifest CAP-* ↔ route_family_id ↔ portfolio mapping for headline caps."""
    headline = {c["capability_id"]: c for c in _load_headline_subset()}
    bridge = _load_bridge()
    row = dict(headline.get(capability_id) or {})
    row["capability_id"] = capability_id
    row["bridge"] = bridge.get(capability_id) or {}
    if row.get("route_family_id") and not row["bridge"]:
        row["bridge"] = {
            "manifest_rf_id": row.get("route_family_id"),
            "task_id": row.get("taskpack_id"),
        }
    return row


def manifest_to_composition_id(capability_id: str) -> str:
    """CompositionGraph edge id when distinct from manifest CAP id (headline: 1:1)."""
    return capability_id


def batch_for_capability(capability_id: str) -> Optional[str]:
    for c in _load_headline_subset():
        if c.get("capability_id") == capability_id:
            return str(c.get("migration_phase") or "")
    return None


def non_headline_manifest_caps(manifest_ids: Optional[List[str]] = None) -> List[str]:
    """Caps outside headline execute surface (legacy audit only)."""
    headline = set(headline_capability_ids())
    if manifest_ids is None:
        manifest_path = _REPO / "docs/final_four/HCG/manifests/capabilities.jsonl"
        manifest_ids = []
        if manifest_path.is_file():
            import json

            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    manifest_ids.append(str(json.loads(line)["capability_id"]))
    return sorted(set(manifest_ids) - headline)
