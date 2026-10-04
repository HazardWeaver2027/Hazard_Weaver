"""Coupled-4M runtime mode — binds HKC/HCG/HWB/HWA on preflight-pass inventory."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping


def coupled_4m_enabled() -> bool:
    return str(os.environ.get("HWA_COUPLED_4M", "0")).strip().lower() in {"1", "true", "yes"}


def hcg_g6_fallback_disabled() -> bool:
    if coupled_4m_enabled():
        return True
    return str(os.environ.get("HWA_DISABLE_HCG_G6_FALLBACK", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def hkc_headline_permit_disabled(task: Mapping[str, Any] | None = None) -> bool:
    if coupled_4m_enabled():
        return True
    if task is not None:
        meta = task.get("metadata") or {}
        if str(meta.get("coupled_4m_protocol") or "") in {"v1", "v2"}:
            return True
    return False


def coupled_typed_allowed_edge_eligible(task: Mapping[str, Any]) -> bool:
    """Coupled L4: when manifest_rf graph is disconnected, use typed CAP routes (HKC-gated)."""
    if not coupled_4m_enabled():
        return False
    meta = task.get("metadata") or {}
    if not (meta.get("hcg_typed_probe") or meta.get("coupled_4m_protocol")):
        return False
    if str(meta.get("headline_route_mode") or "") != "hcg_multi_hop":
        return False
    from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import _allowed_edges

    caps = [e for e in _allowed_edges(task) if str(e).startswith("CAP-")]
    return bool(caps)


def coupled_solve_mandatory_enabled(task: Mapping[str, Any] | None = None) -> bool:
    """Block spurious NO_LEGAL_ROUTE on solve-only coupled cells until CAPs are tried."""
    if not coupled_4m_enabled():
        return False
    raw = str(os.environ.get("HWA_COUPLED_SOLVE_MANDATORY", "1")).strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if task is None:
        return True
    meta = task.get("metadata") or {}
    if not meta.get("coupled_4m_protocol"):
        return False
    expected = str(meta.get("expected_action") or "solve").strip().lower()
    return expected != "abstain"


def stamp_coupled_inventory_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply coupled protocol fields for preflight/runtime parity."""
    out = dict(row)
    out["coupled_4m_protocol"] = "v2"
    out["hcg_typed_probe"] = True
    return out


@contextmanager
def coupled_runtime_env() -> Iterator[None]:
    """Bind env vars matching Coupled-4M SLURM contract ."""
    keys = (
        "HWA_COUPLED_4M",
        "HWA_DISABLE_HCG_G6_FALLBACK",
        "HWA_HEADLINE_HKC_MODE",
        "HWA_HKC_REGISTRY",
        "HWA_HCG_COMPAT_MODE",
    )
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["HWA_COUPLED_4M"] = "1"
    os.environ["HWA_DISABLE_HCG_G6_FALLBACK"] = "1"
    os.environ["HWA_HEADLINE_HKC_MODE"] = "enhance"
    os.environ["HWA_HKC_REGISTRY"] = "k_hkc_frozen_v1"
    os.environ["HWA_HCG_COMPAT_MODE"] = "full"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def apply_coupled_row_metadata(task: Mapping[str, Any], inventory_row: Mapping[str, Any]) -> Dict[str, Any]:
    """Stamp coupled protocol fields on task metadata when inventory row requires it."""
    out = dict(task)
    proto = str(inventory_row.get("coupled_4m_protocol") or "")
    if proto not in {"v1", "v2"}:
        return out
    meta = dict(out.get("metadata") or {})
    meta["coupled_4m_protocol"] = proto
    if inventory_row.get("hcg_typed_probe"):
        meta["hcg_typed_probe"] = True
    if inventory_row.get("hcg_oracle_path"):
        meta["hcg_oracle_path_hwb_only"] = inventory_row.get("hcg_oracle_path")
    out["metadata"] = meta
    return out
