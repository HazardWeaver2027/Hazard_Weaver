"""Round2 frozen HKC registry K_HKC_FROZEN_V1 — solver-side scientific knowledge."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwa.scientific_controller.hkc_route_contract import (
    RouteContractIndex,
    evaluate_A_sci_with_contract,
    load_route_contract_index,
)
from hazardweaver.hwa.scientific_controller.reason_codes import ASciVerdict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

HKC_REGISTRY_K_FROZEN = "k_hkc_frozen_v1"
HKC_REGISTRY_NONE = "none"
HKC_REGISTRY_DEFAULT = HKC_REGISTRY_K_FROZEN

FROZEN_MANIFEST_REL = Path("manifests/K_HKC_FROZEN_V1.jsonl")
FROZEN_CONTRACTS_REL = Path("runs/eskc_compiler/k_hkc_frozen_v1/contracts.jsonl")


def hkc_registry_mode() -> str:
    raw = (
        os.environ.get("HWA_HKC_REGISTRY")
        or os.environ.get("ICLR_HWA_HKC_REGISTRY")
        or HKC_REGISTRY_DEFAULT
    ).strip().lower()
    if raw in {HKC_REGISTRY_NONE, "off", "disabled", "0", "false"}:
        return HKC_REGISTRY_NONE
    if raw in {HKC_REGISTRY_K_FROZEN, "frozen", "k_hkc", "default"}:
        return HKC_REGISTRY_K_FROZEN
    return raw


def hkc_registry_enabled() -> bool:
    return hkc_registry_mode() == HKC_REGISTRY_K_FROZEN


def hkc_registry_disabled() -> bool:
    return hkc_registry_mode() == HKC_REGISTRY_NONE


@dataclass(frozen=True)
class FrozenHKCRegistry:
    manifest_path: Path
    contracts_path: Path
    contract_index: RouteContractIndex
    manifest_sha256: str

    @classmethod
    def load(cls, root: Path | None = None) -> "FrozenHKCRegistry":
        base = root or PROJECT_ROOT
        manifest_path = base / FROZEN_MANIFEST_REL
        contracts_path = base / FROZEN_CONTRACTS_REL
        if not contracts_path.is_file():
            raise FileNotFoundError(f"K_HKC_FROZEN contracts missing: {contracts_path}")
        idx = load_route_contract_index(contracts_path)
        sha = ""
        if manifest_path.is_file():
            sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
        return cls(
            manifest_path=manifest_path,
            contracts_path=contracts_path,
            contract_index=idx,
            manifest_sha256=sha,
        )


_registry_cache: Optional[FrozenHKCRegistry] = None


def get_frozen_hkc_registry() -> FrozenHKCRegistry:
    global _registry_cache
    if _registry_cache is None:
        _registry_cache = FrozenHKCRegistry.load()
    return _registry_cache


def reset_frozen_hkc_registry_cache() -> None:
    global _registry_cache
    _registry_cache = None


def is_headline_inventory_task(task: Mapping[str, Any]) -> bool:
    meta = task.get("metadata") or {}
    return bool(meta.get("hwb_headline_inventory"))


def _route_family_id(route: Mapping[str, Any], task: Mapping[str, Any]) -> str:
    from hazardweaver.hwa.route_controller.admissibility_gate import resolve_hkc_family_id

    fid = resolve_hkc_family_id(task, route)
    if fid:
        return fid
    return str(
        route.get("route_family_id")
        or route.get("hkc_family_id")
        or route.get("family_id")
        or ""
    )


def _scientific_override_inapplicable(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
) -> bool:
    meta = task.get("metadata") or {}
    override = route.get("scientific_condition_override") or meta.get("scientific_condition_override")
    return isinstance(override, Mapping) and bool(override.get("force_inapplicable"))


def resolve_frozen_contract_row(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
    contract_index: RouteContractIndex,
    *,
    explicit_row: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Per-route binding, else headline family representative from frozen K."""
    if explicit_row is not None:
        return dict(explicit_row)
    if not is_headline_inventory_task(task):
        return None
    fid = _route_family_id(route, task)
    if not fid:
        return None
    return contract_index.headline_family_contract(fid)


def hkc_off_permissive_a_sci() -> Dict[str, Any]:
    """HKC-off: no scientific condition gating (Round2 ablation arm)."""
    from hazardweaver.hwa.scientific_controller.reason_codes import ASciVerdict

    return {
        "verdict": ASciVerdict.APPLICABLE.value,
        "codes": [],
        "refs": [],
        "source": "hkc_registry_off",
    }


def evaluate_a_sci_round2(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    contract_row: Optional[Mapping[str, Any]],
    contract_index: Optional[RouteContractIndex] = None,
    theory_arm: str = "verified",
) -> Dict[str, Any]:
    """Round2 A_sci: frozen K (full) vs registry-off (ablation)."""
    if hkc_registry_disabled():
        return hkc_off_permissive_a_sci()
    if _scientific_override_inapplicable(route, task):
        return {
            "verdict": ASciVerdict.INAPPLICABLE_THEORY_MISMATCH.value,
            "codes": ["hkc_intervention_force_inapplicable"],
            "refs": [],
            "source": "frozen_k_hkc_v1",
        }
    if contract_index is not None:
        contract_row = resolve_frozen_contract_row(
            route,
            task,
            contract_index,
            explicit_row=contract_row,
        )
    if contract_row is None:
        return {
            "verdict": ASciVerdict.UNKNOWN_PENDING_THEORY.value,
            "codes": ["missing_frozen_binding"],
            "refs": [],
            "source": "frozen_k_hkc_v1",
        }
    out = evaluate_A_sci_with_contract(route, task, contract_row, theory_arm=theory_arm)
    out["source"] = "frozen_k_hkc_v1"
    from hazardweaver.hwa.experiments.coupled_4m_mode_v1 import hkc_headline_permit_disabled

    if is_headline_inventory_task(task) and not hkc_headline_permit_disabled(task):
        violated = list(out.get("hkc_strict_violated") or [])
        if violated:
            out["verdict"] = ASciVerdict.INAPPLICABLE_THEORY_MISMATCH.value
            out["codes"] = list(out.get("codes") or []) + ["frozen_k_strict_violated"]
        elif out.get("verdict") in {
            ASciVerdict.UNKNOWN_PENDING_THEORY.value,
            "UNKNOWN",
        }:
            out["verdict"] = ASciVerdict.APPLICABLE.value
            out["codes"] = list(out.get("codes") or []) + ["frozen_k_headline_family_permit"]
    return out


def registry_provenance_fields() -> Dict[str, Any]:
    if not hkc_registry_enabled():
        return {"hkc_registry_id": HKC_REGISTRY_NONE, "k_manifest_sha256": None}
    try:
        reg = get_frozen_hkc_registry()
        return {
            "hkc_registry_id": HKC_REGISTRY_K_FROZEN,
            "k_manifest_sha256": reg.manifest_sha256,
            "k_contracts_path": str(reg.contracts_path),
        }
    except FileNotFoundError:
        return {"hkc_registry_id": HKC_REGISTRY_K_FROZEN, "k_manifest_sha256": None, "k_missing": True}
