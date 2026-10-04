"""Capability registry for Hazard Weaver Agent."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from hazardweaver.hwa.contracts import (
    ArtifactSpec,
    CapabilityCard,
    CapabilityContract,
    CapabilityKind,
    ContractField,
    ExecutionSpec,
    ProvenanceSpec,
    RuntimeSpec,
    ValidityCondition,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "configs" / "capabilities" / "pfdf_suite.yaml"


def default_registry_path() -> Path:
    if DEFAULT_REGISTRY_PATH.exists():
        return DEFAULT_REGISTRY_PATH
    return PROJECT_ROOT / "configs" / "capabilities" / "default_cards.yaml"


def _parse_card(item: dict) -> CapabilityCard:
    exec_raw = item.get("execution", {})
    artifact_raw = item.get("artifact", {})
    runtime_raw = item.get("runtime", {})
    prov_raw = item.get("provenance", {})

    if "checkpoint_uri" in exec_raw and not artifact_raw.get("checkpoint_uri"):
        artifact_raw = {**artifact_raw, "checkpoint_uri": exec_raw["checkpoint_uri"]}

    return CapabilityCard(
        capability_id=item["capability_id"],
        kind=CapabilityKind(item["kind"]),
        hazard_scope=item["hazard_scope"],
        description=item.get("description", ""),
        input_contract=[ContractField.model_validate(f) for f in item.get("input_contract", [])],
        output_contract=[ContractField.model_validate(f) for f in item.get("output_contract", [])],
        validity_conditions=[
            ValidityCondition.model_validate(v) for v in item.get("validity_conditions", [])
        ],
        execution=ExecutionSpec.model_validate(exec_raw),
        artifact=ArtifactSpec.model_validate(artifact_raw),
        runtime=RuntimeSpec.model_validate(runtime_raw),
        provenance=ProvenanceSpec.model_validate(prov_raw),
        metadata=item.get("metadata", {}),
    )


class CapabilityRegistry:
    def __init__(self, cards: Optional[List[CapabilityCard]] = None):
        self._cards: Dict[str, CapabilityCard] = {c.capability_id: c for c in (cards or [])}

    def get(self, capability_id: str) -> CapabilityCard:
        if capability_id not in self._cards:
            raise KeyError(f"Unknown capability: {capability_id}")
        return self._cards[capability_id]

    def list_ids(self) -> List[str]:
        return sorted(self._cards.keys())

    def save_yaml(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [c.to_yaml_dict() for c in self._cards.values()]
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: Path) -> "CapabilityRegistry":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cards = [_parse_card(item) for item in raw]
        return cls(cards)


class CapabilityContractRegistry:
    """V2 registry over `CapabilityContract` with legacy YAML compatibility."""

    def __init__(self, contracts: Optional[List[CapabilityContract]] = None):
        self._contracts: Dict[str, CapabilityContract] = {
            c.capability_id: c for c in (contracts or [])
        }

    def get(self, capability_id: str) -> CapabilityContract:
        if capability_id not in self._contracts:
            raise KeyError(f"Unknown capability contract: {capability_id}")
        return self._contracts[capability_id]

    def list_ids(self) -> List[str]:
        return sorted(self._contracts.keys())

    @classmethod
    def load_yaml(cls, path: Path) -> "CapabilityContractRegistry":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or []
        return cls([CapabilityContract.model_validate(item) for item in raw])
