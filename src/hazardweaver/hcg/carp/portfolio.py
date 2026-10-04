"""Load TaskPack scientific capability portfolios (paper + engineering status)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

PORTFOLIOS_ROOT = Path(__file__).resolve().parents[3] / "docs" / "engineering" / "hcg" / "portfolios"

TASK_ID_TO_DIR = {
    "WF-3": "WF-3",
    "FL-2": "FL-2",
    "L2": "L2",
    "E1-E3": "E1-E3",
    "TC-TRK": "TC-TRK",
    "DR-OUT": "DR-OUT",
    "HW-MED": "HW-MED",
    "MH-1": "MH-1",
    "MH-2": "MH-2",
    "MH-3": "MH-3",
    "MH-4": "MH-4",
}


class CapabilitySpec(BaseModel):
    capability_id: str
    family_id: str = ""
    tier: Literal["A", "B", "C"] = "C"
    headline_eligible: bool = True
    validation_level: Literal["L0", "L1", "L2", "L3"] = "L0"
    exact_task_match: bool = True
    anchor_evidence_ref: str = ""
    official_asset_ref: str = ""
    backend: str = "g2_train"
    method_name: str = ""
    evaluator_protocol: str = "native-route"
    paper_side_approval_for_l3: bool = False
    notes: str = ""
    chain: str = ""
    dataset_manifest: str = ""
    split_manifest: str = ""
    metric_min: Optional[float] = None
    legacy_capability_id: str = ""


class RelocatedCapabilitySpec(BaseModel):
    capability_id: str
    validation_level: Literal["L0", "L1", "L2", "L3"] = "L1"
    exact_task_match: bool = False
    headline_eligible: bool = False
    task_mismatch: str = ""
    relocated_to_task: str = ""
    official_asset_ref: str = ""
    notes: str = ""


class HeadlineGates(BaseModel):
    min_families: int = 4
    min_tier_a: int = 2
    min_tier_a_distinct_families: int = 2
    min_l2_per_family: int = 1


class Portfolio(BaseModel):
    task_id: str
    paper_status: Literal["DRAFT", "FROZEN", "BLOCKED"] = "DRAFT"
    engineering_status: Literal["UNVERIFIED", "ACQUIRING", "VERIFIED", "BLOCKED"] = "UNVERIFIED"
    paper_approver: str = ""
    frozen_date: str = ""
    manifest_ref: str = ""
    verdict: str = ""
    canonical_output: str = ""
    evaluator_ref: str = ""
    split_protocol: str = ""
    acquisition_gate: str = ""
    scientific_gate: str = ""
    coverage_tier_target: str = "E"
    headline_gates: HeadlineGates = Field(default_factory=HeadlineGates)
    capabilities: List[CapabilitySpec] = Field(default_factory=list)
    relocated_capabilities: List[RelocatedCapabilitySpec] = Field(default_factory=list)
    families: List[Dict[str, Any]] = Field(default_factory=list)
    bundle_members: List[str] = Field(default_factory=list)
    progression_truth_ref: str = ""
    terminal_bundle: List[str] = Field(default_factory=list)
    legacy_canonical_output: str = ""

    def portfolio_dir(self) -> Path:
        d = TASK_ID_TO_DIR.get(self.task_id, self.task_id)
        return PORTFOLIOS_ROOT / d

    def yaml_path(self) -> Path:
        return self.portfolio_dir() / "portfolio.yaml"

    def capability_by_id(self, capability_id: str) -> Optional[CapabilitySpec]:
        for c in self.capabilities:
            if c.capability_id == capability_id:
                return c
        for r in self.relocated_capabilities:
            if r.capability_id == capability_id:
                return CapabilitySpec(
                    capability_id=r.capability_id,
                    family_id="",
                    validation_level=r.validation_level,
                    exact_task_match=r.exact_task_match,
                    headline_eligible=r.headline_eligible,
                    official_asset_ref=r.official_asset_ref,
                    notes=r.notes or r.task_mismatch,
                )
        return None

    def capabilities_for_family(self, family_id: str) -> List[CapabilitySpec]:
        return [c for c in self.capabilities if c.family_id == family_id]

    def tier_a_families(self) -> List[str]:
        return sorted({c.family_id for c in self.capabilities if c.tier == "A"})


def load_portfolio(task_id: str) -> Portfolio:
    d = TASK_ID_TO_DIR.get(task_id, task_id)
    path = PORTFOLIOS_ROOT / d / "portfolio.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Portfolio.model_validate(raw)


def list_frozen_portfolios() -> List[str]:
    out: List[str] = []
    for task_id in sorted(TASK_ID_TO_DIR):
        dirname = TASK_ID_TO_DIR[task_id]
        p = PORTFOLIOS_ROOT / dirname / "portfolio.yaml"
        if p.is_file():
            port = load_portfolio(task_id)
            if port.paper_status == "FROZEN":
                out.append(task_id)
    return sorted(out)


def list_all_portfolios() -> List[str]:
    out: List[str] = []
    for task_id in sorted(TASK_ID_TO_DIR):
        if (PORTFOLIOS_ROOT / TASK_ID_TO_DIR[task_id] / "portfolio.yaml").is_file():
            out.append(task_id)
    return out
