"""Machine-checkable workflow validity certificate schema."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class AbstainCode(str, Enum):
    UNSUPPORTED_TARGET = "UNSUPPORTED_TARGET"
    MISSING_REQUIRED_INPUT = "MISSING_REQUIRED_INPUT"
    INVALID_DECISION_TIME = "INVALID_DECISION_TIME"
    SPATIAL_SUPPORT_MISMATCH = "SPATIAL_SUPPORT_MISMATCH"
    TEMPORAL_WINDOW_MISMATCH = "TEMPORAL_WINDOW_MISMATCH"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    NO_VALID_ADAPTER = "NO_VALID_ADAPTER"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    TARGET_LEAKAGE_RISK = "TARGET_LEAKAGE_RISK"
    NO_VALID_WORKFLOW = "NO_VALID_WORKFLOW"


class ContractCheck(BaseModel):
    check_id: str
    status: CheckStatus
    message: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)


class WorkflowValidityCertificate(BaseModel):
    request_spec_hash: str
    selected_workflow_dag: Dict[str, Any] = Field(default_factory=dict)
    capabilities_used: List[str] = Field(default_factory=list)
    adapters_inserted: List[str] = Field(default_factory=list)
    interaction_rules_satisfied: List[str] = Field(default_factory=list)
    contract_checks: List[ContractCheck] = Field(default_factory=list)
    decision_time_checks: List[ContractCheck] = Field(default_factory=list)
    missing_or_degraded_inputs: List[str] = Field(default_factory=list)
    uncertainty_path: List[str] = Field(default_factory=list)
    cost_estimate: Optional[Dict[str, Any]] = None
    provenance_bundle: Dict[str, Any] = Field(default_factory=dict)
    abstain_or_warning_codes: List[AbstainCode] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        checks = [*self.contract_checks, *self.decision_time_checks]
        return all(check.status != CheckStatus.FAIL for check in checks) and not any(
            code in self.abstain_or_warning_codes for code in AbstainCode
        )
