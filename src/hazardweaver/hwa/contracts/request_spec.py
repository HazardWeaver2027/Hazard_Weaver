"""Operational request contract for Hazard Weaver Agent."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class PredictionMode(str, Enum):
    FORECAST = "forecast"
    EVENT_CONDITIONED = "event_conditioned"
    POST_EVENT_ASSESSMENT = "post_event_assessment"


class UserIntent(str, Enum):
    RUN = "run"
    PLAN = "plan"
    ASSESS = "assess"
    COMPARE = "compare"


class ObservationSpec(BaseModel):
    """Observation available to a user request."""

    observation_id: str
    variable: str
    schema_id: Optional[str] = None
    spatial_support: Optional[str] = None
    temporal_support: Optional[str] = None
    observed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TimeWindow(BaseModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_order(self) -> "TimeWindow":
        if self.end < self.start:
            raise ValueError("time_window_end_before_start")
        return self


class BudgetSpec(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = None


class UncertaintyRequirement(BaseModel):
    required: bool = False
    max_interval_width: Optional[float] = None
    min_coverage: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RequestSpec(BaseModel):
    """Machine-checkable form of a natural-language hazard request."""

    request_id: str
    hazard_family: str
    target_variable: str
    target_unit: str
    prediction_mode: PredictionMode
    decision_time: datetime
    event_time_window: Optional[TimeWindow] = None
    forecast_horizon: Optional[BudgetSpec] = None
    spatial_support: str
    temporal_support: str
    available_observations: List[ObservationSpec] = Field(default_factory=list)
    allowed_training_or_adaptation: bool = False
    uncertainty_requirement: UncertaintyRequirement = Field(default_factory=UncertaintyRequirement)
    latency_budget: Optional[BudgetSpec] = None
    compute_budget: Optional[BudgetSpec] = None
    user_intent: UserIntent = UserIntent.RUN
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_temporal_contract(self) -> "RequestSpec":
        if self.prediction_mode == PredictionMode.EVENT_CONDITIONED and self.event_time_window is None:
            raise ValueError("event_conditioned_request_requires_event_time_window")
        if self.prediction_mode == PredictionMode.FORECAST and self.forecast_horizon is None:
            raise ValueError("forecast_request_requires_forecast_horizon")
        for obs in self.available_observations:
            if obs.observed_at and obs.observed_at > self.decision_time:
                raise ValueError(f"observation_after_decision_time:{obs.observation_id}")
        return self

    @classmethod
    def pfdf_event_conditioned_volume(
        cls,
        *,
        request_id: str,
        decision_time: datetime,
        event_time_window: TimeWindow,
        available_observations: Optional[List[ObservationSpec]] = None,
        user_intent: UserIntent = UserIntent.RUN,
    ) -> "RequestSpec":
        """Convenience constructor for the PFDF gold request family."""

        return cls(
            request_id=request_id,
            hazard_family="post_fire_debris_flow",
            target_variable="debris_flow_volume",
            target_unit="log1p_m3",
            prediction_mode=PredictionMode.EVENT_CONDITIONED,
            decision_time=decision_time,
            event_time_window=event_time_window,
            spatial_support="watershed",
            temporal_support="storm_event",
            available_observations=available_observations or [],
            user_intent=user_intent,
        )

    def has_observation_variable(self, variable: str) -> bool:
        return any(obs.variable == variable for obs in self.available_observations)
