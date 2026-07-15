from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from embodied_demo.schemas.base import StrictModel
from embodied_demo.schemas.enums import (
    Capability,
    EnvironmentBackend,
    EvaluationProfile,
    FailureType,
    TerminationReason,
    VerificationStatus,
)

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
SchemaVersion = Annotated[str, Field(pattern=r"^\d+\.\d+$")]
TaskVersion = Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]


class LatencyStatistics(StrictModel):
    count: int = Field(ge=0)
    mean_ms: float = Field(ge=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    max_ms: float = Field(ge=0)


class SafetyViolation(StrictModel):
    label: Identifier
    step_id: int = Field(ge=0)
    severity: str = Field(pattern=r"^(info|warning|critical)$")
    detail: str = ""


class EpisodeResult(StrictModel):
    """Auditable episode output aligned with the RoboDojo-inspired protocol."""

    schema_version: SchemaVersion = "1.0"
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    task_id: Identifier
    task_version: TaskVersion
    backend: EnvironmentBackend
    profile: EvaluationProfile
    seed: int
    layout_id: str = Field(min_length=1)
    valid: bool
    episode_success: bool
    progress_score: float = Field(ge=0, le=100)
    completed_stages: int = Field(ge=0)
    total_stages: int = Field(gt=0)
    failure_type: FailureType | None = None
    termination_reason: TerminationReason
    episode_steps: int = Field(ge=0)
    wall_time_s: float = Field(ge=0)
    policy_latency: LatencyStatistics | None = None
    safety_violations: list[SafetyViolation] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome(self) -> "EpisodeResult":
        if self.completed_stages > self.total_stages:
            raise ValueError("completed_stages cannot exceed total_stages")
        if self.episode_success and self.termination_reason != TerminationReason.SUCCESS:
            raise ValueError("successful episodes must terminate with reason 'success'")
        if not self.episode_success and self.termination_reason == TerminationReason.SUCCESS:
            raise ValueError("termination reason 'success' requires episode_success = true")
        if self.failure_type is not None and self.episode_success:
            raise ValueError("successful episodes cannot have a failure_type")
        return self


class TaskAggregate(StrictModel):
    schema_version: SchemaVersion = "1.0"
    task_id: Identifier
    task_version: TaskVersion
    profile: EvaluationProfile
    total_episodes: int = Field(gt=0)
    valid_episodes: int = Field(ge=0)
    task_success_rate: float | None = Field(default=None, ge=0, le=1)
    task_progress_score: float | None = Field(default=None, ge=0, le=100)
    progress_std: float | None = Field(default=None, ge=0)
    confidence_interval_95: tuple[float, float] | None = None

    @model_validator(mode="after")
    def validate_counts(self) -> "TaskAggregate":
        if self.valid_episodes > self.total_episodes:
            raise ValueError("valid_episodes cannot exceed total_episodes")
        metrics = (self.task_success_rate, self.task_progress_score, self.progress_std)
        if self.valid_episodes == 0 and any(value is not None for value in metrics):
            raise ValueError("aggregate metrics must be null when valid_episodes = 0")
        if self.valid_episodes > 0 and any(value is None for value in metrics):
            raise ValueError("aggregate metrics are required when valid_episodes > 0")
        if self.confidence_interval_95 is not None:
            lower, upper = self.confidence_interval_95
            if not (0 <= lower <= upper <= 100):
                raise ValueError("confidence_interval_95 must be ordered within [0, 100]")
        return self


class DimensionAggregate(StrictModel):
    schema_version: SchemaVersion = "1.0"
    dimension: Capability
    task_ids: list[Identifier] = Field(min_length=1)
    score: float = Field(ge=0, le=100)


class EvaluationManifest(StrictModel):
    schema_version: SchemaVersion = "1.0"
    run_id: str = Field(min_length=1)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    task_version: TaskVersion
    evaluator_commit: str = Field(min_length=1)
    profile: EvaluationProfile
    seeds: list[int] = Field(min_length=1)
    checkpoint: str
    normalizer: str
    container_image: str
    policy_transport_version: str
    environment_transport_version: str
    artifacts: dict[str, str] = Field(default_factory=dict)
