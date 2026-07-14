from __future__ import annotations

from typing import Literal

from pydantic import Field

from embodied_demo.schemas.base import StrictModel


class TrainingCheckpointSummary(StrictModel):
    step: int | None = None
    weights: str | None = None
    state: str | None = None


class TrainingEvidence(StrictModel):
    """Normalized evidence that a real training backend ran and produced usable artifacts."""

    schema_version: str = "1.0"
    backend: str
    run_id: str
    source_run_dir: str
    native_output_dir: str | None = None
    mode: str | None = None
    recipe: str | None = None
    task_name: str | None = None
    official_ref: str | None = None
    overlay_ref: str | None = None
    parsed_train_count: int = Field(ge=0)
    initial_loss: float | None = None
    final_loss: float | None = None
    loss_drop_ratio: float | None = None
    loss_decreased: bool | None = None
    final_step: int | None = None
    max_steps: int | None = None
    training_completed: bool
    latest_checkpoint: TrainingCheckpointSummary | None = None
    validation_status: Literal["passed", "warning", "failed"]
    notes: list[str] = Field(default_factory=list)


class DatasetEvidence(StrictModel):
    """Normalized evidence that a LeRobot dataset was readable."""

    schema_version: str = "1.0"
    backend: str = "lerobot"
    repo_id: str
    root: str | None = None
    split: str | None = None
    sample_index: int | None = None
    length: int | None = Field(default=None, ge=0)
    fps: float | None = None
    features: dict[str, object] = Field(default_factory=dict)
    sample: dict[str, object] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)
    allow_download: bool = False
    validation_status: Literal["passed", "warning", "failed"]
    notes: list[str] = Field(default_factory=list)


class InferenceEvidence(StrictModel):
    """Normalized evidence that a LeRobot policy produced an action."""

    schema_version: str = "1.0"
    backend: str = "lerobot"
    policy_type: str
    policy_class: str | None = None
    policy_path: str
    dataset_repo_id: str | None = None
    dataset_root: str | None = None
    sample_index: int | None = None
    device: str
    action: dict[str, object] = Field(default_factory=dict)
    input_sample: dict[str, object] = Field(default_factory=dict)
    latency_ms: float | None = None
    validation_status: Literal["passed", "warning", "failed"]
    notes: list[str] = Field(default_factory=list)
