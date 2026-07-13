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
