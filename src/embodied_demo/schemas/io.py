from __future__ import annotations

from datetime import datetime
from typing import Any, Annotated

from pydantic import Field, model_validator

from embodied_demo.schemas.base import StrictModel
from embodied_demo.schemas.enums import ActionRepresentation, CoordinateFrame

JsonObject = dict[str, Any]
SchemaVersion = Annotated[str, Field(pattern=r"^\d+\.\d+$")]


class VisionFrame(StrictModel):
    uri: str | None = None
    encoding: str | None = None
    shape: list[int] | None = None
    timestamp: datetime | None = None
    metadata: JsonObject = Field(default_factory=dict)


class Observation(StrictModel):
    schema_version: SchemaVersion = "1.0"
    episode_id: str = Field(min_length=1)
    step_id: int = Field(ge=0)
    timestamp: datetime
    instruction: str = Field(min_length=1)
    vision: dict[str, VisionFrame] = Field(default_factory=dict)
    state: JsonObject = Field(default_factory=dict)
    task_context: JsonObject = Field(default_factory=dict)
    metadata: JsonObject = Field(default_factory=dict)


class ActionChunk(StrictModel):
    schema_version: SchemaVersion = "1.0"
    representation: ActionRepresentation
    frame: CoordinateFrame
    control_frequency_hz: float = Field(gt=0)
    horizon: int = Field(gt=0)
    actions: list[JsonObject] = Field(min_length=1)
    valid_mask: list[bool] | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_lengths(self) -> "ActionChunk":
        if len(self.actions) != self.horizon:
            raise ValueError(
                f"horizon ({self.horizon}) must equal number of actions ({len(self.actions)})"
            )
        if self.valid_mask is not None and len(self.valid_mask) != self.horizon:
            raise ValueError(
                "valid_mask length must equal horizon when valid_mask is provided"
            )
        return self
