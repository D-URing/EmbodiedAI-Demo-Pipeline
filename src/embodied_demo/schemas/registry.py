from __future__ import annotations

from pydantic import Field, model_validator

from embodied_demo.schemas.base import StrictModel
from embodied_demo.schemas.enums import RegistryStatus
from embodied_demo.schemas.task import Identifier, SchemaVersion


class TaskRegistryEntry(StrictModel):
    id: Identifier
    path: str = Field(min_length=1)
    status: RegistryStatus


class TaskRegistry(StrictModel):
    schema_version: SchemaVersion = "1.0"
    tasks: list[TaskRegistryEntry]

    @model_validator(mode="after")
    def validate_entries(self) -> "TaskRegistry":
        ids = [entry.id for entry in self.tasks]
        paths = [entry.path for entry in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task registry contains duplicate ids")
        if len(paths) != len(set(paths)):
            raise ValueError("task registry contains duplicate paths")
        return self
