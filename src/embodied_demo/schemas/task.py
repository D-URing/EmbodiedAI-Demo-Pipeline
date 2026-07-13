from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from embodied_demo.schemas.base import StrictModel
from embodied_demo.schemas.enums import (
    ActionRepresentation,
    Capability,
    Difficulty,
    EnvironmentBackend,
    MockRealism,
)

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
SchemaVersion = Annotated[str, Field(pattern=r"^\d+\.\d+$")]
TaskVersion = Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]


class CapabilitySpec(StrictModel):
    primary: list[Capability] = Field(min_length=1)
    secondary: list[Capability] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_capabilities(self) -> "CapabilitySpec":
        if len(set(self.primary)) != len(self.primary):
            raise ValueError("capabilities.primary contains duplicates")
        if len(set(self.secondary)) != len(self.secondary):
            raise ValueError("capabilities.secondary contains duplicates")
        overlap = set(self.primary) & set(self.secondary)
        if overlap:
            raise ValueError(f"primary and secondary capabilities overlap: {sorted(overlap)}")
        return self


class InstructionSpec(StrictModel):
    canonical: str = Field(min_length=1)
    variants: list[str] = Field(default_factory=list)


class SceneSpec(StrictModel):
    scene_id: Identifier
    required_objects: list[Identifier] = Field(min_length=1)
    optional_objects: list[Identifier] = Field(default_factory=list)


class ObservationSpec(StrictModel):
    required: list[Identifier] = Field(min_length=1)
    optional: list[Identifier] = Field(default_factory=list)
    evaluator_only: list[Identifier] = Field(default_factory=list)
    debug_only: list[Identifier] = Field(default_factory=list)
    restricted: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_visibility_groups(self) -> "ObservationSpec":
        groups = {
            "required": self.required,
            "optional": self.optional,
            "evaluator_only": self.evaluator_only,
            "debug_only": self.debug_only,
            "restricted": self.restricted,
        }
        owners: dict[str, str] = {}
        for group_name, fields in groups.items():
            if len(set(fields)) != len(fields):
                raise ValueError(f"observation.{group_name} contains duplicates")
            for field_name in fields:
                if field_name in owners:
                    raise ValueError(
                        f"observation field '{field_name}' appears in both "
                        f"{owners[field_name]} and {group_name}"
                    )
                owners[field_name] = group_name
        return self


class TaskActionSpec(StrictModel):
    supported: list[ActionRepresentation] = Field(min_length=1)


class TerminationSpec(StrictModel):
    max_steps: int = Field(gt=0)
    success: str = Field(min_length=1)
    failure: list[str] = Field(default_factory=list)


class BackendSpec(StrictModel):
    supported: list[EnvironmentBackend] = Field(min_length=1)
    planned: list[EnvironmentBackend] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_backends(self) -> "BackendSpec":
        overlap = set(self.supported) & set(self.planned)
        if overlap:
            raise ValueError(f"supported and planned backends overlap: {sorted(overlap)}")
        return self


class ProgressStage(StrictModel):
    id: Identifier
    description: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    weight: int = Field(gt=0, le=100)


class EvaluationSpec(StrictModel):
    dimensions: list[Capability] = Field(min_length=1)
    progress_stages: list[ProgressStage] = Field(min_length=1)
    standard_layouts: list[str] = Field(default_factory=list)
    randomization_factors: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evaluation(self) -> "EvaluationSpec":
        stage_ids = [stage.id for stage in self.progress_stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("evaluation.progress_stages contains duplicate ids")
        total = sum(stage.weight for stage in self.progress_stages)
        if total != 100:
            raise ValueError(f"progress stage weights must sum to 100, got {total}")
        if len(self.dimensions) != len(set(self.dimensions)):
            raise ValueError("evaluation.dimensions contains duplicates")
        return self


class MockSpec(StrictModel):
    realism: MockRealism
    limitations: list[str] = Field(min_length=1)


class SafetySpec(StrictModel):
    labels: list[Identifier] = Field(default_factory=list)
    real_execution_allowed: bool = False
    notes: list[str] = Field(default_factory=list)


class TaskSpec(StrictModel):
    schema_version: SchemaVersion = "1.0"
    id: Identifier
    version: TaskVersion
    category: Identifier
    display_name: str = Field(min_length=1)
    difficulty: Difficulty
    capabilities: CapabilitySpec
    instruction: InstructionSpec
    scene: SceneSpec
    observation: ObservationSpec
    action: TaskActionSpec
    stages: list[Identifier] = Field(min_length=1)
    termination: TerminationSpec
    backends: BackendSpec
    evaluation: EvaluationSpec
    mock: MockSpec
    safety: SafetySpec

    @model_validator(mode="after")
    def validate_task(self) -> "TaskSpec":
        if len(self.stages) != len(set(self.stages)):
            raise ValueError("stages contains duplicates")
        progress_ids = [stage.id for stage in self.evaluation.progress_stages]
        missing = set(self.stages) - set(progress_ids)
        if missing:
            raise ValueError(
                "every task stage needs a progress stage definition; missing: "
                + ", ".join(sorted(missing))
            )
        declared_dimensions = set(self.capabilities.primary + self.capabilities.secondary)
        undeclared = set(self.evaluation.dimensions) - declared_dimensions
        if undeclared:
            raise ValueError(
                "evaluation dimensions must be declared as task capabilities: "
                + ", ".join(sorted(undeclared))
            )
        return self
