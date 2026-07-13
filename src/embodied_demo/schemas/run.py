from __future__ import annotations

from typing import Any, Annotated, Literal

from pydantic import Field, model_validator

from embodied_demo.schemas.base import StrictModel
from embodied_demo.schemas.enums import (
    ActionRepresentation,
    EnvironmentBackend,
    EvaluationLevel,
    EvaluationProfile,
    Launcher,
    PolicyTransport,
    RuntimeMode,
)
from embodied_demo.schemas.task import SchemaVersion, TaskSpec

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_./-]*$")]


class TaskReference(StrictModel):
    file: str = Field(min_length=1)


class RuntimeSpec(StrictModel):
    mode: RuntimeMode
    launcher: Launcher = Launcher.LOCAL
    seed: int = 0
    deterministic: bool = True
    output_dir: str = "runs"


class PolicySpec(StrictModel):
    name: Identifier
    transport: PolicyTransport = PolicyTransport.INPROC
    action_type: ActionRepresentation
    device: str = "auto"
    supports_batch: bool = False
    endpoint: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_transport(self) -> "PolicySpec":
        if self.transport == PolicyTransport.WEBSOCKET and not self.endpoint:
            raise ValueError("policy.endpoint is required for websocket transport")
        if self.transport == PolicyTransport.INPROC and self.endpoint:
            raise ValueError("policy.endpoint is not valid for inproc transport")
        return self


class EnvironmentSpec(StrictModel):
    backend: EnvironmentBackend
    num_envs: int = Field(default=1, gt=0)
    headless: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class RunEvaluationSpec(StrictModel):
    profile: EvaluationProfile
    level: EvaluationLevel
    evaluator: Identifier = "predicate"
    episodes_per_seed: int = Field(gt=0)
    seeds: list[int] = Field(min_length=1)
    save_video: Literal["never", "failure", "sample", "always"] = "failure"
    confidence_interval: bool = True

    @model_validator(mode="after")
    def validate_seeds(self) -> "RunEvaluationSpec":
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("evaluation.seeds contains duplicates")
        return self


class FeatureFlags(StrictModel):
    enable_viewer: bool = False
    enable_video: bool = False
    enable_remote_policy: bool = False
    enable_simulation: bool = False
    enable_real_robot: bool = False


class ResourceSpec(StrictModel):
    cpus: int = Field(default=2, gt=0)
    memory_gb: int = Field(default=4, gt=0)
    gpus: int = Field(default=0, ge=0)
    gpu_type: str | None = None

    @model_validator(mode="after")
    def validate_gpu_request(self) -> "ResourceSpec":
        if self.gpus == 0 and self.gpu_type is not None:
            raise ValueError("gpu_type requires gpus > 0")
        return self


class ResourceLayout(StrictModel):
    policy: ResourceSpec = Field(default_factory=ResourceSpec)
    environment: ResourceSpec = Field(default_factory=ResourceSpec)
    placement: Literal["same_process", "same_node", "separate_nodes"] = "same_process"
    walltime: str = Field(default="00:30:00", pattern=r"^\d{2,3}:\d{2}:\d{2}$")


class RunSpec(StrictModel):
    schema_version: SchemaVersion = "1.0"
    name: Identifier
    task: TaskReference
    runtime: RuntimeSpec
    policy: PolicySpec
    environment: EnvironmentSpec
    evaluation: RunEvaluationSpec
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    resources: ResourceLayout = Field(default_factory=ResourceLayout)

    @model_validator(mode="after")
    def validate_switches(self) -> "RunSpec":
        expected_mode = {
            EnvironmentBackend.MOCK_2D: RuntimeMode.MOCK,
            EnvironmentBackend.DATASET_REPLAY: RuntimeMode.REPLAY,
            EnvironmentBackend.REAL: RuntimeMode.REAL,
        }.get(self.environment.backend, RuntimeMode.SIM)
        if self.runtime.mode != expected_mode:
            raise ValueError(
                f"runtime.mode '{self.runtime.mode}' does not match backend "
                f"'{self.environment.backend}' (expected '{expected_mode}')"
            )
        if self.environment.num_envs > 1 and not self.policy.supports_batch:
            raise ValueError("environment.num_envs > 1 requires policy.supports_batch = true")
        if self.features.enable_remote_policy != (
            self.policy.transport != PolicyTransport.INPROC
        ):
            raise ValueError(
                "enable_remote_policy must match whether policy.transport is remote"
            )
        if self.features.enable_simulation != (self.runtime.mode == RuntimeMode.SIM):
            raise ValueError("enable_simulation must match runtime.mode = sim")
        if self.features.enable_real_robot != (self.runtime.mode == RuntimeMode.REAL):
            raise ValueError("enable_real_robot must match runtime.mode = real")
        return self


class ResolvedRun(StrictModel):
    schema_version: SchemaVersion = "1.0"
    sources: list[str]
    run: RunSpec
    task_spec: TaskSpec

    @model_validator(mode="after")
    def validate_task_compatibility(self) -> "ResolvedRun":
        if self.run.environment.backend not in self.task_spec.backends.supported:
            raise ValueError(
                f"task '{self.task_spec.id}' does not support backend "
                f"'{self.run.environment.backend}'"
            )
        if self.run.policy.action_type not in self.task_spec.action.supported:
            raise ValueError(
                f"task '{self.task_spec.id}' does not support action type "
                f"'{self.run.policy.action_type}'"
            )
        return self
