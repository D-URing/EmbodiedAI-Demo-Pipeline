from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from embodied_demo.config import load_registry, load_task
from embodied_demo.errors import SchemaValidationError
from embodied_demo.schemas import TaskSpec
from embodied_demo.schemas.enums import RegistryStatus


@dataclass(frozen=True)
class RegisteredTask:
    task: TaskSpec
    status: RegistryStatus
    path: Path


def iter_registered_tasks(registry_path: str | Path) -> list[RegisteredTask]:
    source = Path(registry_path).expanduser().resolve()
    registry = load_registry(source)
    result: list[RegisteredTask] = []
    for entry in registry.tasks:
        task_path = (source.parent / entry.path).resolve()
        task = load_task(task_path)
        if task.id != entry.id:
            raise SchemaValidationError(
                f"registry id '{entry.id}' does not match task id '{task.id}' in {task_path}"
            )
        result.append(RegisteredTask(task=task, status=entry.status, path=task_path))
    return result
