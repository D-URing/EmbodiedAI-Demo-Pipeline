from __future__ import annotations

from pathlib import Path

import pytest

from embodied_demo.config import compose_yaml, load_resolved_run, load_task
from embodied_demo.errors import ConfigurationError, SchemaValidationError

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path, expected_id",
    [
        ("tasks/tabletop_sorting_v1/task.yaml", "tabletop_sorting_v1"),
        ("tasks/towel_folding_v1/task.yaml", "towel_folding_v1"),
    ],
)
def test_task_specs_are_valid(relative_path: str, expected_id: str) -> None:
    task = load_task(ROOT / relative_path)
    assert task.id == expected_id
    assert sum(stage.weight for stage in task.evaluation.progress_stages) == 100


@pytest.mark.parametrize(
    "relative_path, task_id",
    [
        ("configs/runs/tabletop_sorting_mock.yaml", "tabletop_sorting_v1"),
        ("configs/runs/towel_folding_mock.yaml", "towel_folding_v1"),
    ],
)
def test_run_configs_resolve(relative_path: str, task_id: str) -> None:
    resolved = load_resolved_run(ROOT / relative_path)
    assert resolved.task_spec.id == task_id
    assert resolved.run.runtime.mode.value == "mock"
    assert resolved.run.evaluation.profile.value == "smoke"
    assert len(resolved.sources) == 4


def test_unknown_field_fails_with_source_context(tmp_path: Path) -> None:
    task_data = (ROOT / "tasks/tabletop_sorting_v1/task.yaml").read_text(encoding="utf-8")
    invalid = tmp_path / "invalid_task.yaml"
    invalid.write_text(task_data + "unknown_field: true\n", encoding="utf-8")

    with pytest.raises(SchemaValidationError, match="invalid_task.yaml"):
        load_task(invalid)


def test_extends_cycle_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="cycle"):
        compose_yaml(first)
