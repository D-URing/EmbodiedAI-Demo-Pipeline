from __future__ import annotations

from pathlib import Path

import yaml

from embodied_demo.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_validate_run_cli(capsys) -> None:
    exit_code = main(
        [
            "validate",
            "--config",
            str(ROOT / "configs/runs/tabletop_sorting_mock.yaml"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "VALID run=tabletop_sorting_mock" in captured.out


def test_list_tasks_cli(capsys) -> None:
    exit_code = main(
        ["list-tasks", "--registry", str(ROOT / "tasks/registry.yaml")]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "tabletop_sorting_v1" in captured.out
    assert "towel_folding_v1" in captured.out


def test_dry_run_persists_resolved_config(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "resolved.yaml"
    exit_code = main(
        [
            "dry-run",
            "--config",
            str(ROOT / "configs/runs/towel_folding_mock.yaml"),
            "--output",
            str(destination),
        ]
    )
    capsys.readouterr()
    payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["task_spec"]["id"] == "towel_folding_v1"
    assert payload["run"]["evaluation"]["profile"] == "smoke"


def test_export_schema_cli(tmp_path: Path, capsys) -> None:
    exit_code = main(["export-schema", "--output-dir", str(tmp_path)])
    capsys.readouterr()
    assert exit_code == 0
    assert (tmp_path / "task.schema.json").is_file()
    assert (tmp_path / "episode_result.schema.json").is_file()
