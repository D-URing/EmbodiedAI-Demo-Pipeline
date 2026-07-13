from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_demo.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "config_name, expected_steps",
    [
        ("tabletop_sorting_mock.yaml", 7),
        ("towel_folding_mock.yaml", 5),
    ],
)
def test_run_cli_writes_successful_mock_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    config_name: str,
    expected_steps: int,
) -> None:
    exit_code = main(
        [
            "run",
            "--config",
            str(ROOT / "configs/runs" / config_name),
            "--output-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "RUN_COMPLETE" in captured.out

    artifact_dir = Path(captured.out.splitlines()[0].removeprefix("RUN_COMPLETE "))
    assert (artifact_dir / "manifest.yaml").is_file()
    assert (artifact_dir / "events.jsonl").is_file()
    assert (artifact_dir / "result.json").is_file()
    assert (artifact_dir / "metrics.json").is_file()
    assert (artifact_dir / "report.md").is_file()

    result = json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
    assert result["episode_success"] is True
    assert result["progress_score"] == 100
    assert result["termination_reason"] == "success"
    assert result["episode_steps"] == expected_steps
