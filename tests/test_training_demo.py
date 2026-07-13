from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_demo.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "config_name",
    [
        "tabletop_sorting_mock.yaml",
        "towel_folding_mock.yaml",
    ],
)
def test_train_demo_loss_decreases_and_writes_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    config_name: str,
) -> None:
    exit_code = main(
        [
            "train-demo",
            "--config",
            str(ROOT / "configs/runs" / config_name),
            "--output-dir",
            str(tmp_path),
            "--epochs",
            "20",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "TRAIN_COMPLETE" in captured.out
    assert "loss_decreased=true" in captured.out

    artifact_dir = Path(captured.out.splitlines()[0].removeprefix("TRAIN_COMPLETE "))
    assert (artifact_dir / "dataset.jsonl").is_file()
    assert (artifact_dir / "train_log.jsonl").is_file()
    assert (artifact_dir / "checkpoint.json").is_file()
    assert (artifact_dir / "metrics.json").is_file()
    assert (artifact_dir / "report.md").is_file()

    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["loss_decreased"] is True
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["loss_drop_ratio"] > 0.5


@pytest.mark.parametrize(
    "config_name, expected_steps",
    [
        ("tabletop_sorting_mock.yaml", 7),
        ("towel_folding_mock.yaml", 5),
    ],
)
def test_train_eval_demo_loads_checkpoint_and_runs_learned_rollout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    config_name: str,
    expected_steps: int,
) -> None:
    exit_code = main(
        [
            "train-eval-demo",
            "--config",
            str(ROOT / "configs/runs" / config_name),
            "--output-dir",
            str(tmp_path),
            "--epochs",
            "20",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "TRAIN_EVAL_COMPLETE" in captured.out
    assert "true_flow_complete=true" in captured.out
    assert "eval_success=true" in captured.out

    artifact_dir = Path(captured.out.splitlines()[0].removeprefix("TRAIN_EVAL_COMPLETE "))
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    rollout_result = json.loads(
        (artifact_dir / "learned_rollout/result.json").read_text(encoding="utf-8")
    )

    assert metrics["loss_decreased"] is True
    assert metrics["true_flow_complete"] is True
    assert rollout_result["episode_success"] is True
    assert rollout_result["progress_score"] == 100
    assert rollout_result["episode_steps"] == expected_steps
