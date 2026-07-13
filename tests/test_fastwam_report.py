from __future__ import annotations

import json
from pathlib import Path

from embodied_demo.cli import main
from embodied_demo.fastwam_report import generate_fastwam_report
from scripts.fastwam.parse_train_log import write_summary

ROOT = Path(__file__).resolve().parents[1]


def _make_fastwam_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "fastwam_run"
    run_dir.mkdir()
    write_summary(ROOT / "tests/fixtures/fastwam_train_stdout.log", run_dir)
    (run_dir / "backend_manifest.json").write_text(
        json.dumps(
            {
                "backend": "fastwam-realrobot",
                "mode": "pilot",
                "recipe": "joint_base",
                "task_name": "real_robot_joint_2cam224_1e-4",
                "run_id": "20260713-200000",
                "fastwam_native_output_dir": "/shared/FastWAM/runs/real_robot_joint_2cam224_1e-4/20260713-200000",
                "official_ref": "45d8e1458921d83f8ad6cf9ce993d371208dabd0",
                "overlay_ref": "5b9791f7d49956b96e0694786f46ff94e8214eca",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_generate_fastwam_report_writes_demo_chain_artifacts(tmp_path: Path) -> None:
    run_dir = _make_fastwam_run(tmp_path)
    output_dir = tmp_path / "chain_report"

    artifact_dir = generate_fastwam_report(run_dir, output_dir=output_dir)

    assert artifact_dir == output_dir
    assert (artifact_dir / "chain_manifest.yaml").is_file()
    assert (artifact_dir / "training_evidence.json").is_file()
    assert (artifact_dir / "checkpoint_summary.json").is_file()
    assert (artifact_dir / "mock_summary.json").is_file()
    assert (artifact_dir / "report.md").is_file()
    assert (artifact_dir / "handoff.md").is_file()

    evidence = json.loads((artifact_dir / "training_evidence.json").read_text(encoding="utf-8"))
    assert evidence["validation_status"] == "passed"
    assert evidence["loss_decreased"] is True
    assert evidence["loss_drop_ratio"] > 0.5
    assert evidence["latest_checkpoint"]["weights"].endswith("step_000200.pt")


def test_report_fastwam_cli(tmp_path: Path, capsys) -> None:
    run_dir = _make_fastwam_run(tmp_path)
    output_dir = tmp_path / "cli_report"

    exit_code = main(
        [
            "report-fastwam",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "REPORT_FASTWAM_COMPLETE" in captured.out
    assert "loss_decreased=true" in captured.out
    assert (output_dir / "training_evidence.json").is_file()
