from __future__ import annotations

import json
from pathlib import Path

from scripts.fastwam.parse_train_log import write_summary

ROOT = Path(__file__).resolve().parents[1]


def test_fastwam_log_parser_detects_loss_drop_and_checkpoint(tmp_path: Path) -> None:
    summary_path = write_summary(
        ROOT / "tests/fixtures/fastwam_train_stdout.log",
        tmp_path,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["parsed_train_count"] == 4
    assert summary["parsed_eval_count"] == 1
    assert summary["loss_decreased"] is True
    assert summary["initial_loss"] == 1.4862
    assert summary["final_loss"] == 0.701
    assert summary["final_step"] == 200
    assert summary["training_completed"] is True
    assert summary["latest_checkpoint"]["weights"].endswith("step_000200.pt")
    assert summary["metric_summary"]["loss_action"]["final"] == 0.551


def test_fastwam_runner_refuses_cpu_fallback_and_wraps_train_zero1() -> None:
    runner = (ROOT / "scripts/fastwam/run_realrobot_train_eval.sh").read_text(encoding="utf-8")

    assert "torch.cuda.is_available()" in runner
    assert "CPU fallback is intentionally disabled" in runner
    assert "scripts/train_zero1.sh" in runner
    assert "parse_train_log.py" in runner
    assert "FASTWAM_NATIVE_OUTPUT_DIR" in runner


def test_fastwam_prepare_uses_overlay_without_vendoring() -> None:
    prepare = (ROOT / "scripts/fastwam/prepare_fastwam_overlay.sh").read_text(encoding="utf-8")

    assert "FASTWAM_OFFICIAL_REPO" in prepare
    assert "FASTWAM_OVERLAY_REPO" in prepare
    assert "rsync -a" in prepare
    assert "--exclude \"runs/\"" in prepare
    assert "--exclude \"checkpoints/\"" in prepare
