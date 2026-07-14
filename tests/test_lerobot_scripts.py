from __future__ import annotations

import json
from pathlib import Path

from scripts.lerobot.parse_train_log import write_summary
from scripts.lerobot.generate_data_to_inference_report import generate_report

ROOT = Path(__file__).resolve().parents[1]


def test_lerobot_log_parser_detects_loss_drop(tmp_path: Path) -> None:
    summary_path = write_summary(
        ROOT / "tests/fixtures/lerobot_train_stdout.log",
        tmp_path,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["parsed_loss_count"] == 4
    assert summary["loss_decreased"] is True
    assert summary["initial_loss"] == 2.421
    assert summary["final_loss"] == 0.743


def test_lerobot_runner_refuses_cpu_fallback() -> None:
    runner = (ROOT / "scripts/lerobot/run_pusht_act_gpu_smoke.sh").read_text(encoding="utf-8")

    assert "torch.cuda.is_available()" in runner
    assert "CPU fallback is intentionally disabled" in runner
    assert "--policy.device=\"$LEROBOT_POLICY_DEVICE\"" in runner
    assert "LEROBOT_POLICY_DEVICE" in runner


def test_lerobot_dataset_smoke_disables_downloads_by_default() -> None:
    runner = (ROOT / "scripts/lerobot/run_dataset_smoke.sh").read_text(encoding="utf-8")
    config = (ROOT / "configs/lerobot/native_pusht_act_pipeline.sh").read_text(encoding="utf-8")

    assert 'LEROBOT_ALLOW_DOWNLOAD:-0' in config
    assert 'HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"' in runner
    assert "scripts/lerobot/inspect_dataset.py" in runner


def test_lerobot_inference_smoke_requires_local_policy_path() -> None:
    runner = (ROOT / "scripts/lerobot/run_inference_smoke.sh").read_text(encoding="utf-8")

    assert "LEROBOT_POLICY_PATH is required" in runner
    assert "Downloads are disabled by default" in runner
    assert "scripts/lerobot/run_policy_inference_smoke.py" in runner


def test_lerobot_chain_report_uses_dataset_and_inference_evidence(tmp_path: Path) -> None:
    train_summary = write_summary(ROOT / "tests/fixtures/lerobot_train_stdout.log", tmp_path / "train")
    output_dir = tmp_path / "chain"

    generate_report(
        dataset_profile=ROOT / "tests/fixtures/lerobot_dataset_profile.json",
        inference_evidence=ROOT / "tests/fixtures/lerobot_inference_evidence.json",
        training_summary=train_summary,
        output_dir=output_dir,
    )

    assert (output_dir / "chain_manifest.yaml").is_file()
    assert (output_dir / "dataset_profile.json").is_file()
    assert (output_dir / "inference_evidence.json").is_file()
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "lerobot_fastwam_data_to_inference_v0" in report
    assert "policy_type: act" in report
    assert "loss_decreased: True" in report
