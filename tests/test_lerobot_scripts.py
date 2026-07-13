from __future__ import annotations

import json
from pathlib import Path

from scripts.lerobot.parse_train_log import write_summary

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
