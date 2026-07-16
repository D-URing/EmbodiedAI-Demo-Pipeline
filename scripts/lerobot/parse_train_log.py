from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LOSS_PATTERN = re.compile(
    r"(?:^|[^a-zA-Z_])(?:train_)?loss(?:=|:|\s+)([-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)"
)
TRAIN_STEP_PATTERN = re.compile(
    r"step:(?P<step>\d+)\s+"
    r"smpl:(?P<samples>\d+).*?"
    r"loss:(?P<loss>[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?).*?"
    r"updt_s:(?P<update_seconds>[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)\s+"
    r"data_s:(?P<data_seconds>[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)\s+"
    r"smp/s:(?P<samples_per_second>[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)\s+"
    r"mem_gb:(?P<memory_gb>[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)"
)


def read_log_lines(log_path: Path) -> list[str]:
    return log_path.read_text(encoding="utf-8", errors="replace").splitlines()


def parse_losses_from_lines(lines: list[str]) -> list[dict[str, float | int]]:
    losses: list[dict[str, float | int]] = []
    for line_number, line in enumerate(lines, 1):
        for match in LOSS_PATTERN.finditer(line):
            losses.append({"line": line_number, "loss": float(match.group(1))})
    return losses


def parse_losses(log_path: Path) -> list[dict[str, float | int]]:
    return parse_losses_from_lines(read_log_lines(log_path))


def parse_train_records_from_lines(lines: list[str]) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    for line_number, line in enumerate(lines, 1):
        match = TRAIN_STEP_PATTERN.search(line)
        if not match:
            continue
        records.append(
            {
                "line": line_number,
                "step": int(match.group("step")),
                "samples": int(match.group("samples")),
                "loss": float(match.group("loss")),
                "update_seconds": float(match.group("update_seconds")),
                "data_seconds": float(match.group("data_seconds")),
                "samples_per_second": float(match.group("samples_per_second")),
                "memory_gb": float(match.group("memory_gb")),
            }
        )
    return records


def parse_train_records(log_path: Path) -> list[dict[str, float | int]]:
    return parse_train_records_from_lines(read_log_lines(log_path))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def write_summary(log_path: Path, output_dir: Path) -> Path:
    lines = read_log_lines(log_path)
    losses = parse_losses_from_lines(lines)
    train_records = parse_train_records_from_lines(lines)
    if not losses:
        raise SystemExit(f"ERROR: no loss values found in {log_path}")

    initial_loss = float(losses[0]["loss"])
    final_loss = float(losses[-1]["loss"])
    loss_drop = initial_loss - final_loss
    summary = {
        "log": str(log_path),
        "parsed_loss_count": len(losses),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "min_loss": min(float(item["loss"]) for item in losses),
        "max_loss": max(float(item["loss"]) for item in losses),
        "loss_drop": loss_drop,
        "loss_drop_ratio": loss_drop / initial_loss if initial_loss else 0.0,
        "loss_decreased": final_loss < initial_loss,
        "losses": losses,
        "train_records": train_records,
        "step_metrics": {
            "parsed_step_count": len(train_records),
            "mean_update_seconds": mean([float(item["update_seconds"]) for item in train_records]),
            "mean_data_seconds": mean([float(item["data_seconds"]) for item in train_records]),
            "mean_samples_per_second": mean([float(item["samples_per_second"]) for item in train_records]),
            "max_memory_gb": max((float(item["memory_gb"]) for item in train_records), default=None),
            "notes": [
                "These metrics are parsed from LeRobot per-step log lines and exclude model initialization time.",
            ],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "loss_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = output_dir / "loss_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# LeRobot Training Loss Summary",
                "",
                f"- parsed_loss_count: {summary['parsed_loss_count']}",
                f"- initial_loss: {initial_loss:.6f}",
                f"- final_loss: {final_loss:.6f}",
                f"- loss_drop_ratio: {summary['loss_drop_ratio']:.2%}",
                f"- loss_decreased: {str(summary['loss_decreased']).lower()}",
                f"- parsed_step_count: {summary['step_metrics']['parsed_step_count']}",
                f"- mean_samples_per_second: {summary['step_metrics']['mean_samples_per_second']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary_path = write_summary(args.log, args.output_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(
        "LOSS_SUMMARY "
        f"loss_decreased={str(summary['loss_decreased']).lower()} "
        f"initial_loss={summary['initial_loss']:.4f} "
        f"final_loss={summary['final_loss']:.4f} "
        f"drop={summary['loss_drop_ratio']:.2%} "
        f"path={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
