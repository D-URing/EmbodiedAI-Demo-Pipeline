from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LOSS_PATTERN = re.compile(
    r"(?:^|[^a-zA-Z_])(?:train_)?loss(?:=|:|\s+)([-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)"
)


def parse_losses(log_path: Path) -> list[dict[str, float | int]]:
    losses: list[dict[str, float | int]] = []
    for line_number, line in enumerate(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for match in LOSS_PATTERN.finditer(line):
            losses.append({"line": line_number, "loss": float(match.group(1))})
    return losses


def write_summary(log_path: Path, output_dir: Path) -> Path:
    losses = parse_losses(log_path)
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
