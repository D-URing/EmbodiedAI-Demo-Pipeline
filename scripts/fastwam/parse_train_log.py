from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


NUMBER = r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?"
TRAIN_STEP_PATTERN = re.compile(
    rf"\[train\].*?epoch=(?P<epoch>\d+).*?step=(?P<step>\d+)/(?P<max_steps>\d+)"
)
EVAL_STEP_PATTERN = re.compile(rf"\[eval\].*?step=(?P<step>\d+)")
KEY_VALUE_PATTERN = re.compile(rf"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>{NUMBER})")
CKPT_PATTERN = re.compile(
    r"\[(?P<kind>ckpt|done)\].*?step=(?P<step>\d+)(?:.*?weights=(?P<weights>\S+))?(?:.*?state=(?P<state>\S+))?"
)


def _parse_metrics(line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in KEY_VALUE_PATTERN.finditer(line):
        key = match.group("key")
        if key in {"epoch", "step", "max_steps", "eta"}:
            continue
        try:
            metrics[key] = float(match.group("value"))
        except ValueError:
            continue
    return metrics


def parse_log(log_path: Path) -> dict[str, list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    for line_number, line in enumerate(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if "[train]" in line and "loss=" in line:
            step_match = TRAIN_STEP_PATTERN.search(line)
            metrics = _parse_metrics(line)
            if step_match and "loss" in metrics:
                train.append(
                    {
                        "line": line_number,
                        "epoch": int(step_match.group("epoch")),
                        "step": int(step_match.group("step")),
                        "max_steps": int(step_match.group("max_steps")),
                        "metrics": metrics,
                    }
                )
            continue

        if "[eval]" in line:
            step_match = EVAL_STEP_PATTERN.search(line)
            metrics = _parse_metrics(line)
            if step_match and metrics:
                eval_records.append(
                    {
                        "line": line_number,
                        "step": int(step_match.group("step")),
                        "metrics": metrics,
                    }
                )
            continue

        ckpt_match = CKPT_PATTERN.search(line)
        if ckpt_match:
            checkpoints.append(
                {
                    "line": line_number,
                    "kind": ckpt_match.group("kind"),
                    "step": int(ckpt_match.group("step")),
                    "weights": ckpt_match.group("weights"),
                    "state": ckpt_match.group("state"),
                }
            )

    return {"train": train, "eval": eval_records, "checkpoints": checkpoints}


def _metric_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for record in records for key in record["metrics"]})
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(record["metrics"][key]) for record in records if key in record["metrics"]]
        if not values:
            continue
        initial = values[0]
        final = values[-1]
        summary[key] = {
            "initial": initial,
            "final": final,
            "min": min(values),
            "max": max(values),
            "delta": final - initial,
        }
    return summary


def summarize(parsed: dict[str, list[dict[str, Any]]], log_path: Path) -> dict[str, Any]:
    train = parsed["train"]
    checkpoints = parsed["checkpoints"]
    metric_summary = _metric_summary(train)

    if train:
        initial_loss = float(train[0]["metrics"]["loss"])
        final_loss = float(train[-1]["metrics"]["loss"])
        loss_drop = initial_loss - final_loss
        loss_drop_ratio = loss_drop / initial_loss if initial_loss else 0.0
        loss_decreased: bool | None = final_loss < initial_loss if len(train) >= 2 else None
        final_step = int(train[-1]["step"])
        max_steps = int(train[-1]["max_steps"])
    else:
        initial_loss = final_loss = loss_drop = loss_drop_ratio = None
        loss_decreased = None
        final_step = max_steps = None

    latest_checkpoint = checkpoints[-1] if checkpoints else None
    training_completed = bool(latest_checkpoint and latest_checkpoint["kind"] == "done")

    return {
        "log": str(log_path),
        "parsed_train_count": len(train),
        "parsed_eval_count": len(parsed["eval"]),
        "parsed_checkpoint_count": len(checkpoints),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_drop": loss_drop,
        "loss_drop_ratio": loss_drop_ratio,
        "loss_decreased": loss_decreased,
        "final_step": final_step,
        "max_steps": max_steps,
        "training_completed": training_completed,
        "latest_checkpoint": latest_checkpoint,
        "metric_summary": metric_summary,
        "train": train,
        "eval": parsed["eval"],
        "checkpoints": checkpoints,
    }


def write_summary(log_path: Path, output_dir: Path) -> Path:
    parsed = parse_log(log_path)
    summary = summarize(parsed, log_path)
    if summary["parsed_train_count"] == 0:
        raise SystemExit(f"ERROR: no FastWAM train loss records found in {log_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "loss_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    loss_decreased = summary["loss_decreased"]
    loss_decreased_text = "unknown" if loss_decreased is None else str(loss_decreased).lower()
    report_lines = [
        "# FastWAM Training Loss Summary",
        "",
        f"- parsed_train_count: {summary['parsed_train_count']}",
        f"- initial_loss: {summary['initial_loss']:.6f}",
        f"- final_loss: {summary['final_loss']:.6f}",
        f"- loss_drop_ratio: {summary['loss_drop_ratio']:.2%}",
        f"- loss_decreased: {loss_decreased_text}",
        f"- final_step: {summary['final_step']} / {summary['max_steps']}",
        f"- training_completed: {str(summary['training_completed']).lower()}",
    ]
    latest = summary.get("latest_checkpoint")
    if latest:
        report_lines.extend(
            [
                f"- latest_weights: {latest.get('weights')}",
                f"- latest_state: {latest.get('state')}",
            ]
        )
    report_lines.append("")
    report_path = output_dir / "loss_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    summary_path = write_summary(args.log, args.output_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    loss_decreased = summary["loss_decreased"]
    loss_decreased_text = "unknown" if loss_decreased is None else str(loss_decreased).lower()
    print(
        "FASTWAM_LOSS_SUMMARY "
        f"loss_decreased={loss_decreased_text} "
        f"initial_loss={summary['initial_loss']:.4f} "
        f"final_loss={summary['final_loss']:.4f} "
        f"drop={summary['loss_drop_ratio']:.2%} "
        f"steps={summary['final_step']}/{summary['max_steps']} "
        f"path={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
