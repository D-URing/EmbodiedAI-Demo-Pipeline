from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_demo.config import load_resolved_run
from embodied_demo.errors import ConfigurationError
from embodied_demo.schemas.run import ResolvedRun

JsonObject = dict[str, Any]


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


@dataclass(frozen=True)
class TrainingSample:
    sample_id: str
    task_id: str
    step_index: int
    feature_vector: list[float]
    action_label: str
    observation_summary: JsonObject


def _expert_action_labels(task_id: str) -> list[str]:
    if task_id == "tabletop_sorting_v1":
        return [
            "select:red_block",
            "grasp:red_block",
            "place:red_block:red_zone",
            "select:blue_block",
            "grasp:blue_block",
            "place:blue_block:blue_zone",
            "finish",
        ]
    if task_id == "towel_folding_v1":
        return [
            "select_corners:top_left+top_right",
            "fold:1:long",
            "align:1",
            "fold:2:short",
            "align:2",
        ]
    raise ConfigurationError(f"no training demo labels are available for task: {task_id}")


def _build_dataset(resolved: ResolvedRun, repeats: int = 16) -> tuple[list[TrainingSample], list[str]]:
    labels = _expert_action_labels(resolved.task_spec.id)
    action_vocab = list(dict.fromkeys(labels))
    samples: list[TrainingSample] = []
    task_bias = 1.0 if resolved.task_spec.id == "tabletop_sorting_v1" else -1.0
    for repeat in range(repeats):
        for step_index, label in enumerate(labels):
            step_one_hot = [0.0 for _ in labels]
            step_one_hot[step_index] = 1.0
            progress = step_index / max(1, len(labels) - 1)
            feature_vector = [1.0, task_bias, progress, *step_one_hot]
            samples.append(
                TrainingSample(
                    sample_id=f"{resolved.run.name}-r{repeat:02d}-s{step_index:02d}",
                    task_id=resolved.task_spec.id,
                    step_index=step_index,
                    feature_vector=feature_vector,
                    action_label=label,
                    observation_summary={
                        "instruction": resolved.task_spec.instruction.canonical,
                        "stage_hint": resolved.task_spec.stages[
                            min(step_index, len(resolved.task_spec.stages) - 1)
                        ],
                        "progress_hint": round(progress, 4),
                    },
                )
            )
    return samples, action_vocab


def _softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def _cross_entropy(
    weights: list[list[float]],
    samples: list[TrainingSample],
    label_to_index: dict[str, int],
) -> float:
    total = 0.0
    for sample in samples:
        probs = _predict_proba(weights, sample.feature_vector)
        target = label_to_index[sample.action_label]
        total -= math.log(max(probs[target], 1e-12))
    return total / len(samples)


def _predict_proba(weights: list[list[float]], features: list[float]) -> list[float]:
    logits = [
        sum(weight * feature for weight, feature in zip(class_weights, features))
        for class_weights in weights
    ]
    return _softmax(logits)


def _train_softmax_classifier(
    samples: list[TrainingSample],
    action_vocab: list[str],
    epochs: int,
    learning_rate: float,
) -> tuple[list[list[float]], list[JsonObject]]:
    if epochs <= 0:
        raise ConfigurationError("epochs must be > 0")
    if learning_rate <= 0:
        raise ConfigurationError("learning_rate must be > 0")
    feature_dim = len(samples[0].feature_vector)
    label_to_index = {label: index for index, label in enumerate(action_vocab)}
    weights = [[0.0 for _ in range(feature_dim)] for _ in action_vocab]
    log: list[JsonObject] = []

    for epoch in range(epochs + 1):
        loss = _cross_entropy(weights, samples, label_to_index)
        log.append({"epoch": epoch, "train_loss": round(loss, 8)})
        if epoch == epochs:
            break

        gradients = [[0.0 for _ in range(feature_dim)] for _ in action_vocab]
        for sample in samples:
            probs = _predict_proba(weights, sample.feature_vector)
            target = label_to_index[sample.action_label]
            for class_index in range(len(action_vocab)):
                error = probs[class_index] - (1.0 if class_index == target else 0.0)
                for feature_index, feature in enumerate(sample.feature_vector):
                    gradients[class_index][feature_index] += error * feature / len(samples)

        for class_index in range(len(action_vocab)):
            for feature_index in range(feature_dim):
                weights[class_index][feature_index] -= (
                    learning_rate * gradients[class_index][feature_index]
                )

    return weights, log


def train_behavior_cloning_demo(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    epochs: int = 30,
    learning_rate: float = 1.2,
) -> Path:
    config_source = Path(config_path).expanduser().resolve()
    resolved = load_resolved_run(config_source)
    samples, action_vocab = _build_dataset(resolved)
    weights, train_log = _train_softmax_classifier(samples, action_vocab, epochs, learning_rate)
    initial_loss = float(train_log[0]["train_loss"])
    final_loss = float(train_log[-1]["train_loss"])
    loss_drop = initial_loss - final_loss
    loss_drop_ratio = loss_drop / initial_loss if initial_loss else 0.0

    train_id = f"{resolved.run.name}-bc-demo-epochs{epochs}"
    root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else Path(resolved.run.runtime.output_dir).resolve()
    )
    artifact_dir = root / "training" / resolved.run.name / train_id

    _write_text(
        artifact_dir / "dataset.jsonl",
        "".join(
            json.dumps(
                {
                    "sample_id": sample.sample_id,
                    "task_id": sample.task_id,
                    "step_index": sample.step_index,
                    "feature_vector": sample.feature_vector,
                    "action_label": sample.action_label,
                    "observation_summary": sample.observation_summary,
                },
                ensure_ascii=False,
            )
            + "\n"
            for sample in samples
        ),
    )
    _write_text(
        artifact_dir / "train_log.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train_log),
    )
    _write_json(
        artifact_dir / "checkpoint.json",
        {
            "model_type": "softmax_behavior_cloning_demo",
            "task_id": resolved.task_spec.id,
            "feature_schema": ["bias", "task_bias", "progress", "step_one_hot..."],
            "action_vocab": action_vocab,
            "weights": weights,
        },
    )
    _write_json(
        artifact_dir / "metrics.json",
        {
            "task_id": resolved.task_spec.id,
            "run_name": resolved.run.name,
            "reference": "lerobot_style_behavior_cloning_minimal",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "num_samples": len(samples),
            "num_actions": len(action_vocab),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_drop": loss_drop,
            "loss_drop_ratio": loss_drop_ratio,
            "loss_decreased": final_loss < initial_loss,
            "artifacts": {
                "dataset": "dataset.jsonl",
                "train_log": "train_log.jsonl",
                "checkpoint": "checkpoint.json",
                "report": "report.md",
            },
        },
    )
    _write_text(
        artifact_dir / "report.md",
        _render_training_report(
            resolved=resolved,
            epochs=epochs,
            learning_rate=learning_rate,
            num_samples=len(samples),
            initial_loss=initial_loss,
            final_loss=final_loss,
            loss_drop_ratio=loss_drop_ratio,
        ),
    )
    return artifact_dir


def _render_training_report(
    resolved: ResolvedRun,
    epochs: int,
    learning_rate: float,
    num_samples: int,
    initial_loss: float,
    final_loss: float,
    loss_drop_ratio: float,
) -> str:
    status = "PASS" if final_loss < initial_loss else "FAIL"
    return "\n".join(
        [
            f"# Training Demo: {resolved.run.name}",
            "",
            f"- status: {status}",
            f"- task: {resolved.task_spec.id}@{resolved.task_spec.version}",
            "- model: softmax behavior cloning demo",
            "- reference style: LeRobot-style dataset -> policy -> train log -> checkpoint",
            f"- samples: {num_samples}",
            f"- epochs: {epochs}",
            f"- learning_rate: {learning_rate}",
            f"- initial_loss: {initial_loss:.6f}",
            f"- final_loss: {final_loss:.6f}",
            f"- loss_drop_ratio: {loss_drop_ratio:.2%}",
            "",
            "This is a minimal training pipeline demo. It proves that supervised",
            "behavior-cloning training artifacts can be produced and that the",
            "loss decreases on a deterministic mock dataset. It is not a",
            "large VLA model or simulator-backed training result.",
            "",
        ]
    )
