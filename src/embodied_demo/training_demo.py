from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_demo.config import load_resolved_run
from embodied_demo.demo_runner import (
    MockEnvironment,
    _completed_stages,
    _latency_stats,
    _load_scene,
    _progress_score,
    _resolve_scene_path,
)
from embodied_demo.errors import ConfigurationError
from embodied_demo.schemas import ActionChunk, EpisodeResult
from embodied_demo.schemas.enums import CoordinateFrame, TerminationReason
from embodied_demo.schemas.evaluation import TaskAggregate
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


def _feature_vector_for_step(task_id: str, step_index: int, action_vocab: list[str]) -> list[float]:
    task_bias = 1.0 if task_id == "tabletop_sorting_v1" else -1.0
    clamped_step = min(max(step_index, 0), len(action_vocab) - 1)
    step_one_hot = [0.0 for _ in action_vocab]
    step_one_hot[clamped_step] = 1.0
    progress = clamped_step / max(1, len(action_vocab) - 1)
    return [1.0, task_bias, progress, *step_one_hot]


def _action_from_label(label: str) -> JsonObject:
    parts = label.split(":")
    if parts[0] == "select" and len(parts) == 2:
        return {"skill": "select", "object_id": parts[1]}
    if parts[0] == "grasp" and len(parts) == 2:
        return {"skill": "grasp", "object_id": parts[1]}
    if parts[0] == "place" and len(parts) == 3:
        return {"skill": "place", "object_id": parts[1], "target_id": parts[2]}
    if parts[0] == "finish":
        return {"skill": "finish"}
    if parts[0] == "select_corners" and len(parts) == 2:
        return {"skill": "select_corners", "corners": parts[1].split("+")}
    if parts[0] == "fold" and len(parts) == 3:
        return {"skill": "fold", "fold_index": int(parts[1]), "axis": parts[2]}
    if parts[0] == "align" and len(parts) == 2:
        return {"skill": "align", "fold_index": int(parts[1])}
    raise ConfigurationError(f"cannot decode action label: {label}")


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


def _train_accuracy(
    weights: list[list[float]],
    samples: list[TrainingSample],
    action_vocab: list[str],
) -> float:
    correct = 0
    for sample in samples:
        probs = _predict_proba(weights, sample.feature_vector)
        predicted = action_vocab[max(range(len(probs)), key=probs.__getitem__)]
        correct += int(predicted == sample.action_label)
    return correct / len(samples)


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
    train_accuracy = _train_accuracy(weights, samples, action_vocab)

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
            "train_accuracy": train_accuracy,
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


class LearnedBehaviorCloningPolicy:
    """Loads a trained checkpoint and uses it for rollout decisions."""

    def __init__(
        self,
        resolved: ResolvedRun,
        action_vocab: list[str],
        weights: list[list[float]],
    ) -> None:
        self._resolved = resolved
        self._action_vocab = action_vocab
        self._weights = weights
        self._last_observation = None

    def reset(self) -> None:
        self._last_observation = None

    def update_observation(self, observation: Any) -> None:
        self._last_observation = observation

    def get_action(self) -> ActionChunk:
        if self._last_observation is None:
            step_index = 0
        else:
            step_index = self._last_observation.step_id
        features = _feature_vector_for_step(
            self._resolved.task_spec.id,
            step_index,
            self._action_vocab,
        )
        probs = _predict_proba(self._weights, features)
        predicted_index = max(range(len(probs)), key=probs.__getitem__)
        label = self._action_vocab[predicted_index]
        return ActionChunk(
            representation=self._resolved.run.policy.action_type,
            frame=CoordinateFrame.WORLD,
            control_frequency_hz=1.0,
            horizon=1,
            actions=[_action_from_label(label)],
            valid_mask=[True],
            metadata={
                "policy_name": f"{self._resolved.run.policy.name}_learned_bc",
                "policy_family": "learned_behavior_cloning_demo",
                "action_label": label,
                "confidence": round(probs[predicted_index], 6),
                "checkpoint": "checkpoint.json",
            },
        )


def train_and_eval_behavior_cloning_demo(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    epochs: int = 30,
    learning_rate: float = 1.2,
) -> Path:
    artifact_dir = train_behavior_cloning_demo(
        config_path,
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    config_source = Path(config_path).expanduser().resolve()
    resolved = load_resolved_run(config_source)
    checkpoint = json.loads((artifact_dir / "checkpoint.json").read_text(encoding="utf-8"))
    scene_file = resolved.run.environment.config.get("scene_file")
    if not isinstance(scene_file, str) or not scene_file:
        raise ConfigurationError("environment.config.scene_file is required for train-eval demos")
    scene = _load_scene(_resolve_scene_path(config_source, scene_file))
    rollout_dir = artifact_dir / "learned_rollout"
    episode_result = _run_learned_rollout(
        resolved=resolved,
        scene=scene,
        checkpoint=checkpoint,
        artifact_dir=rollout_dir,
    )

    metrics_path = artifact_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["learned_rollout"] = {
        "episode_success": episode_result.episode_success,
        "progress_score": episode_result.progress_score,
        "episode_steps": episode_result.episode_steps,
        "result": "learned_rollout/result.json",
        "events": "learned_rollout/events.jsonl",
        "report": "learned_rollout/report.md",
    }
    metrics["true_flow_complete"] = (
        metrics["loss_decreased"]
        and episode_result.episode_success
        and episode_result.progress_score == 100
    )
    _write_json(metrics_path, metrics)
    _write_text(
        artifact_dir / "report.md",
        _render_train_eval_report(
            resolved=resolved,
            metrics=metrics,
            episode_result=episode_result,
        ),
    )
    return artifact_dir


def _run_learned_rollout(
    resolved: ResolvedRun,
    scene: JsonObject,
    checkpoint: JsonObject,
    artifact_dir: Path,
) -> EpisodeResult:
    action_vocab = checkpoint["action_vocab"]
    weights = checkpoint["weights"]
    seed = resolved.run.evaluation.seeds[0]
    episode_id = f"{resolved.run.name}-learned-bc-seed{seed}-episode000"
    policy = LearnedBehaviorCloningPolicy(resolved, action_vocab, weights)
    environment = MockEnvironment(resolved, scene, episode_id=episode_id, seed=seed)
    policy.reset()
    observation = environment.reset()
    events: list[JsonObject] = [
        {
            "event": "learned_episode_start",
            "episode_id": episode_id,
            "task_id": resolved.task_spec.id,
            "checkpoint": "checkpoint.json",
        }
    ]
    latencies_ms: list[float] = []
    result = None

    while True:
        policy.update_observation(observation)
        action = policy.get_action()
        latencies_ms.append(0.0)
        result = environment.step(action)
        observation = result.observation
        events.append(
            {
                "event": "learned_step",
                "step_id": environment.step_id,
                "action": action.model_dump(mode="json"),
                "stage_progress": result.stage_progress,
                "progress_score": _progress_score(resolved, result.stage_progress),
                "done": result.done,
                "success": result.success,
            }
        )
        if result.done:
            break

    assert result is not None
    progress = _progress_score(resolved, result.stage_progress)
    episode_result = EpisodeResult(
        run_id=episode_id,
        episode_id=episode_id,
        task_id=resolved.task_spec.id,
        task_version=resolved.task_spec.version,
        backend=resolved.run.environment.backend,
        profile=resolved.run.evaluation.profile,
        seed=seed,
        layout_id=resolved.task_spec.evaluation.standard_layouts[0],
        valid=True,
        episode_success=result.success,
        progress_score=progress,
        completed_stages=_completed_stages(resolved, result.stage_progress),
        total_stages=len(resolved.task_spec.evaluation.progress_stages),
        failure_type=result.failure_type,
        termination_reason=result.termination_reason or TerminationReason.TASK_FAILURE,
        episode_steps=environment.step_id,
        wall_time_s=float(environment.step_id),
        policy_latency=_latency_stats(latencies_ms),
        artifacts={
            "events": "events.jsonl",
            "metrics": "metrics.json",
            "report": "report.md",
        },
    )
    aggregate = TaskAggregate(
        task_id=resolved.task_spec.id,
        task_version=resolved.task_spec.version,
        profile=resolved.run.evaluation.profile,
        total_episodes=1,
        valid_episodes=1,
        task_success_rate=1.0 if result.success else 0.0,
        task_progress_score=progress,
        progress_std=0.0,
        confidence_interval_95=(progress, progress),
    )
    events.append(
        {
            "event": "learned_episode_end",
            "success": episode_result.episode_success,
            "progress_score": episode_result.progress_score,
            "termination_reason": episode_result.termination_reason.value,
        }
    )
    _write_text(
        artifact_dir / "events.jsonl",
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
    )
    _write_json(artifact_dir / "result.json", episode_result.model_dump(mode="json"))
    _write_json(
        artifact_dir / "metrics.json",
        {
            "episode": episode_result.model_dump(mode="json"),
            "aggregate": aggregate.model_dump(mode="json"),
            "policy_source": "../checkpoint.json",
        },
    )
    _write_text(artifact_dir / "report.md", _render_learned_rollout_report(episode_result))
    return episode_result


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


def _render_learned_rollout_report(episode_result: EpisodeResult) -> str:
    status = "SUCCESS" if episode_result.episode_success else "FAILED"
    return "\n".join(
        [
            f"# Learned Rollout: {episode_result.task_id}",
            "",
            f"- status: {status}",
            "- policy: checkpoint.json loaded as learned_behavior_cloning_demo",
            f"- progress: {episode_result.progress_score:.1f}/100",
            f"- stages: {episode_result.completed_stages}/{episode_result.total_stages}",
            f"- steps: {episode_result.episode_steps}",
            "",
        ]
    )


def _render_train_eval_report(
    resolved: ResolvedRun,
    metrics: JsonObject,
    episode_result: EpisodeResult,
) -> str:
    status = "PASS" if metrics["true_flow_complete"] else "FAIL"
    return "\n".join(
        [
            f"# Train-Eval Demo: {resolved.run.name}",
            "",
            f"- status: {status}",
            f"- task: {resolved.task_spec.id}@{resolved.task_spec.version}",
            "- flow: mock expert dataset -> train BC policy -> save checkpoint -> load checkpoint -> learned rollout",
            f"- initial_loss: {metrics['initial_loss']:.6f}",
            f"- final_loss: {metrics['final_loss']:.6f}",
            f"- loss_drop_ratio: {metrics['loss_drop_ratio']:.2%}",
            f"- train_accuracy: {metrics['train_accuracy']:.2%}",
            f"- learned_rollout_success: {episode_result.episode_success}",
            f"- learned_rollout_progress: {episode_result.progress_score:.1f}/100",
            f"- learned_rollout_steps: {episode_result.episode_steps}",
            "",
            "This is the first complete local training-to-execution pipeline.",
            "The model is intentionally small and CPU-only, but the checkpoint",
            "is actually loaded and used to choose actions in a rollout.",
            "",
        ]
    )
