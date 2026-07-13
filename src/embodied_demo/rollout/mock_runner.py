from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from embodied_demo.config import dump_yaml, load_resolved_run
from embodied_demo.environments import MockEnvironment, StepResult
from embodied_demo.errors import ConfigurationError
from embodied_demo.policies import ScriptedPolicy
from embodied_demo.schemas import EpisodeResult, EvaluationManifest
from embodied_demo.schemas.enums import TerminationReason
from embodied_demo.schemas.evaluation import LatencyStatistics, TaskAggregate
from embodied_demo.schemas.run import ResolvedRun

JsonObject = dict[str, Any]


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _load_scene(path: Path) -> JsonObject:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"scene file not found: {path}") from exc
    except OSError as exc:
        raise ConfigurationError(f"cannot read scene file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in scene file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"scene file must contain a mapping: {path}")
    return payload


def _resolve_scene_path(config_path: Path, scene_file: str) -> Path:
    raw_path = Path(scene_file).expanduser()
    candidates = (
        [raw_path]
        if raw_path.is_absolute()
        else [Path.cwd() / raw_path, config_path.parent / raw_path]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return candidates[0].resolve()


def _progress_score(resolved: ResolvedRun, stage_progress: dict[str, bool]) -> float:
    return float(
        sum(
            stage.weight
            for stage in resolved.task_spec.evaluation.progress_stages
            if stage_progress.get(stage.id)
        )
    )


def _completed_stages(resolved: ResolvedRun, stage_progress: dict[str, bool]) -> int:
    return sum(
        1
        for stage in resolved.task_spec.evaluation.progress_stages
        if stage_progress.get(stage.id)
    )


def _latency_stats(values: list[float]) -> LatencyStatistics:
    if not values:
        return LatencyStatistics(count=0, mean_ms=0, p50_ms=0, p95_ms=0, max_ms=0)
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return LatencyStatistics(
        count=len(values),
        mean_ms=statistics.fmean(values),
        p50_ms=statistics.median(values),
        p95_ms=ordered[p95_index],
        max_ms=max(values),
    )


def run_mock_demo(config_path: str | Path, output_dir: str | Path | None = None) -> Path:
    config_source = Path(config_path).expanduser().resolve()
    resolved = load_resolved_run(config_source)
    scene_file = resolved.run.environment.config.get("scene_file")
    if not isinstance(scene_file, str) or not scene_file:
        raise ConfigurationError("environment.config.scene_file is required for mock demos")
    scene = _load_scene(_resolve_scene_path(config_source, scene_file))

    seed = resolved.run.evaluation.seeds[0]
    episode_id = f"{resolved.run.name}-seed{seed}-episode000"
    run_id = episode_id
    root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else Path(resolved.run.runtime.output_dir).resolve()
    )
    artifact_dir = root / resolved.run.name / run_id

    _write_text(artifact_dir / "resolved_config.yaml", dump_yaml(resolved))
    _write_text(artifact_dir / "task_snapshot.yaml", dump_yaml(resolved.task_spec))

    policy = ScriptedPolicy(resolved)
    environment = MockEnvironment(resolved, scene, episode_id=episode_id, seed=seed)
    policy.reset()
    observation = environment.reset()
    events: list[JsonObject] = [
        {
            "event": "episode_start",
            "run_id": run_id,
            "episode_id": episode_id,
            "task_id": resolved.task_spec.id,
            "seed": seed,
        }
    ]
    latencies_ms: list[float] = []
    result: StepResult | None = None

    while True:
        policy.update_observation(observation)
        action = policy.get_action()
        latencies_ms.append(0.0)
        result = environment.step(action)
        observation = result.observation
        events.append(
            {
                "event": "step",
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
    wall_time_s = float(environment.step_id)
    final_progress = _progress_score(resolved, result.stage_progress)
    final_completed = _completed_stages(resolved, result.stage_progress)
    total_stages = len(resolved.task_spec.evaluation.progress_stages)
    episode_result = EpisodeResult(
        run_id=run_id,
        episode_id=episode_id,
        task_id=resolved.task_spec.id,
        task_version=resolved.task_spec.version,
        backend=resolved.run.environment.backend,
        profile=resolved.run.evaluation.profile,
        seed=seed,
        layout_id=resolved.task_spec.evaluation.standard_layouts[0],
        valid=True,
        episode_success=result.success,
        progress_score=final_progress,
        completed_stages=final_completed,
        total_stages=total_stages,
        failure_type=result.failure_type,
        termination_reason=result.termination_reason or TerminationReason.TASK_FAILURE,
        episode_steps=environment.step_id,
        wall_time_s=wall_time_s,
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
        task_progress_score=final_progress,
        progress_std=0.0,
        confidence_interval_95=(final_progress, final_progress),
    )
    manifest = EvaluationManifest(
        run_id=run_id,
        task_version=resolved.task_spec.version,
        evaluator_commit="local-working-tree",
        profile=resolved.run.evaluation.profile,
        seeds=[seed],
        checkpoint="scripted_mock_policy",
        normalizer="none",
        container_image="none",
        policy_transport_version=resolved.run.policy.transport.value,
        environment_transport_version=resolved.run.environment.backend.value,
        artifacts={
            "resolved_config": "resolved_config.yaml",
            "task_snapshot": "task_snapshot.yaml",
            "events": "events.jsonl",
            "result": "result.json",
            "metrics": "metrics.json",
            "report": "report.md",
        },
    )

    events.append(
        {
            "event": "episode_end",
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
        },
    )
    _write_text(artifact_dir / "manifest.yaml", dump_yaml(manifest))
    _write_text(artifact_dir / "report.md", _render_report(resolved, episode_result, aggregate))
    return artifact_dir


def _render_report(
    resolved: ResolvedRun,
    episode_result: EpisodeResult,
    aggregate: TaskAggregate,
) -> str:
    status = "SUCCESS" if episode_result.episode_success else "FAILED"
    return "\n".join(
        [
            f"# Demo Run: {resolved.run.name}",
            "",
            f"- status: {status}",
            f"- task: {episode_result.task_id}@{episode_result.task_version}",
            f"- backend: {episode_result.backend.value}",
            f"- profile: {episode_result.profile.value}",
            f"- seed: {episode_result.seed}",
            f"- progress: {episode_result.progress_score:.1f}/100",
            f"- stages: {episode_result.completed_stages}/{episode_result.total_stages}",
            f"- steps: {episode_result.episode_steps}",
            f"- success_rate: {aggregate.task_success_rate:.2f}",
            "",
            "This is a deterministic mock demo. It validates the project pipeline",
            "contract, artifact writing, scripted policy lifecycle, and predicate",
            "evaluation shape. It does not claim simulator or real-robot capability.",
            "",
        ]
    )
