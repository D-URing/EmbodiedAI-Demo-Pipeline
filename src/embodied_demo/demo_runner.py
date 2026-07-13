from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from embodied_demo.config import dump_yaml, load_resolved_run
from embodied_demo.errors import ConfigurationError
from embodied_demo.schemas import ActionChunk, EpisodeResult, EvaluationManifest, Observation
from embodied_demo.schemas.enums import CoordinateFrame, FailureType, TerminationReason
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


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    action: ActionChunk
    stage_progress: dict[str, bool]
    done: bool
    success: bool
    failure_type: FailureType | None = None
    termination_reason: TerminationReason | None = None


class ScriptedPolicy:
    """A tiny XPolicyLab-shaped policy adapter for first runnable demos."""

    def __init__(self, resolved: ResolvedRun) -> None:
        self._resolved = resolved
        self._action_plan = self._build_action_plan(resolved.task_spec.id)
        self._cursor = 0
        self._last_observation: Observation | None = None

    def reset(self) -> None:
        self._cursor = 0
        self._last_observation = None

    def update_observation(self, observation: Observation) -> None:
        self._last_observation = observation

    def get_action(self) -> ActionChunk:
        if self._cursor < len(self._action_plan):
            action = self._action_plan[self._cursor]
        else:
            action = {"skill": "finish"}
        self._cursor += 1
        return ActionChunk(
            representation=self._resolved.run.policy.action_type,
            frame=CoordinateFrame.WORLD,
            control_frequency_hz=1.0,
            horizon=1,
            actions=[action],
            valid_mask=[True],
            metadata={
                "policy_name": self._resolved.run.policy.name,
                "policy_family": "scripted_mock",
                "reference_lifecycle": "xpolicylab_demo_policy_debug_v1",
            },
        )

    def _build_action_plan(self, task_id: str) -> list[JsonObject]:
        if task_id == "tabletop_sorting_v1":
            return [
                {"skill": "select", "object_id": "red_block"},
                {"skill": "grasp", "object_id": "red_block"},
                {"skill": "place", "object_id": "red_block", "target_id": "red_zone"},
                {"skill": "select", "object_id": "blue_block"},
                {"skill": "grasp", "object_id": "blue_block"},
                {"skill": "place", "object_id": "blue_block", "target_id": "blue_zone"},
                {"skill": "finish"},
            ]
        if task_id == "kitchen_counter_sorting_v1":
            return [
                {"skill": "select", "object_id": "tomato"},
                {"skill": "grasp", "object_id": "tomato"},
                {"skill": "place", "object_id": "tomato", "target_id": "prep_tray"},
                {"skill": "select", "object_id": "bowl"},
                {"skill": "grasp", "object_id": "bowl"},
                {"skill": "place", "object_id": "bowl", "target_id": "dish_rack"},
                {"skill": "select", "object_id": "spice_jar"},
                {"skill": "grasp", "object_id": "spice_jar"},
                {"skill": "place", "object_id": "spice_jar", "target_id": "spice_caddy"},
                {"skill": "finish"},
            ]
        if task_id == "drawer_pick_place_v1":
            return [
                {"skill": "locate_handle", "handle_id": "drawer_handle"},
                {"skill": "open_drawer", "drawer_id": "drawer"},
                {"skill": "select", "object_id": "spoon"},
                {"skill": "grasp", "object_id": "spoon"},
                {"skill": "place", "object_id": "spoon", "target_id": "counter_tray"},
                {"skill": "finish"},
            ]
        if task_id == "towel_folding_v1":
            return [
                {"skill": "select_corners", "corners": ["top_left", "top_right"]},
                {"skill": "fold", "fold_index": 1, "axis": "long"},
                {"skill": "align", "fold_index": 1},
                {"skill": "fold", "fold_index": 2, "axis": "short"},
                {"skill": "align", "fold_index": 2},
            ]
        raise ConfigurationError(f"no scripted policy is available for task: {task_id}")


class MockEnvironment:
    def __init__(self, resolved: ResolvedRun, scene: JsonObject, episode_id: str, seed: int) -> None:
        self._resolved = resolved
        self._scene = scene
        self._episode_id = episode_id
        self._seed = seed
        self._step_id = 0
        self._done = False
        self._success = False
        self._failure_type: FailureType | None = None
        self._termination_reason: TerminationReason | None = None
        self._stage_progress = {
            stage.id: False for stage in resolved.task_spec.evaluation.progress_stages
        }
        self._base_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seed)
        self._state = self._initial_state(resolved.task_spec.id, scene)

    def reset(self) -> Observation:
        self._step_id = 0
        return self._make_observation()

    @property
    def step_id(self) -> int:
        return self._step_id

    def step(self, action: ActionChunk) -> StepResult:
        action_payload = action.actions[0]
        skill = action_payload.get("skill")
        task_id = self._resolved.task_spec.id
        if task_id == "tabletop_sorting_v1":
            self._apply_tabletop(skill, action_payload)
        elif task_id == "kitchen_counter_sorting_v1":
            self._apply_kitchen(skill, action_payload)
        elif task_id == "drawer_pick_place_v1":
            self._apply_drawer(skill, action_payload)
        elif task_id == "towel_folding_v1":
            self._apply_towel(skill, action_payload)
        else:
            raise ConfigurationError(f"no mock environment is available for task: {task_id}")

        self._step_id += 1
        if self._step_id >= self._resolved.task_spec.termination.max_steps and not self._done:
            self._done = True
            self._success = False
            self._failure_type = FailureType.TIMEOUT
            self._termination_reason = TerminationReason.MAX_STEPS

        return StepResult(
            observation=self._make_observation(),
            action=action,
            stage_progress=dict(self._stage_progress),
            done=self._done,
            success=self._success,
            failure_type=self._failure_type,
            termination_reason=self._termination_reason,
        )

    def _initial_state(self, task_id: str, scene: JsonObject) -> JsonObject:
        if task_id == "tabletop_sorting_v1":
            objects = {
                item["id"]: {
                    "kind": item.get("kind"),
                    "color": item.get("color"),
                    "category": item.get("category"),
                    "accepts_categories": item.get("accepts_categories", []),
                    "location": item.get("location"),
                    "position": item.get("position"),
                    "placed": item.get("placed", False),
                }
                for item in scene.get("objects", [])
            }
            return {"objects": objects, "selected": None, "held": None, "safe_transport": True}
        if task_id == "kitchen_counter_sorting_v1":
            objects = {
                item["id"]: {
                    "kind": item.get("kind"),
                    "category": item.get("category"),
                    "accepts_categories": item.get("accepts_categories", []),
                    "location": item.get("location", "counter"),
                    "position": item.get("position"),
                    "placed": item.get("placed", False),
                }
                for item in scene.get("objects", [])
            }
            return {"objects": objects, "selected": None, "held": None, "safe_transport": True}
        if task_id == "drawer_pick_place_v1":
            objects = {
                item["id"]: {
                    "kind": item.get("kind"),
                    "category": item.get("category"),
                    "accepts_categories": item.get("accepts_categories", []),
                    "location": item.get("location"),
                    "position": item.get("position"),
                    "placed": item.get("placed", False),
                }
                for item in scene.get("objects", [])
            }
            drawer = scene.get("drawer", {})
            return {
                "drawer": {
                    "id": drawer.get("id", "drawer"),
                    "state": drawer.get("state", "closed"),
                    "handle_id": drawer.get("handle_id", "drawer_handle"),
                    "position": drawer.get("position"),
                },
                "objects": objects,
                "selected": None,
                "held": None,
            }
        if task_id == "towel_folding_v1":
            towel = scene.get("towel", {})
            return {
                "towel": {
                    "id": towel.get("id", "rectangular_towel"),
                    "polygon": towel.get("polygon", []),
                    "selected_corners": [],
                    "folds_completed": 0,
                    "first_alignment": 0.0,
                    "final_alignment": 0.0,
                }
            }
        raise ConfigurationError(f"no mock state initializer is available for task: {task_id}")

    def _make_observation(self) -> Observation:
        return Observation(
            episode_id=self._episode_id,
            step_id=self._step_id,
            timestamp=self._base_time + timedelta(seconds=self._step_id),
            instruction=self._resolved.task_spec.instruction.canonical,
            state=self._state,
            task_context={
                "task_id": self._resolved.task_spec.id,
                "scene_id": self._scene.get("scene_id"),
                "stage_progress": dict(self._stage_progress),
            },
            metadata={"backend": "mock_2d", "seed": self._seed},
        )

    def _apply_tabletop(self, skill: str | None, action: JsonObject) -> None:
        objects = self._state["objects"]
        if skill == "select":
            object_id = action.get("object_id")
            if (
                object_id in objects
                and objects[object_id]["kind"] == "movable"
                and not objects[object_id]["placed"]
            ):
                self._state["selected"] = object_id
                self._stage_progress["select_target"] = True
        elif skill == "grasp":
            object_id = action.get("object_id")
            if object_id == self._state.get("selected"):
                self._state["held"] = object_id
                self._stage_progress["grasp_target"] = True
                self._stage_progress["transport_target"] = True
        elif skill == "place":
            object_id = action.get("object_id")
            target_id = action.get("target_id")
            if object_id == self._state.get("held") and self._is_matching_zone(object_id, target_id):
                objects[object_id]["placed"] = True
                objects[object_id]["position"] = objects[target_id]["position"]
                self._state["held"] = None
                self._stage_progress["place_correct_region"] = True
                if self._all_movable_objects_placed():
                    self._stage_progress["repeat_sorting"] = True
        elif skill == "finish" and self._all_movable_objects_placed():
            self._stage_progress["finalize"] = True
            self._done = True
            self._success = True
            self._termination_reason = TerminationReason.SUCCESS

    def _apply_kitchen(self, skill: str | None, action: JsonObject) -> None:
        objects = self._state["objects"]
        if skill == "select":
            object_id = action.get("object_id")
            if (
                object_id in objects
                and objects[object_id]["kind"] == "movable"
                and not objects[object_id]["placed"]
            ):
                self._state["selected"] = object_id
                self._stage_progress["select_item"] = True
        elif skill == "grasp":
            object_id = action.get("object_id")
            if object_id == self._state.get("selected"):
                self._state["held"] = object_id
                self._stage_progress["grasp_item"] = True
                self._stage_progress["transport_item"] = True
        elif skill == "place":
            object_id = action.get("object_id")
            target_id = action.get("target_id")
            if object_id == self._state.get("held") and self._is_matching_zone(object_id, target_id):
                objects[object_id]["placed"] = True
                objects[object_id]["location"] = target_id
                objects[object_id]["position"] = objects[target_id]["position"]
                self._state["held"] = None
                self._stage_progress["place_matching_zone"] = True
                if self._all_movable_objects_placed():
                    self._stage_progress["repeat_counter_reset"] = True
        elif skill == "finish" and self._all_movable_objects_placed():
            self._stage_progress["finalize"] = True
            self._done = True
            self._success = True
            self._termination_reason = TerminationReason.SUCCESS

    def _apply_drawer(self, skill: str | None, action: JsonObject) -> None:
        drawer = self._state["drawer"]
        objects = self._state["objects"]
        if skill == "locate_handle" and action.get("handle_id") == drawer.get("handle_id"):
            self._stage_progress["locate_drawer_handle"] = True
        elif (
            skill == "open_drawer"
            and action.get("drawer_id") == drawer.get("id")
            and self._stage_progress["locate_drawer_handle"]
        ):
            drawer["state"] = "open"
            self._stage_progress["open_drawer"] = True
        elif skill == "select":
            object_id = action.get("object_id")
            if (
                drawer.get("state") == "open"
                and object_id in objects
                and objects[object_id].get("kind") == "movable"
                and objects[object_id].get("location") == drawer.get("id")
            ):
                self._state["selected"] = object_id
                self._stage_progress["select_target_object"] = True
        elif skill == "grasp":
            object_id = action.get("object_id")
            if (
                object_id == self._state.get("selected")
                and drawer.get("state") == "open"
                and objects[object_id].get("location") == drawer.get("id")
            ):
                self._state["held"] = object_id
                objects[object_id]["location"] = "gripper"
                self._stage_progress["grasp_from_drawer"] = True
        elif skill == "place":
            object_id = action.get("object_id")
            target_id = action.get("target_id")
            if object_id == self._state.get("held") and self._is_matching_zone(object_id, target_id):
                objects[object_id]["placed"] = True
                objects[object_id]["location"] = target_id
                objects[object_id]["position"] = objects[target_id]["position"]
                self._state["held"] = None
                self._stage_progress["place_on_target"] = True
        elif skill == "finish" and self._stage_progress["place_on_target"]:
            self._stage_progress["finalize"] = True
            self._done = True
            self._success = True
            self._termination_reason = TerminationReason.SUCCESS

    def _apply_towel(self, skill: str | None, action: JsonObject) -> None:
        towel = self._state["towel"]
        if skill == "select_corners" and action.get("corners") == ["top_left", "top_right"]:
            towel["selected_corners"] = action["corners"]
            self._stage_progress["select_corners"] = True
        elif skill == "fold" and action.get("fold_index") == 1 and self._stage_progress["select_corners"]:
            towel["folds_completed"] = max(towel["folds_completed"], 1)
            self._stage_progress["first_fold"] = True
        elif skill == "align" and action.get("fold_index") == 1 and self._stage_progress["first_fold"]:
            towel["first_alignment"] = 0.95
            self._stage_progress["first_alignment"] = True
        elif skill == "fold" and action.get("fold_index") == 2 and self._stage_progress["first_alignment"]:
            towel["folds_completed"] = max(towel["folds_completed"], 2)
            self._stage_progress["second_fold"] = True
        elif skill == "align" and action.get("fold_index") == 2 and self._stage_progress["second_fold"]:
            towel["final_alignment"] = 0.92
            self._stage_progress["final_alignment"] = True
            self._done = True
            self._success = True
            self._termination_reason = TerminationReason.SUCCESS

    def _is_matching_zone(self, object_id: str, target_id: str | None) -> bool:
        objects = self._state["objects"]
        if object_id not in objects or target_id not in objects:
            return False
        target = objects[target_id]
        if target.get("kind") != "target":
            return False
        source = objects[object_id]
        if source.get("color") is not None and source.get("color") == target.get("color"):
            return True
        if source.get("category") is not None and source.get("category") == target.get("category"):
            return True
        accepted_categories = target.get("accepts_categories", [])
        return source.get("category") in accepted_categories

    def _all_movable_objects_placed(self) -> bool:
        return all(
            item.get("placed", True)
            for item in self._state["objects"].values()
            if item.get("kind") == "movable"
        )


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
