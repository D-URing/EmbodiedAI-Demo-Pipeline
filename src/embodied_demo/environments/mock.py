from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from embodied_demo.errors import ConfigurationError
from embodied_demo.schemas import ActionChunk, Observation
from embodied_demo.schemas.enums import FailureType, TerminationReason
from embodied_demo.schemas.run import ResolvedRun

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    action: ActionChunk
    stage_progress: dict[str, bool]
    done: bool
    success: bool
    failure_type: FailureType | None = None
    termination_reason: TerminationReason | None = None


class MockEnvironment:
    """Deterministic symbolic/kinematic backend for R1 household demos."""

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
