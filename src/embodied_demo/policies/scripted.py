from __future__ import annotations

from typing import Any

from embodied_demo.errors import ConfigurationError
from embodied_demo.schemas import ActionChunk, Observation
from embodied_demo.schemas.enums import CoordinateFrame
from embodied_demo.schemas.run import ResolvedRun

JsonObject = dict[str, Any]


class ScriptedPolicy:
    """A tiny XPolicyLab-shaped policy adapter for deterministic mock demos.

    This policy is deliberately not a learned model. Its job is to exercise the
    same lifecycle a real policy adapter must satisfy:

    reset -> update_observation -> get_action
    """

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
