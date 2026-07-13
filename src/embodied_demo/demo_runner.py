"""Backward-compatible entry point for deterministic mock demo rollouts.

The implementation now lives in smaller modules:

- ``embodied_demo.policies.scripted`` for scripted policy plans;
- ``embodied_demo.environments.mock`` for symbolic/kinematic mock state;
- ``embodied_demo.rollout.mock_runner`` for rollout, artifacts, and reports.

Keep this module as a stable import path for the CLI and existing tests.
"""

from embodied_demo.environments import MockEnvironment, StepResult
from embodied_demo.policies import ScriptedPolicy
from embodied_demo.rollout import run_mock_demo

__all__ = ["MockEnvironment", "ScriptedPolicy", "StepResult", "run_mock_demo"]
