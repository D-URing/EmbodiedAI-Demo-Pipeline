from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from embodied_demo.schemas import ActionChunk, EpisodeResult, Observation, TrainingEvidence


def test_observation_contract() -> None:
    observation = Observation(
        episode_id="episode-001",
        step_id=0,
        timestamp=datetime.now(UTC),
        instruction="将物品归位",
        state={"gripper_open": True},
    )
    assert observation.schema_version == "1.0"


def test_action_horizon_must_match_payload() -> None:
    with pytest.raises(ValidationError, match="horizon"):
        ActionChunk(
            representation="semantic",
            frame="world",
            control_frequency_hz=2,
            horizon=2,
            actions=[{"skill": "pick", "object_id": "red_block"}],
        )


def test_successful_episode_contract() -> None:
    result = EpisodeResult(
        run_id="run-001",
        episode_id="episode-001",
        task_id="offline_replay_v1",
        task_version="1.0.0",
        backend="dataset_replay",
        profile="smoke",
        seed=0,
        layout_id="layout_a",
        valid=True,
        episode_success=True,
        progress_score=100,
        completed_stages=6,
        total_stages=6,
        termination_reason="success",
        episode_steps=8,
        wall_time_s=0.1,
    )
    assert result.failure_type is None


def test_success_reason_is_consistent() -> None:
    with pytest.raises(ValidationError, match="successful episodes"):
        EpisodeResult(
            run_id="run-001",
            episode_id="episode-001",
            task_id="offline_replay_v1",
            task_version="1.0.0",
            backend="dataset_replay",
            profile="smoke",
            seed=0,
            layout_id="layout_a",
            valid=True,
            episode_success=True,
            progress_score=90,
            completed_stages=5,
            total_stages=6,
            termination_reason="task_failure",
            episode_steps=8,
            wall_time_s=0.1,
        )


def test_training_evidence_contract() -> None:
    evidence = TrainingEvidence(
        backend="fastwam-realrobot",
        run_id="20260713-200000",
        source_run_dir="runs/experiments/custom/fastwam_realrobot_smoke/20260713-200000",
        parsed_train_count=4,
        initial_loss=1.4862,
        final_loss=0.701,
        loss_drop_ratio=0.5283,
        loss_decreased=True,
        final_step=200,
        max_steps=200,
        training_completed=True,
        validation_status="passed",
    )
    assert evidence.schema_version == "1.0"
