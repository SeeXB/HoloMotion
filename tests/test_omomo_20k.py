"""The long imitation run must finish 20,000 real policy optimizer events."""

from types import SimpleNamespace

import pytest

from holomotion.src.algo.supermimic.config import SuperMimicConfig
from holomotion.src.algo.supermimic.schedule import (
    ScheduleConfig,
    UpdateSchedule,
)
from holomotion.src.algo.supermimic.training import training_budget_active


def test_policy_target_counts_optimizer_events_after_world_warmup():
    schedule = UpdateSchedule(
        ScheduleConfig(
            physical_envs=128,
            actor_batch_size=512,
            imagination_horizon=24,
            world_batch_size=256,
            world_window_length=8,
            world_batches_per_update=10,
            warmup_vector_steps=64,
            world_only_updates=100,
        )
    )
    options = SimpleNamespace(
        max_vector_steps=800064,
        max_updates=320000,
        max_actor_updates=20000,
        max_hours=24.0,
    )
    while training_budget_active(options, schedule, 600.0):
        world, actor = schedule.advance()
        for _ in range(world):
            schedule.complete_world()
        for _ in range(actor):
            schedule.complete_actor()
    assert schedule.actor_updates == 20000
    assert schedule.world_updates == 20100
    assert schedule.vector_steps == 50314


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_policy_target_must_be_positive_integer(value, tmp_path):
    with pytest.raises(ValueError, match="max_actor_updates"):
        SuperMimicConfig(
            manifest=tmp_path / "missing",
            robot_urdf=tmp_path / "missing",
            output=tmp_path / "new",
            max_actor_updates=value,
        ).validate()


def test_job_launcher_rejects_shortened_policy_budget(tmp_path):
    from holomotion.scripts.training.run_omomo_20k import command_plan

    with pytest.raises(ValueError, match="at least 20,000"):
        command_plan(tmp_path, ["r1"], 19999, tmp_path)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_export_command_uses_the_precision_specific_input(precision, tmp_path):
    from holomotion.scripts.training.run_omomo_20k import actor_export_command

    training, bundle = tmp_path / "training", tmp_path / "bundle"
    command = actor_export_command(training, bundle, precision)
    if precision == "bf16":
        assert command[2].endswith("export_supermimic_bf16")
        assert command[3:5] == ["--training", str(training)]
    else:
        assert command[2].endswith("export_supermimic")
        assert command[3:5] == [
            "--checkpoint",
            str(training / "checkpoint_final.pt"),
        ]
    assert command[-2:] == ["--output", str(bundle)]


def test_one_failed_job_does_not_prevent_the_other_robot(
    monkeypatch, tmp_path
):
    import json
    import subprocess

    from holomotion.scripts.training.run_omomo_20k import execute

    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    root = tmp_path / "queue"
    plan = dict(
        target_policy_updates=20000,
        robots=["r1", "g1"],
        jobs=[
            dict(
                robot=name,
                output=str(root / name),
                preflight=[name],
                training=[],
            )
            for name in ["r1", "g1"]
        ],
    )
    with pytest.raises(RuntimeError, match="Jobs failed"):
        execute(plan, root)
    assert calls == [["r1"], ["g1"]]
    result = json.loads((root / "status.json").read_text())
    assert result["failed_jobs"] == ["r1", "g1"]
    assert result["status"] == "failed"


def test_isaac_preflight_report_has_a_process_duration(monkeypatch, tmp_path):
    from holomotion.src.algo.supermimic import runtime

    monkeypatch.setattr(
        runtime, "_supervise", lambda options: {"elapsed_s": 1.25}
    )
    options = SimpleNamespace(
        output=tmp_path, skip_evaluation=False, preflight_only=True
    )
    result = runtime.execute_isaac(options)
    assert result["training_process_wall_s"] == 1.25
    assert result["process_wall_s"] >= 0.0
