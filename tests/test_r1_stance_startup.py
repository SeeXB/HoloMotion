"""Verify frame-zero feedback and the grounded-Stance activation boundary."""

from types import SimpleNamespace

import numpy as np
import pytest

from holomotion.src.deployment import supermimic_r1


def test_missing_stance_provenance_blocks_before_hardware_import(monkeypatch):
    monkeypatch.setattr(
        supermimic_r1,
        "OfflineActor",
        lambda _: SimpleNamespace(
            config=dict(
                simulation_passed=True,
                exported_simulation_passed=True,
            )
        ),
    )
    with pytest.raises(RuntimeError, match="grounded official Stance"):
        supermimic_r1.run(None, None, "not-a-hardware-interface")


def test_actor_can_refresh_feedback_without_advancing_reference():
    actor = object.__new__(supermimic_r1.OfflineActor)
    actor.config = dict(
        obs_dim=84,
        reference_orientation=False,
        default_joint_pos=[0.0] * 26,
        action_scale=[0.1] * 26,
        joint_lower=[-3.0] * 26,
        joint_upper=[3.0] * 26,
        history=2,
    )
    actor.goals = np.zeros((3, 2), dtype=np.float32)
    actor.root_rotation = np.repeat(np.eye(3)[None], 3, axis=0)
    actor.frame = 0
    actor.previous = np.zeros(26, dtype=np.float32)
    actor.history = None
    calls = []

    def inference(_, inputs):
        calls.append(inputs["obs_hist"].copy())
        return [np.zeros((1, 26), dtype=np.float32)]

    actor.session = SimpleNamespace(run=inference)
    state = np.zeros(65)
    state[3] = 1
    actor.step(state, advance=False)
    state[7] = 0.2
    actor.step(state, advance=False)
    assert actor.frame == 0
    assert len(calls) == 2
    assert calls[-1][0, -1, 0] == pytest.approx(0.2)
    actor.step(state)
    assert actor.frame == 1


def test_stand_and_terminal_hold_keep_refreshing_live_policy(monkeypatch):
    monkeypatch.setattr(supermimic_r1.time, "sleep", lambda _: None)
    events = iter([0, 1, 0, 0, 0, 0, 2])
    calls, commands = [], []

    class Actor:
        frame = 0
        goals = [None] * 3

        def step(self, state, *, advance):
            calls.append((self.frame, advance))
            if advance:
                self.frame = min(self.frame + 1, 2)
            return np.ones(26) * len(calls) / 100

    hardware = SimpleNamespace(
        get_joystick_msg=lambda: SimpleNamespace(keys=next(events)),
        get_joystick_key=lambda msg: {1: "A", 2: "B"}.get(msg.keys),
        get_low_state=lambda: np.zeros(65),
        send_low_command=lambda q, qd, tau: commands.append(q.copy()),
    )
    actor = Actor()
    supermimic_r1.stream_policy(hardware, actor, np.zeros(26), np.zeros(26))
    assert calls[:2] == [(0, False), (0, False)]
    assert calls[2:4] == [(0, True), (1, True)]
    assert calls[4:] == [(2, False), (2, False)]
    assert len(commands) == 6
    assert actor.frame == 2
