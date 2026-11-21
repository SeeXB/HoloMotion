"""Keep narrow training and real deployment geometry on explicit robot axes."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holomotion.src.env.reference_tracking.box import bottom_clearance
from holomotion.src.env.reference_tracking.r1_box_contract import (
    R1_BOX_AXES,
    R1_REAL_BOX_SIZE_M,
    R1_TRAIN_BOX_SIZE_M,
    use_evaluation_box,
)
from holomotion.src.env.reference_tracking.robot_contract import robot_contract


def test_user_confirmed_profiles_change_only_left_right_dimension():
    assert R1_BOX_AXES == ("left_right", "front_back", "up")
    assert R1_TRAIN_BOX_SIZE_M == (0.2445, 0.245, 0.270)
    assert R1_REAL_BOX_SIZE_M == (0.305, 0.245, 0.270)
    assert R1_TRAIN_BOX_SIZE_M[1:] == R1_REAL_BOX_SIZE_M[1:]
    assert (
        R1_REAL_BOX_SIZE_M[0] - R1_TRAIN_BOX_SIZE_M[0]
    ) / 2 == pytest.approx(0.03025)
    assert {"head_pitch_link", "head_yaw_link"}.issubset(
        robot_contract("r1").tracked
    )


@pytest.mark.parametrize("quat", [[1.0, 0, 0, 0], [0.9, 0.1, -0.2, 0.3]])
def test_real_box_switch_preserves_inputs_and_initial_floor_clearance(quat):
    q = torch.tensor([quat, quat], dtype=torch.float64)
    q /= q.norm(dim=-1, keepdim=True)
    p = torch.tensor([[0.3, 0.2, 0.136], [0.8, 0.1, 0.7]], dtype=torch.float64)
    original_p = p.clone()
    box = dict(size_m=list(R1_TRAIN_BOX_SIZE_M), mass_kg=0.1, friction=0.8)
    real = dict(size_m=list(R1_REAL_BOX_SIZE_M), mass_kg=0.1)
    motion = SimpleNamespace(
        box=box,
        evaluation_box=real,
        box_vertices=None,
        box_p=p,
        box_quat=q,
        clip_start=torch.tensor([0]),
    )
    before = bottom_clearance(p[:1], q[:1], R1_TRAIN_BOX_SIZE_M)
    report = use_evaluation_box(motion)
    after = bottom_clearance(motion.box_p[:1], q[:1], R1_REAL_BOX_SIZE_M)
    torch.testing.assert_close(before, after)
    torch.testing.assert_close(p, original_p, rtol=0, atol=0)
    torch.testing.assert_close(
        motion.box_p[:, :2], original_p[:, :2], rtol=0, atol=0
    )
    torch.testing.assert_close(
        motion.box_p[1:], original_p[1:], rtol=0, atol=0
    )
    assert motion.box_quat is q
    assert box["size_m"] == list(R1_TRAIN_BOX_SIZE_M)
    assert report["evaluation_box"]["size_m"] == list(R1_REAL_BOX_SIZE_M)
    assert motion.box["mass_kg"] == 0.1
    if quat == [1.0, 0, 0, 0]:
        assert report["initial_center_z_shift_m"][0] == 0


def test_undeclared_evaluation_profile_keeps_existing_tasks_unchanged():
    motion = SimpleNamespace(box=None)
    assert use_evaluation_box(motion) is None


def test_physical_rollout_renderer_uses_the_evaluated_box(tmp_path):
    from holomotion.scripts.evaluation.render_omomo_rollout import (
        load_render_model,
    )

    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="visual-test"/>')
    (tmp_path / "g1.xml").write_text(
        '<mujoco><worldbody><body name="pelvis_link"><freejoint/>'
        '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    train_box = dict(size_m=list(R1_TRAIN_BOX_SIZE_M), mass_kg=0.1)
    real_box = dict(size_m=list(R1_REAL_BOX_SIZE_M), mass_kg=0.1)
    (tmp_path / "specification.json").write_text(
        json.dumps(
            dict(
                box=train_box,
                arguments=dict(robot_urdf=str(urdf)),
            )
        )
    )
    (tmp_path / "evaluation_final.json").write_text(
        json.dumps(
            dict(
                box_profile=dict(
                    training_box=train_box, evaluation_box=real_box
                ),
            )
        )
    )
    model, _ = load_render_model(tmp_path, tmp_path / "physical.mp4")
    body = model.body("evaluation_box").id
    geom = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == body)
    np.testing.assert_allclose(model.geom_size[geom] * 2, R1_REAL_BOX_SIZE_M)
