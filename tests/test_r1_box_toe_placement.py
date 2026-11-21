"""Check toe placement independently of the planner's support witnesses."""

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from holomotion.scripts.training.prepare_r1_stance_grip import (
    place_stance_at_box,
)


@pytest.mark.parametrize("box_yaw", [-90, 90])
def test_actual_mesh_box_distance_matches_requested_toe_clearance(box_yaw):
    # A floating robot with 26 joints and known convex foot meshes. The
    # independent native distance query uses the physical 30.5 cm box.
    vertices = " ".join(
        f"{x} {y} {z}"
        for x in (-0.06, 0.16)
        for y in (-0.03, 0.03)
        for z in (0, 0.04)
    )
    auxiliary = "".join(
        f'<body pos="0 0 .1"><joint name="aux{i}"/>'
        '<geom type="sphere" size=".01" mass=".01" '
        'contype="0" conaffinity="0"/></body>'
        for i in range(26)
    )
    xml = (
        f'<mujoco><asset><mesh name="foot" vertex="{vertices}"/></asset>'
        "<worldbody><body><freejoint/>"
        '<geom type="sphere" size=".1" mass="1" '
        'contype="0" conaffinity="0"/>'
        '<body name="left_ankle_roll_link" pos="0 .11 -.7">'
        '<geom type="mesh" mesh="foot"/></body>'
        '<body name="right_ankle_roll_link" pos="0 -.11 -.7">'
        '<geom type="mesh" mesh="foot"/></body>'
        f'{auxiliary}</body><body name="grip_box"><freejoint/>'
        '<geom name="grip_box_surface" type="box" '
        'size=".12 .1225 .135" mass=".1"/></body></worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    stance = np.zeros(40)
    stance[3] = 1
    stance[33:36] = [0.4, 0.02, 0.136]
    stance[36:40] = Rotation.from_euler("z", box_yaw, degrees=True).as_quat()[
        [3, 0, 1, 2]
    ]
    report = place_stance_at_box(
        model, stance, [0, 0, 0], [0.305, 0.245, 0.27], 0.005
    )
    box = model.geom("grip_box_surface").id
    model.geom_size[box] = [0.1525, 0.1225, 0.135]
    data = mujoco.MjData(model)
    data.qpos[:] = stance
    mujoco.mj_fwdPosition(model, data)
    for side in ("left", "right"):
        body = model.body(side + "_ankle_roll_link").id
        geom = next(
            g for g in range(model.ngeom) if model.geom_bodyid[g] == body
        )
        segment = np.zeros(6)
        actual = mujoco.mj_geomDistance(model, data, geom, box, 0.05, segment)
        assert actual == pytest.approx(0.005, abs=1e-6)
        assert np.linalg.norm(segment[3:] - segment[:3]) == pytest.approx(
            actual
        )
        assert report["evaluation_foot_box_gaps_m"][side] == pytest.approx(
            actual
        )
