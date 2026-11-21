"""Check anchor identity and the Jacobians used by contact constraints."""

import json

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from holomotion.scripts.training.prepare_r1_fixed_contacts import (
    ContactEqualities,
    GroundedContactEqualities,
    contact_schedule,
)


def write_contacts(path, *, bad_normal=False, outside_face=False):
    active = np.array(
        [[False, False], [True, True], [True, True], [False, False]]
    )
    points = np.tile(
        [[0.125, -0.0178, 0.0594], [-0.125, 0.058, 0.0533]], (4, 1, 1)
    )
    normals = np.tile([[1.0, 0, 0], [-1.0, 0, 0]], (4, 1, 1))
    if bad_normal:
        normals[1, 0] = [0, 1, 0]
    if outside_face:
        points[1, 0, 1] = 0.15
    np.savez_compressed(
        path,
        active=active,
        object_points_local=points,
        object_normals_local=normals,
        semantic_parts=np.array(["left_hand", "right_hand"]),
        robot_points_local=np.zeros((2, 3)),
        robot_normals_local=np.array([[0, -1.0, 0], [0, 1.0, 0]]),
        metadata_json=json.dumps(
            dict(
                schema="holosoma.retarget_surface_contacts.v2",
                anchor_positions_fixed=True,
                preload_mm=10,
                fps=30,
            )
        ),
    )


def test_width_adaptation_preserves_tangential_anchors_and_contact_timing(
    tmp_path,
):
    path = tmp_path / "contacts.npz"
    write_contacts(path)
    data, _ = contact_schedule(
        path, 6, [0.27, 0.305, 0.245], [0.24, 0.245, 0.27]
    )
    np.testing.assert_array_equal(data["source_frame"], [0, 0, 1, 1, 2, 3])
    np.testing.assert_array_equal(
        data["active"][:, 0], [False, False, True, True, True, False]
    )
    np.testing.assert_allclose(
        data["object_points_local"][2],
        [[0.12, -0.0178, 0.0594], [-0.12, 0.058, 0.0533]],
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        data["object_points_local"][2:5, :, 1:],
        np.tile([[[-0.0178, 0.0594], [0.058, 0.0533]]], (3, 1, 1)),
    )


@pytest.mark.parametrize(
    "bad_normal,outside_face", [(True, False), (False, True)]
)
def test_incompatible_surface_anchors_are_rejected(
    tmp_path, bad_normal, outside_face
):
    path = tmp_path / "contacts.npz"
    write_contacts(path, bad_normal=bad_normal, outside_face=outside_face)
    with pytest.raises(ValueError):
        contact_schedule(path, 6, [0.27, 0.305, 0.245], [0.24, 0.245, 0.27])


@pytest.mark.parametrize("grounded", [False, True])
def test_contact_jacobians_match_independent_finite_differences(grounded):
    wrists = "".join(
        f'<body name="{side}_wrist_roll_link" pos=".2 {y} .3">'
        '<joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/>'
        '<geom type="sphere" size=".01" mass="1"/></body>'
        for side, y in (("left", 0.15), ("right", -0.15))
    )
    feet = "".join(
        f'<body name="{side}_ankle_roll_link" pos="0 {y} -.6">'
        '<joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/>'
        '<geom type="sphere" size=".01" mass="1"/></body>'
        for side, y in (("left", 0.1), ("right", -0.1))
    )
    dummy = feet + "".join(
        '<body><joint/><geom type="sphere" size=".01" mass="1"/></body>'
        for _ in range(14)
    )
    xml = (
        "<mujoco><worldbody><body><freejoint/>"
        '<geom type="sphere" size=".1" mass="1"/>'
        + wrists
        + dummy
        + "</body><body><freejoint/>"
        '<geom type="box" size=".12 .1225 .135" mass=".1"/>'
        "</body></worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    contacts = dict(
        active=np.ones((1, 2), dtype=bool),
        object_points_local=np.array(
            [[[0.12, -0.02, 0.06], [-0.12, 0.058, 0.053]]]
        ),
        object_normals_local=np.array([[[1.0, 0, 0], [-1.0, 0, 0]]]),
        robot_points_local=np.array([[0.11, -0.04, 0.02], [0.11, 0.04, 0.02]]),
        robot_normals_local=np.array([[0, -1.0, 0], [0, 1.0, 0]]),
    )
    problem = (
        GroundedContactEqualities(model, contacts)
        if grounded
        else ContactEqualities(model, contacts, np.arange(6))
    )
    row = np.zeros(40)
    row[:3] = [0.1, -0.2, 0.7]
    row[3:7] = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_quat()[
        [3, 0, 1, 2]
    ]
    row[33:36] = [0.4, 0, 0.2]
    row[36:40] = Rotation.from_euler("xyz", [0.2, 0.1, -0.4]).as_quat()[
        [3, 0, 1, 2]
    ]
    problem.configure(row, 0)
    q = row[problem.q_indices] + 0.15 * np.cos(
        np.arange(len(problem.q_indices))
    )
    step = np.eye(len(q)) * 1e-6
    for value, derivative in (
        (problem.value, problem.jacobian),
        (problem.hemisphere, problem.hemisphere_jacobian),
    ):
        numerical = np.stack(
            [(value(q + s) - value(q - s)) / 2e-6 for s in step], axis=1
        )
        np.testing.assert_allclose(
            derivative(q), numerical, atol=2e-7, rtol=1e-6
        )
