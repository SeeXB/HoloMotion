"""Check source timing, quaternion interpolation and named-array integrity."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from holomotion.scripts.training.prepare_supermimic_r1_omni import (
    resample_reference,
)
from holomotion.src.env.reference_tracking.robot_contract import robot_contract


def source_arrays():
    times = np.arange(31) / 30
    qpos = np.zeros((31, 40))
    qpos[:, :3] = times[:, None] * [1, 2, 3]
    qpos[:, 3:7] = Rotation.from_euler("z", times).as_quat()[:, [3, 0, 1, 2]]
    qpos[:, 7:33] = times[:, None] * np.arange(26) / 26
    qpos[:, 33:36] = qpos[:, :3] + [0.4, 0.2, -0.1]
    qpos[:, 36:40] = qpos[:, 3:7]
    return dict(
        fps=np.array(30),
        time_s=times,
        joint_names=np.array(robot_contract("r1").joints),
        qpos=qpos,
        base_pos=qpos[:, :3].copy(),
        base_quat_wxyz=qpos[:, 3:7].copy(),
        joint_pos=qpos[:, 7:33].copy(),
        object_pos=qpos[:, 33:36].copy(),
        object_quat_wxyz=qpos[:, 36:40].copy(),
    )


def test_resampling_preserves_duration_joint_values_and_shared_frame():
    data = source_arrays()
    result = resample_reference(data)
    q = result["joint_pos"]
    times = np.arange(51) / 50
    assert q.shape == (51, 33)
    np.testing.assert_allclose(
        q[:, :3], times[:, None] * [1, 2, 3], atol=1e-12
    )
    np.testing.assert_allclose(q[-1], data["qpos"][-1, :33], atol=1e-12)
    np.testing.assert_allclose(
        q[:, 7:], times[:, None] * np.arange(26) / 26, atol=1e-12
    )
    rotations = Rotation.from_quat(q[:, [4, 5, 6, 3]])
    np.testing.assert_allclose(rotations.as_rotvec()[:, 2], times, atol=1e-12)
    np.testing.assert_allclose(result["object_quat_w"], q[:, 3:7], atol=1e-12)
    np.testing.assert_allclose(
        result["object_pos_w"] - q[:, :3],
        np.tile([0.4, 0.2, -0.1], (51, 1)),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "field", ["joint_pos", "base_quat_wxyz", "object_pos"]
)
def test_inconsistent_named_export_is_rejected(field):
    data = source_arrays()
    data[field][2, 0] += 0.1
    with pytest.raises(ValueError, match="disagrees with qpos"):
        resample_reference(data)


def test_joint_permutation_is_rejected():
    data = source_arrays()
    data["joint_names"] = data["joint_names"][::-1]
    with pytest.raises(ValueError, match="joint order"):
        resample_reference(data)
