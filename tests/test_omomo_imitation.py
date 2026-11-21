"""Validate shared frames, R1 dimensions and deployment observations."""

import h5py
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from holomotion.scripts.training.prepare_supermimic_omomo import prepare
from holomotion.src.algo.supermimic.buffer import ReplayBuffer
from holomotion.src.algo.supermimic.networks import Policy, WorldModel
from holomotion.src.deployment.supermimic_r1 import observation
from holomotion.src.env.reference_tracking.box import (
    LiftMeter,
    bottom_clearance,
)
from holomotion.src.env.reference_tracking.mdp import (
    MDP,
    flatten_state,
    normalize_xi,
    unflatten_state,
)
from holomotion.src.env.reference_tracking.robot_contract import robot_contract
from holomotion.src.env.reference_tracking.rotations import quat_to_mat


def test_shared_robot_object_frame_and_named_joint_reordering(tmp_path):
    contract = robot_contract("r1")
    rotation = Rotation.from_euler("xyz", [0.1, 0.2, 1.3])
    root = np.tile([2.0, 3.0, 0.76], (50, 1))
    obj = root + rotation.apply([0.4, 0.0, -0.6])
    quat = rotation.as_quat()[[3, 0, 1, 2]]
    joints = np.tile(np.arange(26) * 0.01, (50, 1))
    source = tmp_path / "retarget.npz"
    np.savez(
        source,
        fps=[50],
        joint_pos=np.concatenate(
            (root, np.tile(quat, (50, 1)), joints[:, ::-1]), axis=1
        ),
        joint_names=contract.joints[::-1],
        object_pos_w=obj,
        object_quat_w=np.tile(quat, (50, 1)),
    )
    output = tmp_path / "converted"
    prepare(source, output, "r1", [0.305, 0.245, 0.27])
    with h5py.File(output / "motions.h5") as stream:
        np.testing.assert_allclose(stream["ref_dof_pos"][:], joints, atol=1e-7)
        r = Rotation.from_quat(stream["ref_root_rot"][:])
        relative = r.inv().apply(
            stream["object_pos_w"][:] - stream["ref_root_pos"][:]
        )
        np.testing.assert_allclose(
            relative, np.tile([0.4, 0.0, -0.6], (50, 1)), atol=2e-7
        )


@pytest.mark.parametrize("reference_orientation", [False, True])
def test_r1_deployment_observation_matches_training(reference_orientation):
    contract = robot_contract("r1")
    quaternion = torch.tensor([[0.91, 0.1, 0.2, 0.3]])
    quaternion /= quaternion.norm(dim=-1, keepdim=True)
    rotation = quat_to_mat(quaternion)
    body_gyro = torch.tensor([[0.2, -0.3, 0.7]])
    world_gyro = (rotation @ body_gyro[..., None]).squeeze(-1)
    q = torch.tensor([contract.default]) + 0.03
    qd = torch.arange(26)[None].float() * 0.01
    state = (torch.zeros(1, 3), rotation, torch.zeros(1, 3), world_gyro, q, qd)

    class Kin:
        q_lo = torch.full((26,), -3.0)
        q_hi = torch.full((26,), 3.0)

    class Motion:
        root_R = torch.eye(3)[None]  # noqa: N815

    mdp = MDP(
        Kin(),
        Motion(),
        torch.tensor(contract.default),
        reference_orientation=reference_orientation,
    )
    previous = torch.ones(1, 26) * 0.2
    expected = mdp.obs(state, previous, torch.zeros(1, 30), torch.tensor([0]))
    config = dict(
        default_joint_pos=contract.default,
        reference_orientation=reference_orientation,
        obs_dim=mdp.obs_dim,
    )
    actual = observation(
        config,
        quaternion[0].numpy(),
        q[0].numpy(),
        qd[0].numpy(),
        body_gyro[0].numpy(),
        previous[0].numpy(),
        np.eye(3),
    )
    np.testing.assert_allclose(actual, expected[0].numpy(), atol=2e-7)
    rebuilt = unflatten_state(flatten_state(state))
    for before, after in zip(state, rebuilt, strict=True):
        torch.testing.assert_close(before, after)
    policy = Policy(mdp.obs_dim, 67 * 2, hist=2, num_j=26)
    action = policy(expected[:, None].repeat(1, 2, 1), torch.zeros(1, 134))
    assert action.shape == (1, 26)
    world = WorldModel(hidden=(32, 32), num_j=26)
    prediction = world.step(state, action, normalize_xi(torch.zeros(1, 30)))
    assert prediction[-1].shape == (1, 26)
    sum(v.square().mean() for v in prediction).backward()
    assert world.net[-1].weight.grad.isfinite().all()
    assert ReplayBuffer(4, 1, "cpu", num_j=26).data["s"].shape == (4, 1, 65)


def test_lift_measures_rotated_box_bottom_and_sustained_clearance():
    position = torch.tensor([[0.0, 0.0, 0.3]])
    quat = torch.tensor([[2**-0.5, 2**-0.5, 0.0, 0.0]])
    # A 90 degree roll exchanges the box's Y and Z extents.
    clearance = bottom_clearance(position, quat, [0.305, 0.245, 0.27])
    torch.testing.assert_close(clearance, torch.tensor([0.3 - 0.245 / 2]))
    meter = LiftMeter(1, "cpu")
    for _ in range(26):
        meter.update(clearance, torch.tensor([True]))
    assert meter.report(torch.tensor([False]))[0]["lifted"]
    assert not meter.report(torch.tensor([True]))[0]["lifted"]


def test_failed_simulation_blocks_activation_before_sdk_import(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from holomotion.src.deployment import supermimic_r1

    monkeypatch.setattr(
        supermimic_r1,
        "OfflineActor",
        lambda _: SimpleNamespace(config={"simulation_passed": False}),
    )
    with pytest.raises(RuntimeError, match="has not passed"):
        supermimic_r1.run(
            tmp_path, tmp_path / "absent-sdk", "absent-interface"
        )


def test_object_preflight_rejects_penetrating_box_and_preserves_mesh_origin(
    tmp_path,
):
    import json
    from types import SimpleNamespace

    from holomotion.src.env.reference_tracking.box import load_box_reference

    with h5py.File(tmp_path / "motions.h5", "w") as stream:
        stream["object_pos_w"] = np.array([[0.0, 0.0, 0.12]], dtype=np.float32)
        stream["object_quat_w"] = np.array(
            [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32
        )
    metadata = dict(
        hdf5_shards=[dict(file="motions.h5")],
        clips={"clip": dict(shard=0, start=0, length=1)},
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(metadata))
    motion = SimpleNamespace(
        box=dict(size_m=[0.4, 0.4, 0.4]), clip_start=torch.tensor([0])
    )
    with pytest.raises(ValueError, match="penetrates the floor"):
        load_box_reference(path, motion, "cpu")
    (tmp_path / "box.obj").write_text(
        "v 0 0 -.1\nv .1 0 .1\nv 0 .1 .1\nf 1 2 3\n"
    )
    motion.box["mesh_file"] = "box.obj"
    load_box_reference(path, motion, "cpu")
    clearance = bottom_clearance(
        motion.box_p,
        motion.box_quat,
        motion.box["size_m"],
        vertices=motion.box_vertices,
    )
    torch.testing.assert_close(clearance, torch.tensor([0.02]))
