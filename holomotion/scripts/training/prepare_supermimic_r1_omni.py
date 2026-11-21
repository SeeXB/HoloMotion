"""Validate and resample a named R1 OmniRetarget package."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from holomotion.scripts.training.prepare_supermimic_omomo import prepare
from holomotion.src.env.reference_tracking.model import build_model
from holomotion.src.env.reference_tracking.robot_contract import robot_contract
from holomotion.src.utils.artifacts import atomic_json


def resample_reference(arrays, target_fps=50):
    """Interpolate positions linearly and rotations on SO(3); keep timing."""
    contract = robot_contract("r1")
    names = arrays["joint_names"].tolist()
    if names != contract.joints:
        raise ValueError("Named R1 joint order differs from the SDK contract")
    fps = float(np.asarray(arrays["fps"]).item())
    q = arrays["qpos"]
    if not np.isfinite(fps) or fps <= 0 or q.ndim != 2 or q.shape[1] != 40:
        raise ValueError(
            "Expected R1 floating base, 26 joints and object qpos"
        )
    if len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("Incomplete or nonfinite reference")
    for name, section in [
        ("base_pos", slice(0, 3)),
        ("base_quat_wxyz", slice(3, 7)),
        ("joint_pos", slice(7, 33)),
        ("object_pos", slice(33, 36)),
        ("object_quat_wxyz", slice(36, 40)),
    ]:
        if arrays[name].shape != q[:, section].shape or not np.allclose(
            arrays[name], q[:, section], atol=1e-10, rtol=0
        ):
            raise ValueError(f"Named array disagrees with qpos: {name}")
    times = np.arange(len(q), dtype=float) / fps
    if not np.allclose(arrays["time_s"], times, atol=1e-9, rtol=0):
        raise ValueError("Source timestamps disagree with the declared FPS")
    target_times = np.arange(int(np.floor(times[-1] * target_fps + 1e-8)) + 1)
    target_times = target_times / target_fps

    def linear(values):
        return np.stack(
            [np.interp(target_times, times, v) for v in values.T], axis=-1
        )

    def rotations(values):
        if np.max(abs(np.linalg.norm(values, axis=-1) - 1)) > 1e-5:
            raise ValueError("Source WXYZ quaternions are not unit length")
        r = Rotation.from_quat(values[:, [1, 2, 3, 0]])
        return Slerp(times, r)(target_times).as_quat()[:, [3, 0, 1, 2]]

    raw = np.concatenate(
        [linear(q[:, :3]), rotations(q[:, 3:7]), linear(q[:, 7:33])], axis=-1
    )
    return dict(
        fps=np.array(target_fps),
        joint_names=np.array(names),
        joint_pos=raw,
        object_pos_w=linear(q[:, 33:36]),
        object_quat_w=rotations(q[:, 36:40]),
    )


def convert(source, output, urdf):
    """Preserve the trajectory and box axes; audit the actual training URDF."""
    source, output, urdf = source.resolve(), output.resolve(), urdf.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    named = source / "trajectory/r1_joint_trajectory_30hz.npz"
    source_scene = source / "scene/r1_26dof_w_largebox.xml"
    metadata = json.loads((source / "manifest.json").read_text())
    task_name = metadata.get("task", "sub3_largebox_003")
    original_name = Path(
        metadata.get("artifacts", {}).get(
            "trajectory", f"{task_name}_r1_original_omni.npz"
        )
    ).name
    original = source / "trajectory" / original_name
    if (
        hashlib.sha256(original.read_bytes()).hexdigest()
        != metadata["trajectory_sha256"]
    ):
        raise ValueError("Source trajectory hash differs from its manifest")
    with np.load(named, allow_pickle=False) as package:
        arrays = {k: package[k].copy() for k in package.files}
    with np.load(original, allow_pickle=False) as package:
        if not np.array_equal(package["qpos"], arrays["qpos"]):
            raise ValueError("Original and named source trajectories disagree")
    packed = resample_reference(arrays)
    output.mkdir(parents=True, exist_ok=True)
    scene_tree = ET.parse(source_scene)
    omitted_visual_assets = []
    for mesh in scene_tree.getroot().findall("./asset/mesh"):
        path = urdf.parent / "meshes" / Path(mesh.get("file")).name
        if not path.is_file():
            geometries = [
                g
                for g in scene_tree.getroot().iter("geom")
                if g.get("mesh") == mesh.get("name")
            ]
            if not geometries or any(
                g.get("contype", "1") != "0"
                or g.get("conaffinity", "1") != "0"
                or g.get("density") != "0"
                for g in geometries
            ):
                raise FileNotFoundError(path)
            # This audit needs link transforms, not optional camera cosmetics.
            # Keep the body's explicitly declared inertial and every collision.
            for parent in scene_tree.getroot().iter():
                for geom in geometries:
                    if geom in list(parent):
                        parent.remove(geom)
            scene_tree.getroot().find("asset").remove(mesh)
            omitted_visual_assets.append(mesh.get("name"))
            continue
        mesh.set("file", str(path))
    local_scene = output / "source_scene_local.xml"
    scene_tree.write(local_scene)
    scene = mujoco.MjModel.from_xml_path(str(local_scene))
    sd = mujoco.MjData(scene)
    contract = robot_contract("r1")
    model = build_model(urdf, output / "audit/robot.xml", contract=contract)
    md = mujoco.MjData(model)
    source_joints = [scene.joint(i).name for i in range(scene.njnt)]
    if source_joints[1:-1] != contract.joints or scene.nq != 40:
        raise ValueError("Source scene joint order differs from named qpos")
    common = [
        model.body(i).name
        for i in range(1, model.nbody)
        if model.body(i).name
        in {scene.body(j).name for j in range(scene.nbody)}
    ]
    mi = [model.body(n).id for n in common]
    si = [scene.body(n).id for n in common]
    position_error, angle_error = 0.0, 0.0
    for q in arrays["qpos"]:
        sd.qpos[:] = q
        md.qpos[:] = q[:33]
        mujoco.mj_kinematics(scene, sd)
        mujoco.mj_kinematics(model, md)
        position_error = max(
            position_error,
            float(np.linalg.norm(sd.xpos[si] - md.xpos[mi], axis=-1).max()),
        )
        dots = np.abs(np.sum(sd.xquat[si] * md.xquat[mi], axis=-1))
        angle_error = max(
            angle_error, float((2 * np.arccos(np.clip(dots, 0, 1))).max())
        )
    if position_error > 1e-5 or angle_error > 1e-4:
        raise ValueError(
            "Source scene/URDF kinematics differ: "
            f"{position_error}, {angle_error}"
        )
    limits = np.array([model.joint(n).range for n in contract.joints])
    joints = packed["joint_pos"][:, 7:]
    violation = np.maximum(limits[:, 0] - joints, 0) + np.maximum(
        joints - limits[:, 1], 0
    )
    if violation.max() > 1e-5:
        raise ValueError(
            "Resampled reference exceeds actual URDF joint limits"
        )
    geom = scene.geom("largebox_collision")
    if geom.type != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("Expected the source cuboid collision geometry")
    size = geom.size * 2
    if not np.allclose(sorted(size), sorted([0.305, 0.245, 0.270]), atol=1e-8):
        raise ValueError(
            "Source box differs from the confirmed physical dimensions"
        )
    mass = float(scene.body("largebox_link").mass.item())
    if not np.isclose(mass, 0.1):
        raise ValueError("Source box mass differs from the confirmed 0.1 kg")
    obj_rotation = Rotation.from_quat(packed["object_quat_w"][:, [1, 2, 3, 0]])
    half_height = np.einsum(
        "ni,i->n", abs(obj_rotation.as_matrix()[:, 2]), size / 2
    )
    clearance = packed["object_pos_w"][:, 2] - half_height
    if clearance[0] < -0.01:
        raise ValueError(
            "Initial source box penetrates the floor by more than 1 cm"
        )
    reference = output / "reference_50hz.npz"
    np.savez_compressed(reference, **packed)
    manifest = prepare(
        reference,
        output / "dataset",
        "r1",
        size,
        mass,
        clip_name=task_name,
    )
    provenance = dict(
        source_directory=str(source),
        source_fps=float(arrays["fps"]),
        source_frames=len(arrays["qpos"]),
        source_duration_s=(len(arrays["qpos"]) - 1) / float(arrays["fps"]),
        target_fps=50,
        target_frames=len(joints),
        target_duration_s=(len(joints) - 1) / 50,
        interpolation=(
            "Linear positions/joints; quaternion SLERP; "
            "no smoothing, trimming or standing prefix"
        ),
        source_box_local_xyz_m=size.tolist(),
        common_robot_object_frame_preserved=True,
        head_reference_zero=bool(np.max(abs(joints[:, 24:])) < 1e-8),
        compared_named_links=len(common),
        omitted_nonphysical_audit_visual_assets=omitted_visual_assets,
        source_scene_vs_urdf_max_position_m=position_error,
        source_scene_vs_urdf_max_rotation_rad=angle_error,
        joint_limit_violation_rad=float(violation.max()),
        maximum_joint_velocity_limit_fraction=float(
            (
                abs(np.gradient(joints, 0.02, axis=0))
                / np.array(contract.velocity)
            ).max()
        ),
        initial_box_bottom_m=float(clearance[0]),
        minimum_reference_box_bottom_m=float(clearance.min()),
        peak_reference_box_bottom_m=float(clearance.max()),
        input_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [named, original, source_scene, urdf]
        },
        checks_passed=True,
        dynamics_validated=False,
    )
    manifest["provenance"].update(provenance)
    atomic_json(output / "dataset/manifest.json", manifest)
    atomic_json(output / "validation.json", provenance)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    options = parser.parse_args()
    print(
        json.dumps(
            convert(options.source, options.output, options.robot_urdf),
            indent=2,
        )
    )
