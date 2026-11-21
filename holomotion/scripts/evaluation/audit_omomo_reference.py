"""Check a named robot reference against its source scene and training URDF."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from holomotion.src.env.reference_tracking.model import build_model
from holomotion.src.env.reference_tracking.robot_contract import robot_contract
from holomotion.src.utils.artifacts import atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_robot(path):
    root = ET.parse(path).getroot()
    for extension in root.findall("mujoco"):
        root.remove(extension)
    return ET.canonicalize(
        ET.tostring(root, encoding="unicode"), strip_text=True
    )


def quaternion_error(a, b):
    return 2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1))


def audit(
    source,
    urdf,
    manifest,
    output,
    *,
    original_urdf=None,
    expected_original_sha256=None,
):
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(manifest.read_text())
    if metadata.get("robot") != "r1":
        raise ValueError("This audit expects the R1 OMOMO source scene")
    contract = robot_contract(metadata["robot"])
    with np.load(source, allow_pickle=False) as package:
        arrays = {k: package[k].copy() for k in package.files}
    transition = json.loads(str(arrays["standing_transition_metadata"]))
    scene_path = Path(transition["scene_xml"])
    scene = mujoco.MjModel.from_xml_path(str(scene_path))
    sd = mujoco.MjData(scene)
    model = build_model(urdf, output / "model.xml", contract=contract)
    md = mujoco.MjData(model)
    names = arrays["joint_names"].tolist()
    bodies = arrays["body_names"].tolist()
    if names != contract.joints or len(set(names)) != len(names):
        raise ValueError(
            "Source joint order does not match the named robot contract"
        )
    if arrays["fps"].item() != 50 or arrays["joint_pos"].shape[1] != 7 + len(
        names
    ):
        raise ValueError("Expected 50-Hz MuJoCo floating-base reference")
    model_names = {model.body(i).name for i in range(model.nbody)}
    common = [
        n
        for n in bodies
        if n not in {"world", "largebox_link"} and n in model_names
    ]
    saved_ids = [bodies.index(n) for n in common]
    source_ids = [scene.body(n).id for n in common]
    model_ids = [model.body(n).id for n in common]
    scene_qids = [int(scene.joint(n).qposadr[0]) for n in names]
    object_qid = (
        int(scene.joint("largebox_joint").qposadr[0])
        if any(
            scene.joint(i).name == "largebox_joint" for i in range(scene.njnt)
        )
        else scene.nq - 7
    )
    errors = []
    angles = []
    saved_errors = []
    saved_angles = []
    distances = []

    def collision_geoms(body):
        return [
            i
            for i in range(scene.ngeom)
            if scene.geom_bodyid[i] == scene.body(body).id
            and (scene.geom_contype[i] or scene.geom_conaffinity[i])
        ]

    object_geoms = collision_geoms("largebox_link")
    hand_bodies = [
        "left_wrist_roll_link",
        "left_hand_link",
        "right_wrist_roll_link",
        "right_hand_link",
    ]
    hand_geoms = [collision_geoms(n) for n in hand_bodies]
    for f, raw in enumerate(arrays["joint_pos"]):
        sd.qpos[:7] = raw[:7]
        sd.qpos[scene_qids] = raw[7:]
        sd.qpos[object_qid : object_qid + 7] = np.r_[
            arrays["object_pos_w"][f], arrays["object_quat_w"][f]
        ]
        md.qpos[:] = raw
        mujoco.mj_kinematics(scene, sd)
        mujoco.mj_kinematics(model, md)
        errors.append(
            np.linalg.norm(md.xpos[model_ids] - sd.xpos[source_ids], axis=-1)
        )
        angles.append(
            quaternion_error(md.xquat[model_ids], sd.xquat[source_ids])
        )
        saved_errors.append(
            np.linalg.norm(
                sd.xpos[source_ids] - arrays["body_pos_w"][f, saved_ids],
                axis=-1,
            )
        )
        saved_angles.append(
            quaternion_error(
                sd.xquat[source_ids], arrays["body_quat_w"][f, saved_ids]
            )
        )
        distances.append(
            [
                min(
                    mujoco.mj_geomDistance(scene, sd, g, o, 1.0, None)
                    for g in group
                    for o in object_geoms
                )
                for group in hand_geoms
            ]
        )

    clip = next(iter(metadata["clips"].values()))
    section = slice(clip["start"], clip["start"] + clip["length"])
    shard = manifest.parent / metadata["hdf5_shards"][clip["shard"]]["file"]
    raw = arrays["joint_pos"]
    rotations = Rotation.from_quat(raw[:, [4, 5, 6, 3]])
    mat = rotations[0].as_matrix()
    heading = Rotation.from_euler("z", -np.arctan2(mat[1, 0], mat[0, 0]))
    origin = np.r_[raw[0, :2], 0.0]
    with h5py.File(shard, "r") as stream:
        packed = dict(
            joint_max_abs_rad=float(
                abs(stream["ref_dof_pos"][section] - raw[:, 7:]).max()
            ),
            root_position_max_m=float(
                abs(
                    stream["ref_root_pos"][section]
                    - heading.apply(raw[:, :3] - origin)
                ).max()
            ),
            root_orientation_max_rad=float(
                (
                    Rotation.from_quat(stream["ref_root_rot"][section]).inv()
                    * (heading * rotations)
                )
                .magnitude()
                .max()
            ),
            object_position_max_m=float(
                abs(
                    stream["object_pos_w"][section]
                    - heading.apply(arrays["object_pos_w"] - origin)
                ).max()
            ),
        )
    limits = np.array([model.joint(n).range for n in names])
    violation = np.maximum(limits[:, 0] - raw[:, 7:], 0) + np.maximum(
        raw[:, 7:] - limits[:, 1], 0
    )
    velocity = np.gradient(raw[:, 7:], 0.02, axis=0)
    distances = np.array(distances)
    lift = arrays["object_pos_w"][:, 2] > arrays["object_pos_w"][0, 2] + 0.1
    report = dict(
        scope=(
            "Static asset, kinematics, limits and contact geometry; "
            "dynamics success requires physical evaluation"
        ),
        robot=contract.name,
        frames=len(raw),
        joint_names_match=True,
        compared_named_links=len(common),
        source_only_helper_bodies=[
            n
            for n in bodies
            if n not in common and n not in {"world", "largebox_link"}
        ],
        urdf_vs_source_scene_max_m=float(np.max(errors)),
        urdf_vs_source_scene_max_rad=float(np.max(angles)),
        source_saved_body_vs_scene_max_m=float(np.max(saved_errors)),
        source_saved_body_vs_scene_max_rad=float(np.max(saved_angles)),
        joint_limit_violation_max_rad=float(violation.max()),
        joint_velocity_max_ratio=float(
            (abs(velocity) / np.array(contract.velocity)).max()
        ),
        packed_vs_source=packed,
        hand_contact=dict(
            columns=hand_bodies,
            reference_lift_frames=int(lift.sum()),
            lift_phase_gap_max_m=distances[lift].max(0).tolist(),
            lift_phase_gap_median_m=np.median(
                distances[lift], axis=0
            ).tolist(),
            both_actual_hands_within_1cm_fraction=float(
                (
                    (distances[lift, 0] <= 0.01) & (distances[lift, 2] <= 0.01)
                ).mean()
            ),
        ),
        input_sha256={
            str(p.resolve()): digest(p)
            for p in [source, urdf, manifest, shard, scene_path]
        },
    )
    if original_urdf is not None:
        report["original_urdf_sha256"] = digest(original_urdf)
        report["original_and_adapter_robot_identical"] = canonical_robot(
            original_urdf
        ) == canonical_robot(urdf)
        report["upstream_original_hash_matches"] = (
            expected_original_sha256 is not None
            and digest(original_urdf) == expected_original_sha256
        )
    report["checks_passed"] = bool(
        max(
            report[k]
            for k in [
                "urdf_vs_source_scene_max_m",
                "source_saved_body_vs_scene_max_m",
            ]
        )
        < 1e-5
        and max(packed.values()) < 1e-5
        and violation.max() < 1e-5
        and report["joint_velocity_max_ratio"] <= 1.0
        and (
            original_urdf is None
            or (
                report["original_and_adapter_robot_identical"]
                and report["upstream_original_hash_matches"]
            )
        )
    )
    np.save(output / "hand_box_distances.npy", distances)
    atomic_json(output / "report.json", report)
    if not report["checks_passed"]:
        raise ValueError(f"Reference audit failed: {output / 'report.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-urdf", type=Path)
    parser.add_argument("--expected-original-sha256")
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.source,
                args.urdf,
                args.manifest,
                args.output,
                original_urdf=args.original_urdf,
                expected_original_sha256=args.expected_original_sha256,
            ),
            indent=2,
        )
    )
