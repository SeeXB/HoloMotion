"""Import OMOMO robot retargets, preserving the shared object frame."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from holomotion.src.env.reference_tracking.robot_contract import robot_contract


def prepare(
    source,
    output,
    robot,
    size,
    mass=0.1,
    friction=0.8,
    mesh=None,
    *,
    clip_name="sub3_largebox_003",
):
    contract = robot_contract(robot)
    if (
        len(size) != 3
        or not np.isfinite([*size, mass, friction]).all()
        or min(*size, mass, friction) <= 0
    ):
        raise ValueError(
            "Box dimensions, mass and friction must be finite and positive"
        )
    with np.load(source, allow_pickle=False) as data:
        names = data["joint_names"].tolist()
        if len(set(names)) != len(names) or set(names) != set(contract.joints):
            raise ValueError(
                "Named source joints must exactly match the robot"
            )
        raw = data["joint_pos"].copy()
        if (
            float(data["fps"].item()) != 50
            or raw.ndim != 2
            or raw.shape[1] != 7 + len(names)
            or len(raw) < 44
        ):
            raise ValueError(
                "Expected a complete 50-Hz floating-base robot trajectory"
            )
        obj_p, obj_q = (
            data["object_pos_w"].copy(),
            data["object_quat_w"].copy(),
        )
    if obj_p.shape != (len(raw), 3) or obj_q.shape != (len(raw), 4):
        raise ValueError("Robot and object must share the complete timeline")
    if not all(np.isfinite(x).all() for x in (raw, obj_p, obj_q)):
        raise ValueError("Nonfinite source motion")
    if any(
        (np.abs(np.linalg.norm(q, axis=-1) - 1) > 1e-4).any()
        for q in (raw[:, 3:7], obj_q)
    ):
        raise ValueError("Source quaternions must be normalized WXYZ")
    root_rotation = Rotation.from_quat(raw[:, [4, 5, 6, 3]])
    mat = root_rotation[0].as_matrix()
    heading_inv = Rotation.from_euler("z", -np.arctan2(mat[1, 0], mat[0, 0]))
    origin = raw[0, :3].copy()
    origin[2] = 0
    p = heading_inv.apply(raw[:, :3] - origin)
    root_xyzw = (heading_inv * root_rotation).as_quat()
    obj_p = heading_inv.apply(obj_p - origin)
    obj_xyzw = (
        heading_inv * Rotation.from_quat(obj_q[:, [1, 2, 3, 0]])
    ).as_quat()
    arrays = dict(
        ref_root_pos=p,
        ref_root_rot=root_xyzw,
        ref_dof_pos=raw[:, 7:][:, [names.index(j) for j in contract.joints]],
        object_pos_w=obj_p,
        object_quat_w=obj_xyzw[:, [3, 0, 1, 2]],
    )
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(output / "motions.h5", "w") as stream:
        for name, array in arrays.items():
            stream.create_dataset(
                name, data=array.astype(np.float32), compression="gzip"
            )
    manifest = dict(
        version=2,
        schema_version="root_dof_full_candidate_v1",
        dataset_id=f"omomo_{clip_name}_{robot}",
        robot=robot,
        motion_fps=50.0,
        root_quaternion_order="xyzw",
        dof_names=contract.joints,
        hdf5_shards=[dict(file="motions.h5")],
        clips={
            clip_name: dict(
                shard=0,
                start=0,
                length=len(raw),
                motion_fps=50.0,
                metadata=dict(source_id="omomo", motion_fps=50.0),
            )
        },
        box=dict(size_m=list(size), mass_kg=mass, friction=friction),
        provenance=dict(
            source_name=source.name,
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            frame="Z up, metres; shared initial root XY/yaw removal",
            source_contract="HoloSoma MuJoCo qpos and object WXYZ",
            interaction_rewards=False,
        ),
    )
    if mesh is not None:
        shutil.copyfile(mesh, output / "box.obj")
        manifest["box"].update(
            mesh_file="box.obj",
            mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
        )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", choices=("g1", "r1"), required=True)
    parser.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--box-mass", type=float, default=0.1)
    parser.add_argument("--box-mesh", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source,
                args.output,
                args.robot,
                args.box_size,
                args.box_mass,
                mesh=args.box_mesh,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
