"""Render a prepared R1 reference with an explicit kinematic-only label."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np

from holomotion.scripts.evaluation.render_omomo_reference import label
from holomotion.scripts.evaluation.render_omomo_rollout import robot_visuals
from holomotion.scripts.training.prepare_r1_stance_grip import scene_model


def render(manifest, urdf, output):
    metadata = json.loads(manifest.read_text())
    clip = next(iter(metadata["clips"].values()))
    shard = manifest.parent / metadata["hdf5_shards"][clip["shard"]]["file"]
    section = slice(clip["start"], clip["start"] + clip["length"])
    with h5py.File(shard, "r") as stream:
        arrays = {name: stream[name][section] for name in stream}
    rows = np.c_[
        arrays["ref_root_pos"],
        arrays["ref_root_rot"][:, [3, 0, 1, 2]],
        arrays["ref_dof_pos"],
        arrays["object_pos_w"],
        arrays["object_quat_w"],
    ]
    model = scene_model(
        urdf,
        output.parent / "render-assets",
        metadata["box"]["size_m"],
        metadata["box"]["mass_kg"],
    )
    option = robot_visuals(model, urdf)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.5, 125, -15
    pictures = []
    width, depth, height = metadata["box"]["size_m"]
    caption = (
        f"REF ONLY LR={width * 100:g} FB={depth * 100:g} H={height * 100:g}cm"
    )
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(
            output, fps=50, codec="libx264", quality=8
        ) as writer:
            for i, row in enumerate(rows):
                data.qpos[:] = row
                mujoco.mj_kinematics(model, data)
                camera.lookat[:] = np.r_[row[:2], 0.65]
                renderer.update_scene(data, camera, scene_option=option)
                frame = label(renderer.render(), f"{caption} t={i / 50:.2f}s")
                writer.append_data(frame)
                if i in [0, min(300, len(rows) - 1)]:
                    pictures.append(frame)
    imageio.imwrite(
        output.with_suffix(".png"), np.concatenate(pictures, axis=1)
    )
    return dict(frames=len(rows), reference_only=True, output=str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            render(args.manifest, args.robot_urdf, args.output), indent=2
        )
    )
