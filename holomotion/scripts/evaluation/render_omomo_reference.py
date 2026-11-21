"""Compare the commanded reference with the saved physical policy rollout."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from holomotion.scripts.evaluation.render_omomo_rollout import (
    load_render_model,
)


def label(frame, text):
    picture = Image.fromarray(frame)
    draw = ImageDraw.Draw(picture)
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    draw.rectangle((0, 0, picture.width, 34), fill=(20, 20, 20))
    draw.text((10, 5), text, font=font, fill=(255, 255, 255))
    return np.asarray(picture)


def render(training, output):
    specification = json.loads((training / "specification.json").read_text())
    manifest_path = Path(specification["arguments"]["manifest"])
    manifest = json.loads(manifest_path.read_text())
    clip = next(iter(manifest["clips"].values()))
    shard = (
        manifest_path.parent / manifest["hdf5_shards"][clip["shard"]]["file"]
    )
    section = slice(clip["start"], clip["start"] + clip["length"])
    with h5py.File(shard, "r") as stream:
        p = stream["ref_root_pos"][section]
        q = stream["ref_root_rot"][section]
        j = stream["ref_dof_pos"][section]
        box_p = stream["object_pos_w"][section]
        box_q = stream["object_quat_w"][section]
    trace = torch.load(
        training / "evaluation/rollout_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    states = trace["next_state"].numpy()
    boxes = trace["box_state"].numpy()
    frames = trace["reference_frame"].numpy()
    model, option = load_render_model(training, output)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance = 2.5
    camera.azimuth = 125
    camera.elevation = -15
    n = j.shape[-1]
    box_id = next(
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == model.body("evaluation_box").id
    )
    physical_box_size = model.geom_size[box_id].copy()
    train_box = specification["box"]
    distinct_boxes = not train_box.get("mesh_file") and not np.allclose(
        physical_box_size * 2, train_box["size_m"]
    )
    reference_label = "REFERENCE (kinematic)"
    policy_label = "POLICY (physical)"
    if distinct_boxes:
        reference_label = f"REF train LR={train_box['size_m'][0] * 100:g}cm"
        policy_label = f"POLICY eval LR={physical_box_size[0] * 200:g}cm"
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(
            output, fps=50, codec="libx264", quality=8
        ) as writer:
            for f in range(len(p)):
                if distinct_boxes:
                    model.geom_size[box_id] = np.array(train_box["size_m"]) / 2
                data.qpos[:7] = np.r_[p[f], q[f, [3, 0, 1, 2]]]
                data.qpos[7 : 7 + n] = j[f]
                data.qpos[7 + n :] = np.r_[box_p[f], box_q[f]]
                mujoco.mj_kinematics(model, data)
                camera.lookat[:] = np.r_[p[f, :2], 0.65]
                renderer.update_scene(data, camera, scene_option=option)
                left = label(
                    renderer.render(),
                    f"{reference_label}  t={f / 50:.2f}s",
                )
                index = min(int(np.searchsorted(frames, f)), len(states) - 1)
                state = states[index]
                data.qpos[:7] = state[:7]
                data.qpos[7 : 7 + n] = state[13 : 13 + n]
                data.qpos[7 + n :] = boxes[index]
                if distinct_boxes:
                    model.geom_size[box_id] = physical_box_size
                mujoco.mj_kinematics(model, data)
                camera.lookat[:] = np.r_[state[:2], 0.65]
                renderer.update_scene(data, camera, scene_option=option)
                right = label(
                    renderer.render(),
                    f"{policy_label}  t={frames[index] / 50:.2f}s",
                )
                writer.append_data(np.concatenate((left, right), axis=1))
            # Show the same URDF standing from the front and side.
            if distinct_boxes:
                model.geom_size[box_id] = np.array(train_box["size_m"]) / 2
            data.qpos[:7] = np.r_[p[0], q[0, [3, 0, 1, 2]]]
            data.qpos[7 : 7 + n] = j[0]
            data.qpos[7 + n :] = np.r_[box_p[0], box_q[0]]
            mujoco.mj_kinematics(model, data)
            camera.lookat[:] = np.r_[p[0, :2], 0.65]
            pictures = []
            for azimuth in (180, 90):
                camera.azimuth = azimuth
                renderer.update_scene(data, camera, scene_option=option)
                pictures.append(renderer.render().copy())
            imageio.imwrite(
                output.with_suffix(".png"), np.concatenate(pictures, axis=1)
            )
    return dict(
        frames=len(p),
        reference_is_kinematic=True,
        policy_is_physical=True,
        output=str(output),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.training, args.output), indent=2))
