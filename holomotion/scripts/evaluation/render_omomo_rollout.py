"""Render saved physical evaluation states; never replay object references."""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import torch


def load_render_model(training, output):
    """Create a visualization model without changing recorded physics."""
    import json

    specification = json.loads((training / "specification.json").read_text())
    box = specification["box"]
    evaluation = training / "evaluation_final.json"
    if evaluation.is_file():
        profile = json.loads(evaluation.read_text()).get("box_profile")
        if profile:
            box = profile["evaluation_box"]
    tree = ET.parse(training / "g1.xml")
    world = tree.getroot().find("worldbody")
    body = ET.SubElement(world, "body", name="evaluation_box")
    ET.SubElement(body, "freejoint", name="box_freejoint")
    geometry = dict(
        type="box", size=" ".join(str(x / 2) for x in box["size_m"])
    )
    if box.get("mesh_path"):
        asset = tree.getroot().find("asset")
        if asset is None:
            asset = ET.SubElement(tree.getroot(), "asset")
        ET.SubElement(
            asset, "mesh", name="evaluation_box_surface", file=box["mesh_path"]
        )
        geometry = dict(type="mesh", mesh="evaluation_box_surface")
    ET.SubElement(
        body,
        "geom",
        **geometry,
        rgba=".72 .52 .30 1",
        mass=str(box["mass_kg"]),
    )
    ET.SubElement(
        world, "light", pos="0 -2 4", dir="0 0 -1", diffuse=".8 .8 .8"
    )
    scene_path = output.with_suffix(".xml")
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(scene_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    option = robot_visuals(
        model, Path(specification["arguments"]["robot_urdf"])
    )
    return model, option


def robot_visuals(model, urdf):
    """Show URDF visual materials and hide duplicate collision surfaces."""
    root = ET.parse(urdf).getroot()
    materials = {
        m.get("name"): m.find("color").get("rgba")
        for m in root.findall("material")
        if m.find("color") is not None
    }
    for link in root.findall("link"):
        visual = link.find("visual")
        if visual is None:
            continue
        try:
            body = model.body(link.get("name")).id
        except KeyError:
            continue
        material = visual.find("material")
        color = None
        if material is not None:
            color = material.find("color")
            color = (
                color.get("rgba")
                if color is not None
                else materials.get(material.get("name"))
            )
        indices = [
            i for i in range(model.ngeom) if model.geom_bodyid[i] == body
        ]
        has_visual = any(
            model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0
            for i in indices
        )
        for i in indices:
            is_visual = (
                model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0
            )
            if is_visual:
                model.geom_group[i] = 1
                if color is not None:
                    model.geom_rgba[i] = [float(v) for v in color.split()]
            elif has_visual:
                model.geom_group[i] = 3
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    return option


def render(training, output):
    trace = torch.load(
        training / "evaluation/rollout_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    model, option = load_render_model(training, output)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 0.65]
    camera.distance = 2.5
    camera.azimuth = 125
    camera.elevation = -15
    states = trace["next_state"].numpy()
    boxes = trace["box_state"].numpy()
    n = (states.shape[-1] - 13) // 2
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(
            output, fps=50, codec="libx264", quality=8
        ) as writer:
            for state, box_state in zip(states, boxes, strict=True):
                data.qpos[:7] = state[:7]
                data.qpos[7 : 7 + n] = state[13 : 13 + n]
                data.qpos[7 + n :] = box_state
                mujoco.mj_forward(model, data)
                camera.lookat[:2] = state[:2]
                renderer.update_scene(data, camera, scene_option=option)
                writer.append_data(renderer.render())
    return dict(
        frames=len(states),
        duration_seconds=len(states) / 50,
        output=str(output),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.training, args.output))
