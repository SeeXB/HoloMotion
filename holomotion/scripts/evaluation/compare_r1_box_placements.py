"""Compare two physical box placements using the exact same R1 checkpoint."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from holomotion.scripts.evaluation.render_omomo_reference import label
from holomotion.scripts.evaluation.render_omomo_rollout import (
    load_render_model,
)
from holomotion.src.env.reference_tracking.rotations import quat_to_mat
from holomotion.src.utils.artifacts import atomic_json, fingerprint


def metrics(report, trace):
    time = trace["reference_frame"].float() * trace["control_interval_s"]
    tilt = torch.rad2deg(
        torch.acos(
            quat_to_mat(trace["box_state"][:, 3:])[:, 2, 2].clamp(-1, 1)
        )
    )
    return dict(
        normalized_return=report["normalized_return"],
        failed_clips=report["failed_clips"],
        box_lift=report["clips"][0]["box_lift"],
        peak_box_tilt_deg=float(tilt.max()),
        peak_box_tilt_time_s=float(time[tilt.argmax()]),
        box_tilt_over_85deg_s=float((tilt > 85).sum())
        * trace["control_interval_s"],
        final_box_tilt_deg=float(tilt[-1]),
        peak_box_tilt_before_7s_deg=float(tilt[time < 7].max()),
        mean_box_tilt_before_7s_deg=float(tilt[time < 7].mean()),
    )


def compare(baseline, shifted, output):
    roots = [baseline.resolve(), shifted.resolve()]
    reports = [
        json.loads((p / "evaluation_final.json").read_text()) for p in roots
    ]
    if reports[0]["checkpoint_sha256"] != reports[1]["checkpoint_sha256"]:
        raise ValueError("Placement comparisons require the exact same actor")
    for report in reports:
        if report["evaluated_clips"] != 1 or report["failed_clips"]:
            raise ValueError(
                "This renderer requires one complete successful clip"
            )
    if (
        reports[0]["box_profile"]["evaluation_box"]
        != reports[1]["box_profile"]["evaluation_box"]
    ):
        raise ValueError(
            "The two evaluations use different physical box profiles"
        )
    traces = [
        torch.load(
            p / "evaluation/rollout_final.pt",
            map_location="cpu",
            weights_only=False,
        )
        for p in roots
    ]
    if not torch.equal(
        traces[0]["reference_frame"], traces[1]["reference_frame"]
    ):
        raise ValueError(
            "Comparison requires matching complete reference frames"
        )
    for p, report in zip(roots, reports, strict=True):
        if (
            fingerprint(p / "evaluation/rollout_final.pt")
            != report["rollout_sha256"]
        ):
            raise ValueError("Recorded physical trace changed")
    model, option = load_render_model(shifted, output)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.5, 125, -15
    n = (traces[0]["next_state"].shape[-1] - 13) // 2
    distance = reports[1]["box_profile"]["initial_placement"]["toward_robot_m"]
    titles = ["ORIGINAL PLACEMENT", f"TOWARD ROBOT {distance * 100:g}cm"]
    frames = traces[0]["reference_frame"].numpy()
    keyframes = []
    key_indices = {
        int(np.abs(frames / 50 - t).argmin())
        for t in (3.6, 4.8, 5.6, 7.5, 8.7)
    }
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(
            output, fps=50, codec="libx264", quality=8
        ) as writer:
            for i, frame in enumerate(frames):
                camera.lookat[:] = [
                    *(sum(t["next_state"][i, :2].numpy() for t in traces) / 2),
                    0.65,
                ]
                pictures = []
                for trace, title in zip(traces, titles, strict=True):
                    state = trace["next_state"][i].numpy()
                    data.qpos[:7] = state[:7]
                    data.qpos[7 : 7 + n] = state[13 : 13 + n]
                    data.qpos[7 + n :] = trace["box_state"][i].numpy()
                    mujoco.mj_kinematics(model, data)
                    renderer.update_scene(data, camera, scene_option=option)
                    pictures.append(
                        label(
                            renderer.render(), f"{title}  t={frame / 50:.2f}s"
                        )
                    )
                pair = np.concatenate(pictures, axis=1)
                writer.append_data(pair)
                if i in key_indices:
                    keyframes.append(pair)
    imageio.imwrite(output.with_suffix(".png"), np.concatenate(keyframes))
    result = dict(
        checkpoint_sha256=reports[0]["checkpoint_sha256"],
        actor_updates=reports[0]["actor_updates"],
        video=str(output),
        box_axis_order=reports[0]["box_profile"]["axis_order"],
        physical_box=reports[0]["box_profile"]["evaluation_box"],
        baseline=metrics(reports[0], traces[0]),
        toward_robot=metrics(reports[1], traces[1]),
        initial_placement=reports[1]["box_profile"]["initial_placement"],
        both_panels_are_recorded_physics=True,
        tilt_definition=(
            "Angle between box local up and world up; 90deg is on its side"
        ),
    )
    atomic_json(output.with_suffix(".json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--shifted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(compare(args.baseline, args.shifted, args.output), indent=2)
    )
