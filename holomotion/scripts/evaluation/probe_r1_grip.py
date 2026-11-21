"""Isolate bilateral grip geometry from robot balance in controlled probes."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from holomotion.src.utils.artifacts import atomic_json


def probe(scene, reference, frame, *, gravity_compensation=False):
    with np.load(reference, allow_pickle=False) as source:
        robot = source["joint_pos"][frame]
        box = np.r_[
            source["object_pos_w"][frame], source["object_quat_w"][frame]
        ]
    tree = ET.parse(scene)
    pelvis = tree.getroot().find('.//body[@name="pelvis_link"]')
    pelvis.remove(pelvis.find("freejoint"))
    pelvis.set("pos", " ".join(map(str, robot[:3])))
    pelvis.set("quat", " ".join(map(str, robot[3:7])))
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(tree.getroot(), encoding="unicode")
    )
    model.opt.timestep, model.opt.iterations = 0.001, 50
    box_geom = model.geom("grip_box_surface").id
    for i in range(model.ngeom):
        if i == box_geom:
            model.geom_contype[i], model.geom_conaffinity[i] = 2, 5
        elif model.geom_bodyid[i] == 0:
            model.geom_contype[i], model.geom_conaffinity[i] = 4, 2
        elif model.geom_contype[i] or model.geom_conaffinity[i]:
            model.geom_contype[i], model.geom_conaffinity[i] = 1, 2
    data = mujoco.MjData(model)
    data.qpos[:26], data.qpos[26:] = robot[7:], box
    valid, bottom = True, []
    for i in range(3000):
        mujoco.mj_forward(model, data)
        data.ctrl[:] = robot[7:]
        if gravity_compensation:
            data.ctrl[:] += data.qfrc_bias[:26] / model.actuator_gainprm[:, 0]
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or any(
            w.number for w in data.warning
        ):
            valid = False
            break
        if i % 20 == 19:
            mujoco.mj_kinematics(model, data)
            b = model.body("grip_box").id
            extent = (
                abs(data.xmat[b].reshape(3, 3)[2]) @ model.geom_size[box_geom]
            )
            bottom.append(float(data.xpos[b, 2] - extent))
    drift = float(np.linalg.norm(data.qpos[26:28] - box[:2]))
    held = bool(
        valid
        and bottom
        and min(bottom[-50:]) > 0.05
        and box[2] - data.qpos[28] < 0.05
        and drift < 0.05
    )
    return dict(
        frame=frame,
        gravity_compensation=gravity_compensation,
        valid=valid,
        initial_center_z_m=float(box[2]),
        final_center_z_m=float(data.qpos[28]),
        minimum_bottom_last_second_m=min(bottom[-50:]) if bottom else None,
        box_horizontal_drift_m=drift,
        held=held,
        warnings={
            str(j): int(w.number)
            for j, w in enumerate(data.warning)
            if w.number
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--task-offset", type=int, default=125)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    rows = [
        dict(
            task_time_s=second,
            **probe(
                options.scene,
                options.reference,
                options.task_offset + round(second * 50),
                gravity_compensation=compensated,
            ),
        )
        for compensated in [False, True]
        for second in [1.5, 3.5, 4.5]
    ]
    report = dict(
        scope=(
            "Controlled fixed-base grip probe. "
            "The 0.1 kg box is free under gravity; "
            "robot self/floor contacts disabled only for this diagnostic. "
            "Robot-gravity compensation is an explicit separate condition. "
            "This is not full-body balance or learned-policy validation."
        ),
        held_criterion=(
            "Last second above 5 cm; total vertical drop and horizontal "
            "drift each below 5 cm; no solver warnings"
        ),
        rows=rows,
        all_probes_held=all(r["held"] for r in rows),
    )
    atomic_json(options.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
