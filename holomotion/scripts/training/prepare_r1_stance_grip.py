"""Add measured grounded Stance and geometry-based bilateral arm closure."""

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from holomotion.scripts.training.prepare_supermimic_omomo import prepare
from holomotion.src.env.reference_tracking.model import build_model
from holomotion.src.env.reference_tracking.r1_box_contract import (
    R1_BOX_AXES,
    R1_REAL_BOX_SIZE_M,
    R1_TRAIN_BOX_SIZE_M,
    validate_size,
)
from holomotion.src.env.reference_tracking.robot_contract import robot_contract
from holomotion.src.utils.artifacts import atomic_json


def smootherstep(x):
    x = np.clip(x, 0, 1)
    return x**3 * (10 - 15 * x + 6 * x**2)


def collision_geoms(model, body_name):
    body = model.body(body_name).id
    return [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == body
        and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]


def scene_model(urdf, output, size, mass):
    build_model(urdf, output / "robot.xml", contract=robot_contract("r1"))
    tree = ET.parse(output / "robot.xml")
    box = ET.SubElement(
        tree.getroot().find("worldbody"), "body", name="grip_box"
    )
    ET.SubElement(box, "freejoint", name="grip_box_joint")
    ET.SubElement(
        box,
        "geom",
        name="grip_box_surface",
        type="box",
        size=" ".join(str(s / 2) for s in size),
        mass=str(mass),
        friction="0.8 0.005 0.0001",
        rgba=".72 .52 .30 1",
    )
    path = output / "scene.xml"
    tree.write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.ccd_iterations = 500
    model.opt.ccd_tolerance = 1e-9
    return model


def foot_bottoms(model, data):
    """Compute ground clearance from actual collision surfaces."""
    result = []
    for side in ("left", "right"):
        geoms = collision_geoms(model, side + "_ankle_roll_link")
        bottoms = []
        for g in geoms:
            kind = model.geom_type[g]
            if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                bottoms.append(data.geom_xpos[g, 2] - model.geom_size[g, 0])
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                extent = (
                    abs(data.geom_xmat[g].reshape(3, 3)[2])
                    @ model.geom_size[g]
                )
                bottoms.append(data.geom_xpos[g, 2] - extent)
            elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]
                start = model.mesh_vertadr[mesh]
                count = model.mesh_vertnum[mesh]
                vertices = model.mesh_vert[start : start + count]
                row = data.geom_xmat[g].reshape(3, 3)[2]
                bottoms.append(data.geom_xpos[g, 2] + (vertices @ row).min())
            else:
                raise ValueError("Unsupported physical foot collision shape")
        if not bottoms:
            raise ValueError(f"No collision surfaces on {side} foot")
        result.append(min(bottoms))
    return np.array(result)


def distances(model, data):
    box = model.geom("grip_box_surface").id
    return np.array(
        [
            min(
                mujoco.mj_geomDistance(model, data, g, box, 0.1, np.zeros(6))
                for g in collision_geoms(model, side + "_wrist_roll_link")
            )
            for side in ("left", "right")
        ]
    )


def close_arms(
    model,
    reference,
    size,
    preload,
    max_correction=0.5,
    fixed_jacobian_step=None,
):
    """Use actual wrist collisions; alter ten arm joints only, with bounds."""
    data = mujoco.MjData(model)
    baseline = []
    clearance = []
    for row in reference:
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        baseline.append(distances(model, data))
        rotation = data.xmat[model.body("grip_box").id].reshape(3, 3)
        clearance.append(row[35] - abs(rotation[2]) @ (np.array(size) / 2))
    baseline, clearance = np.array(baseline), np.array(clearance)
    lifted = np.flatnonzero(clearance > 0.025)
    near = np.flatnonzero(np.max(baseline, axis=1) < 0.05)
    if not len(lifted) or not len(near):
        raise ValueError(
            "Cannot identify a bilateral approach and carry phase"
        )
    start = max(0, min(near[0], lifted[0]) - 10)
    full = min(lifted[0], max(start + 1, near[0] + 5))
    end = min(len(reference) - 1, lifted[-1] + 10)
    weight = smootherstep(
        (np.arange(len(reference)) - start) / max(full - start, 1)
    )
    weight *= 1 - smootherstep(
        (np.arange(len(reference)) - lifted[-1]) / max(end - lifted[-1], 1)
    )
    result = reference.copy()
    corrections = np.zeros((len(reference), 2, 5))
    after = baseline.copy()
    for f, row in enumerate(reference):
        if weight[f] < 1e-8:
            continue
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        box_axis = (
            data.xmat[model.body("grip_box").id].reshape(3, 3)[:, 0].copy()
        )
        for side_index, side in enumerate(("left", "right")):
            section = slice(21 + side_index * 5, 26 + side_index * 5)
            body = model.body(side + "_wrist_roll_link").id
            p0 = data.xpos[body].copy()
            r0 = data.xmat[body].reshape(3, 3).copy()
            gap0 = baseline[f, side_index]
            target_gap = gap0 * (1 - weight[f]) - preload * weight[f]
            side_sign = np.sign((p0 - row[33:36]) @ box_axis)
            if side_sign == 0:
                raise ValueError(
                    "Cannot identify the wrist's box contact face"
                )
            normal = box_axis * side_sign
            target_p = p0 + normal * (target_gap - gap0)
            q0 = row[section].copy()
            names = robot_contract("r1").joints[
                14 + side_index * 5 : 19 + side_index * 5
            ]
            limits = np.array([model.joint(n).range for n in names])
            lower = np.maximum(limits[:, 0] - q0 + 1e-7, -max_correction)
            upper = np.minimum(limits[:, 1] - q0 - 1e-7, max_correction)
            if f:
                step_limit = (
                    np.array(
                        robot_contract("r1").velocity[
                            14 + side_index * 5 : 19 + side_index * 5
                        ]
                    )
                    * 0.02
                    * 0.9
                )
                previous_q = result[f - 1, section]
                lower = np.maximum(lower, previous_q - q0 - step_limit)
                upper = np.minimum(upper, previous_q - q0 + step_limit)
            previous = corrections[f - 1, side_index] if f else np.zeros(5)
            forearm = collision_geoms(model, side + "_elbow_link")
            box_geom = model.geom("grip_box_surface").id

            def residual(
                delta,
                section=section,
                q0=q0,
                body=body,
                r0=r0,
                side_index=side_index,
                box_geom=box_geom,
                forearm=forearm,
                target_p=target_p,
                target_gap=target_gap,
                previous=previous,
            ):
                data.qpos[section] = q0 + delta
                mujoco.mj_kinematics(model, data)
                p = data.xpos[body]
                r = data.xmat[body].reshape(3, 3)
                rotation_error = Rotation.from_matrix(r @ r0.T).as_rotvec()
                gap = distances(model, data)[side_index]
                penetration = [
                    max(
                        -0.003
                        - mujoco.mj_geomDistance(
                            model, data, g, box_geom, 2, None
                        ),
                        0,
                    )
                    / 0.001
                    for g in forearm
                ]
                return np.r_[
                    (p - target_p) / 0.003,
                    rotation_error / 0.2,
                    (gap - target_gap) / 0.001,
                    delta / 0.6,
                    (delta - previous) / 0.06,
                    penetration,
                ]

            def fixed_jacobian(delta):
                epsilon = fixed_jacobian_step
                columns = []
                for j in range(len(delta)):
                    step = np.zeros_like(delta)
                    step[j] = epsilon
                    columns.append(
                        (residual(delta + step) - residual(delta - step))
                        / (2 * epsilon)
                    )
                return np.stack(columns, axis=1)

            solution = least_squares(
                residual,
                np.clip(previous, lower + 1e-9, upper - 1e-9),
                bounds=(lower, upper),
                max_nfev=60,
                diff_step=1e-4,
                jac="2-point"
                if fixed_jacobian_step is None
                else fixed_jacobian,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )
            result[f, section] = q0 + solution.x
            corrections[f, side_index] = solution.x
            data.qpos[section] = result[f, section]
            mujoco.mj_kinematics(model, data)
        after[f] = distances(model, data)
    return result, dict(
        active_weight=weight,
        baseline_gaps_m=baseline,
        adjusted_gaps_m=after,
        reference_box_bottom_m=clearance,
        arm_corrections_rad=corrections,
        carry_frames=lifted,
        full_closure_start_frame=full,
        closure_ramp_start_frame=start,
        release_end_frame=end,
    )


def place_stance_at_box(model, stance, measured_rpy, evaluation_size, gap):
    """Face the box with the measured stance at an actual toe clearance."""
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError("Expected a positive foot/box clearance")
    data = mujoco.MjData(model)
    box_rotation = Rotation.from_quat(stance[[37, 38, 39, 36]]).as_matrix()
    forward = -box_rotation[:2, 1]
    forward /= np.linalg.norm(forward)
    data.qpos[:] = stance
    mujoco.mj_kinematics(model, data)
    old_feet_center = np.mean(
        [
            data.xpos[model.body(side + "_ankle_roll_link").id, :2]
            for side in ("left", "right")
        ],
        axis=0,
    )
    if (stance[33:35] - old_feet_center) @ forward < 0:
        forward *= -1
    lateral = np.array([-forward[1], forward[0]])
    yaw = np.arctan2(forward[1], forward[0])
    stance[3:7] = Rotation.from_euler(
        "xyz", [*measured_rpy[:2], yaw]
    ).as_quat()[[3, 0, 1, 2]]
    stance[:2] = stance[33:35] - 0.6 * forward
    stance[2] = 0
    data.qpos[:] = stance
    mujoco.mj_kinematics(model, data)
    stance[2] = -foot_bottoms(model, data).min()
    feet = {
        side: collision_geoms(model, side + "_ankle_roll_link")
        for side in ("left", "right")
    }
    mean_foot = np.mean(
        [
            data.xpos[model.body(side + "_ankle_roll_link").id, :2]
            for side in feet
        ],
        axis=0,
    )
    stance[:2] += lateral * ((stance[33:35] - mean_foot) @ lateral)
    box = model.geom("grip_box_surface").id
    original_size = model.geom_size[box].copy()
    original_bound = model.geom_rbound[box].copy()
    forward_3d = np.r_[forward, 0.0]
    rear = stance[33:36] @ forward_3d - (
        abs(forward_3d @ box_rotation) @ (np.array(evaluation_size) / 2)
    )

    def toe_witnesses():
        data.qpos[:] = stance
        mujoco.mj_kinematics(model, data)
        result = {}
        for side, geoms in feet.items():
            points = []
            for g in geoms:
                if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    raise ValueError(
                        "Expected actual R1 foot collision meshes"
                    )
                mesh = model.geom_dataid[g]
                first, count = (
                    model.mesh_vertadr[mesh],
                    model.mesh_vertnum[mesh],
                )
                vertices = model.mesh_vert[first : first + count]
                points.append(
                    vertices @ data.geom_xmat[g].reshape(3, 3).T
                    + data.geom_xpos[g]
                )
            vertices = np.concatenate(points)
            support = vertices @ forward_3d
            result[side] = vertices[support >= support.max() - 1e-10].mean(0)
        return result

    witnesses = toe_witnesses()
    shift = rear - gap - max(v @ forward_3d for v in witnesses.values())
    stance[:2] += shift * forward
    witnesses = toe_witnesses()
    real_gaps = {
        side: float(rear - v @ forward_3d) for side, v in witnesses.items()
    }
    for side, witness in witnesses.items():
        local = box_rotation.T @ (witness - stance[33:36])
        half = np.array(evaluation_size) / 2
        vertex_distance = np.linalg.norm(local - np.clip(local, -half, half))
        if abs(vertex_distance - real_gaps[side]) > 1e-5:
            raise ValueError(
                "Toe support witness is outside the box's back face"
            )

    try:
        model.geom_size[box] = np.array(evaluation_size) / 2
        model.geom_rbound[box] = np.linalg.norm(model.geom_size[box])
        data.qpos[:] = stance
        mujoco.mj_fwdPosition(model, data)
        contacts = [
            float(c.dist) for c in data.contact if box in (c.geom1, c.geom2)
        ]
        if contacts and min(contacts) < -1e-5:
            raise ValueError("Near-foot reset overlaps robot or floor")
    finally:
        model.geom_size[box] = original_size
        model.geom_rbound[box] = original_bound
    return dict(
        mode="official_stance_facing_box_near_toes",
        target_nearest_real_box_foot_gap_m=gap,
        evaluation_foot_box_gaps_m=real_gaps,
        clearance_method="Exact mesh support plane plus box-face witness",
        training_foot_box_forward_clearance_lower_bound_m=real_gaps,
        reference_box_world_pose_unchanged=True,
        initial_robot_box_penetration_m=max(-min(contacts, default=0), 0),
    )


def prepend_stance(
    model,
    task,
    stance_q,
    measured_rpy,
    hold_s=0.5,
    transition_s=2,
    *,
    near_feet_gap_m=None,
    evaluation_size=None,
):
    n_hold, n_blend = round(hold_s * 50), round(transition_s * 50)
    data = mujoco.MjData(model)
    yaw = Rotation.from_quat(task[0, [4, 5, 6, 3]]).as_euler("xyz")[2]
    stance = task[0].copy()
    stance[3:7] = Rotation.from_euler(
        "xyz", [*measured_rpy[:2], yaw]
    ).as_quat()[[3, 0, 1, 2]]
    stance[7:33] = stance_q
    stance[2] = 0
    foot_ids = [
        model.body(side + "_ankle_roll_link").id for side in ("left", "right")
    ]
    data.qpos[:] = task[0]
    mujoco.mj_kinematics(model, data)
    task_feet = data.xpos[foot_ids].copy()
    data.qpos[:] = stance
    mujoco.mj_kinematics(model, data)
    stand_feet = data.xpos[foot_ids].copy()
    task_line = task_feet[0] - task_feet[1]
    stand_line = stand_feet[0] - stand_feet[1]
    yaw_alignment = np.arctan2(task_line[1], task_line[0]) - np.arctan2(
        stand_line[1], stand_line[0]
    )
    aligned_rotation = Rotation.from_euler("z", yaw_alignment) * (
        Rotation.from_quat(stance[[4, 5, 6, 3]])
    )
    stance[3:7] = aligned_rotation.as_quat()[[3, 0, 1, 2]]
    data.qpos[:] = stance
    mujoco.mj_kinematics(model, data)
    stance[:2] += (task_feet - data.xpos[foot_ids]).mean(0)[:2]
    data.qpos[:] = stance
    mujoco.mj_kinematics(model, data)
    stance[2] = -foot_bottoms(model, data).min()
    placement = None
    if near_feet_gap_m is not None:
        placement = place_stance_at_box(
            model, stance, measured_rpy, evaluation_size, near_feet_gap_m
        )
        placed_yaw = Rotation.from_quat(stance[[4, 5, 6, 3]]).as_euler("xyz")[
            2
        ]
        yaw_alignment = np.arctan2(
            np.sin(placed_yaw - yaw), np.cos(placed_yaw - yaw)
        )
    alpha = smootherstep(np.arange(n_blend) / n_blend)
    blend = stance[None] + alpha[:, None] * (task[0] - stance)[None]
    rotation = Slerp(
        [0, 1],
        Rotation.from_quat(
            np.stack([stance[[4, 5, 6, 3]], task[0, [4, 5, 6, 3]]])
        ),
    )
    blend[:, 3:7] = rotation(alpha).as_quat()[:, [3, 0, 1, 2]]
    prefix = np.concatenate([np.tile(stance, (n_hold, 1)), blend])
    # Ground the approach prefix; leave source task feet unchanged.
    height_adjustments = []
    for row in prefix:
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        shift = max(-float(foot_bottoms(model, data).min()), 0)
        row[2] += shift
        height_adjustments.append(shift)
    terminal = np.repeat(task[-1:], 25, axis=0)
    return np.concatenate([prefix, task, terminal]), dict(
        hold_s=hold_s,
        transition_s=transition_s,
        prefix_frames=len(prefix),
        terminal_hold_s=0.5,
        stance_yaw_alignment_rad=float(yaw_alignment),
        stance_root_xy_from_task_m=(stance[:2] - task[0, :2]).tolist(),
        stance_qpos=stance[:33].tolist(),
        root_height_from_actual_collision_surfaces=True,
        maximum_prefix_ground_correction_m=max(height_adjustments),
        frame_zero_waiting="Freeze policy reference clock at frame zero",
        initial_box_placement=placement,
    )


def audit_motion(model, complete, prefix_frames):
    """Check joint limits, joint speeds and prefix ground clearance."""
    contract = robot_contract("r1")
    joints = complete[:, 7:33]
    limits = np.array([model.joint(n).range for n in contract.joints])
    violation = np.maximum(limits[:, 0] - joints, 0) + np.maximum(
        joints - limits[:, 1], 0
    )
    speed = abs(np.diff(joints, axis=0)) / 0.02
    fraction = float((speed / np.array(contract.velocity)).max())
    data = mujoco.MjData(model)
    feet = []
    box_penetrations = [0.0]
    box_geom = model.geom("grip_box_surface").id
    for row in complete[:prefix_frames]:
        data.qpos[:] = row
        mujoco.mj_fwdPosition(model, data)
        feet.append(foot_bottoms(model, data))
        box_penetrations.extend(
            float(c.dist)
            for c in data.contact
            if c.geom1 == box_geom or c.geom2 == box_geom
        )
    return dict(
        joint_limit_violation_rad=float(violation.max()),
        maximum_direct_joint_velocity_limit_fraction=fraction,
        maximum_joint_step_deg=float(
            np.rad2deg(abs(np.diff(joints, axis=0))).max()
        ),
        initial_foot_bottom_m=feet[0].tolist(),
        prefix_minimum_foot_bottom_m=float(np.min(feet)),
        prefix_maximum_robot_box_penetration_m=max(-min(box_penetrations), 0),
        kinematic_checks_passed=bool(
            violation.max() < 1e-6
            and fraction <= 1
            and np.min(feet) >= -1e-5
            and min(box_penetrations) >= -0.003
        ),
    )


def build(
    source_reference,
    source_manifest,
    stance_capture,
    holosoma_root,
    urdf,
    output,
    preload=0.002,
    training_size=R1_TRAIN_BOX_SIZE_M,
    evaluation_size=R1_REAL_BOX_SIZE_M,
    near_feet_gap_m=None,
    max_arm_correction=0.5,
    fixed_jacobian_step=None,
):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    if not 0 <= preload <= 0.003:
        raise ValueError("Virtual closure preload must be between 0 and 3 mm")
    metadata = json.loads(source_manifest.read_text())
    training_size = validate_size(training_size)
    evaluation_size = validate_size(evaluation_size)
    with np.load(source_reference, allow_pickle=False) as z:
        raw = {k: z[k].copy() for k in z.files}
    contract = robot_contract("r1")
    if (
        raw["joint_names"].tolist() != contract.joints
        or float(raw["fps"]) != 50
    ):
        raise ValueError("Expected the audited named 50-Hz R1 task reference")
    sys.path.insert(0, str(holosoma_root / "tools"))
    from prepend_r1_measured_stance_transition import _load_stance

    q, rpy, capture = _load_stance(stance_capture, contract.joints, 10)
    if (
        capture["capture_metadata"].get("capture_condition")
        != "grounded_official_stance"
    ):
        raise ValueError(
            "Capture is not documented as grounded official Stance"
        )
    output.mkdir(parents=True, exist_ok=True)
    model = scene_model(
        urdf,
        output / "audit",
        training_size,
        metadata["box"]["mass_kg"],
    )
    reference = np.concatenate(
        [raw["joint_pos"], raw["object_pos_w"], raw["object_quat_w"]], axis=-1
    )
    original_reference = reference.copy()
    rotations = Rotation.from_quat(reference[:, [37, 38, 39, 36]]).as_matrix()
    source_half = np.array(metadata["box"]["size_m"]) / 2
    target_half = np.array(training_size) / 2
    box_height_shift = abs(rotations[:, 2]) @ (target_half - source_half)
    reference[:, 35] += box_height_shift
    adjusted, grip = close_arms(
        model,
        reference,
        training_size,
        preload,
        max_correction=max_arm_correction,
        fixed_jacobian_step=fixed_jacobian_step,
    )
    complete, prefix = prepend_stance(
        model,
        adjusted,
        q,
        rpy,
        near_feet_gap_m=near_feet_gap_m,
        evaluation_size=evaluation_size,
    )
    heading = Rotation.from_euler(
        "z", Rotation.from_quat(complete[0, [4, 5, 6, 3]]).as_euler("xyz")[2]
    ).as_matrix()
    box_rotation = Rotation.from_quat(
        complete[0, [37, 38, 39, 36]]
    ).as_matrix()
    relative_axes = heading.T @ box_rotation
    axis_alignment = np.abs(relative_axes[[1, 0, 2], [0, 1, 2]])
    carry = grip["carry_frames"]
    errors = grip["adjusted_gaps_m"][carry]
    report = dict(
        source_reference=str(source_reference),
        box_axis_order=list(R1_BOX_AXES),
        training_box_size_m=training_size,
        evaluation_box_size_m=evaluation_size,
        original_box_size_m=metadata["box"]["size_m"],
        initial_box_center_z_shift_m=float(box_height_shift[0]),
        object_bottom_height_preserved=True,
        initial_box_axis_alignment_cosine=axis_alignment.tolist(),
        official_stance_capture=capture,
        standing_transition=prefix,
        virtual_preload_m=preload,
        grip_scope=(
            "Offline arm IK with actual URDF collision meshes; "
            "no interaction reward"
        ),
        offline_clearance_backend=(
            "MuJoCo native CCD; independent toe mesh witnesses"
        ),
        baseline_carry_gap_range_m=[
            grip["baseline_gaps_m"][carry].min(0).tolist(),
            grip["baseline_gaps_m"][carry].max(0).tolist(),
        ],
        adjusted_carry_gap_range_m=[
            errors.min(0).tolist(),
            errors.max(0).tolist(),
        ],
        maximum_arm_correction_rad=float(
            abs(grip["arm_corrections_rad"]).max()
        ),
        arm_correction_bound_rad=max_arm_correction,
        ik_absolute_jacobian_step_rad=fixed_jacobian_step,
        source_task_nonarm_qpos_preserved=bool(
            np.array_equal(adjusted[:, :21], reference[:, :21])
            and np.array_equal(adjusted[:, 31:33], reference[:, 31:33])
        ),
        source_task_object_xy_orientation_preserved=bool(
            np.array_equal(adjusted[:, 33:35], original_reference[:, 33:35])
            and np.array_equal(
                adjusted[:, 36:40], original_reference[:, 36:40]
            )
        ),
        frames=len(complete),
        duration_s=(len(complete) - 1) / 50,
        input_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                source_reference,
                source_manifest,
                stance_capture,
                urdf,
                holosoma_root
                / "tools/prepend_r1_measured_stance_transition.py",
                holosoma_root / "tools/prepend_r1_standing_transition.py",
            ]
        },
        checks_passed=bool(errors.max() < 0.003 and errors.min() > -0.004),
        dynamics_validated=False,
    )
    report.update(audit_motion(model, complete, prefix["prefix_frames"]))
    report["measured_stance_first_joint_pose_exact"] = bool(
        np.array_equal(complete[0, 7:33], q)
    )
    report["checks_passed"] &= report["kinematic_checks_passed"]
    report["checks_passed"] &= bool(axis_alignment.min() > 0.98)
    np.savez_compressed(output / "grip_audit.npz", **grip)
    np.savez_compressed(
        output / "reference_50hz.npz",
        fps=50,
        joint_names=contract.joints,
        joint_pos=complete[:, :33],
        object_pos_w=complete[:, 33:36],
        object_quat_w=complete[:, 36:40],
    )
    atomic_json(output / "validation.json", report)
    if not report["checks_passed"]:
        raise ValueError(
            "Bilateral mesh clearance audit failed; see validation.json"
        )
    manifest = prepare(
        output / "reference_50hz.npz",
        output / "dataset",
        "r1",
        training_size,
        metadata["box"]["mass_kg"],
        clip_name=next(iter(metadata["clips"])),
    )
    manifest["provenance"].update(
        standing_transition=prefix,
        stance_capture_sha256=report["input_sha256"][str(stance_capture)],
        virtual_preload_m=preload,
    )
    manifest["box_axis_order"] = list(R1_BOX_AXES)
    manifest["evaluation_box"] = dict(manifest["box"], size_m=evaluation_size)
    manifest["dataset_id"] = (
        f"omomo_{next(iter(metadata['clips']))}_r1_"
        f"narrow{training_size[0] * 1000:g}_real{evaluation_size[0] * 1000:g}"
    )
    atomic_json(output / "dataset/manifest.json", manifest)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reference", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--stance-capture", type=Path, required=True)
    parser.add_argument("--holosoma-root", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preload-mm", type=float, default=2)
    parser.add_argument(
        "--training-box-size", type=float, nargs=3, default=R1_TRAIN_BOX_SIZE_M
    )
    parser.add_argument(
        "--evaluation-box-size",
        type=float,
        nargs=3,
        default=R1_REAL_BOX_SIZE_M,
    )
    parser.add_argument("--near-feet-gap-mm", type=float)
    parser.add_argument("--max-arm-correction-rad", type=float, default=0.5)
    parser.add_argument("--ik-jacobian-step", type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.source_reference.resolve(),
                args.source_manifest.resolve(),
                args.stance_capture.resolve(),
                args.holosoma_root.resolve(),
                args.robot_urdf.resolve(),
                args.output.resolve(),
                args.preload_mm / 1000,
                args.training_box_size,
                args.evaluation_box_size,
                None
                if args.near_feet_gap_mm is None
                else args.near_feet_gap_mm / 1000,
                args.max_arm_correction_rad,
                args.ik_jacobian_step,
            ),
            indent=2,
        )
    )
