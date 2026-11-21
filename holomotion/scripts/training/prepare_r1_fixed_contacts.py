"""Resize R1 box references while enforcing the supplied surface anchors."""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from holomotion.scripts.training.prepare_r1_stance_grip import (
    audit_motion,
    prepend_stance,
    scene_model,
)
from holomotion.scripts.training.prepare_supermimic_omomo import prepare
from holomotion.src.env.reference_tracking.r1_box_contract import R1_BOX_AXES
from holomotion.src.env.reference_tracking.robot_contract import robot_contract
from holomotion.src.utils.artifacts import atomic_json, fingerprint


def contact_schedule(path, frames, source_size, target_size, fps=50):
    """Keep source surface Y/Z exactly; move only the opposing X faces."""
    with np.load(path, allow_pickle=False) as package:
        arrays = {key: package[key].copy() for key in package.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if metadata["schema"] != "holosoma.retarget_surface_contacts.v2":
        raise ValueError("Expected named fixed surface contact anchors")
    if not metadata["anchor_positions_fixed"]:
        raise ValueError("Source anchors must be fixed")
    if arrays["semantic_parts"].tolist() != ["left_hand", "right_hand"]:
        raise ValueError("Expected left/right R1 hand contact ordering")
    points = arrays["object_points_local"].astype(float)
    active = arrays["active"]
    normals = arrays["object_normals_local"].astype(float)
    source_half, target_half = (
        np.array(source_size) / 2,
        np.array(target_size) / 2,
    )
    if points.shape != (*active.shape, 3) or normals.shape != points.shape:
        raise ValueError("Invalid contact array shapes")
    expected = np.zeros_like(normals)
    expected[..., 0] = np.sign(normals[..., 0])
    if not np.allclose(normals[active], expected[active], atol=1e-7):
        raise ValueError("Expected contacts on opposing box X faces")
    # The source NPZ encodes a virtual normal preload. Recover its declared
    # surface anchors before adapting geometry; add no new closure preload.
    surface = points + normals * float(metadata["preload_mm"]) / 1000
    if not np.allclose(
        abs(surface[..., 0][active]), source_half[0], atol=1e-7
    ):
        raise ValueError("Source preload and surface geometry disagree")
    if np.any(abs(surface[..., 1:][active]) > target_half[1:] + 1e-7):
        raise ValueError("Original tangential anchors do not fit the new box")
    adapted = surface.copy()
    adapted[..., 0] = normals[..., 0] * target_half[0]
    assert np.array_equal(adapted[..., 1:][active], surface[..., 1:][active])
    indices = np.floor(
        np.arange(frames) / fps * metadata["fps"] + 1e-8
    ).astype(int)
    if indices[-1] >= len(active):
        raise ValueError("Contact timeline does not cover the robot reference")
    return dict(
        active=active[indices].copy(),
        object_points_local=adapted[indices],
        object_normals_local=normals[indices],
        robot_points_local=arrays["robot_points_local"].astype(float),
        robot_normals_local=arrays["robot_normals_local"].astype(float),
        source_frame=indices,
    ), metadata


class ContactEqualities:
    """Point coincidence and opposing normals with analytic Jacobians."""

    def __init__(self, model, contacts, selected):
        self.model, self.contacts = model, contacts
        self.data = mujoco.MjData(model)
        self.selected = np.asarray(selected)
        self.q_indices, self.v_indices = self.selected + 7, self.selected + 6
        self.bodies = [
            model.body(side + "_wrist_roll_link").id
            for side in ("left", "right")
        ]

    def configure(self, row, frame):
        self.row = row.copy()
        self.active = np.flatnonzero(self.contacts["active"][frame])
        self.rotation = Rotation.from_quat(row[[37, 38, 39, 36]]).as_matrix()
        self.targets = (
            row[33:36]
            + self.contacts["object_points_local"][frame] @ self.rotation.T
        )
        self.target_normals = (
            -self.contacts["object_normals_local"][frame] @ self.rotation.T
        )

    def geometry(self, q):
        self.data.qpos[:] = self.row
        self.data.qpos[self.q_indices] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        for hand in self.active:
            body = self.bodies[hand]
            rotation = self.data.xmat[body].reshape(3, 3)
            point = (
                self.data.xpos[body]
                + rotation @ self.contacts["robot_points_local"][hand]
            )
            normal = rotation @ self.contacts["robot_normals_local"][hand]
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jp, jr, point, body)
            dn = np.cross(jr[:, self.v_indices].T, normal).T
            yield hand, point, normal, jp[:, self.v_indices], dn

    def value(self, q):
        rows = []
        for hand, point, normal, _, _ in self.geometry(q):
            rows.extend((point - self.targets[hand]) * 100)
            rows.extend(self.rotation[:, 1:].T @ normal)
        return np.asarray(rows)

    def jacobian(self, q):
        rows = []
        for _, _, _, jp, dn in self.geometry(q):
            rows.extend(jp * 100)
            rows.extend(self.rotation[:, 1:].T @ dn)
        return np.asarray(rows)

    def hemisphere(self, q):
        return np.array(
            [
                normal @ self.target_normals[hand] - 0.5
                for hand, _, normal, _, _ in self.geometry(q)
            ]
        )

    def hemisphere_jacobian(self, q):
        return np.array(
            [
                self.target_normals[hand] @ dn
                for hand, _, _, _, dn in self.geometry(q)
            ]
        )

    def errors(self, q):
        values = []
        for hand, point, normal, _, _ in self.geometry(q):
            distance = np.linalg.norm(point - self.targets[hand])
            angle = np.rad2deg(
                np.arccos(np.clip(normal @ self.target_normals[hand], -1, 1))
            )
            values.append((hand, distance, angle))
        return values


def skew(value):
    x, y, z = value
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def antisymmetric_vector(matrix):
    return (
        np.array(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        )
        / 2
    )


class GroundedContactEqualities(ContactEqualities):
    """Allow pelvis translation while preserving both source foot poses."""

    def __init__(self, model, contacts):
        super().__init__(model, contacts, np.arange(24))
        self.q_indices = np.r_[np.arange(3), np.arange(7, 31)]
        self.v_indices = np.r_[np.arange(3), np.arange(6, 30)]
        self.feet = [
            model.body(side + "_ankle_roll_link").id
            for side in ("left", "right")
        ]

    def configure(self, row, frame):
        super().configure(row, frame)
        self.data.qpos[:] = row
        mujoco.mj_kinematics(self.model, self.data)
        self.foot_positions = self.data.xpos[self.feet].copy()
        self.foot_rotations = self.data.xmat[self.feet].reshape(2, 3, 3).copy()

    def foot_geometry(self, q):
        self.data.qpos[:] = self.row
        self.data.qpos[self.q_indices] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        for i, body in enumerate(self.feet):
            point = self.data.xpos[body].copy()
            relative = (
                self.data.xmat[body].reshape(3, 3) @ self.foot_rotations[i].T
            )
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jp, jr, point, body)
            orientation_jacobian = np.stack(
                [
                    antisymmetric_vector(skew(axis) @ relative)
                    for axis in jr[:, self.v_indices].T
                ],
                axis=1,
            )
            yield (
                i,
                point,
                relative,
                jp[:, self.v_indices],
                orientation_jacobian,
            )

    def value(self, q):
        rows = super().value(q).tolist()
        for i, point, relative, _, _ in self.foot_geometry(q):
            rows.extend((point - self.foot_positions[i]) * 100)
            rows.extend(antisymmetric_vector(relative))
        return np.asarray(rows)

    def jacobian(self, q):
        rows = super().jacobian(q).tolist()
        for _, _, _, jp, jr in self.foot_geometry(q):
            rows.extend(jp * 100)
            rows.extend(jr)
        return np.asarray(rows)

    def foot_errors(self, q):
        return [
            (
                float(np.linalg.norm(point - self.foot_positions[i])),
                float(
                    np.rad2deg(
                        np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
                    )
                ),
            )
            for i, point, relative, _, _ in self.foot_geometry(q)
        ]


def solve_reference(
    model,
    reference,
    contacts,
    *,
    waist_weight=1.0,
    smooth_weight=4.0,
    step_fraction=0.9,
):
    contract = robot_contract("r1")
    problem = GroundedContactEqualities(model, contacts)
    joint_limits = np.array(
        [model.joint(name).range for name in contract.joints[:24]]
    )
    step = (
        np.r_[np.full(3, 1.5), np.array(contract.velocity)[:24]]
        * 0.02
        * (1 if step_fraction is None else step_fraction)
    )
    weights = np.r_[
        np.full(3, 1000.0), np.ones(12), np.full(2, waist_weight), np.ones(10)
    ]
    result = reference.copy()
    measurements = []
    for frame, row in enumerate(reference):
        source = row[problem.q_indices]
        limits = np.vstack(
            [np.c_[row[:3] - 0.06, row[:3] + 0.06], joint_limits]
        )
        if frame:
            predicted = (
                result[frame - 1, problem.q_indices]
                + source
                - reference[frame - 1, problem.q_indices]
            )
            if step_fraction is not None:
                lower = np.maximum(
                    limits[:, 0] + 1e-7,
                    result[frame - 1, problem.q_indices] - step,
                )
                upper = np.minimum(
                    limits[:, 1] - 1e-7,
                    result[frame - 1, problem.q_indices] + step,
                )
            else:
                lower, upper = limits[:, 0] + 1e-7, limits[:, 1] - 1e-7
        else:
            predicted = source.copy()
            lower, upper = limits[:, 0] + 1e-7, limits[:, 1] - 1e-7
        initial = np.clip(predicted, lower, upper)
        problem.configure(row, frame)

        def objective(q, source=source, predicted=predicted):
            return np.sum(
                weights
                * ((q - source) ** 2 + smooth_weight * (q - predicted) ** 2)
            )

        def gradient(q, source=source, predicted=predicted):
            return 2 * weights * (q - source + smooth_weight * (q - predicted))

        # Feet remain constrained during approach and release as well.
        if len(problem.active) or len(problem.feet):
            constraints = [
                dict(type="eq", fun=problem.value, jac=problem.jacobian)
            ]
            if len(problem.active):
                constraints.append(
                    dict(
                        type="ineq",
                        fun=problem.hemisphere,
                        jac=problem.hemisphere_jacobian,
                    )
                )
            solution = minimize(
                objective,
                initial,
                jac=gradient,
                method="SLSQP",
                bounds=list(zip(lower, upper, strict=True)),
                constraints=constraints,
                options=dict(ftol=1e-10, maxiter=200),
            )
            q = solution.x
            errors = problem.errors(q)
            if any(
                distance > 0.0005 or angle > 0.5
                for _, distance, angle in errors
            ):
                for seed in (q, np.clip(source, lower, upper), initial):
                    feasible = least_squares(
                        problem.value,
                        np.clip(seed, lower + 1e-10, upper - 1e-10),
                        jac=problem.jacobian,
                        bounds=(lower, upper),
                        max_nfev=300,
                        ftol=1e-12,
                        xtol=1e-12,
                        gtol=1e-12,
                    )
                    q = feasible.x
                    errors = problem.errors(q)
                    if all(
                        distance <= 0.0005 and angle <= 0.5
                        for _, distance, angle in errors
                    ):
                        break
            if any(
                distance > 0.0005 or angle > 0.5
                for _, distance, angle in errors
            ):
                raise ValueError(
                    f"Contact constraints failed at frame {frame}: "
                    f"{solution.message}; {errors}"
                )
            measurements.extend((frame, *error) for error in errors)
            if any(
                distance > 1e-5 or angle > 0.01
                for distance, angle in problem.foot_errors(q)
            ):
                raise ValueError(
                    f"Source foot constraints failed at frame {frame}"
                )
        else:
            q = np.clip(
                (source + smooth_weight * predicted) / (1 + smooth_weight),
                lower,
                upper,
            )
        result[frame, problem.q_indices] = q
    return result, np.asarray(measurements)


def check_reader(manifest, urdf, contacts, output):
    """Verify contacts through float32 HDF5 and common-frame removal."""
    import torch

    from holomotion.src.env.reference_tracking.rotations import quat_to_mat
    from holomotion.src.env.reference_tracking.task import load_task

    _, motion, mdp = load_task(
        robot_urdf=urdf,
        manifest=manifest,
        output=output,
        device="cpu",
        profile="paper",
    )
    p, r, _, _ = mdp.kin.forward(
        *motion.ref_state(torch.arange(motion.num_frames)), all_bodies=True
    )
    box_rotation = quat_to_mat(motion.box_quat)
    errors, angles = [], []
    for hand, side in enumerate(("left", "right")):
        active = torch.as_tensor(contacts["active"][:, hand])
        body = mdp.kin.body_names.index(side + "_wrist_roll_link")
        local = torch.as_tensor(
            contacts["robot_points_local"][hand], dtype=torch.float32
        )
        normal = torch.as_tensor(
            contacts["robot_normals_local"][hand], dtype=torch.float32
        )
        point = p[:, body] + (r[:, body] @ local[:, None]).squeeze(-1)
        targets = motion.box_p + (
            box_rotation
            @ torch.as_tensor(
                contacts["object_points_local"][:, hand], dtype=torch.float32
            ).unsqueeze(-1)
        ).squeeze(-1)
        n = (r[:, body] @ normal[:, None]).squeeze(-1)
        desired = -(
            box_rotation
            @ torch.as_tensor(
                contacts["object_normals_local"][:, hand], dtype=torch.float32
            ).unsqueeze(-1)
        ).squeeze(-1)
        errors.extend((point - targets).norm(dim=-1)[active].tolist())
        cosine = (n * desired).sum(-1).clamp(-1, 1)
        angles.extend(torch.rad2deg(torch.acos(cosine))[active].tolist())
    return dict(
        maximum_point_error_m=max(errors),
        maximum_normal_error_deg=max(angles),
        active_hand_frames=len(errors),
        passed=max(errors) <= 0.0005 and max(angles) <= 0.5,
    )


def build(
    source_reference,
    source_manifest,
    contact_path,
    stance_capture,
    holosoma_root,
    urdf,
    output,
    training_size=(0.24, 0.245, 0.27),
    evaluation_size=(0.305, 0.245, 0.27),
):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(source_manifest.read_text())
    with np.load(source_reference, allow_pickle=False) as package:
        raw = {key: package[key].copy() for key in package.files}
    contract = robot_contract("r1")
    if (
        raw["joint_names"].tolist() != contract.joints
        or float(raw["fps"]) != 50
    ):
        raise ValueError("Expected the audited 50 Hz named R1 reference")
    reference = np.c_[
        raw["joint_pos"], raw["object_pos_w"], raw["object_quat_w"]
    ]
    original = reference.copy()
    contacts, source_contact_metadata = contact_schedule(
        contact_path, len(reference), metadata["box"]["size_m"], training_size
    )
    rotations = Rotation.from_quat(reference[:, [37, 38, 39, 36]]).as_matrix()
    reference[:, 35] += abs(rotations[:, 2]) @ (
        (np.array(training_size) - metadata["box"]["size_m"]) / 2
    )
    model = scene_model(
        urdf, output / "audit", training_size, metadata["box"]["mass_kg"]
    )
    adjusted, measurements = solve_reference(model, reference, contacts)
    np.savez_compressed(
        output / "task_contact_audit.npz", measurements=measurements
    )
    sys.path.insert(0, str(holosoma_root / "tools"))
    from prepend_r1_measured_stance_transition import _load_stance

    q, rpy, capture = _load_stance(stance_capture, contract.joints, 10)
    if (
        capture["capture_metadata"]["capture_condition"]
        != "grounded_official_stance"
    ):
        raise ValueError("Expected official grounded Stance capture")
    complete, prefix = prepend_stance(
        model,
        adjusted,
        q,
        rpy,
        near_feet_gap_m=0.005,
        evaluation_size=evaluation_size,
    )
    count = prefix["prefix_frames"]
    padded = dict(contacts)
    for key in ("active", "object_points_local", "object_normals_local"):
        a = contacts[key]
        padded[key] = np.concatenate(
            [
                np.zeros((count, *a.shape[1:]), dtype=a.dtype),
                a,
                np.repeat(a[-1:], 25, axis=0),
            ]
        )
    padded["source_frame"] = np.r_[
        np.full(count, -1),
        contacts["source_frame"],
        np.repeat(contacts["source_frame"][-1], 25),
    ]
    np.savez_compressed(output / "fixed_contacts_50hz.npz", **padded)
    np.savez_compressed(
        output / "reference_50hz.npz",
        fps=50,
        joint_names=contract.joints,
        joint_pos=complete[:, :33],
        object_pos_w=complete[:, 33:36],
        object_quat_w=complete[:, 36:40],
    )
    manifest = prepare(
        output / "reference_50hz.npz",
        output / "dataset",
        "r1",
        training_size,
        metadata["box"]["mass_kg"],
        clip_name=next(iter(metadata["clips"])),
    )
    manifest.update(
        box_axis_order=list(R1_BOX_AXES),
        evaluation_box=dict(manifest["box"], size_m=list(evaluation_size)),
    )
    manifest["provenance"].update(
        standing_transition=prefix,
        stance_capture_sha256=fingerprint(stance_capture),
        fixed_contacts=dict(
            file="../fixed_contacts_50hz.npz",
            sha256=fingerprint(output / "fixed_contacts_50hz.npz"),
            source_sha256=fingerprint(contact_path),
            definition=(
                "Original surface Y/Z and normals fixed; "
                "only opposing X faces follow box width"
            ),
        ),
    )
    atomic_json(output / "dataset/manifest.json", manifest)
    reader = check_reader(
        output / "dataset/manifest.json", urdf, padded, output / "reader_audit"
    )
    audit = audit_motion(model, complete, count)
    # Explicitly inspect all robot/box collisions in the source task as well.
    data = mujoco.MjData(model)
    box = model.geom("grip_box_surface").id
    unwanted = []
    feet_errors = []
    foot_problem = GroundedContactEqualities(model, contacts)
    for frame, row in enumerate(adjusted):
        foot_problem.configure(reference[frame], frame)
        feet_errors.extend(
            foot_problem.foot_errors(row[foot_problem.q_indices])
        )
        data.qpos[:] = row
        mujoco.mj_fwdPosition(model, data)
        for contact in data.contact:
            if box not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom1 if contact.geom2 == box else contact.geom2
            body = model.body(model.geom_bodyid[other]).name
            if body not in (
                "left_wrist_roll_link",
                "right_wrist_roll_link",
                "world",
            ):
                unwanted.append(float(contact.dist))
    report = dict(
        **audit,
        checks_passed=bool(
            audit["kinematic_checks_passed"]
            and reader["passed"]
            and min(unwanted, default=0) >= -0.003
        ),
        source_reference=str(source_reference),
        source_contact_metadata=source_contact_metadata,
        fixed_contact_reader_audit=reader,
        maximum_task_point_error_m=float(measurements[:, 2].max()),
        maximum_task_normal_error_deg=float(measurements[:, 3].max()),
        maximum_nonhand_box_penetration_m=max(-min(unwanted, default=0), 0),
        maximum_source_foot_position_error_m=max(x[0] for x in feet_errors),
        maximum_source_foot_rotation_error_deg=max(x[1] for x in feet_errors),
        original_box_size_m=metadata["box"]["size_m"],
        training_box_size_m=list(training_size),
        evaluation_box_size_m=list(evaluation_size),
        box_axis_order=list(R1_BOX_AXES),
        standing_transition=prefix,
        frames=len(complete),
        duration_s=(len(complete) - 1) / 50,
        measured_stance_first_joint_pose_exact=bool(
            np.array_equal(complete[0, 7:33], q)
        ),
        source_task_root_legs_head_preserved=bool(
            np.array_equal(adjusted[:, :19], original[:, :19])
            and np.array_equal(adjusted[:, 31:33], original[:, 31:33])
        ),
        source_task_object_xy_orientation_preserved=bool(
            np.array_equal(adjusted[:, 33:35], original[:, 33:35])
            and np.array_equal(adjusted[:, 36:40], original[:, 36:40])
        ),
        source_task_base_orientation_head_preserved=bool(
            np.array_equal(adjusted[:, 3:7], original[:, 3:7])
            and np.array_equal(adjusted[:, 31:33], original[:, 31:33])
        ),
        maximum_pelvis_translation_m=float(
            np.linalg.norm(adjusted[:, :3] - original[:, :3], axis=-1).max()
        ),
        corrected_joints=contract.joints[:24],
        maximum_waist_correction_rad=float(
            abs(adjusted[:, 19:21] - original[:, 19:21]).max()
        ),
        maximum_arm_correction_rad=float(
            abs(adjusted[:, 21:31] - original[:, 21:31]).max()
        ),
        additional_preload_m=0,
        interaction_rewards=False,
        dynamics_validated=False,
        contact_guarantee_scope=(
            "Offline reference points and normals only; "
            "learned physical contacts require evaluation"
        ),
        preprocessing_source_sha256={
            "holomotion/scripts/training/" + filename: fingerprint(
                Path(__file__).parent / filename
            )
            for filename in (
                "prepare_r1_fixed_contacts.py",
                "prepare_r1_stance_grip.py",
                "prepare_supermimic_omomo.py",
            )
        },
        input_sha256={
            str(p): fingerprint(p)
            for p in (
                source_reference,
                source_manifest,
                contact_path,
                contact_path.with_suffix(".json"),
                stance_capture,
                urdf,
                holosoma_root
                / "tools/prepend_r1_measured_stance_transition.py",
                holosoma_root / "tools/prepend_r1_standing_transition.py",
            )
        },
    )
    atomic_json(output / "validation.json", report)
    if not report["checks_passed"]:
        raise ValueError(
            "Fixed contact reference audit failed; see validation.json"
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-reference",
        "source-manifest",
        "contacts",
        "stance-capture",
        "holosoma-root",
        "robot-urdf",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.source_reference.resolve(),
                args.source_manifest.resolve(),
                args.contacts.resolve(),
                args.stance_capture.resolve(),
                args.holosoma_root.resolve(),
                args.robot_urdf.resolve(),
                args.output.resolve(),
            ),
            indent=2,
        )
    )
