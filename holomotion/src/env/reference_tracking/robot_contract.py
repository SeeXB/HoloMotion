"""Named robot and actuator contracts for compact imitation policies."""

from dataclasses import dataclass

from holomotion.src.env.reference_tracking import robot as g1


@dataclass(frozen=True)
class RobotContract:
    name: str
    joints: list[str]
    tracked: list[str]
    root: str
    torso: str
    default: list[float]
    kp: list[float]
    kd: list[float]
    effort: list[float]
    velocity: list[float]
    armature: list[float]

    @property
    def action_scale(self):
        return [
            0.25 * e / p for e, p in zip(self.effort, self.kp, strict=True)
        ]


def robot_contract(name="g1"):
    if name == "g1":
        motors = [g1.motor_of(j) for j in g1.G1_JOINT_NAMES]
        return RobotContract(
            name,
            g1.G1_JOINT_NAMES,
            g1.TRACKED_BODIES,
            "pelvis",
            "torso_link",
            g1.default_joint_pos_list(),
            [g1._kp(m[2]) for m in motors],
            [g1._kd(m[2]) for m in motors],
            [m[0] for m in motors],
            [m[1] for m in motors],
            [m[2] for m in motors],
        )
    if name != "r1":
        raise ValueError(f"Unsupported compact-policy robot: {name}")
    # HoloSoma R1 26-DOF SDK order, including the two head motors.
    joints = [
        *(
            side + "_" + part + "_joint"
            for side in ("left", "right")
            for part in (
                "hip_pitch",
                "hip_roll",
                "hip_yaw",
                "knee",
                "ankle_pitch",
                "ankle_roll",
            )
        ),
        "waist_roll_joint",
        "waist_yaw_joint",
        *(
            side + "_" + part + "_joint"
            for side in ("left", "right")
            for part in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
            )
        ),
        "head_pitch_joint",
        "head_yaw_joint",
    ]
    tracked = [
        "pelvis_link",
        "waist_yaw_link",
        "head_pitch_link",
        "head_yaw_link",
        *(
            side + "_" + part + "_link"
            for side in ("left", "right")
            for part in (
                "hip_roll",
                "knee",
                "ankle_roll",
                "shoulder_roll",
                "elbow",
                "wrist_roll",
            )
        ),
    ]
    return RobotContract(
        name,
        joints,
        tracked,
        "pelvis_link",
        "waist_yaw_link",
        [-0.1, 0, 0, 0.3, -0.2, 0] * 2
        + [0, 0]
        + [0.35, 0.18, 0, 0.87, 0, 0.35, -0.18, 0, 0.87, 0, 0, 0],
        [100] * 4
        + [40] * 2
        + [100] * 4
        + [40] * 2
        + [100] * 2
        + [40, 40, 20, 20, 20] * 2
        + [50, 10],
        [2] * 14 + [2, 2, 1, 1, 1] * 2 + [2, 0.1],
        [60] * 4
        + [50] * 2
        + [60] * 4
        + [50] * 2
        + [60] * 2
        + [60, 60, 33, 33, 33] * 2
        + [33, 33],
        [18.8] * 4
        + [30] * 2
        + [18.8] * 4
        + [30] * 2
        + [18.8] * 2
        + [18.8, 18.8, 33.4, 33.4, 33.4] * 2
        + [33.4] * 2,
        [0.01] * 26,
    )
