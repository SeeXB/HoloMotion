"""Offline R1 actor with explicit activation and hardware-free checks."""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from holomotion.src.env.reference_tracking.r1_box_contract import (
    R1_BOX_AXES,
    R1_REAL_BOX_SIZE_M,
    validate_size,
)
from holomotion.src.env.reference_tracking.robot_contract import robot_contract


def rotation_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    if (
        quat.shape != (4,)
        or not np.isfinite(quat).all()
        or abs(np.linalg.norm(quat) - 1) > 0.01
    ):
        raise ValueError("Expected a finite normalized WXYZ IMU quaternion")
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )


def observation(
    config, quat, joint_pos, joint_vel, body_gyro, previous, root_reference
):
    rotation = rotation_wxyz(quat)
    pieces = [
        joint_pos - np.asarray(config["default_joint_pos"]),
        joint_vel,
        body_gyro,
        -rotation[2],
        previous,
    ]
    if config["reference_orientation"]:
        relative = rotation.T @ root_reference
        # Same column-based 6D representation as training mat_to_6d.
        pieces.append(relative[:, :2].reshape(-1))
    value = np.concatenate(pieces).astype(np.float32)
    if value.shape != (config["obs_dim"],) or not np.isfinite(value).all():
        raise ValueError("Invalid policy observation")
    return value


def validate_box_profile(config):
    """Validate the exported geometry against its own evaluated recipe."""
    profile = config.get("box_profile")
    if profile is None:
        return
    if tuple(profile["axis_order"]) != R1_BOX_AXES:
        raise ValueError("Invalid box axis order")
    real = validate_size(config["box"]["size_m"])
    training = validate_size(config["training_box"]["size_m"])
    np.testing.assert_allclose(real, R1_REAL_BOX_SIZE_M, atol=1e-8)
    np.testing.assert_allclose(training[1:], real[1:], atol=1e-8)
    if training[0] > real[0]:
        raise ValueError("Training box must not be wider than the real box")
    for key, expected in (
        ("training_box", training),
        ("evaluation_box", real),
    ):
        np.testing.assert_allclose(profile[key]["size_m"], expected, atol=1e-8)
    evaluation = config.get("simulation_evaluation")
    if evaluation is not None:
        if (
            evaluation["checkpoint_sha256"]
            != config["source_checkpoint_sha256"]
        ):
            raise ValueError(
                "Box evaluation belongs to a different checkpoint"
            )
        for key in ("training_box", "evaluation_box"):
            if evaluation["box_profile"][key] != profile[key]:
                raise ValueError("Box profile differs from native evaluation")


def policy_session(directory, config):
    import onnxruntime as ort

    provider = config.get("execution_provider", "CPUExecutionProvider")
    if provider not in {"CPUExecutionProvider", "CUDAExecutionProvider"}:
        raise ValueError(f"Unsupported inference provider: {provider}")
    if provider not in ort.get_available_providers():
        raise RuntimeError(
            f"The bundle requires {provider}, which is unavailable"
        )
    if provider == "CUDAExecutionProvider":
        # Load the task's CUDA libraries before ORT creates its provider.
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("The BF16 CUDA bundle requires a visible GPU")
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(cuda=True, cudnn=True, msvc=False)
    options = ort.SessionOptions()
    if config.get("onnx_graph_optimization") == "disabled":
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
    session = ort.InferenceSession(
        str(directory / "policy.onnx"),
        sess_options=options,
        providers=[provider],
    )
    if session.get_providers()[0] != provider:
        raise RuntimeError(
            "The requested inference provider failed to initialize"
        )
    session.disable_fallback()
    return session


class OfflineActor:
    def __init__(self, directory):
        self.config = json.loads((directory / "manifest.json").read_text())
        cfg = self.config
        validate_box_profile(cfg)
        contract = robot_contract("r1")
        if (
            cfg["format"] != "supermimic_offline_v1"
            or cfg["robot"] != "r1"
            or cfg["joint_names"] != contract.joints
        ):
            raise ValueError("Expected an R1 26-DOF SuperMimic bundle")
        if cfg["control_dt"] != 0.02 or cfg["quaternion_order"] != "wxyz":
            raise ValueError("Invalid policy clock or quaternion contract")
        for name, digest in cfg["files"].items():
            path = (directory / name).resolve()
            if (
                not path.is_relative_to(directory.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError(f"Artifact checksum mismatch: {name}")
        for name, expected in (
            ("kp", contract.kp),
            ("kd", contract.kd),
            ("action_scale", contract.action_scale),
        ):
            np.testing.assert_allclose(cfg[name], expected, atol=1e-7)
        with np.load(
            directory / "reference.npz", allow_pickle=False
        ) as reference:
            self.goals = reference["goals"].copy()
            self.first_q = reference["first_joint_pos"].copy()
        self.root_rotation = np.load(
            directory / "root_rotation.npy", allow_pickle=False
        )
        if (
            self.goals.shape[1:] != (cfg["goal_dim"],)
            or not np.isfinite(self.goals).all()
        ):
            raise ValueError("Invalid reference goals")
        self.session = policy_session(directory, cfg)
        if [(v.name, v.shape) for v in self.session.get_inputs()] != [
            ("obs_hist", [1, cfg["history"], cfg["obs_dim"]]),
            ("goal", [1, cfg["goal_dim"]]),
        ]:
            raise ValueError("ONNX actor inputs disagree with bundle")
        self.frame = 0
        self.previous = np.zeros(26, dtype=np.float32)
        self.history = None

    def step(self, state, *, advance=True):
        cfg = self.config
        state = np.asarray(state).reshape(-1)
        if state.shape != (65,) or not np.isfinite(state).all():
            raise ValueError("Expected finite dense R1 state (65,)")
        obs = observation(
            cfg,
            state[3:7],
            state[7:33],
            state[39:65],
            state[36:39],
            self.previous,
            self.root_rotation[self.frame],
        )
        self.history = (
            np.repeat(obs[None], cfg["history"], axis=0)
            if self.history is None
            else np.concatenate((self.history[1:], obs[None]))
        )
        action = self.session.run(
            None,
            dict(
                obs_hist=self.history[None],
                goal=self.goals[self.frame : self.frame + 1],
            ),
        )[0].reshape(-1)
        if action.shape != (26,) or not np.isfinite(action).all():
            raise ValueError("Invalid policy action")
        target = (
            np.asarray(cfg["default_joint_pos"])
            + np.asarray(cfg["action_scale"]) * action
        )
        # Fail closed: do not silently execute a clipped, unvalidated policy.
        if np.any(target < np.asarray(cfg["joint_lower"])) or np.any(
            target > np.asarray(cfg["joint_upper"])
        ):
            raise ValueError("Policy PD target exceeds robot limits")
        self.previous = action
        if advance:
            self.frame = min(self.frame + 1, len(self.goals) - 1)
        return target


def check(directory):
    actor = OfflineActor(directory)
    state = np.zeros(65)
    state[3] = 1.0
    state[7:33] = actor.first_q
    started = time.monotonic()
    for _ in range(30):
        actor.step(state)
    return dict(
        hardware_initialized=False,
        motors_started=False,
        finite_inferences=30,
        mean_inference_ms=(time.monotonic() - started) / 30 * 1000,
        robot="r1",
        training_box=actor.config.get("training_box"),
        deployment_box=actor.config.get("box"),
        box_axis_order=["left_right", "front_back", "up"],
        inference_precision=actor.config.get("inference_precision", "fp32"),
        execution_provider=actor.config.get(
            "execution_provider", "CPUExecutionProvider"
        ),
        simulation_passed=actor.config.get("simulation_passed", False),
        exported_simulation_passed=actor.config.get(
            "exported_simulation_passed", False
        ),
        real_robot_validated=False,
    )


def run(directory, holosoma_root, interface):
    actor = OfflineActor(directory)
    if not (
        actor.config.get("simulation_passed", False)
        and actor.config.get("exported_simulation_passed", False)
    ):
        raise RuntimeError(
            "This checkpoint has not passed native and exported "
            "complete-clip box-lift simulation"
        )
    if actor.config.get("numerical_precision_passed") is False:
        raise RuntimeError(
            "Exported actions have not matched native policy precision"
        )
    if actor.config.get("startup", {}).get("reference_start") != (
        "grounded_official_stance"
    ):
        raise RuntimeError(
            "Active R1 bundles must start at grounded official Stance"
        )
    # All SDK imports and construction are deliberately below the check branch.
    sys.path.insert(0, str(holosoma_root / "src/holosoma_inference"))
    from holosoma_inference.config.config_values.r1 import r1_26dof
    from holosoma_inference.realworld.r1_startup import (
        request_official_stance,
        wait_for_grounded_stance_confirmation,
    )
    from holosoma_inference.sdk.unitree.r1_sdk2_cpp_interface import (
        R1Sdk2CppInterface,
    )

    cfg = actor.config
    robot_cfg = replace(
        r1_26dof, motor_kp=tuple(cfg["kp"]), motor_kd=tuple(cfg["kd"])
    )
    # Retain the official RPC client until cleanup. Request success is checked
    # before constructing the read-only bridge or relinquishing control.
    official = request_official_stance(0, interface)
    hardware = None
    try:
        hardware = R1Sdk2CppInterface(
            robot_cfg, interface_str=interface, defer_low_level_takeover=True
        )
        # Warm ONNX while the official controller owns the grounded robot.
        for _ in range(5):
            actor.step(hardware.get_low_state(), advance=False)
        actor.previous[:] = 0
        actor.history = None
        wait_for_grounded_stance_confirmation(hardware)
        hardware.activate_low_level(
            takeover_blend_seconds=0.2, authority_settle_seconds=1.0
        )
        initial = hardware.get_low_state().reshape(-1)[7:33].copy()
        zeros = np.zeros(26)
        print(
            "模型闭环保持官方站姿。松开后再按 A 播放；B 或 L1+R1 停止。",
            flush=True,
        )
        stream_policy(hardware, actor, initial, zeros)
    finally:
        try:
            if hardware is not None and hardware.low_level_active:
                hardware.enter_damping()
        finally:
            if hardware is not None:
                hardware.close()
            del official


def stream_policy(hardware, actor, initial, zeros):
    """Refresh feedback while holding frame zero and after the final frame."""
    playing, finished, released, pressed = False, False, False, False
    takeover = time.monotonic()
    while True:
        started = time.monotonic()
        message = hardware.get_joystick_msg()
        key = (
            hardware.get_joystick_key(message) if message is not None else None
        )
        if key in {"B", "L1+R1"} or "select" in (key or "").lower():
            return
        if not playing and not finished and message is not None:
            if message.keys == 0:
                if pressed:
                    playing = True
                released = True
            elif released and key == "A":
                pressed = True
        state = hardware.get_low_state()
        target = actor.step(state, advance=playing)
        alpha = np.clip((started - takeover) / 0.2, 0, 1)
        blend = 6 * alpha**5 - 15 * alpha**4 + 10 * alpha**3
        target = (1 - blend) * initial + blend * target
        if time.monotonic() - started > 0.02:
            raise RuntimeError("Policy missed its 50-Hz deadline")
        hardware.send_low_command(target, zeros, zeros)
        if playing and actor.frame == len(actor.goals) - 1:
            playing, finished = False, True
            print("动作结束，模型闭环保持末帧；B 或 L1+R1 停止。", flush=True)
        time.sleep(max(0.0, 0.02 - (time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--holosoma-root", type=Path)
    parser.add_argument("--interface")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check(args.bundle), indent=2))
        return
    if args.holosoma_root is None or not args.interface:
        parser.error("Active runtime requires --holosoma-root and --interface")
    run(args.bundle, args.holosoma_root, args.interface)


if __name__ == "__main__":
    main()
