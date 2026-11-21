"""Export a compact actor and its offline reference contract."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from holomotion.src.algo.supermimic.networks import Policy
from holomotion.src.env.reference_tracking.r1_box_contract import (
    use_evaluation_box,
)
from holomotion.src.env.reference_tracking.task import load_task


def export(checkpoint, output):
    import onnxruntime as ort

    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty export directory")
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    spec = bundle["specification"]
    args = spec["arguments"]
    repository = Path(__file__).resolve().parents[3]
    # Physics and data-production code may evolve after a completed run.
    # Preserve that run's source archive and require the executing actor,
    # kinematics and reference reconstruction implementations to match.
    reconstruction_sources = {
        name: digest
        for name, digest in spec["source_sha256"].items()
        if (
            name.startswith("holomotion/src/env/reference_tracking/")
            and Path(name).name
            not in {"box.py", "isaac_backend.py", "warp_backend.py"}
        )
        or name
        in {
            "holomotion/src/algo/supermimic/networks.py",
            "holomotion/src/training/h5_dataloader.py",
            "holomotion/src/training/reference_fk.py",
        }
    }
    for path, expected in [
        *(
            (Path(name), digest)
            for name, digest in spec["input_sha256"].items()
        ),
        *(
            (checkpoint.parent / "source_snapshot" / name, digest)
            for name, digest in spec["source_sha256"].items()
        ),
        *(
            (repository / name, digest)
            for name, digest in reconstruction_sources.items()
        ),
    ]:
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(
                f"Training source or input changed before export: {path}"
            )
    output.mkdir(parents=True, exist_ok=True)
    _, motion, mdp = load_task(
        robot_urdf=Path(args["robot_urdf"]),
        manifest=Path(args["manifest"]),
        output=output / "kinematics",
        device="cpu",
        profile=args["profile"],
    )
    training_box = dict(motion.box) if motion.box else None
    box_profile = use_evaluation_box(motion)
    if len(motion.names) != 1:
        raise ValueError(
            "Offline deployment currently requires exactly one clip"
        )
    n, k, hist = mdp.num_j, args["goal_frames"], args["hist"]
    policy = Policy(mdp.obs_dim, (15 + 2 * n) * k, hist=hist, num_j=n).eval()
    policy.load_state_dict(bundle["pi"], strict=True)
    frames = torch.arange(motion.num_frames)
    goals = motion.goal(frames, k)
    observations = mdp.obs(
        motion.ref_state(frames),
        torch.zeros(len(frames), n),
        torch.zeros(len(frames), n + 4),
        frames,
    )
    inputs = (observations[:1, None].repeat(1, hist, 1), goals[:1])
    path = output / "policy.onnx"
    torch.onnx.export(
        policy,
        inputs,
        str(path),
        input_names=["obs_hist", "goal"],
        output_names=["action"],
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    maximum = 0.0
    # Test varied physical observations, previous actions and reference phases.
    generator = torch.Generator().manual_seed(42)
    for i in np.linspace(0, len(frames) - 1, 20, dtype=int):
        history = observations[i : i + 1, None].repeat(1, hist, 1)
        history = history + 0.03 * torch.randn(
            history.shape, generator=generator
        )
        goal = goals[i : i + 1]
        with torch.no_grad():
            expected = policy(history, goal).numpy()
        actual = session.run(
            None, dict(obs_hist=history.numpy(), goal=goal.numpy())
        )[0]
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    contract = mdp.kin.contract
    evaluation_path = checkpoint.parent / "evaluation_final.json"
    evaluation = (
        json.loads(evaluation_path.read_text())
        if evaluation_path.is_file()
        else None
    )
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    simulation_passed = bool(
        evaluation
        and evaluation.get("checkpoint_sha256") == checkpoint_digest
        and evaluation.get("failed_clips") == 0
        and evaluation.get("evaluated_clips") == len(motion.names)
        and all(
            clip.get("box_lift", {}).get("lifted", False)
            for clip in evaluation.get("clips", [])
        )
    )
    np.savez_compressed(
        output / "reference.npz",
        goals=goals.numpy(),
        first_joint_pos=motion.q[0].numpy(),
    )
    manifest = dict(
        format="supermimic_offline_v1",
        robot=contract.name,
        joint_names=contract.joints,
        quaternion_order="wxyz",
        control_dt=0.02,
        history=hist,
        obs_dim=mdp.obs_dim,
        goal_dim=goals.shape[-1],
        reference_orientation=args["profile"] == "vibed",
        default_joint_pos=contract.default,
        kp=contract.kp,
        kd=contract.kd,
        action_scale=contract.action_scale,
        joint_lower=mdp.kin.q_lo.tolist(),
        joint_upper=mdp.kin.q_hi.tolist(),
        source_checkpoint_sha256=checkpoint_digest,
        training_source_sha256=spec["source_sha256"],
        reconstruction_source_sha256=reconstruction_sources,
        simulation_passed=simulation_passed,
        simulation_evaluation=evaluation,
        onnx_max_abs_error=maximum,
        parity_samples=20,
        files={
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("policy.onnx", "reference.npz")
        },
        box=motion.box,
        training_box=training_box,
        box_profile=box_profile,
        real_robot_validated=False,
    )
    reference_manifest = json.loads(Path(args["manifest"]).read_text())
    provenance = reference_manifest.get("provenance", {})
    if provenance.get("stance_capture_sha256"):
        manifest["startup"] = dict(
            reference_start="grounded_official_stance",
            stance_capture_sha256=provenance["stance_capture_sha256"],
            standing_transition=provenance["standing_transition"],
            hold_control="Live policy feedback with reference clock frozen",
        )
    # The optional orientation profile also needs each current root target.
    np.save(output / "root_rotation.npy", motion.root_R.numpy())
    manifest["files"]["root_rotation.npy"] = hashlib.sha256(
        (output / "root_rotation.npy").read_bytes()
    ).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(export(options.checkpoint, options.output), indent=2))
