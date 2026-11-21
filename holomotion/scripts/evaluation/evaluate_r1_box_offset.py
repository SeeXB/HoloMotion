"""Evaluate a saved R1 actor with the physical box initially nearer the robot.

This is a separate placement experiment. Robot references, weights, box size,
orientation and height stay fixed; only the box's reset XY position changes.
"""

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from holomotion.src.utils.artifacts import atomic_json, fingerprint


def move_box_toward_robot(motion, distance):
    """Translate only reset frames along the negative initial robot heading."""
    if not math.isfinite(distance) or distance < 0:
        raise ValueError(
            "The distance toward the robot must be finite and nonnegative"
        )
    ids = motion.clip_start
    forward = motion.root_R[ids, :2, 0]
    norm = forward.norm(dim=-1, keepdim=True)
    if (norm < 1e-6).any():
        raise ValueError("Initial robot heading has no horizontal component")
    shift = -distance * forward / norm
    before = motion.box_p[ids].clone()
    motion.box_p = motion.box_p.clone()
    motion.box_p[ids, :2] += shift
    return dict(
        toward_robot_m=distance,
        direction="negative initial robot heading, projected onto the floor",
        initial_center_before_m=before.cpu().tolist(),
        initial_center_after_m=motion.box_p[ids].cpu().tolist(),
        initial_center_xy_shift_m=shift.cpu().tolist(),
        height_unchanged=True,
        orientation_unchanged=True,
        robot_reference_unchanged=True,
        box_pose_replayed=False,
    )


def evaluate_offset(options, *, launched=None):
    """Use the native full-clip scorer with an explicit reset intervention."""
    from holomotion.src.algo.supermimic import networks
    from holomotion.src.env.reference_tracking.r1_box_contract import (
        use_evaluation_box,
    )
    from holomotion.src.env.reference_tracking.task import load_task
    from holomotion.src.evaluation.reference_tracking.isaac_evaluation import (
        IsaacFullClipEvaluator,
    )

    started = time.monotonic() if launched is None else launched
    torch.manual_seed(options.seed)
    torch.set_num_threads(8)
    torch.cuda.set_device(options.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    networks.AMP_DTYPE = torch.bfloat16 if options.amp == "bf16" else None
    checkpoint = options.output.parent / "checkpoint_final.pt"
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Resolve relative source paths against the runtime used to reconstruct
    # this actor, including archived runtimes for ongoing training jobs.
    from holomotion.src.algo.supermimic import evaluation as native_evaluation

    repository = Path(native_evaluation.__file__).resolve().parents[4]
    specification = bundle["specification"]
    inputs = [
        (repository / name, digest)
        for name, digest in specification["source_sha256"].items()
    ] + [
        (Path(name), digest)
        for name, digest in specification["input_sha256"].items()
    ]
    checkpoint_sha = fingerprint(checkpoint)
    experiment_source = Path(__file__).resolve()
    experiment_sha = fingerprint(experiment_source)
    inputs += [
        (checkpoint, checkpoint_sha),
        (experiment_source, experiment_sha),
    ]

    def verify_contract():
        if any(not p.is_file() or fingerprint(p) != sha for p, sha in inputs):
            raise ValueError("Evaluation source, input or checkpoint changed")

    verify_contract()
    model, motion, mdp = load_task(
        robot_urdf=options.robot_urdf,
        manifest=options.manifest,
        output=options.output,
        device=options.device,
        profile=options.profile,
    )
    profile = use_evaluation_box(motion)
    if profile is None:
        raise ValueError(
            "This evaluation requires the explicit R1 real-size box profile"
        )
    placement = move_box_toward_robot(motion, options.toward_robot_m)
    profile["box_center_xy_unchanged"] = options.toward_robot_m == 0
    profile["initial_placement"] = placement
    policy = networks.Policy(
        mdp.obs_dim,
        (15 + 2 * mdp.num_j) * options.goal_frames,
        options.hist,
        num_j=mdp.num_j,
    ).to(options.device)
    policy.load_state_dict(bundle["pi"], strict=True)
    evaluator = IsaacFullClipEvaluator(
        model,
        motion,
        mdp,
        device=options.device,
        history_length=options.hist,
        future_frames=options.goal_frames,
        robot_urdf=options.robot_urdf,
        cache_dir=options.output / "isaac_usd",
    )
    print(
        json.dumps(
            dict(phase="box_offset_evaluation_started", placement=placement)
        ),
        flush=True,
    )
    result = evaluator.evaluate(policy)
    verify_contract()
    trace_path = options.output / "rollout_final.pt"
    evaluator.last_trace["initial_placement"] = placement
    torch.save(evaluator.last_trace, trace_path)
    result.update(
        scope=(
            "Separate real-size box placement experiment; "
            "unchanged robot reference and actor"
        ),
        evaluation_protocol=dict(
            mode="r1_box_position_offset_v1",
            base_mode="native",
            placement=placement,
        ),
        new_evaluation_protocol=True,
        baseline_rebuild_required=True,
        box_profile=profile,
        label=f"actor_{bundle['schedule']['actor_updates']:06d}_box_offset",
        rollout_file=trace_path.name,
        rollout_sha256=fingerprint(trace_path),
        checkpoint_sha256=checkpoint_sha,
        experiment_source_sha256=experiment_sha,
        training_wall_s=bundle["training_wall_s"],
        **{
            key: bundle["schedule"][key]
            for key in ("actor_updates", "world_updates", "vector_steps")
        },
        evaluation_process_wall_s=time.monotonic() - started,
    )
    return result


def preview(checkpoint, output, toward_robot_m):
    from holomotion.src.algo.supermimic.runtime import _supervise

    checkpoint, output = checkpoint.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    specification = bundle["specification"]
    arguments = specification["arguments"]
    source_sha = fingerprint(checkpoint)
    del bundle
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoint_final.pt").symlink_to(checkpoint)
    (output / "source_snapshot").symlink_to(
        checkpoint.parent / "source_snapshot", target_is_directory=True
    )
    atomic_json(output / "specification.json", specification)
    options = SimpleNamespace(
        **{
            **arguments,
            "output": output / "evaluation",
            "manifest": Path(arguments["manifest"]),
            "robot_urdf": Path(arguments["robot_urdf"]),
            "toward_robot_m": toward_robot_m,
        }
    )
    result = _supervise(options, training_operation=evaluate_offset)
    result.update(
        source_checkpoint=str(checkpoint),
        final_policy_evaluation=False,
        evaluation_scope=(
            "Separate box reset position comparison; no training changes"
        ),
        rollout_file="evaluation/rollout_final.pt",
    )
    atomic_json(output / "evaluation_final.json", result)
    (output / "g1.xml").symlink_to(output / "evaluation/g1.xml")
    if fingerprint(checkpoint) != source_sha:
        raise ValueError("The source checkpoint changed during evaluation")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--toward-robot-m", type=float, default=0.02)
    args = parser.parse_args()
    print(
        json.dumps(
            preview(args.checkpoint, args.output, args.toward_robot_m),
            indent=2,
        )
    )
