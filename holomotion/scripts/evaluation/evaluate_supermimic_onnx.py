"""Evaluate the exported R1 actor inside the same dynamic-box Isaac task."""

import argparse
import hashlib
import json
from pathlib import Path

from holomotion.src.utils.artifacts import atomic_json


def evaluate_exported(options, *, launched=None):
    import numpy as np
    import torch

    from holomotion.src.deployment.supermimic_r1 import OfflineActor
    from holomotion.src.env.reference_tracking.r1_box_contract import (
        use_evaluation_box,
    )
    from holomotion.src.env.reference_tracking.task import load_task
    from holomotion.src.evaluation.reference_tracking.isaac_evaluation import (
        IsaacFullClipEvaluator,
    )

    torch.manual_seed(options.seed)
    torch.set_num_threads(8)
    torch.cuda.set_device(options.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    bundle = options.bundle
    actor = OfflineActor(bundle)
    model, motion, mdp = load_task(
        robot_urdf=options.robot_urdf,
        manifest=options.manifest,
        output=options.output,
        device=options.device,
        profile=options.profile,
    )
    box_profile = use_evaluation_box(motion)
    if box_profile is not None:
        np.testing.assert_allclose(
            motion.box["size_m"], actor.config["box"]["size_m"], atol=1e-8
        )
    goal = motion.goal(
        torch.arange(motion.num_frames, device=options.device),
        options.goal_frames,
    )
    np.testing.assert_allclose(
        goal.cpu().numpy(), actor.goals, atol=1e-5, rtol=1e-5
    )
    if mdp.obs_dim != actor.config["obs_dim"]:
        raise ValueError("Exported actor observation contract changed")

    def policy(history, reference):
        output = actor.session.run(
            None,
            dict(obs_hist=history.cpu().numpy(), goal=reference.cpu().numpy()),
        )[0]
        if not np.isfinite(output).all():
            raise FloatingPointError("Nonfinite ONNX action")
        return torch.tensor(output, device=options.device)

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
    result = evaluator.evaluate(policy)
    torch.save(evaluator.last_trace, options.output / "rollout_onnx.pt")
    result.update(
        actor_runtime=f"onnxruntime {actor.session.get_providers()[0]}",
        inference_precision=actor.config.get("inference_precision", "fp32"),
        artifact_sha256=actor.config["files"]["policy.onnx"],
        checkpoint_sha256=actor.config["source_checkpoint_sha256"],
        box_profile=box_profile,
    )
    return result


def main():
    from types import SimpleNamespace

    from holomotion.src.algo.supermimic.runtime import _supervise

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory")
    specification = json.loads(
        (args.training / "specification.json").read_text()
    )["arguments"]
    options = SimpleNamespace(
        **{
            **specification,
            "bundle": args.bundle.resolve(),
            "output": args.output.resolve(),
            "device": args.device,
            "manifest": Path(specification["manifest"]),
            "robot_urdf": Path(specification["robot_urdf"]),
        }
    )
    result = _supervise(options, training_operation=evaluate_exported)
    manifest_path = args.bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["exported_simulation_passed"] = bool(
        result["failed_clips"] == 0
        and result["evaluated_clips"] == 1
        and all(
            clip.get("box_lift", {}).get("lifted", False)
            for clip in result["clips"]
        )
    )
    report = args.output / "report.json"
    manifest["exported_simulation"] = dict(
        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        normalized_return=result["normalized_return"],
        clips=result["clips"],
    )
    atomic_json(manifest_path, manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
