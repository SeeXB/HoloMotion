"""Evaluate an immutable saved checkpoint alongside an ongoing training run."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from holomotion.src.algo.supermimic.runtime import _supervise
from holomotion.src.utils.artifacts import atomic_json, fingerprint


def preview(checkpoint, output):
    checkpoint = checkpoint.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty preview output directory")
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    specification = bundle["specification"]
    arguments = specification["arguments"]
    schedule = bundle["schedule"]
    recorded = dict(
        source_checkpoint=str(checkpoint),
        checkpoint_sha256=fingerprint(checkpoint),
        actor_updates=schedule["actor_updates"],
        world_updates=schedule["world_updates"],
        vector_steps=schedule["vector_steps"],
        training_wall_s=bundle["training_wall_s"],
    )
    del bundle
    output.mkdir(parents=True, exist_ok=True)
    # The existing native evaluator reads a fixed filename. This alias never
    # mutates the original checkpoint or the running trainer's directory.
    (output / "checkpoint_final.pt").symlink_to(checkpoint)
    (output / "source_snapshot").symlink_to(
        checkpoint.parent / "source_snapshot", target_is_directory=True
    )
    atomic_json(output / "specification.json", specification)
    atomic_json(output / "preview_checkpoint.json", recorded)
    options = SimpleNamespace(
        **{
            **arguments,
            "output": (output / "evaluation").resolve(),
            "manifest": Path(arguments["manifest"]),
            "robot_urdf": Path(arguments["robot_urdf"]),
        }
    )
    result = _supervise(options, evaluation=True)
    result.update(
        recorded,
        label=f"actor_{schedule['actor_updates']:06d}",
        final_policy_evaluation=False,
        evaluation_scope="Intermediate checkpoint; main training continues",
    )
    result["rollout_file"] = "evaluation/rollout_final.pt"
    atomic_json(output / "evaluation_final.json", result)
    (output / "g1.xml").symlink_to(output.resolve() / "evaluation/g1.xml")
    if fingerprint(checkpoint) != recorded["checkpoint_sha256"]:
        raise ValueError("The source checkpoint changed during preview")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preview(args.checkpoint, args.output), indent=2))
