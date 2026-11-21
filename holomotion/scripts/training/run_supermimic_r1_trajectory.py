"""Train one audited R1 trajectory and evaluate its 10k checkpoint."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path

from holomotion.scripts.training.run_omomo_20k import execute
from holomotion.src.env.reference_tracking.robot_contract import robot_contract
from holomotion.src.utils.artifacts import atomic_json

REPOSITORY = Path(__file__).resolve().parents[3]
PREVIEW_UPDATES = 10000


def command_plan(manifest, urdf, output, target):
    if target < 20000:
        raise ValueError(
            "Keep the requested 20k training target and preview at 10k"
        )
    metadata = json.loads(manifest.read_text())
    audit = manifest.parent.parent / "validation.json"
    report = json.loads(audit.read_text())
    if metadata["robot"] != "r1" or not report["checks_passed"]:
        raise ValueError("An audited R1 trajectory is required")
    for path, expected in report["input_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Audited source changed: {path}")
    if not {"head_pitch_link", "head_yaw_link"}.issubset(
        robot_contract("r1").tracked
    ):
        raise ValueError("Use the isolated runtime with head tracking enabled")
    training = REPOSITORY / "holomotion/src/algo/supermimic/training.py"
    if 'save("actor_010000")' not in training.read_text():
        raise ValueError("The runtime must save the exact 10k checkpoint")
    directory = output / "r1"
    arguments = [
        sys.executable,
        "-m",
        "holomotion.scripts.training.train_supermimic",
        "--manifest",
        str(manifest),
        "--robot-urdf",
        str(urdf),
        "--physics-backend",
        "isaac_physx",
        "--num-envs",
        "128",
        "--buffer-steps",
        "500",
        "--horizon",
        "24",
        "--wm-batch",
        "256",
        "--pi-batch",
        "512",
        "--pi-microbatch",
        "128",
        "--wm-batches",
        "10",
        "--world-only-updates",
        "100",
        "--lr",
        "0.0004",
        "--seed",
        "42",
        "--no-capture-graph",
        "--checkpoint-minutes",
        "15",
        "--max-actor-updates",
        str(target),
        "--max-updates",
        "320000",
        "--max-vector-steps",
        "800064",
        "--max-hours",
        "24",
    ]
    return dict(
        target_policy_updates=target,
        preview_policy_updates=PREVIEW_UPDATES,
        initialization="From scratch; no previous policy or optimizer loaded",
        robots=["r1"],
        reference_audit=str(audit),
        runtime_root=str(REPOSITORY),
        tracked_bodies=robot_contract("r1").tracked,
        interaction_rewards=False,
        export_precision="bf16",
        jobs=[
            dict(
                robot="r1",
                output=str(directory),
                manifest=str(manifest),
                urdf=str(urdf),
                preflight=[
                    *arguments,
                    "--output",
                    str(directory / "preflight"),
                    "--num-envs",
                    "2",
                    "--preflight-only",
                    "--warmup-steps",
                    "4",
                    "--no-compile",
                ],
                training=[*arguments, "--output", str(directory / "training")],
            )
        ],
    )


def watch_preview(output, stop):
    checkpoint = output / "r1/training/checkpoint_actor_010000.pt"
    while not checkpoint.is_file():
        if stop.wait(5):
            return
    preview = output / "r1/preview_actor_010000"
    status_path = output / "preview_status.json"
    environment = os.environ.copy()
    environment["MUJOCO_GL"] = "osmesa"
    try:
        atomic_json(
            status_path,
            dict(status="evaluating", actor_updates=PREVIEW_UPDATES),
        )
        commands = [
            [
                sys.executable,
                "-m",
                "holomotion.scripts.evaluation.evaluate_supermimic_snapshot",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(preview),
            ],
            [
                sys.executable,
                "-m",
                "holomotion.scripts.evaluation.render_omomo_reference",
                "--training",
                str(preview),
                "--output",
                str(preview / "reference-vs-policy.mp4"),
            ],
        ]
        with (output / "preview.log").open("w") as stream:
            for command in commands:
                subprocess.run(
                    command,
                    cwd=REPOSITORY,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        result = json.loads((preview / "evaluation_final.json").read_text())
        atomic_json(
            status_path,
            dict(
                status="completed",
                actor_updates=PREVIEW_UPDATES,
                clips=result["clips"],
                video=str(preview / "reference-vs-policy.mp4"),
            ),
        )
    except Exception:
        atomic_json(
            status_path, dict(status="failed", error=traceback.format_exc())
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-updates", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    plan = command_plan(
        args.manifest.resolve(),
        args.robot_urdf.resolve(),
        output,
        args.policy_updates,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    # The watcher writes nothing until the atomic 10k checkpoint exists.
    stop = threading.Event()
    watcher = threading.Thread(
        target=watch_preview, args=(output, stop), daemon=True
    )
    watcher.start()
    try:
        execute(plan, output)
    finally:
        stop.set()
        watcher.join(timeout=60)


if __name__ == "__main__":
    main()
