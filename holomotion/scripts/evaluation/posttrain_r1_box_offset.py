"""Run an additional box-position evaluation after a training pipeline ends."""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from holomotion.src.utils.artifacts import atomic_json


def run(root, runtime, distance):
    root, runtime = root.resolve(), runtime.resolve()
    status_path = root / "box_offset_after_training.json"
    output = (
        root / "r1" / f"eval_actor_020000_toward_robot_{distance * 100:g}cm"
    )

    def status(phase, **fields):
        atomic_json(
            status_path,
            dict(
                status=phase,
                toward_robot_m=distance,
                output=str(output),
                **fields,
            ),
        )

    status("waiting_for_training_pipeline")
    try:
        deadline = time.monotonic() + 12 * 3600
        while True:
            state = json.loads((root / "status.json").read_text())
            if state["status"] in {"completed", "failed"}:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Training pipeline did not finish within 12 hours"
                )
            time.sleep(20)
        baseline = root / "r1/training"
        original = json.loads((baseline / "evaluation_final.json").read_text())
        if original["actor_updates"] != 20000:
            raise ValueError("Expected the complete 20k actor evaluation")
        status("evaluating", actor_updates=20000)
        with (root / "box_offset_after_training.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "holomotion.scripts.evaluation.evaluate_r1_box_offset",
                    "--checkpoint",
                    str(baseline / "checkpoint_final.pt"),
                    "--output",
                    str(output),
                    "--toward-robot-m",
                    str(distance),
                ],
                cwd=runtime,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
            shifted = json.loads(
                (output / "evaluation_final.json").read_text()
            )
            video = output / "placement-comparison.mp4"
            if original["failed_clips"] == 0 and shifted["failed_clips"] == 0:
                status("rendering", actor_updates=20000)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "holomotion.scripts.evaluation.compare_r1_box_placements",
                        "--baseline",
                        str(baseline),
                        "--shifted",
                        str(output),
                        "--output",
                        str(video),
                    ],
                    cwd=runtime,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            else:
                video = output / "physical-rollout.mp4"
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "holomotion.scripts.evaluation.render_omomo_rollout",
                        "--training",
                        str(output),
                        "--output",
                        str(video),
                    ],
                    cwd=runtime,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        status(
            "completed",
            actor_updates=20000,
            shifted_clips=shifted["clips"],
            video=str(video),
        )
    except BaseException:
        status("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--toward-robot-m", type=float, required=True)
    args = parser.parse_args()
    run(args.run_root, args.runtime, args.toward_robot_m)
