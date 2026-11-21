"""Run independent R1 policies sequentially, with a 10k preview for each."""

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from holomotion.scripts.training.run_supermimic_r1_trajectory import (
    command_plan,
)
from holomotion.src.utils.artifacts import atomic_json, fingerprint

REPOSITORY = Path(__file__).resolve().parents[3]


def validate(recipe, output):
    for path, expected in recipe["input_sha256"].items():
        if fingerprint(Path(path)) != expected:
            raise ValueError(f"Frozen batch input changed: {path}")
    jobs = recipe["jobs"]
    names = [job["task"] for job in jobs]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate batch task names")
    plans = []
    for job in jobs:
        manifest = Path(job["manifest"])
        metadata = json.loads(manifest.read_text())
        if list(metadata["clips"]) != [job["task"]]:
            raise ValueError("The task name and selected clip differ")
        if metadata["box"]["size_m"] != [0.24, 0.245, 0.27]:
            raise ValueError("Expected a 24 cm training width")
        if metadata["evaluation_box"]["size_m"] != [0.305, 0.245, 0.27]:
            raise ValueError("Expected the 30.5 cm physical evaluation width")
        placement = metadata["provenance"]["standing_transition"][
            "initial_box_placement"
        ]
        nearest = min(placement["evaluation_foot_box_gaps_m"].values())
        if abs(nearest - 0.005) > 1e-4:
            raise ValueError(
                "Expected the real-size box within 5 mm of the toes"
            )
        plans.append(
            command_plan(
                manifest, Path(job["robot_urdf"]), output / job["task"], 20000
            )
        )
    return plans


def execute(recipe_path, output):
    recipe = json.loads(recipe_path.read_text())
    validate(recipe, output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty batch output directory")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "recipe.json", recipe)
    state = dict(
        status="running",
        scheduling="sequential on one GPU",
        initialization="Three independent policies from scratch",
        target_actor_updates_per_task=20000,
        preview_actor_updates_per_task=10000,
        started_utc=datetime.now(timezone.utc).isoformat(),
        runtime_root=str(REPOSITORY),
        jobs={job["task"]: dict(status="queued") for job in recipe["jobs"]},
    )
    atomic_json(output / "status.json", state)
    for job in recipe["jobs"]:
        name = job["task"]
        try:
            validate(recipe, output)
            state["active_task"] = name
            state["jobs"][name].update(
                status="running",
                started_utc=datetime.now(timezone.utc).isoformat(),
            )
            atomic_json(output / "status.json", state)
            with (output / f"{name}.log").open("x") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "holomotion.scripts.training.run_supermimic_r1_trajectory",
                        "--manifest",
                        job["manifest"],
                        "--robot-urdf",
                        job["robot_urdf"],
                        "--output",
                        str(output / name),
                        "--policy-updates",
                        "20000",
                    ],
                    cwd=REPOSITORY,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            validate(recipe, output)
            state["jobs"][name]["status"] = "completed"
        except Exception:
            state["jobs"][name].update(
                status="failed", error=traceback.format_exc()
            )
        state["jobs"][name]["finished_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        atomic_json(output / "status.json", state)
    state.update(
        status="failed"
        if any(j["status"] == "failed" for j in state["jobs"].values())
        else "completed",
        active_task=None,
        finished_utc=datetime.now(timezone.utc).isoformat(),
    )
    atomic_json(output / "status.json", state)
    if state["status"] == "failed":
        raise RuntimeError(
            "One or more policies failed; see batch status.json"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                validate(
                    json.loads(args.recipe.read_text()), args.output.resolve()
                ),
                indent=2,
            )
        )
    else:
        execute(args.recipe.resolve(), args.output.resolve())
