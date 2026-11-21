"""Own sequential 20k-update imitation jobs and their final validation."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from holomotion.src.utils.artifacts import atomic_json

REPOSITORY = Path(__file__).resolve().parents[3]


def actor_export_command(training, bundle, precision):
    if precision == "bf16":
        module = "holomotion.scripts.training.export_supermimic_bf16"
        source = ["--training", str(training)]
    elif precision == "fp32":
        module = "holomotion.scripts.training.export_supermimic"
        source = ["--checkpoint", str(training / "checkpoint_final.pt")]
    else:
        raise ValueError(f"Unsupported actor export precision: {precision}")
    return [sys.executable, "-m", module, *source, "--output", str(bundle)]


def command_plan(output, robots, target, holosoma):
    if target < 20000:
        raise ValueError(
            "This launcher requires at least 20,000 policy updates"
        )
    python = Path(sys.executable).absolute()
    jobs = []
    audit_path = (
        REPOSITORY / "not_for_commit/r1-audit/reproducible/report.json"
    )
    if "r1" in robots:
        audit = json.loads(audit_path.read_text())
        if not audit["checks_passed"]:
            raise ValueError("R1 reference audit has not passed")
        for name, expected in audit["input_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Audited R1 input changed: {name}")
    for robot in robots:
        manifest = REPOSITORY / (
            "data/omomo-g1-mesh/manifest.json"
            if robot == "g1"
            else "data/omomo-r1/manifest.json"
        )
        urdf = holosoma / (
            "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_halfspherehand.urdf"
            if robot == "g1"
            else (
                "src/holosoma_retargeting/holosoma_retargeting/"
                "demo_data/models/r1/r1_26dof.urdf"
            )
        )
        if not manifest.is_file() or not urdf.is_file():
            raise FileNotFoundError(f"Missing {robot} manifest or URDF")
        directory = output / robot
        # Preserve the validated logical batches and update-to-data ratio.
        arguments = [
            str(python),
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
        jobs.append(
            dict(
                robot=robot,
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
        )
    return dict(
        target_policy_updates=target,
        robots=robots,
        jobs=jobs,
        reference_audit=str(audit_path) if "r1" in robots else None,
        interpretation=(
            "One iter is one complete policy optimizer update; "
            "world-model events and vector steps have separate counters"
        ),
        launcher_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    )


def execute(plan, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new job output directory")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "plan.json", plan)
    environment = os.environ.copy()
    environment.update(
        OMNI_KIT_ACCEPT_EULA="YES",
        ACCEPT_EULA="YES",
        HDF5_USE_FILE_LOCKING="FALSE",
    )
    environment.setdefault(
        "XDG_CACHE_HOME", str(REPOSITORY / "not_for_commit/omomo-cache")
    )
    for key, folder in [
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("WARP_CACHE_PATH", "warp"),
        ("MPLCONFIGDIR", "matplotlib"),
    ]:
        environment[key] = str(Path(environment["XDG_CACHE_HOME"]) / folder)
    environment["MUJOCO_GL"] = "osmesa"
    state = dict(
        status="running",
        started_utc=datetime.now(timezone.utc).isoformat(),
        target_policy_updates=plan["target_policy_updates"],
        jobs={name: dict(status="queued") for name in plan["robots"]},
    )

    def progress(robot, phase):
        state["active_robot"] = robot
        state["phase"] = phase
        state["jobs"][robot]["status"] = phase
        atomic_json(output / "status.json", state)
        print(json.dumps(dict(robot=robot, phase=phase)), flush=True)

    def run(command, log):
        print(json.dumps(dict(command=command, log=str(log))), flush=True)
        with log.open("w") as stream:
            subprocess.run(
                command,
                cwd=REPOSITORY,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def run_job(job):
        robot = job["robot"]
        directory = Path(job["output"])
        directory.mkdir()
        progress(robot, "preflight")
        run(job["preflight"], directory / "preflight.log")
        preflight = json.loads(
            (directory / "preflight/run_status.json").read_text()
        )
        if preflight != dict(completed=True, phase="finished"):
            raise RuntimeError(f"{robot} physical preflight did not complete")
        progress(robot, "training")
        run(job["training"], directory / "training.log")
        training = directory / "training"
        evaluation = json.loads(
            (training / "evaluation_final.json").read_text()
        )
        updates = evaluation["actor_updates"]
        if updates < plan["target_policy_updates"]:
            raise RuntimeError(
                f"{robot} stopped at {updates} policy updates "
                "before the target"
            )
        state["jobs"][robot].update(
            policy_updates=updates, native_evaluation=evaluation["clips"]
        )
        progress(robot, "rendering")
        run(
            [
                sys.executable,
                "-m",
                "holomotion.scripts.evaluation.render_omomo_rollout",
                "--training",
                str(training),
                "--output",
                str(directory / "evaluation.mp4"),
            ],
            directory / "render.log",
        )
        run(
            [
                sys.executable,
                "-m",
                "holomotion.scripts.evaluation.render_omomo_reference",
                "--training",
                str(training),
                "--output",
                str(directory / "reference-vs-policy.mp4"),
            ],
            directory / "reference_render.log",
        )
        if robot == "r1":
            progress(robot, "exporting")
            bundle = directory / "bundle"
            run(
                actor_export_command(
                    training, bundle, plan.get("export_precision", "fp32")
                ),
                directory / "export.log",
            )
            progress(robot, "onnx_evaluation")
            run(
                [
                    sys.executable,
                    "-m",
                    "holomotion.scripts.evaluation.evaluate_supermimic_onnx",
                    "--training",
                    str(training),
                    "--bundle",
                    str(bundle),
                    "--output",
                    str(directory / "onnx-evaluation"),
                ],
                directory / "onnx_evaluation.log",
            )
            run(
                [
                    sys.executable,
                    "-m",
                    "holomotion.src.deployment.supermimic_r1",
                    "--bundle",
                    str(bundle),
                    "--check",
                ],
                directory / "no_action_check.log",
            )
        progress(robot, "completed")

    errors = []
    for job in plan["jobs"]:
        try:
            run_job(job)
        except Exception:
            error = traceback.format_exc()
            state["jobs"][job["robot"]].update(status="failed", error=error)
            errors.append(job["robot"])
            atomic_json(output / "status.json", state)
    state.update(
        status="failed" if errors else "completed",
        phase="finished",
        failed_jobs=errors,
        finished_utc=datetime.now(timezone.utc).isoformat(),
    )
    atomic_json(output / "status.json", state)
    if errors:
        raise RuntimeError(f"Jobs failed; inspect status.json: {errors}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--robots", nargs="+", choices=("r1", "g1"), default=["r1", "g1"]
    )
    parser.add_argument("--policy-updates", type=int, default=20000)
    parser.add_argument(
        "--holosoma-root", type=Path, default=REPOSITORY.parent / "holosoma"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.robots)) != len(args.robots):
        parser.error("Each robot may appear only once")
    plan = command_plan(
        args.output.resolve(),
        args.robots,
        args.policy_updates,
        args.holosoma_root.resolve(),
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    execute(plan, args.output.resolve())


if __name__ == "__main__":
    main()
