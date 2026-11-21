"""Run the fixed-contact 032 experiment after the existing Omni batch."""

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from holomotion.scripts.training.run_supermimic_r1_trajectory import (
    command_plan,
)
from holomotion.src.utils.artifacts import atomic_json, fingerprint

RUNTIME = Path(__file__).resolve().parents[3]


def validate(recipe, output):
    provenance_path = RUNTIME / "runtime_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        for relative, expected in provenance["source_sha256"].items():
            if fingerprint(RUNTIME / relative) != expected:
                raise ValueError(f"Frozen runtime changed: {relative}")
    for filename, expected in recipe["input_sha256"].items():
        if fingerprint(Path(filename)) != expected:
            raise ValueError(f"Frozen input changed: {filename}")
    manifest = Path(recipe["manifest"])
    report = json.loads(
        (manifest.parent.parent / "validation.json").read_text()
    )
    for relative, expected in report["preprocessing_source_sha256"].items():
        if fingerprint(RUNTIME / relative) != expected:
            raise ValueError(f"The validated preparer differs: {relative}")
    if (
        not report["checks_passed"]
        or not report["fixed_contact_reader_audit"]["passed"]
    ):
        raise ValueError("The contact reference has not passed validation")
    if report["maximum_source_foot_position_error_m"] > 1e-5:
        raise ValueError("The source foot support has changed")
    return command_plan(
        manifest,
        Path(recipe["robot_urdf"]),
        output / "sub10_largebox_032",
        20000,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    plan = validate(recipe, args.output)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty queue output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "recipe.json", recipe)
    state = dict(
        status="queued",
        phase="waiting_for_existing_omni_batch",
        created_utc=datetime.now(timezone.utc).isoformat(),
        task="sub10_largebox_032",
        source_variant="b4_fixed_contacts",
        target_actor_updates=20000,
        preview_actor_updates=10000,
        wait_for_batch=recipe["wait_for_batch"],
        contact_reference_audit=recipe["contact_reference_audit"],
        real_robot_actions=False,
    )

    def save():
        atomic_json(args.output / "status.json", state)

    save()
    started = time.monotonic()
    try:
        while True:
            old = json.loads(Path(recipe["wait_for_batch"]).read_text())
            expected = {f"sub10_largebox_{x}" for x in ("032", "087", "089")}
            if set(old["jobs"]) != expected:
                raise ValueError(
                    "The prerequisite batch is not the expected Omni trio"
                )
            state["existing_omni_active_task"] = old.get("active_task")
            state["existing_omni_jobs"] = {
                k: v["status"] for k, v in old["jobs"].items()
            }
            save()
            if (
                old["status"] in ("completed", "failed")
                and old.get("active_task") is None
            ):
                if not all(
                    j["status"] in ("completed", "failed")
                    for j in old["jobs"].values()
                ):
                    raise RuntimeError(
                        "The prerequisite batch ended with incomplete jobs"
                    )
                state["existing_omni_final_status"] = old["status"]
                break
            service = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "is-active",
                    recipe["wait_for_service"],
                ],
                capture_output=True,
                text=True,
            )
            if service.returncode != 0:
                raise RuntimeError(
                    "The prerequisite Omni service stopped "
                    "before its batch finished"
                )
            if time.monotonic() - started > 12 * 3600:
                raise TimeoutError(
                    "The prerequisite batch did not finish within 12 hours"
                )
            time.sleep(15)
        validate(recipe, args.output)
        state.update(
            status="running",
            phase="preflight_and_training",
            started_utc=datetime.now(timezone.utc).isoformat(),
        )
        save()
        command = [
            sys.executable,
            "-m",
            "holomotion.scripts.training.run_supermimic_r1_trajectory",
            "--manifest",
            recipe["manifest"],
            "--robot-urdf",
            recipe["robot_urdf"],
            "--output",
            str(args.output / "sub10_largebox_032"),
            "--policy-updates",
            "20000",
        ]
        with (args.output / "launch.log").open("x") as log:
            subprocess.run(
                command,
                cwd=RUNTIME,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        validate(recipe, args.output)
        state.update(status="completed", phase="finished")
    except Exception:
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
