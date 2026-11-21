"""Build a local overlay around an existing Isaac Sim Python environment.

Run with the existing Isaac Python, e.g. <isaac-env>/bin/python
environments/setup_omomo.py. No packages in the base environment are changed.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import venv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".venv-omomo"))
    args = parser.parse_args()
    isaac = importlib.util.find_spec("isaacsim")
    if isaac is None:
        parser.error(
            "Run with a Python environment containing Isaac Sim and Isaac Lab"
        )
    root = Path(isaac.origin).parent
    bundled = sorted(root.glob("extscache/omni.warp.core-*/warp"))
    if len(bundled) != 1:
        parser.error("Expected one unambiguous Isaac-bundled Warp package")
    if args.output.exists():
        parser.error("Choose a new environment directory")
    venv.create(args.output, system_site_packages=True, with_pip=True)
    python = args.output / "bin/python"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "mujoco==3.11.0",
            "tabulate==0.9.0",
            "accelerate==1.10.1",
            "ruff==0.17.1",
            "onnx==1.19.0",
            "onnxruntime==1.29.0",
            "etils==1.13.0",
        ],
        check=True,
    )
    site = subprocess.check_output(
        [
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        text=True,
    ).strip()
    (Path(site) / "warp").symlink_to(
        bundled[0].resolve(), target_is_directory=True
    )
    report = dict(
        base_python=sys.executable,
        isaacsim=importlib.metadata.version("isaacsim"),
        isaaclab=importlib.metadata.version("isaaclab"),
        warp_source=str(bundled[0]),
        environment=str(args.output.resolve()),
    )
    (args.output / "isaac_overlay.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
