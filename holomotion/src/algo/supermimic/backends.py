"""Select physics and complete-clip evaluation without changing the learning method."""

from functools import partial


def backend_class(args):
    import json

    metadata = json.loads(args.manifest.read_text())
    if args.physics_backend != "isaac_physx" and (
        metadata.get("box") or metadata.get("robot", "g1") != "g1"
    ):
        raise ValueError(
            "R1 and passive-box tasks currently require Isaac PhysX"
        )
    if getattr(args, "physics_backend", "mujoco_warp") == "isaac_physx":
        from holomotion.src.env.reference_tracking.isaac_backend import (
            IsaacTrackingBackend,
        )

        return partial(
            IsaacTrackingBackend,
            robot_urdf=args.robot_urdf,
            cache_dir=args.output / "isaac_usd",
        )
    from holomotion.src.env.reference_tracking.warp_backend import (
        WarpTrackingBackend,
    )

    return WarpTrackingBackend


def evaluator_class(args):
    if getattr(args, "physics_backend", "mujoco_warp") == "isaac_physx":
        from holomotion.src.evaluation.reference_tracking.isaac_evaluation import (
            IsaacFullClipEvaluator,
        )

        return partial(
            IsaacFullClipEvaluator,
            robot_urdf=args.robot_urdf,
            cache_dir=args.output / "isaac_usd",
        )
    from holomotion.src.evaluation.reference_tracking.evaluation import (
        FullClipEvaluator,
    )

    return FullClipEvaluator
