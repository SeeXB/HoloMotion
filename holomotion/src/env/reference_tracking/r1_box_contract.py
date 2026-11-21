"""User-confirmed R1 box dimensions: left/right, front/back, height."""

import math

R1_BOX_AXES = ("left_right", "front_back", "up")
R1_TRAIN_BOX_SIZE_M = (0.2445, 0.245, 0.270)
R1_REAL_BOX_SIZE_M = (0.305, 0.245, 0.270)


def validate_size(size):
    if len(size) != 3 or any(not math.isfinite(x) or x <= 0 for x in size):
        raise ValueError("Expected positive left/right, front/back and height")
    return [float(x) for x in size]


def use_evaluation_box(motion):
    """Use the real evaluation box without changing robot references."""
    from holomotion.src.env.reference_tracking.box import bottom_clearance

    target = getattr(motion, "evaluation_box", None)
    if target is None:
        return None
    if motion.box_vertices is not None:
        raise ValueError("R1 train/real box profiles expect cuboid geometry")
    source = dict(motion.box)
    evaluation = dict(source, **target)
    source_size = validate_size(source["size_m"])
    target_size = validate_size(evaluation["size_m"])
    if evaluation["mass_kg"] != source["mass_kg"]:
        raise ValueError("The train/real box test must preserve box mass")
    initial = motion.box_p[motion.clip_start].clone()
    quat = motion.box_quat[motion.clip_start]
    before = bottom_clearance(initial, quat, source_size)
    after = bottom_clearance(initial, quat, target_size)
    shift = before - after
    motion.box_p = motion.box_p.clone()
    motion.box_p[motion.clip_start, 2] += shift
    motion.box = evaluation
    return dict(
        axis_order=list(R1_BOX_AXES),
        training_box=source,
        evaluation_box=evaluation,
        initial_center_z_shift_m=shift.cpu().tolist(),
        robot_reference_unchanged=True,
        box_center_xy_unchanged=True,
        box_orientation_unchanged=True,
        box_pose_replayed=False,
    )
