"""Accept each evaluated training width without weakening profile checks."""

import copy

import pytest

from holomotion.src.deployment.supermimic_r1 import validate_box_profile


def configuration(width):
    training = dict(size_m=[width, 0.245, 0.27], mass_kg=0.1, friction=0.8)
    real = dict(size_m=[0.305, 0.245, 0.27], mass_kg=0.1, friction=0.8)
    profile = dict(
        axis_order=["left_right", "front_back", "up"],
        training_box=training,
        evaluation_box=real,
    )
    return copy.deepcopy(
        dict(
            box=real,
            training_box=training,
            box_profile=profile,
            source_checkpoint_sha256="actor",
            simulation_evaluation=dict(
                checkpoint_sha256="actor", box_profile=profile
            ),
        )
    )


@pytest.mark.parametrize("width", [0.24, 0.2445])
def test_accepts_width_in_the_actual_native_evaluation(width):
    validate_box_profile(configuration(width))


@pytest.mark.parametrize(
    "mutation",
    ["declared", "native", "axis", "checkpoint", "depth", "wide", "nonfinite"],
)
def test_rejects_inconsistent_geometry_or_lineage(mutation):
    cfg = configuration(0.24)
    # Detach aliases so one corrupted field cannot update its evidence too.
    cfg["box_profile"] = copy.deepcopy(cfg["box_profile"])
    cfg["simulation_evaluation"] = copy.deepcopy(cfg["simulation_evaluation"])
    if mutation == "declared":
        cfg["training_box"]["size_m"][0] = 0.2445
    elif mutation == "native":
        cfg["simulation_evaluation"]["box_profile"]["training_box"]["size_m"][
            0
        ] = 0.2445
    elif mutation == "axis":
        cfg["box_profile"]["axis_order"] = ["up", "front_back", "left_right"]
    elif mutation == "checkpoint":
        cfg["simulation_evaluation"]["checkpoint_sha256"] = "other"
    elif mutation == "depth":
        cfg["training_box"]["size_m"][1] = 0.25
    elif mutation == "wide":
        cfg["training_box"]["size_m"][0] = 0.32
    else:
        cfg["training_box"]["size_m"][0] = float("nan")
    with pytest.raises((ValueError, AssertionError)):
        validate_box_profile(cfg)
