"""Load the shared physical task for training and saved-policy evaluation."""

import json

import torch

from holomotion.src.env.reference_tracking.kinematics import G1Kinematics
from holomotion.src.env.reference_tracking.mdp import MDP
from holomotion.src.env.reference_tracking.model import build_model
from holomotion.src.env.reference_tracking.motion import MotionLib
from holomotion.src.env.reference_tracking.robot_contract import robot_contract


def load_task(*, robot_urdf, manifest, output, device, profile):
    metadata = json.loads(manifest.read_text())
    contract = robot_contract(metadata.get("robot", "g1"))
    model = build_model(
        robot_urdf,
        output / "g1.xml",
        self_collisions=profile == "paper",
        contract=contract,
    )
    kinematics = G1Kinematics(
        device, model, tracked=contract.tracked, contract=contract
    )
    motion = MotionLib.from_hdf5(
        manifest,
        kinematics,
        output / "g1.xml" if metadata.get("box") else robot_urdf,
        device,
    )
    mdp = MDP(
        kinematics,
        motion,
        torch.tensor(contract.default, device=device),
        reference_orientation=profile == "vibed",
    )
    motion.box = metadata.get("box")
    motion.evaluation_box = metadata.get("evaluation_box")
    if motion.box:
        from holomotion.src.env.reference_tracking.box import (
            load_box_reference,
        )

        load_box_reference(manifest, motion, device)
    return model, motion, mdp
