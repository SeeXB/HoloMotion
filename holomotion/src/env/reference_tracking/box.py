"""Passive dynamic box physics and evaluation, without interaction rewards."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch


def load_box_reference(manifest, motion, device):
    metadata = json.loads(manifest.read_text())
    positions, rotations = [], []
    for clip in metadata["clips"].values():
        shard = metadata["hdf5_shards"][clip["shard"]]["file"]
        section = slice(clip["start"], clip["start"] + clip["length"])
        with h5py.File(manifest.parent / shard, "r") as stream:
            positions.append(stream["object_pos_w"][section])
            rotations.append(stream["object_quat_w"][section])
    motion.box_p = torch.tensor(np.concatenate(positions), device=device)
    motion.box_quat = torch.tensor(np.concatenate(rotations), device=device)
    motion.box_vertices = None
    if motion.box.get("mesh_file"):
        mesh = (manifest.parent / motion.box["mesh_file"]).resolve()
        vertices, _ = read_obj(mesh)
        motion.box_vertices = torch.tensor(vertices, device=device)
        motion.box["mesh_path"] = str(mesh)
    clearance = bottom_clearance(
        motion.box_p[motion.clip_start],
        motion.box_quat[motion.clip_start],
        motion.box["size_m"],
        vertices=motion.box_vertices,
    )
    if bool((clearance < -0.01).any()):
        raise ValueError(
            "Initial object penetrates the floor by more than 1 cm: "
            f"{clearance.tolist()}; use the original collision mesh"
        )


def read_obj(path):
    vertices, faces = [], []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        if fields and fields[0] == "f":
            face = [int(v.split("/")[0]) for v in fields[1:]]
            faces.append([v - 1 if v > 0 else len(vertices) + v for v in face])
    if not vertices or not faces:
        raise ValueError("Expected an OBJ surface with vertices and faces")
    return np.asarray(vertices, dtype=np.float32), faces


def box_asset_config(box, cache_dir):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    if box.get("mesh_path"):
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        vertices, faces = read_obj(box["mesh_path"])
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "box.usda"
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Box")
        stage.SetDefaultPrim(root.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(
            root.GetPrim()
        ).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(box["mass_kg"])
        mesh = UsdGeom.Mesh.Define(stage, "/Box/geometry")
        mesh.CreatePointsAttr(vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
        mesh.CreateFaceVertexIndicesAttr(
            [index for face in faces for index in face]
        )
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([(0.72, 0.52, 0.30)])
        UsdPhysics.CollisionAPI.Apply(
            mesh.GetPrim()
        ).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(
            mesh.GetPrim()
        ).CreateApproximationAttr("convexHull")
        material = UsdShade.Material.Define(stage, "/Box/Material")
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(box["friction"])
        physics.CreateDynamicFrictionAttr(box["friction"])
        physics.CreateRestitutionAttr(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(
            material.GetPrim()
        ).CreateFrictionCombineModeAttr("max")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            material, materialPurpose="physics"
        )
        stage.GetRootLayer().Save()
        return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(path),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False, max_depenetration_velocity=1.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=tuple(box["size_m"]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=box["mass_kg"]),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=box["friction"],
                dynamic_friction=box["friction"],
                restitution=0.0,
                friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.72, 0.52, 0.30)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


def reset_box(env, ids, frames):
    # Never replay the box trajectory in physics. Only initialize at reset.
    state = torch.zeros(len(ids), 13, device=env.device)
    state[:, :3] = env.motion.box_p[frames] + env.origins[ids]
    state[:, 3:7] = env.motion.box_quat[frames]
    env.box.write_root_state_to_sim(state, env_ids=ids)
    env.box.reset(ids)


def bottom_clearance(position, quaternion, size, *, vertices=None):
    """Lowest cuboid corner above z=0, accounting for arbitrary orientation."""
    from holomotion.src.env.reference_tracking.rotations import quat_to_mat

    rotation = quat_to_mat(quaternion)
    if vertices is not None:
        return position[..., 2] + (rotation[..., 2, :] @ vertices.T).amin(-1)
    half = torch.as_tensor(size, device=position.device) / 2
    return position[..., 2] - (rotation[..., 2, :].abs() * half).sum(-1)


class LiftMeter:
    def __init__(self, count, device, threshold=0.05, hold_seconds=0.5):
        self.threshold = threshold
        self.hold_seconds = hold_seconds
        self.peak = torch.full((count,), -torch.inf, device=device)
        self.current = torch.zeros(count, device=device)
        self.longest = torch.zeros(count, device=device)

    def update(self, clearance, active):
        if not torch.isfinite(clearance[active]).all():
            raise FloatingPointError("Nonfinite dynamic box state")
        self.peak = torch.where(
            active, torch.maximum(self.peak, clearance), self.peak
        )
        self.current = torch.where(
            active & (clearance >= self.threshold), self.current + 0.02, 0.0
        )
        self.longest = torch.maximum(self.longest, self.current)

    def report(self, failed):
        return [
            dict(
                peak_bottom_clearance_m=float(self.peak[i]),
                longest_lift_seconds=float(self.longest[i]),
                lifted=bool(
                    self.longest[i] >= self.hold_seconds and not failed[i]
                ),
                threshold_m=self.threshold,
                required_hold_seconds=self.hold_seconds,
            )
            for i in range(len(self.peak))
        ]
