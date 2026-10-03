"""Props in the scene: primitives or USD files, spawned into the live stage.

Spawned as plain USD prims carrying physics APIs, not as Isaac Lab
RigidObjects: a RigidObject only initializes on the timeline's PLAY event,
which has already fired by the time a command can run. PhysX picks the new
prims up from USD change notices -- but ONLY while Kit is rendering: headless
with rendering off, a runtime-spawned ball neither moved nor reported a
velocity, which is why scripts/simd.py always enables offscreen rendering.

Poses are read from fabric (omni:fabric:worldMatrix), the copy the renderer
draws from. Not from USD: Isaac Lab simulates with fabric, so a moving body's
USD transform stays at its spawn pose. Not from the PhysX CPU query either:
GPU readback is suppressed, so it returns the spawn pose too. And not through
a new PhysX tensor view: creating one mid-simulation on the GPU pipeline
crashed the daemon with a CUDA illegal memory access. (All measured.)

Import only after AppLauncher has started Kit.
"""

from __future__ import annotations

import re
from typing import Any

import isaaclab.sim as sim_utils
import omni.usd

from .scene import expand_usd

ROOT = "/World/Objects"
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SHAPES = {
    "cuboid": (sim_utils.CuboidCfg, ("size",)),
    "sphere": (sim_utils.SphereCfg, ("radius",)),
    "cylinder": (sim_utils.CylinderCfg, ("radius", "height")),
    "capsule": (sim_utils.CapsuleCfg, ("radius", "height")),
    "cone": (sim_utils.ConeCfg, ("radius", "height")),
}


def spawn(name: str, shape: str, pos: list[float], rot: list[float] | None = None,
          size: list[float] | None = None, radius: float | None = None,
          height: float | None = None, mass: float = 0.1, static: bool = False,
          kinematic: bool = False, color: list[float] | None = None,
          friction: float = 0.8, usd: str | None = None,
          scale: list[float] | None = None) -> dict[str, Any]:
    """Spawn one prop at /World/Objects/<name>.

    static: a collider with no rigid body (a table, a wall) — it never moves.
    kinematic: a rigid body moved only by you, never by contact.
    rot is a (w, x, y, z) quaternion.
    """
    if not _NAME.match(name):
        raise ValueError(f"object name must be an identifier: {name!r}")
    path = f"{ROOT}/{name}"
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(path).IsValid():
        raise ValueError(f"an object named {name!r} already exists; remove it first")

    rigid = None if static else sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic)
    mass_props = None if static else sim_utils.MassPropertiesCfg(mass=float(mass))
    collision = sim_utils.CollisionPropertiesCfg()
    pos_t = tuple(float(v) for v in pos)
    rot_t = tuple(float(v) for v in rot) if rot else (1.0, 0.0, 0.0, 0.0)

    if shape == "usd":
        if not usd:
            raise ValueError("shape='usd' needs usd=<path>")
        cfg = sim_utils.UsdFileCfg(
            usd_path=expand_usd(usd), rigid_props=rigid, mass_props=mass_props,
            collision_props=collision,
            scale=tuple(float(v) for v in scale) if scale else None)
    elif shape in SHAPES:
        cls, needed = SHAPES[shape]
        given = {"size": tuple(float(v) for v in size) if size else None,
                 "radius": radius, "height": height}
        missing = [k for k in needed if given[k] is None]
        if missing:
            raise ValueError(f"shape {shape!r} needs {missing}")
        cfg = cls(
            **{k: given[k] for k in needed},
            rigid_props=rigid, mass_props=mass_props, collision_props=collision,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=friction, dynamic_friction=friction),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(color) if color else (0.8, 0.3, 0.2)),
        )
    else:
        raise ValueError(f"unknown shape {shape!r}; use one of {sorted(SHAPES)} or 'usd'")

    cfg.func(path, cfg, translation=pos_t, orientation=rot_t)
    return {"name": name, "path": path, "shape": shape, "static": static,
            "kinematic": kinematic, "pos": list(pos_t), "rot": list(rot_t)}


def _bodies() -> list[Any]:
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(ROOT)
    return list(root.GetChildren()) if root.IsValid() else []


def poses() -> list[dict[str, Any]]:
    """Every prop's current world pose: pos (m) and rot (w, x, y, z)."""
    import usdrt
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    rt = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    out = []
    for prim in _bodies():
        path = str(prim.GetPath())
        attr = rt.GetPrimAtPath(path).GetAttribute("omni:fabric:worldMatrix")
        value = attr.Get() if attr and attr.IsValid() else None
        if value is not None:
            m = Gf.Matrix4d(*[[float(value[r][c]) for c in range(4)] for r in range(4)])
        else:  # not in fabric yet (e.g. static and never rendered)
            m = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
        t = m.ExtractTranslation()
        q = m.RemoveScaleShear().ExtractRotationQuat()
        im = q.GetImaginary()
        out.append({"name": prim.GetName(), "path": path,
                    "pos": [float(t[0]), float(t[1]), float(t[2])],
                    "rot": [float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])]})
    return out


def remove(name: str) -> dict[str, Any]:
    stage = omni.usd.get_context().get_stage()
    path = f"{ROOT}/{name}"
    if not stage.GetPrimAtPath(path).IsValid():
        raise ValueError(f"no object named {name!r}; have {[p.GetName() for p in _bodies()]}")
    stage.RemovePrim(path)
    return {"removed": name}
