"""One spawned articulation, plus the bookkeeping to drive and measure it.

isaaclab and pxr are imported at module scope, so this module must only be
imported after AppLauncher has started Kit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from pxr import Usd, UsdPhysics

from ..robot import Robot, match


def expand_usd(path: str) -> str:
    """Substitute Isaac's Nucleus placeholders; check local files exist."""
    if "{" in path:
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

        path = path.format(ISAAC_NUCLEUS_DIR=ISAAC_NUCLEUS_DIR,
                           ISAACLAB_NUCLEUS_DIR=ISAACLAB_NUCLEUS_DIR)
    if "://" not in path:
        local = Path(path).resolve()
        if not local.is_file():
            raise FileNotFoundError(f"USD not found: {local}")
        path = str(local)
    return path


@dataclass
class SceneParams:
    """Everything the scene is built from, split by cost.

    hot    — stiffness, damping, effort: written onto the live articulation.
    frozen — usd, pos_iters, self_collisions: properties of how the prim was
             SPAWNED. Replacing a spawned prim needs SimulationContext.stop(),
             which deadlocks when called from inside the daemon's command loop
             (see server.respawn). Restart the daemon to change them.
    """

    usd: str
    pos_iters: int = 32
    # Spawn-time, and it OVERRIDES whatever the USD authored — a stage exported
    # with self-collision off still spawns with it on unless this says so.
    self_collisions: bool = True
    stiffness: float = 100.0
    damping: float = 10.0
    effort: float = 100.0

    HOT = ("stiffness", "damping", "effort")
    FROZEN = ("usd", "pos_iters", "self_collisions")

    @classmethod
    def from_robot(cls, robot: Robot) -> SceneParams:
        return cls(usd=robot.usd, pos_iters=robot.pos_iters,
                   self_collisions=robot.self_collisions, stiffness=robot.stiffness,
                   damping=robot.damping, effort=robot.effort)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_usd(usd_path: str) -> dict[str, Any]:
    """Report a USD's joint topology without building an Articulation.

    Distinguishes articulation DOFs (which need actuator coverage) from joints
    excluded from the articulation (loop closures — constraints, not DOFs).
    The quickest check that edits made in the Isaac GUI actually saved.
    """
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"could not open stage: {usd_path}")

    dofs: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint.GetExcludeFromArticulationAttr().Get():
            excluded.append({
                # Full path, not GetName(): the GUI names every joint it
                # creates "RevoluteJoint", so the leaf name cannot tell them apart.
                "path": str(prim.GetPath()),
                "body0": [p.name for p in joint.GetBody0Rel().GetTargets()],
                "body1": [p.name for p in joint.GetBody1Rel().GetTargets()],
                "axis": prim.GetAttribute("physics:axis").Get(),
            })
        elif prim.GetTypeName() != "PhysicsFixedJoint":
            dofs.append({"name": prim.GetName(), "type": prim.GetTypeName()})

    return {
        "usd": usd_path,
        "dofs": dofs,
        "excluded": excluded,
        "roots": [str(p.GetPath()) for p in stage.Traverse()
                  if p.HasAPI(UsdPhysics.ArticulationRootAPI)],
    }


# Below this fraction of its own commanded travel, a joint is jammed rather
# than merely slow.
ARRIVED = 0.90


class RobotScene:
    """One spawned robot, plus the bookkeeping needed to drive and measure it."""

    def __init__(self, sim: Any, robot: Robot, params: SceneParams,
                 prim_path: str = "/World/Robot"):
        self.sim = sim
        self.robot = robot
        self.params = params
        self.prim_path = prim_path
        self.couplings = list(robot.couplings)
        self.art = self._spawn()
        # bind() must be called after sim.reset(): joint limits come from the
        # PhysX articulation view, which does not exist until then.

    # ---- construction -------------------------------------------------

    def _spawn(self) -> Articulation:
        p, r = self.params, self.robot
        # ImplicitActuatorCfg IS the controller: stiffness/damping become a PD
        # position drive inside PhysX, written over whatever the USD carried.
        actuators = {
            "driven": ImplicitActuatorCfg(
                joint_names_expr=r.driven_joints,
                stiffness=p.stiffness,
                damping=p.damping,
                effort_limit_sim=p.effort,
                velocity_limit_sim=r.velocity,
            ),
        }
        if r.passive_joints:
            actuators["passive"] = ImplicitActuatorCfg(
                joint_names_expr=r.passive_joints,
                stiffness=0.0, damping=0.01, effort_limit_sim=0.0,
            )
        cfg = ArticulationCfg(
            prim_path=self.prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=expand_usd(p.usd),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    # Loop closures are maximal-coordinate constraints and
                    # stretch at low iteration counts; raise this if a
                    # closed-chain mechanism visibly separates.
                    solver_position_iteration_count=p.pos_iters,
                    solver_velocity_iteration_count=r.vel_iters,
                    enabled_self_collisions=p.self_collisions,
                    fix_root_link=r.fix_root_link,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(r.spawn_pos),
                                                     joint_pos=dict(r.init_joint_pos)),
            actuators=actuators,
        )
        return Articulation(cfg)

    def bind(self) -> None:
        """Resolve joint indices, limits and couplings. Call after sim.reset()."""
        names = list(self.art.joint_names)
        driven = match(self.robot.driven_joints, names)
        passive = set(match(self.robot.passive_joints, names))
        driven = [n for n in driven if n not in passive]
        if not driven:
            raise RuntimeError(
                f"driven_joints {self.robot.driven_joints} match no joint; joints={names}")
        self.driven_ids, self.driven_names = self.art.find_joints(driven, preserve_order=True)
        self.lower = self.art.data.soft_joint_pos_limits[0, self.driven_ids, 0]
        self.upper = self.art.data.soft_joint_pos_limits[0, self.driven_ids, 1]
        self._bind_coupling()
        self.reset_stats()

    def _bind_coupling(self) -> None:
        ids: list[int] = []
        src: list[int] = []
        ratios: list[float] = []
        for c in self.couplings:
            if not c.ratio:
                continue
            if c.leader not in self.driven_names:
                raise ValueError(f"coupling leader {c.leader!r} is not a driven joint")
            found, _ = self.art.find_joints([c.follower], preserve_order=True)
            if not found:
                raise ValueError(f"coupling follower {c.follower!r} is not a joint")
            ids.append(found[0])
            src.append(self.driven_names.index(c.leader))
            ratios.append(c.ratio)
        self.coupled_ids = ids
        self.coupled_src = src
        self.coupled_ratio = torch.tensor(ratios, device=self.sim.device)
        if ids:
            # A commanded follower needs real gains; an uncoupled passive one
            # must stay at zero stiffness or it fights its constraint.
            p = self.params
            self.art.write_joint_stiffness_to_sim(p.stiffness, joint_ids=ids)
            self.art.write_joint_damping_to_sim(p.damping, joint_ids=ids)
            self.art.write_joint_effort_limit_to_sim(p.effort, joint_ids=ids)

    def set_coupling(self, ratios: dict[str, float]) -> list[dict[str, Any]]:
        """Change coupling ratios live, keyed by follower name. 0 disables one."""
        by_follower = {c.follower: c for c in self.couplings}
        unknown = set(ratios) - set(by_follower)
        if unknown:
            raise ValueError(f"no coupling for followers {sorted(unknown)}; "
                             f"known={sorted(by_follower)}")
        released = []
        for follower, ratio in ratios.items():
            if not ratio and by_follower[follower].ratio:
                released.append(follower)
            by_follower[follower].ratio = float(ratio)
        if released:
            # An uncommanded follower goes limp, so whatever constraint owns
            # it decides its angle.
            ids, _ = self.art.find_joints(released, preserve_order=True)
            self.art.write_joint_stiffness_to_sim(0.0, joint_ids=ids)
            self.art.write_joint_damping_to_sim(0.01, joint_ids=ids)
            self.art.write_joint_effort_limit_to_sim(0.0, joint_ids=ids)
        self._bind_coupling()
        return [vars(c) for c in self.couplings]

    # ---- driving ------------------------------------------------------

    @property
    def num_driven(self) -> int:
        return len(self.driven_ids)

    def unit_to_target(self, unit: torch.Tensor) -> torch.Tensor:
        """Map 0..1 per driven joint into that joint's own limit range."""
        return (self.lower + unit * (self.upper - self.lower)).unsqueeze(0)

    def target_to_unit(self, q: torch.Tensor) -> torch.Tensor:
        span = (self.upper - self.lower).clamp_min(1e-9)
        return ((q - self.lower) / span).clamp(0.0, 1.0)

    def wave(self, t: float) -> torch.Tensor:
        """Scripted 0.5 Hz sweep of every driven joint, phase-offset per joint."""
        phase = torch.tensor([t * math.pi - 0.35 * i for i in range(self.num_driven)],
                             device=self.sim.device)
        return 0.5 * (1.0 - torch.cos(phase))

    def apply(self, unit: torch.Tensor) -> torch.Tensor:
        target = self.unit_to_target(unit)
        self.art.set_joint_position_target(target, joint_ids=self.driven_ids)
        if self.coupled_ids:
            self.art.set_joint_position_target(
                self.coupled_ratio * target[:, self.coupled_src], joint_ids=self.coupled_ids)
        self.art.write_data_to_sim()
        self._last_target = target
        return target

    def step(self, dt: float) -> None:
        self.sim.step()
        self.art.update(dt)
        q = self.art.data.joint_pos[0]
        self.q_min = torch.minimum(self.q_min, q)
        self.q_max = torch.maximum(self.q_max, q)
        if self._last_target is not None:
            self.err_sum += (self._last_target[0] - q[self.driven_ids]).abs()
        self.steps += 1

    # ---- measurement --------------------------------------------------

    def reset_stats(self) -> None:
        q = self.art.data.joint_pos[0]
        self.q_min = q.clone()
        self.q_max = q.clone()
        self.err_sum = torch.zeros(self.num_driven, device=self.sim.device)
        self.steps = 0
        self._last_target: torch.Tensor | None = None

    def stats(self) -> dict[str, Any]:
        """Per-joint travel and tracking error since the last reset.

        Travel on an undriven joint is the proof a linkage transmits: with zero
        stiffness it only moves if a constraint (or gravity) moves it. A driven
        joint with a large mean error means the gains are soft or it is blocked.
        """
        rows = []
        for i, name in enumerate(self.art.joint_names):
            row: dict[str, Any] = {
                "joint": name,
                "travel": (self.q_max[i] - self.q_min[i]).item(),
                "driven": i in self.driven_ids,
            }
            if row["driven"] and self.steps:
                row["mean_err"] = self.err_sum[self.driven_ids.index(i)].item() / self.steps
            rows.append(row)
        return {"steps": self.steps, "joints": rows}

    def joint_state(self) -> dict[str, Any]:
        q = self.art.data.joint_pos[0]
        return {
            "names": list(self.art.joint_names),
            "pos": q.tolist(),
            "vel": self.art.data.joint_vel[0].tolist(),
            "driven_names": list(self.driven_names),
            "driven_unit": self.target_to_unit(q[self.driven_ids]).tolist(),
            "driven_lower": self.lower.tolist(),
            "driven_upper": self.upper.tolist(),
        }

    # ---- range test ---------------------------------------------------

    def go_to_lower(self, dt: float, settle: int = 120, drive_steps: int = 400,
                    pos_tol: float = 5e-3) -> dict[str, Any]:
        """Put every driven joint at its lower limit and CONFIRM it got there.

        Writes the state directly, then lets the solver settle any passive
        linkage around it. Teleporting is fine here — the point is an identical
        start for every variant. Waiting on velocity instead does NOT work: a
        joint held against its limit reports the drive's attempted velocity
        forever, not motion.
        """
        q = self.art.data.joint_pos[0].clone()
        q[self.driven_ids] = self.lower
        v = torch.zeros_like(self.art.data.joint_vel[0])
        self.art.write_joint_state_to_sim(q.unsqueeze(0), v.unsqueeze(0))

        zeros = torch.zeros(self.num_driven, device=self.sim.device)
        for _ in range(settle):
            self.apply(zeros)
            self.step(dt)
        res = float((self.art.data.joint_pos[0, self.driven_ids] - self.lower).abs().max())
        if res < pos_tol:
            return {"ok": True, "steps": settle, "pos_residual": res}

        # Contact is pushing a joint off the limit; fall back to driving.
        for _ in range(drive_steps):
            self.apply(zeros)
            self.step(dt)
        res = float((self.art.data.joint_pos[0, self.driven_ids] - self.lower).abs().max())
        return {"ok": res < pos_tol, "steps": settle + drive_steps, "pos_residual": res}

    def range_test(self, steps: int, dt: float, target: float = 1.0,
                   reset: bool = True) -> dict[str, Any]:
        """Drive every driven joint from its lower limit toward `target`, report how far each got.

        `frac` is the fraction of COMMANDED travel (lower limit -> target),
        a fixed denominator. Measuring from wherever the joint happened to
        start makes results depend on the previous test's end state.
        A joint below ARRIVED is blocked: self-collision, a binding linkage,
        or an effort limit too low for the load.
        """
        reset_info = self.go_to_lower(dt) if reset else None
        start = self.art.data.joint_pos[0, self.driven_ids].clone()
        cmd = torch.full((self.num_driven,), float(target), device=self.sim.device)
        for _ in range(int(steps)):
            self.apply(cmd)
            self.step(dt)

        q = self.art.data.joint_pos[0, self.driven_ids]
        goal = self.lower + float(target) * (self.upper - self.lower)
        rows = []
        for i, name in enumerate(self.driven_names):
            lo, g, f = float(self.lower[i]), float(goal[i]), float(q[i])
            span = g - lo
            frac = (f - lo) / span if abs(span) > 1e-6 else 1.0
            rows.append({"joint": name, "start": float(start[i]), "final": f,
                         "target": g, "frac": frac,
                         "status": "ok" if frac >= ARRIVED else "short" if frac >= 0.5 else "jammed"})

        worst = min(rows, key=lambda r: r["frac"])
        return {
            "arrived": sum(r["frac"] >= ARRIVED for r in rows), "n": len(rows),
            "mean_frac": sum(r["frac"] for r in rows) / len(rows),
            "worst_joint": worst["joint"], "worst_frac": worst["frac"],
            "reset": reset_info, "joints": rows,
        }
