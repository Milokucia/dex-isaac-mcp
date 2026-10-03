"""Robot description: which USD to spawn, which joints to drive, named poses.

Stdlib-only, so the daemon (inside Kit) and any host-side tooling can both
load it. A robot is a JSON file; every field is optional except `usd`:

    {
      "name": "franka",
      "usd": "{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
      "fix_root_link": true,
      "spawn_pos": [0, 0, 0],
      "init_joint_pos": {"panda_joint4": -2.81, ".*": 0.0},
      "driven_joints": ["panda_joint.*", "panda_finger_joint.*"],
      "passive_joints": [],
      "couplings": [{"leader": "a", "follower": "b", "ratio": 1.0}],
      "actuator": {"stiffness": 400, "damping": 40, "effort": 87, "velocity": 2.0},
      "solver": {"pos_iters": 32, "vel_iters": 4, "self_collisions": true},
      "camera": {"eye": [1.5, 1.5, 1.2], "target": [0, 0, 0.4]},
      "poses": {"home": {"panda_joint.*": 0.5}}
    }

Joint patterns are regular expressions matched against the articulation's
joint names (full match, as in Isaac Lab's `joint_names_expr`).

`init_joint_pos` is in the joints' own units (rad / m), passed to Isaac Lab's
InitialStateCfg; it must lie inside every joint's limits or spawning fails.

`usd` may use {ISAAC_NUCLEUS_DIR} / {ISAACLAB_NUCLEUS_DIR}; a relative path is
resolved against the JSON file's own directory.

Poses are partial maps from joint pattern to a NORMALIZED target: 0 = the
joint's lower limit, 1 = its upper limit. The same convention as
sim_set_targets, so a pose is portable across robots without knowing radians.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Coupling:
    """A software mimic joint: follower target = ratio x leader target (radians).

    Applied as a drive target, not a PhysX mimic constraint, so a large ratio
    cannot blow up the solver. ratio 0 disables it; the follower then falls
    back to whatever `passive_joints` / the USD says.
    """

    leader: str
    follower: str
    ratio: float = 1.0


@dataclass
class Robot:
    usd: str
    name: str = "robot"
    fix_root_link: bool = True
    spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_joint_pos: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
    # Joints that take position commands. Default: every joint.
    driven_joints: list[str] = field(default_factory=lambda: [".*"])
    # Joints whose angle is owned by a constraint (a closed-chain linkage, a
    # loop joint). They get a zero-stiffness drive so the drive does not fight
    # the constraint — a live PD drive there makes the mechanism jitter.
    passive_joints: list[str] = field(default_factory=list)
    couplings: list[Coupling] = field(default_factory=list)
    stiffness: float = 100.0
    damping: float = 10.0
    effort: float = 100.0
    velocity: float | None = None
    pos_iters: int = 32
    vel_iters: int = 4
    self_collisions: bool = True
    camera_eye: tuple[float, float, float] = (1.0, 1.0, 1.0)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.3)
    poses: dict[str, dict[str, float]] = field(default_factory=dict)
    source: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any], base_dir: Path | None = None) -> Robot:
        known = {"name", "usd", "fix_root_link", "spawn_pos", "init_joint_pos", "driven_joints",
                 "passive_joints", "couplings", "actuator", "solver", "camera", "poses"}
        unknown = set(d) - known
        if unknown:
            # A typo'd key silently falling back to a default is the worst
            # failure a config file can have.
            raise ValueError(f"unknown robot config keys: {sorted(unknown)}")
        if "usd" not in d:
            raise ValueError("robot config needs a 'usd'")

        act = d.get("actuator", {})
        sol = d.get("solver", {})
        cam = d.get("camera", {})
        for section, keys, val in (
            ("actuator", {"stiffness", "damping", "effort", "velocity"}, act),
            ("solver", {"pos_iters", "vel_iters", "self_collisions"}, sol),
            ("camera", {"eye", "target"}, cam),
        ):
            bad = set(val) - keys
            if bad:
                raise ValueError(f"unknown {section} keys: {sorted(bad)}")

        usd = d["usd"]
        if base_dir is not None and "{" not in usd and "://" not in usd and not Path(usd).is_absolute():
            usd = str((base_dir / usd).resolve())

        defaults = cls(usd=usd)
        return cls(
            usd=usd,
            name=d.get("name", defaults.name),
            fix_root_link=bool(d.get("fix_root_link", True)),
            spawn_pos=tuple(d.get("spawn_pos", defaults.spawn_pos)),
            init_joint_pos={k: float(v) for k, v in
                            d.get("init_joint_pos", defaults.init_joint_pos).items()},
            driven_joints=list(d.get("driven_joints", defaults.driven_joints)),
            passive_joints=list(d.get("passive_joints", [])),
            couplings=[Coupling(**c) for c in d.get("couplings", [])],
            stiffness=float(act.get("stiffness", defaults.stiffness)),
            damping=float(act.get("damping", defaults.damping)),
            effort=float(act.get("effort", defaults.effort)),
            velocity=act.get("velocity"),
            pos_iters=int(sol.get("pos_iters", defaults.pos_iters)),
            vel_iters=int(sol.get("vel_iters", defaults.vel_iters)),
            self_collisions=bool(sol.get("self_collisions", True)),
            camera_eye=tuple(cam.get("eye", defaults.camera_eye)),
            camera_target=tuple(cam.get("target", defaults.camera_target)),
            poses={k: {j: float(v) for j, v in p.items()} for k, p in d.get("poses", {}).items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> Robot:
        path = Path(path).resolve()
        robot = cls.from_dict(json.loads(path.read_text()), base_dir=path.parent)
        robot.source = str(path)
        return robot


def match(patterns: list[str], names: list[str]) -> list[str]:
    """Names fully matching any pattern, in `names` order."""
    compiled = [re.compile(p) for p in patterns]
    return [n for n in names if any(c.fullmatch(n) for c in compiled)]


def match_ordered(patterns: list[str], names: list[str]) -> list[str]:
    """Names fully matching any pattern, in PATTERN order (then `names` order within one).

    Used for the driven joints, so the order a config lists them in is the
    order of every `unit` vector, not whatever order the USD happens to store.
    """
    out: list[str] = []
    for p in patterns:
        c = re.compile(p)
        out += [n for n in names if c.fullmatch(n) and n not in out]
    return out


def resolve_pose(pose: dict[str, float], names: list[str], amount: float = 1.0,
                 base: dict[str, float] | None = None) -> dict[str, float]:
    """Expand a pose's joint patterns to concrete names, blended toward `base`.

    amount=1 is the pose as written; amount=0.5 is halfway from `base` (default
    all zeros, i.e. every joint at its lower limit). Later patterns win, so a
    pose can say {".*": 0, "thumb.*": 1}. A pattern that matches nothing is an
    error: it is almost always a typo, and silently ignoring it would leave the
    joint wherever it was.
    """
    base = base or {}
    out: dict[str, float] = {}
    for pattern, value in pose.items():
        hit = match([pattern], names)
        if not hit:
            raise ValueError(f"pose pattern {pattern!r} matches no driven joint; driven={names}")
        for n in hit:
            b = base.get(n, 0.0)
            out[n] = b + (value - b) * amount
    return out
