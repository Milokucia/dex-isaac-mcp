"""MCP bridge to the Isaac Sim daemon.

Runs on the HOST (plain python + the `mcp` package, no isaaclab). It forwards
JSON over the Unix socket that scripts/simd.py listens on inside the
container, so every tool call lands in a Kit process that is already up — a
parameter change costs a frame instead of a full Kit relaunch.

Register with Claude Code (from the repo root):

    claude mcp add isaac -- python -m dex_isaac_mcp

Nothing here starts Kit until sim_up is called.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent

from . import training
from .protocol import DEFAULT_SOCKET, Client, ProtocolError

_COMPOSE_DIR = training.COMPOSE_DIR
_SIMD_SERVICE = os.environ.get("ISAAC_MCP_SIMD_SERVICE", "simd")
_SIMD_CONTAINER = "isaacmcp-simd"
# Robot config sim_up loads when none is passed (container path, relative to the repo).
_DEFAULT_ROBOT = os.environ.get("ISAAC_MCP_ROBOT")

mcp = MCPServer(
    name="isaac",
    instructions=(
        "Drive a live Isaac Sim session of one articulated robot. A daemon keeps Kit up "
        "between calls, so parameter changes cost a frame instead of a relaunch. Call "
        "sim_status first; if it reports down, call sim_up.\n\n"
        "Joint targets are NORMALIZED: 0 = a joint's lower limit, 1 = its upper limit. "
        "For requests like 'go home' or 'make a fist', prefer sim_set_pose with a pose "
        "from sim_list_poses over guessing raw values, then sim_step and sim_screenshot "
        "to verify. Add props (a table, a ball to grasp) with sim_spawn_object and read "
        "where they ended up with sim_list_objects.\n\n"
        "stiffness, damping, effort and coupling ratios are live (sim_set_params, "
        "sim_set_coupling, sim_sweep). The USD, solver iterations and self-collision are "
        "spawn properties: change them with sim_reload.\n\n"
        "Training runs (train_*) are separate containers, not the daemon: train_start, "
        "poll train_status/train_metrics, read train_logs, train_stop. They keep running "
        "after this MCP session ends."
    ),
)


# Every deliberate failure is a ToolError: MCP passes its text to the client.
# Any other exception reaches the client only as "Error executing tool <name>",
# which would throw away every explanation written below.


class DaemonError(ToolError):
    """The daemon answered with an error (e.g. a diverging solver mid-sweep)."""


def _call(cmd: str, **args: Any) -> Any:
    """One request against the daemon, with a legible error if it is not up."""
    try:
        with Client(DEFAULT_SOCKET) as client:
            return client.call(cmd, **args)
    except FileNotFoundError:
        raise ToolError(f"no daemon socket at {DEFAULT_SOCKET}. Start it with sim_up.") from None
    except ConnectionRefusedError:
        raise ToolError(f"stale socket at {DEFAULT_SOCKET} — the daemon died without "
                        "cleaning up. Call sim_up to restart it.") from None
    except ProtocolError as exc:
        raise DaemonError(str(exc)) from None


def _training(fn, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except training.TrainingError as exc:
        raise ToolError(str(exc)) from None


# ---- lifecycle ---------------------------------------------------------


@mcp.tool()
def sim_status() -> dict[str, Any]:
    """Report whether the sim daemon is up, its robot, driven joints, poses and parameters."""
    try:
        return {"up": True, **_call("status")}
    except ToolError as exc:
        return {"up": False, "reason": str(exc)}


@mcp.tool()
def sim_up(robot: str | None = None, usd: str | None = None, gui: bool = True,
           ground: bool = True, timeout: float = 300.0,
           extra_args: list[str] | None = None) -> dict[str, Any]:
    """Start the daemon container and wait until it answers. Idempotent.

    robot: a robot config JSON, path relative to the repo root
    (e.g. examples/robots/franka.json); defaults to $ISAAC_MCP_ROBOT. usd: a USD path instead of / overriding
    the config's (every joint driven if no config). gui=True is needed for
    sim_screenshot; the host must have run `xhost +local:docker` once.
    ground=False spawns without a floor (use for range tests).
    extra_args go to scripts/simd.py verbatim (e.g. ["--pos-iters", "64"]).
    The first start downloads Nucleus assets and builds shader caches: minutes.
    """
    try:
        return {"already_up": True, **_call("status")}
    except ToolError:
        pass
    if not (_COMPOSE_DIR / "docker-compose.yaml").is_file():
        raise ToolError(
            f"no docker-compose.yaml in {_COMPOSE_DIR}. The daemon runs from a clone of "
            "https://github.com/Milokucia/dex-isaac-mcp: set ISAAC_MCP_HOME to its path "
            "(or ISAAC_MCP_COMPOSE_DIR to its docker/ dir).")

    # A socket left by a hard-killed daemon would make the wait succeed against nothing.
    sock = Path(DEFAULT_SOCKET)
    if sock.exists():
        sock.unlink()
    # `run --name` containers are ours, not compose's; clear a dead one.
    subprocess.run(["docker", "rm", "-f", _SIMD_CONTAINER], capture_output=True)

    cmd = ["docker", "compose", "run", "-d", "--rm", "--name", _SIMD_CONTAINER,
           _SIMD_SERVICE, "scripts/simd.py", "--gui" if gui else "--headless"]
    robot = robot or _DEFAULT_ROBOT
    if robot:
        cmd += ["--robot", robot]
    if usd:
        cmd += ["--usd", usd]
    if not ground:
        cmd.append("--no-ground")
    cmd += extra_args or []
    result = subprocess.run(cmd, cwd=_COMPOSE_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise ToolError(f"docker compose run failed: {result.stderr.strip()}")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sock.exists():
            try:
                return {"already_up": False, **_call("status")}
            except (ToolError, OSError):
                pass
        elif subprocess.run(["docker", "inspect", _SIMD_CONTAINER],
                            capture_output=True).returncode != 0:
            # --rm already deleted it, and its traceback with it.
            raise ToolError(
                "the daemon container exited during startup. Re-run it in the foreground "
                f"to see the error: cd {_COMPOSE_DIR} && docker compose run --rm "
                f"{_SIMD_SERVICE} {' '.join(cmd[cmd.index(_SIMD_SERVICE) + 1:])}")
        time.sleep(1.0)
    raise ToolError(f"daemon did not come up within {timeout:.0f}s. "
                       f"Check: docker logs {_SIMD_CONTAINER}")


@mcp.tool()
def sim_down() -> dict[str, Any]:
    """Stop the sim daemon; its container removes itself."""
    try:
        return {"stopped": True, "result": _call("shutdown")}
    except ToolError as exc:
        return {"stopped": False, "reason": str(exc)}


@mcp.tool()
def sim_reload(robot: str | None = None, usd: str | None = None, pos_iters: int | None = None,
               self_collisions: bool | None = None, gui: bool = True,
               ground: bool = True) -> dict[str, Any]:
    """Restart the daemon to change a spawn property: robot, USD, solver iterations, self-collision.

    One Kit start rather than one per experiment; gains stay live via sim_set_params.
    Omitted robot/usd fall back to the defaults, not to what was loaded before.
    """
    sim_down()
    sock = Path(DEFAULT_SOCKET)
    for _ in range(60):  # let the old daemon release the socket before starting anew
        if not sock.exists():
            break
        time.sleep(0.5)
    extra: list[str] = []
    if pos_iters is not None:
        extra += ["--pos-iters", str(int(pos_iters))]
    if self_collisions is False:
        extra.append("--no-self-collision")
    return sim_up(robot=robot, usd=usd, gui=gui, ground=ground, extra_args=extra)


# ---- inspection --------------------------------------------------------


@mcp.tool()
def sim_inspect_joints(usd: str | None = None) -> dict[str, Any]:
    """List a USD's articulation DOFs, joints excluded from the articulation (loop closures), and roots.

    Reads the file, not the live scene, so it shows edits saved from the Isaac
    GUI. Defaults to the loaded USD.
    """
    return _call("inspect_joints", **({"usd": usd} if usd else {}))


@mcp.tool()
def sim_get_joint_state() -> dict[str, Any]:
    """Joint positions (rad/m) and velocities, plus driven joints' normalized positions and limits."""
    return _call("get_joint_state")


@mcp.tool()
def sim_get_stats() -> dict[str, Any]:
    """Per-joint travel and driven-joint mean tracking error since the last reset.

    Travel on a passive joint proves a linkage transmits: with zero stiffness it
    moves only if its constraint moves it.
    """
    return _call("get_stats")


# ---- driving -----------------------------------------------------------


@mcp.tool()
def sim_set_targets(unit: list[float] | None = None,
                    joints: dict[str, float] | None = None) -> dict[str, Any]:
    """Command the driven joints, normalized 0..1 (lower..upper limit).

    Pass `unit` as a full vector in driven order (see sim_status), or `joints`
    to set individual joints by name. Targets persist; advance with sim_step.
    """
    args: dict[str, Any] = {}
    if unit is not None:
        args["unit"] = unit
    if joints is not None:
        args["joints"] = joints
    return _call("set_targets", **args)


@mcp.tool()
def sim_list_poses() -> dict[str, Any]:
    """Named poses from the robot config, as {pose: {joint_pattern: 0..1}}."""
    return _call("list_poses")


@mcp.tool()
def sim_set_pose(name: str, amount: float = 1.0, from_current: bool = False) -> dict[str, Any]:
    """Command a named pose. amount blends toward it: 1 = as authored, 0.5 = halfway.

    Halfway from the lower limits by default, or from the current targets with
    from_current=True. Follow with sim_step, then sim_screenshot to verify.
    """
    return _call("set_pose", name=name, amount=amount, from_current=from_current)


@mcp.tool()
def sim_step(n: int = 60) -> dict[str, Any]:
    """Advance the simulation n physics steps (dt from sim_status, default 1/120 s)."""
    return _call("step", n=n)


@mcp.tool()
def sim_play(playing: bool = True) -> dict[str, Any]:
    """Run the sim continuously in real time (True) or pause it (False)."""
    return _call("play" if playing else "pause")


@mcp.tool()
def sim_wave(n: int = 240) -> dict[str, Any]:
    """Sweep every driven joint through its range for n steps; return travel stats."""
    return _call("wave", n=n)


@mcp.tool()
def sim_range_test(steps: int = 400, target: float = 1.0, reset: bool = True) -> dict[str, Any]:
    """Drive every driven joint from its lower limit toward `target`, report the fraction reached.

    A joint below 0.9 is blocked: self-collision, a binding linkage, or too
    little effort for the load. reset=True first puts every joint at its lower
    limit and verifies it. Start the daemon with ground=False if the robot can
    reach the floor, or the test measures the floor.
    """
    return _call("range_test", steps=steps, target=target, reset=reset)


@mcp.tool()
def sim_set_params(stiffness: float | None = None, damping: float | None = None,
                   effort: float | None = None) -> dict[str, Any]:
    """Change the driven joints' PD gains and effort limit on the live articulation."""
    changes = {k: v for k, v in {"stiffness": stiffness, "damping": damping,
                                 "effort": effort}.items() if v is not None}
    if not changes:
        raise ToolError("pass at least one parameter")
    return _call("set_params", **changes)


@mcp.tool()
def sim_set_coupling(ratios: dict[str, float]) -> dict[str, Any]:
    """Change software-coupling ratios live, keyed by follower joint. 0 releases a follower.

    Couplings are declared in the robot config; a follower is driven to
    ratio x its leader's target.
    """
    return _call("set_coupling", ratios=ratios)


@mcp.tool()
def sim_spawn_object(name: str, shape: str, pos: list[float], size: list[float] | None = None,
                     radius: float | None = None, height: float | None = None,
                     mass: float = 0.1, static: bool = False, kinematic: bool = False,
                     color: list[float] | None = None, friction: float = 0.8,
                     rot: list[float] | None = None, usd: str | None = None,
                     scale: list[float] | None = None) -> dict[str, Any]:
    """Add a prop to the live scene: a primitive or a USD file, with collision.

    shape: cuboid (size=[x,y,z]), sphere (radius), cylinder / capsule / cone
    (radius, height), or usd (usd=<path>, optional scale). Units are meters and kg.
    static=True: a fixed collider (a table, a wall). kinematic=True: a rigid body
    that contact cannot move. rot is a (w, x, y, z) quaternion. color is RGB 0..1.
    Spawns at /World/Objects/<name>; sim_step to let it fall and settle.
    """
    args = {k: v for k, v in dict(
        name=name, shape=shape, pos=pos, size=size, radius=radius, height=height,
        mass=mass, static=static, kinematic=kinematic, color=color, friction=friction,
        rot=rot, usd=usd, scale=scale).items() if v is not None}
    return _call("spawn_object", **args)


@mcp.tool()
def sim_list_objects() -> dict[str, Any]:
    """Every spawned prop with its current world pose (pos in m, rot as w, x, y, z)."""
    return _call("list_objects")


@mcp.tool()
def sim_remove_object(name: str) -> dict[str, Any]:
    """Delete a spawned prop."""
    return _call("remove_object", name=name)


@mcp.tool()
def sim_set_camera(eye: list[float], target: list[float]) -> dict[str, Any]:
    """Point the viewport camera, e.g. eye=[1.2, 1.2, 1.0] target=[0, 0, 0.3]."""
    return _call("set_camera", eye=eye, target=target)


@mcp.tool()
def sim_screenshot() -> ImageContent:
    """Capture the viewport as an image. Requires the daemon started with gui=True."""
    result = _call("screenshot")
    return ImageContent(type="image", data=result["png_base64"], mimeType="image/png")


# ---- sweeps ------------------------------------------------------------


@mcp.tool()
def sim_sweep(param: str, values: list[float], steps: int = 240,
              test: str = "wave") -> dict[str, Any]:
    """Compare several values of one live parameter inside the single session.

    param: stiffness, damping, effort, or "coupling:<follower>". For each value
    it applies the parameter and runs `test` ("wave" or "range") for `steps`.
    The original value is restored afterwards, even on error. A diverging
    solver is recorded as a result, not raised.
    """
    if test not in ("wave", "range"):
        raise ToolError("test must be 'wave' or 'range'")
    status = _call("status")
    if param.startswith("coupling:"):
        follower = param.split(":", 1)[1]
        current = {c["follower"]: c["ratio"] for c in status["couplings"]}
        if follower not in current:
            raise ToolError(f"no coupling for {follower!r}; known: {sorted(current)}")
        original: Any = current[follower]
        apply = lambda v: _call("set_coupling", ratios={follower: v})  # noqa: E731
    elif param in ("stiffness", "damping", "effort"):
        original = status["params"][param]
        apply = lambda v: _call("set_params", **{param: v})  # noqa: E731
    else:
        raise ToolError(f"cannot sweep {param!r} live. Sweepable: stiffness, damping, effort, "
                         "coupling:<follower>. For spawn properties use sim_reload between runs.")

    rows = []
    try:
        for value in values:
            v = float(value)
            apply(v)
            _call("reset_stats")
            try:
                if test == "wave":
                    stats = _call("wave", n=steps)
                    passive = [j for j in stats["joints"] if not j["driven"]]
                    driven = [j for j in stats["joints"] if j["driven"]]
                    rows.append({
                        param: v,
                        "max_passive_travel": max((j["travel"] for j in passive), default=0.0),
                        "mean_driven_err": (sum(j.get("mean_err", 0.0) for j in driven)
                                            / len(driven) if driven else 0.0),
                        "joints": stats["joints"],
                    })
                else:
                    rep = _call("range_test", steps=steps)
                    rows.append({param: v, **{k: rep[k] for k in
                                 ("arrived", "n", "mean_frac", "worst_joint", "worst_frac")},
                                 "joints": rep["joints"]})
            except DaemonError as exc:
                rows.append({param: v, "error": str(exc)})
    finally:
        apply(original)
    return {"param": param, "test": test, "restored_to": original, "results": rows}


# ---- training control ---------------------------------------------------


@mcp.tool()
def train_start(task: str, num_envs: int | None = None, max_iterations: int | None = None,
                checkpoint: str | None = None, seed: int | None = None,
                run_name: str | None = None, extra_args: list[str] | None = None,
                device: int | None = None) -> dict[str, Any]:
    """Launch a headless Isaac Lab training run in its own container; returns at once.

    task is a registered gym id (e.g. Isaac-Cartpole-v0). run_name defaults to
    '<task>_<timestamp>'; keep it to find the run later. extra_args go to the
    training script verbatim (e.g. Hydra overrides). device pins one GPU index.
    Independent of the daemon and of this MCP session.
    """
    return _training(training.start, task=task, num_envs=num_envs,
                     max_iterations=max_iterations, checkpoint=checkpoint, seed=seed,
                     run_name=run_name, extra_args=extra_args, device=device)


@mcp.tool()
def train_list() -> dict[str, Any]:
    """List training runs: live or recent containers, and log directories holding checkpoints."""
    return _training(training.list_runs)


@mcp.tool()
def train_status(run_name: str) -> dict[str, Any]:
    """Container state, checkpoints, and latest TensorBoard scalar values for one run."""
    return _training(training.status, run_name)


@mcp.tool()
def train_logs(run_name: str, tail: int = 200) -> dict[str, Any]:
    """Tail a running training container's stdout/stderr."""
    return _training(training.logs, run_name, tail=tail)


@mcp.tool()
def train_stop(run_name: str) -> dict[str, Any]:
    """Stop a training run's container."""
    return _training(training.stop, run_name)


@mcp.tool()
def train_checkpoints(run_name: str) -> dict[str, Any]:
    """List a run's saved checkpoints with step and size."""
    return _training(training.checkpoints, run_name)


@mcp.tool()
def train_metrics(run_name: str, tag: str | None = None, max_points: int = 200) -> dict[str, Any]:
    """TensorBoard scalars for a run: no tag lists tag names; a tag returns its series, downsampled."""
    return _training(training.metrics, run_name, tag=tag, max_points=max_points)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
