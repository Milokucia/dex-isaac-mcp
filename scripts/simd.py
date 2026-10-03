"""Start the sim daemon: one Kit process that stays up and takes commands.

Runs INSIDE the Isaac Lab container (or any Isaac Lab python):

    scripts/simd.py --robot examples/robots/franka.json --gui
    scripts/simd.py --usd /path/to/robot.usd --headless

then drive it from the host through the MCP server (dex_isaac_mcp/mcp_server.py)
or dex_isaac_mcp.protocol.Client.

Screenshots need a real viewport: pass --gui (and run `xhost +local:docker`
on the host once). Everything else works headless.
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# Kit teardown is hard enough that a block-buffered stdout never flushes, so
# every print from a container-side daemon would be lost on exit.
sys.stdout.reconfigure(line_buffering=True)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dex_isaac_mcp.protocol import DEFAULT_SOCKET  # noqa: E402  (stdlib-only)
from dex_isaac_mcp.robot import Robot  # noqa: E402  (stdlib-only)

parser = argparse.ArgumentParser(description="Persistent Isaac Sim daemon.")
parser.add_argument("--robot", default=None,
                    help="robot config JSON (see dex_isaac_mcp/robot.py). "
                         "Default: examples/robots/franka.json unless --usd is given")
parser.add_argument("--usd", default=None,
                    help="override the config's USD (or use a bare USD with every joint driven)")
parser.add_argument("--socket", default=str(DEFAULT_SOCKET), help="unix socket to listen on")
parser.add_argument("--gui", action="store_true",
                    help="run with a window. Required for screenshots; needs `xhost +local:docker`.")
parser.add_argument("--no-ground", action="store_true",
                    help="spawn without a floor. Use for range tests on anything whose "
                         "links can reach the ground, or the test measures the floor.")
parser.add_argument("--no-self-collision", action="store_true",
                    help="spawn with self-collision off (overrides the USD and the config)")
parser.add_argument("--pos-iters", type=int, default=None,
                    help="articulation solver position iterations (spawn property)")
parser.add_argument("--stiffness", type=float, default=None)
parser.add_argument("--damping", type=float, default=None)
parser.add_argument("--effort", type=float, default=None)
parser.add_argument("--dt", type=float, default=1 / 120, help="physics step, seconds")
parser.add_argument("--sliders", action="store_true",
                    help="show an omni.ui panel with one slider per driven joint. "
                         "Implies --gui and starts the sim playing.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# AppLauncher reads args.headless; --gui is the readable inverse, since a
# daemon defaults to headless but is far more useful with a viewport.
if args.sliders:
    args.gui = True
if args.gui:
    args.headless = False
elif not getattr(args, "headless", False):
    args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Only now is isaaclab importable.
from dex_isaac_mcp.daemon.scene import SceneParams  # noqa: E402
from dex_isaac_mcp.daemon.server import SimDaemon  # noqa: E402


def _load_robot() -> Robot:
    if args.robot:
        robot = Robot.load(args.robot)
    elif args.usd:
        # A bare USD: every joint driven, generic gains.
        robot = Robot(usd=args.usd, name=Path(args.usd).stem)
    else:
        robot = Robot.load(_ROOT / "examples/robots/franka.json")
    if args.usd:
        robot.usd = args.usd
    if args.no_self_collision:
        robot.self_collisions = False
    for key in ("pos_iters", "stiffness", "damping", "effort"):
        value = getattr(args, key)
        if value is not None:
            setattr(robot, key, value)
    return robot


def _attach_sliders(daemon) -> None:
    from dex_isaac_mcp.daemon.sliders import build_slider_window

    daemon._slider_window = build_slider_window(daemon)
    daemon.playing = True  # a slider panel over a paused sim looks broken


def main() -> None:
    robot = _load_robot()
    daemon = SimDaemon(Path(args.socket), robot, SceneParams.from_robot(robot),
                       device=args.device, dt=args.dt, ground=not args.no_ground)
    if args.sliders:
        daemon.on_world_built = _attach_sliders
    try:
        daemon.serve(simulation_app)
    except KeyboardInterrupt:
        daemon.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
