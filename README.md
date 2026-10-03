# dex-isaac-mcp

<!-- mcp-name: io.github.Milokucia/dex-isaac-mcp -->

![An agent driving the Allegro hand through dex-isaac-mcp](https://raw.githubusercontent.com/Milokucia/dex-isaac-mcp/main/docs/demo.gif)

*Recorded headless with the server's own tools (`sim_frame_robot`, `sim_set_backdrop`, `sim_record_start`), driving the Allegro hand example. Each caption is the request and the tool call it became.*

An [MCP](https://modelcontextprotocol.io) server that lets an AI agent (Claude Code, or any MCP client) drive a **live, persistent Isaac Sim session** and launch **Isaac Lab training runs**.

Kit takes tens of seconds to boot. If every experiment is a fresh launch, most of your time goes to waiting. Here Kit starts once inside a daemon and stays up. Each tool call lands in that running session, so changing a gain, stepping physics or taking a screenshot costs a frame, not a relaunch.

```
 MCP client ──stdio──▶ dex_isaac_mcp (host, plain python)
                          │  newline-delimited JSON over a Unix socket
                          ▼
                       scripts/simd.py (Isaac Lab container, Kit stays up)
                          └─ one articulation, described by a robot JSON
```

- **Any articulated robot.** Point it at a USD and a small JSON config. Franka and Allegro examples are included.
- **Normalized joint control.** Targets are `0..1`, where 0 is a joint's lower limit and 1 its upper limit, so agents don't need to know radians or meters.
- **Named poses** per robot (`home`, `fist`, …), which can be blended part-way.
- **Headless capture and GIF recording** from an auto-framed camera, with captions. The clip above was made this way.
- **Props:** spawn a table, a ball or a USD into the running scene and read back where they settle.
- **Measurements:** joint state, per-joint travel and tracking error, a range test that finds blocked joints, and parameter sweeps inside one session.
- **Training control:** each run is a detached `docker compose run`. You can poll its status, TensorBoard scalars, checkpoints and logs.

## How it differs from other Isaac Sim MCP servers

[NVIDIA's official Isaac Sim MCP](https://docs.isaacsim.omniverse.nvidia.com/latest/development_tools/isaac_sim_mcp.html) is a documentation search for coding assistants, and works well alongside this one. Servers like [`isaacsim-mcp-server`](https://github.com/InstinctRobotics/isaacsim-mcp-server) and [`omni-mcp/isaac-sim-mcp`](https://github.com/omni-mcp/isaac-sim-mcp) run inside the Isaac Sim GUI and cover scene building broadly. This one runs headless, as a daemon in Docker, and focuses on tuning, measuring and training one robot.

| | NVIDIA Isaac Sim MCP | In-GUI servers (`isaacsim-mcp-server`, `omni-mcp`) | **dex-isaac-mcp** |
|---|---|---|---|
| What it is | Docs and code search | Kit extension inside a running Isaac Sim | External daemon; Kit stays up in a container |
| Controls the simulation | No | Yes | Yes |
| Headless / remote GPU box | n/a | Needs the GUI | Yes: headless capture, Docker, Unix socket |
| Scene building (lights, materials, sensors, asset library) | n/a | Broad | Minimal: primitive and USD props |
| Robot setup | n/a | Built-in robot library | Any USD plus a small JSON config (joints, poses, couplings, gains) |
| Measurement | n/a | Joint and prim state | Range tests that find blocked joints, tracking error, sweeps that restore the original value |
| Live tuning | n/a | Not documented | PD gains, effort, software mimic couplings |
| Isaac Lab training | n/a | Not documented | Launch, poll, TensorBoard metrics, checkpoints, stop |
| Recording | n/a | Camera captures | Captioned GIFs from an auto-framed camera, headless |
| Platform | Any | Wherever Isaac Sim runs | Linux, NVIDIA GPU, Docker, NGC |

**Pick an in-GUI server** to build a scene by talking to it, interactively, with a wide robot and asset library.

**Pick this one** to have an agent tune, debug and train a specific robot (your own, from a USD), unattended or on a remote box, with numbers it can act on instead of only screenshots.

## Requirements

- Linux with an NVIDIA GPU that Isaac Sim supports, plus the NVIDIA Container Toolkit
- Docker with the Compose plugin
- An NGC login to pull the Isaac Lab image: `docker login nvcr.io` (user `$oauthtoken`, password: your NGC API key)
- Python ≥ 3.10 on the host, for the MCP server only

## Quick start

```bash
git clone <this repo> dex-isaac-mcp && cd dex-isaac-mcp

# 1. Build the image (Isaac Lab 2.3.2 base, pinned by digest)
cd docker && docker compose build && cd ..

# 2. Install the host-side server, editable so it finds this clone
pip install -e .            # add [metrics] for train_metrics: pip install -e '.[metrics]'

# 3. Register it with Claude Code
claude mcp add isaac -- python -m dex_isaac_mcp

# 4. Allow the container to open windows (once per login; needed for screenshots)
xhost +local:docker
```

Then ask the agent something like *"start the sim with the Franka, move it to ready and show me a screenshot."* It will call `sim_up`, `sim_set_pose`, `sim_step` and `sim_screenshot`.

**From PyPI instead:** `pip install dex-isaac-mcp`. The daemon still runs from a clone (it needs `docker/` and `scripts/simd.py`), so point the server at it:

```bash
claude mcp add isaac -e ISAAC_MCP_HOME=/abs/path/dex-isaac-mcp -- dex-isaac-mcp
```

Other MCP clients can launch `python -m dex_isaac_mcp` (or the `dex-isaac-mcp` script) over stdio.

The first `sim_up` takes several minutes: Kit builds its shader cache and downloads Nucleus assets. Later starts are much faster.

### Running the daemon by hand

`sim_up` deletes its container (`--rm`) when the daemon exits, so a crash on startup takes its traceback with it. To see the error, run the daemon in the foreground:

```bash
cd docker
docker compose run --rm simd scripts/simd.py --gui --robot examples/robots/franka.json
docker compose run --rm simd scripts/simd.py --headless --usd /path/in/container/robot.usd
docker compose run --rm simd scripts/simd.py --sliders --robot examples/robots/allegro_hand.json
```

`--sliders` opens an omni.ui panel with one slider per driven joint. The socket stays live alongside it.

## Tools

### Session

| Tool | What it does |
|---|---|
| `sim_status` | Whether the daemon is up, plus its robot, driven joints, poses, couplings and gains |
| `sim_up` | Start the daemon container (`robot`, `usd`, `gui`, `ground`, `extra_args`) and wait until it answers. Does nothing if it is already up |
| `sim_down` | Stop the daemon |
| `sim_reload` | Restart with a different spawn property: robot, USD, `pos_iters`, self-collision |

### Inspect

| Tool | What it does |
|---|---|
| `sim_inspect_joints` | A USD's articulation DOFs, its loop-closure joints (excluded from the articulation) and its articulation roots. Reads the file, so it shows edits saved from the GUI |
| `sim_get_joint_state` | Positions and velocities, plus driven joints' normalized positions and limits |
| `sim_get_stats` | Per-joint travel and mean tracking error since the last reset |
| `sim_screenshot` | Viewport capture, returned as an image. Needs `gui=True` |
| `sim_set_camera` | Point the viewport camera |

### Drive

| Tool | What it does |
|---|---|
| `sim_set_targets` | Normalized targets, as a full vector or `{joint: value}` |
| `sim_list_poses` / `sim_set_pose` | Named poses from the robot config. `amount` blends toward a pose, starting from the lower limits or from the current targets |
| `sim_step` | Advance N physics steps (default dt 1/120 s) |
| `sim_play` | Run continuously, or pause |
| `sim_wave` | Sweep every driven joint through its range and return travel stats |
| `sim_range_test` | Drive every joint from its lower limit toward a target and report the fraction of travel reached. Below 0.9 counts as blocked (self-collision, a binding linkage, too little effort) |

### Scene

| Tool | What it does |
|---|---|
| `sim_spawn_object` | Add a prop to the live scene: `cuboid`, `sphere`, `cylinder`, `capsule`, `cone` or a USD file, with collision. `static` for a fixed table or wall, `kinematic` for a body contact cannot move |
| `sim_list_objects` | Every prop's current world pose |
| `sim_remove_object` | Delete a prop |

### Capture and record

These work headless, with no GUI or viewport. They use a dedicated camera that is independent of the GUI view.

| Tool | What it does |
|---|---|
| `sim_frame_robot` | Aim the capture camera so the whole robot fills the frame, from a given direction. `raise_frac` leaves room for captions |
| `sim_set_capture_camera` | Place the capture camera by hand |
| `sim_set_backdrop` | A plain colored panel behind the robot. Use it with `ground=False` for clean footage |
| `sim_capture` | One frame, returned as an image |
| `sim_record_start` / `sim_record_caption` / `sim_record_stop` | Record every Nth physics step, with a caption drawn on each frame, to an animated GIF under `.cache/recordings/`. Anything that steps the sim is recorded |

### Tune

| Tool | What it does |
|---|---|
| `sim_set_params` | Live `stiffness` / `damping` / `effort` on the driven joints |
| `sim_set_coupling` | Live software-mimic ratios, keyed by follower joint |
| `sim_sweep` | Try several values of one live parameter (`stiffness`, `damping`, `effort`, `coupling:<follower>`), running `wave` or `range` for each. Restores the original value afterwards. A diverging solver is recorded as a result rather than raised |

### Train

| Tool | What it does |
|---|---|
| `train_start` | Launch a headless training run in its own container and return at once. `extra_args` go to the script verbatim (e.g. Hydra overrides). `device` pins a GPU |
| `train_list` | Running and recent runs, and log directories holding checkpoints |
| `train_status` | Container state, checkpoints and latest scalars |
| `train_logs` | Tail a running container's output |
| `train_metrics` | List TensorBoard tags, or get a downsampled series for one tag |
| `train_checkpoints` | Checkpoints with step and size |
| `train_stop` | Stop a run |

Training runs do not depend on the daemon or on the MCP session. They keep going after the client disconnects.

## Robot config

A robot is one JSON file. Only `usd` is required. Unknown keys are rejected, so a typo fails loudly instead of quietly falling back to a default.

```json
{
  "name": "franka",
  "usd": "{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
  "fix_root_link": true,
  "spawn_pos": [0, 0, 0],
  "init_joint_pos": {"panda_joint4": -2.81, "panda_joint6": 3.04, ".*": 0.0},
  "driven_joints": ["panda_joint[1-7]", "panda_finger_joint.*"],
  "passive_joints": [],
  "couplings": [{"leader": "joint_a", "follower": "joint_b", "ratio": 1.0}],
  "actuator": {"stiffness": 400, "damping": 40, "effort": 87, "velocity": null},
  "solver": {"pos_iters": 32, "vel_iters": 4, "self_collisions": true},
  "camera": {"eye": [1.8, 1.8, 1.4], "target": [0, 0, 0.4]},
  "poses": {"ready": {"panda_joint[1357]": 0.5, "panda_finger_joint.*": 1.0}}
}
```

| Key | Meaning |
|---|---|
| `usd` | Local path (relative paths resolve from the JSON's own directory), a URL, or a path using `{ISAAC_NUCLEUS_DIR}` / `{ISAACLAB_NUCLEUS_DIR}` |
| `init_joint_pos` | Spawn pose in the joints' own units (rad / m), keyed by regex. **It must lie inside every joint's limits or spawning fails.** The default is all zeros, which is out of range for e.g. Franka's joint 4 |
| `driven_joints` | Regexes (full match) for joints that take commands. Default `.*` |
| `passive_joints` | Joints whose angle is owned by a constraint, such as a closed-chain linkage. They get a zero-stiffness drive, because a live PD drive fights the constraint and the mechanism jitters |
| `couplings` | Software mimic joints: follower target = `ratio` × leader target. They are applied as drive targets rather than PhysX mimic constraints, so a large ratio cannot blow up the solver |
| `actuator` | Implicit PD gains and effort/velocity limits for the driven joints. Live-tunable |
| `solver` | Spawn properties. Changing them needs `sim_reload` |
| `poses` | `{name: {joint_regex: 0..1}}`. Later patterns win, so `{".*": 0, "thumb.*": 1}` works. A pattern that matches no driven joint is an error |

## Using your own robot without forking

Keep the robot config and assets in your own repo, and add them to the container with a compose override that you list in `COMPOSE_FILE`. Set it in the MCP server's environment, using absolute paths:

```yaml
# my-robot/mcp-compose.yaml
services:
  simd:
    volumes:
      - /abs/path/my-robot:/workspace/my-robot
```

```bash
claude mcp add isaac \
  -e COMPOSE_FILE=/abs/path/dex-isaac-mcp/docker/docker-compose.yaml:/abs/path/my-robot/mcp-compose.yaml \
  -e ISAAC_MCP_ROBOT=/workspace/my-robot/robot.json \
  -- python -m dex_isaac_mcp
```

## Training defaults

Out of the box, `train_start` runs Isaac Lab's stock skrl script inside the `isaac-lab` service. It tags the run name onto the log directory (`logs/skrl/<experiment>/<timestamp>_ppo_torch_<run_name>/`), which the other `train_*` tools use to find the run. To use your own launcher, set these in the environment the MCP server starts in:

| Variable | Default |
|---|---|
| `ISAAC_MCP_HOME` | the clone this package was installed from (editable install); **required for a PyPI install** |
| `ISAAC_MCP_COMPOSE_DIR` | `<repo>/docker` |
| `ISAAC_MCP_TRAIN_SERVICE` | `isaac-lab` |
| `ISAAC_MCP_TRAIN_WORKDIR` | `/workspace/isaaclab` |
| `ISAAC_MCP_TRAIN_SCRIPT` | `scripts/reinforcement_learning/skrl/train.py` |
| `ISAAC_MCP_LOGS_DIR` | `<repo>/logs` (mounted at `/workspace/isaaclab/logs`) |
| `ISAAC_MCP_RUN_NAME_ARG` | `agent.agent.experiment.experiment_name={run_name}` (empty = don't pass one) |
| `ISAAC_MCP_TRAIN_ARGS` | `hydra.run.dir=/tmp/hydra hydra.output_subdir=null`: appended to every run. The stock script otherwise writes Hydra's `outputs/` into the root-owned `/workspace/isaaclab` and dies. Set it empty for a non-Hydra script |
| `ISAAC_MCP_SIMD_SERVICE` | `simd` |
| `ISAAC_MCP_ROBOT` | robot config `sim_up` loads when none is given (unset: Franka example) |
| `ISAAC_MCP_SOCKET` | `<repo>/.cache/simd.sock` |

The script must accept `--task`, `--headless` and, when given, `--num_envs`, `--seed`, `--max_iterations` and `--checkpoint`. Tasks from your own extension need to be importable inside the container, either installed into the image or mounted.

## Design notes

These are the constraints the code is built around. Most were learned by breaking them.

- **Every Kit call happens on the main thread.** Kit, PhysX and USD are not thread-safe. Socket threads only parse JSON and queue requests, and the main loop executes them between physics steps. Answering from a reader thread appears to work, then corrupts the stage under load.
- **Spawn properties are frozen.** Replacing a spawned articulation needs `SimulationContext.stop()`, which blocks on a timeline event that only advances while the Kit loop pumps. A command runs *on* that loop, so the call never returns. `omni.usd` `new_stage()` has the same trap. So the USD, solver iterations and self-collision need a restart (`sim_reload`), and gains stay live.
- **A Unix socket, not TCP.** The repo is bind-mounted and the container runs as the host uid, so the host sees the socket file directly, with no port mapping. Paths are capped at 107 bytes (`AF_UNIX`). If your checkout is deep, set `ISAAC_MCP_SOCKET`.
- **The host side imports no Isaac code.** `protocol.py`, `robot.py` and `training.py` are stdlib-only. The MCP server adds only `mcp`. Nothing on the host needs isaaclab, torch or a GPU.
- **Cache directories are committed with `.gitkeep`.** If Docker auto-creates a bind-mount source, it is root-owned, and Kit then dies with `registry cache path is not set` before any script runs.
- **The base image is pinned by digest.** A re-pulled tag once shipped `/isaac-sim` as mode 750, and every non-root container lost its Python.
- **Recorded GIFs are stabilized.** The renderer's denoiser shimmers: between two frames of a motionless scene, about 9% of background pixels change slightly, and a GIF re-encodes every one of them. Holding sub-threshold changes and using one shared palette took a 9-second clip from 15 MB to 1.2 MB.
- **The daemon always renders, even headless** (`enable_cameras`). Without rendering, PhysX never registers a prop spawned at runtime. Prop poses are read from fabric, because the USD transform and the PhysX CPU query both stay at the spawn pose, and creating a PhysX tensor view mid-simulation crashes CUDA.
- **No floor for range tests** (`ground=False`) on anything whose links can reach the ground. Otherwise the test measures the floor, not the robot.

## Development

```bash
python -m unittest discover tests   # host-side tests: no Isaac, no GPU
ruff check .
```

`dex_isaac_mcp/protocol.Client` is a handy debugging client:

```python
from dex_isaac_mcp.protocol import Client
with Client() as c:
    print(c.call("status"))
    c.call("set_pose", name="ready"); c.call("step", n=240)
```

## Status

Tested against Isaac Lab 2.3.2 (Isaac Sim 5.x) and `mcp` 2.3, over the stdio protocol, with the GUI on:

- **Franka** and **Allegro** examples, plus a custom closed-linkage hand through a compose override: `sim_up`, poses, screenshots, `sim_range_test`, `sim_sweep` (restores the original value), `sim_down`.
- Headless daemon: gains, frozen-parameter rejection, wave.
- Headless capture and recording, Allegro and Franka: auto-framing, backdrop, captions, GIF output (the clip at the top).
- Props, GUI and headless: a sphere and a cylinder dropped onto a static table settle at exactly table height plus their radius and half-height.

- Training, against Isaac Lab's stock skrl script: `Isaac-Cartpole-v0` launched, polled, logged, checkpointed and read back through every `train_*` tool, plus a run stopped mid-training. skrl's `write_interval: auto` writes no TensorBoard scalars on a very short run (5 iterations), so `train_metrics` comes back empty there; 50 iterations gives 18 tags.

Issues and PRs are welcome.

## License

MIT, see [LICENSE](LICENSE).
