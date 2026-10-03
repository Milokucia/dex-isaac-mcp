"""Host-side control of Isaac Lab training runs, one container per run.

Stdlib-only (tensorboard optional, for metrics). Training is deliberately NOT
routed through the sim daemon: the daemon holds one interactive scene for
tuning, while a training run needs hundreds of parallel headless envs, takes
no live commands, and legitimately runs for hours. Each run is its own
`docker compose run -d --rm` container, named so it can be found again.

Defaults target Isaac Lab's stock skrl script, which logs to
logs/skrl/<experiment>/<timestamp>_<algo>_<framework>_<run_name>/. Point it at
your own launcher with environment variables:

    ISAAC_MCP_COMPOSE_DIR     dir holding docker-compose.yaml   (<repo>/docker)
    ISAAC_MCP_TRAIN_SERVICE   compose service to run            (isaac-lab)
    ISAAC_MCP_TRAIN_WORKDIR   working dir inside the container  (/workspace/isaaclab)
    ISAAC_MCP_TRAIN_SCRIPT    script, relative to the workdir   (scripts/reinforcement_learning/skrl/train.py)
    ISAAC_MCP_LOGS_DIR        host dir the container logs into  (<repo>/logs)
    ISAAC_MCP_RUN_NAME_ARG    how the run name reaches the script, with {run_name}
                              (agent.agent.experiment.experiment_name={run_name}; empty = not passed)
    ISAAC_MCP_TRAIN_ARGS      args appended to every run, shell-split
                              (hydra.run.dir=/tmp/hydra hydra.output_subdir=null; empty = none)

The Hydra default exists because the stock script writes Hydra's outputs/
into its working dir, /workspace/isaaclab, which is root-owned in the image
and unwritable to the non-root container user: every run died at startup
with PermissionError: 'outputs'.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .protocol import REPO_ROOT as _ROOT

COMPOSE_DIR = Path(os.environ.get("ISAAC_MCP_COMPOSE_DIR") or _ROOT / "docker")
SERVICE = os.environ.get("ISAAC_MCP_TRAIN_SERVICE", "isaac-lab")
WORKDIR = os.environ.get("ISAAC_MCP_TRAIN_WORKDIR", "/workspace/isaaclab")
SCRIPT = os.environ.get("ISAAC_MCP_TRAIN_SCRIPT", "scripts/reinforcement_learning/skrl/train.py")
LOGS_DIR = Path(os.environ.get("ISAAC_MCP_LOGS_DIR") or _ROOT / "logs")
RUN_NAME_ARG = os.environ.get("ISAAC_MCP_RUN_NAME_ARG",
                              "agent.agent.experiment.experiment_name={run_name}")

TRAIN_ARGS = shlex.split(os.environ.get("ISAAC_MCP_TRAIN_ARGS",
                                        "hydra.run.dir=/tmp/hydra hydra.output_subdir=null"))

CONTAINER_PREFIX = "isaacmcp-train-"

# Valid docker names; also keeps task/run_name out of argv in a way that
# cannot smuggle flags.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class TrainingError(RuntimeError):
    pass


def _check_name(value: str, what: str) -> str:
    if not _SAFE_NAME.match(value):
        raise TrainingError(f"invalid {what}: {value!r}")
    return value


def container_name(run_name: str) -> str:
    return f"{CONTAINER_PREFIX}{_check_name(run_name, 'run_name')}"


def make_run_name(task: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", task)
    return f"{slug}_{datetime.now():%Y%m%d_%H%M%S}"


def _docker(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], cwd=cwd, capture_output=True, text=True)


def find_log_dir(run_name: str) -> Path | None:
    """The run's log directory: LOGS_DIR/<run_name>, else any dir ending in _<run_name>."""
    _check_name(run_name, "run_name")
    direct = LOGS_DIR / run_name
    if direct.is_dir():
        return direct
    if not LOGS_DIR.is_dir():
        return None
    hits = [p for p in LOGS_DIR.rglob(f"*_{run_name}") if p.is_dir()]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


# ---- lifecycle --------------------------------------------------------


def start(task: str, num_envs: int | None = None, max_iterations: int | None = None,
          checkpoint: str | None = None, seed: int | None = None,
          run_name: str | None = None, extra_args: list[str] | None = None,
          device: int | None = None) -> dict[str, Any]:
    """Launch a training run in its own container and return immediately.

    `device` pins the run to one GPU index via CUDA_VISIBLE_DEVICES. Without it
    every concurrent run lands on GPU 0 — a single-process trainer does not
    spread across cards, so separate runs need separate devices.
    """
    _check_name(task, "task")
    run_name = _check_name(run_name or make_run_name(task), "run_name")
    name = container_name(run_name)

    if _docker(["inspect", name]).returncode == 0:
        raise TrainingError(f"a run named {run_name!r} already exists (container {name}); "
                            "pick another run_name or stop it first")

    cmd = ["compose", "run", "-d", "--rm", "--name", name, "-w", WORKDIR]
    if device is not None:
        # Before the service name: `compose run` treats everything after it as
        # the command, so a later -e would be handed to the training script.
        cmd += ["-e", f"CUDA_VISIBLE_DEVICES={int(device)}"]
    cmd += [SERVICE, SCRIPT, "--task", task, "--headless"]
    if num_envs is not None:
        cmd += ["--num_envs", str(int(num_envs))]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    if max_iterations is not None:
        cmd += ["--max_iterations", str(int(max_iterations))]
    if checkpoint is not None:
        cmd += ["--checkpoint", checkpoint]
    cmd += list(extra_args or [])
    cmd += TRAIN_ARGS
    if RUN_NAME_ARG:
        cmd.append(RUN_NAME_ARG.format(run_name=run_name))

    result = _docker(cmd, cwd=COMPOSE_DIR)
    if result.returncode != 0:
        raise TrainingError(f"docker compose run failed: {result.stderr.strip()}")
    return {"run_name": run_name, "container": name, "task": task,
            "command": cmd[cmd.index(SERVICE) + 1:], "logs_dir": str(LOGS_DIR)}


def stop(run_name: str) -> dict[str, Any]:
    name = container_name(run_name)
    result = _docker(["stop", name])
    if result.returncode != 0:
        raise TrainingError(f"could not stop {name}: {result.stderr.strip()}")
    return {"run_name": run_name, "container": name, "stopped": True}


def logs(run_name: str, tail: int = 200) -> dict[str, Any]:
    name = container_name(run_name)
    result = _docker(["logs", "--tail", str(int(tail)), name])
    if result.returncode != 0:
        raise TrainingError(f"could not read logs for {name} (containers are --rm, so a "
                            f"finished run has none): {result.stderr.strip()}")
    return {"run_name": run_name, "text": result.stdout + result.stderr}


# ---- inspection ---------------------------------------------------------


def _docker_ps() -> list[dict[str, Any]]:
    result = _docker(["ps", "-a", "--filter", f"name={CONTAINER_PREFIX}",
                      "--format", "{{json .}}"])
    if result.returncode != 0:
        return []
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def list_runs() -> dict[str, Any]:
    """Runs with a live or recently exited container, plus any log dir with checkpoints."""
    containers = {row["Names"]: row for row in _docker_ps()}
    runs: dict[str, dict[str, Any]] = {}
    for cname, row in containers.items():
        if cname.startswith(CONTAINER_PREFIX):
            rn = cname[len(CONTAINER_PREFIX):]
            runs[rn] = {"run_name": rn, "running": row.get("State") == "running",
                        "container_status": row.get("Status")}
    if LOGS_DIR.is_dir():
        for ckpt in LOGS_DIR.rglob("checkpoints"):
            if ckpt.is_dir():
                d = ckpt.parent
                key = next((rn for rn in runs if d.name == rn or d.name.endswith("_" + rn)),
                           str(d.relative_to(LOGS_DIR)))
                runs.setdefault(key, {"run_name": key, "running": False})["log_dir"] = str(d)
    return {"runs": sorted(runs.values(), key=lambda r: r["run_name"])}


def _resolve(run_name: str) -> Path | None:
    # list_runs reports runs started elsewhere by their path under LOGS_DIR.
    if "/" in run_name:
        p = (LOGS_DIR / run_name).resolve()
        if LOGS_DIR.resolve() not in p.parents or not p.is_dir():
            raise TrainingError(f"no run directory {run_name!r} under {LOGS_DIR}")
        return p
    return find_log_dir(run_name)


def status(run_name: str) -> dict[str, Any]:
    log_dir = _resolve(run_name)
    container = None
    if "/" not in run_name:
        name = container_name(run_name)
        container = next((r for r in _docker_ps() if r.get("Names") == name), None)
    out: dict[str, Any] = {
        "run_name": run_name,
        "running": bool(container and container.get("State") == "running"),
        "container_status": container.get("Status") if container else "not found",
        "log_dir": str(log_dir) if log_dir else None,
        "checkpoints": _list_checkpoints(log_dir),
    }
    if log_dir is not None:
        latest = _latest_scalars(log_dir)
        if latest is not None:
            out["latest_metrics"] = latest
    return out


def _list_checkpoints(log_dir: Path | None) -> list[dict[str, Any]]:
    if log_dir is None or not (log_dir / "checkpoints").is_dir():
        return []
    out = []
    for f in (log_dir / "checkpoints").glob("*.pt"):
        m = re.search(r"_(\d+)\.pt$", f.name)
        out.append({"file": str(f), "name": f.name, "step": int(m.group(1)) if m else None,
                    "bytes": f.stat().st_size})
    out.sort(key=lambda r: (r["step"] is None, r["step"] or 0))
    return out


def checkpoints(run_name: str) -> dict[str, Any]:
    return {"run_name": run_name, "checkpoints": _list_checkpoints(_resolve(run_name))}


# ---- TensorBoard metrics --------------------------------------------------


def _event_accumulator(log_dir: Path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise TrainingError("reading metrics needs `pip install tensorboard` on the host") from exc
    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def _latest_scalars(log_dir: Path) -> dict[str, float] | None:
    try:
        acc = _event_accumulator(log_dir)
    except TrainingError:
        return None
    out = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        if events:
            out[tag] = events[-1].value
    return out or None


def metrics(run_name: str, tag: str | None = None, max_points: int = 200) -> dict[str, Any]:
    """List scalar tags, or the (step, value) series for one tag, downsampled."""
    log_dir = _resolve(run_name)
    if log_dir is None:
        raise TrainingError(f"no log directory for {run_name!r} under {LOGS_DIR}")
    acc = _event_accumulator(log_dir)
    tags = acc.Tags().get("scalars", [])
    if tag is None:
        return {"run_name": run_name, "tags": tags}
    if tag not in tags:
        raise TrainingError(f"no tag {tag!r} in {run_name!r}; available: {tags}")
    events = acc.Scalars(tag)
    n = len(events)
    if n > max_points >= 2:
        # Evenly spaced, always including the first and the final point: the
        # final one is what anyone polling a run wants.
        events = [events[round(i * (n - 1) / (max_points - 1))] for i in range(max_points)]
    elif n > max_points:
        events = events[-max_points:]
    return {"run_name": run_name, "tag": tag,
            "points": [{"step": e.step, "value": e.value} for e in events]}
