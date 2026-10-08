"""The sim daemon: one Kit process, many commands.

Kit costs several seconds to "app ready" and more before PhysX has a CUDA
device, so a launch-per-experiment loop spends most of its time booting. Here
Kit starts once and stays up; a parameter change costs a frame.

THREAD AFFINITY IS THE LOAD-BEARING CONSTRAINT. Kit, PhysX and USD are not
thread-safe. The socket threads below only ever parse JSON and push onto
_pending; every call that touches the simulation happens on the main loop
thread in serve(). Do not "simplify" this by answering a request from the
reader thread — it will appear to work and then corrupt the stage under load.

Import only after AppLauncher has started Kit; see scripts/simd.py.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import queue
import socket
import threading
import traceback
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
import torch
from isaaclab.sim import SimulationCfg, SimulationContext

from ..protocol import LineReader, check_socket_path, encode
from ..robot import Robot, resolve_pose
from .scene import RobotScene, SceneParams, expand_usd, inspect_usd


class _Request:
    """One command awaiting execution on the main thread."""

    __slots__ = ("cmd", "args", "req_id", "conn")

    def __init__(self, cmd: str, args: dict[str, Any], req_id: Any, conn: socket.socket):
        self.cmd = cmd
        self.args = args
        self.req_id = req_id
        self.conn = conn


class SimDaemon:
    def __init__(self, socket_path: Path, robot: Robot, params: SceneParams, device: str,
                 dt: float = 1 / 120, ground: bool = True):
        self.socket_path = Path(check_socket_path(socket_path))
        self.robot = robot
        self.params = params
        self.device = device
        self.dt = dt
        # A floor is right for driving and screenshots and WRONG for a range
        # test on anything that can reach it: the links stop on the ground
        # before their own limits and the test measures the floor.
        self.ground = ground

        self.sim: SimulationContext | None = None
        self.scene: RobotScene | None = None
        self.playing = False
        # Optional callback fired once the scene exists; see serve(). Used by
        # simd.py --sliders to attach a UI panel that writes into self.unit.
        self.on_world_built = None
        self.unit: torch.Tensor | None = None

        self._capturer = None
        # The capture camera starts on the same side as the config's viewport
        # camera, so a robot whose front is not +x (e.g. a hand) is framed head-on.
        self._view_dir = [e - t for e, t in zip(robot.camera_eye, robot.camera_target, strict=True)]

        self._pending: queue.Queue[_Request] = queue.Queue()
        self._stop = threading.Event()
        self._server: socket.socket | None = None

    # ---- socket side (background threads) ------------------------------

    def start_listening(self) -> None:
        # A stale socket file from a hard-killed container makes bind() fail
        # with EADDRINUSE even though nothing is listening.
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(self.socket_path))
        srv.listen(8)
        srv.settimeout(0.5)  # so the accept loop can notice _stop
        self._server = srv
        os.chmod(self.socket_path, 0o660)

        threading.Thread(target=self._accept_loop, name="simd-accept", daemon=True).start()
        print(f"[simd] listening on {self.socket_path}", flush=True)

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=self._conn_loop, args=(conn,), name="simd-conn",
                             daemon=True).start()

    def _conn_loop(self, conn: socket.socket) -> None:
        reader = LineReader()
        with conn:
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(65536)
                except OSError:
                    return
                if not chunk:
                    return
                try:
                    msgs = reader.feed(chunk)
                except json.JSONDecodeError as exc:
                    with contextlib.suppress(OSError):
                        conn.sendall(encode({"id": None, "ok": False,
                                             "error": f"malformed JSON: {exc}"}))
                    return
                for msg in msgs:
                    self._pending.put(_Request(cmd=msg.get("cmd", ""),
                                               args=msg.get("args") or {},
                                               req_id=msg.get("id"), conn=conn))

    @staticmethod
    def _reply(req: _Request, ok: bool, payload: Any) -> None:
        body = {"id": req.req_id, "ok": ok}
        body["result" if ok else "error"] = payload
        with contextlib.suppress(OSError):  # client hung up mid-command
            req.conn.sendall(encode(body))

    # ---- main loop (owns every Kit call) -------------------------------

    def serve(self, simulation_app: Any) -> None:
        self.build_world()
        if self.on_world_built is not None:
            self.on_world_built(self)
        self.start_listening()

        while simulation_app.is_running() and not self._stop.is_set():
            # Drain every queued command before stepping, so a burst of
            # set_params + step arrives as one coherent change.
            while True:
                try:
                    req = self._pending.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._reply(req, True, self.dispatch(req.cmd, req.args))
                except Exception as exc:
                    traceback.print_exc()
                    self._reply(req, False, f"{type(exc).__name__}: {exc}")

            if self.playing:
                self._advance(1)
            else:
                # Still pump Kit while paused, or the GUI freezes and
                # screenshots return a stale frame.
                simulation_app.update()

        self.shutdown()

    def _advance(self, n: int) -> None:
        assert self.scene is not None
        for _ in range(n):
            if self.unit is not None:
                self.scene.apply(self.unit)
            self.scene.step(self.dt)

    # ---- world construction --------------------------------------------

    def build_world(self) -> None:
        sim = SimulationContext(SimulationCfg(dt=self.dt, device=self.device))
        sim.set_camera_view(eye=list(self.robot.camera_eye), target=list(self.robot.camera_target))
        if self.ground:
            sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
        light = sim_utils.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light", light)
        self.sim = sim

        # Spawned once. Replacing a spawned articulation requires
        # SimulationContext.stop() to drop the PhysX views first, and stop()
        # blocks on a timeline event that only advances when the Kit app loop
        # pumps. A command runs ON that loop, so the loop is not pumping and
        # the daemon wedges with no traceback. omni.usd new_stage() has the
        # same trap. Hence the frozen params: restart the daemon instead.
        scene = RobotScene(sim, self.robot, self.params)
        sim.reset()
        scene.bind()
        self.scene = scene
        # Hold the pose the robot spawned in rather than snapping to all-lower.
        self.unit = scene.target_to_unit(scene.art.data.joint_pos[0, scene.driven_ids]).clone()
        # The spawn state sim_reset returns to, so a bad experiment costs a few
        # frames instead of a Kit restart.
        self._home_q = scene.art.data.joint_pos[0].clone()
        self._home_unit = self.unit.clone()
        self._home_params = {k: getattr(self.params, k) for k in SceneParams.HOT}
        self._home_couplings = {c.follower: c.ratio for c in scene.couplings}
        print(f"[simd] spawned {self.robot.name}: {len(scene.driven_names)} driven of "
              f"{scene.art.num_joints} joints; {self.params.to_dict()}", flush=True)

    # ---- commands -------------------------------------------------------

    def dispatch(self, cmd: str, args: dict[str, Any]) -> Any:
        handler = getattr(self, f"cmd_{cmd}", None)
        if handler is None:
            raise ValueError(f"unknown command {cmd!r}")
        return handler(**args)

    def _require_scene(self) -> RobotScene:
        if self.scene is None:
            raise RuntimeError("no scene loaded")
        return self.scene

    def cmd_ping(self) -> str:
        return "pong"

    def cmd_status(self) -> dict[str, Any]:
        scene = self.scene
        return {
            "robot": self.robot.name,
            "robot_config": self.robot.source,
            "playing": self.playing,
            "device": self.device,
            "dt": self.dt,
            "ground": self.ground,
            "params": self.params.to_dict(),
            "num_joints": scene.art.num_joints if scene else 0,
            "driven": list(scene.driven_names) if scene else [],
            "couplings": [vars(c) for c in scene.couplings] if scene else [],
            "poses": sorted(self.robot.poses),
            "steps": scene.steps if scene else 0,
        }

    def cmd_inspect_joints(self, usd: str | None = None) -> dict[str, Any]:
        return inspect_usd(expand_usd(usd or self.params.usd))

    def cmd_set_params(self, **changes: Any) -> dict[str, Any]:
        """Write PD gains onto the live driven joints — no respawn."""
        unknown = set(changes) - set(self.params.to_dict())
        if unknown:
            raise ValueError(f"unknown params: {sorted(unknown)}")
        frozen = sorted(set(changes) & set(SceneParams.FROZEN))
        if frozen:
            raise ValueError(f"{frozen} are spawn properties and cannot change on a live "
                             "daemon. Restart it (sim_reload) with the new value.")
        scene = self._require_scene()
        ids, art = scene.driven_ids, scene.art
        if "stiffness" in changes:
            art.write_joint_stiffness_to_sim(float(changes["stiffness"]), joint_ids=ids)
        if "damping" in changes:
            art.write_joint_damping_to_sim(float(changes["damping"]), joint_ids=ids)
        if "effort" in changes:
            art.write_joint_effort_limit_to_sim(float(changes["effort"]), joint_ids=ids)
        for k, v in changes.items():
            setattr(self.params, k, float(v))
        return {"params": self.params.to_dict()}

    def cmd_set_coupling(self, ratios: dict[str, float]) -> dict[str, Any]:
        return {"couplings": self._require_scene().set_coupling(ratios)}

    def cmd_set_targets(self, unit: list[float] | None = None,
                        joints: dict[str, float] | None = None) -> dict[str, Any]:
        """Command the driven joints, normalized 0..1 (lower..upper limit)."""
        scene = self._require_scene()
        if unit is None and joints is None:
            raise ValueError("pass unit=[...] or joints={name: value}")
        if unit is not None:
            if len(unit) != scene.num_driven:
                raise ValueError(f"expected {scene.num_driven} values for "
                                 f"{scene.driven_names}, got {len(unit)}")
            self.unit = torch.tensor(unit, dtype=torch.float32, device=scene.sim.device)
        if joints:
            assert self.unit is not None
            for name, value in joints.items():
                if name not in scene.driven_names:
                    raise ValueError(f"{name!r} is not driven; driven={scene.driven_names}")
                self.unit[scene.driven_names.index(name)] = float(value)
        assert self.unit is not None
        self.unit.clamp_(0.0, 1.0)
        return {"unit": self.unit.tolist(), "names": list(scene.driven_names)}

    def cmd_list_poses(self) -> dict[str, Any]:
        return {"poses": self.robot.poses}

    def cmd_set_pose(self, name: str, amount: float = 1.0,
                     from_current: bool = False) -> dict[str, Any]:
        """Command a named pose from the robot config, optionally part-way."""
        scene = self._require_scene()
        if name not in self.robot.poses:
            raise ValueError(f"unknown pose {name!r}; known: {sorted(self.robot.poses)}")
        base = None
        if from_current and self.unit is not None:
            base = dict(zip(scene.driven_names, self.unit.tolist(), strict=True))
        joints = resolve_pose(self.robot.poses[name], list(scene.driven_names), amount, base)
        return self.cmd_set_targets(joints=joints)

    def cmd_step(self, n: int = 1) -> dict[str, Any]:
        scene = self._require_scene()
        self._advance(int(n))
        return {"steps": scene.steps, "sim_time": scene.steps * self.dt}

    def cmd_reset(self, keep_objects: bool = False, settle: int = 60,
                  pos_tol: float = 5e-3) -> dict[str, Any]:
        """Return to the spawn state without restarting Kit.

        Props go first, so none is left pressing on a joint during the settle.
        Then the gains and coupling ratios the daemon spawned with, then the
        root and every joint written back to their spawn values at zero
        velocity. The settle lets passive linkages close around that state;
        the residual says whether they did. Spawn properties (USD, solver
        iterations, self-collision) are untouched: those still need sim_reload,
        and so does a scene whose state went non-finite.
        """
        from . import objects

        scene = self._require_scene()
        art = scene.art
        removed = [] if keep_objects else objects.remove_all()
        self.cmd_set_params(**self._home_params)
        couplings = scene.set_coupling(self._home_couplings) if self._home_couplings else []

        root = art.data.default_root_state.clone()
        art.write_root_pose_to_sim(root[:, :7])
        art.write_root_velocity_to_sim(root[:, 7:])
        art.write_joint_state_to_sim(self._home_q.unsqueeze(0),
                                     torch.zeros_like(self._home_q).unsqueeze(0))
        self.unit = self._home_unit.clone()
        # At least one step: before it, joint_pos is the state just written,
        # so the residual below would read 0 whatever physics does.
        self._advance(max(1, int(settle)))

        q = art.data.joint_pos[0]
        finite = bool(torch.isfinite(q).all() and torch.isfinite(art.data.joint_vel[0]).all())
        res = float((q[scene.driven_ids] - self._home_q[scene.driven_ids]).abs().max())
        scene.reset_stats()
        ok = finite and res < pos_tol
        out = {"ok": ok, "removed": removed, "params": self.params.to_dict(),
               "couplings": couplings, "pos_residual": res, "finite": finite}
        if not ok:
            out["hint"] = ("state is not finite; sim_reload" if not finite else
                           "a joint did not return to its spawn value; sim_reset again "
                           "with a longer settle, or sim_reload")
        return out

    def cmd_wave(self, n: int = 240) -> dict[str, Any]:
        """Run the scripted sweep for n steps and return travel stats."""
        scene = self._require_scene()
        for _ in range(int(n)):
            scene.apply(scene.wave(scene.steps * self.dt))
            scene.step(self.dt)
        return scene.stats()

    def cmd_range_test(self, steps: int = 400, target: float = 1.0,
                       reset: bool = True) -> dict[str, Any]:
        rep = self._require_scene().range_test(steps, self.dt, target=target, reset=reset)
        rep["params"] = self.params.to_dict()
        return rep

    def cmd_spawn_object(self, **kwargs: Any) -> dict[str, Any]:
        from . import objects

        self._require_scene()
        out = objects.spawn(**kwargs)
        # One frame so PhysX has parsed the new prim before anyone asks its pose.
        self._advance(1)
        return out

    def cmd_list_objects(self) -> dict[str, Any]:
        from . import objects

        return {"objects": objects.poses()}

    def cmd_remove_object(self, name: str) -> dict[str, Any]:
        from . import objects

        return objects.remove(name)

    def _cap(self):
        if self._capturer is None:
            from .capture import Capturer

            self._capturer = Capturer()
            scene = self._require_scene()
            scene.post_step = self._capturer.on_step
            # Start framed on the robot: an unplaced camera sits at the origin
            # looking at nothing, and a first sim_capture came back blank.
            self._capturer.fit(scene.prim_path, self._view_dir)
        return self._capturer

    def _recordings(self) -> Path:
        return self.socket_path.parent / "recordings"

    def cmd_frame_robot(self, direction: list[float] | None = None, margin: float = 1.15,
                        width: int = 640, height: int = 480,
                        raise_frac: float = 0.0) -> dict[str, Any]:
        """Aim the capture camera so the whole robot fills the frame."""
        if direction is not None:
            self._view_dir = list(direction)
        scene = self._require_scene()
        return self._cap().fit(scene.prim_path, self._view_dir, margin, width, height, raise_frac)

    def cmd_set_capture_camera(self, eye: list[float], target: list[float]) -> dict[str, Any]:
        self._cap().look_at(eye, target)
        self._view_dir = [e - t for e, t in zip(eye, target, strict=True)]
        return {"eye": eye, "target": target}

    def cmd_set_backdrop(self, color: list[float] | None = None, size: float = 4.0) -> dict[str, Any]:
        """A plain panel behind the robot, facing the capture camera. color=None removes it."""
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        box = cache.ComputeWorldBound(
            stage.GetPrimAtPath(self._require_scene().prim_path)).ComputeAlignedRange()
        c = box.GetMidpoint()
        return self._cap().backdrop(color, [c[0], c[1], c[2]], self._view_dir, size)

    def cmd_capture(self, width: int = 640, height: int = 480) -> dict[str, Any]:
        """A frame from the capture camera. Works headless."""
        return {"png_base64": self._cap().png_base64(int(width), int(height))}

    def cmd_record_start(self, every: int = 2, width: int = 640, height: int = 480,
                         caption: str = "") -> dict[str, Any]:
        cap = self._cap()
        cap.caption = caption
        cap.start(every, int(width), int(height))
        return {"recording": True, "every": cap.every}

    def cmd_record_caption(self, text: str) -> dict[str, Any]:
        self._cap().caption = text
        return {"caption": text}

    def cmd_record_stop(self, name: str = "recording", fps: float = 30.0,
                        hold_last: float = 1.0) -> dict[str, Any]:
        safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_") or "recording"
        return self._cap().stop(self._recordings() / f"{safe}.gif", fps, hold_last)

    def cmd_play(self) -> dict[str, Any]:
        self.playing = True
        return {"playing": True}

    def cmd_pause(self) -> dict[str, Any]:
        self.playing = False
        return {"playing": False}

    def cmd_get_joint_state(self) -> dict[str, Any]:
        return self._require_scene().joint_state()

    def cmd_get_stats(self) -> dict[str, Any]:
        return self._require_scene().stats()

    def cmd_reset_stats(self) -> dict[str, Any]:
        self._require_scene().reset_stats()
        return {"steps": 0}

    def cmd_set_camera(self, eye: list[float], target: list[float]) -> dict[str, Any]:
        if self.sim is None:
            raise RuntimeError("no sim")
        self.sim.set_camera_view(eye=eye, target=target)
        return {"eye": eye, "target": target}

    def cmd_screenshot(self, settle: int = 4) -> dict[str, Any]:
        """Capture the viewport and return it as base64 PNG.

        The capture is asynchronous inside Kit, so frames are pumped until the
        file appears rather than reading a half-written PNG.
        """
        try:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        except ImportError as exc:
            raise RuntimeError("viewport extension unavailable — run the daemon with --gui") from exc
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("no active viewport — run the daemon with --gui")

        out = (self.socket_path.parent / "simd-capture.png").resolve()
        if out.exists():
            out.unlink()

        import omni.kit.app

        capture_viewport_to_file(viewport, str(out))
        app = omni.kit.app.get_app()
        for _ in range(max(int(settle), 1) + 60):
            app.update()
            if out.exists() and out.stat().st_size > 0:
                break
        if not out.exists():
            raise RuntimeError(f"capture did not produce {out}")
        data = out.read_bytes()
        return {"path": str(out), "bytes": len(data),
                "png_base64": base64.b64encode(data).decode()}

    def cmd_shutdown(self) -> str:
        self._stop.set()
        return "stopping"

    # ---- teardown --------------------------------------------------------

    def shutdown(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.close()
            self._server = None
        if self.socket_path.exists():
            self.socket_path.unlink()
        print("[simd] stopped", flush=True)
