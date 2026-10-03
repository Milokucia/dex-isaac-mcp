"""Host-side tests: no Isaac, no GPU, stdlib only.  python -m unittest discover tests"""

import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from dex_isaac_mcp import training
from dex_isaac_mcp.protocol import Client, LineReader, ProtocolError, encode
from dex_isaac_mcp.robot import Robot, match, match_ordered, resolve_pose

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "robots"


class ProtocolTest(unittest.TestCase):
    def test_partial_lines_reassemble(self):
        r = LineReader()
        data = encode({"id": 1, "ok": True, "result": "x" * 100000})
        self.assertEqual(r.feed(data[:5000]), [])
        self.assertEqual(r.feed(data[5000:])[0]["result"], "x" * 100000)

    def test_newline_in_payload_stays_one_message(self):
        self.assertEqual(LineReader().feed(encode({"s": "a\nb"})), [{"s": "a\nb"}])

    def test_round_trip_and_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "s.sock")
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(path)
            srv.listen(1)

            def serve():
                conn, _ = srv.accept()
                reader = LineReader()
                with conn:
                    while chunk := conn.recv(4096):
                        for msg in reader.feed(chunk):
                            if msg["cmd"] == "boom":
                                conn.sendall(encode({"id": msg["id"], "ok": False, "error": "bad"}))
                            else:
                                conn.sendall(encode({"id": msg["id"], "ok": True, "result": msg["args"]}))

            threading.Thread(target=serve, daemon=True).start()
            with Client(path, timeout=5) as c:
                self.assertEqual(c.call("echo", n=3), {"n": 3})
                with self.assertRaises(ProtocolError):
                    c.call("boom")
            srv.close()


class RobotTest(unittest.TestCase):
    def test_examples_load(self):
        for f in EXAMPLES.glob("*.json"):
            with self.subTest(f.name):
                r = Robot.load(f)
                self.assertTrue(r.usd)
                self.assertTrue(r.poses)

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            Robot.from_dict({"usd": "a.usd", "drivenjoints": []})
        with self.assertRaises(ValueError):
            Robot.from_dict({"usd": "a.usd", "actuator": {"stifness": 1}})

    def test_relative_usd_resolves_against_config(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            p.write_text(json.dumps({"usd": "assets/r.usd"}))
            self.assertEqual(Robot.load(p).usd, str((Path(d) / "assets/r.usd").resolve()))

    def test_match_is_full_match(self):
        self.assertEqual(match(["joint_1"], ["joint_1", "joint_10"]), ["joint_1"])

    def test_match_ordered_follows_patterns(self):
        self.assertEqual(match_ordered(["c", "a.*"], ["a1", "b", "c", "a2"]), ["c", "a1", "a2"])

    def test_pose_blend_and_override(self):
        names = ["a_0", "a_1", "b_0"]
        self.assertEqual(resolve_pose({".*": 0.0, "a_.*": 1.0}, names),
                         {"a_0": 1.0, "a_1": 1.0, "b_0": 0.0})
        self.assertEqual(resolve_pose({"a_0": 1.0}, names, amount=0.5, base={"a_0": 0.5}),
                         {"a_0": 0.75})
        with self.assertRaises(ValueError):
            resolve_pose({"typo": 1.0}, names)

    def test_example_poses_resolve_against_names(self):
        franka = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
        for pose in Robot.load(EXAMPLES / "franka.json").poses.values():
            resolve_pose(pose, franka)
        allegro = [f"{f}_joint_{i}" for f in ("index", "middle", "ring", "thumb") for i in range(4)]
        for pose in Robot.load(EXAMPLES / "allegro_hand.json").poses.values():
            resolve_pose(pose, allegro)


class TrainingNameTest(unittest.TestCase):
    def test_names_cannot_smuggle_flags(self):
        for bad in ("--rm", "a b", "x;y", "../up"):
            with self.assertRaises(training.TrainingError):
                training.container_name(bad)

    def test_find_log_dir_suffix(self):
        with tempfile.TemporaryDirectory() as d:
            old = training.LOGS_DIR
            training.LOGS_DIR = Path(d)
            try:
                run = Path(d) / "skrl" / "cartpole" / "2026-01-01_00-00-00_ppo_torch_myrun"
                (run / "checkpoints").mkdir(parents=True)
                self.assertEqual(training.find_log_dir("myrun"), run)
                self.assertIsNone(training.find_log_dir("other"))
            finally:
                training.LOGS_DIR = old


if __name__ == "__main__":
    unittest.main()
