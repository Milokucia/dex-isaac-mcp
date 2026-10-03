"""Wire protocol for the sim daemon.

Newline-delimited JSON over a Unix domain socket, one JSON object per line:

    request   {"id": 7, "cmd": "step", "args": {"n": 60}}
    response  {"id": 7, "ok": true, "result": {...}}
              {"id": 7, "ok": false, "error": "no scene loaded"}

Deliberately stdlib-only and free of any isaaclab import. The daemon runs
inside the Isaac container; the MCP bridge and any ad-hoc client run on the
host, where isaaclab does not exist. Both ends import this same file, so the
framing cannot drift between them.

A Unix socket rather than TCP because the repo is bind-mounted into the
container and the container runs as the host uid (see
docker/docker-compose.yaml), so the socket file is reachable from the host
with no port mapping and no permission fixups.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

# Under the repo's bind mount this resolves to the same file on both sides:
#   host       <repo>/.cache/simd.sock
#   container  /workspace/isaac-sim-mcp/.cache/simd.sock
# ISAAC_MCP_SOCKET overrides it (e.g. for a non-editable install, or a repo
# path long enough to hit the AF_UNIX limit below).
DEFAULT_SOCKET = Path(
    os.environ.get("ISAAC_MCP_SOCKET")
    or Path(__file__).resolve().parents[1] / ".cache" / "simd.sock"
)

# Kit's first frame after a cold start can take several seconds, and load_usd
# on a large asset is slower still. A short default timeout would look like a
# daemon crash.
DEFAULT_TIMEOUT = 120.0


# Linux caps sun_path at 108 bytes including the NUL. Over that, bind() and
# connect() both fail with a bare "AF_UNIX path too long", which is a confusing
# thing to hit after moving the repo somewhere deeper.
_SUN_PATH_MAX = 107


class ProtocolError(RuntimeError):
    """The daemon replied, but with an error, or the connection died mid-call."""


def check_socket_path(path: str | Path) -> str:
    """Fail early and legibly on an over-long socket path."""
    text = str(path)
    if len(text.encode()) > _SUN_PATH_MAX:
        raise ValueError(
            f"socket path is {len(text.encode())} bytes, over the {_SUN_PATH_MAX}-byte "
            f"AF_UNIX limit: {text}\nSet ISAAC_MCP_SOCKET / --socket to a shorter path."
        )
    return text


def encode(obj: dict[str, Any]) -> bytes:
    """One request/response as a single newline-terminated line.

    Rejects embedded newlines by construction: json.dumps escapes them, so a
    string field can never split one message into two.
    """
    return (json.dumps(obj) + "\n").encode()


class LineReader:
    """Reassembles newline-delimited JSON from a stream socket.

    A socket read returns whatever bytes happen to have arrived, which for a
    base64 screenshot is reliably a partial line. Feed raw chunks in, get
    complete objects out.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        self._buf.extend(chunk)
        out: list[dict[str, Any]] = []
        while True:
            nl = self._buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(self._buf[:nl])
            del self._buf[: nl + 1]
            if line.strip():
                out.append(json.loads(line))
        return out


class Client:
    """Blocking client. Used by the MCP bridge and by hand for debugging."""

    def __init__(self, path: str | Path = DEFAULT_SOCKET, timeout: float = DEFAULT_TIMEOUT):
        self.path = check_socket_path(path)
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._reader = LineReader()
        self._next_id = 0

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.path)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> Client:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def call(self, cmd: str, **args: Any) -> Any:
        if self._sock is None:
            self.connect()
        assert self._sock is not None

        self._next_id += 1
        req_id = self._next_id
        self._sock.sendall(encode({"id": req_id, "cmd": cmd, "args": args}))

        while True:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ProtocolError(
                    f"daemon closed the connection during {cmd!r}; "
                    "check the container log for a Kit-side traceback"
                )
            for msg in self._reader.feed(chunk):
                # Requests are answered in order on one connection, but match
                # on id anyway so a future async command cannot desync us.
                if msg.get("id") != req_id:
                    continue
                if not msg.get("ok"):
                    raise ProtocolError(msg.get("error", "unknown daemon error"))
                return msg.get("result")
