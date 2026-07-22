"""A tiny stdlib HTTP server for the shell — no web framework.

Routes:

* ``GET  /``            → the single-page UI
* ``GET  /api/config``  → current settings (UI-safe, no key)
* ``POST /api/config``  → update settings
* ``POST /api/new``     → start a fresh conversation
* ``POST /api/chat``    → run a turn, streaming events back as Server-Sent Events
* ``POST /api/quit``    → ask the app to exit (used by the UI's Quit button)

The chat endpoint streams the agent's event stream verbatim — each glia
:class:`~glia.trajectory.Event` becomes one ``data:`` line — so the front end is
just a thin renderer over the same events the library emits.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler
from importlib import resources

from ..trajectory import Trajectory
from .backend import build_agent
from .config import Config


class ShellState:
    """Holds the current config, the ongoing conversation, and a quit signal."""

    def __init__(self) -> None:
        self.config = Config.load()
        self.trajectory = Trajectory.new(system=self.config.system)
        self.lock = threading.Lock()
        self.quit_event = threading.Event()

    def reset(self) -> None:
        self.trajectory = Trajectory.new(system=self.config.system)


def _index_html() -> bytes:
    return resources.files("glia.shell").joinpath("web/index.html").read_bytes()


def make_handler(state: ShellState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "glia-shell"

        def log_message(self, *args: object) -> None:  # keep the console quiet
            pass

        # -- helpers -----------------------------------------------------------

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                return json.loads(raw or b"{}")
            except Exception:  # noqa: BLE001
                return {}

        # -- routes ------------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, _index_html(), "text/html; charset=utf-8")
            elif path == "/api/config":
                self._send(200, json.dumps(state.config.public()).encode())
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            data = self._json_body()  # always drain the request body first
            if path == "/api/config":
                self._update_config(data)
                self._send(200, json.dumps(state.config.public()).encode())
            elif path == "/api/new":
                state.reset()
                self._send(200, b'{"ok":true}')
            elif path == "/api/quit":
                self._send(200, b'{"ok":true}')
                state.quit_event.set()
            elif path == "/api/chat":
                self._chat(data.get("message", ""))
            else:
                self._send(404, b'{"error":"not found"}')

        # -- handlers ----------------------------------------------------------

        def _update_config(self, data: dict) -> None:
            config = state.config
            if data.get("mode"):
                config.mode = data["mode"]
            if data.get("model"):
                config.model = data["model"]
            if "system" in data:
                config.system = data["system"]
            if "use_tools" in data:
                config.use_tools = bool(data["use_tools"])
            if data.get("anthropic_api_key"):  # only overwrite when a value is given
                config.anthropic_api_key = data["anthropic_api_key"]
            if data.get("clear_key"):
                config.anthropic_api_key = ""
            config.save()

        def _chat(self, message: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(obj: dict) -> None:
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()

            # One turn at a time — this is a single-user local app.
            with state.lock:
                agent = build_agent(state.config, stream=True)

                async def run() -> None:
                    async for event in agent.run_events(message, trajectory=state.trajectory):
                        emit(event.to_dict())

                try:
                    asyncio.run(run())
                except Exception as exc:  # noqa: BLE001 - report failures to the UI
                    emit({"kind": "error", "message": str(exc)})
                emit({"kind": "__done__"})

    return Handler
