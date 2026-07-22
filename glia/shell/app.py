"""Entry point: start the local server and open the app.

By default it opens a **native desktop window** via ``pywebview``. If that isn't
available — or you pass ``--web``, or you're running a frozen binary — it falls
back to opening your default browser. Either way it's the same local app.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser
from http.server import ThreadingHTTPServer

from .server import ShellState, make_handler


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glia-shell", description="The glia desktop shell.")
    parser.add_argument("--web", action="store_true", help="Open in your browser instead of a native window.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port.")
    args = parser.parse_args(argv)

    port = args.port or _free_port()
    state = ShellState()
    server = ThreadingHTTPServer((args.host, port), make_handler(state))
    url = f"http://{args.host}:{port}/"

    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"glia shell running at {url}")

    # Frozen binaries and --web use the browser; a normal install tries a window.
    want_window = not args.web and not getattr(sys, "frozen", False)
    if want_window and _open_native_window(url):
        server.shutdown()
        return 0

    webbrowser.open(url)
    print("Close this window (or use the Quit button in the app) to stop.")
    state.quit_event.wait()  # set by the UI's Quit button
    server.shutdown()
    return 0


def _open_native_window(url: str) -> bool:
    """Open a pywebview window (blocking until closed). Returns False if pywebview
    isn't available, so the caller can fall back to the browser."""
    try:
        import webview  # type: ignore
    except Exception:  # noqa: BLE001
        print("(native window unavailable — opening your browser instead)")
        return False
    webview.create_window("glia", url, width=1180, height=800, min_size=(760, 540))
    webview.start()
    return True


if __name__ == "__main__":
    raise SystemExit(main())
