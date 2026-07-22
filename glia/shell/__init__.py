"""The glia desktop shell — a downloadable graphical chat with a live glass box.

A small, dependency-light desktop app: a stdlib HTTP server serves a
self-contained web UI, wrapped in a native window via ``pywebview`` (with a
browser fallback). It drives an ordinary :class:`glia.Agent`, streams the reply,
and shows every event — model calls, tool calls, approvals — in a side panel, so
users can *see* the glass box, not just read about it.

Run it with the ``glia-shell`` command (``pip install "glia-agents[shell]"``),
or download a standalone binary from the GitHub releases.
"""

from __future__ import annotations
