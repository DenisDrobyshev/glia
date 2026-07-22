"""Allow ``python -m glia.shell`` and give PyInstaller a stable entry point."""

from __future__ import annotations

from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
