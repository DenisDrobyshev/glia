"""Allow ``python -m glia.shell`` and give PyInstaller a stable entry point.

Uses an absolute import on purpose: PyInstaller runs this file as the top-level
``__main__`` with no package context, so a relative import (``from .app``) would
fail with "attempted relative import with no known parent package". The absolute
form works both frozen and under ``python -m glia.shell``.
"""

from __future__ import annotations

from glia.shell.app import main

if __name__ == "__main__":
    raise SystemExit(main())
