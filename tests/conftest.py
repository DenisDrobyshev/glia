"""Shared test fixtures/helpers.

Tests use ``asyncio.run`` directly rather than a pytest-asyncio plugin, so the
suite has zero test-time dependencies beyond pytest itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion in a fresh event loop."""
    return asyncio.run(coro)
