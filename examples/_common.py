"""Shared helper for the examples.

Every example runs offline by default with the deterministic :class:`EchoLLM`,
so you can `python examples/01_hello_agent.py` with no API key. Set
``ANTHROPIC_API_KEY`` (or run `ant auth login`) to route the same code through
real Claude instead — the agent code never changes.
"""

from __future__ import annotations

import os
import sys

# Let the examples run straight from a fresh clone, before `pip install -e .`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glia.providers import EchoLLM  # noqa: E402


def make_llm(script: list | None = None):
    """Return ClaudeLLM if credentials are present, else a scripted EchoLLM."""
    if os.getenv("ANTHROPIC_API_KEY"):
        from glia.providers import ClaudeLLM

        return ClaudeLLM()  # model defaults to claude-opus-4-8
    return EchoLLM(script or [])
