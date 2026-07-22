"""Provider adapters. Each is a thin, self-contained implementation of the
:class:`~glia.llm.LLM` protocol.

* :class:`EchoLLM` — deterministic, offline, no API key. For tests and demos.
* :class:`ClaudeLLM` — Claude via the Anthropic SDK (optional dependency).

``ClaudeLLM`` is imported lazily so that ``import glia`` never requires the
``anthropic`` package.
"""

from __future__ import annotations

from typing import Any

from .echo import EchoLLM, call, reply
from .ollama import OllamaLLM  # stdlib-only, safe to import eagerly

__all__ = ["EchoLLM", "OllamaLLM", "ClaudeLLM", "call", "reply"]


def __getattr__(name: str) -> Any:
    # ClaudeLLM stays lazy so `import glia` never requires the anthropic package.
    if name == "ClaudeLLM":
        from .anthropic import ClaudeLLM

        return ClaudeLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
