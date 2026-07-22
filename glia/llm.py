"""The provider boundary: one small Protocol every model backend implements.

glia never imports a vendor SDK in its core. Instead it talks to an
:class:`LLM` — a single ``async generate(request) -> LLMResponse`` method. A
Claude adapter and a deterministic in-memory adapter both satisfy it, and so can
anything you write in ~40 lines. This is the seam that keeps the library
provider-agnostic without a plugin system.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .types import Message, Usage


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """A tool as the model sees it: a name, a description, and a JSON schema.

    This is the wire-level shape, separate from the callable that runs it (see
    :mod:`glia.tools`). Providers only ever receive this, never your Python
    function.
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """Everything a provider needs to produce one turn. A plain value object —
    build it, log it, hash it, or replay it."""

    messages: list[Message]
    system: str | None = None
    tools: list[ToolSchema] = field(default_factory=list)
    tool_choice: str | None = None
    """``None``/``"auto"`` lets the model decide, ``"any"`` forces some tool,
    or a tool name forces that specific tool."""
    max_tokens: int = 4096
    temperature: float | None = None
    thinking: bool = False
    """When true, ask the provider to enable its reasoning mode if it has one."""
    stop: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """One model turn plus why it stopped and what it cost."""

    message: Message
    stop_reason: str
    usage: Usage = field(default_factory=Usage)
    raw: Any = None
    """The untouched provider response, for when you need to escape the abstraction."""

    @property
    def wants_tools(self) -> bool:
        return bool(self.message.tool_uses())


@runtime_checkable
class LLM(Protocol):
    """The entire provider contract. Implement this and glia can drive it."""

    async def generate(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """One piece of a streaming response.

    A chunk carries an incremental ``text`` delta, or — on the **final** chunk —
    the complete :class:`LLMResponse`. A stream is therefore a sequence of text
    deltas followed by exactly one terminal chunk with ``response`` set.
    """

    text: str = ""
    response: LLMResponse | None = None


@runtime_checkable
class StreamingLLM(Protocol):
    """An optional capability: providers that can stream implement this too.

    The agent uses it only when you ask for streaming *and* the provider
    supports it; otherwise it falls back to :meth:`LLM.generate`. Streaming is
    purely a delivery detail — the final :class:`LLMResponse` is identical either
    way.
    """

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:  # pragma: no cover - protocol
        ...
