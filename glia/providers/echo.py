"""A deterministic, offline LLM for tests, examples, and local development.

:class:`EchoLLM` returns a scripted sequence of assistant turns, then falls back
to echoing the last user message. Because it needs no API key and behaves
identically every run, you can exercise the entire agent loop — tool calls,
subagents, compaction, checkpoint/resume — in CI with zero cost and zero
flakiness. It also records every request it received, so tests can assert on
exactly what the agent sent.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..llm import LLMRequest, LLMResponse, StreamChunk
from ..types import Block, Message, Text, ToolUse, Usage

Turn = str | Block | list[Block] | Callable[[LLMRequest], list[Block]]


class EchoLLM:
    """Replay scripted turns; then echo. Satisfies the :class:`~glia.llm.LLM`
    protocol.

    Each entry in ``turns`` is one assistant response, taken in order:

    * a ``str`` becomes a text reply,
    * a single block or a list of blocks is returned as-is,
    * a callable ``(request) -> list[Block]`` is computed on demand.
    """

    def __init__(self, turns: list[Turn] | None = None, *, default_reply: str | None = None) -> None:
        self.turns = list(turns or [])
        self.default_reply = default_reply
        self.calls: list[LLMRequest] = []
        self._index = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)

        if self._index < len(self.turns):
            blocks = _normalise(self.turns[self._index], request)
            self._index += 1
        else:
            text = self.default_reply if self.default_reply is not None else _last_user_text(request)
            blocks = [Text(text)]

        message = Message(role="assistant", blocks=blocks)
        stop_reason = "tool_use" if message.tool_uses() else "end_turn"
        usage = Usage(
            input_tokens=_approx_tokens_request(request),
            output_tokens=_approx_tokens_blocks(blocks),
        )
        return LLMResponse(message=message, stop_reason=stop_reason, usage=usage, raw=None)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Stream the scripted turn word-by-word, then a final response chunk.

        Deterministic: it computes the exact same response as :meth:`generate`
        and simply chunks the text, so streaming code paths can be exercised
        offline with no surprises.
        """
        response = await self.generate(request)
        for piece in _word_chunks(response.message.text()):
            yield StreamChunk(text=piece)
        yield StreamChunk(response=response)


def _word_chunks(text: str) -> list[str]:
    return re.findall(r"\S+\s*", text) if text else []


# -- helpers for building scripted turns --------------------------------------


def reply(text: str) -> str:
    """A plain text turn (identity helper for readability in scripts)."""
    return text


def call(name: str, arguments: dict[str, Any] | None = None, *, id: str | None = None) -> ToolUse:
    """A turn where the model calls one tool."""
    return ToolUse(id=id or f"call_{name}", name=name, input=arguments or {})


def _normalise(turn: Turn, request: LLMRequest) -> list[Block]:
    if callable(turn) and not isinstance(turn, (str, ToolUse, Text)):
        turn = turn(request)
    if isinstance(turn, str):
        return [Text(turn)]
    if isinstance(turn, list):
        return list(turn)
    return [turn]  # a single Block


def _last_user_text(request: LLMRequest) -> str:
    for message in reversed(request.messages):
        if message.role == "user":
            text = message.text()
            if text:
                return f"echo: {text}"
    return "echo"


def _approx_tokens_request(request: LLMRequest) -> int:
    chars = len(request.system or "")
    for message in request.messages:
        chars += len(message.text())
    return chars // 4


def _approx_tokens_blocks(blocks: list[Block]) -> int:
    return sum(len(b.text) for b in blocks if isinstance(b, Text)) // 4
