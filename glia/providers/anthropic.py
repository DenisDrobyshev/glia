"""Claude adapter, built on the official Anthropic SDK.

This is the whole integration — a translation layer between glia's blocks and
the Messages API, in about a hundred readable lines. The ``anthropic`` package
is an optional dependency, imported lazily, so the core of glia has no vendor
requirement.

Defaults follow current Anthropic guidance: model ``claude-opus-4-8``, and
adaptive thinking when ``thinking`` is requested. Note that Opus 4.8 rejects
``temperature`` — glia only sends it when you set it explicitly, so the default
path is safe.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..errors import ProviderError
from ..llm import LLMRequest, LLMResponse, StreamChunk
from ..types import Block, Message, Text, Thinking, ToolResult, ToolUse, Usage

DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeLLM:
    """An :class:`~glia.llm.LLM` backed by Claude via the Anthropic SDK."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        client: Any = None,
        api_key: str | None = None,
        thinking_display: str = "summarized",
    ) -> None:
        self.model = model
        self.thinking_display = thinking_display
        self._client = client
        self._api_key = api_key

    @property
    def client(self) -> Any:
        """Lazily construct an ``AsyncAnthropic`` client. Credentials resolve
        from the environment / ``ant`` profile just like the SDK default."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ProviderError(
                    "the 'anthropic' package is required for ClaudeLLM — "
                    "install it with: pip install 'glia[anthropic]'"
                ) from exc
            self._client = AsyncAnthropic(api_key=self._api_key) if self._api_key else AsyncAnthropic()
        return self._client

    def _build_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": [_message_to_api(m) for m in request.messages],
        }
        if request.system:
            kwargs["system"] = request.system
        if request.tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in request.tools
            ]
            if request.tool_choice and request.tool_choice != "auto":
                kwargs["tool_choice"] = _tool_choice(request.tool_choice)
        if request.thinking:
            kwargs["thinking"] = {"type": "adaptive", "display": self.thinking_display}
        elif request.temperature is not None:
            # Only sent when explicitly requested and thinking is off — current
            # Opus/Sonnet models reject sampling params.
            kwargs["temperature"] = request.temperature
        if request.stop:
            kwargs["stop_sequences"] = request.stop
        return kwargs

    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            response = await self.client.messages.create(**self._build_kwargs(request))
        except Exception as exc:  # noqa: BLE001 - normalise to our error boundary
            raise ProviderError(f"Claude request failed: {exc}") from exc
        return _response_from_api(response)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Stream text deltas, then a final chunk carrying the full response.

        Uses the SDK's ``messages.stream`` helper; the final chunk's response is
        assembled from ``get_final_message()``, so it is byte-identical to what
        ``generate`` would have returned.
        """
        final = None
        try:
            async with self.client.messages.stream(**self._build_kwargs(request)) as s:
                async for text in s.text_stream:
                    yield StreamChunk(text=text)
                final = await s.get_final_message()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Claude stream failed: {exc}") from exc
        yield StreamChunk(response=_response_from_api(final))


# -- request conversion --------------------------------------------------------


def _message_to_api(message: Message) -> dict[str, Any]:
    return {"role": message.role, "content": [_block_to_api(b) for b in message.blocks if _keep(b)]}


def _keep(block: Block) -> bool:
    # Thinking blocks are dropped on the way back in: glia does not capture the
    # provider signature needed to replay them. Keep thinking off for multi-turn
    # tool loops, or extend this adapter to preserve signatures.
    return not isinstance(block, Thinking)


def _block_to_api(block: Block) -> dict[str, Any]:
    if isinstance(block, Text):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUse):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResult):
        payload: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
        }
        if block.is_error:
            payload["is_error"] = True
        return payload
    raise ProviderError(f"cannot send block of type {type(block).__name__} to Claude")


def _tool_choice(choice: str) -> dict[str, Any]:
    if choice == "any":
        return {"type": "any"}
    return {"type": "tool", "name": choice}


# -- response conversion -------------------------------------------------------


def _response_from_api(response: Any) -> LLMResponse:
    blocks: list[Block] = []
    for block in response.content:
        kind = getattr(block, "type", None)
        if kind == "text":
            blocks.append(Text(block.text))
        elif kind == "tool_use":
            blocks.append(ToolUse(id=block.id, name=block.name, input=dict(block.input or {})))
        elif kind == "thinking":
            blocks.append(Thinking(getattr(block, "thinking", "") or ""))
        # redacted_thinking and any future block types are ignored for display.

    stop_reason = getattr(response, "stop_reason", None) or "end_turn"
    if stop_reason == "refusal" and not any(isinstance(b, Text) for b in blocks):
        details = getattr(response, "stop_details", None)
        reason = getattr(details, "explanation", None) or "the model declined this request"
        blocks.append(Text(f"[refusal] {reason}"))

    return LLMResponse(
        message=Message(role="assistant", blocks=blocks),
        stop_reason=stop_reason,
        usage=_usage_from_api(getattr(response, "usage", None)),
        raw=response,
    )


def _usage_from_api(usage: Any) -> Usage:
    if usage is None:
        return Usage()
    return Usage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )
