"""OpenAI (and any OpenAI-compatible endpoint) via stdlib HTTP.

Talks to the Chat Completions API (``/v1/chat/completions``) using only the
standard library — no ``openai`` SDK dependency. Because it takes a ``base_url``,
the same adapter drives OpenAI, Groq, Together, OpenRouter, a local vLLM, or
anything that speaks the OpenAI wire format.

    from glia import Agent
    from glia.providers import OpenAILLM

    agent = Agent(OpenAILLM("gpt-4o-mini", api_key="sk-..."), system="Be concise.")

Streaming and tool calling are supported. The key resolves from the constructor
or the ``OPENAI_API_KEY`` environment variable.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import AsyncIterator
from typing import Any
from urllib.request import Request, urlopen

from ..errors import ProviderError
from ..llm import LLMRequest, LLMResponse, StreamChunk
from ..types import Block, Message, Text, ToolResult, ToolUse, Usage

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class OpenAILLM:
    """An :class:`~glia.llm.LLM` backed by an OpenAI-compatible chat API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- LLM protocol ----------------------------------------------------------

    async def generate(self, request: LLMRequest) -> LLMResponse:
        payload = self._payload(request, stream=False)

        def call() -> dict[str, Any]:
            with urlopen(self._request(payload), timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())

        try:
            data = await asyncio.to_thread(call)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenAI request failed ({self.base_url}): {exc}") from exc

        choice = (data.get("choices") or [{}])[0]
        return _response_from(choice.get("message") or {}, choice.get("finish_reason"), data.get("usage") or {})

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        payload = self._payload(request, stream=True)
        payload["stream_options"] = {"include_usage": True}
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker() -> None:
            try:
                with urlopen(self._request(payload), timeout=self.timeout) as resp:
                    for raw in resp:
                        line = raw.decode().strip()
                        if line.startswith("data:"):
                            loop.call_soon_threadsafe(queue.put_nowait, ("data", line[5:].strip()))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("end", None))

        threading.Thread(target=worker, daemon=True).start()

        text_parts: list[str] = []
        tool_acc: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        while True:
            kind, item = await queue.get()
            if kind == "end":
                break
            if kind == "error":
                raise ProviderError(f"OpenAI stream failed ({self.base_url}): {item}")
            if item == "[DONE]":
                continue
            obj = json.loads(item)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if delta.get("content"):
                text_parts.append(delta["content"])
                yield StreamChunk(text=delta["content"])
            for call in delta.get("tool_calls") or []:
                _accumulate_tool_call(tool_acc, call)

        message = {
            "content": "".join(text_parts),
            "tool_calls": [tool_acc[i] for i in sorted(tool_acc)],
        }
        yield StreamChunk(response=_response_from(message, finish_reason, usage))

    # -- request building ------------------------------------------------------

    def _request(self, payload: dict[str, Any]) -> Request:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )

    def _payload(self, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(request.messages, request.system),
            "stream": stream,
            "max_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in request.tools
            ]
            body["tool_choice"] = _tool_choice(request.tool_choice)
        return body


# -- conversion ----------------------------------------------------------------


def _to_openai_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for message in messages:
        if message.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.text() or None}
            calls = [
                {"id": u.id, "type": "function", "function": {"name": u.name, "arguments": json.dumps(u.input)}}
                for u in message.tool_uses()
            ]
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
        else:
            texts = [b.text for b in message.blocks if isinstance(b, Text)]
            if texts:
                out.append({"role": "user", "content": "".join(texts)})
            for block in message.blocks:
                if isinstance(block, ToolResult):
                    out.append({"role": "tool", "tool_call_id": block.tool_use_id, "content": block.content})
    return out


def _tool_choice(choice: str | None) -> Any:
    if choice in (None, "auto"):
        return "auto"
    if choice == "any":
        return "required"
    return {"type": "function", "function": {"name": choice}}


def _accumulate_tool_call(acc: dict[int, dict[str, Any]], delta: dict[str, Any]) -> None:
    index = delta.get("index", 0)
    slot = acc.setdefault(index, {"id": None, "function": {"name": "", "arguments": ""}})
    if delta.get("id"):
        slot["id"] = delta["id"]
    fn = delta.get("function") or {}
    if fn.get("name"):
        slot["function"]["name"] = fn["name"]
    if fn.get("arguments"):
        slot["function"]["arguments"] += fn["arguments"]


def _response_from(message: dict[str, Any], finish_reason: str | None, usage: dict[str, Any]) -> LLMResponse:
    blocks: list[Block] = []
    content = message.get("content") or ""
    if content:
        blocks.append(Text(content))

    tool_calls = message.get("tool_calls") or []
    for i, call in enumerate(tool_calls):
        fn = call.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        blocks.append(ToolUse(id=call.get("id") or f"openai_{i}_{fn.get('name', 'tool')}", name=fn.get("name", ""), input=args))

    stop = "tool_use" if tool_calls else {"length": "max_tokens"}.get(finish_reason or "", "end_turn")
    tokens = Usage(
        input_tokens=usage.get("prompt_tokens", 0) or 0,
        output_tokens=usage.get("completion_tokens", 0) or 0,
    )
    return LLMResponse(message=Message(role="assistant", blocks=blocks), stop_reason=stop, usage=tokens, raw={"usage": usage})
