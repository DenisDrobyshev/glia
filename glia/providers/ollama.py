"""Local models via Ollama — run Qwen, DeepSeek, Llama, etc. with no API key.

Talks to a local Ollama server (``http://localhost:11434`` by default) over its
``/api/chat`` endpoint using only the standard library — no vendor SDK, no
third-party HTTP client. Supports streaming and tool calling for models that
offer it.

    from glia import Agent
    from glia.providers import OllamaLLM

    agent = Agent(OllamaLLM("qwen2.5"), system="Be concise.")
    result = await agent.run("Say hello.")

Prerequisite: install Ollama (https://ollama.com) and pull a model, e.g.
``ollama pull qwen2.5`` or ``ollama pull deepseek-r1``.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from typing import Any
from urllib.request import Request, urlopen

from ..errors import ProviderError
from ..llm import LLMRequest, LLMResponse, StreamChunk
from ..types import Block, Message, Text, ToolResult, ToolUse, Usage

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5"


class OllamaLLM:
    """An :class:`~glia.llm.LLM` backed by a local Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        host: str = DEFAULT_HOST,
        options: dict[str, Any] | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.options = dict(options or {})
        self.timeout = timeout

    # -- LLM protocol ----------------------------------------------------------

    async def generate(self, request: LLMRequest) -> LLMResponse:
        payload = self._payload(request, stream=False)

        def call() -> dict[str, Any]:
            with urlopen(self._request(payload), timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())

        try:
            data = await asyncio.to_thread(call)
        except Exception as exc:  # noqa: BLE001 - normalise to our error boundary
            raise ProviderError(f"Ollama request failed ({self.host}): {exc}") from exc
        return _response_from(data.get("message") or {}, data)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        payload = self._payload(request, stream=True)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker() -> None:
            try:
                with urlopen(self._request(payload), timeout=self.timeout) as resp:
                    for line in resp:
                        if line.strip():
                            loop.call_soon_threadsafe(queue.put_nowait, ("line", line))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("end", None))

        threading.Thread(target=worker, daemon=True).start()

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        final: dict[str, Any] = {}
        while True:
            kind, item = await queue.get()
            if kind == "end":
                break
            if kind == "error":
                raise ProviderError(f"Ollama stream failed ({self.host}): {item}")
            obj = json.loads(item.decode())
            message = obj.get("message") or {}
            content = message.get("content") or ""
            if content:
                text_parts.append(content)
                yield StreamChunk(text=content)
            if message.get("tool_calls"):
                tool_calls = message["tool_calls"]
            if obj.get("done"):
                final = obj

        message = {"content": "".join(text_parts), "tool_calls": tool_calls}
        yield StreamChunk(response=_response_from(message, final))

    # -- request building ------------------------------------------------------

    def _request(self, payload: dict[str, Any]) -> Request:
        return Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _payload(self, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": _to_ollama_messages(request.messages, request.system),
            "stream": stream,
        }
        if request.tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in request.tools
            ]
        options = dict(self.options)
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens:
            options["num_predict"] = request.max_tokens
        if options:
            body["options"] = options
        return body


# -- conversion ----------------------------------------------------------------


def _to_ollama_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    id_to_name: dict[str, str] = {}
    for message in messages:
        if message.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.text()}
            calls = []
            for use in message.tool_uses():
                id_to_name[use.id] = use.name
                calls.append({"function": {"name": use.name, "arguments": use.input}})
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
        else:
            # A user turn may carry plain text and/or tool results. Ollama wants
            # tool results as their own `role: "tool"` messages.
            texts = [b.text for b in message.blocks if isinstance(b, Text)]
            if texts:
                out.append({"role": "user", "content": "".join(texts)})
            for block in message.blocks:
                if isinstance(block, ToolResult):
                    tool_msg: dict[str, Any] = {"role": "tool", "content": block.content}
                    name = id_to_name.get(block.tool_use_id)
                    if name:
                        tool_msg["tool_name"] = name
                    out.append(tool_msg)
    return out


def _response_from(message: dict[str, Any], meta: dict[str, Any]) -> LLMResponse:
    blocks: list[Block] = []
    content = message.get("content") or ""
    if content:
        blocks.append(Text(content))

    tool_calls = message.get("tool_calls") or []
    for i, call in enumerate(tool_calls):
        fn = call.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {"_raw": args}
        blocks.append(ToolUse(id=f"ollama_{i}_{fn.get('name', 'tool')}", name=fn.get("name", ""), input=args))

    stop_reason = "tool_use" if tool_calls else {"length": "max_tokens"}.get(meta.get("done_reason", ""), "end_turn")
    usage = Usage(
        input_tokens=meta.get("prompt_eval_count", 0) or 0,
        output_tokens=meta.get("eval_count", 0) or 0,
    )
    return LLMResponse(message=Message(role="assistant", blocks=blocks), stop_reason=stop_reason, usage=usage, raw=meta)
