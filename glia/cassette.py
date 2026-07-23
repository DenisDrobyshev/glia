"""Record/replay cassettes — VCR for the ``LLM`` protocol.

Record a real provider's responses once, then replay them deterministically with
no network and no API key. Because glia's requests and responses are plain
serialisable data, a cassette is just a readable JSON file you can inspect and
commit alongside your tests.

    from glia import Agent, use_cassette
    from glia.providers import ClaudeLLM

    # First run records (needs a key); later runs replay offline — same output.
    llm = use_cassette("tests/cassettes/weather.json", ClaudeLLM)
    agent = Agent(llm, tools=[get_weather])
    result = await agent.run("What's the weather in Paris?")

Matching prefers an exact request key (so replay reproduces the recorded run),
and falls back to the next unused interaction in order for minor drift. Pass
``mode="record"`` / ``"replay"`` to force a direction, or ``strict=True`` on
:class:`ReplayLLM` to require exact key matches.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import ProviderError
from .llm import LLM, LLMRequest, LLMResponse, StreamChunk
from .types import Message, Usage

# --- serialisation ------------------------------------------------------------


def _serialize_request(req: LLMRequest) -> dict[str, Any]:
    return {
        "system": req.system,
        "messages": [m.to_dict() for m in req.messages],
        "tools": [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in req.tools],
        "tool_choice": req.tool_choice,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "thinking": req.thinking,
        "stop": req.stop,
    }


def _request_key(req: LLMRequest) -> str:
    canonical = json.dumps(_serialize_request(req), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _serialize_response(resp: LLMResponse) -> dict[str, Any]:
    return {"message": resp.message.to_dict(), "stop_reason": resp.stop_reason, "usage": resp.usage.to_dict()}


def _deserialize_response(data: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        message=Message.from_dict(data["message"]),
        stop_reason=data["stop_reason"],
        usage=Usage(**data.get("usage", {})),
        raw=None,
    )


def _word_chunks(text: str) -> list[str]:
    return re.findall(r"\S+\s*", text) if text else []


# --- cassette -----------------------------------------------------------------


class Cassette:
    """A recorded list of ``{key, request, response}`` interactions; JSON on disk."""

    def __init__(self, interactions: list[dict[str, Any]] | None = None) -> None:
        self.interactions = interactions or []

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "interactions": self.interactions}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cassette:
        return cls(interactions=list(data.get("interactions", [])))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Cassette:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --- recording ----------------------------------------------------------------


class RecordingLLM:
    """Wraps a real :class:`~glia.llm.LLM`, forwarding calls and recording each
    ``(request, response)`` to a cassette file (written after every call)."""

    def __init__(self, inner: LLM, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.cassette = Cassette.load(self.path) if self.path.exists() else Cassette()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        response = await self.inner.generate(request)
        self._append(request, response)
        return response

    async def stream(self, request: LLMRequest):
        inner = self.inner
        if hasattr(inner, "stream"):
            response: LLMResponse | None = None
            async for chunk in inner.stream(request):
                if chunk.response is not None:
                    response = chunk.response
                yield chunk
            if response is not None:
                self._append(request, response)
        else:
            response = await inner.generate(request)
            self._append(request, response)
            for piece in _word_chunks(response.message.text()):
                yield StreamChunk(text=piece)
            yield StreamChunk(response=response)

    def _append(self, request: LLMRequest, response: LLMResponse) -> None:
        self.cassette.interactions.append(
            {"key": _request_key(request), "request": _serialize_request(request), "response": _serialize_response(response)}
        )
        self.cassette.save(self.path)


# --- replay -------------------------------------------------------------------


class ReplayLLM:
    """Replays a cassette deterministically — satisfies :class:`~glia.llm.LLM`
    and the streaming protocol, with no network."""

    def __init__(self, path: str | Path, *, strict: bool = False) -> None:
        self.cassette = Cassette.load(path)
        self.strict = strict
        self._used: set[int] = set()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return _deserialize_response(self._next(_request_key(request)))

    async def stream(self, request: LLMRequest):
        response = await self.generate(request)
        for piece in _word_chunks(response.message.text()):
            yield StreamChunk(text=piece)
        yield StreamChunk(response=response)

    def _next(self, key: str) -> dict[str, Any]:
        # Prefer an unused interaction whose request matches exactly.
        for i, interaction in enumerate(self.cassette.interactions):
            if i not in self._used and interaction["key"] == key:
                self._used.add(i)
                return interaction["response"]
        # Fall back to the next unused interaction in record order.
        if not self.strict:
            for i, interaction in enumerate(self.cassette.interactions):
                if i not in self._used:
                    self._used.add(i)
                    return interaction["response"]
        raise ProviderError(
            f"cassette has no recorded response for this request (key {key[:8]}…). "
            "Re-record the cassette, or set strict=False to allow ordered fallback."
        )


# --- convenience --------------------------------------------------------------


def use_cassette(
    path: str | Path,
    real_llm_factory: Callable[[], LLM] | LLM,
    *,
    mode: str = "auto",
) -> LLM:
    """Record on first use, replay thereafter.

    ``mode="auto"`` (default) records if the cassette file is missing and replays
    if it exists; ``"record"`` / ``"replay"`` force a direction. When recording,
    ``real_llm_factory`` is used to build the real provider (a factory is only
    called if recording, so replay never needs a key).
    """
    p = Path(path)
    replaying = mode == "replay" or (mode == "auto" and p.exists())
    if replaying:
        return ReplayLLM(p)
    inner = real_llm_factory() if callable(real_llm_factory) else real_llm_factory
    return RecordingLLM(inner, p)
