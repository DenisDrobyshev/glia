"""Tests for the Ollama provider, run fully offline by faking ``urlopen``.

No local Ollama server is needed — we monkeypatch the module-level ``urlopen``
with a fake that returns canned ``/api/chat`` responses (single JSON for
``generate``, newline-delimited JSON for ``stream``).
"""

from __future__ import annotations

import json

import pytest
from conftest import run

from glia.errors import ProviderError
from glia.llm import LLMRequest, ToolSchema
from glia.providers import OllamaLLM
from glia.providers import ollama as ollama_mod
from glia.providers.ollama import _to_ollama_messages
from glia.types import Message, Text, ToolResult, ToolUse, user


class FakeResp:
    def __init__(self, body: bytes = b"", lines: list[bytes] | None = None):
        self._body = body
        self._lines = lines or []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)


def _patch(monkeypatch, resp, capture=None):
    def fake_urlopen(req, timeout=None):
        if capture is not None:
            capture["url"] = req.full_url
            capture["body"] = json.loads(req.data.decode())
        return resp

    monkeypatch.setattr(ollama_mod, "urlopen", fake_urlopen)


# --- generate -----------------------------------------------------------------


def test_generate_parses_content_and_usage(monkeypatch):
    data = {"message": {"content": "hi there"}, "done": True, "done_reason": "stop",
            "prompt_eval_count": 12, "eval_count": 5}
    cap = {}
    _patch(monkeypatch, FakeResp(body=json.dumps(data).encode()), cap)
    out = run(OllamaLLM("qwen2.5").generate(LLMRequest(messages=[user("hey")])))
    assert out.message.text() == "hi there"
    assert out.stop_reason == "end_turn"
    assert (out.usage.input_tokens, out.usage.output_tokens) == (12, 5)
    assert cap["body"]["model"] == "qwen2.5"
    assert cap["body"]["stream"] is False
    assert "localhost:11434/api/chat" in cap["url"]


def test_generate_tool_call(monkeypatch):
    data = {"message": {"content": "", "tool_calls": [{"function": {"name": "add", "arguments": {"a": 1, "b": 2}}}]},
            "done": True}
    _patch(monkeypatch, FakeResp(body=json.dumps(data).encode()))
    out = run(OllamaLLM().generate(
        LLMRequest(messages=[user("add")], tools=[ToolSchema("add", "d", {"type": "object"})])
    ))
    assert out.wants_tools
    call = out.message.tool_uses()[0]
    assert call.name == "add" and call.input == {"a": 1, "b": 2}
    assert out.stop_reason == "tool_use"


def test_tool_arguments_as_json_string(monkeypatch):
    data = {"message": {"content": "", "tool_calls": [{"function": {"name": "f", "arguments": '{"x": 5}'}}]},
            "done": True}
    _patch(monkeypatch, FakeResp(body=json.dumps(data).encode()))
    out = run(OllamaLLM().generate(LLMRequest(messages=[user("q")])))
    assert out.message.tool_uses()[0].input == {"x": 5}


def test_payload_includes_tools_and_options(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeResp(body=json.dumps({"message": {"content": "x"}, "done": True}).encode()), cap)
    run(OllamaLLM("m", options={"seed": 1}).generate(
        LLMRequest(messages=[user("q")], tools=[ToolSchema("t", "d", {"type": "object"})],
                   temperature=0.5, max_tokens=256)
    ))
    body = cap["body"]
    assert body["tools"][0]["function"]["name"] == "t"
    assert body["options"]["temperature"] == 0.5
    assert body["options"]["num_predict"] == 256
    assert body["options"]["seed"] == 1


def test_error_is_wrapped(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(ollama_mod, "urlopen", boom)
    with pytest.raises(ProviderError):
        run(OllamaLLM().generate(LLMRequest(messages=[user("q")])))


# --- stream -------------------------------------------------------------------


def test_stream_yields_deltas_then_final(monkeypatch):
    lines = [
        json.dumps({"message": {"content": "Hel"}, "done": False}).encode() + b"\n",
        json.dumps({"message": {"content": "lo"}, "done": False}).encode() + b"\n",
        json.dumps({"message": {"content": ""}, "done": True, "eval_count": 3, "prompt_eval_count": 4}).encode() + b"\n",
    ]
    _patch(monkeypatch, FakeResp(lines=lines))

    async def collect():
        deltas, final = [], None
        async for chunk in OllamaLLM().stream(LLMRequest(messages=[user("hi")])):
            if chunk.response is not None:
                final = chunk.response
            else:
                deltas.append(chunk.text)
        return deltas, final

    deltas, final = run(collect())
    assert "".join(deltas) == "Hello"
    assert final.message.text() == "Hello"
    assert final.usage.output_tokens == 3


# --- conversion ---------------------------------------------------------------


def test_message_conversion_with_tool_results():
    msgs = [
        user("hi"),
        Message("assistant", [Text("let me look"), ToolUse("id1", "search", {"q": "x"})]),
        Message("user", [ToolResult("id1", "found it")]),
    ]
    out = _to_ollama_messages(msgs, "you are helpful")
    assert out[0] == {"role": "system", "content": "you are helpful"}
    assert out[1] == {"role": "user", "content": "hi"}
    assert out[2]["role"] == "assistant"
    assert out[2]["tool_calls"][0]["function"]["name"] == "search"
    assert out[3] == {"role": "tool", "content": "found it", "tool_name": "search"}


def test_host_trailing_slash_stripped():
    assert OllamaLLM(host="http://x:1/").host == "http://x:1"
