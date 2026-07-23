"""Tests for the OpenAI provider, run fully offline by faking ``urlopen``."""

from __future__ import annotations

import json

import pytest
from conftest import run

from glia.errors import ProviderError
from glia.llm import LLMRequest, ToolSchema
from glia.providers import OpenAILLM
from glia.providers import openai as openai_mod
from glia.providers.openai import _to_openai_messages, _tool_choice
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
            capture["headers"] = dict(req.headers)
            capture["body"] = json.loads(req.data.decode())
        return resp

    monkeypatch.setattr(openai_mod, "urlopen", fake_urlopen)


def _completion(message, finish_reason="stop", usage=None):
    return {"choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 8, "completion_tokens": 4}}


# --- generate -----------------------------------------------------------------


def test_generate_parses_content_and_usage(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeResp(body=json.dumps(_completion({"content": "hi"})).encode()), cap)
    out = run(OpenAILLM("gpt-4o-mini", api_key="sk-test").generate(LLMRequest(messages=[user("hey")])))
    assert out.message.text() == "hi"
    assert out.stop_reason == "end_turn"
    assert (out.usage.input_tokens, out.usage.output_tokens) == (8, 4)
    assert cap["headers"]["Authorization"] == "Bearer sk-test"
    assert cap["url"].endswith("/chat/completions")


def test_generate_tool_call_parses_json_string_args(monkeypatch):
    message = {"content": None, "tool_calls": [
        {"id": "call_1", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}}]}
    _patch(monkeypatch, FakeResp(body=json.dumps(_completion(message, "tool_calls")).encode()))
    out = run(OpenAILLM(api_key="k").generate(
        LLMRequest(messages=[user("add")], tools=[ToolSchema("add", "d", {"type": "object"})])
    ))
    assert out.wants_tools and out.stop_reason == "tool_use"
    call = out.message.tool_uses()[0]
    assert call.id == "call_1" and call.name == "add" and call.input == {"a": 1, "b": 2}


def test_payload_has_tools_and_tool_choice(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeResp(body=json.dumps(_completion({"content": "x"})).encode()), cap)
    run(OpenAILLM(api_key="k").generate(
        LLMRequest(messages=[user("q")], tools=[ToolSchema("t", "d", {"type": "object"})],
                   tool_choice="any", temperature=0.2, max_tokens=99)
    ))
    body = cap["body"]
    assert body["tools"][0]["function"]["name"] == "t"
    assert body["tool_choice"] == "required"  # 'any' -> 'required'
    assert body["temperature"] == 0.2 and body["max_tokens"] == 99


def test_tool_choice_variants():
    assert _tool_choice(None) == "auto"
    assert _tool_choice("auto") == "auto"
    assert _tool_choice("any") == "required"
    assert _tool_choice("f") == {"type": "function", "function": {"name": "f"}}


def test_error_is_wrapped(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("401 unauthorized")

    monkeypatch.setattr(openai_mod, "urlopen", boom)
    with pytest.raises(ProviderError):
        run(OpenAILLM(api_key="k").generate(LLMRequest(messages=[user("q")])))


def test_api_key_from_env_and_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    llm = OpenAILLM(base_url="https://groq.example/v1/")
    assert llm.api_key == "sk-env"
    assert llm.base_url == "https://groq.example/v1"  # trailing slash stripped


# --- stream -------------------------------------------------------------------


def test_stream_text_deltas_and_usage(monkeypatch):
    lines = [
        b'data: ' + json.dumps({"choices": [{"delta": {"content": "Hel"}}]}).encode() + b"\n",
        b'data: ' + json.dumps({"choices": [{"delta": {"content": "lo"}}]}).encode() + b"\n",
        b'data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode() + b"\n",
        b'data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}).encode() + b"\n",
        b"data: [DONE]\n",
    ]
    _patch(monkeypatch, FakeResp(lines=lines))

    async def collect():
        deltas, final = [], None
        async for chunk in OpenAILLM(api_key="k").stream(LLMRequest(messages=[user("hi")])):
            if chunk.response is not None:
                final = chunk.response
            else:
                deltas.append(chunk.text)
        return deltas, final

    deltas, final = run(collect())
    assert "".join(deltas) == "Hello"
    assert final.message.text() == "Hello"
    assert final.usage.output_tokens == 2


def test_stream_accumulates_tool_calls(monkeypatch):
    lines = [
        b'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_9", "function": {"name": "add", "arguments": ""}}]}}]}).encode() + b"\n",
        b'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"a":'}}]}}]}).encode() + b"\n",
        b'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '1}'}}]}, "finish_reason": "tool_calls"}]}).encode() + b"\n",
        b"data: [DONE]\n",
    ]
    _patch(monkeypatch, FakeResp(lines=lines))

    async def collect():
        final = None
        async for chunk in OpenAILLM(api_key="k").stream(LLMRequest(messages=[user("go")])):
            if chunk.response is not None:
                final = chunk.response
        return final

    final = run(collect())
    call = final.message.tool_uses()[0]
    assert call.id == "call_9" and call.name == "add" and call.input == {"a": 1}
    assert final.stop_reason == "tool_use"


# --- conversion ---------------------------------------------------------------


def test_message_conversion_round_trips_tool_calls():
    msgs = [
        user("hi"),
        Message("assistant", [Text("sure"), ToolUse("call_1", "search", {"q": "x"})]),
        Message("user", [ToolResult("call_1", "found")]),
    ]
    out = _to_openai_messages(msgs, "sys")
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "hi"}
    assert out[2]["role"] == "assistant"
    tc = out[2]["tool_calls"][0]
    assert tc["id"] == "call_1" and tc["function"]["name"] == "search"
    assert json.loads(tc["function"]["arguments"]) == {"q": "x"}
    assert out[3] == {"role": "tool", "tool_call_id": "call_1", "content": "found"}
