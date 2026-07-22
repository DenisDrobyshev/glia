from __future__ import annotations

import pytest
from conftest import run

from glia.llm import LLM, LLMRequest, LLMResponse, StreamChunk, StreamingLLM, ToolSchema
from glia.providers import EchoLLM, call, reply
from glia.types import Message, Text, ToolUse


def test_llm_request_defaults():
    req = LLMRequest(messages=[])
    assert req.max_tokens == 4096
    assert req.tools == [] and req.tool_choice is None
    assert req.thinking is False


def test_llm_response_wants_tools():
    with_tool = LLMResponse(Message("assistant", [ToolUse("t", "f", {})]), "tool_use")
    plain = LLMResponse(Message("assistant", [Text("hi")]), "end_turn")
    assert with_tool.wants_tools is True
    assert plain.wants_tools is False


def test_stream_chunk_defaults():
    assert StreamChunk().text == "" and StreamChunk().response is None


def test_echo_and_streaming_protocol_membership():
    e = EchoLLM()
    assert isinstance(e, LLM)
    assert isinstance(e, StreamingLLM)


def test_echo_scripted_then_echo_fallback():
    llm = EchoLLM([reply("first")])
    r1 = run(llm.generate(LLMRequest(messages=[Message("user", [Text("hello")])])))
    assert r1.message.text() == "first"
    # script exhausted -> echoes the last user text
    r2 = run(llm.generate(LLMRequest(messages=[Message("user", [Text("again")])])))
    assert r2.message.text() == "echo: again"
    assert len(llm.calls) == 2  # every request is recorded


def test_echo_default_reply_and_tool_turn():
    llm = EchoLLM(default_reply="canned")
    assert run(llm.generate(LLMRequest(messages=[]))).message.text() == "canned"

    tool_llm = EchoLLM([call("f", {"a": 1}, id="x")])
    resp = run(tool_llm.generate(LLMRequest(messages=[])))
    assert resp.stop_reason == "tool_use"
    assert resp.message.tool_uses()[0].id == "x"


def test_echo_callable_turn_receives_request():
    seen = {}

    def make(request):
        seen["system"] = request.system
        return [Text("computed")]

    llm = EchoLLM([make])
    out = run(llm.generate(LLMRequest(messages=[], system="S")))
    assert out.message.text() == "computed"
    assert seen["system"] == "S"


def test_echo_stream_chunks_text_then_final():
    llm = EchoLLM(["alpha beta gamma"])

    async def collect():
        deltas, final = [], None
        async for chunk in llm.stream(LLMRequest(messages=[])):
            if chunk.response is not None:
                final = chunk.response
            else:
                deltas.append(chunk.text)
        return deltas, final

    deltas, final = run(collect())
    assert len(deltas) == 3
    assert "".join(deltas) == "alpha beta gamma"
    assert final.message.text() == "alpha beta gamma"


def test_echo_stream_tool_turn_has_no_text_deltas():
    llm = EchoLLM([call("f", {})])

    async def collect():
        return [c async for c in llm.stream(LLMRequest(messages=[]))]

    chunks = run(collect())
    assert all(c.text == "" for c in chunks)  # only the final response chunk
    assert chunks[-1].response is not None


def test_tool_schema_is_a_value():
    s = ToolSchema("n", "d", {"type": "object"})
    assert (s.name, s.description) == ("n", "d")


def test_providers_lazy_claude_import_and_bad_attr():
    import glia.providers as providers

    assert providers.ClaudeLLM.__name__ == "ClaudeLLM"  # lazily resolved
    with pytest.raises(AttributeError):
        _ = providers.DoesNotExist
