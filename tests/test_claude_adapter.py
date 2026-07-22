"""Tests for the Claude adapter, run fully offline via a fake Anthropic client.

The real ``anthropic`` SDK is never imported: ``ClaudeLLM(client=...)`` accepts
any object with a ``messages.create`` / ``messages.stream`` surface, so we
exercise every conversion path — request building, response parsing, streaming,
refusals, usage, thinking-block dropping — with no network and no API key.
"""

from __future__ import annotations

from conftest import run

from glia.errors import ProviderError
from glia.llm import LLMRequest, ToolSchema
from glia.providers.anthropic import (
    ClaudeLLM,
    _block_to_api,
    _keep,
    _message_to_api,
    _tool_choice,
    _usage_from_api,
)
from glia.types import Message, Text, Thinking, ToolResult, ToolUse, user

# --- fake SDK objects ---------------------------------------------------------


class _Blk:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Usage:
    def __init__(self, i=10, o=5, cr=0, cw=0):
        self.input_tokens = i
        self.output_tokens = o
        self.cache_read_input_tokens = cr
        self.cache_creation_input_tokens = cw


class _Resp:
    def __init__(self, content, stop_reason="end_turn", usage=None, stop_details=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or _Usage()
        self.stop_details = stop_details


class _Stream:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    @property
    def text_stream(self):
        async def gen():
            for block in self._resp.content:
                if getattr(block, "type", None) == "text":
                    yield block.text

        return gen()

    async def get_final_message(self):
        return self._resp


class _Messages:
    def __init__(self, resp):
        self._resp = resp
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._resp

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        return _Stream(self._resp)


class FakeAnthropic:
    def __init__(self, resp):
        self.messages = _Messages(resp)


# --- response parsing ---------------------------------------------------------


def test_generate_parses_text_and_usage():
    resp = _Resp([_Blk(type="text", text="hi")], usage=_Usage(11, 7, 3, 2))
    llm = ClaudeLLM(client=FakeAnthropic(resp))
    out = run(llm.generate(LLMRequest(messages=[user("q")])))
    assert out.message.text() == "hi"
    assert out.stop_reason == "end_turn"
    assert (out.usage.input_tokens, out.usage.output_tokens) == (11, 7)
    assert (out.usage.cache_read_tokens, out.usage.cache_write_tokens) == (3, 2)
    assert out.raw is resp


def test_generate_parses_tool_use():
    resp = _Resp([_Blk(type="tool_use", id="t1", name="add", input={"a": 1})], stop_reason="tool_use")
    out = run(ClaudeLLM(client=FakeAnthropic(resp)).generate(LLMRequest(messages=[user("q")])))
    assert out.wants_tools
    call = out.message.tool_uses()[0]
    assert (call.id, call.name, call.input) == ("t1", "add", {"a": 1})


def test_generate_parses_thinking_block():
    resp = _Resp([_Blk(type="thinking", thinking="because"), _Blk(type="text", text="answer")])
    out = run(ClaudeLLM(client=FakeAnthropic(resp)).generate(LLMRequest(messages=[user("q")])))
    assert out.message.thinking() == "because"
    assert out.message.text() == "answer"


def test_refusal_gets_a_readable_text_block():
    resp = _Resp([], stop_reason="refusal", stop_details=_Blk(explanation="not allowed"))
    out = run(ClaudeLLM(client=FakeAnthropic(resp)).generate(LLMRequest(messages=[user("q")])))
    assert out.stop_reason == "refusal"
    assert "[refusal]" in out.message.text()
    assert "not allowed" in out.message.text()


# --- request building ---------------------------------------------------------


def test_build_kwargs_maps_all_fields():
    fake = FakeAnthropic(_Resp([_Blk(type="text", text="ok")]))
    llm = ClaudeLLM("claude-test", client=fake)
    request = LLMRequest(
        messages=[user("q")],
        system="be terse",
        tools=[ToolSchema("t", "d", {"type": "object"})],
        tool_choice="t",
        max_tokens=123,
        stop=["END"],
    )
    run(llm.generate(request))
    kw = fake.messages.last_kwargs
    assert kw["model"] == "claude-test"
    assert kw["max_tokens"] == 123
    assert kw["system"] == "be terse"
    assert kw["tools"][0] == {"name": "t", "description": "d", "input_schema": {"type": "object"}}
    assert kw["tool_choice"] == {"type": "tool", "name": "t"}
    assert kw["stop_sequences"] == ["END"]


def test_thinking_enabled_and_temperature_suppressed():
    fake = FakeAnthropic(_Resp([_Blk(type="text", text="ok")]))
    llm = ClaudeLLM(client=fake)
    run(llm.generate(LLMRequest(messages=[user("q")], thinking=True, temperature=0.7)))
    kw = fake.messages.last_kwargs
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "temperature" not in kw  # sampling params suppressed when thinking is on


def test_temperature_sent_only_when_thinking_off():
    fake = FakeAnthropic(_Resp([_Blk(type="text", text="ok")]))
    run(ClaudeLLM(client=fake).generate(LLMRequest(messages=[user("q")], temperature=0.3)))
    assert fake.messages.last_kwargs["temperature"] == 0.3


def test_tool_choice_variants():
    assert _tool_choice("any") == {"type": "any"}
    assert _tool_choice("mytool") == {"type": "tool", "name": "mytool"}

    fake = FakeAnthropic(_Resp([_Blk(type="text", text="ok")]))
    # 'auto' / None must NOT set tool_choice (Anthropic defaults to auto).
    run(ClaudeLLM(client=fake).generate(
        LLMRequest(messages=[user("q")], tools=[ToolSchema("t", "d", {"type": "object"})], tool_choice="auto")
    ))
    assert "tool_choice" not in fake.messages.last_kwargs


# --- streaming ----------------------------------------------------------------


def test_stream_yields_deltas_then_final_response():
    resp = _Resp([_Blk(type="text", text="hello world")])
    llm = ClaudeLLM(client=FakeAnthropic(resp))

    async def collect():
        deltas, final = [], None
        async for chunk in llm.stream(LLMRequest(messages=[user("q")])):
            if chunk.response is not None:
                final = chunk.response
            else:
                deltas.append(chunk.text)
        return deltas, final

    deltas, final = collect_result = run(collect())
    assert "".join(deltas) == "hello world"
    assert final.message.text() == "hello world"
    assert collect_result[1].stop_reason == "end_turn"


# --- message/block conversion -------------------------------------------------


def test_message_to_api_shapes():
    assistant_msg = Message("assistant", [Text("hi"), ToolUse("t1", "add", {"a": 1})])
    api = _message_to_api(assistant_msg)
    assert api["role"] == "assistant"
    assert api["content"][0] == {"type": "text", "text": "hi"}
    assert api["content"][1] == {"type": "tool_use", "id": "t1", "name": "add", "input": {"a": 1}}


def test_tool_result_block_and_is_error():
    ok = _block_to_api(ToolResult("t1", "done"))
    assert ok == {"type": "tool_result", "tool_use_id": "t1", "content": "done"}
    err = _block_to_api(ToolResult("t2", "boom", is_error=True))
    assert err["is_error"] is True


def test_thinking_blocks_are_dropped_on_send():
    assert _keep(Text("x")) is True
    assert _keep(Thinking("y")) is False
    msg = Message("assistant", [Thinking("reasoning"), Text("visible")])
    api = _message_to_api(msg)
    assert [b["type"] for b in api["content"]] == ["text"]  # thinking omitted


def test_block_to_api_rejects_unknown():
    import pytest

    with pytest.raises(ProviderError):
        _block_to_api(object())  # type: ignore[arg-type]


def test_usage_from_none_is_zero():
    u = _usage_from_api(None)
    assert (u.input_tokens, u.output_tokens) == (0, 0)


def test_generate_wraps_provider_errors():
    import pytest

    class Boom:
        class messages:  # noqa: N801
            @staticmethod
            async def create(**kw):
                raise RuntimeError("network down")

    with pytest.raises(ProviderError):
        run(ClaudeLLM(client=Boom()).generate(LLMRequest(messages=[user("q")])))
