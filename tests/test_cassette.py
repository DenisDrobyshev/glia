from __future__ import annotations

import pytest
from conftest import run

from glia import Agent, Cassette, RecordingLLM, ReplayLLM, tool, use_cassette
from glia.errors import ProviderError
from glia.llm import LLMRequest
from glia.providers import EchoLLM
from glia.providers import call as tcall
from glia.types import user


def _req(text: str) -> LLMRequest:
    return LLMRequest(messages=[user(text)])


def test_record_then_replay(tmp_path):
    path = tmp_path / "c.json"
    rec = RecordingLLM(EchoLLM(["hello there"]), path)
    r1 = run(rec.generate(_req("hi")))
    assert r1.message.text() == "hello there"
    assert path.exists()

    replayed = run(ReplayLLM(path).generate(_req("hi")))
    assert replayed.message.text() == "hello there"
    assert replayed.usage.input_tokens == r1.usage.input_tokens  # usage preserved


def test_use_cassette_auto_records_then_replays(tmp_path):
    path = tmp_path / "c.json"
    first = use_cassette(path, lambda: EchoLLM(["recorded answer"]))
    assert isinstance(first, RecordingLLM)
    run(first.generate(_req("q")))

    def boom():
        raise AssertionError("factory must not run on replay")

    second = use_cassette(path, boom)  # file now exists -> replay, factory untouched
    assert isinstance(second, ReplayLLM)
    assert run(second.generate(_req("q"))).message.text() == "recorded answer"


def test_full_agent_record_then_replay(tmp_path):
    path = tmp_path / "run.json"

    @tool
    async def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    real = EchoLLM([tcall("add", {"a": 2, "b": 3}), "The sum is 5."])
    rec_run = run(Agent(RecordingLLM(real, path), tools=[add]).run("add 2 and 3"))
    assert rec_run.output == "The sum is 5."

    # Replay the whole run — no real provider, deterministic tools re-run.
    replay_run = run(Agent(ReplayLLM(path), tools=[add]).run("add 2 and 3"))
    assert replay_run.output == "The sum is 5."
    assert "add" in {e.name for e in replay_run.trajectory.events_of("tool_called")}  # type: ignore[attr-defined]


def test_streaming_record_and_replay(tmp_path):
    path = tmp_path / "s.json"

    async def drain(llm):
        deltas, final = [], None
        async for chunk in llm.stream(_req("hi")):
            if chunk.response is not None:
                final = chunk.response
            else:
                deltas.append(chunk.text)
        return deltas, final

    d1, _ = run(drain(RecordingLLM(EchoLLM(["one two three"]), path)))
    assert "".join(d1) == "one two three"

    d2, final = run(drain(ReplayLLM(path)))
    assert "".join(d2) == "one two three"
    assert final.message.text() == "one two three"


def test_strict_replay_raises_on_unknown_request(tmp_path):
    path = tmp_path / "c.json"
    run(RecordingLLM(EchoLLM(["a"]), path).generate(_req("first")))
    with pytest.raises(ProviderError):
        run(ReplayLLM(path, strict=True).generate(_req("a completely different prompt")))


def test_sequential_fallback_when_not_strict(tmp_path):
    path = tmp_path / "c.json"
    run(RecordingLLM(EchoLLM(["only answer"]), path).generate(_req("recorded q")))
    # Different request, but non-strict falls back to the next unused interaction.
    out = run(ReplayLLM(path).generate(_req("some other q")))
    assert out.message.text() == "only answer"


def test_cassette_round_trips(tmp_path):
    interaction = {
        "key": "k",
        "request": {},
        "response": {"message": {"role": "assistant", "blocks": [{"type": "text", "text": "x"}]},
                     "stop_reason": "end_turn", "usage": {}},
    }
    Cassette([interaction]).save(tmp_path / "c.json")
    assert Cassette.load(tmp_path / "c.json").interactions == [interaction]


def test_record_mode_forces_recording_even_if_file_exists(tmp_path):
    path = tmp_path / "c.json"
    run(use_cassette(path, lambda: EchoLLM(["first"]), mode="record").generate(_req("q")))
    forced = use_cassette(path, lambda: EchoLLM(["second"]), mode="record")
    assert isinstance(forced, RecordingLLM)
