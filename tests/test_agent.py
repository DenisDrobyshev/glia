from __future__ import annotations

from conftest import run

from glia import Agent, tool
from glia.providers import EchoLLM, call


@tool
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def test_plain_answer_no_tools():
    agent = Agent(EchoLLM(["Hello there."]))
    result = run(agent.run("hi"))
    assert result.output == "Hello there."
    assert result.stop_reason == "end_turn"
    assert result.steps == 1


def test_tool_loop_runs_tool_then_answers():
    llm = EchoLLM([call("add", {"a": 2, "b": 3}), "The sum is 5."])
    agent = Agent(llm, tools=[add])
    result = run(agent.run("add 2 and 3"))

    assert result.output == "The sum is 5."
    assert result.steps == 2

    # The tool actually ran and its result went back to the model.
    kinds = [e.kind for e in result.trajectory.events]
    assert "tool_called" in kinds and "tool_returned" in kinds
    returned = result.trajectory.events_of("tool_returned")[0]
    assert returned.content == "5"  # type: ignore[attr-defined]

    # Usage accumulated across both model calls.
    assert result.usage.input_tokens > 0


def test_hooks_observe_every_event():
    seen: list[str] = []
    llm = EchoLLM([call("add", {"a": 1, "b": 1}), "done"])
    agent = Agent(llm, tools=[add], hooks=[lambda e: seen.append(e.kind)])
    run(agent.run("go"))
    assert seen[0] == "run_started"
    assert seen[-1] == "run_finished"
    assert "model_call" in seen and "tool_returned" in seen


def test_run_events_streams_in_order():
    llm = EchoLLM(["answer"])
    agent = Agent(llm)

    async def collect() -> list[str]:
        return [e.kind async for e in agent.run_events("hi")]

    kinds = run(collect())
    assert kinds == ["run_started", "model_call", "model_response", "run_finished"]


def test_max_steps_exceeded_raises():
    import pytest

    from glia.errors import MaxStepsExceeded

    # The model keeps calling a tool forever; the loop must give up.
    llm = EchoLLM([call("add", {"a": 1, "b": 1}) for _ in range(20)])
    agent = Agent(llm, tools=[add], max_steps=3)
    with pytest.raises(MaxStepsExceeded):
        run(agent.run("loop forever"))


def test_subagent_as_tool():
    child = Agent(EchoLLM(["child says hi"]), name="child")
    parent = Agent(
        EchoLLM([call("child_agent", {"request": "greet"}), "parent wraps: child says hi"]),
        tools=[child.as_tool()],
    )
    result = run(parent.run("delegate"))
    assert "child says hi" in result.output
    assert "child_agent" in {e.name for e in result.trajectory.events_of("tool_called")}  # type: ignore[attr-defined]
