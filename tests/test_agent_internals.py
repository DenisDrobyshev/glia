from __future__ import annotations

from conftest import run

from glia import Agent, TrimmingCompactor
from glia.providers import EchoLLM
from glia.trajectory import Trajectory
from glia.types import assistant, user


def test_request_reflects_agent_config():
    llm = EchoLLM(["ok"])
    agent = Agent(
        llm,
        system="be terse",
        max_tokens=99,
        temperature=0.5,
        thinking=True,
        tool_choice="any",
    )
    run(agent.run("hi"))
    sent = llm.calls[0]
    assert sent.system == "be terse"
    assert sent.max_tokens == 99
    assert sent.temperature == 0.5
    assert sent.thinking is True
    assert sent.tool_choice == "any"


def test_compaction_fires_mid_run_and_emits_event():
    # Pre-load a long trajectory so the compactor triggers on the first step.
    traj = Trajectory.new(system="s")
    for i in range(4):
        traj.add_message(user(f"q{i}"))
        traj.add_message(assistant(f"a{i}"))  # 8 messages

    agent = Agent(EchoLLM(["done"]), compactor=TrimmingCompactor(max_messages=5, keep_last=2))
    result = run(agent.run("new question", trajectory=traj))

    assert result.output == "done"
    compacted = result.trajectory.events_of("compacted")
    assert len(compacted) == 1
    assert compacted[0].freed_messages > 0  # type: ignore[attr-defined]


def test_async_hook_is_awaited():
    seen: list[str] = []

    async def hook(event):
        seen.append(event.kind)

    agent = Agent(EchoLLM(["hi"]), hooks=[hook])
    run(agent.run("go"))
    assert "run_started" in seen and "run_finished" in seen


def test_trajectory_passed_in_is_the_one_returned():
    traj = Trajectory.new()
    agent = Agent(EchoLLM(["answer"]))
    result = run(agent.run("q", trajectory=traj))
    assert result.trajectory is traj  # no copy; you own the state


def test_pending_tool_uses_helper():
    from glia.types import Message, ToolUse

    traj = Trajectory.new()
    assert traj.pending_tool_uses() == []  # empty trajectory
    traj.add_message(Message("assistant", [ToolUse("t1", "f", {"a": 1})]))
    assert traj.pending_tool_uses()[0].name == "f"
    # A trailing user turn means no pending calls.
    traj.add_user("thanks")
    assert traj.pending_tool_uses() == []
