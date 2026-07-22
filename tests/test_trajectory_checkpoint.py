from __future__ import annotations

from pathlib import Path

from conftest import run

from glia import Agent, Trajectory, tool
from glia.checkpoint import checkpointer, dumps, load, loads
from glia.providers import EchoLLM, call
from glia.types import Text, ToolResult, ToolUse, user


@tool
async def add(a: int, b: int) -> int:
    """Add."""
    return a + b


def test_trajectory_round_trips_through_json():
    traj = Trajectory.new(system="be nice")
    traj.add_user("hi")
    traj.add_message(
        __import__("glia").Message(role="assistant", blocks=[Text("ok"), ToolUse("t1", "add", {"a": 1})])
    )
    traj.add_tool_results([ToolResult("t1", "1")])

    restored = Trajectory.from_dict(traj.to_dict())
    assert restored.system == "be nice"
    assert len(restored.messages) == 3
    assert restored.messages[1].tool_uses()[0].name == "add"
    assert restored.messages[2].blocks[0].content == "1"


def test_dumps_loads_symmetry():
    traj = Trajectory.new()
    traj.add_message(user("hello"))
    assert loads(dumps(traj)).messages[0].text() == "hello"


def test_checkpoint_and_resume(tmp_path: Path):
    path = tmp_path / "run.json"

    # First run: model calls the tool, then answers. Checkpoint along the way.
    traj = Trajectory.new()
    llm1 = EchoLLM([call("add", {"a": 2, "b": 2}), "The total is 4."])
    agent1 = Agent(llm1, tools=[add], hooks=[checkpointer(traj, path)])
    result1 = run(agent1.run("add 2 and 2", trajectory=traj))
    assert result1.output == "The total is 4."
    assert path.exists()

    # Reload the finished trajectory from disk and confirm state survived.
    reloaded = load(path)
    assert reloaded.messages[-1].role == "assistant"
    assert reloaded.last_assistant_text() == "The total is 4."


def test_resume_continues_a_conversation():
    # Simulate a saved conversation, then continue it with a new prompt.
    traj = Trajectory.new(system="s")
    traj.add_user("first question")
    traj.add_message(__import__("glia").assistant("first answer"))

    saved = loads(dumps(traj))
    agent = Agent(EchoLLM(["second answer"]))
    result = run(agent.run("second question", trajectory=saved))
    assert result.output == "second answer"
    # History was preserved: original turns + new user turn + new answer.
    assert len(result.trajectory.messages) == 4
