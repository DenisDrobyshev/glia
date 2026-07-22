"""Durable execution: a run is a JSON file you can save and resume.

Run: python examples/05_checkpoint_resume.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from _common import make_llm

from glia import Agent, Trajectory
from glia.checkpoint import checkpointer, load


async def main() -> None:
    path = Path(tempfile.gettempdir()) / "glia_run.json"

    # Run an agent, checkpointing to disk after every step.
    traj = Trajectory.new(system="You are helpful.")
    agent = Agent(make_llm(["The three primary colours are red, green, and blue."]), hooks=[checkpointer(traj, path)])
    result = await agent.run("Name the three primary colours.", trajectory=traj)
    print("ANSWER:", result.output)
    print("checkpoint written to:", path)

    # Later — even in a new process — reload and continue the conversation.
    resumed = load(path)
    print("\nreloaded", len(resumed.messages), "messages from disk")
    follow_up = Agent(make_llm(["Green is a primary colour of light."]))
    result2 = await follow_up.run("Which of those is a colour of light too?", trajectory=resumed)
    print("CONTINUED:", result2.output)


if __name__ == "__main__":
    asyncio.run(main())
