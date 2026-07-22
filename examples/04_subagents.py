"""Subagents: any agent can be exposed as a tool for another agent.

Run: python examples/04_subagents.py
"""

from __future__ import annotations

import asyncio

from _common import make_llm

from glia import Agent, tool
from glia.providers import call


@tool
async def web_search(query: str) -> str:
    """Search the web (stubbed for the demo)."""
    return "glia is a glass-box agent library."


async def main() -> None:
    # A specialist researcher agent...
    researcher = Agent(
        make_llm([call("web_search", {"query": "glia library"}), "Found: glia is a glass-box agent library."]),
        tools=[web_search],
        name="researcher",
    )

    # ...becomes a single tool the lead agent can delegate to.
    lead = Agent(
        make_llm([call("researcher_agent", {"request": "what is glia?"}), "In short: glia is a glass-box agent library."]),
        tools=[researcher.as_tool("researcher_agent", "Delegate research to a specialist")],
        name="lead",
    )

    result = await lead.run("Look up what glia is and summarise it.")
    print("ANSWER:", result.output)
    print("\nsub-task delegated via tool:", [e.name for e in result.trajectory.events_of("tool_called")])  # type: ignore[attr-defined]


if __name__ == "__main__":
    asyncio.run(main())
