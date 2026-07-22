"""Tools are plain typed functions. The agent loop runs them for you.

Run: python examples/02_tools.py
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from _common import make_llm

from glia import Agent, tool
from glia.providers import call


@tool
async def get_weather(city: Annotated[str, "The city to look up"]) -> str:
    """Get the current weather for a city."""
    fake = {"Paris": "18°C, cloudy", "Tokyo": "24°C, clear"}
    return fake.get(city, "unknown")


async def main() -> None:
    # Offline script: the model calls the tool, then answers with the result.
    llm = make_llm([call("get_weather", {"city": "Paris"}), "It's 18°C and cloudy in Paris."])
    agent = Agent(llm, tools=[get_weather], system="Use tools when you need live data.")

    result = await agent.run("What's the weather in Paris?")
    print("ANSWER:", result.output)

    print("\n--- what the tool actually did ---")
    for event in result.trajectory.events_of("tool_returned"):
        print(f"  {event.name}({event.tool_use_id}) -> {event.content!r}")  # type: ignore[attr-defined]


if __name__ == "__main__":
    asyncio.run(main())
