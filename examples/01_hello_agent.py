"""The smallest possible agent — and how to watch it think.

Run: python examples/01_hello_agent.py
"""

from __future__ import annotations

import asyncio

from _common import make_llm

from glia import Agent


async def main() -> None:
    agent = Agent(
        make_llm(["Bonjour! The capital of France is Paris."]),
        system="You are concise and friendly.",
    )

    # Option A: just get the answer.
    result = await agent.run("What is the capital of France?")
    print("ANSWER:", result.output)
    print("COST:  ", result.usage)

    # Option B: watch every step of the glass box as it happens.
    print("\n--- event stream ---")
    async for event in agent.run_events("Say hello in French."):
        print(f"  {event.kind}")


if __name__ == "__main__":
    asyncio.run(main())
