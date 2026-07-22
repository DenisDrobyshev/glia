"""Streaming output, parallel tools, and a human-in-the-loop approval gate —
all visible in one event stream.

Run: python examples/07_streaming_and_approval.py
"""

from __future__ import annotations

import asyncio

from _common import make_llm

from glia import Agent, tool
from glia.approval import deny
from glia.providers import call


@tool
async def search(q: str) -> str:
    """Search the knowledge base."""
    return f"3 results for {q!r}"


@tool
async def delete_all(confirm: bool = False) -> str:
    """Delete everything (the dangerous one)."""
    return "deleted everything"


async def main() -> None:
    # One turn asks for TWO tools at once (they run in parallel); the next turn
    # streams the final answer word by word.
    llm = make_llm(
        [
            [call("search", {"q": "glia"}, id="1"), call("delete_all", {"confirm": True}, id="2")],
            "I searched successfully. The delete was blocked, so I left your data alone.",
        ]
    )

    agent = Agent(
        llm,
        tools=[search, delete_all],
        stream=True,            # tokens arrive incrementally
        parallel_tools=True,    # the two tool calls run concurrently
        approval=deny("delete_all"),  # gate: this tool is never allowed to run
    )

    print("--- live trace ---")
    async for event in agent.run_events("Search for glia, then wipe the database."):
        if event.kind == "model_delta":
            print(event.text, end="", flush=True)  # streamed tokens
        elif event.kind == "approval_resolved":
            verdict = "ALLOWED" if event.allowed else f"DENIED ({event.reason})"
            print(f"\n  [approval] {event.name}: {verdict}")
        elif event.kind == "tool_returned":
            flag = " (blocked)" if event.is_error else ""
            print(f"  [tool] {event.name} -> {event.content!r}{flag}")
    print("\n--- end ---")


if __name__ == "__main__":
    asyncio.run(main())
