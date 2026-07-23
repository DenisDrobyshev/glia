"""Record/replay cassettes: record a run once, replay it deterministically offline.

Run: python examples/09_record_replay.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

# Let the example run straight from a clone, before `pip install -e .`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glia import Agent, tool, use_cassette  # noqa: E402
from glia.providers import EchoLLM, call  # noqa: E402

CASSETTE = os.path.join(tempfile.gettempdir(), "glia_cassette_demo.json")


@tool
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def real_provider() -> EchoLLM:
    # Stands in for a real, paid provider you don't want to call twice.
    return EchoLLM([call("add", {"a": 2, "b": 3}), "The sum is 5."])


async def main() -> None:
    if os.path.exists(CASSETTE):
        os.remove(CASSETTE)

    print("recording (would hit the real provider)…")
    rec = await Agent(use_cassette(CASSETTE, real_provider, mode="record"), tools=[add]).run("add 2 and 3")
    print("  →", rec.output)

    print("replaying (fully offline, no provider)…")
    rep = await Agent(use_cassette(CASSETTE, real_provider, mode="replay"), tools=[add]).run("add 2 and 3")
    print("  →", rep.output)

    print("cassette saved to:", CASSETTE)


if __name__ == "__main__":
    asyncio.run(main())
