"""Evals as tests: a regression suite you can run in CI.

Run: python examples/06_evals.py
"""

from __future__ import annotations

import asyncio

from _common import make_llm

from glia import Agent, tool
from glia.evals import Case, contains, did_not_error, evaluate, used_tool
from glia.providers import call


@tool
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def make_agent() -> Agent:
    # A fresh agent per case (the EchoLLM script is single-use).
    return Agent(make_llm([call("add", {"a": 2, "b": 2}), "The answer is 4."]), tools=[add])


async def main() -> None:
    suite = [
        Case("arithmetic answer", "what is 2 + 2?", [contains("4"), did_not_error]),
        Case("uses the tool", "what is 2 + 2?", [used_tool("add")]),
    ]
    report = await evaluate(suite, make_agent)
    print(report)
    print("\nsuite passed:", report.ok)


if __name__ == "__main__":
    asyncio.run(main())
