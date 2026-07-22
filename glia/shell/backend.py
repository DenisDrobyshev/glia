"""Build a glia Agent from the shell's config, plus a couple of demo tools.

Offline ("demo") mode uses the deterministic :class:`EchoLLM`, so the app runs
with zero setup — it streams your message back and the glass box lights up. Add
an Anthropic key in settings to switch to real Claude, where the demo tools
actually get called.
"""

from __future__ import annotations

import datetime

from ..agent import Agent
from ..providers import EchoLLM
from ..tools import tool
from .config import Config


@tool
async def current_time() -> str:
    """Return the current local date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
async def word_count(text: str) -> str:
    """Count the words and characters in a piece of text."""
    return f"{len(text.split())} words, {len(text)} characters"


DEMO_TOOLS = [current_time, word_count]


def build_agent(config: Config, *, stream: bool = True) -> Agent:
    """Assemble the Agent the shell will run for the next turn."""
    if config.mode == "claude" and config.anthropic_api_key:
        from ..providers import ClaudeLLM

        llm = ClaudeLLM(model=config.model, api_key=config.anthropic_api_key)
        tools = DEMO_TOOLS if config.use_tools else []
        return Agent(llm, tools=tools, system=config.system, stream=stream, name="glia")

    # Offline demo: an echo bot. It streams your message back so you can try the
    # UI and watch the event stream without any API key.
    return Agent(EchoLLM(default_reply=None), system=config.system, stream=stream, name="glia")
