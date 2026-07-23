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


def build_agent(config: Config, *, stream: bool = True, approval=None) -> Agent:
    """Assemble the Agent the shell will run for the next turn."""
    tools = DEMO_TOOLS if config.use_tools else []
    common = {"tools": tools, "system": config.system, "stream": stream, "approval": approval, "name": "glia"}

    if config.mode == "claude" and config.anthropic_api_key:
        from ..providers import ClaudeLLM

        return Agent(ClaudeLLM(model=config.model, api_key=config.anthropic_api_key), **common)

    if config.mode == "ollama":
        from ..providers import OllamaLLM

        return Agent(OllamaLLM(model=config.ollama_model, host=config.ollama_host), **common)

    if config.mode == "openai" and config.openai_api_key:
        from ..providers import OpenAILLM

        llm = OpenAILLM(model=config.openai_model, api_key=config.openai_api_key, base_url=config.openai_base_url)
        return Agent(llm, **common)

    # Offline demo: an echo bot. It streams your message back so you can try the
    # UI and watch the event stream without any API key or local server.
    return Agent(EchoLLM(default_reply=None), system=config.system, stream=stream, name="glia")
