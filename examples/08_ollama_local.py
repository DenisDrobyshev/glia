"""Run a local open model (Qwen, DeepSeek, …) via Ollama — no API key.

Prerequisite: install Ollama from https://ollama.com, then pull a model:
    ollama pull qwen2.5      # or: ollama pull deepseek-r1

Run: python examples/08_ollama_local.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Let the example run straight from a clone, before `pip install -e .`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glia import Agent  # noqa: E402
from glia.errors import ProviderError  # noqa: E402
from glia.providers import OllamaLLM  # noqa: E402

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")


async def main() -> None:
    agent = Agent(OllamaLLM(MODEL), system="You are concise.", stream=True)
    print(f"Talking to local model '{MODEL}' via Ollama…\n")
    try:
        async for event in agent.run_events("In one sentence, what is a glial cell?"):
            if event.kind == "model_delta":
                print(event.text, end="", flush=True)
        print()
    except ProviderError as exc:
        print(f"Could not reach Ollama: {exc}\n")
        print("Install it from https://ollama.com, then run:")
        print(f"  ollama pull {MODEL}")
        print("and try again. (Set OLLAMA_MODEL to use a different model.)")


if __name__ == "__main__":
    asyncio.run(main())
