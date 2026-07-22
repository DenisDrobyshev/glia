# glia

**A glass-box, minimal library for building LLM agents.** Every model call, tool
call, and state transition is a plain object you can log, snapshot, and replay.
No hidden control flow. The whole loop fits in one file you can read in an
afternoon.

> _glia_ (n.): the cells that support and connect neurons. This is the connective
> tissue for LLM agents — not a framework you submit to, a small library you
> build on.

## Why glia?

The 2026 agent-framework field is crowded, and the loudest complaint about the
incumbents is the same: **too much abstraction, hidden control flow, painful to
debug.** glia is the opposite bet. It ships the modern techniques — tools,
structured outputs, streaming, parallel tool execution, context compaction,
durable checkpoints, guardrails, a human-in-the-loop approval gate, subagents,
and evals-as-tests — as **opt-in primitives you can read**, not a monolith you
must trust.

## Install

```bash
pip install glia-agents               # core — no dependencies
pip install "glia-agents[anthropic]"  # + the Claude provider
```

The distribution is `glia-agents`; the import stays `import glia`.

## 30 seconds

```python
import asyncio
from glia import Agent, tool
from glia.providers import ClaudeLLM

@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return {"Paris": "18°C, cloudy"}.get(city, "unknown")

async def main():
    agent = Agent(ClaudeLLM(), tools=[get_weather], system="Be concise.")
    result = await agent.run("What's the weather in Paris?")
    print(result.output)   # the answer
    print(result.usage)    # what it cost

asyncio.run(main())
```

No API key? Every example runs offline with the deterministic `EchoLLM`
provider — same code, no network.

## Next

- [Getting started](getting-started.md) — install, first agent, streaming, offline testing
- [Guide](guide.md) — every primitive, with code
- [Architecture](ARCHITECTURE.md) — how the whole thing works
- [Strategy](STRATEGY.md) — market analysis and positioning
