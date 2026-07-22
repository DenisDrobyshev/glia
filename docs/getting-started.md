# Getting started

## Install

```bash
pip install "glia-agents[anthropic]"   # core + Claude provider
```

For local development from a clone:

```bash
git clone https://github.com/DenisDrobyshev/glia
cd glia
pip install -e ".[anthropic,dev]"
```

The distribution name is `glia-agents`; the import is always `import glia`.

## Your first agent

```python
import asyncio
from glia import Agent, tool
from glia.providers import ClaudeLLM

@tool
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

async def main():
    agent = Agent(ClaudeLLM(), tools=[add], system="You are precise.")
    result = await agent.run("What is 21 + 21?")
    print(result.output)

asyncio.run(main())
```

`ClaudeLLM` defaults to the `claude-opus-4-8` model and resolves credentials the
same way the Anthropic SDK does (`ANTHROPIC_API_KEY`, or an `ant auth login`
profile).

## Run it offline — no API key

glia ships a deterministic provider, `EchoLLM`, that replays a scripted sequence
of turns. It satisfies the same `LLM` protocol as `ClaudeLLM`, so your agent code
never changes:

```python
from glia.providers import EchoLLM, call

# Turn 1: the model calls the tool. Turn 2: it answers.
llm = EchoLLM([call("add", {"a": 21, "b": 21}), "The answer is 42."])
agent = Agent(llm, tools=[add])
result = await agent.run("What is 21 + 21?")
assert result.output == "The answer is 42."
```

This is what makes glia testable: the entire loop — tools, subagents,
compaction, approval — runs in CI with no cost and no flakiness.

## Watch every step

The agent emits an event for everything it does. Consume the event stream to see
the glass box working:

```python
async for event in agent.run_events("What is 21 + 21?"):
    print(event.kind)
# run_started → model_call → model_response → tool_called → tool_returned → ... → run_finished
```

## Turn on streaming

```python
agent = Agent(ClaudeLLM(), tools=[add], stream=True)
async for event in agent.run_events("Explain addition briefly."):
    if event.kind == "model_delta":
        print(event.text, end="", flush=True)   # tokens as they arrive
```

## Where to next

- The [Guide](guide.md) covers every primitive with examples.
- The `examples/` folder in the repo has seven runnable scripts (all offline).
