# Guide

Everything glia can do, with runnable code. Every feature is off by default and
shows up in the event stream.

## Core concepts

Three objects carry the whole model:

- **`Agent`** — a model, its tools, and the loop connecting them.
- **`Trajectory`** — the complete, JSON-serialisable state of a run: the system
  prompt, the message list, an append-only log of events, and token usage.
- **`Event`** — a record of one thing the agent did (`ModelCall`,
  `ModelResponse`, `ToolCalled`, `ToolReturned`, `ApprovalResolved`, …).

`agent.run(prompt)` runs to completion and returns a `RunResult`.
`agent.run_events(prompt)` is the same loop exposed as an async event stream —
consume it to observe, or break early to steer.

```python
result = await agent.run("hello")
result.output        # the model's final text
result.stop_reason   # why it stopped
result.steps         # how many model turns
result.usage         # accumulated token usage
result.trajectory    # the full inspectable record
```

## Tools

A tool is a plain, typed Python function. The `@tool` decorator reads its type
hints and docstring to build a JSON schema — no schema DSL, no base class.

```python
from typing import Annotated, Literal
from glia import tool

@tool
async def search(
    query: Annotated[str, "What to search for"],
    limit: int = 5,
    sort: Literal["relevance", "date"] = "relevance",
) -> str:
    """Search the knowledge base."""
    ...
```

Sync and async functions both work. A tool that raises is turned into an error
result the model can see and recover from — it never crashes the loop.

## Providers

glia talks to models through one small protocol, `LLM`, with a single
`async generate(request) -> LLMResponse` method. Four adapters ship:

- **`ClaudeLLM`** — Claude via the Anthropic SDK (optional `[anthropic]` extra).
- **`OpenAILLM`** — OpenAI **or any OpenAI-compatible endpoint** (Groq, Together,
  OpenRouter, a local vLLM…) via a `base_url`. **Zero dependencies** — stdlib HTTP.
- **`OllamaLLM`** — local open models (Qwen, DeepSeek, Llama…) via a local
  [Ollama](https://ollama.com) server. **Zero dependencies**. Free and offline.
- **`EchoLLM`** — deterministic, offline, for tests and demos.

```python
from glia import Agent
from glia.providers import OllamaLLM

# after: `ollama pull qwen2.5` (or `ollama pull deepseek-r1`)
agent = Agent(OllamaLLM("qwen2.5"), system="Be concise.")
result = await agent.run("Say hello.")
```

`OllamaLLM(model, host="http://localhost:11434")` supports streaming and tool
calling for models that offer it. Writing your own adapter is ~40 lines — see
[Architecture](ARCHITECTURE.md).

## Streaming

Set `stream=True`. When the provider supports it, text deltas are re-emitted as
`ModelDelta` events; the buffered `ModelResponse` still follows, so nothing
downstream changes. Providers that can't stream fall back to `generate`
automatically.

```python
agent = Agent(ClaudeLLM(), stream=True)
async for event in agent.run_events("Write a haiku."):
    if event.kind == "model_delta":
        print(event.text, end="", flush=True)
```

## Parallel tool execution

When the model requests several tools in one turn, glia runs them concurrently
(`parallel_tools=True`, the default) while keeping the `ToolCalled` /
`ToolReturned` events and results in the original call order.

```python
agent = Agent(llm, tools=[search, fetch], parallel_tools=True)
```

## Human-in-the-loop approval

Gate any tool behind an inspectable verdict. An approval policy is any callable
`(ApprovalRequest) -> Decision` (sync or async; a bare `bool` works too). Denied
calls never execute and return an error result the model can react to.

```python
from glia import Agent
from glia.approval import deny, allow_only, prompt_in_terminal

# Block a specific tool:
agent = Agent(llm, tools=[search, delete_all], approval=deny("delete_all"))

# Or allow only a safe set:
agent = Agent(llm, tools=[...], approval=allow_only("search", "read"))

# Or ask a human on the terminal (reference policy):
agent = Agent(llm, tools=[...], approval=prompt_in_terminal)

# Or your own logic:
def policy(request):
    return request.name != "delete_all"   # allow everything but delete_all
```

Each decision emits `ApprovalRequested` → `ApprovalResolved` events.

## Structured output

Get a typed object back instead of a string to parse. glia forces the model to
call one tool whose schema *is* the shape you want, then reads the validated
arguments — provider-agnostic, works even with `EchoLLM`.

```python
from dataclasses import dataclass
from glia import generate_structured

@dataclass
class Contact:
    name: str
    email: str
    wants_demo: bool

contact = await generate_structured(
    llm, "Extract: Ada (ada@x.io) asked for a demo.", Contact
)
# -> Contact(name='Ada', email='ada@x.io', wants_demo=True)
```

`schema` may be a JSON-schema `dict` (you get a `dict`), a `dataclass` type, or a
Pydantic model (you get an instance). Pydantic is imported only if you use it.

## Context engineering (compaction)

The finite context window is the scarcest resource a long-running agent has. A
`Compactor` decides when the trajectory is too big and how to shrink it; the
agent calls it once per step and emits a `Compacted` event.

```python
from glia import Agent, SummarizingCompactor, TrimmingCompactor

# Fold old turns into an LLM-written summary (provenance-preserving):
agent = Agent(llm, compactor=SummarizingCompactor(max_messages=40, keep_last=12))

# Or just drop the oldest turns (cheap, no model call):
agent = Agent(llm, compactor=TrimmingCompactor(max_messages=40, keep_last=20))
```

Both keep the most recent turns verbatim and never split a tool call from its
results.

## Durable execution (checkpoint & resume)

A run is a JSON document. Save it and resume later — even in a new process.

```python
from glia import Agent, Trajectory
from glia.checkpoint import checkpointer, save, load

traj = Trajectory.new()
agent = Agent(llm, hooks=[checkpointer(traj, "run.json")])   # snapshots each step
await agent.run("start the task", trajectory=traj)

resumed = load("run.json")
await agent.run("continue", trajectory=resumed)              # picks up where it stopped
```

## Guardrails

A guardrail is any `(text) -> None` callable that raises `GuardrailTripped` to
reject. Input guardrails run on each prompt; output guardrails run on the final
answer.

```python
from glia import Agent
from glia.guardrails import max_length, block_pattern, no_secrets

agent = Agent(
    llm,
    input_guardrails=[max_length(4000)],
    output_guardrails=[block_pattern("confidential"), no_secrets()],
)
```

## Subagents

Any agent can be exposed as a tool another agent calls — the whole of
"subagents" in glia.

```python
researcher = Agent(llm, tools=[web_search], name="researcher")
lead = Agent(llm, tools=[researcher.as_tool("research", "Look things up online")])
```

## MCP tools

Use any [Model Context Protocol](https://modelcontextprotocol.io) server's tools
as glia tools. Install the optional extra (`pip install "glia-agents[mcp]"`) and
open a session — the tools are live for the duration of the `async with` block:

```python
from glia import Agent
from glia.integrations.mcp import mcp_stdio_tools

async with mcp_stdio_tools("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]) as tools:
    agent = Agent(llm, tools=tools)
    await agent.run("List the files in /tmp")
```

`mcp_http_tools(url)` connects over Streamable HTTP instead. The adapter itself
(`tools_from_mcp`) is dependency-free, so it's easy to test.

## Evals as tests

Treat evals like unit tests: a suite of prompts plus pytest-style checks, run in
CI, gating deploys.

```python
from glia.evals import Case, evaluate, contains, used_tool, did_not_error

suite = [
    Case("answers", "what is 2+2?", [contains("4"), did_not_error]),
    Case("uses tool", "what is 2+2?", [used_tool("add")]),
]
report = await evaluate(suite, lambda: Agent(llm, tools=[add]))
assert report.ok, report
```

## Observability with hooks

A hook is any callable that receives each `Event` (sync or async). Every event
flows through it — log it, trace it, or drive a UI.

```python
def log(event):
    print(event.kind, event.to_dict())

agent = Agent(llm, hooks=[log])
```
