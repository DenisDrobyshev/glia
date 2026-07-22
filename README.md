# glia

**English** · [Русский](README.ru.md) · [📖 Docs](https://denisdrobyshev.github.io/glia/)

**A glass-box, minimal library for building LLM agents.** Every model call, tool
call, and state transition is a plain object you can log, snapshot, and replay.
No hidden control flow. The whole loop fits in one file you can read in an
afternoon.

> _glia_ (n.): the cells that support and connect neurons. This is the connective
> tissue for LLM agents — not a framework you submit to, a small library you
> build on.

[![CI](https://github.com/DenisDrobyshev/glia/actions/workflows/ci.yml/badge.svg)](https://github.com/DenisDrobyshev/glia/actions/workflows/ci.yml)
&nbsp;Python 3.10+ &nbsp;·&nbsp; MIT &nbsp;·&nbsp; zero required dependencies &nbsp;·&nbsp; typed

---

## Why another agent library?

The 2026 agent-framework field is crowded, and the loudest, most consistent
complaint about the incumbents is the same: **too much abstraction, hidden
control flow, painful to debug.** Developers keep stripping the framework out to
call the model API directly, just to see what's happening.

glia is the opposite bet. It ships the modern techniques — tools, structured
outputs, context compaction, durable checkpoints, guardrails, subagents,
evals-as-tests — as **opt-in primitives you can read**, not a monolith you must
trust. The design goal is understandability and control, not feature count.

If you want a graph engine, use [LangGraph](https://github.com/langchain-ai/langgraph).
If you want role-play crews, use [CrewAI](https://github.com/crewAIInc/crewAI).
If you want a small, transparent loop you fully understand — glia.

See [docs/STRATEGY.md](docs/STRATEGY.md) for the full market analysis.

## Install

```bash
pip install glia-agents               # core — no dependencies
pip install "glia-agents[anthropic]"  # + the Claude provider
```

> The distribution is `glia-agents` (the bare name `glia` was taken on PyPI);
> the import stays `import glia`. For development: `git clone` then
> `pip install -e ".[anthropic,dev]"`.

## 30-second tour

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
    print(result.output)          # the answer
    print(result.usage)           # what it cost

asyncio.run(main())
```

No API key handy? Every example runs offline with the deterministic `EchoLLM`
provider — same code, no network:

```python
from glia.providers import EchoLLM, call
llm = EchoLLM([call("get_weather", {"city": "Paris"}), "It's 18°C and cloudy."])
```

## See the whole glass box

Because the loop emits an event for everything it does, you can watch it work:

```python
async for event in agent.run_events("What's the weather in Paris?"):
    print(event.kind)
# run_started → model_call → model_response → tool_called → tool_returned → model_call → ... → run_finished
```

And the entire run state is one serialisable object:

```python
from glia.checkpoint import save, load
save(result.trajectory, "run.json")     # durable execution: it's just JSON
resumed = load("run.json")
await agent.run("follow-up question", trajectory=resumed)   # pick up where you left off
```

## What's in the box

| Primitive | What it gives you |
|---|---|
| **Transparent loop** | `agent.run()` / `agent.run_events()` — no hidden control flow |
| **Streaming** | `Agent(..., stream=True)` — tokens arrive as `ModelDelta` events |
| **Typed tools** | `@tool` on a plain function; JSON schema derived from type hints |
| **Parallel tools** | a turn's tool calls run concurrently, results kept in order |
| **Approval gate** | `approval=...` — an inspectable human-in-the-loop verdict before any tool runs |
| **Provider boundary** | one ~40-line `LLM` protocol; Claude, local (Ollama), and offline adapters |
| **Trajectory** | the full, JSON-serialisable run state and event log |
| **Structured output** | `generate_structured(...)` → a dataclass / Pydantic model / dict |
| **Context engineering** | `SummarizingCompactor`, `TrimmingCompactor` |
| **Durable execution** | checkpoint & resume — a run is a JSON file |
| **Guardrails** | `(text) -> None` validators for input and output |
| **Subagents** | `agent.as_tool(...)` — any agent becomes a tool |
| **Evals-as-tests** | a pytest-style regression harness for agent behaviour |

## Examples

Runnable, offline, no API key needed:

```bash
python examples/01_hello_agent.py       # basic agent + event stream
python examples/02_tools.py             # tools
python examples/03_structured_output.py # typed output
python examples/04_subagents.py         # subagent as a tool
python examples/05_checkpoint_resume.py # durable execution
python examples/06_evals.py             # eval suite
python examples/07_streaming_and_approval.py # streaming + parallel tools + approval gate
python examples/08_ollama_local.py           # run a local model (Qwen/DeepSeek) via Ollama
```

## Desktop app

glia ships a **graphical shell** — a chat window that also shows the live glass
box (streaming tokens, tool calls, approvals) as it happens.

- **Download** a standalone build for your OS from the [latest release](https://github.com/DenisDrobyshev/glia/releases/latest) (`glia-shell-windows.exe` / `-macos` / `-linux`) — no Python needed.
- **Or** `pip install "glia-agents[shell]"` then `glia-shell` for a native window.

It works offline in demo mode out of the box; add an Anthropic API key in
Settings to chat with real Claude. The whole app is stdlib + one HTML file — see
[docs/app.md](docs/app.md).

## Design in one picture

```
Agent.run() → [ call model → (tools? run them, loop) : done ]
                     │              │
                every step emits an Event you can see, log, and replay
                     │
              all state lives in one serialisable Trajectory
```

Full details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project docs

- 📖 **[Documentation site](https://denisdrobyshev.github.io/glia/)** — full guide, available in **English** and **Русский**
- [Strategy & market analysis](docs/STRATEGY.md) — why glia, and who it competes with
- [Architecture](docs/ARCHITECTURE.md) — how the whole thing works
- [Roadmap](docs/ROADMAP.md) — where it's going
- [Contributing](CONTRIBUTING.md)

## Status

**v0.4 — alpha.** The core thesis is proven end-to-end with a full test suite
(129 offline tests) and green CI. v0.4 adds local models via Ollama; v0.3 added a
downloadable desktop shell; v0.2 added streaming, parallel tool execution, and a
human-in-the-loop approval gate. APIs may still change before 1.0. Feedback and
issues welcome.

## License

MIT — see [LICENSE](LICENSE).
