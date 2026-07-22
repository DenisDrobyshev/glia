# Architecture

glia has one rule: **no hidden control flow**. If you want to know what the agent
does, you read `agent.py`. If you want to know what state it holds, you read
`trajectory.py`. That's the whole system.

## The pieces

```
          ┌─────────────────────────────────────────────┐
          │                  Agent                       │
          │   the loop: call model → run tools → repeat  │
          └───────┬───────────────┬──────────────┬───────┘
                  │               │              │
          ┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼───────┐
          │     LLM      │ │ ToolRegistry│ │  Trajectory │
          │  (protocol)  │ │  (@tool fns)│ │ state+events│
          └───────┬──────┘ └─────────────┘ └─────────────┘
                  │
        ┌─────────┴─────────┐
   ┌────▼─────┐        ┌─────▼──────┐
   │ ClaudeLLM│        │  EchoLLM   │
   │ (Anthropic)        │(offline/CI)│
   └──────────┘        └────────────┘
```

| Module | Responsibility | Lines you'd read to understand it |
|---|---|---|
| `types.py` | Content blocks, `Message`, `Usage`. Immutable, JSON-serialisable. | ~170 |
| `llm.py` | The provider boundary: `LLMRequest`, `LLMResponse`, and the `LLM` protocol. | ~90 |
| `tools.py` | `@tool` decorator (schema from type hints), `ToolRegistry`. | ~230 |
| `trajectory.py` | The run state + the `Event` types. The glass box itself. | ~250 |
| `agent.py` | The loop. `run()` and `run_events()`. | ~230 |
| `memory.py` | Context engineering: compactors. | ~140 |
| `guardrails.py` | Input/output validators. | ~90 |
| `structured.py` | Structured output via a forced tool call. | ~110 |
| `checkpoint.py` | Save/load a trajectory; a checkpointer hook. | ~70 |
| `evals.py` | Evals-as-tests harness. | ~140 |
| `providers/` | `EchoLLM` (offline) and `ClaudeLLM` (Anthropic). | ~120 each |

## The loop, precisely

`Agent.run_events()` is the single source of behaviour. Each iteration:

1. **(optional) compact** — if a `Compactor` says the trajectory is too big,
   shrink it first, emitting a `Compacted` event.
2. **build an `LLMRequest`** from the current trajectory + tool schemas.
3. **call the provider** (`ModelCall` → `ModelResponse`). When `stream=True` and
   the provider supports it, text deltas are re-emitted as `ModelDelta` events
   first. The assistant message and its `Usage` are appended to the trajectory.
4. **if the model asked for tools:** announce all calls (`ToolCalled`), optionally
   run each past the `approval` gate (`ApprovalRequested` → `ApprovalResolved`),
   execute the approved ones — concurrently by default (`asyncio.gather`) —
   emit `ToolReturned` in call order, append all results as one user turn, loop.
5. **otherwise:** run output guardrails and finish (`RunFinished`).

`run()` is just `run_events()` drained to completion. That's the whole thing.

## Why these choices

- **Blocks are a closed union, not a class hierarchy.** You `isinstance`/match on
  four types; there are no subclasses to discover or plugins to register.
- **Tool results are a `user` turn.** That's the Anthropic convention, and it
  keeps our message model and the wire format aligned with no translation debt.
- **Events are records, not commands.** Hooks observe; they never mutate the
  loop. This keeps behaviour readable and hooks safe.
- **The provider boundary is tiny on purpose.** Adapting to a new model or an
  API change touches one file. Vendor churn can't ripple through the codebase.
- **`EchoLLM` is a first-class citizen.** Deterministic, offline testing of the
  entire loop is a design goal, not an afterthought.

## Extending it

- **New provider:** implement `async def generate(request) -> LLMResponse`. ~40
  lines. See `providers/echo.py` for the simplest possible example.
- **New context strategy:** implement the `Compactor` protocol
  (`should_compact` + `compact`). See `memory.py`.
- **New guardrail:** write a `(text) -> None` function that raises
  `GuardrailTripped`. That's the whole interface.
- **Observability:** add a hook. Every event flows through it; export to logs,
  a tracer, or a UI.
