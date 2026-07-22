# Roadmap

glia follows one principle when adding features: **a new primitive must not add
hidden control flow.** If a feature can't be made inspectable, it doesn't ship.

## v0.1 — thesis proven (this release)
- Transparent agent loop (`run` / `run_events`) with a full event stream
- Tools from typed functions (`@tool`) + `ToolRegistry`
- Provider boundary + `ClaudeLLM` (Anthropic) and `EchoLLM` (offline)
- Serialisable `Trajectory`; checkpoint/resume (durable execution)
- Context engineering: `SummarizingCompactor`, `TrimmingCompactor`
- Guardrails (input/output validators)
- Structured outputs (provider-agnostic, via forced tool call)
- Subagents (any agent → a tool)
- Evals-as-tests harness
- Zero-dependency core, `py.typed`, green CI, runnable offline examples

## v0.2 — ergonomics & throughput
- [ ] Parallel tool execution (`asyncio.gather`) with preserved event ordering
- [ ] Streaming token output through the event stream (`ModelDelta` events)
- [ ] Retry/backoff policy as an explicit, inspectable wrapper (not hidden magic)
- [ ] `RunResult` niceties: per-tool timings, cost summary helpers

## v0.3 — interop
- [ ] MCP tool bridge: expose MCP servers as glia tools, and glia tools as MCP
- [ ] OpenTelemetry span exporter driven off the event stream
- [ ] Prompt-caching hints on the Claude adapter (stable-prefix breakpoints)

## v0.4 — reliability
- [ ] Pluggable persistence backends for checkpoints (file, sqlite, redis)
- [ ] Deterministic replay: re-run a saved trajectory against a recorded provider
- [ ] Human-in-the-loop tool approval as a first-class, inspectable gate

## Later
- [ ] TypeScript port once the Python core stabilises (same glass-box contract)
- [ ] A tiny local trace viewer that renders a `Trajectory` as a timeline

## Non-goals (on purpose)
- A graph/DSL orchestration engine — use LangGraph.
- A hosted runtime/sandbox — use the Claude Agent SDK / Managed Agents.
- A role-play "crew" abstraction — use CrewAI.
- Anything that requires the core to grow a heavy dependency.
