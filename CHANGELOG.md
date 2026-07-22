# Changelog

All notable changes to glia are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[Semantic Versioning](https://semver.org/).

## [0.4.0] — 2026-07-23

### Added
- **`OllamaLLM` provider** — run local open models (Qwen, DeepSeek, Llama, …)
  via a local [Ollama](https://ollama.com) server. Free, offline, no API key.
  - **Zero dependencies:** talks to `/api/chat` over stdlib HTTP (`urllib`),
    with async streaming via a background reader thread.
  - Supports streaming and tool calling for models that offer it; maps glia
    messages/tool-results to Ollama's format and back.
  - Exported as `glia.providers.OllamaLLM` (eager import — no vendor SDK).
- **Ollama mode in the desktop shell** — a third provider option in Settings
  (host + model, with a datalist of common models), so the shell can drive a
  local model. New config fields `ollama_host` / `ollama_model`.
- Example `08_ollama_local.py`; Ollama documented in the Guide and Desktop app
  pages (EN + RU).

## [0.3.0] — 2026-07-23

### Added
- **Desktop shell (`glia-shell`)** — a downloadable graphical chat app with a
  live glass-box event panel. It runs a real `glia.Agent` and streams every
  event (model calls, tool calls, approvals) to a self-contained single-page UI.
  - Pure-stdlib local server (`http.server` + Server-Sent Events); the only
    optional dependency is `pywebview` for the native window (`[shell]` extra),
    with an automatic browser fallback.
  - Works offline in demo mode (EchoLLM) out of the box; add an Anthropic API
    key in Settings — stored locally, never returned to the UI — for real Claude.
  - New console entry point `glia-shell` and `python -m glia.shell`.
  - CI workflow builds standalone Windows/macOS/Linux binaries and attaches them
    to each release.
- Docs: a "Desktop app" page (EN + RU).

## [0.2.1] — 2026-07-22

### Changed
- `_stringify` (tool-result rendering) now JSON-encodes only containers and uses
  `str()` for everything else — a plain object no longer comes back wrapped in
  quotes.

### Docs & tests
- Test suite expanded to **110 offline tests, ~98% coverage** (adds full offline
  coverage of the Claude adapter via a fake client, plus types, llm/echo,
  approval, structured, evals, memory, tools, and trajectory edge cases). CI now
  enforces a coverage floor.
- **Bilingual (EN/RU) documentation site** (MkDocs Material + i18n): home,
  getting started, guide, and architecture in English and Russian, with
  auto-deploy to GitHub Pages. Added `README.ru.md`.

## [0.2.0] — 2026-07-22

Refinements that deepen the glass-box thesis. Every new capability is visible in
the event stream and off by default.

### Added
- **Streaming output.** `Agent(..., stream=True)` re-emits provider text deltas
  as `ModelDelta` events; the buffered `ModelResponse` still follows. New
  `StreamChunk` type and optional `StreamingLLM` protocol; both `ClaudeLLM`
  (via `messages.stream`) and `EchoLLM` (deterministic word chunks) implement
  it. Providers without streaming fall back to `generate` automatically.
- **Parallel tool execution.** A turn's tool calls run concurrently
  (`asyncio.gather`) by default (`parallel_tools=True`), with `ToolCalled` /
  `ToolReturned` events kept in the original call order.
- **Human-in-the-loop approval.** `approval=<policy>` gates every tool call
  behind an inspectable verdict (`ApprovalRequested` / `ApprovalResolved`
  events). Denied calls never execute and return an error result the model can
  react to. Built-in policies: `approve_all`, `deny_all`, `allow_only`, `deny`,
  and a reference interactive `prompt_in_terminal`.

## [0.1.0] — 2026-07-22

Initial release. Proves the glass-box thesis end-to-end.

### Added
- Transparent agent loop: `Agent.run()` and `Agent.run_events()` with a full
  event stream (`RunStarted`, `ModelCall`, `ModelResponse`, `ToolCalled`,
  `ToolReturned`, `Compacted`, `RunFinished`).
- `@tool` decorator deriving JSON schemas from type hints (scalars, `Literal`,
  `Optional`/PEP 604 unions, `list`/`dict`, `Annotated` descriptions).
- `LLM` provider protocol with two adapters: `ClaudeLLM` (Anthropic SDK,
  optional dependency) and `EchoLLM` (deterministic, offline, for tests/CI).
- Serialisable `Trajectory` with checkpoint/resume (`glia.checkpoint`).
- Context engineering: `SummarizingCompactor` and `TrimmingCompactor`.
- Input/output guardrails (`glia.guardrails`).
- Provider-agnostic structured outputs (`generate_structured`).
- Subagents via `Agent.as_tool(...)`.
- Evals-as-tests harness (`glia.evals`).
- Zero-dependency core, `py.typed`, six runnable offline examples, 26 tests,
  green CI.
