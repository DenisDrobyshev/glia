# Changelog

All notable changes to glia are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[Semantic Versioning](https://semver.org/).

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
