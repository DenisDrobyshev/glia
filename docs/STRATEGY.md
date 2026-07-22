# glia — Market Analysis & Strategy

_Last updated: July 2026._

This document explains why glia exists, who it competes with, and how it wins a
defensible slice of a crowded market. It is deliberately honest about where the
field already has strong answers — glia does not try to beat LangGraph at being
LangGraph.

## 1. The market in one paragraph

Agent frameworks are past the hype phase and into consolidation. Six libraries
dominate production deployments — LangGraph, CrewAI, the OpenAI Agents SDK, the
Claude Agent SDK, Google ADK, and the merged Microsoft Agent Framework — and the
lanes are largely claimed. What is _not_ settled is developer sentiment: the most
consistent complaint across every 2026 "field guide" is that these frameworks
over-abstract, hide control flow, and are painful to debug. That complaint is the
opening glia is built for.

## 2. Competitive landscape

| Framework | Lane it owns | Traction (mid-2026) | Weakness glia targets |
|---|---|---|---|
| **LangGraph** | Controllable, stateful, production orchestration (graph model) | ~34k★, ~38M downloads/mo, v1.0 GA | Heavy concepts; you debug the framework, not your logic |
| **CrewAI** | Role-based multi-agent "teams" | ~44.6k★, millions of runs/mo | Ergonomic but opaque; hard to see/what/why at runtime |
| **OpenAI Agents SDK** | Fast path inside the OpenAI ecosystem | ~19k★ | Vendor-centric; thin on portability |
| **Claude Agent SDK** | Batteries-included coding/filesystem agent on Claude | Anthropic-backed | A full harness, not a small library to build _on_ |
| **PydanticAI** | Typed Python outputs | V2 stable | Great at types; not positioned as a minimal glass-box loop |
| **Microsoft Agent Framework** | Enterprise (Semantic Kernel + AutoGen merged) | 1.0 (Apr 2026) | Enterprise weight and surface area |
| **Mastra / Vercel AI SDK** | TypeScript agents | Growing fast | Different runtime; not the Python/ML audience |

**Read of the field:** the "controllable production orchestration" slot
(LangGraph) and the "role-based team" slot (CrewAI) are taken and defended.
Competing head-on there is a losing game for a new entrant.

## 3. The gap

Three findings from the 2026 literature point at the same unmet need:

1. **Over-abstraction is the #1 complaint.** Developers report debugging the
   framework's internals instead of their own logic; many strip the framework
   out and call the model API directly to cut latency and regain visibility.
2. **The modern techniques are known, but scattered.** Context engineering
   (compaction, trimming), structured outputs, durable execution
   (checkpoint/replay), guardrails, subagents, and evals-as-tests are the 2026
   toolkit — but they arrive either baked into a monolith or as separate
   products to integrate.
3. **"Read the source" is a real buying criterion now.** After two years of API
   churn, teams increasingly prefer small libraries they can fully understand
   over large frameworks they must trust.

**The gap:** a _small, transparent, fully-debuggable_ library that ships the
modern techniques as **opt-in primitives** rather than a framework you submit to.

## 4. Positioning

> **glia is the glass-box agent library.** Every model call, tool call, and state
> transition is a plain object you can log, snapshot, and replay. No hidden
> control flow. The whole loop fits in one file you can read in an afternoon.

Analogy: glia aims to be the **`httpx`/`FastAPI` of agents** — small, typed,
composable, boring in the best way — not the Django. It competes on
_understandability and control_, not on feature count.

### What glia deliberately is NOT
- Not a graph engine (use LangGraph if you want explicit graph orchestration).
- Not a hosted runtime (use the Claude Agent SDK / Managed Agents for that).
- Not a role-play team abstraction (use CrewAI).
- Not a kitchen sink. The core has **zero required dependencies**.

## 5. Differentiators (the proof points)

1. **Transparency.** The agent loop is ~200 readable lines; the entire run state
   is one serialisable `Trajectory`. Everything emits an inspectable `Event`.
2. **Provider-agnostic, Claude-first.** One ~40-line `LLM` protocol; a Claude
   adapter and a deterministic offline adapter both satisfy it. No vendor lock.
3. **Modern techniques as primitives, not a monolith:** tools from typed
   functions, structured outputs, `SummarizingCompactor` / `TrimmingCompactor`
   (context engineering), checkpoint/resume (durable execution), guardrails,
   subagents-as-tools, and an evals-as-tests harness.
4. **Testable by construction.** The `EchoLLM` provider makes the full loop
   run deterministically in CI with no API key and no flakiness — a property
   most frameworks lack.
5. **Typed and dependency-light.** Ships `py.typed`; core installs nothing but
   the standard library.

## 6. Target users

- **Backend/ML engineers** who want to build an agent without adopting a
  framework's worldview.
- **Teams burned by abstraction churn** who value auditability and control.
- **Educators and learners** — the codebase doubles as a readable reference for
  how a modern agent loop actually works.

## 7. Go-to-market (open source)

1. **Lead with the anti-framework message.** README hero: "You can read the
   whole thing." Show the loop. Show the event stream. Show the offline test.
2. **Ship credibility signals:** green CI, typed, examples that run offline,
   an evals suite, a clear architecture doc.
3. **Content wedge:** short posts that reproduce the exact pain points
   ("debugging an agent when you can see every step") and a side-by-side with a
   heavier framework.
4. **Adjacent-tool interop, not lock-in:** MCP-compatible tool interface;
   document how glia sits _next to_ LangGraph/CrewAI rather than replacing them.

## 8. Risks & honest caveats

- **Crowded market, discovery is hard.** Mitigation: a sharp, single-sentence
  wedge and a codebase small enough to evaluate in minutes.
- **"Minimal" can read as "toy."** Mitigation: production-relevant primitives
  (durable execution, evals, guardrails) present from v0.1, with tests.
- **Provider drift.** Mitigation: the provider boundary is one tiny protocol;
  adapting to model/API changes touches ~1 file.

## 9. Roadmap headline

v0.1 (this release) proves the thesis end-to-end. Next: parallel tool
execution, streaming token output, an MCP tool bridge, a tracing exporter
(OpenTelemetry), and a TypeScript port once the Python core stabilises. Full
detail in [ROADMAP.md](ROADMAP.md).

## Sources

Market research conducted July 2026:

- [Comparing Open-Source AI Agent Frameworks — Langfuse](https://langfuse.com/blog/2025-03-19-ai-agent-comparison)
- [Best Open-Source AI Agent Frameworks 2026 — AI Magicx](https://www.aimagicx.com/blog/best-open-source-ai-agent-frameworks-2026)
- [Best AI Agent SDKs Compared 2026 — Requesty](https://www.requesty.ai/blog/best-ai-agent-sdks-compared-2026-langchain-crewai-openai-anthropic-google)
- [Top Agentic Frameworks 2026 — JetBrains](https://blog.jetbrains.com/pycharm/2026/06/top-agentic-frameworks-for-building-applications-2026/)
- [Best open-source frameworks for building AI agents 2026 — Firecrawl](https://www.firecrawl.dev/blog/best-open-source-agent-frameworks)
- [Why Developers Say LangChain Is "Bad" — Designveloper](https://www.designveloper.com/blog/is-langchain-bad/)
- [AI Agent Frameworks 2026: Only 2 Production-Ready — fp8.co](https://fp8.co/articles/AI-Agent-Frameworks-Complete-Guide-2026)
- [Effective context engineering for AI agents — Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Shipping AI Agents to Production: 2026 Context Engineering Recipe Book — Pento](https://www.pento.ai/blog/shipping-ai-agents-to-production-recipe-book)
- [State of AI Agent Memory 2026 — mem0](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
