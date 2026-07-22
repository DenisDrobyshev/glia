"""glia — a glass-box, minimal library for building LLM agents.

Design in one sentence: everything the agent does is a plain, inspectable object
you can log, snapshot, and replay — no hidden control flow, no magic. Modern
techniques (tools, structured outputs, context compaction, durable checkpoints,
guardrails, subagents, evals-as-tests) ship as opt-in primitives, not a monolith.

Quick start::

    from glia import Agent, tool
    from glia.providers import ClaudeLLM

    @tool
    async def add(a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b

    agent = Agent(ClaudeLLM(), tools=[add], system="You are precise.")
    result = await agent.run("What is 2 + 2?")
    print(result.output)

Read the source — that's the point. The whole loop is in ``agent.py`` and the
whole state is in ``trajectory.py``.
"""

from __future__ import annotations

from .agent import Agent, Hook, RunResult
from .errors import (
    GliaError,
    GuardrailTripped,
    MaxStepsExceeded,
    ProviderError,
    StructuredOutputError,
    ToolError,
)
from .llm import LLM, LLMRequest, LLMResponse, ToolSchema
from .memory import Compactor, SummarizingCompactor, TrimmingCompactor
from .structured import generate_structured
from .tools import Tool, ToolRegistry, tool
from .trajectory import (
    Compacted,
    Event,
    ModelCall,
    ModelResponse,
    RunFinished,
    RunStarted,
    ToolCalled,
    ToolReturned,
    Trajectory,
)
from .types import Message, Text, Thinking, ToolResult, ToolUse, Usage, assistant, user

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # core
    "Agent",
    "RunResult",
    "Hook",
    "tool",
    "Tool",
    "ToolRegistry",
    # llm boundary
    "LLM",
    "LLMRequest",
    "LLMResponse",
    "ToolSchema",
    # trajectory + events
    "Trajectory",
    "Event",
    "RunStarted",
    "ModelCall",
    "ModelResponse",
    "ToolCalled",
    "ToolReturned",
    "Compacted",
    "RunFinished",
    # types
    "Message",
    "Text",
    "Thinking",
    "ToolUse",
    "ToolResult",
    "Usage",
    "user",
    "assistant",
    # context engineering
    "Compactor",
    "SummarizingCompactor",
    "TrimmingCompactor",
    # structured output
    "generate_structured",
    # errors
    "GliaError",
    "ToolError",
    "GuardrailTripped",
    "MaxStepsExceeded",
    "ProviderError",
    "StructuredOutputError",
]
