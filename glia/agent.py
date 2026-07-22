"""The agent: a transparent loop over a model and its tools.

There is no hidden control flow here. ``run()`` is a plain ``for`` loop: call the
model, and if it asked for tools, run them, append the results, and loop again;
otherwise stop. Every iteration emits events into the trajectory and to your
hooks, so you can watch, log, snapshot, and replay the entire thing.

The modern-technique primitives — guardrails, compaction, subagents — plug in as
ordinary constructor arguments. Nothing is on unless you turn it on.

    from glia import Agent
    from glia.providers import ClaudeLLM

    agent = Agent(ClaudeLLM(), tools=[get_weather], system="Be concise.")
    result = await agent.run("What's the weather in Paris?")
    print(result.output)          # the answer
    print(result.trajectory.usage)  # what it cost
"""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .errors import MaxStepsExceeded
from .guardrails import Guardrail, run_guardrails
from .llm import LLM, LLMRequest
from .memory import Compactor
from .tools import Tool, ToolRegistry
from .tools import tool as _tool
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
from .types import Usage

Hook = Callable[[Event], Any]
"""A hook observes events. Sync or async; return value is ignored."""


@dataclass(slots=True)
class RunResult:
    """The outcome of a run. The trajectory is the full record; ``output`` is
    the model's final text."""

    trajectory: Trajectory
    output: str
    stop_reason: str
    steps: int

    @property
    def usage(self) -> Usage:
        return self.trajectory.usage


class Agent:
    """A model, its tools, and the loop that connects them."""

    def __init__(
        self,
        llm: LLM,
        *,
        tools: list[Any] | ToolRegistry | None = None,
        system: str | None = None,
        input_guardrails: list[Guardrail] | None = None,
        output_guardrails: list[Guardrail] | None = None,
        compactor: Compactor | None = None,
        hooks: list[Hook] | None = None,
        max_steps: int = 12,
        max_tokens: int = 4096,
        temperature: float | None = None,
        thinking: bool = False,
        tool_choice: str | None = None,
        name: str = "agent",
    ) -> None:
        self.llm = llm
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.system = system
        self.input_guardrails = list(input_guardrails or [])
        self.output_guardrails = list(output_guardrails or [])
        self.compactor = compactor
        self.hooks = list(hooks or [])
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking = thinking
        self.tool_choice = tool_choice
        self.name = name

    # -- running ---------------------------------------------------------------

    async def run(self, prompt: str | None = None, *, trajectory: Trajectory | None = None) -> RunResult:
        """Run to completion and return the result. Pass an existing
        ``trajectory`` to resume a checkpointed run."""
        traj = trajectory or Trajectory.new(system=self.system)
        finished: RunFinished | None = None
        async for event in self.run_events(prompt, trajectory=traj):
            if isinstance(event, RunFinished):
                finished = event
        assert finished is not None  # run_events always ends with RunFinished
        return RunResult(
            trajectory=traj,
            output=finished.output,
            stop_reason=finished.stop_reason,
            steps=finished.step,
        )

    async def run_events(
        self, prompt: str | None = None, *, trajectory: Trajectory | None = None
    ) -> AsyncIterator[Event]:
        """The loop, as an event stream. Consume it to observe every model call,
        tool call, and compaction as it happens — or break early to steer."""
        traj = trajectory or Trajectory.new(system=self.system)

        if prompt is not None:
            run_guardrails(self.input_guardrails, prompt)
            traj.add_user(prompt)

        async for event in self._emit(traj, RunStarted(prompt=prompt)):
            yield event

        output = ""
        stop_reason = "end_turn"
        step = 0

        for step in range(1, self.max_steps + 1):
            # Context engineering happens before the call, so the request we
            # build reflects the compacted state.
            if self.compactor and self.compactor.should_compact(traj):
                before = len(traj.messages)
                summary = await self.compactor.compact(traj, self.llm)
                if summary:
                    async for event in self._emit(
                        traj, Compacted(freed_messages=before - len(traj.messages), summary=summary)
                    ):
                        yield event

            request = LLMRequest(
                messages=traj.messages,
                system=traj.system,
                tools=self.tools.schemas(),
                tool_choice=self.tool_choice,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                thinking=self.thinking,
            )
            async for event in self._emit(
                traj, ModelCall(step=step, message_count=len(traj.messages), tool_count=len(self.tools))
            ):
                yield event

            response = await self.llm.generate(request)
            traj.add_message(response.message)
            traj.add_usage(response.usage)
            async for event in self._emit(
                traj,
                ModelResponse(
                    step=step,
                    text=response.message.text(),
                    stop_reason=response.stop_reason,
                    tool_uses=tuple(t.name for t in response.message.tool_uses()),
                    usage=response.usage,
                ),
            ):
                yield event

            tool_uses = response.message.tool_uses()
            if not tool_uses:
                output = response.message.text()
                stop_reason = response.stop_reason
                run_guardrails(self.output_guardrails, output)
                break

            # Execute each requested tool, then hand all results back in one
            # user turn (the provider convention). Sequential and readable;
            # swap in asyncio.gather here if you need parallelism.
            results = []
            for call in tool_uses:
                async for event in self._emit(
                    traj, ToolCalled(step=step, tool_use_id=call.id, name=call.name, arguments=call.input)
                ):
                    yield event
                started = time.perf_counter()
                result = await self.tools.invoke(call.id, call.name, call.input)
                async for event in self._emit(
                    traj,
                    ToolReturned(
                        step=step,
                        tool_use_id=call.id,
                        name=call.name,
                        content=result.content,
                        is_error=result.is_error,
                        duration_s=time.perf_counter() - started,
                    ),
                ):
                    yield event
                results.append(result)
            traj.add_tool_results(results)
        else:
            raise MaxStepsExceeded(self.max_steps)

        async for event in self._emit(
            traj, RunFinished(step=step, output=output, stop_reason=stop_reason)
        ):
            yield event

    # -- composition -----------------------------------------------------------

    def as_tool(self, name: str | None = None, description: str | None = None) -> Callable[..., Awaitable[str]]:
        """Expose this agent as a tool another agent can call — the whole of
        "subagents" in glia. The child runs on a fresh trajectory and returns
        its final text.

            researcher = Agent(llm, tools=[web_search], name="researcher")
            lead = Agent(llm, tools=[researcher.as_tool("research", "Look things up online")])
        """
        child = self

        async def subagent(request: str) -> str:
            result = await child.run(request)
            return result.output

        subagent.__name__ = name or f"{self.name}_agent"
        subagent.__doc__ = description or f"Delegate a sub-task to the {self.name} sub-agent."
        return _tool(subagent)

    # -- internals -------------------------------------------------------------

    async def _emit(self, trajectory: Trajectory, event: Event) -> AsyncIterator[Event]:
        """Record an event, notify hooks, and yield it to the stream."""
        trajectory.record(event)
        for hook in self.hooks:
            result = hook(event)
            if inspect.isawaitable(result):
                await result
        yield event


__all__ = ["Agent", "RunResult", "Hook", "Tool", "field"]

_ = field  # re-export convenience for dataclass-based hooks
