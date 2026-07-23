"""Evals as tests: a tiny regression harness for agent behaviour.

The 2026 consensus is to treat evals like unit tests — a suite of representative
trajectories with assertions, run in CI, gating deploys. glia keeps that literal:
a :class:`Case` is a prompt plus a list of pytest-style checks (each raises
``AssertionError`` on failure), and :func:`evaluate` runs them against a fresh
agent per case and returns a :class:`Report`.

    suite = [
        Case("greets", "say hello", [contains("hello")]),
        Case("uses tool", "weather in Paris?", [used_tool("get_weather")]),
    ]
    report = await evaluate(suite, lambda: Agent(make_llm(), tools=[get_weather]))
    assert report.ok, report
"""

from __future__ import annotations

import re
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field

from .agent import Agent, RunResult

Check = Callable[[RunResult], None]
"""A check raises ``AssertionError`` to fail, or returns ``None`` to pass."""

AgentFactory = Callable[[], Agent]


@dataclass(slots=True)
class Case:
    """One eval: a prompt and the checks its result must satisfy."""

    name: str
    prompt: str
    checks: list[Check] = field(default_factory=list)


@dataclass(slots=True)
class CaseResult:
    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    result: RunResult | None = None


@dataclass(slots=True)
class Report:
    """The outcome of a suite. ``ok`` is true only if every case passed."""

    results: list[CaseResult]

    @property
    def ok(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    def __str__(self) -> str:
        lines = [f"eval report: {self.passed}/{len(self.results)} passed"]
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{mark}] {r.name}")
            for failure in r.failures:
                lines.append(f"         - {failure}")
        return "\n".join(lines)


async def evaluate(cases: list[Case], make_agent: AgentFactory | Agent) -> Report:
    """Run every case against a fresh agent and collect the results."""
    results: list[CaseResult] = []
    for case in cases:
        agent = make_agent() if callable(make_agent) else make_agent
        failures: list[str] = []
        result: RunResult | None = None
        try:
            result = await agent.run(case.prompt)
        except Exception as exc:  # noqa: BLE001 - a crashing run is a failing case
            failures.append(f"run raised {type(exc).__name__}: {exc}")
        else:
            for check in case.checks:
                try:
                    check(result)
                except AssertionError as exc:
                    failures.append(str(exc) or "assertion failed")
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"check errored: {exc}\n{traceback.format_exc(limit=1)}")
        results.append(CaseResult(name=case.name, passed=not failures, failures=failures, result=result))
    return Report(results=results)


# -- reusable checks -----------------------------------------------------------


def contains(substring: str, *, ignore_case: bool = True) -> Check:
    def check(result: RunResult) -> None:
        haystack = result.output.lower() if ignore_case else result.output
        needle = substring.lower() if ignore_case else substring
        assert needle in haystack, f"output missing {substring!r}: {result.output!r}"

    return check


def matches(pattern: str) -> Check:
    compiled = re.compile(pattern)

    def check(result: RunResult) -> None:
        assert compiled.search(result.output), f"output does not match /{pattern}/: {result.output!r}"

    return check


def used_tool(name: str) -> Check:
    def check(result: RunResult) -> None:
        called = {e.name for e in result.trajectory.events_of("tool_called")}  # type: ignore[attr-defined]
        assert name in called, f"tool {name!r} was not called (called: {sorted(called) or 'none'})"

    return check


def did_not_error(result: RunResult) -> None:
    """A check (not a factory): no tool returned an error."""
    errored = [
        getattr(e, "name", "?") for e in result.trajectory.events_of("tool_returned")
        if getattr(e, "is_error", False)
    ]
    assert not errored, f"tools errored: {errored}"


def within_steps(limit: int) -> Check:
    def check(result: RunResult) -> None:
        assert result.steps <= limit, f"took {result.steps} steps, expected <= {limit}"

    return check
