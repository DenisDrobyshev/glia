from __future__ import annotations

from conftest import run

from glia import Agent, tool
from glia.evals import (
    Case,
    contains,
    did_not_error,
    evaluate,
    matches,
    used_tool,
    within_steps,
)
from glia.providers import EchoLLM, call


@tool
async def ok(x: str) -> str:
    """ok."""
    return x


@tool
async def broken() -> str:
    """always fails."""
    raise RuntimeError("nope")


def test_matches_check():
    check = matches(r"\d+")
    check(_result("answer is 42"))  # passes
    import pytest

    with pytest.raises(AssertionError):
        check(_result("no digits here"))


def test_within_steps_check_fails_when_exceeded():
    import pytest

    result = run(Agent(EchoLLM([call("ok", {"x": "a"}), "done"]), tools=[ok]).run("go"))
    within_steps(5)(result)  # passes
    with pytest.raises(AssertionError):
        within_steps(1)(result)


def test_did_not_error_detects_tool_error():
    import pytest

    result = run(Agent(EchoLLM([call("broken", {}), "recovered"]), tools=[broken]).run("go"))
    with pytest.raises(AssertionError):
        did_not_error(result)


def test_report_str_and_failure_accounting():
    def make_pass():
        return Agent(EchoLLM(["hello world"]))

    suite = [
        Case("good", "hi", [contains("hello")]),
        Case("bad", "hi", [contains("absent")]),
    ]
    report = run(evaluate(suite, make_pass))
    assert report.passed == 1 and report.failed == 1
    text = str(report)
    assert "PASS" in text and "FAIL" in text


def test_evaluate_records_a_crashing_run_as_failure():
    class Boom:
        async def generate(self, request):
            raise RuntimeError("kaboom")

    report = run(evaluate([Case("c", "x", [used_tool("ok")])], lambda: Agent(Boom())))
    assert not report.ok
    assert "kaboom" in report.results[0].failures[0]


# -- helper --------------------------------------------------------------------


def _result(text: str):
    return run(Agent(EchoLLM([text])).run("q"))
