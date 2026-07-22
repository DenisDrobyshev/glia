from __future__ import annotations

from dataclasses import dataclass

from conftest import run

from glia import Agent, generate_structured, tool
from glia.evals import Case, contains, did_not_error, evaluate, used_tool, within_steps
from glia.providers import EchoLLM, call


@dataclass
class Person:
    name: str
    age: int


def test_structured_output_into_dataclass():
    llm = EchoLLM([call("respond", {"name": "Ada", "age": 36})])
    person = run(generate_structured(llm, "who?", Person))
    assert person == Person(name="Ada", age=36)


def test_structured_output_into_dict():
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    llm = EchoLLM([call("respond", {"ok": True})])
    data = run(generate_structured(llm, "?", schema))
    assert data == {"ok": True}


def test_structured_output_forces_tool_choice():
    llm = EchoLLM([call("respond", {"name": "X", "age": 1})])
    run(generate_structured(llm, "?", Person))
    # The request must have forced the 'respond' tool.
    assert llm.calls[0].tool_choice == "respond"


@tool
async def add(a: int, b: int) -> int:
    """Add."""
    return a + b


def test_evaluate_suite_passes():
    def make_agent() -> Agent:
        return Agent(EchoLLM([call("add", {"a": 2, "b": 2}), "The result is 4."]), tools=[add])

    suite = [
        Case("mentions result", "add 2+2", [contains("4"), did_not_error, within_steps(3)]),
        Case("uses the tool", "add 2+2", [used_tool("add")]),
    ]
    report = run(evaluate(suite, make_agent))
    assert report.ok, str(report)
    assert report.passed == 2


def test_evaluate_reports_failures():
    def make_agent() -> Agent:
        return Agent(EchoLLM(["nope"]))

    suite = [Case("wants tool", "hi", [used_tool("add")])]
    report = run(evaluate(suite, make_agent))
    assert not report.ok
    assert report.failed == 1
    assert "was not called" in report.results[0].failures[0]
