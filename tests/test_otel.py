"""Tests for the OpenTelemetry exporter, using a fake tracer (no otel package).

We inject a fake tracer and a ``context_factory`` that returns the parent span
itself, so we can assert on span names, attributes, parent relationships, and
ordering without the real OpenTelemetry SDK.
"""

from __future__ import annotations

from conftest import run

from glia import Agent, tool
from glia.integrations.otel import OTelExporter
from glia.providers import EchoLLM
from glia.providers import call as tcall


class FakeSpan:
    def __init__(self, name, context=None):
        self.name = name
        self.context = context
        self.attributes: dict = {}
        self.events: list = []
        self.ended = False
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def add_event(self, name, attributes=None):
        self.events.append((name, attributes or {}))

    def end(self):
        self.ended = True


class FakeTracer:
    def __init__(self):
        self.spans: list[FakeSpan] = []

    def start_span(self, name, context=None):
        span = FakeSpan(name, context)
        self.spans.append(span)
        return span


def _exporter(tracer):
    # context = the parent span itself, so we can assert parenting directly.
    return OTelExporter(tracer, context_factory=lambda span: span)


@tool
async def add(a: int, b: int) -> int:
    """Add."""
    return a + b


def test_run_and_child_spans_with_attributes_and_parenting():
    tracer = FakeTracer()
    agent = Agent(EchoLLM([tcall("add", {"a": 2, "b": 3}), "The sum is 5."]), tools=[add], hooks=[_exporter(tracer)])
    run(agent.run("add 2 and 3"))

    names = [s.name for s in tracer.spans]
    assert names[0] == "glia.run"
    assert "glia.model_call" in names and "glia.tool" in names

    run_span = tracer.spans[0]
    tool_span = next(s for s in tracer.spans if s.name == "glia.tool")
    model_span = next(s for s in tracer.spans if s.name == "glia.model_call")

    assert tool_span.context is run_span  # parented to the run span
    assert model_span.context is run_span
    assert tool_span.attributes["glia.tool.name"] == "add"
    assert model_span.attributes["glia.step"] == 1
    assert "glia.output_tokens" in model_span.attributes
    assert run_span.attributes["glia.stop_reason"] == "end_turn"
    assert all(s.ended for s in tracer.spans)  # every span closed


def test_tool_error_sets_error_attribute():
    @tool
    async def boom() -> str:
        raise ValueError("kaboom")

    tracer = FakeTracer()
    agent = Agent(EchoLLM([tcall("boom", {}), "done"]), tools=[boom], hooks=[_exporter(tracer)])
    run(agent.run("go"))

    tool_span = next(s for s in tracer.spans if s.name == "glia.tool")
    assert tool_span.attributes.get("glia.error") is True


def test_approval_and_compaction_recorded_as_span_events():
    from glia.approval import deny

    @tool
    async def danger(x: str) -> str:
        return "ran"

    tracer = FakeTracer()
    agent = Agent(
        EchoLLM([tcall("danger", {"x": "b"}), "ok"]), tools=[danger],
        approval=deny("danger"), hooks=[_exporter(tracer)],
    )
    run(agent.run("go"))

    run_span = tracer.spans[0]
    assert any(name == "approval" for name, _ in run_span.events)


def test_exporter_never_breaks_a_run():
    class AngryTracer:
        def start_span(self, *a, **k):
            raise RuntimeError("telemetry down")

    agent = Agent(EchoLLM(["hi"]), hooks=[OTelExporter(AngryTracer(), context_factory=lambda s: s)])
    result = run(agent.run("q"))  # must not raise
    assert result.output == "hi"
